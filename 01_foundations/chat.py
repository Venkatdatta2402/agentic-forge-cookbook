import asyncio
import os
from pathlib import Path

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).parent / ".tiktoken_cache"))

import tiktoken
from openai import BadRequestError
from pydantic import BaseModel

from llm import async_client, chat, current_model, execute_tool_call, extract

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(_encoding.encode(text))


def _role(m):
    return getattr(m, "role", None) or m.get("role")


def _content(m):
    return getattr(m, "content", None) if hasattr(m, "content") else m.get("content")


def _tokens(m):
    content = _content(m)
    return count_tokens(content) if content else 0


def render_transcript(messages):
    lines = []
    for m in messages:
        role, content = _role(m), _content(m)
        if role in ("user", "assistant") and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


class ConversationMemory(BaseModel):
    key_facts: list[str]


def trim_to_budget(messages, max_tokens, reserve_turns=2):
    system = [m for m in messages if _role(m) == "system"]
    rest = [m for m in messages if _role(m) != "system"]

    # A turn is a user message plus everything the assistant produced in
    # response to it — one message for plain text, several when tool calls
    # and their results are involved. Grouping this way means a tool round
    # trip gets dropped as a whole, never leaving an orphaned tool result
    # without the tool-call message it belongs to.
    turns = []
    for m in rest:
        if _role(m) == "user" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)

    num_turns = len(turns)
    total_tokens = sum(_tokens(m) for turn in turns for m in turn)
    avg_turn_tokens = total_tokens / num_turns if num_turns else 0
    history_budget = max_tokens - reserve_turns * avg_turn_tokens

    def current_tokens():
        return sum(_tokens(m) for m in system) + sum(_tokens(m) for turn in turns for m in turn)

    while turns and current_tokens() > history_budget:
        turns.pop(0)

    return system + [m for turn in turns for m in turn]


class Conversation:
    def __init__(self, system=None):
        self.messages = []
        self.notes = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def send(self, content, **kwargs):
        self.messages.append({"role": "user", "content": content})
        reply = chat(self.messages, **kwargs)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def trim(self, max_tokens, reserve_turns=2, remember=False):
        trimmed = trim_to_budget(self.messages, max_tokens, reserve_turns)

        if remember:
            dropped = [m for m in self.messages if not any(m is kept for kept in trimmed)]
            transcript = render_transcript(dropped)
            if transcript:
                memory = extract(
                    "Extract only genuine decisions, constraints, preferences, or open/unanswered "
                    "questions worth remembering long after this excerpt is gone. A stated personal "
                    "preference counts even if it seems minor (a favorite number, color, or similar) "
                    "— what matters is that the user stated something about themselves, not how "
                    "consequential it is. Do NOT include factual answers, tool results, or a recap "
                    "of what was discussed — a turn that's just a factual query with no decision, "
                    "constraint, preference, or open question behind it should produce nothing at "
                    "all. Return JSON with a single field \"key_facts\": a list of short strings, "
                    f"one per fact. If nothing qualifies, use an empty list.\n\n{transcript}",
                    ConversationMemory,
                )
                self.notes.extend(memory.key_facts)

        self.messages = trimmed

    # model=None rather than model=DEFAULT_MODEL, so the active model is looked up when the call
    # happens instead of frozen when this method was defined -- see llm.use().
    async def asend_with_tools(self, content, tools, functions, schemas=None, retries=3, model=None):
        self.messages.append({"role": "user", "content": content})
        for _ in range(retries):
            try:
                response = await async_client.chat.completions.create(
                    model=model or current_model(), messages=self.messages, tools=tools
                )
            except BadRequestError:
                self.messages.append({
                    "role": "user",
                    "content": "That tool call could not be processed. Try again, calling one of the available tools correctly.",
                })
                continue

            message = response.choices[0].message
            self.messages.append(message)

            if not message.tool_calls:
                return message.content

            async def run(tool_call):
                try:
                    return await asyncio.to_thread(execute_tool_call, tool_call, functions, schemas)
                except Exception as e:
                    return f"Error: {e}"

            results = await asyncio.gather(*(run(tc) for tc in message.tool_calls))
            for tool_call, result in zip(message.tool_calls, results):
                self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        raise ValueError(f"Failed to get a final answer after {retries} attempts")


class Agent:
    def __init__(self, name="Agent", system=None, tools=None, functions=None, schemas=None):
        self.name = name
        self.conversation = Conversation(system=system)
        self.tools = tools
        self.functions = functions
        self.schemas = schemas

    async def ask(self, message):
        if self.tools:
            answer = await self.conversation.asend_with_tools(message, self.tools, self.functions, self.schemas)
        else:
            answer = await asyncio.to_thread(self.conversation.send, message)
        print(f"{self.name}: {answer}")
