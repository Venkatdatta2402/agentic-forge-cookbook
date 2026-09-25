import asyncio
import os
from pathlib import Path

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).parent / ".tiktoken_cache"))

import tiktoken
from openai import BadRequestError
from pydantic import BaseModel

from llm import async_client, chat, execute_tool_call, extract, DEFAULT_MODEL

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
    """The messages an agent is carrying, plus the live things it needs to do its work.

    `messages` and `notes` are data -- they can be written to a file and read back. `resources`
    is the opposite: connections, sessions, database handles, things that are *open* and have
    to be closed. Keeping them here rather than in a closure is what makes them reachable,
    which is what makes them replaceable.

    Two chapters asked for this independently. `04_memory` notebook 5 found that nothing on a
    Conversation could hold a memory store, so `remember` had to close over one. `07_mcp`
    notebook 6 found the same thing for a live MCP session, where it is worse: a session whose
    server has died needs replacing, and a thing sealed inside a closure cannot be replaced by
    anyone.
    """

    def __init__(self, system=None):
        self.messages = []
        self.notes = []
        # name -> a live thing. Anything with a `close()` is closed by `close()` below; the
        # names are the caller's to choose ("mcp", "store"), because nothing here needs to
        # know what is in it.
        self.resources = {}
        if system:
            self.messages.append({"role": "system", "content": system})

    def close(self):
        """Close every resource that knows how, and forget them all.

        Deliberately NOT called by `Runtime.run()`, which was the first design and is wrong: a
        long-running agent does several runs on one conversation, and closing its connection
        at the end of run one would break run two. Cleanup belongs to whoever opened the
        conversation, which is why this is also a context manager -- `with Conversation() as c`
        is the form that cannot forget.
        """
        problems = []
        for name, resource in list(self.resources.items()):
            closer = getattr(resource, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as e:  # noqa: BLE001 -- one bad resource must not orphan the rest
                    problems.append(f"{name}: {type(e).__name__}: {e}")
        self.resources.clear()
        return problems

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

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

    async def asend_with_tools(self, content, tools, functions, schemas=None, retries=3, model=DEFAULT_MODEL):
        self.messages.append({"role": "user", "content": content})
        for _ in range(retries):
            try:
                response = await async_client.chat.completions.create(
                    model=model, messages=self.messages, tools=tools
                )
            except BadRequestError:
                self.messages.append({
                    "role": "user",
                    "content": "That tool call could not be processed. Try again, calling one of the available tools correctly.",
                })
                continue

            message = response.choices[0].message
            # `.model_dump()` rather than the SDK object itself, so `messages` holds one type
            # and not two. Appending the object works right up until something tries to WRITE
            # the conversation out -- json.dump raises TypeError on a ChatCompletionMessage --
            # and a persistence boundary is the worst place to discover that a list you have
            # been treating as data is half objects. The API accepts the dict form unchanged,
            # tool_calls included; `message` is still the object below, so nothing else moves.
            self.messages.append(message.model_dump(exclude_none=True))

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
