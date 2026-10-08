"""Stand-ins for every model, so the whole system runs offline -- on the real corpus, really split and indexed.

    WordEmbedding   a deterministic bag-of-words embedding, counting the texts it embeds. Crude, but retrieval
                    over this repo works with it, and it scores on its own scale, not Gemini's
    answering()     LlamaIndex's own `MockLLM`, which answers with the prompt it was given
    scripted()      LlamaIndex's `MockFunctionCallingLLM`, scripted to search, open the file the search found,
                    then answer citing it -- so the agent's tool loop really runs
"""

import hashlib
import math
import re

from llama_index.core.base.embeddings.base import BaseEmbedding


class WordEmbedding(BaseEmbedding):
    """Each word hashed into one of 256 buckets, normalised. `texts` counts what has been embedded."""

    dim: int = 256
    texts: int = 0

    def _vector(self, text):
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z_]{3,}", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _get_text_embedding(self, text):
        self.texts += 1
        return self._vector(text)

    def _get_text_embeddings(self, texts):
        self.texts += len(texts)
        return [self._vector(t) for t in texts]

    def _get_query_embedding(self, query):
        return self._vector(query)

    async def _aget_query_embedding(self, query):
        return self._vector(query)


def answering():
    from llama_index.core.llms import MockLLM
    return MockLLM(max_tokens=40)


def scripted(question_file="04_memory/recall.py", query="similarity floor before rescaling"):
    """An agent model that searches, opens what it found, and answers citing it."""
    from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
    from llama_index.core.llms import MockFunctionCallingLLM

    def respond(messages, **kwargs):
        done = [m for m in messages if m.role == MessageRole.TOOL]
        if not done:
            return ChatMessage(role=MessageRole.ASSISTANT, blocks=[
                ToolCallBlock(tool_call_id="1", tool_name="search_cookbook", tool_kwargs={"query": query})])
        if len(done) == 1:
            return ChatMessage(role=MessageRole.ASSISTANT, blocks=[
                ToolCallBlock(tool_call_id="2", tool_name="open_file",
                              tool_kwargs={"path": question_file, "line_start": 1, "line_end": 30})])
        return ChatMessage(role=MessageRole.ASSISTANT,
                           content=f"The floor runs on raw similarity, before rescaling [{question_file}].")

    return MockFunctionCallingLLM(response_generator=respond, is_chat_model=True)


def scripted_client(steps, seen=None):
    """A stand-in for chapter 1's `llm.client.chat.completions.create`, playing back `steps` in order.

    A step is ("tool", name, {args}) or ("answer", text). `seen`, if given, gets the messages each call was
    sent, so a test can measure how big the calls grew.
    """
    import json
    from types import SimpleNamespace as NS
    position = {"n": 0}

    def create(model=None, messages=None, tools=None, **kwargs):
        if seen is not None:
            seen.append(messages)
        step = steps[min(position["n"], len(steps) - 1)]
        position["n"] += 1
        if not tools:
            # offered no tools -- chapter 2's graceful finish -- so it can only answer
            step = ("answer", "best answer from what was found [stand-in]")
        if step[0] == "tool":
            arguments = json.dumps(step[2])
            call = NS(id=f"c{position['n']}", type="function", function=NS(name=step[1], arguments=arguments))
            dumped = {"role": "assistant", "content": None, "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": step[1], "arguments": arguments}}]}
            message = NS(content=None, tool_calls=[call], model_dump=lambda **k: dumped)
        else:
            message = NS(content=step[1], tool_calls=None,
                         model_dump=lambda **k: {"role": "assistant", "content": step[1]})
        return NS(choices=[NS(message=message, finish_reason="stop")],
                  usage=NS(prompt_tokens=sum(len(str(m.get("content") or "")) for m in messages) // 4,
                           completion_tokens=20, total_tokens=0))
    return create
