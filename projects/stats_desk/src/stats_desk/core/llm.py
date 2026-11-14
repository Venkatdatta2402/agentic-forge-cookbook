"""The one way this app calls a model: async, governed, metered.

From 01_foundations/llm.py, which had a sync client for the notebooks and an async one used only
for streaming. Here everything is async, and every call passes through the governor (so the
server stays inside Groq's limits) and the meter (so every token is attributed to a question and
an agent). `tool_call_failure` is carried over unchanged in spirit: Groq's 400 for a malformed
tool call says exactly what was wrong, and that is what goes back to the model.

The backend is swappable, so tests run the real code paths against a scripted model.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

import tiktoken
from openai import APIStatusError, AsyncOpenAI, AuthenticationError, BadRequestError

from .. import config
from . import meter, trace
from .governor import DailyLimitReached, TooLarge, account_of
from .governor import governor as default_governor

_encoding = tiktoken.get_encoding("o200k_base")
# Our count of a request (its messages and tools as JSON text, by o200k_base) against Groq's own
# prompt_tokens, measured 2026-11-13 on the five calls of one real question (lead and analyst
# contexts, 1.4-2.6k tokens): 7-16% HIGH on every one -- the JSON's quotes, keys and escapes cost
# more than Groq's own rendering adds. So no padding: it erred high already (the earlier
# x1.06 + 60 made it 16-26% high, holding back minute tokens calls never used).
TYPICAL_REPLY = 500     # analyst replies measured at 76-610 tokens, reasoning included

# What a call Groq REJECTED (a malformed tool call) most likely wrote, reasoning included: Groq
# charges it (probe 2026-11-13: two rejected calls cost 447 and 567 tokens of the minute's
# allowance, with max_tokens 300 and 1,000 -- so what was written, not the allowance) but says
# no number. Saved replies (span tests 2026-11-12, baseline 2026-11-11): analysts median 84,
# 9 in 10 under 117; the lead's 99 / 192, writing the brief 246 / 264. Rejected calls ran longer
# (~300-450 in the probe), so these sit above the 9-in-10 sizes. The visible rejected text, which
# Groq does return, is counted exactly when it is longer.
REJECTED_REPLY = {"analyst": 200, "lead": 300}


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def estimate(messages, tools=None) -> int:
    text = json.dumps(messages, ensure_ascii=False, default=str)
    if tools:
        text += json.dumps(tools, ensure_ascii=False)
    return count_tokens(text)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(Protocol):
    async def create(self, **kwargs) -> tuple[Reply, dict[str, str]]: ...


class GroqBackend:
    def __init__(self, api_key=None):
        self._client = AsyncOpenAI(api_key=api_key or os.environ["GROQ_API_KEY"],
                                   base_url=config.GROQ_BASE_URL,
                                   # the governor keeps us inside the limits; the SDK's retries
                                   # remain for 5xx and the rare 429 from spend we couldn't see
                                   max_retries=3)

    async def models(self) -> set[str]:
        """The models this key may use: Groq's model list, which costs no tokens. Raises
        AuthenticationError for a key Groq doesn't accept."""
        return {m.id async for m in self._client.models.list()}

    async def close(self):
        await self._client.close()

    async def create(self, **kwargs):
        raw = await self._client.chat.completions.with_raw_response.create(**kwargs)
        response = raw.parse()
        message = response.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments)
                 for c in (message.tool_calls or [])]
        usage = response.usage
        reply = Reply(message.content, calls, response.choices[0].finish_reason,
                      usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)
        return reply, dict(raw.headers)


_backend: Backend | None = None              # set by tests: a scripted model, used for everything
_default: Backend | None = None              # the server's own key (scripts, evals)
_question: ContextVar[Backend | None] = ContextVar("question_backend", default=None)
_governor = default_governor


def set_backend(backend):
    """Swap the model (tests use a scripted one). Returns the previous backend."""
    global _backend
    previous, _backend = _backend, backend
    return previous


def set_governor(governor):
    global _governor
    previous, _governor = _governor, governor
    return previous


def backend() -> Backend:
    global _default
    if _backend is not None:
        return _backend
    if (question := _question.get()) is not None:
        return question
    if _default is None:
        _default = GroqBackend()
    return _default


class QuestionKeys:
    """The Groq key one question runs on, and -- for a free question -- how to get another when
    Groq says this one's day is spent (decided 2026-11-13: a free question that runs a key dry
    carries on on the desk's other free key, if one has room and is free).

    `spare(spent_key, retry_after)` returns the next key, or None. Several analysts may hit the
    limit at once: the first switches, the others see the switch has happened (`generation`)
    and simply retry on the new key. `on_usage(key, tokens)`, if given, hears what every call
    used, on which key (the desk counts its free keys' last 24 hours with it)."""

    def __init__(self, key: str, spare=None, on_usage=None):
        self.spare, self.on_usage, self.generation = spare, on_usage, 0
        self._lock = asyncio.Lock()
        self._clients: list[GroqBackend] = []
        self._use(key)

    def _use(self, key):
        self.key, self.account = key, account_of(key)
        self.client = GroqBackend(api_key=key)
        self._clients.append(self.client)

    async def create(self, **kwargs):
        key = self.key
        try:
            reply, headers = await self.client.create(**kwargs)
        except APIStatusError as e:
            e.stats_desk_key = key                 # whose count a rejected call's estimate goes to
            raise
        await self.note_usage(key, reply.prompt_tokens + reply.completion_tokens)
        return reply, headers

    async def note_usage(self, key, tokens):
        if self.on_usage is not None:
            try:
                await self.on_usage(key, tokens)
            except Exception:  # noqa: BLE001 -- a count that fails to save must not fail the call
                pass

    async def switch(self, generation, retry_after) -> bool:
        """True if there is a key to retry on (just switched to, or by another task already)."""
        async with self._lock:
            if generation != self.generation:
                return True
            new = await self.spare(self.key, retry_after) if self.spare else None
            if new is None:
                return False
            self._use(new)                 # the old client may still be finishing calls: closed at the end
            self.generation += 1
            return True

    async def close(self):
        for client in self._clients:
            await client.close()


@asynccontextmanager
async def using_key(key: str, spare=None, on_usage=None):
    """Every model call made inside this block -- the question's agents, in every task they
    start -- goes out on `key` (then on what `spare` gives, see QuestionKeys) and draws on that
    key's own bucket in the governor. Keys live in this block's clients only, and are gone when
    the block ends: never stored, never logged."""
    keys = QuestionKeys(key, spare, on_usage)
    token = _question.set(keys)
    try:
        yield keys
    finally:
        _question.reset(token)
        await keys.close()


async def complete(model, messages, tools=None, max_tokens=800, **kwargs) -> Reply:
    """One model call. Raises TooLarge/DailyLimitReached before sending, BadRequestError as Groq does.
    A question's key whose day is spent is swapped for another (QuestionKeys) and the call retried."""
    while True:
        used = backend()
        generation = getattr(used, "generation", 0)
        try:
            return await _complete(used, model, messages, tools, max_tokens, **kwargs)
        except DailyLimitReached as e:
            if not hasattr(used, "switch") or not await used.switch(generation, e.retry_after):
                raise


async def _complete(used, model, messages, tools, max_tokens, **kwargs) -> Reply:
    asked = time.monotonic()
    estimated = estimate(messages, tools)
    # The per-request cap counts max_tokens; the minute's bucket doesn't need to. Measured: Groq
    # charges what a call uses (max_tokens 100 and 4,000 left the same 7,906 remaining), and
    # reserving the whole cap made every analyst call wait for room it rarely used.
    ticket = await _governor.reserve(model, estimated + min(max_tokens, TYPICAL_REPLY),
                                     request=estimated + max_tokens, account=getattr(used, "account", None))
    started = time.monotonic()
    request = {"model": model, "messages": messages, "max_tokens": max_tokens, **kwargs}
    if tools:
        request["tools"] = tools
    observed = trace.generation(model, messages, tools, max_tokens)
    try:
        reply, headers = await used.create(**request)
    except APIStatusError as e:
        headers = getattr(e.response, "headers", {}) or {}
        remaining = _remaining(headers)
        # A rejected call is still charged: its context, and what the model wrote (Groq gives no
        # number: REJECTED_REPLY). Not the whole reservation, which would leak max_tokens from
        # the bucket on every rejection. Only a 400 ran the model (a 413 or 429 was turned away
        # before it, and goes to no key's count).
        wrote = rejected_reply(e, max_tokens) if e.status_code == 400 else 0
        _governor.settle(ticket, estimated + wrote, remaining, _header_int(headers, "x-ratelimit-limit-tokens"))
        meter.record(model, estimated, wrote, time.monotonic() - started, started - asked, remaining,
                     error=_describe(e))
        observed.finish(estimated, wrote, error=_describe(e), waited=started - asked)
        if wrote and hasattr(used, "note_usage"):
            await used.note_usage(getattr(e, "stats_desk_key", None) or used.key, estimated + wrote)
        if e.status_code == 413:
            raise TooLarge(f"Groq refused the request as too large: {e}") from e
        if e.status_code == 429 and "per day" in str(e).lower():
            raise DailyLimitReached(f"Groq's daily limit for {model}: {e}",
                                    retry_after=_header_int(headers, "retry-after") or 3600) from e
        raise
    except Exception as e:          # a dropped connection, a timeout: recorded, then raised as is
        observed.finish(error=f"{type(e).__name__}: {str(e)[:300]}", waited=started - asked)
        raise
    remaining = _remaining(headers)
    _governor.settle(ticket, reply.prompt_tokens + reply.completion_tokens, remaining,
                     _header_int(headers, "x-ratelimit-limit-tokens"))
    meter.record(model, reply.prompt_tokens, reply.completion_tokens, time.monotonic() - started,
                 started - asked, remaining)
    observed.finish(reply.prompt_tokens, reply.completion_tokens, waited=started - asked, output=(
        {"tool calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls]}
        if reply.tool_calls else reply.content),
        decided=(", ".join(c.name for c in reply.tool_calls) if reply.tool_calls
                 else "wrote text" if (reply.content or "").strip() else "nothing"),
        finish=reply.finish_reason)
    return reply


def _error_body(error):
    """Groq's error object, whether the SDK kept the outer {"error": ...} or not."""
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return {}
    return body["error"] if isinstance(body.get("error"), dict) else body


def _describe(error):
    err = _error_body(error)
    parts = [str(getattr(error, "status_code", "")), str(err.get("code") or ""),
             str(err.get("message") or "")[:300]]
    if err.get("failed_generation"):
        parts.append("generated: " + str(err["failed_generation"])[:300])
    return " | ".join(p for p in parts if p)


def _header_int(headers, name):
    value = headers.get(name)
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def _remaining(headers):
    return _header_int(headers, "x-ratelimit-remaining-tokens")


def rejected_reply(error, max_tokens) -> int:
    """Our estimate of what a rejected call wrote: the visible rejected text, counted exactly, or
    a typical reply for the agent making the call (REJECTED_REPLY), whichever is larger; never
    more than the call's max_tokens."""
    agent = meter._agent.get() or ""
    typical = REJECTED_REPLY["analyst" if agent.startswith("analyst") else "lead"]
    visible = count_tokens(str(failed_generation(error) or ""))
    return min(max_tokens, max(visible, typical))


def failed_generation(error) -> str | None:
    """What the model generated, when Groq rejected it as a malformed tool call."""
    return _error_body(error).get("failed_generation")


def tool_call_failure(error) -> str | None:
    """What Groq said about a rejected tool call, phrased back at the model. None if it wasn't one.

    From 01_foundations/llm.py. Groq validates tool calls server-side and its 400 body names the
    bad field, or carries the exact malformed text in `failed_generation`; a generic "try again"
    leaves the model repeating the same mistake.
    """
    err = _error_body(error)
    if err.get("code") != "tool_use_failed":
        return None
    parts = [f"Your last tool call was rejected: {err.get('message', 'it could not be processed')}"]
    if err.get("failed_generation"):
        parts.append(f"This is exactly what you generated, and it is not valid:\n"
                     f"{str(err['failed_generation'])[:1500]}")
    parts.append(
        "Issue the call again as one complete, valid tool call, not as text. Make one call, not "
        "several. Close every brace and bracket, match every declared type, and leave out any "
        "optional field you do not need.")
    return "\n\n".join(parts)


__all__ = ["AuthenticationError", "BadRequestError", "DailyLimitReached", "GroqBackend", "QuestionKeys", "Reply", "ToolCall",
           "TooLarge", "complete", "count_tokens", "estimate", "set_backend", "set_governor",
           "tool_call_failure", "using_key"]
