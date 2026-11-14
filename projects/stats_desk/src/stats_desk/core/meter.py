"""Who spent which tokens, without passing a counter through every call.

From 06_orchestration/meter.py. That version patched the sync client's `create` for the length of
a `with` block, which counts everything in the process: fine in a notebook, wrong in a server
answering several questions at once. Here a ContextVar holds the meters that are open in the
current task, and asyncio copies it into every task started inside it, so the calls an analyst
makes inside `asyncio.gather` are counted for that analyst and for its question, and for nobody else.

    async with metered() as usage:
        await agent.run(task)
    usage.total
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_open: ContextVar[tuple["Usage", ...]] = ContextVar("open_meters", default=())
_agent: ContextVar[str] = ContextVar("agent", default="")


@dataclass
class Call:
    model: str
    agent: str
    prompt: int
    completion: int
    seconds: float            # in the model
    waited: float = 0.0       # in the governor, before the call was sent
    remaining: int | None = None
    error: str | None = None  # set for a call the API rejected (it still took time and budget)

    @property
    def total(self):
        return self.prompt + self.completion


@dataclass
class Usage:
    calls: list[Call] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    @property
    def rejected(self):
        return [c for c in self.calls if c.error]

    @property
    def total(self):
        return sum(c.total for c in self.calls)

    def by(self, key):
        out = {}
        for c in self.calls:
            out[getattr(c, key)] = out.get(getattr(c, key), 0) + c.total
        return out

    def summary(self):
        return {"calls": len(self.calls), "rejected": len(self.rejected), "total": self.total,
                "prompt": sum(c.prompt for c in self.calls),
                "completion": sum(c.completion for c in self.calls),
                "by_model": self.by("model"), "by_agent": self.by("agent"),
                "errors": [f"{c.agent}: {c.error}" for c in self.rejected],
                "model_seconds": round(sum(c.seconds for c in self.calls), 1),
                "waited_seconds": round(sum(c.waited for c in self.calls), 1),
                "seconds": round(time.monotonic() - self.started, 1)}


@contextmanager
def metered():
    usage = Usage()
    token = _open.set(_open.get() + (usage,))
    try:
        yield usage
    finally:
        _open.reset(token)


@contextmanager
def acting_as(agent: str):
    """Label every model call made inside this block with the agent making it."""
    token = _agent.set(agent)
    try:
        yield
    finally:
        _agent.reset(token)


def record(model, prompt, completion, seconds, waited=0.0, remaining=None, error=None):
    call = Call(model, _agent.get(), prompt, completion, seconds, waited, remaining, error)
    for usage in _open.get():
        usage.calls.append(call)
    return call
