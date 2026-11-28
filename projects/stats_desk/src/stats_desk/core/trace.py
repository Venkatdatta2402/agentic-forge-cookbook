"""One record per question of everything that happened in it, kept with the run and sent to Langfuse.

From ask-the-cookbook's traces.py, made async and nested for a team of agents:

    Trace        one question: every model call (agent, tokens in and out, seconds in the model,
                 seconds waiting for Groq's minute limit, Groq's error), every analysis run (who
                 ran it, its arguments, the result's id or the refusal, seconds), how it ended,
                 totals, and what the same tokens would cost on Groq's paid tier
    span()       an agent at work, so what it does nests under it: the lead, then under its
                 ask_analysts call each analyst, then under each analyst its own calls
    tool()       one analysis run, by an agent or by code
    generation() one model call

Nesting follows the asyncio task, like the meter: analysts running at the same time each nest
their calls under themselves and under nobody else. With no Trace open, all of these do nothing.

Langfuse gets the same tree, one observation at a time as it happens (Langfuse 5 takes no start
time after the fact). It is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and
STATS_DESK_LANGFUSE isn't "off" (the tests set it off), and nothing it does can reach an answer:
every call to it is caught, and a failure is said once. Langfuse sees the question and what each
agent was shown; the log line (`quiet`) doesn't carry the question or the brief.
"""

import contextvars
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

# Groq's paid-tier prices for gpt-oss-120b, as ask-the-cookbook took them from its model page on
# 8 October 2026: $0.15 per million input tokens, $0.60 per million output. The free tier pays
# nothing; this is what the same tokens would cost.
PRICE = {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000}
RESULT_CHARS = 4000            # of a tool's result, into Langfuse

CURRENT = contextvars.ContextVar("trace", default=None)
PARENT = contextvars.ContextVar("trace_parent", default=None)     # the observation to nest under
PATH = contextvars.ContextVar("trace_path", default=())           # the enclosing agents and tools


def cost(tokens_in, tokens_out):
    return round(tokens_in * PRICE["input"] + tokens_out * PRICE["output"], 6)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment():
    return os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", "local")


# --------------------------------------------------------------------------- Langfuse, never in the way

class _Langfuse:
    def __init__(self):
        self.client, self.broken = None, False

    def on(self):
        return (bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
                and os.environ.get("STATS_DESK_LANGFUSE", "on") != "off" and not self.broken)

    def get(self):
        if self.client is None and self.on():
            try:
                from langfuse import Langfuse
                self.client = Langfuse(base_url=os.environ.get("LANGFUSE_BASE_URL") or None,
                                       environment=environment())
            except Exception as error:                     # noqa: BLE001 -- tracing is optional
                _complain(error)
                self.broken = True
        return self.client if not self.broken else None


LANGFUSE = _Langfuse()
_COMPLAINED = set()


def _complain(error):
    if type(error).__name__ not in _COMPLAINED:
        _COMPLAINED.add(type(error).__name__)
        print(f"langfuse: {type(error).__name__}: {str(error)[:200]} (runs are unaffected; said once)", flush=True)


def _safe(do):
    try:
        return do()
    except Exception as error:                             # noqa: BLE001 -- tracing is optional
        _complain(error)
        return None


def flush():
    """Send what Langfuse still holds; a script calls this before it exits."""
    if LANGFUSE.client is not None:
        _safe(LANGFUSE.client.flush)


def _child(name, as_type, **fields):
    parent = PARENT.get()
    return _safe(lambda: parent.start_observation(name=name, as_type=as_type, **fields)) if parent else None


def _end(obs, **fields):
    if obs is not None:
        _safe(lambda: obs.update(**{k: v for k, v in fields.items() if v is not None}))
        _safe(obs.end)


# --------------------------------------------------------------------------- one question's record

class Trace:
    """`via`: who asked -- "script", "eval", "api". `session`: a conversation's id, so Langfuse
    groups a question with its follow-ups. `tags`: more labels for Langfuse (an eval question's id).
    In Langfuse the trace is named by its question."""

    def __init__(self, question, via="script", session=None, tags=()):
        self.tags = [via, *tags]
        self.record = {"id": uuid.uuid4().hex, "via": via, "session": session, "tags": self.tags,
                       "environment": environment(), "started": _now(), "question": question,
                       "outcome": None, "stopped": None, "error": None, "seconds": None,
                       "tokens": {"in": 0, "out": 0, "total": 0}, "cost if paid": 0.0,
                       "waited seconds": 0.0, "model calls": [], "tools": []}
        self._began, self._root, self._attributes, self._tokens = time.monotonic(), None, None, []

    def __enter__(self):
        self._tokens = [CURRENT.set(self)]
        lf = LANGFUSE.get()
        if lf is not None:
            def start():
                from langfuse import propagate_attributes
                r = self.record
                name = r["question"] if len(r["question"]) <= 100 else r["question"][:97] + "..."
                self._attributes = propagate_attributes(session_id=r["session"], trace_name=name,
                                                        tags=self.tags, metadata={"via": r["via"]})
                self._attributes.__enter__()
                self._root = lf.start_observation(name=name, as_type="chain", input={"question": r["question"]})
                r["langfuse trace"] = self._root.trace_id
            _safe(start)
        self._tokens += [PARENT.set(self._root), PATH.set(())]
        return self

    def __exit__(self, kind, error, tb):
        if self.record["outcome"] is None:
            self.finish(outcome="error" if error else "ended", error=error)
        for var, token in zip((PATH, PARENT, CURRENT), reversed(self._tokens)):
            var.reset(token)
        return False

    def finish(self, outcome, brief=None, stopped=None, error=None, **more):
        r = self.record
        r["outcome"], r["stopped"] = outcome, stopped
        r["error"] = f"{type(error).__name__}: {str(error)[:300]}" if error is not None else None
        tokens_in = sum(c["in"] or 0 for c in r["model calls"])
        tokens_out = sum(c["out"] or 0 for c in r["model calls"])
        r["tokens"] = {"in": tokens_in, "out": tokens_out, "total": tokens_in + tokens_out}
        r["cost if paid"] = cost(tokens_in, tokens_out)
        r["waited seconds"] = round(sum(c["waited"] for c in r["model calls"]), 1)
        r["seconds"] = round(time.monotonic() - self._began, 1)
        r.update(more)
        print("trace " + json.dumps(quiet(r), ensure_ascii=False), flush=True)
        if self._root is not None:
            _end(self._root, output=brief, level="ERROR" if error is not None else None,
                 status_message=r["error"] or stopped,
                 metadata={k: r[k] for k in ("outcome", "tokens", "cost if paid", "seconds", "waited seconds")})
            self._root = None
        if self._attributes is not None:
            _safe(lambda: self._attributes.__exit__(None, None, None))
            self._attributes = None
        return r


def quiet(record):
    """The record for the server's log: no question, no brief."""
    return {k: v for k, v in record.items() if k not in ("question", "brief")} | {
        "model calls": len(record["model calls"]), "tools": len(record["tools"])}


# --------------------------------------------------------------------------- the hooks

def note(kind, **data):
    """Something worth counting that isn't a call: e.g. a rejected call recovered as a real one."""
    trace = CURRENT.get()
    if trace is not None:
        trace.record.setdefault("notes", []).append({"kind": kind, "path": " > ".join(PATH.get()), **data})

class _Handle:
    """What span(), tool() and generation() give back; does nothing when no Trace is open."""

    def __init__(self, entry=None, obs=None):
        self.entry, self.obs, self.began = entry, obs, time.monotonic()
        self.output = None


@contextmanager
def span(name, as_type="agent", input=None, label=None):
    """An agent at work: everything inside it nests under it. `label`: its name in Langfuse when
    that should say more than `name` ("analyst f1: has unemployment risen..."); the record and
    the paths use `name`."""
    trace = CURRENT.get()
    if trace is None:
        yield _Handle()
        return
    handle = _Handle(obs=_child(label or name, as_type, input=input))
    tokens = (PARENT.set(handle.obs or PARENT.get()), PATH.set(PATH.get() + (name,)))
    try:
        yield handle
    finally:
        PATH.reset(tokens[1])
        PARENT.reset(tokens[0])
        _end(handle.obs, output=handle.output)


@contextmanager
def tool(name, arguments, by=None):
    """One analysis run (or the lead's ask_analysts). `by`: "code" when no model decided it.
    Set `handle.output` to the Result (kind, content) before leaving."""
    trace = CURRENT.get()
    if trace is None:
        yield _Handle()
        return
    entry = {"n": len(trace.record["tools"]) + 1, "path": " > ".join(PATH.get()), "tool": name,
             "args": arguments, "by": by or "model", "kind": None, "result": None, "said": None,
             "seconds": None}
    trace.record["tools"].append(entry)
    handle = _Handle(entry, _child(_tool_label(name, arguments, by), "tool", input=arguments))
    tokens = (PARENT.set(handle.obs or PARENT.get()), PATH.set(PATH.get() + (name,)))
    try:
        yield handle
    finally:
        PATH.reset(tokens[1])
        PARENT.reset(tokens[0])
        entry["seconds"] = round(time.monotonic() - handle.began, 2)
        result = handle.output
        if result is not None:
            kind = getattr(result, "kind", "ok")
            content = getattr(result, "content", None) or str(result)
            first = content.split("\n", 1)[0]
            entry["kind"] = kind
            entry["result"] = first.removeprefix("result ") if first.startswith("result ") else None
            entry["said"] = None if entry["result"] else content[:200]
            _end(handle.obs, output=content[:RESULT_CHARS],
                 level="WARNING" if kind != "ok" else None, status_message=None if kind == "ok" else kind)
        else:
            entry["kind"] = "not finished"
            _end(handle.obs, level="ERROR", status_message="not finished")


def _tool_label(name, arguments, by):
    """An analysis's name in Langfuse: what ran, on what ("gap ur_annual", "recheck compare")."""
    args = arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    args = args if isinstance(args, dict) else {}
    on = args.get("table") or args.get("x_table") or args.get("analysis") or args.get("name") or ""
    if isinstance(args.get("names"), list):
        on = ", ".join(args["names"])
    if isinstance(args.get("tables"), list):          # join
        on = " + ".join(map(str, args["tables"]))
    return " ".join(p for p in (name, str(on), "(by code)" if by == "code" else "") if p)


def generation(model, messages, tools=None, max_tokens=None):
    """A model call, opened just before it is sent: call `.finish(...)` with what came back. In
    Langfuse it is named, once it is back, by what it decided ("analyst f1 -> gap")."""
    trace = CURRENT.get()
    if trace is None:
        return _Generation(None)
    entry = {"n": len(trace.record["model calls"]) + 1, "path": " > ".join(PATH.get()),
             "agent": PATH.get()[-1] if PATH.get() else "", "model": model, "in": None, "out": None,
             "seconds": None, "waited": 0.0, "error": None}
    trace.record["model calls"].append(entry)
    obs = _child(f"{entry['agent'] or 'model'} thinking", "generation", model=model, input=messages,
                 model_parameters={"max_tokens": max_tokens} if max_tokens else None,
                 metadata={"tools": [t["function"]["name"] for t in tools or []]})
    return _Generation(entry, obs)


class _Generation(_Handle):
    def __init__(self, entry, obs=None):
        super().__init__(entry, obs)

    def finish(self, tokens_in=None, tokens_out=None, output=None, error=None, waited=0.0, decided=None,
               finish=None):
        """`decided`: what the reply did -- the tools it called, "wrote text", or "nothing".
        `finish`: why the model stopped ("stop", "tool_calls", "length": out of output tokens)."""
        if self.entry is None:
            return
        e = self.entry
        e.update({"in": tokens_in, "out": tokens_out, "seconds": round(time.monotonic() - self.began, 2),
                  "waited": round(waited, 2), "error": error, "decided": decided, "finish": finish})
        if finish == "length":
            decided = f"{decided} (ran out of output tokens)"
        usage = {"input": tokens_in or 0, "output": tokens_out or 0}
        called = re.search(r'generated: \{"name": "(\w+)"', error or "")
        what = (f"rejected by Groq (called '{called[1]}')" if called else "rejected by Groq") if error else decided
        _end(self.obs, name=f"{e['agent'] or 'model'} -> {what}" if what else None,
             output=output, usage_details=usage,
             cost_details={"input": cost(usage["input"], 0), "output": cost(0, usage["output"])},
             metadata={"waited seconds": e["waited"]},
             level="ERROR" if error else None, status_message=error)
