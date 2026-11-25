"""The agent loop: Think decides, Act carries out, Observe records. Async.

From 02_agent_runtime (runtime.py, think.py, act.py, observe.py), cut down to what this app uses
and made async. Kept as the chapter measured it:

  - history in the native tool-calling shape (assistant message with tool_calls, then tool
    messages with matching ids), because narrated history made the model repeat calls
  - budgets on iterations, tool calls, tokens and seconds, checked before every component
  - the doom-loop checks: identical calls repeated, the same error twice, oscillation, and the
    same question in other words getting the same answer
  - a stopped run still ends on an answer: the last Think gets one turn with no tools, told to
    use only what the tool results say (`finish`)

Changed:
  - several tool calls in one turn run concurrently (`asyncio.gather` in Act). Chapter 2 had one
    call per turn and got parallelism from forks; here parallel delegation is just two calls.
  - Observe passes results through verbatim. Chapter 2 had a model restate each result, which
    costs a call per step; these tools already return compact text (06_orchestration did the
    same with its `verbatim` extractor).
  - the token budget counts tokens actually spent by this run (the meter), not the length of
    the conversation.
"""

import asyncio
import difflib
import json
import re
import time
import uuid
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import events, llm, meter, trace
from .tools import Registry, Result, invoke


@dataclass
class Conversation:
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    stopped: str | None = None
    remaining_budget: dict = field(default_factory=dict)
    pending: list | None = None


def pending_decision(conversation):
    return next((m for m in reversed(conversation.messages)
                 if m.get("role") == "decision" and not m.get("applied")), None)


def history(conversation, shorten=(), keep_recent=2):
    """The conversation as API messages, in the native tool-calling shape.

    `shorten` names tools whose OLD results (all but the last `keep_recent` results) are cut to
    their first line. 05_context's selection, applied per call: a long intermediate result is
    needed for the next step, then only paid for again on every later call. In
    live run 6 the analyst's prompt grew from 1,320 to 3,719 tokens over 8 calls that way.
    Results that carry the numbers being reported are never in `shorten`.
    """
    out = []
    results = [i for i, m in enumerate(conversation.messages) if m["role"] == "tool"]
    old = set(results[:-keep_recent]) if keep_recent else set(results)
    for i, m in enumerate(conversation.messages):
        role = m["role"]
        if role in ("user", "assistant"):
            out.append({"role": role, "content": m["content"]})
        elif role == "decision" and m.get("type") == "tool_calls":
            out.append({"role": "assistant", "content": m.get("reasoning") or "",
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}}
                                       for c in m["calls"]]})
        elif role == "tool":
            content = m["content"]
            if i in old and m.get("name") in shorten and content.count("\n") > 1:
                # two lines: a result's id, and what it is ("PROFILE unemployment rate...")
                content = "\n".join(content.split("\n")[:2]) + "  [earlier result, shortened]"
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": content})
    return out


GUIDANCE = (
    "\n\nIf a tool result below already answers the request, respond with that answer in plain "
    "text; do not call the tool again. Give the answer itself, with nothing about your process.")


def parse_call(text):
    """A tool call written as text -- {"name": ..., "arguments": {...}}, or the arguments alone --
    as (name or None, arguments), or None if the text holds no JSON object."""
    if not text:
        return None
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        found = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(found, dict):
        return None
    if "arguments" in found and isinstance(found.get("name"), str):
        args = found["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        return (found["name"], args) if isinstance(args, dict) else None
    return None, found


class NoDecision(RuntimeError):
    """A step that got no usable reply: every try rejected or empty, or its retries spent the
    budget. The runtime ends the agent with this as the reason, as at any budget; it doesn't
    crash the question (span test: an analyst's 4 rejected tries raised out of the whole run)."""


# Groq's words when the model wrote text where a tool call was required: "did not call a tool",
# or, for half a thought ("We have change result. Need claim."), "could not be parsed"
PROSE = ("did not call a tool", "could not be parsed")


def fitting(tools, args):
    """The tools whose arguments these are."""
    out = []
    for t in tools:
        try:
            t.validate(args)
            out.append(t)
        except ValidationError:
            pass
    return out


class Think:
    def __init__(self, tools: Registry | None = None, model=None, max_tokens=800,
                 reasoning_effort=None, retries=4, shorten=(), keep_recent=2, require_tool=False, choose=None,
                 when_stuck=None):
        """`require_tool`: every reply must call a tool (the API's tool_choice "required"), for
        agents that only ever finish by handing in a form: a live run lost its brief
        to a lead writing it as text three times.

        Every tool stays on offer. Tried and dropped (span tests): hiding the lead's
        brief form until a finding was in. The model called it anyway, from its instructions,
        Groq refused each call ("not in request.tools"), and after 4 tries the turn failed.

        `choose(conversation)`: the name of the tool this call must use, or None (the API's
        tool_choice naming a function: the model generates a call to that tool and no other).
        For a workflow's fixed steps: the lead reads profiles, then plans, before it may hand in
        (span tests: where a profile already showed the answer, the lead wrote its
        brief from it instead of planning in 8 of 12 runs, whatever its instructions said)."""
        self.require_tool, self.choose, self.when_stuck = require_tool, choose, when_stuck
        self.tools = tools or Registry()
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self.shorten = tuple(shorten)
        self.keep_recent = keep_recent

    def _recover(self, error, forced=None):
        """A rejected call whose arguments fit exactly one of our tools, made as that call.

        gpt-oss often gets the name of a call wrong and its contents right: in the 5 baseline
        runs it called 'json' or 'commentary' 9 times, with a complete hand-in as
        the arguments, and Groq threw each one away. Exactly one tool must fit, or nothing is
        guessed; the call then goes through the same checks as any other. While a tool is
        named (`forced`), only that one: span tests had the lead, told to call
        ask_analysts, write a brief instead, Groq refuse it, and this recover it as the brief."""
        parsed = parse_call(llm.failed_generation(error))
        if parsed is None:
            return None
        name, args = parsed
        pool = [self.tools[forced]] if forced else [self.tools[name]] if name in self.tools else list(self.tools)
        candidates = fitting(pool, args)
        if len(candidates) != 1:
            return None
        chosen = candidates[0].name
        trace.note("recovered call", called=name, as_=chosen)
        events.emit("recovered", called=name, tool=chosen)
        return {"role": "decision", "type": "tool_calls",
                "reasoning": f"(the model called '{name}'; its arguments fit `{chosen}`, so it was made as that)",
                "calls": [{"id": f"recovered_{uuid.uuid4().hex[:8]}", "name": chosen,
                           "arguments": json.dumps(args, ensure_ascii=False)}]}

    def _rejected(self, error):
        """What goes back to the model after a rejected call: Groq's reason, and the real names.
        A model that wrote prose where a tool call was required gets its own words back and the
        next move that fits (`when_stuck`): live run, an analyst retried on Groq's
        "issue the call again" wrote "this cannot be computed directly with a single tool" three
        times and spent its budget without one analysis."""
        names = ", ".join(f"`{n}`" for n in self.tools.names)
        message = str(getattr(error, "message", "") or error) + " " + str(llm._error_body(error).get("message", ""))
        wrote = (llm.failed_generation(error) or "").strip()
        if any(p in message for p in PROSE) and wrote:
            return (f"You wrote this instead of calling a tool:\n{wrote[:800]}\n\n"
                    + (self.when_stuck or f"Act on it now with one of your tools: {names}."))
        return ((llm.tool_call_failure(error) or "That request could not be processed.")
                + f"\n\nYour tools are exactly: {names}. Call one of them by that name"
                + ("." if self.require_tool else ", or answer in text."))

    def _kwargs(self, with_tools=True):
        kwargs = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        if self.require_tool and with_tools and len(self.tools):
            kwargs["tool_choice"] = "required"
        return kwargs

    def _system(self, conversation):
        budget = conversation.remaining_budget or {}
        note = ""
        if budget.get("turns") is not None and len(self.tools):
            note = (f"\nBudget left before this run is stopped: about {budget['turns']} turns. "
                    "When it runs out you will have to answer from what the results above show.")
        if budget.get("tokens") is not None and budget["tokens"] < 4_000 and len(self.tools):
            note += (f"\nOnly about {budget['tokens']:,} tokens of budget are left. Finish now: "
                     "run any last query you need and hand in your result.")
        # GUIDANCE says to answer in plain text: not for an agent whose every reply must be a tool
        # call (it told the analyst both, on every call)
        guidance = "" if self.require_tool else GUIDANCE
        return {"role": "system", "content": conversation.system + guidance + note}

    async def run(self, conversation):
        base = [self._system(conversation), *history(conversation, self.shorten, self.keep_recent)]
        messages = list(base)
        schemas = self.tools.schemas
        kwargs = self._kwargs()
        forced = self.choose(conversation) if self.choose else None
        # choose may say (tool, True): keep naming it on every try (an analyst out of budget: its
        # own choice after a refused try was another analysis, which stopped it with its figures
        # lost, in an eval)
        forced, keep = forced if isinstance(forced, tuple) else (forced, False)
        forced = forced if forced in self.tools else None
        # every try is paid in full, rejected or not: retries stop at the budget (span test: an
        # analyst's 4 rejected tries cost 17k tokens on a 12k budget)
        left, spent = (conversation.remaining_budget or {}).get("tokens"), 0
        for attempt in range(self.retries):
            if attempt and left is not None and spent >= left:
                raise NoDecision("its token budget went on replies Groq rejected")
            # one attempt with the tool named: Groq checks the call after it is generated, so a
            # model set on another tool is refused, not steered; after that, its own choice
            # (a step done too early is refused by its tool, which costs a call, not the turn)
            named = forced if attempt == 0 or keep else None
            if named:
                kwargs = {**kwargs, "tool_choice": {"type": "function", "function": {"name": named}}}
            elif forced:
                kwargs = self._kwargs()
            spent += llm.estimate(messages, schemas)
            try:
                reply = await llm.complete(self.model, messages, tools=schemas or None,
                                           max_tokens=self.max_tokens, **kwargs)
            except llm.BadRequestError as e:
                recovered = self._recover(e, named)
                if recovered is not None:
                    return recovered
                # Groq names what was wrong with a malformed call; hand exactly that back, once,
                # on top of a clean context rather than a growing stack of failures
                messages = [*base, {"role": "user", "content": self._rejected(e)
                                    + (f" Call `{named}` now." if named else "")}]
                continue
            if reply.tool_calls:
                return {"role": "decision", "type": "tool_calls", "reasoning": reply.content or "",
                        "calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                                  for c in reply.tool_calls]}
            if reply.content and reply.content.strip():
                return {"role": "decision", "type": "respond", "content": reply.content.strip()}
            # gpt-oss can spend its whole output budget reasoning and return nothing
            messages = [*base, {"role": "user", "content":
                                "Your last reply was empty. Reply now: call a tool"
                                + ("." if self.require_tool else ", or answer briefly in text.")}]
        raise NoDecision(f"no usable reply from {self.model} in {self.retries} tries")

    async def finish(self, conversation, reason):
        """A stopped run's last word, from the results only, with no tools on offer."""
        messages = [{"role": "system", "content": conversation.system}, *history(conversation)]
        if messages[-1].get("tool_calls"):
            messages.pop()        # a call whose results never arrived: the API rejects it
        messages.append({"role": "user", "content": (
            f"[The program running this conversation has stopped you ({reason}). Not a message "
            "from the user.]\nYou cannot call any more tools. Reply to the request using ONLY "
            "facts that appear in the tool results above -- nothing from general knowledge, no "
            "estimates. For any part they do not answer, say that you could not establish it. "
            "Do not claim that data does not exist or is not published: not having retrieved "
            "it is not evidence that it is missing.")})
        for _ in range(2):
            try:
                reply = await llm.complete(self.model, messages, max_tokens=self.max_tokens,
                                           **self._kwargs(with_tools=False))
            except llm.BadRequestError:
                messages.append({"role": "user", "content":
                                 "There are no tools. Reply in plain text, from the results above only."})
                continue
            if reply.content and reply.content.strip():
                return reply.content.strip()
            messages.append({"role": "user", "content": "Write that reply out as text."})
        return None


class Act:
    def __init__(self, tools: Registry, timeout=60, max_result_tokens=1500):
        self.tools = tools
        self.timeout = timeout
        self.max_result_tokens = max_result_tokens

    async def _one(self, call) -> Result:
        with trace.tool(call["name"], call["arguments"]) as observed:
            observed.output = await self._run(call)
            return observed.output

    async def _run(self, call) -> Result:
        name = call["name"]
        if name not in self.tools:
            return Result(name, "refused", f"There is no tool named '{name}'. Available: "
                          f"{', '.join(self.tools.names)}.")
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError as e:
            return Result(name, "bad_arguments", f"The arguments were not valid JSON: {e}")
        if not isinstance(args, dict):
            return Result(name, "bad_arguments", "The arguments must be a JSON object.")
        return await invoke(self.tools[name], args, timeout=self.timeout,
                            max_tokens=self.max_result_tokens)

    async def run(self, conversation):
        decision = pending_decision(conversation)
        if decision is None:
            return None
        decision["applied"] = True
        if decision["type"] == "respond":
            return {"role": "assistant", "content": decision["content"]}
        for c in decision["calls"]:
            events.emit("tool_call", tool=c["name"], arguments=c["arguments"])
        results = await asyncio.gather(*(self._one(c) for c in decision["calls"]))
        for call, result in zip(decision["calls"], results):
            call["result"] = result.as_seen
            call["kind"] = result.kind
            events.emit("tool_result", tool=call["name"], result_kind=result.kind,
                        preview=result.as_seen[:300])
        conversation.pending = list(zip(decision["calls"], results))
        return None


class Observe:
    async def run(self, conversation):
        pending, conversation.pending = conversation.pending, None
        if not pending:
            return None
        return [{"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                 "kind": result.kind, "content": result.as_seen} for call, result in pending]


def _similar(a, b, threshold=0.8):
    def values(text):
        try:
            args = json.loads(text)
        except (TypeError, ValueError):
            return str(text)
        return " ".join(str(v) for v in args.values()) if isinstance(args, dict) else str(args)
    return difflib.SequenceMatcher(None, values(a), values(b)).ratio() >= threshold


class AsyncRuntime:
    def __init__(self, loop, repeat_from=0, max_iterations=None, max_tool_calls=None,
                 max_tokens=None, max_seconds=None, cancel=None, answer_on_stop=True):
        self.loop = list(loop)
        # False for agents whose result must be a form: a prose answer written at a stop has
        # no SQL behind it. Measured (live run 5, analyst on gpt-oss-120b, stopped at budget):
        # its closing table gave 16 monthly values, and 4 of them -- months it had never been
        # shown -- were invented (27.5 where MoSPI says 26.4, 27.3 for 24.9, 27.1 for 24.5).
        self.answer_on_stop = answer_on_stop
        self.repeat_from = repeat_from
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.cancel = cancel
        self._position = 0
        self._iterations = 0

    def _next(self):
        component = self._peek()
        self._position = self.loop.index(component) + 1
        return component

    def _peek(self):
        return self.loop[self.repeat_from if self._position >= len(self.loop) else self._position]

    async def run(self, conversation):
        self._started = time.monotonic()
        conversation.stopped = None
        with meter.metered() as usage:
            self._usage = usage
            while True:
                reason = self._stop_reason(conversation)
                # A limit stops the next model call, not the carrying out of a decision already
                # made: that costs no tokens. Live run 2: a skeptic called submit with its verdict
                # on the call that crossed its budget, and the verdict was thrown away.
                carry_on = (reason is not None and reason in self._limits(conversation)
                            and not isinstance(self._peek(), Think))
                if reason is not None and not carry_on:
                    return await self._end_gracefully(conversation, reason)
                conversation.remaining_budget = self._remaining(conversation)
                try:
                    output = await self._next().run(conversation)
                except NoDecision as stuck:
                    return await self._end_gracefully(conversation, str(stuck))
                if isinstance(output, list):
                    conversation.messages.extend(output)
                elif output is not None:
                    conversation.messages.append(output)
                self._iterations += 1
                if conversation.messages and conversation.messages[-1]["role"] == "assistant":
                    return conversation

    # --- budgets ----------------------------------------------------------------------------

    def _calls(self, conversation):
        return [c for m in conversation.messages
                if m["role"] == "decision" and m.get("type") == "tool_calls" for c in m["calls"]]

    def _remaining(self, conversation):
        cycle = len(self.loop) - self.repeat_from
        left = None if self.max_iterations is None else max(0, self.max_iterations - self._iterations)
        return {"turns": None if left is None else left // max(cycle, 1),
                "tokens": None if self.max_tokens is None else max(0, self.max_tokens - self._usage.total)}

    def _limits(self, conversation):
        """The budgets this run has reached (empty if none)."""
        reached = []
        if self.max_iterations is not None and self._iterations >= self.max_iterations:
            reached.append(f"reached the maximum of {self.max_iterations} iterations")
        if self.max_tool_calls is not None and len(self._calls(conversation)) >= self.max_tool_calls:
            reached.append(f"reached the maximum of {self.max_tool_calls} tool calls")
        if self.max_tokens is not None and self._usage.total >= self.max_tokens:
            reached.append(f"reached its token budget of {self.max_tokens:,}")
        if self.max_seconds is not None and time.monotonic() - self._started >= self.max_seconds:
            reached.append(f"reached the time limit of {self.max_seconds}s")
        return reached

    def _stop_reason(self, conversation):
        if self.cancel is not None and self.cancel():
            return "cancelled"
        limits = self._limits(conversation)
        return limits[0] if limits else self._doom_loop(conversation)

    def _doom_loop(self, conversation, window=4):
        calls = self._calls(conversation)
        recent = calls[-window:]
        if len(recent) >= 2 and len({(c["name"], c["arguments"]) for c in recent[-2:]}) == 1:
            return "the same tool call with identical arguments repeated"
        errors = [m["content"] for m in conversation.messages
                  if m["role"] == "tool" and m.get("kind") not in (None, "ok")]
        if len(errors) >= 2 and errors[-1] == errors[-2]:
            return "the same error repeated"
        if len(recent) == window:
            pairs = [(c["name"], c["arguments"]) for c in recent]
            if pairs[0] == pairs[2] and pairs[1] == pairs[3] and pairs[0] != pairs[1]:
                return "oscillating between the same two calls"
        answered = [c for c in recent if "result" in c]
        if len(answered) >= 2:
            before, last = answered[-2:]
            if (before["name"] == last["name"] and before["result"] == last["result"]
                    and _similar(before["arguments"], last["arguments"])):
                return f"`{last['name']}` was asked the same thing again and returned the same result"
        return None

    async def _end_gracefully(self, conversation, reason):
        conversation.stopped = reason
        if reason == "cancelled":
            return conversation
        decided = pending_decision(conversation)
        if decided is not None and decided["type"] == "respond":
            decided["applied"] = True
            conversation.messages.append({"role": "assistant", "content": decided["content"],
                                          "stopped": reason})
            return conversation
        finisher = next((c for c in reversed(self.loop) if hasattr(c, "finish")), None)
        gathered = any(m["role"] == "tool" for m in conversation.messages)
        if finisher is not None and gathered and self.answer_on_stop:
            answer = await finisher.finish(conversation, reason)
            if answer:
                conversation.messages.append({"role": "assistant", "content": answer,
                                              "stopped": reason})
                return conversation
        conversation.messages.append({"role": "assistant", "stopped": reason, "content":
                                      f"Stopping: {reason}. This is partial progress, not an answer."})
        return conversation
