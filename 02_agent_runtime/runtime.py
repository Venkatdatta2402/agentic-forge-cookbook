import concurrent.futures
import difflib
import json
import time

from act import pending_decision
from chat import count_tokens
from llm import chat
from think import latest_plan, render, resolve_args, step_dependencies


def _similar_args(a, b, threshold=0.8):
    # compared on the argument VALUES, not the JSON text: the keys are identical in any two calls
    # to the same tool, and counting them made '{"timezone": "Asia/Tokyo"}' and
    # '{"timezone": "Asia/Seoul"}' look 85% alike
    def values(text):
        try:
            args = json.loads(text)
        except (TypeError, ValueError):
            return str(text)
        return " ".join(str(v) for v in args.values()) if isinstance(args, dict) else str(args)

    return difflib.SequenceMatcher(None, values(a), values(b)).ratio() >= threshold


def _finisher(components):
    # Which component gets the last word when a run is stopped: the last Think in the loop,
    # looking inside a Loop too. Every pattern ends on a Think that can answer -- ReWOO's
    # solver, the search patterns' way out, the debate's moderator -- and a pattern with no
    # Think at all has nothing that could answer, so it keeps the plain stopping message.
    found = None
    for component in components:
        if hasattr(component, "components"):
            found = _finisher(component.components) or found
        elif callable(getattr(component, "finish", None)):
            found = component
    return found


class Runtime:
    def __init__(
        self,
        loop=None,
        repeat_from=None,
        max_iterations=None,
        max_tool_calls=None,
        max_tokens=None,
        max_seconds=None,
        cancel=None,
    ):
        self.loop = list(loop) if loop else []
        self.repeat_from = repeat_from
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.cancel = cancel
        self._position = 0
        self._iterations = 0
        self._started_at = None

    def add(self, component):
        self.loop.append(component)
        return self

    @property
    def finished(self):
        return self.repeat_from is None and self._position >= len(self.loop)

    def next(self):
        if self._position >= len(self.loop):
            self._position = self.repeat_from
        component = self.loop[self._position]
        self._position += 1
        return component

    def run(self, conversation):
        self._started_at = time.monotonic()
        # why this run was cut short, or None if it ended on an answer the model chose to give.
        # Set on the conversation rather than read out of the last message: since a stopped run
        # now ends with a real answer (see _end_gracefully), the text alone no longer says so.
        conversation.stopped = None
        while not self.finished:
            reason = self._stop_reason(conversation)
            if reason is not None:
                return self._end_gracefully(conversation, reason, self._drain(conversation))

            conversation.remaining_budget = self._remaining_budget(conversation)

            component = self.next()
            output = component.run(conversation)
            if output is not None:
                conversation.messages.append(output)
            self._iterations += 1

            if conversation.messages and conversation.messages[-1].get("role") == "assistant":
                self._drain(conversation)
                return conversation

        self._drain(conversation)
        return conversation

    def _drain(self, conversation):
        # Every way out of run() passes through here, because a background branch that outlives
        # the run costs real money for an answer nobody will read.
        #
        # Nothing is appended to `messages`. A great deal of code reads messages[-1] as "the
        # answer" -- Act._fork does it for every branch -- so writing a note after the final
        # assistant message would quietly turn that answer into a note. The record goes on the
        # conversation instead, and into the stopping message where one is being written anyway.
        running = getattr(conversation, "running", None)
        if not running:
            return []

        abandoned = list(running)
        if getattr(conversation, "abandoned", None) is not None:
            conversation.abandoned.set()          # branches stop at their next iteration
        for branch_id in abandoned:
            running.pop(branch_id).cancel()       # only bites if it never started
        conversation.abandoned_branches = abandoned

        # the run is over, so nothing is promised to anyone any more
        if getattr(conversation, "reserved_budget", None) is not None:
            conversation.reserved_budget = {"tokens": 0, "tool_calls": 0}
            conversation.reserved_by_branch = {}

        pool = getattr(conversation, "pool", None)
        if pool is not None:
            # not waited on: the cancel flag is what stops them, and blocking here would spend
            # exactly the time this is meant to save
            pool.shutdown(wait=False, cancel_futures=True)
            conversation.pool = None
        return abandoned

    def _tool_calls_made(self, conversation):
        return sum(1 for m in conversation.messages if m.get("role") == "decision" and m.get("type") == "tool_call")

    def _remaining_budget(self, conversation):
        # all four budgets Runtime tracks, not just time, so whatever decides to fork can hand
        # any of them to a branch. Note they are NOT all the same kind of thing: tokens and tool
        # calls are really consumed and so must be divided between branches, while seconds and
        # iterations are ceilings each branch faces on its own -- branches run concurrently, and
        # each Runtime counts its own iterations. think._POOLED_BUDGETS / _PER_BRANCH_BUDGETS is
        # where that distinction is stated to the model.
        tokens_used = sum(count_tokens(m["content"]) for m in conversation.messages if m.get("content"))
        # what branches left running were promised but have not necessarily spent yet. Subtracted
        # so the figure advertised to a fork is UNPROMISED budget rather than merely unspent
        # budget -- otherwise the same tokens get handed to a second branch while the first is
        # still working through them. Only the pooled budgets appear here; see the note above.
        reserved = getattr(conversation, "reserved_budget", None) or {}
        iterations = None if self.max_iterations is None else max(0, self.max_iterations - self._iterations)
        # iterations count COMPONENTS, so "16" on a Think/Act/Observe loop is about five model
        # turns, which nobody reading the number guesses. A Think is told turns, because that is
        # the unit it spends in. Only meaningful for a loop that cycles.
        cycle = len(self.loop) - self.repeat_from if self.repeat_from is not None else 0
        return {
            "iterations": iterations,
            "turns": None if iterations is None or cycle <= 0 else iterations // cycle,
            "tool_calls": None if self.max_tool_calls is None else max(
                0, self.max_tool_calls - self._tool_calls_made(conversation) - reserved.get("tool_calls", 0)),
            "tokens": None if self.max_tokens is None else max(
                0, self.max_tokens - tokens_used - reserved.get("tokens", 0)),
            "seconds": None if self.max_seconds is None else max(0.0, self.max_seconds - (time.monotonic() - self._started_at)),
        }

    def _stop_reason(self, conversation):
        if self.cancel is not None and self.cancel():
            return "cancelled"
        if self.max_iterations is not None and self._iterations >= self.max_iterations:
            return f"reached the maximum of {self.max_iterations} iterations"
        if self.max_tool_calls is not None and self._tool_calls_made(conversation) >= self.max_tool_calls:
            return f"reached the maximum of {self.max_tool_calls} tool calls"
        if self.max_tokens is not None:
            used = sum(count_tokens(m["content"]) for m in conversation.messages if m.get("content"))
            if used >= self.max_tokens:
                return f"reached the maximum token budget of {self.max_tokens}"
        if self.max_seconds is not None and time.monotonic() - self._started_at >= self.max_seconds:
            return f"reached the maximum execution time of {self.max_seconds}s"
        return self._doom_loop(conversation)

    def _doom_loop(self, conversation, window=4):
        decisions = [m for m in conversation.messages if m.get("role") == "decision" and m.get("type") == "tool_call"]
        recent = decisions[-window:]

        if len(recent) >= 2 and len({(d["tool"], d["content"]) for d in recent[-2:]}) == 1:
            return "the same tool call with identical arguments repeated"

        errors = [m["content"] for m in conversation.messages if m.get("role") == "tool" and m["content"].startswith("Error")]
        if len(errors) >= 2 and errors[-1] == errors[-2]:
            return "the same error repeated"

        if len(recent) == window:
            pairs = [(d["tool"], d["content"]) for d in recent]
            if pairs[0] == pairs[2] and pairs[1] == pairs[3] and pairs[0] != pairs[1]:
                return "oscillating between the same two calls"

        # The same question in slightly different words, getting the same answer back. The
        # identical-arguments check above never sees this, and it is the commonest way a model
        # re-searches: 04_memory's recall agent asked "checkout slowed 6th of November ..." and
        # then "checkout slowed ON 6th of November ...", and got the same listing twice. A repeat
        # that returns the same thing has taught the agent nothing, so it stops here and answers
        # from what it has (see _end_gracefully). Measured cost: that agent searched again because
        # Observe had dropped the id it needed, and a second listing sometimes kept the id -- so
        # this can cut off a lucky recovery. The fix for THAT is an extractor that keeps the
        # listing verbatim, not a looser check here.
        #
        # All three conditions, not just "same result". Bare values coincide legitimately:
        # get_current_time for Tokyo and for Seoul both return "20:49", and stopping there would
        # cut a three-city question short. Arguments that are nearly the same text are what make
        # it a repeat rather than a coincidence.
        answered = [d for d in recent if "result" in d]
        if len(answered) >= 2:
            before, last = answered[-2:]
            if (before["tool"] == last["tool"] and before["result"] == last["result"]
                    and _similar_args(before["content"], last["content"])):
                return f"`{last['tool']}` was asked the same thing again and returned the same result"

        return None

    def _end_gracefully(self, conversation, reason, abandoned=()):
        conversation.stopped = reason

        # A budget or doom-loop stop used to end on "Stopping: ... partial progress", discarding
        # whatever the tools had already found -- even when the answer was sitting in the last
        # result and the model had merely failed to notice. So the Think gets one more turn,
        # with no tools, and answers from what is there, saying what it could not find.
        #
        # Not when cancelled: `cancel` is how a caller says "stop now" -- a user pressing stop,
        # an abandoned branch, chapter 6's `submit` and handoffs -- and every one of those wants
        # the run over, not another model call. And not when no tool has returned anything,
        # because then there is nothing to answer from and the honest message is the stop.
        if reason != "cancelled":
            # the budget can run out in the one step between the model deciding to answer and
            # Act delivering it -- in which case the answer already exists and only needs saying
            decided = pending_decision(conversation)
            if decided is not None and decided.get("type") == "respond":
                decided["applied"] = True
                conversation.messages.append(
                    {"role": "assistant", "content": decided["content"], "stopped": reason})
                return conversation

        finisher = _finisher(self.loop)
        # a result counts whether or not Observe got to it: a stop between Act and Observe leaves
        # it only on the call, and Think.finish knows to look there
        gathered = any(m.get("role") == "tool" or "result" in m for m in conversation.messages)
        if reason != "cancelled" and finisher is not None and gathered:
            answer = finisher.finish(conversation, reason)
            if answer:
                conversation.messages.append({"role": "assistant", "content": answer, "stopped": reason})
                return conversation

        last = conversation.messages[-1] if conversation.messages else None
        progress = f" Last step: {last['role']} -> {last['content']!r}." if last else " No steps completed."
        # said out loud rather than dropped: work was paid for and thrown away, and a run that
        # keeps doing that is one where wait_for is being used wrongly
        cut = (f" Cut short while still running, and their work discarded: {', '.join(abandoned)}."
               if abandoned else "")
        conversation.messages.append({
            "role": "assistant",
            "content": f"Stopping: {reason}.{progress}{cut} This is partial progress, not a final answer.",
            "stopped": reason,
        })
        return conversation


class Loop:
    # a bounded inner cycle: repeats a full pass through `components` until `until(conversation)`
    # is true after a pass, then hands control back silently (no message appended) -- unlike
    # Runtime's own stop conditions, this is a normal handoff, not a doom/abnormal stop, so it
    # never explains itself in the conversation. Also stops immediately, mid-pass, the moment any
    # component produces a final assistant message -- that can happen before `until` would ever
    # fire (e.g. a decision to respond, which has nothing to do with `until`'s own condition).
    # Doesn't get Runtime's budget/doom-loop protection -- that stays the outer Runtime's job.
    # `until` is content-driven -- a predicate over the conversation, for when the number of
    # passes depends on what actually happens. `count` is a plain fixed number, for when it
    # doesn't. Either works alone; how many rounds a task deserves is the caller's call, which
    # is the whole reason this is a parameter and not a constant inside some pattern.
    def __init__(self, components, until=None, count=None, max_iterations=50):
        assert until is not None or count is not None, "Loop needs either `until` or `count`"
        self.components = list(components)
        self.until = until
        self.count = count
        self.max_iterations = max_iterations

    def run(self, conversation):
        passes = 0
        for _ in range(self.max_iterations):
            for component in self.components:
                output = component.run(conversation)
                if output is not None:
                    conversation.messages.append(output)
                if conversation.messages and conversation.messages[-1].get("role") == "assistant":
                    return None
            passes += 1
            if self.count is not None and passes >= self.count:
                return None
            if self.until is not None and self.until(conversation):
                return None
        return None


class Graph:
    # runs a DAG-shaped plan (see think.step_dependencies): each step starts the instant its
    # own dependencies finish, never gated behind a "wave" of unrelated steps that just happen
    # to be ready around the same time. Unlike Runtime/Loop, this one genuinely needs multiple
    # steps in flight at once, so it's the one place in this chapter that imports from think.py
    # -- it's inherently plan-specific, not a general primitive like Loop.
    def __init__(self, act, extractors=None, max_workers=8):
        self.act = act
        self.extractors = extractors or {}
        self.max_workers = max_workers

    def run(self, conversation):
        # Deciding to answer is itself an Act -- a graph of exactly one. Without this the loop
        # could never finish: the planner would conclude "done, here is the answer", nothing
        # would turn that decision into an assistant message, and control would come back here
        # to re-run an already-finished plan until the budget stopped it.
        last = pending_decision(conversation)
        if last is not None and last.get("type") == "respond":
            return self.act.run(conversation)

        steps = latest_plan(conversation)["steps"]
        deps = [step_dependencies(step) for step in steps]

        # $stepN is numbered globally, across every round -- a replan continues the count rather
        # than restarting it (see PlannerThink._continue_or_finish, which tells the planner
        # exactly which index to carry on from). So earlier rounds' outputs are seeded here as
        # already-satisfied dependencies, and this round's steps take the indices after them.
        # Resolving against a fresh per-round dict instead would silently make "$step0" mean
        # this round's first step, and a replan referring back to round one would read the
        # wrong value.
        step_outputs = getattr(conversation, "step_outputs", None)
        if step_outputs is None:
            step_outputs = conversation.step_outputs = []
        base = len(step_outputs)
        results = dict(enumerate(step_outputs))
        step_outputs.extend([None] * len(steps))
        started = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            in_flight = {}

            def launch_ready():
                for i, step in enumerate(steps):
                    if i in started or not (deps[i] <= results.keys()):
                        continue
                    started.add(i)
                    args = resolve_args(step, results)
                    future = pool.submit(self.act.execute, step["tool"], json.dumps(args))
                    in_flight[future] = i

            launch_ready()
            while in_flight:
                finished = next(concurrent.futures.as_completed(in_flight))
                i = in_flight.pop(finished)
                effect = finished.result()
                results[base + i] = effect["content"]
                step_outputs[base + i] = effect["content"]  # by index: completion order varies
                conversation.messages.append({"role": "tool", "content": self._phrase(effect, conversation)})
                launch_ready()

        return None

    def _phrase(self, effect, conversation):
        extractor = self.extractors.get(effect["tool"])
        if extractor is not None:
            return extractor(effect)
        return chat([{
            "role": "user",
            "content": (
                f"Conversation so far:\n{render(conversation)}\n\n"
                f"The tool `{effect['tool']}` was just called and returned:\n{effect['content']}\n\n"
                "State this result as one plain factual sentence, using the conversation only to "
                "phrase it correctly (e.g. which units, which entity). Do not speculate about why "
                "it matters, what happens next, or how it will be used. No preamble, no "
                "commentary -- the fact only."
            ),
        }])
