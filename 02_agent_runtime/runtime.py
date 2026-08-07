import concurrent.futures
import json
import time

from chat import count_tokens
from llm import chat
from think import plan_progress, render, resolve_args, step_dependencies


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
        while not self.finished:
            reason = self._stop_reason(conversation)
            if reason is not None:
                return self._end_gracefully(conversation, reason)

            conversation.remaining_budget = self._remaining_budget(conversation)

            component = self.next()
            output = component.run(conversation)
            if output is not None:
                conversation.messages.append(output)
            self._iterations += 1

            if conversation.messages and conversation.messages[-1].get("role") == "assistant":
                return conversation

        return conversation

    def _tool_calls_made(self, conversation):
        return sum(1 for m in conversation.messages if m.get("role") == "decision" and m.get("type") == "tool_call")

    def _remaining_budget(self, conversation):
        # all four budgets Runtime tracks, not just time -- whatever decides to fork should be
        # able to split any (or all) of them across branches, not only max_seconds.
        tokens_used = sum(count_tokens(m["content"]) for m in conversation.messages if m.get("content"))
        return {
            "iterations": None if self.max_iterations is None else max(0, self.max_iterations - self._iterations),
            "tool_calls": None if self.max_tool_calls is None else max(0, self.max_tool_calls - self._tool_calls_made(conversation)),
            "tokens": None if self.max_tokens is None else max(0, self.max_tokens - tokens_used),
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

        return None

    def _end_gracefully(self, conversation, reason):
        last = conversation.messages[-1] if conversation.messages else None
        progress = f" Last step: {last['role']} -> {last['content']!r}." if last else " No steps completed."
        conversation.messages.append({
            "role": "assistant",
            "content": f"Stopping: {reason}.{progress} This is partial progress, not a final answer.",
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
    # `until` is content-driven (e.g. plan_done); `count` is a plain fixed number of passes --
    # not every use of Loop has (or needs) a predicate to check against.
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
        plan, _ = plan_progress(conversation)
        steps = plan["steps"]
        deps = [step_dependencies(step) for step in steps]
        results = {}
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
                results[i] = effect["content"]
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
