"""A scripted model, so the real loop, tools and governor run with no tokens spent."""

import json
from itertools import count

from stats_desk.core.llm import Reply, ToolCall

_ids = count(1)


def call(name, **arguments):
    """One tool call, as the model would emit it."""
    return Reply(None, [ToolCall(f"call_{next(_ids)}", name, json.dumps(arguments))],
                 "tool_calls", 300, 40)


def calls(*pairs):
    """Several tool calls in one turn: calls(("ask_a", {...}), ("ask_b", {...}))."""
    return Reply(None, [ToolCall(f"call_{next(_ids)}", n, json.dumps(a)) for n, a in pairs],
                 "tool_calls", 300, 60)


def plan(*tasks, later=(), approach="Test what the question asks, then follow up."):
    """ask_analysts' arguments (agents/plan.py)."""
    return {"approach": approach, "tasks": list(tasks), **({"later": list(later)} if later else {})}


def say(text):
    return Reply(text, [], "stop", 300, 40)


class Scripted:
    """Replies in order. A step may be a function of the request, to answer from what it saw."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("the model was called more times than the script allows")
        step = self.steps.pop(0)
        reply = step(request) if callable(step) else step
        return reply, {}

    def last_tool_result(self, request=None):
        request = request or self.requests[-1]
        return next(m["content"] for m in reversed(request["messages"]) if m["role"] == "tool")
