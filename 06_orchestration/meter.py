"""Real calls and tokens for a stretch of code, from the API's own `usage` field.

Every component that talks to a model -- Think, Observe, extract -- goes through the one client in
`llm.py`, so wrapping that client's `create` sees all of them, including calls made inside other
agents and in other threads. Notebook 1 wrote this inline; it moved here when notebook 2 needed it too.

    with Meter() as meter:
        agent.run(task)
    meter.summary()   # {"calls": 3, "prompt": ..., "completion": ..., "total": ..., "seconds": ...}
"""

import time

import llm


class Meter:
    def __enter__(self):
        self.calls = []
        real = self._real = llm.client.chat.completions.create

        def create(*args, **kwargs):
            response = real(*args, **kwargs)
            self.calls.append({
                # the first user message says which task a call was working on, which is how calls
                # made by different agents at the same time are told apart afterwards
                "asked": next((m["content"] for m in kwargs.get("messages", [])
                               if m.get("role") == "user"), ""),
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
            })
            return response

        llm.client.chat.completions.create = create
        self._started = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.seconds = time.monotonic() - self._started
        llm.client.chat.completions.create = self._real

    def summary(self):
        prompt = sum(c["prompt"] for c in self.calls)
        completion = sum(c["completion"] for c in self.calls)
        return {"calls": len(self.calls), "prompt": prompt, "completion": completion,
                "total": prompt + completion, "seconds": round(self.seconds)}
