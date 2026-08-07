import json
import re

from llm import chat
from think import plan_progress, render


class Observe:
    def __init__(self, event=None, extractors=None):
        self.event = event
        self.extractors = extractors or {}

    def run(self, conversation):
        if self.event is not None:
            return {"role": "user", "content": self.event}

        effect = getattr(conversation, "pending", None)
        conversation.pending = None

        if isinstance(effect, list):
            # a fork's (or advance_branch's) results: one entry per branch
            content = "\n".join(f"[{b.get('goal') or b.get('id')}] {b['content']}" for b in effect)
            return {"role": "tool", "content": content}

        extractor = self.extractors.get(effect["tool"])
        if extractor is not None:
            content = extractor(effect)
        else:
            content = chat([{
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

        step_outputs = getattr(conversation, "step_outputs", None)
        if step_outputs is None:
            step_outputs = conversation.step_outputs = []
        step_outputs.append(effect["content"])

        observation = {"role": "tool", "content": content}

        next_decision = self._next_plan_decision(conversation)
        if next_decision is None:
            return observation

        conversation.messages.append(observation)
        return next_decision

    def _next_plan_decision(self, conversation):
        plan, completed = plan_progress(conversation)
        if plan is None or completed >= len(plan["steps"]):
            return None
        step = plan["steps"][completed]
        args = {k: self._resolve(v, conversation) for k, v in step["args"].items()}
        return {"role": "decision", "type": "tool_call", "tool": step["tool"], "content": json.dumps(args)}

    def _resolve(self, value, conversation):
        match = re.fullmatch(r"\$(?:step)?(\d+)(?:\.(\w+))?", value) if isinstance(value, str) else None
        if match is None:
            return value
        index, field = match.groups()
        raw = conversation.step_outputs[int(index)]
        return json.loads(raw)[field] if field is not None else raw
