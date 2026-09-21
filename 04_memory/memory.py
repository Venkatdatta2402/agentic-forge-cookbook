"""Working and short-term memory -- the parts that need no store at all.

Everything here is a list, a dict, or a file. That is not a simplification standing in for a
database: it is what these kinds of memory actually are. A scratchpad is a dict because a dict
is the right shape for something small, current, and thrown away at the end of the session.

The stores arrive in `02_stores.ipynb` and sit BESIDE this, for long-term memory. Nothing in
this file moves into one.
"""

import json
from pathlib import Path


class Scratchpad:
    """Short-term memory the agent is allowed to revise.

    The conversation cannot be revised. `Conversation.messages` only ever grows, so a plan that
    changes leaves BOTH versions sitting in context, and the model has to work out on its own
    which one is current -- usually by position, which is exactly the kind of reasoning it is
    worst at. Chapter 2's `replan` runs straight into this: the superseded plan is still there,
    word for word, several messages up.

    A scratchpad is keyed, so writing "plan" twice replaces rather than accumulates. That single
    difference is the whole reason it exists as something separate from the conversation, and it
    is why `render()` is short no matter how many times the agent changed its mind.

    Not persisted on purpose. What belongs here is working state for one session -- the current
    plan, a count, a partial result. Anything that should outlive the session is a fact, and
    facts go somewhere else (see `03_operations.ipynb`).
    """

    def __init__(self, **entries):
        self.entries = dict(entries)

    def write(self, key, value):
        self.entries[key] = value
        return self

    def read(self, key, default=None):
        return self.entries.get(key, default)

    def erase(self, key=None):
        if key is None:
            self.entries.clear()
        else:
            self.entries.pop(key, None)
        return self

    def render(self, header="Scratchpad"):
        """The whole pad as one block of text, ready to go into a prompt.

        Returns "" when empty rather than a header with nothing under it -- an empty section is
        worse than no section, because the model reads it as "there is nothing to know here"
        instead of "this has not been used yet".
        """
        if not self.entries:
            return ""
        lines = [f"{header}:"] + [f"- {k}: {v}" for k, v in self.entries.items()]
        return "\n".join(lines)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"Scratchpad({self.entries!r})"


def _as_dict(message):
    """A message that json.dump will accept.

    `messages` is a plain public list and this is where its contents stop being Python and start
    being text, so anything that is not already a dict gets `.model_dump()` called on it. Every
    path in this repo puts dicts in -- chapter 2's components hand `Runtime` a dict to append,
    and `Conversation` appends `message.model_dump(exclude_none=True)` -- so in practice this is
    a pass-through. It costs one isinstance check, against a crash at save time.
    """
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return dict(message)


def save_session(path, conversation):
    """Write a session's short-term memory to a file, and return what was written.

    Both halves go in. `messages` is the conversation itself; `notes` is what chapter 1's
    `Conversation.trim(remember=True)` extracted from the turns it dropped -- long-term content
    that has, until now, had nowhere long-term to live.
    """
    payload = {
        "messages": [_as_dict(m) for m in conversation.messages],
        "notes": list(getattr(conversation, "notes", [])),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_session(path, conversation):
    """Read a saved session back into a conversation, in place.

    Takes the conversation rather than building one, because a `Conversation` is constructed
    with its system message and that is the caller's decision, not this function's. Anything
    already in `messages` is replaced; a system message in the saved file comes back with it.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    conversation.messages = list(payload.get("messages", []))
    conversation.notes = list(payload.get("notes", []))
    return conversation
