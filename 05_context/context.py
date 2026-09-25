"""Context is what one model call is given. Everything else is information that could have been.

The runtime keeps a lot of information around -- the conversation, the raw result of every tool
call, a scratchpad, notes, whatever a memory store holds. None of that is context yet. Context
is the slice of it handed to ONE operation: this Think, this Observe, this summarizer. A
different operation over the same run gets a different slice, and should.

This file turns what a run is holding into a flat list of `Item`s -- the *available*
information -- so that the other operations in this chapter have one shape to work on:

    available()     conversation, scratchpad, notes, constraints   -> [Item]
    selection.py    [Item] -> the ones this operation gets
    (retrieval)     a memory store -> more [Item], fetched rather than chosen

Nothing here decides anything. `available()` only says what exists and what kind of thing each
piece is -- a constraint, an observation, a raw tool result, a piece of task state -- because
every later decision depends on knowing that, and a message's `role` does not say it. A user
message can be a request, a constraint, or small talk; all three arrive as `role: "user"`.
"""

from dataclasses import dataclass

from chat import count_tokens

# What kind of thing an item is. The distinctions are the ones a selection actually acts on:
#
#   instruction  the system message -- how to behave, not information to reason over
#   request      what is being asked RIGHT NOW. The one item every operation needs
#   constraint   a rule stated once that must hold for the rest of the run
#   message      an ordinary turn of dialogue
#   observation  what Observe said a tool returned -- one sentence, already phrased
#   tool_result  what the tool actually returned -- exact, and often large
#   decision     a tool call Think has made that nothing has reported on yet. Only exists
#                mid-loop: once Observe phrases the result, the pair becomes one observation
#   state        where the task stands: the current plan, a scratchpad entry
#   fact         something known about the user or the world (chapter 1's `notes`)
#   memory       something RETRIEVED from a store -- present because a query found it
#   summary      several items compressed into one (compression.py). Stands in for them
KINDS = ("instruction", "request", "constraint", "message", "observation", "decision",
         "tool_result", "state", "fact", "memory", "summary")


@dataclass
class Item:
    """One piece of available information.

    `turn` is where it sits in the run, counted in messages -- so "how old is this" has an
    answer. Items from outside the conversation (a scratchpad entry, a retrieved memory) take
    the turn they were brought in at.

    `about` names what the item is *of*. Two items with the same `about` are two versions of
    the same thing -- a tool called twice with the same arguments, a plan revised, a scratchpad
    key rewritten -- and only the later one is current. Left empty when the item is not a
    version of anything.

    `pinned` means preserved: this item goes into every context built from this list,
    whatever the budget and whatever it scores. Constraints and the current request are
    pinned; almost nothing else should be.
    """

    kind: str
    content: str
    turn: int = 0
    about: str = ""
    source: str = "conversation"
    pinned: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {KINDS}")

    @property
    def tokens(self):
        return count_tokens(self.content)


def tokens(items):
    return sum(i.tokens for i in items)


def available(conversation, scratchpad=None, constraints=()):
    """Everything this run is holding, as `Item`s, oldest first.

    Reads the conversation the way the runtime writes it, not the way the API does:

      - the LAST user message is the `request`, and is pinned. Earlier user turns are
        `message`s -- they were requests once, and are history now.
      - a `tool_call` decision and the `tool` message after it are one event, so they become
        one `observation` whose `about` is the tool and its arguments. Same pairing
        `Think._build_messages` relies on.
      - that call's raw output is in `conversation.step_outputs`, in call order. It comes out
        as a separate `tool_result` with the same `about`, because it IS separate information:
        the observation is a sentence about the result, the result is the result.
      - a `plan` is `state`, about "plan", so a replan supersedes it. A `critique` is an
        observation about the work.
      - `decision`s other than tool calls, and `respond`s, are skipped: a fork decision is
        followed by its results, and a respond by the assistant message that carries it.

    `constraints` are rules stated outside the conversation, pinned. Chapter 1's
    `conversation.notes` come in as facts, and a chapter 4 `Scratchpad`'s entries as state.
    """
    messages = conversation.messages
    items = []

    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    raw_outputs = list(getattr(conversation, "step_outputs", None) or [])
    call_index = 0
    pending_call = None

    for turn, m in enumerate(messages):
        role, content = m.get("role"), m.get("content") or ""

        if role == "system":
            items.append(Item("instruction", content, turn, pinned=True))
        elif role == "user":
            if turn == last_user:
                items.append(Item("request", content, turn, pinned=True))
            else:
                items.append(Item("message", f"user: {content}", turn))
        elif role == "assistant":
            items.append(Item("message", f"assistant: {content}", turn))
        elif role == "decision" and m.get("type") == "tool_call":
            pending_call = (m["tool"], m["content"], turn)
        elif role == "tool" and pending_call is not None:
            tool, args, _ = pending_call
            about = f"{tool} {args}"
            items.append(Item("observation", content, turn, about=about))
            if call_index < len(raw_outputs):
                items.append(Item("tool_result", str(raw_outputs[call_index]), turn,
                                  about=about, source=f"step_outputs[{call_index}]"))
            call_index += 1
            pending_call = None
        elif role == "tool":
            # merged branch results, a search round's survivors -- no call of our own before it
            items.append(Item("observation", content, turn))
        elif role == "plan":
            items.append(Item("state", content, turn, about="plan"))
        elif role == "critique":
            items.append(Item("observation", content, turn, about="critique"))

    if pending_call is not None:
        # A tool call with nothing reported on it yet, which is where the loop is when `Observe`
        # runs: `Act` has carried the decision out, and the tool message that would fuse the two
        # into one observation is the thing `Observe` is about to produce. Left out, the only
        # component whose job is to describe that call could not see which call it was -- it
        # gets the tool's name from `conversation.pending` and never its arguments.
        tool, args, at = pending_call
        items.append(Item("decision", f"{tool}({args})", at, about=f"{tool} {args}"))

    now = len(messages)
    for note in getattr(conversation, "notes", None) or []:
        items.append(Item("fact", note, now, source="notes"))
    if scratchpad is not None:
        for key, value in scratchpad.entries.items():
            items.append(Item("state", f"{key}: {value}", now, about=f"scratchpad:{key}",
                              source="scratchpad"))
    for rule in constraints:
        items.append(Item("constraint", rule, 0, source="constraints", pinned=True))

    return sorted(items, key=lambda i: i.turn)


def render(items):
    """One line per item, labelled by kind. Good enough to read, and to send in a pinch.

    This is not construction. It says nothing about which item is an instruction the model
    should follow and which is data it should reason over, and it puts everything in one
    message. `04_context_construction.ipynb` replaces it with something that does.
    """
    return "\n".join(f"[{i.kind}] {i.content}" for i in items)
