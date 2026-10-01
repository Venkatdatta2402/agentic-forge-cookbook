"""Memory, in the shape `02_agent_runtime` already accepts.

Nothing in the runtime changes to support any of this. A component is anything with
`.run(conversation)`, and a tool is anything `03_tools` can wrap -- so memory arrives as one
of each, by two different routes, for two different reasons.

    Retrieve      a COMPONENT. Runs before Think, every turn, whether or not anyone asked.
    remember      a TOOL. The model calls it when it decides something is worth keeping.
    open_memory   a TOOL. Fetches one memory in full, after Retrieve showed its summary.

**Why reading is injected and writing is a tool.**

Measured over five questions the journal could answer, with `recall` offered as a tool the
model could call: it called it four times. The fifth answered *"how long can search results be
stale?"* out of general knowledge, fluently, while the journal held the actual answer. That is
the failure mode worth designing against -- not an empty result, which is visible, but a
confident answer that never consulted anything, which is not.

Writing is the opposite. There is no cost to *not* writing something down, judging what is
worth keeping is a judgement, and a model is good at judgements. So the model decides.

**`EXTRACTORS` exists because of what `Observe` does by default.**

`Observe` sends a tool's result to a model and asks for it back as one plain sentence. For a
tool returning raw JSON that is a service. For anything memory returns it is damage, and the
two cases differ in how badly.

*A listing loses its ids, and that breaks the loop.* `recall` returns lines like
`#1  2026-11-06  Backfill job locking orders table...`, and the `#1` is the only thing making
the entry openable. Measured in a live loop, three runs: the ids survived once. In the two runs
they did not, one model called `recall` again -- there was nothing left to open -- and the
other answered without opening anything. **This is the entry that matters**, because losing it
does not degrade the answer, it removes the second stage entirely.

*A body loses detail, which may or may not matter.* Summarising an 868-character write-up gave
back about 130 characters. The compression is aimed at the question, so the two facts that
question asked for came through; asked how many of six specific facts survive, the answer was
two. Lossy, not usually fatal.

`Observe(extractors=...)` skips the model entirely for a named tool. The name says "extractor",
but what it means here is **this tool's output is already final -- do not restate it**.
"""


def last_question(conversation):
    """The most recent thing the user asked, or None."""
    return next((m.get("content") for m in reversed(conversation.messages)
                 if m.get("role") == "user"), None)


class Retrieve:
    """A loop component that puts memory in front of `Think` before it thinks.

    Placed anywhere before `Think` in a `Runtime` loop. Returns a message or `None`, which is
    the whole contract -- the same one `Observe` satisfies.

    It returns `None` rather than an empty note when nothing clears the floor, because a line
    reading "no relevant memories" is a line the model has to read on every turn that has
    none, and it says nothing an absent line does not.

    **What it injects depends on the store, not on a setting.** A journal that writes summaries
    has a second stage, so this shows summaries and lets `open_memory` fetch a body. A journal
    without them has no second stage, so the entry itself is what goes in.

    Getting that wrong in the generous direction is worse than it looks. Injecting full bodies
    from a summarising store puts every candidate in the prompt -- measured here, 2,136
    characters for three memories -- and the model then has no reason to open anything, because
    it already has everything. Two-stage recall stops happening and nothing reports that it
    stopped: the answers stay correct and the bill quietly triples.
    """

    def __init__(self, engine, k=3, header="From earlier sessions:"):
        self.engine = engine
        self.k = k
        self.header = header

    def run(self, conversation):
        question = last_question(conversation)
        if not question:
            return None
        hits = self.engine(question, k=self.k)
        if not hits:
            return None

        lines = []
        for _, entry_id, at, text in hits:
            row = self.engine.journal.get(entry_id)
            shown = (row["summary"] if row and row["summary"] else text)
            lines.append(f"#{entry_id}  {at}  {shown}")

        note = self.header + "\n" + "\n".join(lines)
        if any(self.engine.journal.get(e)["summary"] for _, e, _, _ in hits):
            note += "\n\nThese are summaries. Call open_memory(memory_id=N) for the full text."
        return {"role": "tool", "content": note}


def memory_tools(journal, today, write=None):
    """`remember` and `open_memory`, closed over a store.

    The closure is the one thing memory genuinely needs that the runtime does not provide.
    Nothing on `Conversation` holds a store handle, and a tool's signature is what the model
    sees -- so a `journal` parameter would be a parameter the model has to fill in, with a
    value it has no way to know. It is captured here instead, which is why these are built by
    a function rather than declared at module level.

    `write` is where `04_types_of_memory`'s routing plugs in: a callable taking the text and
    doing whatever that kind of memory needs -- appending an event, or running the
    add/update/ignore check for a fact -- and returning a line to tell the model. Left out, a
    memory is appended, which is right for an event and leaves a fact's duplicate check to
    whoever wires this up.
    """
    from typing import Annotated

    from tools import tool

    def append(text, about):
        entry_id = journal.write([(today, about, "concluded", text)], now=today)[0]
        return f"remembered as #{entry_id}"

    save = write or append

    @tool
    def remember(text: Annotated[str, "the memory, in one or two sentences"],
                 about: Annotated[str, "what it concerns, e.g. a service name"]) -> str:
        """Write something to memory that should outlive this session."""
        return save(text, about)

    @tool
    def open_memory(memory_id: Annotated[int, "an id from the memories above"]) -> str:
        """Read one memory in full. Call this before answering from a summary."""
        row = journal.get(memory_id)
        if row is None:
            return f"No memory #{memory_id}."
        journal.used([memory_id], today)      # being opened is use; being listed is not
        return row["text"]

    return [remember, open_memory]


def _already_final(effect):
    return effect["content"]


# Pass to `Observe(extractors=EXTRACTORS)`. Each entry means: this tool's result is the
# artifact, not a report about one, so hand it over untouched and make no model call.
#
# `recall` is here even though `Retrieve` -- the injected route -- never reaches `Observe` at
# all. Anyone exposing recall as a tool instead needs it far more than `open_memory` does, and
# an entry for a tool that is not registered costs nothing.
EXTRACTORS = {
    "recall": _already_final,
    "open_memory": _already_final,
}
