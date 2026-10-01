"""The four operations, assembled -- and assembled as a system rather than a pipeline.

A pipeline would be: everything goes through every stage, in one order, every call. That is
wrong twice over. Compression is not a stage: it costs model calls and loses information, and
most calls need none of it. And selection and retrieval are not sequential: neither needs the
other's output, so making one wait for the other buys nothing.

What actually happens for one component, on one call:

    available()                     everything the run holds, as items
       |
       +-- why_compress()?          a trigger, not a stage. Nothing fires -> nothing runs
       |      yes -> compact() / truncate()
       |
       +-- boundary                 what selection cannot say: "this pass only". First, so
       |                            selection chooses within it rather than against it
       |
       +-- select() || retrieve()   at the same time, because neither needs the other
       |
       v
    to_messages() -> a component      OR      build() -> a direct API call

The last line is the one thing this file does not decide. Construction has two targets, and
they are genuinely different: a chapter 2 component expects a `conversation`, so its context is
rebuilt into the runtime's own message shapes (`focus.to_messages`); a call we make ourselves
wants API messages with roles and a fence (`construction.build`). Same chosen items, two
renderings, picked by who is consuming it.

`Policy` is what one component gets, written as data. `ContextSystem` holds what is shared --
the scratchpad, the rules, a memory store, the history limit -- and applies a policy per call.
Every application returns a `Trace`, because "what did this component actually get, and why"
is a question you will ask far more often than you expect.
"""

from dataclasses import dataclass, field

from compression import compact, compressible, truncate, why_compress
from context import available, by_route, tokens
from focus import Focus
from selection import gather, overlap, retrieved, select


@dataclass
class Policy:
    """What one component is given. Data, so it can be read, compared and changed in one place.

    `boundary` is for what a score cannot express -- `focus.current_pass` is the example: this
    iteration of the loop, whatever it scores. It runs LAST, after selection and retrieval,
    because a boundary is a statement about what is admissible at all.

    `compress` is what this component is allowed to do when a trigger fires: "compact" for
    conversation compression, "truncate" to cut oversized items down, or None to leave the
    context alone and let selection drop what does not fit.
    """

    kinds: tuple = None
    budget: int = None
    keep_last: int = 0
    query: str = None
    relevance: object = overlap
    retrieve: object = None            # callable(query) -> [Item], usually a chapter 4 Recall
    retrieve_k: int = 2
    compress: str = None              # "compact" | "truncate" | None
    # how much of the context must be old enough to compact before it is worth the model calls.
    # A trigger can fire on a run that has nothing compactable: everything pinned, task state,
    # or inside the recent window compaction leaves alone.
    compress_when: float = 0.4
    boundary: object = None           # callable(items, conversation) -> items
    # {group: fewest items that must survive}, matched against a route or a kind. What a
    # component must not be without: a skills catalogue it is expected to use, the plan it is
    # working to. See `select(floors=)` for the failure this exists for.
    floors: dict = None

    def __str__(self):
        name = lambda f: getattr(f, "__name__", type(f).__name__)
        said = [f"{field}={value}" for field, value in
                (("kinds", self.kinds), ("budget", self.budget), ("keep_last", self.keep_last or None),
                 ("compress", self.compress)) if value is not None]
        if self.retrieve is not None:
            said.append(f"retrieve={name(self.retrieve)}(k={self.retrieve_k})")
        if self.relevance is not overlap:
            said.append(f"relevance={name(self.relevance)}")
        if self.boundary is not None:
            said.append(f"boundary={name(self.boundary)}")
        return "Policy(" + ", ".join(said) + ")"


@dataclass
class Trace:
    """What happened on one application of a policy."""

    available: int = 0
    compressed: str = ""
    bounded: int = None
    selected: int = 0
    retrieved: int = 0
    final: int = 0
    reasons: list = field(default_factory=list)
    routes: dict = field(default_factory=dict)      # where the final tokens came from
    # routes that had something to offer and ended with nothing. Reported whether or not a
    # floor was asked for, because the failure worth naming is not that a route was dropped --
    # that is often right -- but that it was dropped without anyone seeing it.
    emptied: list = field(default_factory=list)

    def __str__(self):
        parts = [f"available {self.available}t"]
        parts.append(f"compressed ({self.compressed})" if self.compressed else "no compression")
        if self.bounded is not None:
            parts.append(f"bounded {self.bounded}t")
        parts.append(f"selected {self.selected}t")
        if self.retrieved:
            parts.append(f"retrieved {self.retrieved}t")
        parts.append(f"-> {self.final}t")
        if self.routes:
            parts.append("(" + ", ".join(f"{name} {n}t" for name, n in self.routes.items()) + ")")
        if self.emptied:
            parts.append("!! emptied: " + ", ".join(self.emptied))
        return "  ".join(parts)


class ContextSystem:
    """Everything shared between components, and the one method that applies a policy."""

    def __init__(self, scratchpad=None, constraints=(), recall=None, history_limit=None,
                 profile=None, catalogue=()):
        self.scratchpad = scratchpad
        self.constraints = constraints
        self.recall = recall
        self.history_limit = history_limit
        # the pasted and always-present routes. Held here rather than per policy because every
        # component draws on the same profile and the same catalogue; what differs is how much
        # of them survives that component's budget.
        self.profile = profile
        self.catalogue = catalogue
        self.traces = []
        # (messages at the time, the compacted items that stand in for them). Notebook 3 was
        # emphatic that compression must never edit the record, and `compact()` obeys that: it
        # returns a new list and leaves `conversation.messages` alone. The cost of that showed up
        # the first time a loop ran -- every `Think` call recompacted the same history from
        # scratch, two model calls each, six in a two-turn run. So the result is kept here
        # instead: compact once, reuse it, and recompact only when the tail since then is itself
        # worth compacting.
        self._checkpoint = None

    def apply(self, items, conversation, policy):
        trace = Trace(available=tokens(items))

        # 1. compression, if something calls for it. `why_compress` reports; the policy decides
        #    what this component may do about it; most calls do neither.
        trace.reasons = why_compress(items, policy.budget or 10 ** 9, self.history_limit)
        if trace.reasons and policy.compress == "compact":
            enough = lambda part: tokens(part) >= policy.compress_when * tokens(items)
            at, kept = self._checkpoint or (None, [])
            tail = [i for i in items if at is not None and i.turn >= at and not i.pinned]
            if at is not None and not enough(compressible(tail)):
                # the old history is already compacted; only what has happened since is new
                items = [i for i in items if i.pinned or i.turn >= at] + kept
                trace.compressed = "reused"
            elif enough(compressible(items)):
                items = compact(items).items
                self._checkpoint = (len(conversation.messages),
                                    [i for i in items if not i.pinned])
                trace.compressed = "compact"
            else:
                trace.reasons.append(
                    f"not compacted: only {tokens(compressible(items))}t of {tokens(items)}t is "
                    "old enough to replace")
        elif trace.reasons and policy.compress == "truncate" and policy.budget:
            items = [truncate(i, policy.budget // 2, keep="ends") if i.tokens > policy.budget else i
                     for i in items]
            trace.compressed = "truncate"

        # what selection is choosing from, measured AFTER compression: compaction moves the run
        # and the raw results into `compressed` legitimately, and reading this any earlier
        # reported "emptied: called" every time a compaction did its job.
        offered = by_route(items)

        # 2. the boundary, BEFORE selection. It says what is admissible at all, so scoring
        #    anything outside it is work thrown away -- measured: with the boundary applied last,
        #    a single Observe context scored 2002 tokens and then discarded all but 22 of them.
        #    Selection now chooses within the boundary instead of against it.
        if policy.boundary is not None:
            items = policy.boundary(items, conversation)
            trace.bounded = tokens(items)

        # 3. selection and retrieval, at the same time. A retrieval that returns nothing costs
        #    the wait; a retrieval nobody asked for costs a query, so `retrieve` is per policy.
        #    Retrieved memories are not subject to the boundary: a memory is not part of this
        #    pass of the loop, and a policy that asks for one is asking for it on purpose.
        query = policy.query or next((i.content for i in reversed(items)
                                      if i.kind == "request"), "")
        chosen, memories = gather(
            lambda: select(items, budget=policy.budget, query=query, kinds=policy.kinds,
                           relevance=policy.relevance, keep_last=policy.keep_last,
                           floors=policy.floors),
            *([lambda: retrieved(policy.retrieve, query, k=policy.retrieve_k)]
              if policy.retrieve is not None else []),
        )
        trace.selected, trace.retrieved = chosen.tokens, tokens(memories)
        items = chosen.kept + memories

        trace.final = tokens(items)
        trace.routes = by_route(items)
        # a boundary is a deliberate "this pass only", so the routes it excludes are not news
        if policy.boundary is None:
            trace.emptied = [name for name in offered if name not in trace.routes]
        self.traces.append(trace)
        return items

    def available(self, conversation):
        return available(conversation, self.scratchpad, self.constraints,
                         profile=self.profile, catalogue=self.catalogue)

    def focus(self, component, policy):
        """`component`, wrapped so that every call gets a context built by this system."""
        return Focus(component, scratchpad=self.scratchpad, constraints=self.constraints,
                     profile=self.profile, catalogue=self.catalogue,
                     prepare=lambda items, conversation: self.apply(items, conversation, policy))

    def items(self, conversation, policy):
        """The context a policy produces, without a component -- for a call you make yourself."""
        return self.apply(self.available(conversation), conversation, policy)
