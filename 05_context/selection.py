"""Choosing which of the information you already have goes into one operation's context.

Selection works on what is already in hand -- `context.available()`'s list. Retrieval is the
other way in: it asks a memory store a question and gets back things that were NOT in hand.
They are different operations with different failure modes, so they are different code:

    select()      [Item] in, [Item] out. Local, cheap, and it can only ever drop things.
    retrieved()   a chapter 4 `Recall` and a query in, [Item] out. It can add things, and it
                  can be wrong about what is relevant in ways a selection over known items
                  cannot.
    gather()      runs one selection and any number of retrievals at the same time. Neither
                  needs the other's output, so neither waits for it.

`select()` runs in a fixed order, and the order is the design:

    1. supersede  items with the same `about` are versions of one thing; keep the newest.
                  An identity question, not a scoring one -- an old plan is not "less
                  relevant" than the new one, it is wrong.
    2. pin        pinned items go in, always. If they alone overflow the budget, selection
                  refuses: it can drop things, but it cannot make a constraint smaller.
                  That is compression's job.
    3. score      what is left is ranked by relevance to the request, importance of its kind,
                  and recency -- each min-max normalised across the candidates, as in chapter
                  4's `Recall`, so a weight is a share of the decision.
    4. fill       best first, until the budget is spent.

The result keeps the original order. Selection decides WHAT goes in; where each thing sits in
the model's input is construction's decision, and doing it here would decide it by accident.
"""

import concurrent.futures
import re
from dataclasses import dataclass, field

from context import Item, route, tokens
from recall import unit

# Importance is a property of what KIND of thing an item is, fixed in advance. Not rated by a
# model per item: chapter 4 measured that, and a model asked "how important is this, 1-10"
# answers 6-8 for nearly everything -- a call per item that moves no ranking.
#
# Pinned kinds are listed for completeness; they never reach scoring.
IMPORTANCE = {
    "instruction": 1.0,
    "request": 1.0,
    "constraint": 1.0,
    "state": 0.8,        # where the task stands -- what the next step starts from
    "decision": 0.8,     # the call in flight: what the next observation is about
    "fact": 0.7,
    "summary": 0.7,      # stands in for a whole stretch of the run that is no longer there
    "observation": 0.6,  # a sentence per tool call, already phrased for reading
    "memory": 0.6,
    "message": 0.4,
    "tool_result": 0.3,  # exact but large; the observation usually says what matters in it
}

STOPWORDS = frozenset(
    "a an and are as at be but by can did do does for from had has have how i if in is it its "
    "me my of on or so that the them then there this to was we were what when where which who "
    "why will with you your about into just should would could".split()
)


def words(text):
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def overlap(query, text):
    """Share of the query's words that appear in the text. The cheapest relevance there is.

    Knows nothing about meaning: "slow" and "latency" score zero against each other. Any
    callable `(query, text) -> float` can replace it -- see `select(relevance=)`.
    """
    q = words(query)
    return len(q & words(text)) / len(q) if q else 0.0


@dataclass
class Selection:
    kept: list
    dropped: list = field(default_factory=list)   # [(item, why)]

    @property
    def tokens(self):
        return tokens(self.kept)

    def report(self, width=70):
        lines = [f"kept {len(self.kept)} items, {self.tokens} tokens"]
        for item in self.kept:
            mark = "pinned" if item.pinned else "      "
            lines.append(f"  + {mark} t{item.turn:<3} {item.kind:<11} {_short(item.content, width)}")
        if self.dropped:
            lines.append(f"dropped {len(self.dropped)}")
            for item, why in self.dropped:
                lines.append(f"  -        t{item.turn:<3} {item.kind:<11} "
                             f"{_short(item.content, width - 20)}  <- {why}")
        return "\n".join(lines)


def _short(text, width):
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def supersede(items):
    """Newest version of every `about`, and everything with no `about` at all.

    Returns (current, [(old_item, why)]).
    """
    newest = {}
    for item in items:
        if item.about:
            # the same `about` can hold an observation AND a raw result; they are versions
            # only of their own kind, never of each other
            newest[(item.kind, item.about)] = item.turn
    current, dropped = [], []
    for item in items:
        latest = newest.get((item.kind, item.about)) if item.about else None
        if latest is not None and item.turn < latest:
            dropped.append((item, f"superseded at t{latest}"))
        else:
            current.append(item)
    return current, dropped


def select(items, budget=None, query=None, kinds=None, relevance=overlap, keep_last=0,
           floors=None, w_relevance=0.5, w_importance=0.3, w_recency=0.2, half_life=8):
    """The items this operation should be given, within `budget` tokens.

    `query` is what relevance is measured against. Defaults to the request item's content,
    which is the right thing for a Think; an Observe or a summarizer may want something else.

    `kinds`, if given, is the only kinds this operation can see at all -- a filter applied
    before anything is scored. An operation checking a number wants `tool_result`; one
    answering the user usually does not.

    `half_life` is in turns: an item that many turns old scores half on recency.

    `floors` is the other guarantee: `{"catalogue": 2, "state": 1}` means at least that many
    items from that group survive, whatever they score, with the group's best-scoring ones
    chosen. A name is matched against the item's route (`context.route`) or its kind, so it can
    speak about either.

    It exists because of a measured failure. Once a skills catalogue became items, selection
    dropped **all of it** at a 700-token budget: the entries are unpinned, they scored below the
    incident's own findings, and the model was then asked what to do about a connection pool
    with `resize_pool` invisible to it. A dropped fact makes an answer worse; a dropped
    catalogue entry removes something the agent could do, and nothing in the answer says so.

    Like the pinned items, a floor that does not fit the budget is refused rather than quietly
    broken.

    `keep_last` is a GUARANTEE, and `w_recency` is not. Recency is one term in a score: an old
    item with strong relevance outranks a recent one, and the greedy fill can skip a recent item
    that does not fit while taking a smaller older one. Weighting recency higher does not fix
    that -- it only shifts the odds. `keep_last=N` takes the N most recent items (after
    superseding) and treats them exactly like pinned ones: in, whatever they score. If they do
    not fit the budget, selection refuses rather than quietly dropping one, for the same reason
    it refuses on pinned items.
    """
    current, dropped = supersede(items)

    if kinds is not None:
        dropped += [(i, f"kind {i.kind} not used here") for i in current
                    if i.kind not in kinds and not i.pinned]
        current = [i for i in current if i.kind in kinds or i.pinned]

    # the last `keep_last` items join the pinned ones as must-keeps, in their own order
    recent = sorted(current, key=lambda i: i.turn)[-keep_last:] if keep_last else []
    recent_ids = set(map(id, recent))
    pinned = [i for i in current if i.pinned or id(i) in recent_ids]
    candidates = [i for i in current if not (i.pinned or id(i) in recent_ids)]

    if budget is not None and tokens(pinned) > budget:
        what = "pinned items" if not keep_last else f"pinned items and the last {keep_last}"
        raise ValueError(
            f"{what} alone are {tokens(pinned)} tokens, over the budget of {budget}. "
            "Selection can leave things out; it cannot make what must go in any smaller. "
            "Raise the budget or compress them."
        )

    if query is None:
        query = next((i.content for i in reversed(items) if i.kind == "request"), "")

    chosen = set(map(id, pinned))
    if candidates:
        now = max(i.turn for i in items)
        rel = unit([relevance(query, i.content) for i in candidates])
        imp = unit([IMPORTANCE[i.kind] for i in candidates])
        rec = unit([0.5 ** ((now - i.turn) / half_life) for i in candidates])
        scored = sorted(
            ((w_relevance * r + w_importance * m + w_recency * c, n, item)
             for n, (item, r, m, c) in enumerate(zip(candidates, rel, imp, rec))),
            key=lambda s: (-s[0], s[1]),
        )
        # floors, chosen AFTER scoring so each group keeps its best rather than its first
        promised = []
        for name, least in (floors or {}).items():
            group = [item for _, _, item in scored
                     if name in (route(item), item.kind) and id(item) not in chosen]
            for item in group[:least]:
                chosen.add(id(item))
                promised.append(item)
        if budget is not None and tokens(pinned) + tokens(promised) > budget:
            raise ValueError(
                f"pinned items and the floors {floors} come to "
                f"{tokens(pinned) + tokens(promised)} tokens, over the budget of {budget}. A "
                "floor is a promise, so this is refused rather than half-kept. Raise the budget, "
                "lower a floor, or compress what they hold."
            )

        spent = tokens(pinned) + tokens(promised)
        for score, _, item in scored:
            if id(item) in chosen:
                continue
            if budget is None or spent + item.tokens <= budget:
                chosen.add(id(item))
                spent += item.tokens
            else:
                dropped.append((item, f"over budget (score {score:.2f})"))

    kept = [i for i in current if id(i) in chosen]
    return Selection(kept, sorted(dropped, key=lambda d: d[0].turn))


SUMMARY_HINT = ("The memories above are summaries. Call open_memory(memory_id=N) for the full "
                "text of one before answering from it.")


def retrieved(recall, query, k=3, turn=0, hint=True):
    """A chapter 4 `Recall` asked `query`, its hits as `memory` items.

    `Recall` already did the ranking -- similarity floor, reinforcement, top k -- so this does
    not rank again. It only says what came back and where from.

    **What goes in depends on the store, not on a setting** -- the rule `04_memory`'s `Retrieve`
    settles. A journal that writes summaries has a second stage, so the summary goes in and
    `open_memory` fetches a body when the model wants one. A journal without them has no second
    stage, so the entry itself goes in. Injecting bodies from a summarising store is the
    expensive mistake: 2,136 characters for three memories against 363, measured in
    `04_memory/05_memory_in_the_loop`, and the model then has no reason to open anything, so
    two-stage recall silently stops happening while the answers stay right.

    **The id is in the CONTENT, not only in `about`.** It is what makes the entry openable, so
    it has to survive every renderer: `about` only reaches the model when construction is asked
    for labels, and `04_memory` measured ids surviving one live run in three once a listing had
    been through a paraphrase. A fact the loop depends on does not belong in an attribute.
    """
    items, summarised = [], False
    for _, eid, at, text in recall(query, k=k):
        row = recall.journal.get(eid)
        summary = row["summary"] if row is not None and row["summary"] else None
        summarised |= summary is not None
        items.append(Item("memory", f"#{eid}  {at}  {summary or text}", turn,
                          about=f"journal#{eid}", source=f"journal#{eid}"))
    if items and summarised and hint:
        # An instruction about how to use the data, so it renders as one -- and unpinned, because
        # it is true only for the calls that actually retrieved a summary. `instruction` carries
        # the top importance, so a budget has to be very tight before it goes.
        items.append(Item("instruction", SUMMARY_HINT, turn, source="retrieval"))
    return items


def gather(select_fn, *retrieve_fns):
    """Run one selection and any number of retrievals at once; return (Selection, [Item]).

    Each argument is a zero-argument callable, so the caller decides everything about how
    each one runs -- budget, query, k -- and this only decides WHEN: all at the same time.
    Threads rather than asyncio for the same reason as chapter 2's `Graph`: a notebook already
    runs an event loop, and this work is blocking I/O.
    """
    with concurrent.futures.ThreadPoolExecutor() as pool:
        selection = pool.submit(select_fn)
        memories = [pool.submit(fn) for fn in retrieve_fns]
        return selection.result(), [item for f in memories for item in f.result()]
