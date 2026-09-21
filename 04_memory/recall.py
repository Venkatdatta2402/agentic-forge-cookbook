"""Ranking a retrieve -- the part a vector index does not do for you.

`Journal.find_like` returns the k nearest vectors and nothing else. It cannot say "I have
nothing for that", it has no idea which of two equally-similar memories the agent has actually
been living in, and its scores sit in a narrow band that makes them awkward to combine.

    over-fetch   ask the index for more than you need -- the reordering below can promote a
                 memory that was not in the top k by similarity alone
    floor        drop candidates on RAW similarity, before any rescaling. The only step that
                 can return nothing, and it has to run first, because the rescaling below
                 destroys the number it depends on
    normalise    min-max each term across the surviving candidates, so a weight is a share of
                 the decision rather than a share of a range
    weight       combine, and take the k best

**What the second term is, and what it is not.**

It is *reinforcement*: how recently and how often this memory has been used. It is the
ACT-R effect -- what you reached for lately comes to mind more easily -- and it is measured
from the `accesses` table, not from when the entry was written.

It is **not** "the user asked for something current". That is a different question with a
different mechanism. Whether a fact is still true is settled by supersession -- replace the
memory, or give it a validity range like `Graph` edges have -- and never by nudging a stale
row a few places down a ranking, which leaves it in the results for the model to read anyway.

Confusing the two is easy and expensive, so: this term answers *which of these relevant
memories is live in the agent's working life*. It has no opinion about truth.

**Frequency does not compete with recency. It slows it down.**

The tempting shape is ACT-R's literal one -- sum an independently decaying term per access --
and it is wrong here, because it puts the two signals in opposition: ten uses six weeks ago
outscore one use yesterday, and a memory the agent has not touched in a month outranks the one
it read this morning. That is not what "frequently used things are remembered longer" means.

So frequency enters through **lambda itself**, which stops being a global constant and becomes
a property of each memory:

    lambda(f) = lambda_0 / (1 + alpha * f)

`lambda_0` is the fastest decay, applying to a memory used once. `f` is how many times it has
been accessed. `alpha` is a consolidation factor: how aggressively use protects a memory.

Since a half-life is the reciprocal of a decay rate, that is the same thing as

    half_life(f) = half_life_0 * (1 + alpha * f)

which is how it is computed below. Decay still runs from the *most recent* access; frequency
only sets how fast it runs. One signal answers *when did this last matter*, the other *how
firmly is it held*, and they multiply rather than trade.

Linear consolidation looks like it should run away -- sixty uses buys a 930-day half-life, and a
memory that surfaces more gets used more, which buys more half-life again. Damping `f` to
`ln(1 + f)` (`damped=True`) cuts that same half-life to 92 days, and against this journal the
two produce **identical rankings** at every weight tried.

That is a fact about this corpus, not about the formulas, and the reason is worth keeping in
view. Activation is `0.5 ** (age / half_life)`, which saturates once `age` is small next to
`half_life`. Three weeks against 930 days and three weeks against 92 days both sit near the top
of the curve -- 0.9845 and 0.8532 -- and a 0.13 gap does not reorder anything against a
similarity term carrying four fifths of the weight. The two forms only separate as memories get
old:

    days since last use      linear      ln(1+f)       gap
                     21      0.9845       0.8532      0.13
                    120      0.9144       0.4036      0.51
                    300      0.7996       0.1035      0.70

This journal spans about five months, so it cannot reach the region where the choice matters.
Treat the question as open at this scale rather than settled -- the same shape as BM25's rarity
term in `02_stores.ipynb`, which also has nothing to show on a corpus this size.

One thing the measurement does settle: damping is not what bounds the **runaway**. At
`w_reinforcement=0.5` a single memory takes first place on every query and every access in the
corpus, under both forms. The weight is the knob that controls that, not the shape of lambda.

A memory never used decays from its write date at the base half-life -- being written is the
one access it is known to have had.

The floor and the weights are both measured, not chosen. `0.6` separates answerable from
unanswerable questions on this journal with `gemini-embedding-001`; change either and it moves.
"""

import math
from datetime import date


def unit(values):
    """Min-max to [0, 1]. All-equal collapses to 0.5 rather than dividing by zero."""
    lo, hi = min(values), max(values)
    return [0.5 if hi == lo else (v - lo) / (hi - lo) for v in values]


class Recall:
    """Similarity and reinforcement, on a scale where the weights mean what they say.

    `over_fetch` multiplies the k asked of the index. It is close to free -- one query, the
    same graph walk -- and it matters because the scoring reorders: a memory outside the top k
    on similarity can finish inside it once reinforcement counts. Fetch k, score k, and all the
    scoring can do is shuffle what similarity already chose.

    `record_use` decides whether a retrieve counts as a use, and **it works itself out from
    the store**: a journal with a `summarize` hook has a second stage, so `open_memory` records
    and this does not; a journal without one has no second stage, so the retrieve is the last
    thing that could record and it does.

    That is the rule -- *the last stage records* -- and deriving it beats a flag, because the
    setting is not a preference. Getting it wrong in one direction reinforces memories for
    being listed and skipped; in the other, nothing is ever recorded and every activation
    falls back to the write date.

    Pass it explicitly only to *measure*: an instrument that mutates what it measures makes
    every result depend on how many times the earlier ones ran.

    `evict_below` is a **policy, and it is off by default**. Left as `None`, nothing is ever
    deleted for having decayed: memories rank lower as they go unused and stay in the store
    forever, which is the right answer for most agents -- a journal of a few thousand entries
    costs nothing worth reclaiming. Set it to an activation figure and `sweep()` will offer up
    everything below that line. Which figure is a question about your own corpus's timespan,
    not a constant: on a journal spanning five months at a thirty-day half-life, `0.01` matches
    nothing at all, because the oldest entry in it has only reached `0.0298`.
    """

    def __init__(self, journal, today, half_life=30.0, floor=0.6,
                 w_similarity=0.8, w_reinforcement=0.2, over_fetch=3,
                 alpha=0.5, damped=False, evict_below=None, record_use=None):
        self.journal = journal
        self.today = today
        self.half_life = half_life
        self.floor = floor
        self.w_similarity = w_similarity
        self.w_reinforcement = w_reinforcement
        self.over_fetch = over_fetch
        self.alpha = alpha
        self.damped = damped
        self.evict_below = evict_below
        # a store that writes summaries has a second stage, and the second stage is what
        # records. One without them does not, so the retrieve is the last thing that can.
        self.record_use = (journal.summarize is None) if record_use is None else record_use

    def age(self, at):
        return (date.fromisoformat(self.today) - date.fromisoformat(at)).days

    def half_life_of(self, uses):
        """lambda(f) = lambda_0 / (1 + alpha*f), expressed as its reciprocal."""
        f = math.log1p(uses) if self.damped else uses
        return self.half_life * (1 + self.alpha * f)

    def activation(self, entry_id, written_at):
        """Decay from the last time this memory mattered, at a rate set by how often it has.

        "Mattered" is the later of two things: when it was last **used**, and when it was last
        **written**. Rewriting a memory makes it current, so an entry corrected today starts
        again from full activation even though nobody has retrieved it yet.

        Pass `written` for `written_at`, not `at`. `at` is what the entry is about and can be
        months older than the row.
        """
        times = self.journal.accesses(entry_id)
        last = max(times + [written_at]) if times else written_at
        return 0.5 ** (self.age(last) / self.half_life_of(len(times)))

    # -- forgetting, which decay makes safe rather than performs ---------------------------

    def stale(self, threshold=None):
        """Entries whose activation has fallen below `threshold`, weakest first.

        Falls back to `evict_below`, which is `None` by default -- so unless a policy was
        chosen, nothing is ever stale and this returns an empty list.

        Decay on its own forgets nothing. A decayed row is still in the table, still costs
        index and storage, and is still eligible for every query. Ranking it low is not the
        same as removing it, and for most journals never removing anything is a perfectly
        good answer.

        What decay offers, if you do want a bound, is a defensible basis for one. A single
        low score is a moment; a score that has stayed low is evidence that nothing has
        needed this in a long time. That turns eviction from a guess into a report.

        Pinned entries never qualify, however far they decay.
        """
        threshold = self.evict_below if threshold is None else threshold
        if threshold is None:
            return []
        rows = self.journal.db.execute("SELECT id, at, pinned FROM entries")
        found = [(self.activation(r["id"], r["at"]), r["id"]) for r in rows if not r["pinned"]]
        return sorted(a for a in found if a[0] < threshold)

    def sweep(self, threshold=None, dry_run=True):
        """Delete what stayed decayed. Reports by default; deletes only when asked.

        The thing to keep in view is what activation actually measures: **use**. A memory
        nobody has needed for months looks identical to one that was never worth keeping, and
        the store cannot tell them apart. `pin()` is the way to say which is which, and the
        dry run exists so the list can be read before anything is irreversible.
        """
        victims = self.stale(threshold)
        if not dry_run:
            for _, entry_id in victims:
                self.journal.forget(entry_id)
        return victims

    def __call__(self, query, k=3, where=None):
        """Returns [(score, id, at, text)], best first -- or [] when nothing clears the floor."""
        hits = self.journal.find_like(query, k=k * self.over_fetch, where=where)

        # raw, before normalising: the only number here that means the same thing twice
        kept = [(sim, eid, self.journal.get(eid)["at"], text)
                for sim, eid, text in hits if sim >= self.floor]
        if not kept:
            return []

        nsim = unit([sim for sim, _, _, _ in kept])
        nact = unit([self.activation(eid, at) for _, eid, at, _ in kept])
        scored = [(self.w_similarity * a + self.w_reinforcement * b, eid, at, text)
                  for (_, eid, at, text), a, b in zip(kept, nsim, nact)]
        best = sorted(scored, reverse=True)[:k]
        if self.record_use and best:
            self.journal.used([eid for _, eid, _, _ in best], self.today)
        return best
