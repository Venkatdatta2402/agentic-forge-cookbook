"""Composing stores, when one question needs more than one of them.

`02_stores.ipynb` introduces the four stores one at a time, each for a question the previous
one could not answer. Plenty of real questions need two:

    "why did the latest billing deploy fail?"     SQL finds *which* deploy. Prose says *why*.
    "what went wrong on anything Meera owns?"     The graph says what she owns. Vectors rank
                                                  what went wrong inside that.
    "has this happened before, and who fixed it?" Vectors find the incidents. The graph says
                                                  who was attached to them.

The shape underneath all three is the same, and it is not "pick the right store". Each store
takes a turn, and every turn either **answers** or **narrows what the next one may look at**.

So a step returns two things:

    answer      rows, for the caller
    narrowing   a small dict the next step is restricted to

and a pipeline is a list of steps with the narrowing threaded through.

**The narrowing is deliberately neutral**, not any one store's dialect:

    {"about": ["billing-api"], "since": "2026-11-18"}

because the stores disagree about everything else. Chroma wants `{"at": {"$gte": 20261118}}`
with an integer; SQL wants `since="2026-11-18"` as a string; Cypher wants `$at`. Each step
translates, which is the same lesson `03_operations` makes about the five verbs -- the
signatures are portable and nothing underneath them is.

**Empty narrowing means carry on unnarrowed.** A step that finds nothing to restrict by has
not failed; it has merely nothing to contribute, and the next step should see everything. The
alternative -- treat "narrowed to nothing" as "the answer is nothing" -- is right only when the
constraint is a *rule* rather than a hint, which is why `fail_open=False` exists and why a
tenant or permission filter is the case to use it for. Returning another tenant's rows because
a lookup came back empty is a bug; returning unranked prose because no service was mentioned
is a Tuesday.
"""

from datetime import date, timedelta


def _shift(day, days):
    return (date.fromisoformat(day[:10]) + timedelta(days=days)).isoformat()


def _chroma_where(narrowing):
    """Translate the neutral narrowing into Chroma's dialect, or None for no restriction."""
    clauses = []
    if narrowing.get("about"):
        values = list(narrowing["about"])
        clauses.append({"about": values[0] if len(values) == 1 else {"$in": values}})
    if narrowing.get("since"):
        clauses.append({"at": {"$gte": int(narrowing["since"][:10].replace("-", ""))}})
    if narrowing.get("until"):
        clauses.append({"at": {"$lte": int(narrowing["until"][:10].replace("-", ""))}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Step:
    """One store, taking its turn.

    `fail_open` decides what happens when the *incoming* narrowing is empty because an earlier
    step found nothing: True searches everything, False stops the pipeline.
    """

    fail_open = True
    name = "step"

    def run(self, question, narrowing):
        raise NotImplementedError


class FromProfile(Step):
    """A dict lookup. Answers by name; narrows nothing, because a key is not a filter."""

    name = "profile"

    def __init__(self, profile, *keys):
        self.profile = profile
        self.keys = keys

    def run(self, question, narrowing):
        answer = [(k, self.profile.read(k)) for k in self.keys]
        return answer, {}


class FromRecords(Step):
    """Filter, order, limit.

    Answers *which one* and *how many*. Narrows to the services it returned, and to a
    **window around** their dates -- not a lower bound at them.

    That distinction is the whole reason `window` exists, and it is easy to get wrong: an
    event's date is not a lower bound on its explanation. Ask why the deploy on the 19th
    failed and narrow to `at >= 2026-08-19`, and every note about the cause is excluded for
    predating the symptom -- which is the usual direction for a cause. Measured on this
    journal, that mistake returns a paging rule instead of an explanation.

    `window=None` narrows by service alone, which is the safe choice when the connection
    between an event and the notes about it is not a matter of days.
    """

    name = "records"

    def __init__(self, records, window=14, fail_open=True, **filters):
        self.records = records
        self.window = window
        self.fail_open = fail_open
        self.filters = filters

    def run(self, question, narrowing):
        # anything named on this step wins over what an earlier step narrowed to: an explicit
        # `service="billing-api"` is the caller being specific, not a default to be overridden
        filters = dict(self.filters)
        filters.setdefault("service", (narrowing.get("about") or [None])[0])
        filters.setdefault("since", narrowing.get("since"))
        rows = self.records.recent(**filters)
        if not rows:
            return [], {}

        produced = {"about": sorted({r["service"] for r in rows})}
        if self.window:
            produced["since"] = _shift(min(r["at"] for r in rows), -self.window)
            produced["until"] = _shift(max(r["at"] for r in rows), self.window)
        return rows, produced


class FromGraph(Step):
    """A walk.

    Answers *who is connected to what*. Narrows to the things it reached, which is how
    "anything Meera owns" becomes a filter on a vector search.
    """

    name = "graph"

    def __init__(self, graph, start=None, hops=2, at=None, fail_open=True):
        self.graph = graph
        self.start = start
        self.hops = hops
        self.at = at
        self.fail_open = fail_open

    def run(self, question, narrowing):
        starts = [self.start] if self.start else list(narrowing.get("about") or [])
        if not starts:
            return [], {}
        reached = {}
        for start in starts:
            reached.update(self.graph.connected(start, hops=self.hops, at=self.at))
        answer = sorted(reached.items(), key=lambda kv: (kv[1], kv[0]))
        return answer, {"about": sorted(set(starts) | set(reached))}


class FromJournal(Step):
    """Resemblance, restricted to whatever qualified.

    Answers *why*, which nothing else here can. Narrows to the services of the entries it
    returned, so a later graph step knows where to start walking.

    `03_operations` replaces `find_like` with `Recall` at exactly this point -- same position
    in the pipeline, same `where`, with ranking and a floor on top.
    """

    name = "journal"

    def __init__(self, journal, k=3, fail_open=True):
        self.journal = journal
        self.k = k
        self.fail_open = fail_open

    def run(self, question, narrowing):
        hits = self.journal.find_like(question, k=self.k, where=_chroma_where(narrowing))
        if not hits:
            return [], {}
        about = {self.journal.get(eid)["about"] for _, eid, _ in hits}
        return hits, {"about": sorted(about)}


class Pipeline:
    """Steps in order, with the narrowing threaded through.

    `run` returns the last step's answer. `trace` returns every step's, which is the thing
    worth looking at: a pipeline that returns nothing useful is nearly always a pipeline where
    an early step narrowed to something wrong, and the answer alone does not show you that.
    """

    def __init__(self, steps):
        self.steps = list(steps)

    def trace(self, question, narrowing=None):
        narrowing = dict(narrowing or {})
        out = []
        for step in self.steps:
            if not narrowing and out and not step.fail_open:
                out.append((step.name, "stopped: nothing qualified and fail_open is False", {}))
                break
            answer, produced = step.run(question, narrowing)
            out.append((step.name, answer, produced))
            if produced:
                narrowing = produced
        return out

    def run(self, question, narrowing=None):
        steps = self.trace(question, narrowing)
        return steps[-1][1] if steps else []
