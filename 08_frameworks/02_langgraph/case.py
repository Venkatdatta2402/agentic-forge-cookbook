"""A case that can wait for people for days, built from this repo's parts.

The one thing this notebook needs that no chapter had is a run that stops, waits for a person, and
is picked up later by another process. `02_agent_runtime`'s `Runtime` cannot: a run lives in one
process. But the pieces are here already, and they suggest a different design from LangGraph's:

    the case      chapter 6's `state.Shared` board. Each field has ONE owner who may write it: the
                  model that reads invoices owns `invoice`, the investigator owns `recommendation`,
                  Tomasz owns `approval:tomasz`, and the desk's own code owns the money -- `amount`,
                  `signers`, `status`. A write by anyone else is refused, not recorded.
    on disk       every write is also a row in chapter 4's `Records`, a SQLite table. The board is
                  rebuilt from its rows, so another process sees exactly what this one wrote -- and
                  the rows are the audit trail: who set what, from what, to what.
    the steps     plain functions of the case. A step returns the name of the next one, or a `Wait`.
    waiting       a `Wait` ENDS its step. It names the field that will hold the answer, the person
                  who owns that field, and the step that continues once it is written. Answering is
                  that person writing that field; then the named step runs.

The last line is where this differs from LangGraph and from this notebook's first version of it.
There, a node that waits is run again from its first line when the answer comes, so anything it
did before the wait happens twice, and answers are matched to questions by their position. Here
nothing is ever re-run to get back to where it was: a step that asks is finished, and the answer
starts the next one. And an answer is a write to a named field, so it lands where it belongs, and
only its owner can make it.

What this does NOT do, and LangGraph does: run steps in parallel, stream, rewind and fork a run,
cache, time out a step, or serve any of it over an API.
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _chapter in ("04_memory", "06_orchestration"):
    if str(_REPO / _chapter) not in sys.path:
        sys.path.append(str(_REPO / _chapter))

from state import Denied, Shared  # noqa: E402,F401 -- Denied is re-exported for callers
from stores import Records  # noqa: E402

END = "__end__"
BOOKKEEPING = ("next", "waiting")     # where the case is, rather than what it has established


@dataclass
class Wait:
    """What a step returns when it needs somebody: the field for the answer, who owns it, what comes next."""

    field: str
    party: str
    then: str
    shown: dict = field(default_factory=dict)      # what the person is shown: the question, the packet

    def ask(self):
        return {"field": self.field, "party": self.party, "then": self.then, **self.shown}


class CaseFile(Shared):
    """A chapter 6 board whose every write is also a row in a chapter 4 `Records` table.

    `Records` has columns for a feed of events, and a write to a board is one: `kind` holds the
    field, `status` who wrote it, `service` which case, `text` the value as JSON.
    """

    def __init__(self, log, thread, **board):
        super().__init__(**board)
        self.log, self.thread = log, thread
        # Rebuilt from the rows, in the order they were written. The owner check is not run again:
        # it ran when each row was written, which is the only time it means anything.
        for row in log.db.execute("SELECT kind, status, text FROM records WHERE service = ? ORDER BY id",
                                  (thread,)):
            before = self.values.get(row["kind"])
            self.values[row["kind"]] = json.loads(row["text"])
            self.history.append({"author": row["status"], "field": row["kind"], "from": before,
                                 "to": self.values[row["kind"]]})

    def set(self, author, field, value):
        if field not in self.owner:
            # chapter 6's board lets anyone write a field nobody owns. A case file holding money does not.
            raise Denied(f"{field} has no owner, so nobody may write it")
        before = super().set(author, field, value)          # raises Denied for anyone but the owner
        self.log.write(datetime.now().isoformat(), field, author, self.thread, json.dumps(value))
        return before


class Desk:
    """Steps over case files. Every case lives in one SQLite file, so any process can carry one on."""

    def __init__(self, steps, start, fields, owner, path=":memory:"):
        self.steps, self.start = dict(steps), start
        self.fields, self.owner = fields, owner
        self.log = Records(path)

    def case(self, thread):
        return CaseFile(self.log, thread, fields=self.fields, owner=self.owner)

    def open(self, thread, opening):
        """Start a case with the desk's opening facts. Returns what it waits on, or None if it finished."""
        case = self.case(thread)
        for name, value in opening.items():
            case.set("desk", name, value)
        case.set("desk", "next", self.start)
        return self._go(case)

    def answer(self, thread, party, field, value):
        """`party` answers the case's question by writing `field`. Then the step the question named runs."""
        case = self.case(thread)
        waiting = case.values.get("waiting")
        if not waiting or waiting["field"] != field:
            raise ValueError(f"{thread} is not waiting on {field!r} (waiting on: {waiting})")
        case.set(party, field, value)                       # only the field's owner gets past this
        case.set("desk", "waiting", None)
        case.set("desk", "next", waiting["then"])
        return self._go(case)

    def resume(self, thread):
        """Carry on a case whose step failed part-way: that step runs again."""
        return self._go(self.case(thread))

    def _go(self, case):
        while case.values.get("next") not in (None, END):
            after = self.steps[case.values["next"]](case)
            if isinstance(after, Wait):
                case.set("desk", "waiting", after.ask())
                case.set("desk", "next", None)
                return case.values["waiting"]
            case.set("desk", "next", after or END)
        return None

    def state(self, thread):
        case = self.case(thread)
        return {**case.values, "trail": [f"{w['author']}: {w['field']} = {_short(w['field'], w['to'])}"
                                         for w in case.history if w["field"] not in BOOKKEEPING]}

    def waiting(self, thread):
        return self.case(thread).values.get("waiting")

    def history(self, thread):
        return self.case(thread).history


def _short(field, value):
    """One readable line for a write, for the trail."""
    if isinstance(value, dict) and "action" in value:
        return f"{value['action']} -- {value['reason']}"
    if isinstance(value, dict) and "vendor_name" in value:
        return f"{value['invoice_number']} from {value['vendor_name']}"
    if field == "findings":
        return ", ".join(f["kind"] for f in value) or "matched"
    if field == "outbox":
        return f"wrote to {value[-1]['to']}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    text = " ".join(str(value).split())
    return text if len(text) <= 120 else text[:117] + "..."
