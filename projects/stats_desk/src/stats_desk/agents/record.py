"""One question's record: everything about it in one object, the single source that every
agent's context is built from and the thing that is saved (decided 2026-11-13).

The standard shape (LangGraph's state + checkpointer, Google ADK's session, the OpenAI Agents
SDK's sessions): one record per question, saved as is; each model call's context is a VIEW of it,
built by code. Before this, findings, results, transcripts and usage lived in separate objects
and the saved brief was assembled at the end from some of them -- which is how it came to keep
too little for a follow-up.

Not one chat shared by all agents: each agent keeps its own messages inside the record
(`transcripts`), because an analyst paying for the lead's whole chat on every call would waste
tokens. The rules for what each agent sees (lead.py, analyst.py) read from here.

    id, question, created, finished, status    running | done | no brief | failed | stopped
                                               (page closed) | paused (key's day spent)
    release                                    which release the figures come from
    conversation, parent, thread               the first question's id, the question this
                                               follows up, the questions before it
    earlier, earlier_brief, earlier_release    the previous question's findings (as e1, e2...),
                                               its brief, its release: what a follow-up starts from
    tasks, findings, results                   every task, every finding (or why it stopped),
                                               every analysis result by id
    transcripts                                each agent's own messages
    brief                                      answer, findings, ruled_out, cannot_tell, figures_from
    stopped, error, usage, free, public
"""

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime

from .agent import Stopped
from .analyst import Finding


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Record:
    id: str
    question: str
    release: str = ""
    created: str = field(default_factory=now)
    finished: str | None = None
    status: str = "running"
    conversation: str = ""
    parent: str | None = None
    thread: list = field(default_factory=list)
    earlier: dict = field(default_factory=dict)         # "e1" -> Finding, from the previous question
    earlier_brief: dict = field(default_factory=dict)   # its question and brief
    earlier_release: str | None = None
    tasks: dict = field(default_factory=dict)           # "f1" -> the lead's task
    findings: dict = field(default_factory=dict)        # "f1" -> Finding | Stopped | None (running)
    results: dict = field(default_factory=dict)         # result id -> its text
    transcripts: dict = field(default_factory=dict)     # agent -> its messages
    brief: dict = field(default_factory=dict)
    stopped: str | None = None
    error: str | None = None
    usage: dict = field(default_factory=dict)           # this run's
    spent: int = 0                                      # tokens over every run of it (resumes too)
    free: bool = False                                  # run on the desk's shared keys
    key_switches: int = 0                               # free keys run dry and swapped mid-question
    resumes: int = 0                                    # times carried on after a pause or a closed page
    public: bool = False

    def __post_init__(self):
        self.conversation = self.conversation or self.id

    # ------------------------------------------------------------------ follow-ups

    @classmethod
    def follow_up(cls, record_id, question, parent: "Record", release) -> "Record":
        """A follow-up's record: the previous question's findings renumbered e1, e2... (this
        question's own are f1, f2... again), and its results loaded back in so they can be cited
        -- unless the release changed since, when they can't be mixed with today's figures."""
        same = parent.release == release
        earlier, results = {}, {}
        for f in parent.findings.values():
            if isinstance(f, Finding):
                eid = f"e{len(earlier) + 1}"
                earlier[eid] = replace(f, id=eid)
                if same:
                    results |= {i: parent.results[i] for i in f.results + f.checked_with if i in parent.results}
        renamed = {f.id: eid for eid, f in zip(earlier, (f for f in parent.findings.values()
                                                         if isinstance(f, Finding)))}
        return cls(record_id, question, release, conversation=parent.conversation, parent=parent.id,
                   thread=[*parent.thread, parent.question], earlier=earlier,
                   earlier_brief={"question": parent.question, **parent.brief,
                                  "findings": [renamed[i] for i in parent.brief.get("findings", [])
                                               if i in renamed]},
                   earlier_release=parent.release, results=results)

    @property
    def earlier_citable(self) -> bool:
        return bool(self.earlier) and self.earlier_release == self.release

    # ------------------------------------------------------------------ saving

    def to_doc(self) -> dict:
        """As stored: a few fields to find it by, and the whole record as one JSON text (one
        field, so the database indexes a handful of fields, not every message)."""
        data = asdict(self) | {
            "findings": {k: _finding_doc(f) for k, f in self.findings.items()},
            "earlier": {k: _finding_doc(f) for k, f in self.earlier.items()}}
        return {"id": self.id, "question": self.question, "status": self.status, "created": self.created,
                "conversation": self.conversation, "public": self.public,
                "answer": self.brief.get("answer", ""), "record": json.dumps(data, ensure_ascii=False, default=str)}

    @classmethod
    def from_doc(cls, doc: dict) -> "Record":
        data = json.loads(doc["record"])
        data["findings"] = {k: _finding_from(v) for k, v in data["findings"].items()}
        data["earlier"] = {k: _finding_from(v) for k, v in data["earlier"].items()}
        data["public"] = doc.get("public", data.get("public", False))    # set apart from the record
        return cls(**data)

    def view(self) -> dict:
        """The brief as a reader gets it: no transcripts, no unused results."""
        cited = self.brief.get("findings", [])
        every = {**self.earlier, **self.findings}
        used = [every[i] for i in cited if isinstance(every.get(i), Finding)]
        evidence = {i: self.results[i] for f in used for i in f.results + f.checked_with if i in self.results}
        return {"id": self.id, "question": self.question, "status": self.status, "created": self.created,
                "finished": self.finished, "release": self.release, "conversation": self.conversation,
                "parent": self.parent, "thread": self.thread, "free": self.free,
                **{k: v for k, v in self.brief.items() if k != "findings"},
                "findings": [_finding_doc(f) for f in used], "evidence": evidence,
                "stopped": self.stopped, "error": self.error, "resumable": self.resumable,
                "usage": {"tokens": self.spent or self.usage.get("total", 0),
                          "model_calls": self.usage.get("calls", 0), "seconds": self.usage.get("seconds", 0)}}

    RESUMES = 3

    @property
    def resumable(self) -> bool:
        """Stopped partway (a closed page, a key's day spent): it can carry on from what it has."""
        return self.status in ("stopped", "paused") and self.resumes < self.RESUMES


def _finding_doc(f):
    if isinstance(f, Finding):
        return {"finding": asdict(f)}
    if isinstance(f, Stopped):
        return {"stopped": f.model_dump()}
    return {"running": True}


def _finding_from(d):
    if "finding" in d:
        return Finding(**d["finding"])
    if "stopped" in d:
        return Stopped(**d["stopped"])
    return None
