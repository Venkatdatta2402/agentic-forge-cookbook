"""The running coach again, with memory built from this repo's parts -- chapter 4's design, not Mem0's.

Mem0 keeps every memory the same way: a sentence, embedded, found again by search. Chapter 4 decides where a memory
goes by what will be asked of it later, and a coach asks three different things:

    the profile    "when is her race, what is her goal, which days does she run" -- facts with names, so chapter
                   4's `KeyValue`: one value per name. A new value REPLACES the old one, so a race that moves
                   leaves one race date behind, not two for a model to choose between
    the runs       "how have her long runs built up" -- events, filtered and ordered, so a table in the shape of
                   chapter 4's `Records`. Always appended: two long runs a week apart look like duplicates to a
                   similarity check and are two events (chapter 4's types-of-memory trap)
    notes          anything else worth knowing next time, which nobody could design a field for -- chapter 4's
                   `QdrantJournal` (its `Journal`, on one engine), found again by resemblance through `Recall`
                   and its floor

and it uses them the way chapter 4's last notebook settled on: RECALL BY INJECTION, WRITE BY TOOL.

    reading        `Recollect`, a loop component before Think, as chapter 4's `Retrieve` is: the whole profile,
                   the run log, and the notes that bear on the message, every check-in, whether asked or not
    writing        one `remember` tool, a chapter 3 `Tool`. The coach decides what is worth keeping; the code
                   decides where it goes -- the routing by type of memory that chapter 4 built
    the loop       chapter 2's `Runtime` with chapter 6's `Think` (an output ceiling that can be set) and
                   `verbatim`, so the tool's answer reaches the coach as it is
    one runner     a folder per runner. Two runners' memories are not filtered apart: they are never together
"""

import json
import re
import sqlite3
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, create_model

import runners as R
import writing  # noqa: E402 -- chapter 4's, on the path `runners` sets

ROOT = R.STORE / "scratch"
ANSWERS = R.STORE / "answers" / "scratch.json"

PROFILE = {"race": "the race they are training for, and its date",
           "goal": "their goal time",
           "days": "the days of the week they run",
           "time_of_day": "when in the day they run",
           "injuries": "anything that hurts, and how it stands now",
           "food": "their diet, and what they eat and drink on runs"}
Field_ = Literal["race", "goal", "days", "time_of_day", "injuries", "food"]
assert set(Field_.__args__) == set(PROFILE)

WRITE = """
Before you reply, call `remember` once with what in the message is worth keeping for later check-ins:
- profile: a field whose value has changed or is new. Write the whole current value -- it replaces the old one.
- runs: every run the runner reports, with its date worked out from today's date.
- notes: anything else worth knowing at a later check-in that no profile field holds.
If nothing in the message is new, reply without calling it."""


class Change(BaseModel):
    field: Field_ = Field(description="; ".join(f"{k}: {v}" for k, v in PROFILE.items()))
    value: str = Field(description="the whole current value, e.g. 'Riverside Half Marathon, Sunday 1 November'")


class Run(BaseModel):
    on: str = Field(description="the date of the run, YYYY-MM-DD")
    kind: Literal["long", "easy", "tempo", "intervals", "hills", "race"]
    km: float
    minutes: Optional[float] = Field(default=None, description="how long it took, in minutes, if said")
    felt: str = Field(default="", description="how it went, in a few words")


# Round 2. In round 1 the coach wrote memory as a side job of replying, and did it carelessly: a race without its
# date, a race that moved never written, changed days put in a note instead of the profile, runs dated wrong. Mem0
# gives writing a step of its own; so does round 2, with this repo's parts: after the reply, one chapter 1
# `extract()` is shown what is on record and asked what the message changes. The routing is the same `keep`.
ROUND2 = R.STORE / "scratch-round2"
ANSWERS2 = R.STORE / "answers" / "scratch.round2.json"

RECORD = """You keep a running coach's records about {name}. Today is {today}.

On record now:
{record}

{name}'s message today:
{message}

Return what to change in the record -- only what {name} says, nothing a coach recommends.
- profile: every field the message changes or adds. Write the field's whole new value, complete on its own: a race
  with its name and its date ("Riverside Half Marathon, Sunday 1 November"), every day they run, and so on. It
  replaces what is on record, so leave out a field the message does not change.
- runs: every run the message reports, each dated by working back from today, {today}.
- notes: anything else worth knowing at a later check-in that no profile field holds.
Leave a list empty when nothing applies."""


class Changes(BaseModel):
    profile: list[Change] = Field(default_factory=list)
    runs: list[Run] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# Round 3. Round 2 got every named fact right at the last check-in but still wrote three kinds of wrong fact, and a
# profile field shows a wrong value as a plain fact:
#   a field left out    "gels worked" went into a run's note; `food` kept "water only". The profile was an optional
#                       list, so a field nobody thought of was silently "unchanged"
#   a value made up     Tom's "five days a week" saved as "Monday, Tuesday, Wednesday, Thursday, Friday"
#   a miss hidden       a replaced value leaves nothing beside it, so a missed change cannot be seen
# The fixes: every field has a slot that must be answered; every change carries the runner's own words and is
# checked against the message in code (chapter 3's `verified()`: check the artifact, not the report); and the coach
# is shown each field's previous value with its date, while each check-in's message is kept, dated, in the journal.
ROUND3 = R.STORE / "scratch-round3"
ANSWERS3 = R.STORE / "answers" / "scratch.round3.json"

CHECKED = """You keep a running coach's records about {name}. Today is {today}.

On record now:
{record}

{name}'s message today:
{message}

Fill in the record form -- only what {name} says, nothing a coach recommends.
- profile: answer EVERY field. "unchanged" if the message does not change it; otherwise its whole new value,
  complete on its own (a race with its name and its date). With each new value, copy the words of the message it
  comes from into `quote`, exactly. The value must say nothing the quote does not: no day, date or number that is
  not in the quote.
- runs: every run the message reports, each dated by working back from today, {today}, with its words in `quote`.
- notes: anything else worth knowing at a later check-in that no profile field holds, with its words in `quote`."""


class Slot(BaseModel):
    value: str = Field(description='"unchanged", or the whole new value')
    quote: str = Field(default="", description="the words of the message the new value comes from, copied exactly")


ProfileForm = create_model(
    "ProfileForm", **{field: (Slot, Field(description=about)) for field, about in PROFILE.items()})


class QuotedRun(Run):
    quote: str = Field(description="the words of the message that report this run, copied exactly")


class Note(BaseModel):
    text: str
    quote: str = Field(description="the words of the message this comes from, copied exactly")


class Form(BaseModel):
    profile: ProfileForm
    runs: list[QuotedRun] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)


_NAMES = set("monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
             "august september october november december".split())


def _words(text):
    """Words and numbers, apart: "10k" is 10 and k, "28km" is 28 and km; a time like 1:45 stays whole."""
    return re.findall(r"[a-z]+|\d+(?::\d+)?", text.lower())


def unquoted(value, quote, message):
    """Why a change is refused, or None. The quote must be the message's own words, in order; and every day, month
    and number in the value must be in the quote -- the parts a model fills in when it fills a form."""
    said, words = _words(quote), _words(message)
    if not said:
        return "no quote"
    if not any(words[i:i + len(said)] == said for i in range(len(words) - len(said) + 1)):
        return "the quote is not in the message"
    claimed = [w for w in _words(value) if w in _NAMES or re.fullmatch(r"\d+(:\d+)?", w)]
    missing = [w for w in dict.fromkeys(claimed) if w not in said]
    return f"not in the quote: {', '.join(missing)}" if missing else None


# Round 4. Round 3's fixes worked on their targets, but three of its own choices were mistakes:
#   one line, two facts   `food` held both diet and race fuel, and a new value replaces the whole line -- writing
#                         "gels" erased "vegetarian"
#   a check on copying    a quote had to match the message word for word; one word changed ("the" for "my") and a
#                         correct fact was thrown away
#   whole messages        kept in the journal beside the notes; a message covers many topics, so it matches almost
#                         any question, and three of them filled Recall's three places
# And one thing no round had fixed: the model working out dates ("last Sunday") and times ("1:38") itself.
# Round 4, still one form per check-in:
#   - diet and race fuel are separate fields, and the form can add a field when none fits -- chapter 4's `Profile`,
#     which also keeps each field's previous value and dates
#   - no quotes: every day, date, number and name in a value must be in the message or already on record; what fails
#     goes back to the model once, with the reason, instead of being dropped -- chapter 4's `writing.fill`
#   - runs give the runner's words for the day and the time; code works out the date (chapter 4's `day_of`) and the
#     minutes (`minutes_of`, running times being this project's own)
#   - notes are one fact each, and messages are not kept
ROUND4 = R.STORE / "scratch-round4"
ANSWERS4 = R.STORE / "answers" / "scratch.round4.json"

FIELDS4 = {"race": "the race they are training for, and its date",
           "goal": "their goal time",
           "days": "the days of the week they run",
           "time_of_day": "when in the day they run",
           "injuries": "anything that hurts, and how it stands now",
           "diet": "what they eat in general, e.g. vegetarian",
           "fuel": "what they eat and drink before and during runs and races"}

CARD = """You keep a running coach's records about {name}. Today is {today}.

On record now:
{record}

{name}'s message today:
{message}

Fill in the record form -- only what {name} says, nothing a coach recommends.
- profile: answer EVERY field: "unchanged" if the message does not change it, otherwise its whole new value,
  complete on its own (a race with its name and its date). A day, date, number or name in a value must be one the
  message or the record says.
- new_fields: only for something worth keeping that fits none of the fields: a short field name, what it holds,
  and its value.
- runs: every run the message reports. For `when` and `time`, copy {name}'s own words ("last Sunday", "1:38"):
  the dates and minutes are worked out from them.
- notes: anything else worth knowing at a later check-in -- ONE fact per note, in a short sentence of its own.
  Not questions {name} asks."""


class SaidRun(BaseModel):
    when: str = Field(description="the runner's own words for the day, e.g. 'last Sunday', 'on Saturday'")
    kind: Literal["long", "easy", "tempo", "intervals", "hills", "race"]
    km: float
    time: str = Field(default="", description="the runner's own words for how long it took, e.g. '1:38', or ''")
    felt: str = Field(default="", description="how it went, in a few words")


def minutes_of(words):
    """Minutes from a running time as said: "1:38" is 1 h 38, "49:30" is 49 min 30 s (a first part over 5 is
    minutes), "86 min" is 86. None when nothing is said."""
    clock = re.search(r"(\d+):(\d{2})", words or "")
    if clock:
        first, second = int(clock.group(1)), int(clock.group(2))
        return first * 60 + second if first <= 5 else round(first + second / 60, 2)
    plain = re.search(r"(\d+(?:\.\d+)?)\s*min", words or "")
    return float(plain.group(1)) if plain else None


class RunLog:
    """A runner's runs: chapter 4's `Records`, with a runner's columns. Filtered, ordered and counted -- never
    searched by resemblance, and never merged: each row is one run."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS runs (
        id      INTEGER PRIMARY KEY,
        on_day  TEXT NOT NULL,          -- ISO 8601, so string order is time order
        kind    TEXT NOT NULL,
        km      REAL NOT NULL,
        minutes REAL,
        felt    TEXT NOT NULL,
        UNIQUE (on_day, kind, km)       -- the same run told twice is one run
    );
    """

    def __init__(self, path):
        from stores import _connect
        self.db = _connect(str(path))
        self.db.executescript(self.SCHEMA)

    def write(self, run):
        added = self.db.execute("INSERT OR IGNORE INTO runs (on_day, kind, km, minutes, felt) VALUES (?, ?, ?, ?, ?)",
                                (run.on, run.kind, run.km, run.minutes, run.felt))
        self.db.commit()
        return added.rowcount

    def recent(self, kind=None, since=None, limit=10):
        """Filter, order, limit -- oldest first, so a build-up reads as one."""
        where, args = [], []
        if kind:
            where.append("kind = ?"); args.append(kind)
        if since:
            where.append("on_day >= ?"); args.append(since)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = self.db.execute(f"SELECT * FROM runs {clause} ORDER BY on_day DESC LIMIT ?", [*args, limit]).fetchall()
        return list(reversed(rows))

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]


class Memory:
    """One runner's three stores, in one folder of their own."""

    def __init__(self, runner, root=ROOT, before=False):
        from stores import KeyValue, QdrantJournal
        self.runner, self.folder, self.before = runner, root / runner, before
        self.folder.mkdir(parents=True, exist_ok=True)
        self.profile = KeyValue(self.folder / "profile.json")
        self.runs = RunLog(self.folder / "runs.db")
        self.notes = QdrantJournal(self.folder / "journal")

    def keep(self, today, profile=(), runs=(), notes=()):
        """The routing: each kind of memory to its store, the way that kind is written."""
        said = []
        for change in profile:
            old = self.profile.read(change.field) or {}
            self.profile.write(change.field, {"value": change.value, "since": today,
                                              "was": old.get("value"), "was_since": old.get("since")})
            said.append(f"{change.field} = {change.value}")
        logged = sum(self.runs.write(run) for run in runs)
        if runs:
            said.append(f"{logged} run(s) logged" + (f", {len(runs) - logged} already there" if logged < len(runs) else ""))
        known = {row["text"] for row in self.notes.recent(limit=1000)}
        fresh = [n for n in dict.fromkeys(notes) if n not in known]
        if fresh:
            self.notes.write([(today, self.runner, "coach", n) for n in fresh], now=today)
            said.append(f"{len(fresh)} note(s) kept")
        return "Kept: " + "; ".join(said) if said else "Nothing new kept."

    def lines(self):
        """The profile and the latest runs, one line each. With `before`, each field also shows what it said before
        and since when -- so a change that was missed still has its old value in view, dated, not a bare fact."""
        def field(k, v):
            earlier = (f"; before: {v['was']}" + (f" (since {v['was_since']})" if v.get("was_since") else "")
                       if self.before and v.get("was") else "")
            return f"- {k}: {v['value']} (since {v['since']}{earlier})"
        profile = [field(k, v) for k, v in self.profile.data.items()]
        return profile, self.run_lines()

    def run_lines(self):
        return [f"- {r['on_day']} {r['kind']} {r['km']:g} km" + (f" in {r['minutes']:g} min" if r["minutes"] else "") +
                (f" -- {r['felt']}" if r["felt"] else "") for r in self.runs.recent(limit=12)]

    def render(self, today, message, floor=0.6, share=None, cap=None, search="meaning"):
        """What the coach is shown before it replies: all of the profile, the runs, and the notes that bear on this.

        search="meaning"    what rounds 1-7 ran: the top 3 above the floor, by meaning alone (or with `share` and
                            `cap`, that share of all notes, at most `cap`)
        search="questions"  chapter 4's hybrid `Recall`, one search per question in the message, merged; and every
                            habit note shown as well, searched or not -- a runner has few, and they bear on every
                            check-in"""
        from recall import Recall, questions
        name = R.RUNNERS[self.runner]
        profile, runs = self.lines()
        habits = []
        if not len(self.notes):
            notes = []
        elif search == "questions":
            habit_rows = self.notes.recent(source="habit", limit=1000)
            habits = [f"- {row['at']} {row['text']}" for row in habit_rows]
            picked = Recall(self.notes, today=today, floor=floor).many(questions(message), k=3, share=share, cap=cap)
            shown = {row["id"] for row in habit_rows}
            notes = [f"- {at} {text}" for _, eid, at, text in picked if eid not in shown]
        else:
            recall = Recall(self.notes, today=today, floor=floor, hybrid=False)
            notes = [f"- {at} {text}" for _, _, at, text in recall(message, k=3, share=share, cap=cap)]
        if not (profile or runs or notes or habits):
            return None
        parts = [f"What you know about {name} from earlier check-ins."]
        parts += ["Profile:", *profile] if profile else ["Profile: nothing yet"]
        parts += ["Runs, oldest first:", *runs] if runs else ["Runs: none logged yet"]
        if habits:
            parts += ["Habits:", *habits]
        if notes:
            parts += ["Notes that bear on this message:", *notes]
        return "\n".join(parts)


class Recollect:
    """Chapter 4's `Retrieve` for a coach: a component before Think, returning the memory as a message, or None."""

    def __init__(self, memory, today):
        self.memory, self.today = memory, today
        self.shown = None

    def run(self, conversation):
        from memory_tools import last_question
        self.shown = self.memory.render(self.today, last_question(conversation) or "")
        return {"role": "tool", "content": self.shown} if self.shown else None


def remember_tool(memory, today, log=None):
    """`remember`, closed over one runner's stores and today's date -- a parameter the model has no way to know."""
    from tools import tool

    @tool
    def remember(profile: Annotated[Optional[list[Change]], "profile fields that changed or are new"] = None,
                 runs: Annotated[Optional[list[Run]], "every run reported in the message"] = None,
                 notes: Annotated[Optional[list[str]], "anything else worth knowing at a later check-in"] = None
                 ) -> str:
        """Keep what is worth knowing at later check-ins with this runner."""
        to_model = lambda items, cls: [cls.model_validate(i) if isinstance(i, dict) else i for i in items or []]
        said = memory.keep(today, to_model(profile, Change), to_model(runs, Run), notes or [])
        if log is not None:
            log.append({"profile": profile or [], "runs": runs or [], "notes": notes or [], "said": said})
        return said

    return remember


def write_step(memory, session):
    """Round 2's writing: after the reply, one `extract()` shown the record and the message, its changes routed by
    `keep`. Today's weekday goes in, because "last Sunday" is arithmetic the date alone leaves to the model."""
    from datetime import date

    from llm import extract
    name = R.RUNNERS[session["runner"]]
    today = f"{date.fromisoformat(session['date']):%A} {session['date']}"
    profile, runs = memory.lines()
    record = "\n".join(["Profile:", *(profile or ["- nothing yet"]), "Runs:", *(runs or ["- none logged"])])
    changes = extract(RECORD.format(name=name, today=today, record=record, message=session["message"]), Changes)
    said = memory.keep(session["date"], changes.profile, changes.runs, changes.notes)
    return {**changes.model_dump(), "said": said}


def checked_step(memory, session):
    """Round 3's writing: a form with a slot for every field; each change checked against the message's own words
    before it is kept, the refused ones recorded with why; and the message itself kept in the journal, dated."""
    from datetime import date

    from llm import extract
    name, message = R.RUNNERS[session["runner"]], session["message"]
    today = f"{date.fromisoformat(session['date']):%A} {session['date']}"
    profile, runs = memory.lines()
    record = "\n".join(["Profile:", *(profile or ["- nothing yet"]), "Runs:", *(runs or ["- none logged"])])
    form = extract(CHECKED.format(name=name, today=today, record=record, message=message), Form)
    refused, changes, kept_runs, kept_notes = [], [], [], []
    for field in PROFILE:
        slot = getattr(form.profile, field)
        if slot.value.strip().strip('"').lower() in ("unchanged", ""):
            continue
        why = unquoted(slot.value, slot.quote, message)
        if why:
            refused.append({"field": field, "value": slot.value, "quote": slot.quote, "why": why})
        else:
            changes.append(Change(field=field, value=slot.value))
    for run in form.runs:
        why = unquoted(f"{run.km:g}", run.quote, message)
        (refused.append({"run": run.model_dump(), "why": why}) if why else kept_runs.append(run))
    for note in form.notes:
        why = unquoted("", note.quote, message)
        (refused.append({"note": note.text, "quote": note.quote, "why": why}) if why else kept_notes.append(note.text))
    said = memory.keep(session["date"], changes, kept_runs, kept_notes)
    entry = f"{name} said: {message}"
    if entry not in {row["text"] for row in memory.notes.recent(limit=1000)}:
        memory.notes.write([(session["date"], memory.runner, "runner said", entry)], now=session["date"])
    return {"form": form.model_dump(), "refused": refused, "said": said}


class Profiled(Memory):
    """Round 4's stores: chapter 4's `Profile` in place of the plain `KeyValue` -- named facts that keep their
    previous value and dates, and a field list that can grow -- with the same run log and notes."""

    def __init__(self, runner, root=ROUND4):
        super().__init__(runner, root, before=True)
        from stores import Profile
        self.profile = Profile(self.folder / "profile.json", FIELDS4)

    def keep(self, today, profile=(), runs=(), notes=()):
        for field, value in profile:
            self.profile.write(field, value, today)
        rest = super().keep(today, (), runs, notes)
        said = [f"{f} = {v}" for f, v in profile] + ([rest[len("Kept: "):]] if rest.startswith("Kept: ") else [])
        return "Kept: " + "; ".join(said) if said else "Nothing new kept."

    def lines(self):
        return self.profile.lines(before=True), self.run_lines()


def card_step(memory, session):
    """Round 4's writing, on chapter 4's `writing`: one form with a slot for every field the runner has and room to
    add one; every day, date, number and name checked against the message and the record; anything that fails sent
    back once with the reason; the runner's words for days and times turned into dates and minutes by code."""
    from datetime import date

    import writing as W
    name, message, today = R.RUNNERS[session["runner"]], session["message"], session["date"]
    fields = memory.profile.fields()
    profile, runs = memory.lines()
    record = "\n".join(["Profile:", *(profile or ["- nothing yet"]), "Runs:", *(runs or ["- none logged"])])
    source = message + "\n" + record
    form = W.profile_form(fields, runs=(list[SaidRun], Field(default_factory=list)),
                          notes=(list[str], Field(default_factory=list)))

    def problems(card):
        found = W.profile_problems(card, fields, source)
        for i, run in enumerate(card.runs):
            if W.day_of(run.when, today) is None:
                found.append({"key": ("run", i), "what": f"run '{run.when}'", "why": "no day can be worked out from these words"})
            elif f"{run.km:g}" not in W.words(message):
                found.append({"key": ("run", i), "what": f"run '{run.when}' {run.km:g} km", "why": "the distance is not in the message"})
        for i, note in enumerate(card.notes):
            if bad := W.unsupported(note, source):
                found.append({"key": ("note", i), "what": f"note '{note}'", "why": "not said: " + ", ".join(bad)})
        return found

    prompt = CARD.format(name=name, today=f"{date.fromisoformat(today):%A} {today}", record=record, message=message)
    card, still, sent_back = W.fill(prompt, form, problems)
    refused = {p["key"] for p in still}
    changes = [(f, v) for f, v in W.changed(card, fields) if ("profile", f) not in refused]
    for new in card.new_fields:
        key = re.sub(r"\W+", "_", new.name.lower()).strip("_")
        if key and key not in fields and ("profile", key) not in refused:
            memory.profile.add_field(key, new.about)
    kept_runs = [Run(on=W.day_of(r.when, today), kind=r.kind, km=r.km, minutes=minutes_of(r.time), felt=r.felt)
                 for i, r in enumerate(card.runs) if ("run", i) not in refused]
    kept_notes = [n for i, n in enumerate(card.notes) if ("note", i) not in refused]
    said = memory.keep(today, changes, kept_runs, kept_notes)
    return {"card": card.model_dump(), "sent back": sent_back,
            "refused": [{k: v for k, v in p.items() if k != "key"} for p in still], "said": said}


# Round 5. Measured against `answer_key` (see `baseline.py`), round 4 saved 45 of 50 facts right, and what it
# missed was a pattern: habits and small qualifiers ("as I always seem to at race starts", "hills included").
# And three things it did wrong: unchanged values re-written (now ignored by chapter 4's `Profile`), a note that
# restated a profile field going stale, and a field the run log could work out (`longest_run`). Round 5:
#   - the form says what belongs where: profile, run log, note -- and a note never restates a field
#   - habits are a kind of their own, saved as notes
#   - every statement in the message is listed with where it went, and code checks each is accounted for
#   - each note names the fields it depends on; when one of them changes, chapter 4's `QdrantJournal.retire`
#     takes the note out of every search
#   - a new field the run log can work out is refused
# All but the running words are chapter 4's: `writing.rules`, `Note`, `Statement`, `uncovered`,
# `derived_problems`, `note_problems`; `Profile` ignoring unchanged values; `QdrantJournal.retire`.
ROUND5 = R.STORE / "scratch-round5"
ANSWERS5 = R.STORE / "answers" / "scratch.round5.json"
# Round 6. Round 5 copied earlier runs back into the form at six check-ins, and each copy, refused, cost a second
# call. Two changes: chapter 4's `writing.rules` now asks for the runs THIS message reports (round 5 ran with "every
# one that happened"); and a run whose distance this message never gives is dropped -- `fill` removes a problem
# marked "drop" without sending the form back -- so a copy costs nothing even if the model still makes one.
ROUND6 = R.STORE / "scratch-round6"
ANSWERS6 = R.STORE / "answers" / "scratch.round6.json"
# Round 7: round 6 unchanged here; chapter 4's check now compares what days, dates and numbers MEAN. Round 6 refused
# Tom's race written "2026-11-08" against his "8 November", and "5 days a week" against "five" -- so his race was
# never stored, and at his last check-in the coach made one up.
# How many of a runner's notes the coach is shown: 30% of them, at most 5 (chapter 4's `Recall(share=, cap=)`).
NOTES_SHARE, NOTES_CAP = 0.3, 5
ROUND7 = R.STORE / "scratch-round7"
ANSWERS7 = R.STORE / "answers" / "scratch.round7.json"

FIELDS5 = {"race": "the race they are training for, and its date",
           "goal": "their goal time for that race",
           "days": "the days of the week they run",
           "time_of_day": "when in the day they run",
           "injuries": "anything that hurts, and how it stands now",
           "diet": "what they eat in general, e.g. vegetarian",
           "fuel": "what they eat and drink before and during runs and races"}

# The placement rules, notes, statements and the derived-field check are chapter 4's (`writing.py`): they are about
# writing any memory down. What is the coach's own: the running fields, its examples, runs in the runner's words.
CARD5 = """You keep a running coach's records about {name}. Today is {today}.

On record now:
{record}

{name}'s message today:
{message}

Fill in the record form -- only what {name} says, nothing a coach recommends. """ + writing.rules(
    events="runs", example_states="their race, goal, days, diet",
    example_habit="always starts races too fast") + """
For each run, copy {name}'s own words for `when` and `time` ("last Sunday", "1:38"): dates and minutes are worked
out from them."""

RUNNING_DERIVED = ["pb", "mileage", "weekly", "pace"]      # chapter 4's DERIVED, plus what a run log also answers


class Profiled5(Profiled):
    """Round 5's stores: as round 4, with notes that depend on fields and are retired when those change."""

    def __init__(self, runner, root=ROUND5):
        super().__init__(runner, root)
        from stores import Profile
        self.profile = Profile(self.folder / "profile.json", FIELDS5)

    def keep5(self, today, changes, runs, notes):
        said, retired = [], []
        for field, value in changes:
            if self.profile.write(field, value, today):          # False when the value has not changed
                said.append(f"{field} = {value}")
                retired += self.notes.retire(field, today)
        logged = sum(self.runs.write(run) for run in runs)
        if runs:
            said.append(f"{logged} run(s) logged")
        known = {row["text"] for row in self.notes.recent(limit=1000, live=False)}
        fresh = [n for n in notes if n.text not in known]
        if fresh:
            self.notes.write([(today, self.runner, n.kind, n.text, n.depends_on) for n in fresh], now=today)
            said.append(f"{len(fresh)} note(s) kept")
        if retired:
            said.append(f"{len(retired)} note(s) retired")
        return "Kept: " + "; ".join(said) if said else "Nothing new kept."


def card5_step(memory, session):
    """Round 5's writing: round 4's form with the store rules, habits, notes that depend on fields, and every
    statement accounted for -- checked in code and sent back once if anything fails."""
    from datetime import date

    import writing as W
    name, message, today = R.RUNNERS[session["runner"]], session["message"], session["date"]
    fields = memory.profile.fields()
    profile, runs = memory.lines()
    record = "\n".join(["Profile:", *(profile or ["- nothing yet"]), "Runs:", *(runs or ["- none logged"])])
    source = message + "\n" + record
    form = W.profile_form(fields, runs=(list[SaidRun], Field(default_factory=list)),
                          notes=(list[W.Note], Field(default_factory=list)),
                          statements=(list[W.Statement], Field(default_factory=list)))

    def problems(card):
        found = W.profile_problems(card, fields, source) + W.derived_problems(card, RUNNING_DERIVED)
        for i, run in enumerate(card.runs):
            # a run whose distance today's message never gives was not reported today -- copied from the record --
            # so it is dropped, not sent back: asking again would cost a call to remove it
            if f"{run.km:g}" not in W.words(message):
                found.append({"key": ("run", i), "what": f"run '{run.when}' {run.km:g} km",
                              "why": "not in this message", "drop": True})
            elif W.day_of(run.when, today) is None:
                found.append({"key": ("run", i), "what": f"run '{run.when}'", "why": "no day can be worked out from these words"})
        return found + W.note_problems(card, fields, source) + W.uncovered(card, fields, card.runs, memory.profile.read)

    prompt = CARD5.format(name=name, today=f"{date.fromisoformat(today):%A} {today}", record=record, message=message)
    card, still, sent_back = W.fill(prompt, form, problems)
    refused = {p["key"] for p in still}
    for new in card.new_fields:
        if W.key(new.name) and W.key(new.name) not in fields and ("profile", W.key(new.name)) not in refused:
            memory.profile.add_field(W.key(new.name), new.about)
    changes = [(f, v) for f, v in W.changed(card, fields) if ("profile", f) not in refused]
    kept_runs = [Run(on=W.day_of(r.when, today), kind=r.kind, km=r.km, minutes=minutes_of(r.time), felt=r.felt)
                 for i, r in enumerate(card.runs) if ("run", i) not in refused]
    kept_notes = [n for i, n in enumerate(card.notes) if ("note", i) not in refused]
    said = memory.keep5(today, changes, kept_runs, kept_notes)
    return {"card": card.model_dump(), "sent back": sent_back,
            "refused": [{k: v for k, v in p.items() if k != "key"} for p in still], "said": said}


def coach(memory, today, log=None, model=R.COACH_MODEL, write="tool"):
    """The loop: Recollect once, then Think -> Act -> Observe until the coach replies. With write="step" or
    "checked" the coach has no tool and only replies; the writing is done afterwards."""
    from act import Act
    from observe import Observe
    from runtime import Runtime
    from supervisor import Think, verbatim
    recollect = Recollect(memory, today)
    if write in ("step", "checked", "card", "card5"):
        # Act stays, with nothing to call: Think's answer is a "respond" decision, and Act is what turns it into
        # the reply that ends the run
        return Runtime(loop=[recollect, Think(model=model, tools=[], allow_fork=False, explain=False,
                                              max_tokens=R.REPLY_TOKENS), Act()],
                       repeat_from=1, max_iterations=7), recollect
    remember = remember_tool(memory, today, log)
    runtime = Runtime(loop=[recollect,
                            Think(model=model, tools=[remember.schema], allow_fork=False, explain=False,
                                  max_tokens=R.REPLY_TOKENS),
                            Act(functions={"remember": remember.function}, schemas={"remember": remember.args}),
                            Observe(extractors={"remember": verbatim})],
                      repeat_from=1, max_iterations=13)       # four turns: more than one write and a reply need
    return runtime, recollect


def check_in(session, root=ROOT, write="tool"):
    """One check-in: what the coach was shown, what was kept, its reply, and what the calls cost."""
    from chat import Conversation
    from meter import Meter
    memory = (Profiled5(session["runner"], root) if write == "card5" else Profiled(session["runner"], root)
              if write == "card" else Memory(session["runner"], root, before=write == "checked"))
    writes = []
    runtime, recollect = coach(memory, session["date"], writes, write=write)
    persona = R.COACH.format(date=session["date"], name=R.RUNNERS[session["runner"]])
    conversation = Conversation(system=persona + (WRITE if write == "tool" else ""))
    conversation.messages.append({"role": "user", "content": session["message"]})
    with Meter() as meter:
        runtime.run(conversation)
        if write == "step":
            writes.append(write_step(memory, session))
        elif write == "checked":
            writes.append(checked_step(memory, session))
        elif write == "card":
            writes.append(card_step(memory, session))
        elif write == "card5":
            writes.append(card5_step(memory, session))
    last = conversation.messages[-1]
    return {"shown": recollect.shown, "kept": writes, "reply": last.get("content") or "",
            "stopped": conversation.stopped, "usage": meter.summary()}


def run(sessions=R.SESSIONS, root=ROOT, path=ANSWERS, log=print, write="tool"):
    """Every check-in in order, each saved as it finishes; a stopped run carries on from where it stopped.

    A check-in that fails part-way may already have written: the profile is replaced and runs and notes are not
    written twice, so doing it again leaves the stores as one clean pass would."""
    import openai
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for session in sessions:
        if "reply" in done.get(session["id"], {}):
            continue
        try:
            record = check_in(session, root, write)
        except openai.APIStatusError as error:
            log(f"  {session['id']}: stopped -- {type(error).__name__}: {str(error)[:200]}")
            return done, False
        done[session["id"]] = {"runner": session["runner"], "date": session["date"], **record}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"  {session['id']}: kept {len(record['kept'])} time(s), {record['usage']['calls']} calls, "
            f"{record['usage']['total']:,} tokens")
    return done, True


def holdings(runner, root=ROOT):
    """Everything kept for a runner: the profile (with what each field said before), the runs, the notes."""
    saved = root / runner / "profile.json"
    profiled = saved.exists() and '"facts"' in saved.read_text(encoding="utf-8")     # round 4's chapter 4 `Profile`
    memory = Profiled(runner, root) if profiled else Memory(runner, root)
    held = {"profile": memory.profile.facts if profiled else memory.profile.data,
            "runs": [dict(r) for r in memory.runs.recent(limit=1000)],
            "notes": [dict(r) for r in memory.notes.recent(limit=1000, live=False)]}    # retired ones too, marked
    if profiled:
        held["fields"] = memory.profile.fields()
    return held
