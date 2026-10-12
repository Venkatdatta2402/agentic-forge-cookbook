"""Writing memory down, checked before it is kept.

Everything else in this chapter assumes what is stored is right. `08_frameworks/06_mem0` measured what happens when it
is not: a coach whose stores did exactly what they were built for, and still told a runner the wrong race date,
because the wrong things had been written. A wrong memory outlives every other kind of failure -- it is read back
at every later turn, as fact. Chapter 3's `verified()` checks an artifact rather than the report of it; this is that
check, at the moment of writing.

    profile_form   a form with a required slot for every field, answered "unchanged" or with the new value, and
                   room to add a field. A slot that must be answered cannot be skipped by not thinking of it --
                   an optional list of changes can, silently
    unsupported    the days, dates, numbers and names in a value that the source never says: the parts a model
                   fills in when it fills a form ("five days a week" saved as "Monday to Friday")
    day_of         the date a runner's or user's words mean ("last Sunday", said on a Tuesday). Date arithmetic is
                   done here, not by the model -- measured, the model got it wrong even when told the weekday
    fill           extract the form, check it, and send it back ONCE with the reasons if anything fails, keeping
                   what passes. Refusing without a second chance throws good facts away. A problem marked "drop"
                   (something that should just not be kept) is removed without sending the form back
    rules          where each thing goes -- profile, events, notes -- as a form's instructions, with `Note` (a
                   habit is a kind of its own) and `Statement`, every statement listed with where it went:
                   `uncovered` checks each is really in the form, `derived_problems` refuses a field the records
                   could work out, `note_problems` checks notes as values are checked

What is not checked: words that only describe ("knee cleared" for "my knee is fine"). Rewording them is the job, so
an exact-words check refuses correct facts -- `06_mem0`'s third round did, on one changed word. Checking meaning
would take another model call, and a model judging support was unreliable in `07_langsmith`.
"""

import re
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, create_model

WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
MONTHS = "january february march april may june july august september october november december".split()


def words(text):
    """Words and numbers, apart: "10k" is 10 and k; a time like 1:45 stays whole."""
    return re.findall(r"[a-z]+|\d+(?::\d+)?", text.lower())


NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())} | {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60"}
_MONTH = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|" \
         r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATES = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"                                        # 2026-11-08
                    r"|\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH + r"\b(?:,?\s+(\d{4}))?"    # 8 November (2026)
                    r"|\b" + _MONTH + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?", re.IGNORECASE)


def _month(name):
    return next(i for i, m in enumerate(MONTHS, 1) if m.startswith(name.lower()[:3]))


def dates(text):
    """Every date in the text, in any of the ways it is written, as ((start, end), (month, day))."""
    found = []
    for m in _DATES.finditer(text):
        g = m.groups()
        if g[0]:
            when = (int(g[1]), int(g[2]))
        elif g[3]:
            when = (_month(g[4]), int(g[3]))
        else:
            when = (_month(g[6]), int(g[7]))
        found.append((m.span(), when))
    return found


def as_digits(text):
    """Number words as digits: "five days a week" reads as "5 days a week"."""
    return re.sub(r"\b(" + "|".join(NUMBERS) + r")\b", lambda m: NUMBERS[m.group(1).lower()], text,
                  flags=re.IGNORECASE)


def unsupported(value, source):
    """The days, dates, numbers and names in `value` that `source` never says, or [].

    Compared by what they mean, not how they are written: "5" is "five", and "2026-11-08", "8 November 2026" and
    "Nov 8" are one day (a year beside a date that matches is part of it). Measured in `06_mem0`: compared as
    written, "8 November" saved as "2026-11-08" was refused, and a runner's race was never stored.

    A name is a capitalised word that is not the first word of the value or of a clause in it -- "Riverside Half
    Marathon, Sunday 1 November" claims Half, Marathon, Sunday and the date. Pass the record as part of the source
    as well as the new message, so a value may repeat what is already known.
    """
    said_dates = {when for _, when in dates(source)}
    missing_dates, rest = [], value
    for (start, end), when in reversed(dates(value)):
        if when not in said_dates:
            missing_dates.append(value[start:end].lower())
        rest = rest[:start] + " " + rest[end:]
    # the source's number words count as digits, so a "5" in the value matches "five"; a number word in the
    # value is not a claim, as before -- only digits are
    value, source = rest, as_digits(source)
    said = set(words(source))
    singular = lambda w: w[:-1] if w.endswith("s") and not w.endswith("ss") else w      # "Sundays" is "Sunday"
    said |= {singular(w) for w in said}
    claimed = [w for w in words(value) if singular(w) in WEEKDAYS or w in MONTHS or re.fullmatch(r"\d+(:\d+)?", w)]
    for clause in re.split(r"[,.;:()\-–—]", value):
        found = re.findall(r"[A-Za-z]+", clause)
        claimed += [w.lower() for w in found[1:] if w[0].isupper()]
    return list(reversed(missing_dates)) + [w for w in dict.fromkeys(claimed) if w not in said and singular(w) not in said]


def day_of(text, today):
    """The date `text` means, counted from `today` (YYYY-MM-DD) -- or None when it does not say.

    A weekday is the most recent one before today: "Sunday's long run", said on a Tuesday, is two days ago, and
    "Tuesday", said on a Tuesday, is a week ago. A date without a year is this year's, or last year's if that would
    be in the future.
    """
    today, text = date.fromisoformat(today), text.lower()
    written = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if written:
        return written.group(1)
    named = re.search(r"(\d{1,2})(?:st|nd|rd|th)? (" + "|".join(MONTHS) + r")|(" + "|".join(MONTHS) + r") (\d{1,2})",
                      text)
    if named:
        day, month = (named.group(1), named.group(2)) if named.group(1) else (named.group(4), named.group(3))
        on = date(today.year, MONTHS.index(month) + 1, int(day))
        return (on if on <= today else on.replace(year=today.year - 1)).isoformat()
    if "yesterday" in text:
        return (today - timedelta(days=1)).isoformat()
    if "today" in text or "this morning" in text:
        return today.isoformat()
    ago = re.search(r"(\d+) days? ago", text)
    if ago:
        return (today - timedelta(days=int(ago.group(1)))).isoformat()
    for i, name in enumerate(WEEKDAYS):
        if name in text:
            return (today - timedelta(days=(today.weekday() - i) % 7 or 7)).isoformat()
    return None


class NewField(BaseModel):
    name: str = Field(description="a short field name")
    about: str = Field(description="what the field holds")
    value: str


def profile_form(fields, **more):
    """A form with a required slot for every field -- "unchanged", or the whole new value -- and `new_fields` for
    what fits none of them. `more` adds sections of the caller's own, as `create_model` takes them."""
    profile = create_model("Profile", **{f: (str, Field(description=f'{about}: "unchanged", or the whole new value'))
                                         for f, about in fields.items()})
    return create_model("Form", profile=(profile, ...), new_fields=(list[NewField], Field(default_factory=list)),
                        **more)


def changed(form, fields):
    """[(field, value)] for every slot not answered "unchanged", then every new field (its name made a key)."""
    out = [(f, getattr(form.profile, f).strip()) for f in fields
           if getattr(form.profile, f).strip().strip('"').lower() not in ("unchanged", "")]
    out += [(key(n.name), n.value.strip()) for n in form.new_fields]
    return out


def profile_problems(form, fields, source):
    """Every changed value with a day, date, number or name the source does not say."""
    return [{"key": ("profile", f), "what": f"{f} = {v}", "why": "not said: " + ", ".join(bad)}
            for f, v in changed(form, fields) if (bad := unsupported(v, source))]


# -- where each thing goes ----------------------------------------------------------------
#
# This chapter's principle -- a memory goes where what will be asked of it decides -- as instructions for a form.
# Measured in `06_mem0`: without them a model files a changed day in a note, a habit as a one-off event, and a
# longest run as a profile field the run log already answers.

DERIVED = ["longest", "total", "average", "count", "best", "latest", "fastest", "most"]


def rules(events, example_states, example_habit):
    """The placement rules for a form with a profile, an events log and notes. `events` names the log ("runs"), the
    examples are the domain's own.

    The events line says THIS message's events, not merely the ones that happened: asked for "every one that
    happened", a model shown the record copied every earlier run back into the form, at six of nine check-ins in
    `06_mem0`'s fifth round -- each refused by the checks, each costing a second call."""
    return f"""Each thing goes in ONE place:
- profile: the current state of something that stays true until it changes ({example_states}). Answer EVERY field:
  "unchanged", or its whole new value, complete on its own. A day, date, number or name in a value must be one the
  message or the record says.
- new_fields: a current state that fits none of the fields, with its name, what it holds and its value. Never
  something the {events} can work out (longest, totals, averages, best, latest).
- {events}: every one THIS message reports -- never the ones already on record.
- notes: something lasting that is neither a state nor one of the {events} -- ONE fact per note. A habit (something
  that keeps happening, e.g. "{example_habit}") is a note of kind "habit". A note never restates a profile field.
  In `depends_on`, list the profile fields the note would stop being true without.
- statements: every statement made about themselves, each in a few words, with where it went: "profile" (and its
  field), "events" (the {events}), "note", or "nothing" (a question, or not worth keeping)."""


class Note(BaseModel):
    text: str
    kind: Literal["habit", "fact"]
    depends_on: list[str] = Field(default_factory=list, description="profile fields this note depends on")


class Statement(BaseModel):
    said: str = Field(description="the statement, in a few words")
    went_to: Literal["profile", "events", "note", "nothing"]
    field: str = Field(default="", description="the profile field, when it went to the profile")


def key(name):
    """A field name as a key: "pre-run eating" is pre_run_eating."""
    return re.sub(r"\W+", "_", name.lower()).strip("_")


def derived_problems(form, words=()):
    """New fields an events log could work out -- by name, against DERIVED and the domain's own `words`."""
    pattern = re.compile("|".join(map(re.escape, [*DERIVED, *words])))
    return [{"key": ("profile", key(n.name)), "what": f"new field {key(n.name)}",
             "why": "the records can work this out; leave it to them"}
            for n in form.new_fields if pattern.search(key(n.name))]


def note_problems(form, fields, source):
    """Notes with a day, date, number or name nobody said, or that depend on a field that does not exist."""
    known = set(fields) | {key(n.name) for n in form.new_fields}
    found = []
    for i, note in enumerate(form.notes):
        if bad := unsupported(note.text, source):
            found.append({"key": ("note", i), "what": f"note '{note.text}'", "why": "not said: " + ", ".join(bad)})
        if unknown := [f for f in note.depends_on if f not in known]:
            found.append({"key": ("note", i), "what": f"note '{note.text}'", "why": f"no such field: {', '.join(unknown)}"})
    return found


def uncovered(form, fields, events, on_record):
    """Statements the form says went somewhere they did not: a profile field neither changed nor already holding
    something, an events log with nothing in it, notes with none. `on_record(field)` says whether a field already
    holds a value -- a statement repeating what is known can go to an unchanged field."""
    done = {f for f, _ in changed(form, fields)}
    return [{"key": ("statement", s.said), "what": f"'{s.said}' -> {s.went_to} {s.field}".strip(),
             "why": "it is not in the form"}
            for s in form.statements
            if (s.went_to == "profile" and key(s.field) not in done and not on_record(key(s.field)))
            or (s.went_to == "events" and not events) or (s.went_to == "note" and not form.notes)]


def fill(prompt, form, problems, extract=None):
    """Extract `form`; if `problems(filled)` finds any, send it back once with the reasons; return the form, what is
    still wrong, and what was sent back. Callers keep what is not in the returned problems."""
    if extract is None:
        from llm import extract
    filled = extract(prompt, form)
    found, sent_back = problems(filled), []
    # a problem marked "drop" is an item that should simply not be kept -- an event copied from the record, say --
    # and asking again costs a whole call to remove what code can remove. Only the others are worth a second try.
    fixable = [p for p in found if not p.get("drop")]
    if fixable:
        sent_back = [f"{p['what']} -- {p['why']}" for p in fixable]
        filled = extract(prompt + "\n\nYour last form had these problems. Fix them, writing only what the message or "
                         "the record says:\n" + "\n".join(f"- {s}" for s in sent_back), form)
        found = problems(filled)
    return filled, found, sent_back
