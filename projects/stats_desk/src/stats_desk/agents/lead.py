"""The lead, and one investigation from question to brief.

The lead directs; it never runs an analysis (chapter 6's supervisor with agents as tools). It
reads the profiles that bear on the question, forms hypotheses, and gives them to analysts with
`ask_analysts`, which runs one analyst per task at the same time. Each finding then gets code's
checks (checks.py: reruns in the other period and on the other measure, warnings), and the lead
reads the finding, judges it, and decides: more tasks, or the brief. The brief's numbers are
checked like everyone else's.

What the lead is shown, call by call:
  first      the question and the list of profiles, each with the table it comes from
  then       the profiles it asked for
  once findings arrive   the profiles cut to two lines each (they were needed to form the
             hypotheses, not after: on live run 3 six profiles rode along in every call, 26k
             tokens of a 49k question); each finding as the analyst's claim, its figures written
             short by code, the reruns, the warnings
  at a follow-up         also the earlier question, from its record (follow_up_context): its
             brief, and its findings as e1, e2... with their figures, citable like its own

Everything lives in the question's record (record.py): tasks, findings, results, each agent's
messages, the brief. It is what is saved, after each finding and at the end.

Limits, in code: at most MAX_ANALYSTS tasks, and a token budget for the whole question. The
lead's runtime counts every token spent under it, its analysts' included (the meter follows the
task), so one budget covers the question; `ask_analysts` refuses tasks the budget can no longer
pay for, saying how many it can, and leaves enough for the brief.
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

from .. import config
from ..analysis.analyses import NOTE
from ..analysis.data import MEASURES, TOTAL
from ..core import events, llm, meter, trace
from ..core.tools import Registry, Tool, ToolError
from ..datasets import DATASET
from . import plan
from .agent import FORM, Agent, Stopped, answer
from .analyst import BUDGET as ANALYST_BUDGET
from .analyst import INDICATORS, Finding, check_numbers, run_analyst, tables_for, ways_to_say
from .checks import check, outliers_in
from .record import Record, now

# a brief saying something isn't published (matched on plain()), and the server's refusals that say so
# only words that say MoSPI doesn't publish something (eval: "the figures do not provide the
# reasons why", an honest cannot_tell, was read as such a claim, and the brief refused)
UNPUBLISHED = re.compile(r"\b(?:not|never) (?:been )?published\b|\bunpublished\b"
                         r"|\b(?:isn't|aren't|wasn't|weren't|is not|are not|was not|were not) published\b"
                         r"|\b(?:doesn't|does not|don't|do not|didn't|did not) publish\b|\bno published\b")
# not "published only for <another age group or measure>": that says it IS published, for that
# (eval: taken as evidence, it let "MoSPI doesn't publish unemployment by education" through)
PUBLISHED_REFUSAL = re.compile(rf"not published|nothing published|has no (?:field|period|{DATASET.across_plural})|no \w+ '")
DEFINITION_WORDS = ("by definition", "built in", "computed from", "mechanical", "by construction", "follows from")
SKIPPED = re.compile(r"not published(?:, left out)?: ((?:\d{4}-\d{2}(?:, )?)+)")       # series, trend; average
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december")


def month_name(period: str) -> str:
    """"2026-03" -> "march 2026", as plain() reads a brief."""
    return f"{MONTHS[int(period[5:7]) - 1]} {period[:4]}"


# words that say a figure is shaky, for a brief naming a state set apart as an outlier
OUTLIER_WORDS = ("outlier", "small sample", "unreliable", "caution", "noisy", "small number")   # matched on plain()
REWRITES = 2       # small repair calls per hand-in before the problem goes back to the lead

REWRITE = ("You correct a brief so that it passes a check. Change only what the check asks, adding only the "
           "words the fix needs: no explanations or definitions of your own. Keep every number, finding id "
           f"and claim as it is, and add no new number. Reply by calling `{FORM}` with the corrected brief, "
           "all fields.")


def plain(text):
    """Text as the checks read it: "usual‑status" (a non-breaking hyphen) and "PS + SS" as "usual
    status" and "ps+ss", the forms gpt-oss writes the measure in."""
    text = (text or "").lower().replace("‑", " ").replace("‐", " ").replace("-", " ").replace(" ", " ")
    # every kind of space as one (eval: "Andaman &" + a narrow no-break space +
    # "Nicobar" slipped past the outlier check); \s takes the narrow and no-break spaces too
    text = re.sub(r"\s+", " ", text).replace("’", "'")              # and a curly apostrophe as '
    return re.sub(r"\s*\+\s*", "+", text)


class Rewrite(ToolError):
    """A problem with a brief that its own words can fix, without the findings: fixed by a small
    repair call before it would ever go back to the lead."""

MAX_ANALYSTS = 4
FIRST_ROUND = 3            # tasks in the first ask_analysts call: a live run spent all 4 at
                           # once, three failed, and nothing was left to give them out again
QUESTION_BUDGET = 60_000
TASK_RESERVE = ANALYST_BUDGET   # an analyst's budget; code's checks spend no tokens
BRIEF_RESERVE = 6_000

# the dataset's own words in the instructions (datasets/plfs.toml)
_D, _EX = DATASET, DATASET.examples
ACROSS, ACROSS_PLURAL = DATASET.across, DATASET.across_plural

PERSONA = f"""You are the lead analyst at a desk that answers questions about {_D.subject}: {_D.publisher}'s {_D.name}, with {_D.holds}.

You direct; you never run analyses yourself. Every reply is a tool call: profile, ask_analysts, or {FORM} to hand in. Never reply in prose.
1. Read the profiles that bear on the question; the list below says which table each comes from. Ask for all of them in one profile call. A profile says what code found striking in its table's latest release.
2. From them, pick what bears on the question and turn it into 2-3 hypotheses: possible answers the figures can test. A profile only points: every number in your brief must come from an analyst's finding. Never hand in the brief before analysts have reported, even when a profile seems to answer the question.
3. Give the hypotheses to analysts with ask_analysts, all in one call, one task each. Say what to find out, not how: the analyst chooses the analysis, and doesn't see the profiles, so name every group exactly as the profiles write it. Ask only for combinations the list of profiles says are published together; when the question's groups are not, ask for the nearest ones that are, each on its own, and say in the brief that the combination itself isn't published: the nearest figures stand in for it, they don't answer it. Never piece a combination together from two published ones.
   Analysts can:
   - look up figures; rank the groups of a field; take the gap between groups and whether it widened or narrowed since a year (or a month) earlier
   - over time: the change since the previous year or month; the months described (peaks, dips, streaks); an average over a stretch of months; whether it is rising, falling or has no clear trend
   - across {ACROSS_PLURAL}: whether two figures go together; and find which split shows the biggest differences
   - do each of these for any groups a task names: every group of a field (e.g. {_EX['every_group']}), a few of one field (e.g. {_EX['few_groups']}), or groups that differ in more than one way (e.g. {_EX['differ_groups']}), always with the total
   They cannot: average across groups, {_D.cannot}, test statistical significance, or give a combination {_D.publisher} doesn't publish.
4. Each finding comes back with the analyst's claim, its figures, the same figures rerun by code in the other period and on the other measure, and warnings from code. Judge it yourself: does it hold there too; are the groups compared alike in all but one way; could an overall figure hide the opposite pattern within its groups; does it rest on a small {ACROSS} set apart as an outlier; is it true by definition; is it a link you would expect anyway, because the {ACROSS_PLURAL} share something that moves both? A link you would expect anyway is not a finding: don't present it as one. Then go on with your plan: give the later tasks as the findings now make them look, drop the ones they make unnecessary, and add one only if something that matters for the answer is still open. A task that came back empty can be given again. Stop when more tasks are unlikely to change the answer; if the figures can't settle the question, say so. At most {MAX_ANALYSTS} tasks in all.
5. Hand in the brief with {FORM}:
- answer: the direct answer in 2-5 sentences. Quote numbers exactly as the findings show them, each with its unit ({_D.unit} or points), group, year and measure, and keep levels apart from changes. If what was asked is not published, say so, and give the nearest figures that are.
- findings: ids of the findings it rests on (f1, f2, ...)
- ruled_out: what was checked and did not hold, and why
- cannot_tell: what these figures cannot say. They show where and for whom, not why.
If the question asks whether one thing makes or causes another, don't answer yes or no: say what the figures show, and that they can't show cause.
Correlations across {ACROSS_PLURAL} describe {ACROSS_PLURAL} ({_D.describes}), not causes. Never work a number out yourself."""


class BriefForm(BaseModel):
    answer: Annotated[str, Field(description="the direct answer, 2-5 sentences, numbers exactly as the findings show them; "
                                             "for the reader, so no finding ids (they go in findings)")]
    findings: Annotated[list[str], Field(description="ids of the findings it rests on, e.g. f1")]
    # required: left optional, the lead left it out of 6 of 8 briefs in the span tests
    cannot_tell: Annotated[str, Field(description="what these figures cannot say; they show where and for whom, not why")]
    ruled_out: str = ""


@dataclass
class Brief:
    question: str
    answer: str = ""
    findings: list = field(default_factory=list)       # the Findings it rests on
    ruled_out: str = ""
    cannot_tell: str = ""
    all_findings: dict = field(default_factory=dict)   # every task: Finding or Stopped
    evidence: dict = field(default_factory=dict)       # result id -> the result's text
    stopped: str | None = None                         # why there is no brief, if there isn't
    usage: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)          # the question's record (core/trace.py)
    figures_from: list = field(default_factory=list)   # written by code: which tables, periods, measures

    def text(self):
        if self.stopped:
            return f"No brief: {self.stopped}"
        lines = [f"Q: {self.question}", "", self.answer, ""]
        if self.figures_from:
            lines += [f"Figures from: {'; '.join(self.figures_from)}", ""]
        for f in self.findings:
            lines += [f"[{f.id}] {f.claim}", f"     evidence: {', '.join(f.results + f.checked_with)}"]
            if f.flags:
                lines.append(f"     warnings: {'; '.join(f.flags)}")
        if self.ruled_out:
            lines += ["", f"Ruled out: {self.ruled_out}"]
        if self.cannot_tell:
            lines += ["", f"What these figures can't tell: {self.cannot_tell}"]
        return "\n".join(lines)


def settings_of(text) -> list[tuple[str, str]]:
    """(table and period, measure) of a result, from its own description line(s): "unemployment
    rate (UR), annual, 2025 | usual status (PS+SS) | ..." (a relate result has one per side)."""
    out = []
    for line in text.split("\n")[1:3]:
        parts = line.removeprefix("x: ").removeprefix("y: ").split(" | ")
        if len(parts) > 1 and parts[1] in MEASURES.values():
            setting = (parts[0].removesuffix(", averages over periods"), parts[1])
            if setting not in out:
                out.append(setting)
    return out


def figures_from(results):
    """Which table, period and measure each result's figures come from: written into the brief by
    code. Annual figures come on two measures that differ a lot (unemployment 2025: 3.1% usual
    status, 5.3% weekly status); in the span tests half the briefs named none, and a
    model asked to add it also added a wrong definition of its own."""
    out = []
    for text in results:
        for where, measure in settings_of(text):
            if f"{where}, {measure}" not in out:
                out.append(f"{where}, {measure}")
    return out


def not_comparable(findings, results) -> str:
    """A sentence for the answer when the findings it rests on use different measures, or mix
    annual and monthly figures: written by code, no model call. The first live
    question set a fall in the annual figure (usual status, 2024 to 2025) against a rise in the
    monthly one (weekly status, Apr 2025 to Sep 2026) with "However", as if they disagreed; the
    "Figures from" line named both measures, but apart from the answer. The figures are named by
    what they are, never by finding id: ids are for the findings list, not the reader."""
    settings = list(dict.fromkeys(s for f in findings for i in f.results if i in results
                                  for s in settings_of(results[i])))
    measures = {m for _, m in settings}
    kinds = {"monthly" if ", monthly" in w else "annual" for w, _ in settings}
    if len({f.id for f in findings}) < 2 or (len(measures) < 2 and len(kinds) < 2):
        return ""
    named = []
    for where, measure in settings:                  # "unemployment rate (UR), monthly, 2025-04 to 2026-09"
        kind = "monthly" if ", monthly" in where else "annual"
        period = where.rsplit(", ", 1)[-1]
        named.append(f"the {kind} figures ({period}, {measure})")
    why = ("different measures and periods" if len(measures) > 1 and len(kinds) > 1
           else "different measures" if len(measures) > 1 else "annual and monthly figures")
    return (f"Note: {', '.join(named[:-1])} and {named[-1]} are not directly comparable: {why}.")


# a finding cited by id in prose -- "(f1)", "(f1, f2)", "(f1 and e2)", "[f3]" -- with its leading space
FINDING_IDS = re.compile(r"\s*[\(\[]\s*(?:see\s+)?[fe]\d+(?:\s*(?:,|and|&)\s*[fe]\d+)*\s*[\)\]]", re.I)


def without_ids(text: str) -> str:
    """Prose for the reader, with the finding ids the lead cites by left out: they are for the
    findings list behind the brief."""
    return FINDING_IDS.sub("", text or "")


def profile_list(tables_text):
    """The profiles, each with the table it comes from and what that table holds, from the
    `tables` result: what the lead needs to choose which to read, and no more."""
    tables = {}
    for line in tables_text.split("\n"):
        m = re.match(r"(\w+_(?:annual|monthly)): (.*)$", line)
        if m:
            tables[m[1]] = m[2]
    names = next(line for line in tables_text.split("\n") if line.startswith("profiles: "))
    out = ["Profiles you can read:"]
    for name in names.removeprefix("profiles: ").split(", "):
        table = "_".join(name.split("_")[:2]).lower()
        out.append(f"- {name}: from table {table} -- {tables.get(table, '')}")
    out += [line for line in tables_text.split("\n") if line.startswith("what the groups mean: ")]
    return "\n".join(out)


SETTINGS = DATASET.settings        # measure, age: the dimensions with no total


def catalogue(tables_text):
    """From the tables list: each table's fields, and every field's values."""
    fields, values = {}, {}
    for line in tables_text.split("\n"):
        table = re.match(r"(\w+_(?:annual|monthly)): .*?; fields ([^;]+)", line)
        if table:
            fields[table[1]] = [f.strip() for f in table[2].split(",")]
        listed = re.match(r"(\w+) values: (.*)$", line)
        if listed:
            values[listed[1]] = [v.strip() for v in listed[2].split(", ")]
    return fields, values


def _words(text):
    """A task as it is matched: lower case, one kind of space and of hyphen."""
    text = re.sub(r"[‐‑‒–]", "-", (text or "").lower())
    return re.sub(r"\s+", " ", text)


def named_in(task, values):
    """What a task names: the settings ([(field, value)] of age and measure) and one group of each
    other field ({field: value}), matched on the values as the tables list writes them ("7.graduate"
    also as "graduate"). Totals are no group."""
    said = _words(task)
    settings = [(f, v) for f in SETTINGS for v in values.get(f, [])
                if any(w in said for w in ways_to_say(f, v))]
    groups = {}
    for f, vs in values.items():
        if f in SETTINGS:
            continue
        for v in vs:
            name = re.sub(r"^\d+\.", "", v).lower()
            if v != TOTAL.get(f) and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", said):
                groups.setdefault(f, v)
    return settings, groups


async def unpublished_settings(analyses, tables_text, task):
    """A measure or age group a task names that its groups aren't published on: the server's own
    note, else None. Code's calls, no tokens: each group of the task, on the settings it names, in
    every table of the task's indicator that has the group's field; it counts only if no table
    publishes it (a table without the note does). Span test: told which measure each split is on,
    and to name none unless asked, a lead still wrote "all ages" and "CWS" for graduates, both
    unpublished; an analyst then dropped the measure and its claim never said so."""
    fields, values = catalogue(tables_text)
    settings, groups = named_in(task, values)
    if not settings:
        return None
    measure_field, band_field = DATASET.measure, DATASET.band
    measures = [v for f, v in settings if f == measure_field] or [None]
    ages = [v for f, v in settings if f == band_field] or [None]
    wanted = {i for i, pattern in INDICATORS.items() if re.search(pattern, task, re.I)} or set(INDICATORS)
    tables = [t for t in fields if t.split("_")[0] in wanted]
    # annual or monthly, when the task's own words say which (a year alone: annual)
    months = r"\bmonth|\b\d{4}-\d{2}\b|\b(?:" + "|".join(m[:3] for m in MONTHS) + r")[a-z]*\b"
    kind = ("monthly" if re.search(months, task, re.I) else
            "annual" if re.search(r"\bannual|\byears?\b|\b20\d\d\b", task, re.I) else None)
    tables = [t for t in tables if kind is None or t.endswith(kind)]
    for field, group in [*groups.items(), (None, None)]:
        for measure in measures:
            for age in ages:
                where = {k: v for k, v in ((measure_field, measure), (band_field, age), (field, group)) if k and v}
                notes, published = [], False
                for table in tables:
                    if any(k not in fields[table] for k in where if k != measure_field):   # every table has one
                        continue
                    try:
                        text = await analyses.call("value", table=table, where=where)
                    except ToolError:
                        continue                   # refused for something else: not this check's to say
                    rid = text.partition("\n")[0].removeprefix("result ").strip()
                    analyses.results.pop(rid, None)    # a probe, never a result to cite
                    analyses.calls.pop(rid, None)
                    found = NOTE.findall(text)
                    if not found:
                        published = True
                        break
                    notes.append((table, found))
                if notes and not published:
                    table, found = notes[0]
                    what = f"{group} ({field})" if field else "the figures"
                    asked = " and ".join(f"{k} {v}" for k, v in where.items() if k in SETTINGS)
                    nearest = ", ".join(f"{f} {used}" for f, used, _ in found)
                    return f"{what} on {asked} isn't published together in {table} (the nearest that is: {nearest})"
    return None


FOLLOW_UP_CAP = 1_500      # tokens for everything a follow-up is shown of the earlier question


def follow_up_context(record: Record) -> str:
    """What the lead is shown of the earlier question, written by code from the record:
    the questions before it one line each, the last brief, and its findings as e1, e2... with
    their figures written short, citable when the release hasn't changed. Kept under
    FOLLOW_UP_CAP tokens: findings the brief didn't use lose their figures first, then
    themselves, then the oldest questions go, then the used findings' figures."""
    if not record.parent:
        return ""
    brief, used = record.earlier_brief, set(record.earlier_brief.get("findings", []))

    def render(unused_figures=True, unused=True, questions=None, used_figures=True):
        lines = ["This is a follow-up. Earlier in this conversation:"]
        thread = record.thread[:-1] if questions is None else record.thread[:-1][-questions:] if questions else []
        if thread:
            lines.append("asked before: " + " | ".join(thread))
        lines += [f"last question: {brief.get('question', '')}", f"its brief: {brief.get('answer', '')}"]
        if brief.get("figures_from"):
            lines.append(f"figures from: {'; '.join(brief['figures_from'])}")
        if brief.get("ruled_out"):
            lines.append(f"ruled out: {brief['ruled_out']}")
        if brief.get("cannot_tell"):
            lines.append(f"what those figures can't tell: {brief['cannot_tell']}")
        if record.earlier_citable:
            lines.append("Its findings: cite them as e1, e2... in your brief, like your own; ask analysts "
                         "only for what they don't cover.")
        elif record.earlier:
            lines.append(f"{DATASET.publisher} re-released the figures since ({record.earlier_release}; now {record.release}): "
                         "the earlier findings can't be cited or mixed with today's. Ask analysts for any "
                         "figure you need.")
        for eid, f in record.earlier.items():
            if eid not in used and not unused:
                continue
            head = f"{eid}{'' if eid in used else ' (not used in that brief)'} -- task: {f.task}"
            lines += [head, f"claim: {f.claim}"]
            lines += [f"WARNING from code: {w}" for w in f.flags]
            if (used_figures if eid in used else unused_figures) and record.earlier_citable:
                lines += [f"  [{i}] " + figures.replace("\n", "\n    ") for i, figures in f.figures.items()]
        return "\n".join(lines)

    for option in ({}, {"unused_figures": False}, {"unused_figures": False, "unused": False},
                   {"unused_figures": False, "unused": False, "questions": 2},
                   {"unused_figures": False, "unused": False, "questions": 0, "used_figures": False}):
        text = render(**option)
        if llm.count_tokens(text) <= FOLLOW_UP_CAP:
            return text
    return text


class Investigation:
    def __init__(self, analyses, question: str | None = None, record: Record | None = None, save=None):
        """`record`: the question's record (the API makes it; a script gets a new one).
        `save`: an async function given the record, called as each finding comes in."""
        self.record = record or Record(uuid.uuid4().hex, question)
        self.analyses, self.question, self.save = analyses, self.record.question, save
        # one record of results: what the agents get back is what is kept (earlier ones included)
        self.record.results.update(analyses.results)
        analyses.results = self.record.results
        self.usage = None
        self.tables = ""
        self.probed_unpublished = ""      # the server's refusal of the question's groups taken together
        self.unfinished: dict = {}        # a resumed question's tasks that never finished

    @property
    def findings(self) -> dict:
        return self.record.findings

    @findings.setter
    def findings(self, value):
        self.record.findings = value

    @property
    def transcripts(self) -> dict:
        return self.record.transcripts

    async def _save(self):
        if self.save is not None:
            try:
                await self.save(self.record)
            except Exception as error:          # noqa: BLE001 -- a failed save mustn't stop the question
                trace.note("save failed", error=type(error).__name__)

    def _citable(self) -> dict:
        """The findings a brief may rest on: this question's, and the earlier question's when
        their figures are from the same release."""
        done = {k: f for k, f in self.findings.items() if isinstance(f, Finding)}
        return ({**self.record.earlier, **done} if self.record.earlier_citable else done)

    async def ask_analysts(self, approach: str, tasks: list, later: list = ()) -> str:
        """Hand in your plan, and give its tasks that need no other task's answer to analysts,
        one task each. They run at the same time; each comes back as a finding with code's
        checks. Later tasks wait for those answers: give them in another call, as the findings
        then make them look."""
        tasks = [str(t) for t in tasks]
        room = MAX_ANALYSTS - len(self.findings)
        if room <= 0:
            raise ToolError(f"All {MAX_ANALYSTS} analyst tasks are used. Hand in the brief now.")
        affordable = max(0, (QUESTION_BUDGET - BRIEF_RESERVE - self.usage.total) // TASK_RESERVE)
        if affordable == 0:
            raise ToolError("This question's token budget can't pay for another task. "
                            "Hand in the brief now from the findings you have.")
        first = not self.findings                  # keep room to follow up on what the first round finds
        cap = min(room, FIRST_ROUND) if first else room
        most = min(cap, affordable)
        if not tasks or len(tasks) > most:
            why = (f"the token budget left pays for {affordable}" if affordable < cap
                   else f"in your first call, keeping room to follow up on what they find; {MAX_ANALYSTS} in all"
                   if first else f"{MAX_ANALYSTS} in all per question")
            raise ToolError(f"Send at most {most} task{'s' if most > 1 else ''} now ({why}).")
        await self.check_tasks(tasks)
        # Parallel by code, not by the model's choice: on live runs 1 and 2 the lead was told to
        # ask for independent tasks in one turn and made one call per turn both times.
        # numbered after every task ever given, resumed runs' unfinished ones included
        last = max((int(k[1:]) for k in {**self.record.tasks, **self.findings}), default=0)
        fids = [f"f{last + i + 1}" for i in range(len(tasks))]
        self.findings.update({fid: None for fid in fids})
        self.record.tasks.update(zip(fids, tasks))
        later = [str(x) for x in later]
        self.record.plan.append({"approach": approach, "tasks": dict(zip(fids, tasks)), "later": later})
        events.emit("plan", approach=approach, tasks=dict(zip(fids, tasks)), later=later)
        found = await asyncio.gather(*(self._task(fid, task) for fid, task in zip(fids, tasks)))
        return "\n\n".join(found) + plan.later_text(later)

    async def check_tasks(self, tasks: list):
        """Refuse the tasks, none given out, when one names a measure or age group its groups aren't
        published on (`unpublished_settings`)."""
        problems = []
        for n, task in enumerate(tasks, 1):
            found = await unpublished_settings(self.analyses, self.tables, task)
            if found:
                problems.append(f"task {n}: {found}")
        if problems:
            raise ToolError("No task was given out: " + "; ".join(problems) + ". Give the tasks again without "
                            f"the {DATASET.measure} or {DATASET.band} group the question doesn't ask about: "
                            "analysts use the published one.")

    async def _task(self, fid, task):
        """One analyst on one task, then code's checks on its finding."""
        events.emit("task", finding=fid, task=task)
        found = await run_analyst(self.analyses, fid, task, tables_for(self.tables, task),
                                  transcript=self.transcripts.setdefault(f"analyst {fid}", []))
        if isinstance(found, Stopped):
            self.findings[fid] = found
            await self._save()
            events.emit("finding", finding=fid, stopped=found.reason)
            return self._unfinished(fid, found)
        await check(self.analyses, found)
        self.findings[fid] = found
        await self._save()
        events.emit("finding", finding=fid, claim=found.claim, warnings=len(found.flags), flags=found.flags,
                    figures={self._label(i): t for i, t in found.figures.items()},
                    elsewhere={f"rerun by code: {self._label(i)}": t for i, t in found.elsewhere.items()})
        return found.text()

    def _unfinished(self, fid, stopped):
        """A task that came back with nothing, as the lead reads it. Not "the figures don't
        exist": a live run had three analysts stop on a wrong group name and their
        budgets, and the lead then wrote that unemployment by education isn't published -- its
        own profile had shown it at 11.2% for graduates. What the server refused is passed on:
        that, and only that, says what is published."""
        refused = []
        for m in self.transcripts.get(f"analyst {fid}", []):
            if m.get("role") == "tool" and m.get("kind") not in (None, "ok"):
                line = (m.get("content") or "").strip().split("\n")[0][:240]
                if line and line not in refused:
                    refused.append(line)
        lines = [f"{fid}: the analyst could not finish ({stopped.reason}); nothing from it can be cited. "
                 "That says nothing about whether the figures are published."]
        if refused:
            lines.append("What the analysis server refused it (these do say what is published):")
            lines += [f"  - {r}" for r in refused[-3:]]
        lines.append("If the answer still needs this, give the task again, naming groups exactly as "
                     "the profiles and the refusals above write them.")
        return "\n".join(lines)

    def _label(self, result_id):
        """A result as a reader knows it: "gap on ur_annual by gender", not its id."""
        analysis, args = self.analyses.calls.get(result_id, (None, None))
        if analysis == "recheck" and isinstance(args, dict):
            analysis, args = args.get("analysis"), args.get("args")
        return events.label(analysis, args) if analysis else result_id

    async def _build_brief(self, form: BriefForm):
        """The brief's checks. A problem its own words can fix (`Rewrite`) is fixed by a small
        repair call that gets only the brief and what to fix (`_rewrite`); anything else goes back
        to the lead, with its whole context, as a refusal. Span tests: a brief refused
        for a missing measure was retried with the lead's full ~2,500-token context."""
        said = plain(" ".join((form.answer, form.ruled_out, form.cannot_tell)))
        if UNPUBLISHED.search(said) and not self._refused_as_unpublished():
            await self._probe_combination()
        for _ in range(REWRITES):
            try:
                return self._check_brief(form)
            except Rewrite as problem:
                fixed = await self._rewrite(form, problem)
                if fixed is None:
                    raise
                form = fixed
        return self._check_brief(form)

    async def _rewrite(self, form: BriefForm, problem):
        """One small call: the brief, what the check said, and the facts that fix it. Nothing else."""
        from ..core.runtime import parse_call
        from .agent import form_tool
        messages = [{"role": "system", "content": REWRITE},
                    {"role": "user", "content": f"The brief:\n{form.model_dump_json(indent=1)}\n\nThe check said: {problem}"}]
        schema = form_tool(BriefForm, []).schema
        with trace.span("lead: rewrite brief", input=str(problem)):
            try:
                reply = await llm.complete(config.LEAD_MODEL, messages, tools=[schema], max_tokens=1200,
                                           reasoning_effort="low", tool_choice="required")
                args = (json.loads(reply.tool_calls[0].arguments) if reply.tool_calls
                        else (parse_call(reply.content) or (None, None))[1])
            except llm.BadRequestError as e:          # a hand-in under the wrong name, as in Think
                args = (parse_call(llm.failed_generation(e)) or (None, None))[1]
            except ValueError:
                args = None
        try:
            return BriefForm(**args) if isinstance(args, dict) else None
        except ValidationError:
            return None

    def _check_brief(self, form: BriefForm):
        if not form.answer.strip():
            raise ToolError("The answer is empty. Say what the figures show; if what was asked is not "
                            "published, say so and give the nearest figures that are.")
        done = self._citable()
        # span tests: on a trend question the lead twice wrote its brief straight from
        # the monthly profile, before giving any task; say what to do, not only what's wrong
        if not done and any(ch.isdigit() for ch in form.answer) and not self.findings:
            raise ToolError("You have no findings yet, and the numbers in a brief must come from findings. "
                            "Give the analysts their tasks first, with ask_analysts.")
        for fid in form.findings:
            if fid not in done:
                raise ToolError(f"There is no finding {fid}. Findings: {', '.join(done) or 'none'}.")
            # the span tests: shown this warning, the lead answered "Yes, they move
            # together very closely" -- a link true by definition is not a finding to rest on
            built = next((w for w in done[fid].flags if "TRUE BY DEFINITION" in w), None)
            if built:
                raise Rewrite(f"{fid} is a correlation true by definition ({built}): it is not a finding. "
                              f"Take {fid} out of findings, and say in the answer that the link holds by definition.")
        if not form.cannot_tell.strip():
            raise Rewrite("cannot_tell is empty: say what these figures cannot tell. They show where and for "
                          "whom, not why.")
        # live run: the lead wrote that unemployment by gender in cities isn't
        # published -- it is -- having never asked; only the analysis server's refusal knows
        said_unpublished = UNPUBLISHED.search(plain(" ".join((form.answer, form.ruled_out, form.cannot_tell))))
        if said_unpublished and not self._refused_as_unpublished():
            # a wording fix by the small repair call, not a refusal to the lead: in the eval the
            # lead sent the same brief back until the run was stopped with none
            raise Rewrite(
                f"The brief says something isn't published (\"{said_unpublished.group(0)}\"), but nothing in this "
                "question shows that it isn't. Take that claim out, and say only what the findings show.")
        # only the findings' results and code's reruns of them: on live run 3 profiles were allowed
        # too, and three invented numbers passed because they happened to be in some profile
        cited = [i for f in done.values() for i in f.results + f.checked_with]
        check_numbers(self.analyses, cited, form.answer, form.ruled_out, form.cannot_tell)
        said = plain(" ".join((form.answer, form.ruled_out, form.cannot_tell)))
        # (which measure the figures are on is written into the brief by code: `figures_from`)
        # a state set apart as an outlier, named without saying so (span tests: "Andaman & Nicobar
        # recorded the highest... 16.4%", the warning passed over)
        for name in dict.fromkeys(n for i in cited for n in outliers_in(self.analyses.results.get(i, ""))):
            if plain(name) in plain(form.answer) and not any(w in said for w in OUTLIER_WORDS):
                raise Rewrite(f"The answer names {name}, which code set apart as a statistical outlier. "
                              "Say that its figure may rest on a small sample.")
        # a link true by definition, shown by any finding, is said, whichever findings the brief
        # cites (eval: barred from citing the flagged correlation, the lead cited top-
        # and bottom-3 state lists showing the same link, and answered "Yes")
        built = next((w for f in done.values() for w in f.flags if "TRUE BY DEFINITION" in w), None)
        if built and not any(w in said for w in DEFINITION_WORDS):
            why = built.split("TRUE BY DEFINITION (", 1)[-1].rsplit("): it is not", 1)[0]
            raise Rewrite(f"Code found this link true by definition: {why}. "
                          f"Say in the answer that it holds by definition, so it is not a finding about {ACROSS_PLURAL}.")
        # months MoSPI didn't publish, inside the figures the brief rests on (eval:
        # every result said "not published: 2026-03"; the brief didn't)
        rests_on = [i for fid in form.findings if fid in done for i in done[fid].results]
        skipped = sorted({m for i in rests_on for m in SKIPPED.findall(self.analyses.results.get(i, ""))
                          for m in re.findall(r"\d{4}-\d{2}", m)})
        if skipped and not any(m in said or month_name(m) in said for m in skipped):
            raise Rewrite(f"The monthly figures skip {', '.join(month_name(m).title() for m in skipped)}: "
                          f"{DATASET.publisher} didn't publish it. Say so in the answer.")
        return form

    def _refused_as_unpublished(self) -> bool:
        """Whether this question has the server's word that something isn't published: a request
        refused as not published, or a result saying so (a series lists the months MoSPI didn't
        publish, "not published: 2026-03"; the eval's Q2 expects the brief to say it)."""
        refused = any(m.get("role") == "tool" and m.get("kind") not in (None, "ok")
                      and PUBLISHED_REFUSAL.search(m.get("content") or "")
                      for messages in self.transcripts.values() for m in messages)
        return (refused or bool(self.probed_unpublished)
                or any("not published" in text for text in self.analyses.results.values()))

    async def _probe_combination(self):
        """Ask the server, once, whether the groups this question's analyses asked for, taken
        together, are published: e.g. religion=Islam from one finding and state=Kerala from
        another (the eval's Q4: the lead knew from the tables list that the combination isn't
        published, planned the nearest figures, and was refused for saying so without proof).
        Code's own call: no tokens. Its refusal, if it is one, is the evidence."""
        merged: dict = {}
        for name, args in list(self.analyses.calls.values()):
            if name == "recheck" or not isinstance(args, dict) or not args.get("table"):
                continue
            if not isinstance(args.get("where") or {}, dict):
                continue                # several groups, each set on its own: not one combination
            where = merged.setdefault(args["table"], {})
            for k, v in (args.get("where") or {}).items():
                if k not in ("period", DATASET.measure) and v is not None and not isinstance(v, list):
                    where.setdefault(k, v)
        for table, where in merged.items():
            if len(where) < 2:
                continue
            try:
                await self.analyses.call("value", table=table, where=where)
            except ToolError as refused:
                if PUBLISHED_REFUSAL.search(str(refused)):
                    self.probed_unpublished = str(refused)
                    return

    def lead(self) -> Agent:
        """The lead as this investigation runs it (the span tests run its steps one at a time)."""
        ask = Tool(self.ask_analysts, name="ask_analysts", args=plan.Plan)
        return Agent("lead", PERSONA, tools=Registry([self.analyses.tool("profile"), ask]),
                     model=config.LEAD_MODEL, max_tokens=1500, max_tool_calls=12,
                     budget=QUESTION_BUDGET, answer_on_stop=False, form_build=self._build_brief,
                     tool_timeout=900, shorten=("profile",), keep_recent=1, choose=self._next_step,
                     when_stuck=("Act on it with a tool call. With no findings yet, give tasks with ask_analysts, "
                                 "including any check of what is published; once findings answer the question, "
                                 f"hand in the brief with {FORM}."))

    def _next_step(self, conversation) -> str | None:
        """The lead's fixed first steps: read profiles, then hand out tasks; after that, its own
        choice. Not in a follow-up whose earlier findings can be cited (they may answer it)."""
        if self.findings or self.record.earlier_citable:
            return None
        read = any(m.get("role") == "tool" and m.get("name") == "profile" and m.get("kind") in (None, "ok")
                   for m in conversation.messages)
        return "ask_analysts" if read else "profile"

    def context(self):
        """What the lead is shown before the question: the profiles by name and table, for a
        follow-up what code writes of the earlier question, and for a resumed question what it
        had found before it stopped -- all from the record."""
        parts = [profile_list(self.tables), follow_up_context(self.record), self.resume_context()]
        return "\n\n".join(p for p in parts if p)

    def resume_context(self) -> str:
        if not self.unfinished and not self.record.resumes:
            return ""
        lines = ["You were answering this question and had to stop partway. Carry on from what you "
                 "have: don't ask again for what these findings already show."]
        done = [f for f in self.findings.values() if isinstance(f, Finding)]
        lines += [f.text() for f in done] or ["No finding had come back yet."]
        if self.unfinished:
            lines.append("Tasks that didn't finish (give them again if you still need them):")
            lines += [f"- {task}" for task in self.unfinished.values()]
        return "\n\n".join(lines)

    def _prepare_resume(self):
        """A question carried on from its record: keep its findings, take out the tasks that
        never finished (they are listed for the lead instead), and keep the lead's earlier
        messages under their own name -- the new run starts a fresh conversation from the record
        rather than replaying a half-done one."""
        record = self.record
        self.unfinished = {fid: record.tasks.get(fid, "") for fid, f in record.findings.items()
                           if not isinstance(f, Finding)}
        for fid in self.unfinished:
            del record.findings[fid]
        if "lead" in record.transcripts:
            runs = sum(1 for k in record.transcripts if k.startswith("lead"))
            record.transcripts[f"lead, run {runs}"] = record.transcripts.pop("lead")

    async def run(self) -> Brief:
        events.emit("code_call", tool="tables")
        self.tables = await self.analyses.call("tables")
        brief = Brief(self.question)
        record = self.record
        self.unfinished = {}
        if record.resumes:
            self._prepare_resume()
        try:
            with meter.metered() as usage:
                self.usage = usage
                result = await answer(self.lead(), self.question, BriefForm, context=self.context(),
                                      transcript=self.transcripts.setdefault("lead", []))
        finally:                    # also when stopped halfway: the record keeps what was spent
            record.usage = usage.summary()
            record.spent += usage.total
        brief.usage = record.usage
        brief.all_findings = dict(self.findings)
        record.finished = now()
        if isinstance(result, Stopped):
            brief.stopped = record.stopped = result.reason
            record.status = "no brief"
            return brief
        cited = self._citable()
        brief.answer, brief.ruled_out, brief.cannot_tell = (without_ids(result.answer), without_ids(result.ruled_out),
                                                            without_ids(result.cannot_tell))
        brief.findings = [cited[f] for f in result.findings]
        used = {i for f in brief.findings for i in f.results + f.checked_with}
        brief.evidence = {i: self.analyses.results[i] for i in used}
        rests_on = brief.findings or [f for f in self.findings.values() if isinstance(f, Finding)]
        brief.figures_from = figures_from(self.analyses.results[i] for f in rests_on for i in f.results)
        note = not_comparable(brief.findings, self.analyses.results)
        if note:
            brief.answer = f"{brief.answer.rstrip()} {note}"
        record.brief = {"answer": brief.answer, "findings": list(result.findings), "ruled_out": brief.ruled_out,
                        "cannot_tell": brief.cannot_tell, "figures_from": brief.figures_from}
        record.status = "done"
        return brief


async def investigate(analyses, question=None, via="script", session=None, tags=(), record=None,
                      save=None) -> tuple[Brief, Investigation]:
    """One question, traced (core/trace.py): `brief.trace` is its trace. `record`, `save`: see
    Investigation (the API passes both; a script needs neither)."""
    investigation = Investigation(analyses, question, record=record, save=save)
    question = investigation.question
    with trace.Trace(question, via=via, session=session or investigation.record.conversation, tags=tags) as recorded:
        try:
            brief = await investigation.run()
        except Exception as error:
            recorded.finish("error", error=error)
            raise
        recorded.finish("no brief" if brief.stopped else "brief", brief=None if brief.stopped else brief.text(),
                        stopped=brief.stopped)
    brief.trace = recorded.record
    return brief, investigation
