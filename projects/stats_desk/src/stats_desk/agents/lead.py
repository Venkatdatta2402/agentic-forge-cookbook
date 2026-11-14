"""The lead, and one investigation from question to brief.

The lead directs; it never runs an analysis (chapter 6's supervisor with agents as tools). It
reads the profiles that bear on the question, forms hypotheses, and gives them to analysts with
`ask_analysts`, which runs one analyst per task at the same time. Each finding then gets code's
checks (checks.py: reruns in the other period and on the other measure, warnings), and the lead
reads the finding, judges it, and decides: more tasks, or the brief. The brief's numbers are
checked like everyone else's.

What the lead is shown, call by call (decided 2026-11-11):
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
from ..analysis.data import MEASURES
from ..core import events, llm, meter, trace
from ..core.tools import Registry, Tool, ToolError
from .agent import FORM, Agent, Stopped, answer
from .analyst import Finding, check_numbers, run_analyst, tables_for
from .checks import check, outliers_in
from .record import Record, now

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
    return re.sub(r"\s*\+\s*", "+", text)


class Rewrite(ToolError):
    """A problem with a brief that its own words can fix, without the findings: fixed by a small
    repair call before it would ever go back to the lead."""

MAX_ANALYSTS = 4
QUESTION_BUDGET = 60_000
TASK_RESERVE = 12_000      # an analyst's budget; code's checks spend no tokens
BRIEF_RESERVE = 6_000

PERSONA = f"""You are the lead analyst at a desk that answers questions about India's official labour statistics: MoSPI's PLFS, with the unemployment rate (UR), labour force participation rate (LFPR) and worker population ratio (WPR); annual by state, gender, sector, age, education, religion and social group, and monthly by gender, sector and age.

You direct; you never run analyses yourself.
1. Read the profiles that bear on the question; the list below says which table each comes from. Ask for all of them in one profile call. A profile says what code found striking in its table's latest release.
2. From them, pick what bears on the question and turn it into 2-3 hypotheses: possible answers the figures can test.
3. Give the hypotheses to analysts with ask_analysts, all in one call, one task each. Say what to find out, not how: the analyst chooses the analysis, and doesn't see the profiles.
   Analysts can: look up a figure; compare every group of a field; take the gap between two groups, overall and within each group of another field; give the change since the previous year, month, or the same month a year earlier; follow a figure month by month (streaks, against its own history, 3-month averages); average a figure over a stretch of months; say whether a figure is rising or falling over the months; say whether two figures go together across states; find which split shows the biggest differences. They cannot: average across groups, give shares of people (only rates are published), adjust for prices, test statistical significance, or give a combination MoSPI doesn't publish.
4. Each finding comes back with the analyst's claim, its figures, the same figures rerun by code in the other period and on the other measure, and warnings from code. Judge it yourself: does it hold there too; are the groups compared alike in all but one way; does it rest on a small state set apart as an outlier; is it true by definition; is it a link you would expect anyway, because the states share something that moves both? A link you would expect anyway is not a finding: don't present it as one. If something that matters for the answer is still open, you may ask again. At most {MAX_ANALYSTS} tasks in all.
5. Hand in the brief with {FORM}:
- answer: the direct answer in 2-5 sentences. Quote numbers exactly as the findings show them, each with its unit (% or points), group, year and measure. If what was asked is not published, say so, and give the nearest figures that are.
- findings: ids of the findings it rests on (f1, f2, ...)
- ruled_out: what was checked and did not hold, and why
- cannot_tell: what these figures cannot say. They show where and for whom, not why.
If the question asks whether one thing makes or causes another, don't answer yes or no: say what the figures show, and that they can't show cause.
Correlations across states describe states (people's chances there), not causes. Never work a number out yourself."""


class BriefForm(BaseModel):
    answer: Annotated[str, Field(description="the direct answer, 2-5 sentences, numbers exactly as the findings show them; "
                                             "for the reader, so no finding ids (they go in findings)")]
    findings: Annotated[list[str], Field(description="ids of the findings it rests on, e.g. f1")]
    # required: left optional, the lead left it out of 6 of 8 briefs in the span tests (2026-11-12)
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
    status, 5.3% weekly status); in the span tests (2026-11-12) half the briefs named none, and a
    model asked to add it also added a wrong definition of its own."""
    out = []
    for text in results:
        for where, measure in settings_of(text):
            if f"{where}, {measure}" not in out:
                out.append(f"{where}, {measure}")
    return out


def not_comparable(findings, results) -> str:
    """A sentence for the answer when the findings it rests on use different measures, or mix
    annual and monthly figures: written by code, no model call (2026-11-14). The first live
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
    findings list behind the brief (user, 2026-11-14)."""
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
    return "\n".join(out)


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
            lines.append(f"MoSPI re-released the figures since ({record.earlier_release}; now {record.release}): "
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

    async def ask_analysts(self, tasks: Annotated[
            list[str], "independent tasks, one per analyst; each says what to find out"]) -> str:
        """Give tasks to analysts, one each. They run at the same time; each comes back as a
        finding with code's checks."""
        room = MAX_ANALYSTS - len(self.findings)
        if room <= 0:
            raise ToolError(f"All {MAX_ANALYSTS} analyst tasks are used. Hand in the brief now.")
        affordable = max(0, (QUESTION_BUDGET - BRIEF_RESERVE - self.usage.total) // TASK_RESERVE)
        if affordable == 0:
            raise ToolError("This question's token budget can't pay for another task. "
                            "Hand in the brief now from the findings you have.")
        most = min(room, affordable)
        if not tasks or len(tasks) > most:
            raise ToolError(f"Send at most {most} task{'s' if most > 1 else ''} now"
                            + (f" (the token budget left pays for {affordable})." if affordable < room else
                               f" ({MAX_ANALYSTS} in all per question)."))
        # Parallel by code, not by the model's choice: on live runs 1 and 2 the lead was told to
        # ask for independent tasks in one turn and made one call per turn both times.
        # numbered after every task ever given, resumed runs' unfinished ones included
        last = max((int(k[1:]) for k in {**self.record.tasks, **self.findings}), default=0)
        fids = [f"f{last + i + 1}" for i in range(len(tasks))]
        self.findings.update({fid: None for fid in fids})
        self.record.tasks.update(zip(fids, tasks))
        found = await asyncio.gather(*(self._task(fid, task) for fid, task in zip(fids, tasks)))
        return "\n\n".join(found)

    async def _task(self, fid, task):
        """One analyst on one task, then code's checks on its finding."""
        events.emit("task", finding=fid, task=task)
        found = await run_analyst(self.analyses, fid, task, tables_for(self.tables, task),
                                  transcript=self.transcripts.setdefault(f"analyst {fid}", []))
        if isinstance(found, Stopped):
            self.findings[fid] = found
            await self._save()
            events.emit("finding", finding=fid, stopped=found.reason)
            return f"{fid}: the analyst could not finish ({found.reason}). Nothing from this task can be used."
        await check(self.analyses, found)
        self.findings[fid] = found
        await self._save()
        events.emit("finding", finding=fid, claim=found.claim, warnings=len(found.flags))
        return found.text()

    async def _build_brief(self, form: BriefForm):
        """The brief's checks. A problem its own words can fix (`Rewrite`) is fixed by a small
        repair call that gets only the brief and what to fix (`_rewrite`); anything else goes back
        to the lead, with its whole context, as a refusal. Span tests (2026-11-12): a brief refused
        for a missing measure was retried with the lead's full ~2,500-token context."""
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
                                           reasoning_effort="low")
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
        # span tests (2026-11-12): on a trend question the lead twice wrote its brief straight from
        # the monthly profile, before giving any task; say what to do, not only what's wrong
        if not done and any(ch.isdigit() for ch in form.answer) and not self.findings:
            raise ToolError("You have no findings yet, and the numbers in a brief must come from findings. "
                            "Give the analysts their tasks first, with ask_analysts.")
        for fid in form.findings:
            if fid not in done:
                raise ToolError(f"There is no finding {fid}. Findings: {', '.join(done) or 'none'}.")
            # the span tests (2026-11-12): shown this warning, the lead answered "Yes, they move
            # together very closely" -- a link true by definition is not a finding to rest on
            built = next((w for w in done[fid].flags if "TRUE BY DEFINITION" in w), None)
            if built:
                raise Rewrite(f"{fid} is a correlation true by definition ({built}): it is not a finding. "
                              f"Take {fid} out of findings, and say in the answer that the link holds by definition.")
        if not form.cannot_tell.strip():
            raise Rewrite("cannot_tell is empty: say what these figures cannot tell. They show where and for "
                          "whom, not why.")
        # only the findings' results and code's reruns of them: on live run 3 profiles were allowed
        # too, and three invented numbers passed because they happened to be in some profile
        cited = [i for f in done.values() for i in f.results + f.checked_with]
        check_numbers(self.analyses, cited, form.answer, form.ruled_out, form.cannot_tell)
        said = plain(" ".join((form.answer, form.ruled_out, form.cannot_tell)))
        # (which measure the figures are on is written into the brief by code: `figures_from`)
        # a state set apart as an outlier, named without saying so (span tests: "Andaman & Nicobar
        # recorded the highest... 16.4%", the warning passed over)
        for name in dict.fromkeys(n for i in cited for n in outliers_in(self.analyses.results.get(i, ""))):
            if name in form.answer and not any(w in said for w in OUTLIER_WORDS):
                raise Rewrite(f"The answer names {name}, which code set apart as a statistical outlier. "
                              "Say that its figure may rest on a small sample.")
        return form

    def lead(self) -> Agent:
        """The lead as this investigation runs it (the span tests run its steps one at a time)."""
        return Agent("lead", PERSONA, tools=Registry([self.analyses.tool("profile"), Tool(self.ask_analysts)]),
                     model=config.LEAD_MODEL, max_tokens=1500, max_tool_calls=12,
                     budget=QUESTION_BUDGET, answer_on_stop=False, form_build=self._build_brief,
                     tool_timeout=900, shorten=("profile",), keep_recent=1)

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
