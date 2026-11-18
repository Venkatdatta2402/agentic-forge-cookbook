"""The analyst: carries out one task the lead gives it, with the analysis tools, and hands in a
finding. Several run at once when the lead's tasks are independent; each sees only its own task
(chapter 2: a branch gets its goal, not the parent's conversation).

The hand-in is one sentence and the ids of the results it rests on. Before,
it was a free summary, which is where a number got the wrong label, lost its unit, or gained a
"because". Now the figures the lead reads are written by code from the results themselves
(`compact`), and only the one sentence is the analyst's own words; code checks every number in it
against the results it cites.
"""

import re
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, Field

from .. import config
from ..core.tools import Registry, ToolError
from .agent import FORM, Agent, Stopped, answer

TOOLS = ("value", "compare", "gap", "change", "series", "average", "trend", "relate", "drill", "tables")
BUDGET = 12_000        # tokens for one task

PERSONA = f"""You are an analyst of India's official labour statistics (MoSPI's PLFS). You carry out ONE task from the lead analyst with the analysis tools, then hand in what you found.

How to work:
- Pick the analysis whose description fits the task. One analysis that answers it beats several lookups: groups side by side come from one gap, compare, or an analysis with `by`.
- Compare like with like: groups that differ in one field only. If the task compares groups that differ in several fields, compare like with like instead and say so.
- The tools compute every difference, change and correlation. Never do arithmetic yourself.
- If a tool refuses, its message says what is valid or published: fix the call from it. If the data can't answer the task, say so in your claim.
- If no single analysis does all of the task, run the one that gets closest, then the next. Never explain instead of acting.
- 2-4 analyses are usually enough. Analyses that don't depend on each other can go in the same turn.
- Every reply is a tool call: an analysis, or {FORM} to hand in. Never reply in prose.

Hand in with {FORM}:
- claim: one sentence answering the task: what the results show, which way and how much. Quote numbers exactly as a result shows them. The lead also gets the figures themselves from code, so don't list them all.
- results: the ids of the results the claim rests on (each result's first line, e.g. gap-83c1ba).
Say what the figures show, not why: the data has no causes in it."""


class Report(BaseModel):
    claim: Annotated[str, Field(description="one sentence answering the task, numbers exactly as the results show them")]
    results: Annotated[list[str], Field(description="ids of the results the claim rests on")]


@dataclass
class Finding:
    id: str
    task: str
    claim: str
    results: list
    figures: dict = field(default_factory=dict)        # result id -> its figures, written short by code
    checked_with: list = field(default_factory=list)   # recheck result ids (checks.py)
    elsewhere: dict = field(default_factory=dict)      # recheck id -> its figures, written short
    flags: list = field(default_factory=list)          # what code saw wrong or worth a look

    def text(self):
        """The finding as the lead reads it: warnings first, each saying what to do (span tests:
        with the warnings last, the lead passed over two of them)."""
        lines = [f"{self.id} -- task: {self.task}"]
        lines += [f"WARNING from code: {w}" for w in self.flags]
        lines += [f"claim: {self.claim}", "figures:"]
        lines += [f"  [{i}] " + figures.replace("\n", "\n    ") for i, figures in self.figures.items()]
        if self.elsewhere:
            lines.append("the same, rerun by code in the other period and on the other measure:")
            lines += [f"  [{i}] " + figures.replace("\n", "\n    ") for i, figures in self.elsewhere.items()]
        if not self.flags:
            lines.append("warnings from code: none")
        return "\n".join(lines)


# lines a reader of a finding doesn't need: every state's pair of values, every month's value
DROPPED = ("values: ", "states (x, y): ", "this describes states")
KEEP = 3        # of a long list: the first and last 3


def compact(text):
    """A result written short for the lead: its first line (the id) and lines of every value
    dropped, long lists cut to their first and last 3. A comparison of 36 states keeps the
    total, the top 3, the bottom 3 and the outliers."""
    out = []
    for line in text.split("\n")[1:]:
        if line.startswith(DROPPED):
            continue
        head, sep, items = line.partition(": ")
        if sep:
            parts = items.split("; " if "; " in items else ", ")
            if len(parts) > 2 * KEEP + 1:
                joiner = "; " if "; " in items else ", "
                items = joiner.join(parts[:KEEP] + [f"... {len(parts) - 2 * KEEP} more ..."] + parts[-KEEP:])
            line = head + sep + items
        out.append(line)
    return "\n".join(out)


# which indicator a task is about, by its words: "unemployed" must not count as "employed"
INDICATORS = {"ur": r"unemploy|\bUR\b",
              "lfpr": r"participat|labou?r force|\bLFPR\b",
              "wpr": r"\bworkers?\b|population ratio|\bWPR\b|\bemployed\b|\bemployment\b|in work\b|working"}


def tables_for(tables_text, task):
    """The tables list an analyst gets with its task: only the tables of the indicators the task
    names (all of them if it names none), and no profiles (analysts never read them). The three
    annual tables share every value, so this cuts table lines, ~210 tokens a call."""
    wanted = {i for i, pattern in INDICATORS.items() if re.search(pattern, task, re.I)} or set(INDICATORS)
    out = []
    for line in tables_text.split("\n")[1:]:
        table = re.match(r"(ur|lfpr|wpr)_(annual|monthly): ", line)
        if (table and table[1] not in wanted) or line.startswith("profiles: "):
            continue
        out.append(line)
    return "\n".join(out)


def check_numbers(analyses, cited, *texts):
    """Refuse a hand-in whose numbers aren't all in the results it cites."""
    missing = list(dict.fromkeys(n for t in texts if t for n in analyses.unbacked(t, cited)))
    if missing:
        raise ToolError(
            f"These numbers are not in the results you cited: {', '.join(missing)}. Quote numbers "
            "exactly as the results show them and cite every result they come from. Don't work "
            "numbers out yourself: gap, change and relate compute differences.")


async def run_analyst(analyses, finding_id, task, context, transcript=None):
    """One analyst instance on one task. Returns a Finding, or Stopped.

    Two lookups for one comparison (eval: women and men looked up one by one three
    times over) are headed off in `value`'s description and PERSONA, not refused at hand-in: a
    refusal there would cost the analyst more calls, not fewer."""
    async def build(report: Report):
        check_numbers(analyses, report.results, report.claim)
        return Finding(finding_id, task, report.claim, report.results,
                       figures={i: compact(analyses.results[i]) for i in report.results})

    short = re.sub(r"\s+", " ", task)[:70]
    agent = Agent(f"analyst {finding_id}", PERSONA, tools=Registry([analyses.tool(n) for n in TOOLS]),
                  model=config.ANALYST_MODEL, max_tokens=800, max_tool_calls=8, budget=BUDGET,
                  answer_on_stop=False, shorten=("drill", "tables"), form_build=build,
                  label=f"analyst {finding_id}: {short}",
                  # the backstop for when the rule in PERSONA slips (it says the same, once more)
                  when_stuck=("Act on it with a tool call. If no single analysis does all of the task, run the one "
                              "that gets closest now, then the next; then hand in with json what the results show, "
                              "and say in the claim what you could not get."))
    return await answer(agent, f"Task from the lead: {task}", Report, context=context,
                        transcript=transcript)


__all__ = ["Finding", "Report", "Stopped", "check_numbers", "compact", "run_analyst"]
