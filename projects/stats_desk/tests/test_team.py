"""The investigation, end to end, with scripted agents: the real lead and analyst code, code's
checks, the real analysis server (in-process) on the test release, the real governor, meter and
trace. No tokens.

Analysts run at the same time, so their model calls interleave in no fixed order. The scripted
model therefore answers each agent from its own script, picked by who is asking.
"""

import json
import re

import pytest
from stubs import call, calls, plan, say

from stats_desk.agents.agent import Stopped
from stats_desk.agents.lead import MAX_ANALYSTS, Investigation, investigate, profile_list
from stats_desk.analysis.client import connect
from stats_desk.analysis.server import create
from stats_desk.core import llm, meter
from stats_desk.core.governor import Governor
from stats_desk.core.tools import ToolError


class Routed:
    """Replies from one script per agent: 'lead', 'analyst <task>', and 'task check' (the task
    check's grouping call; without a script it finds no groups, so every task goes through)."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.asked = []

    async def create(self, **request):
        from stats_desk.agents.grouping import INSTRUCTIONS
        messages = request["messages"]
        system, users = messages[0]["content"], [m["content"] for m in messages if m["role"] == "user"]
        if system == INSTRUCTIONS:
            self.asked.append(("task check", request))
            script = self.scripts.get("task check")
            return (script.pop(0) if script else call("groups", groups=[])), {}
        if system.startswith("You are the lead"):
            key = "lead"
        else:
            key = "analyst " + next(u for u in users if u.startswith("Task from the lead: "))[20:]
        self.asked.append((key, request))
        step = self.scripts[key].pop(0)
        return (step(request) if callable(step) else step), {}


def result_id(request, kind):
    """The id of the analysis result of this kind that the agent has seen."""
    tool = next(m["content"] for m in request["messages"]
                if m["role"] == "tool" and m["content"].startswith(f"result {kind}-"))
    return tool.split("\n")[0].removeprefix("result ")


def findings_by_task(request):
    """{task: finding id} from the ask_analysts results the lead has seen."""
    return {m.group(2): m.group(1) for msg in request["messages"] if msg["role"] == "tool"
            for m in re.finditer(r"^(f\d+) -- task: (.*)$", msg["content"], re.M)}


@pytest.fixture
def routed():
    holder = {}

    def install(scripts):
        holder["model"] = Routed(scripts)
        holder["previous"] = llm.set_backend(holder["model"])
        holder["governor"] = llm.set_governor(Governor(limits={}))
        return holder["model"]
    yield install
    if "model" in holder:
        llm.set_backend(holder["previous"])
        llm.set_governor(holder["governor"])


GAP = "compare urban young women's and men's unemployment"
CHANGE = "how did unemployment change since 2024"


async def test_an_investigation_runs_from_question_to_checked_brief(release_path, routed):
    def brief(request):
        ids = findings_by_task(request)
        return call("json", answer="Among urban 15-29 year olds, women's unemployment is 18.9% against "
                    "men's 11.8% in 2025 (usual status), a gap of 7.1 points; in villages it is 0.4.",
                    findings=[ids[GAP]], ruled_out="A fall since 2024: -0.1 points, within rounding.",
                    cannot_tell="Why the gap exists.")

    model = routed({
        "lead": [calls(("profile", {"names": ["UR_annual_2025"]})),
                 calls(("ask_analysts", plan(GAP, CHANGE))), brief],
        f"analyst {GAP}": [
            call("gap", table="ur_annual", field="gender", a="female", b="male", across="sector",
                 where={"age": "15-29 years"}),
            lambda r: call("json", claim="Young urban women's unemployment is 7.1 points above young urban "
                           "men's; in villages the gap is 0.4.", results=[result_id(r, "gap")])],
        f"analyst {CHANGE}": [
            call("change", table="ur_annual"),
            lambda r: call("json", claim="Unemployment fell from 3.2 to 3.0.", results=[result_id(r, "change")]),
            lambda r: call("json", claim="Unemployment went from 3.2% in 2024 to 3.1% in 2025 (-0.1 points).",
                           results=[result_id(r, "change")])],
    })
    async with connect(server=create(release_path)) as analyses:
        brief_, run = await investigate(analyses, "Why do young women in cities struggle to find work?")

    assert brief_.stopped is None, brief_.stopped
    assert "18.9%" in brief_.answer and [f.task for f in brief_.findings] == [GAP]
    # every turn must be a tool call: a live run lost its brief to a lead writing it as text;
    # the lead's first two are fixed: read profiles, then hand out tasks (span tests)
    lead_choices = [r.get("tool_choice") for key, r in model.asked if key == "lead"]
    assert [c if isinstance(c, str) else c["function"]["name"] for c in lead_choices] == ["profile", "ask_analysts", "required"]
    assert {r.get("tool_choice") for key, r in model.asked if key != "lead"} == {"required"}
    # the analyst's invented 3.0 was refused, with the number named, and fixed
    change_id = next(k for k, f in run.findings.items() if f.task == CHANGE)
    assert "These numbers are not in the results you cited: 3.0" in json.dumps(run.transcripts[f"analyst {change_id}"])
    # what the lead read of each finding: the claim, the figures written by code, the reruns, warnings
    lead_saw = [r for key, r in model.asked if key == "lead"][-1]
    finding = next(m["content"] for m in lead_saw["messages"] if m["role"] == "tool" and "-- task:" in m["content"])
    assert "claim: Young urban women's unemployment is 7.1 points" in finding
    assert "urban 18.9 vs 11.8 = +7.1" in finding                     # the figures, from the result itself
    assert "rerun by code in the other period and on the other measure" in finding and "in 2024:" in finding
    assert "warnings from code: none" in finding
    # by then the profile it read is cut to two lines
    profile = next(m["content"] for m in lead_saw["messages"] if m["role"] == "tool" and m["content"].startswith("result profile-"))
    assert profile.endswith("[earlier result, shortened]") and profile.count("\n") == 1
    # the trace: each agent's calls under it, code's reruns recorded as code's
    paths = {c["path"] for c in brief_.trace["model calls"]}
    assert {"lead", "lead > ask_analysts > analyst f1", "lead > ask_analysts > analyst f2"} <= paths
    assert any(t["tool"] == "recheck" and t["by"] == "code" for t in brief_.trace["tools"])
    assert brief_.trace["outcome"] == "brief"


async def test_the_lead_is_first_shown_the_profiles_by_name_and_table_only(release_path):
    async with connect(server=create(release_path)) as analyses:
        listed = profile_list(await analyses.call("tables"))
    assert "- UR_annual_2025: from table ur_annual -- unemployment rate (UR), annual; periods 2024 to 2025" in listed
    assert "headline" not in listed                                  # names, not the profiles


async def test_the_lead_cannot_ask_for_more_tasks_than_the_limit_or_the_budget(release_path):
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        run.findings = {f"f{i}": None for i in range(1, MAX_ANALYSTS + 1)}
        with meter.metered() as run.usage:
            with pytest.raises(ToolError, match=f"All {MAX_ANALYSTS} analyst tasks are used"):
                await run.ask_analysts(**plan("one more"))
        run.findings = {}
        with meter.metered() as run.usage:
            meter.record("m", 35_000, 0, 1.0)                        # 35k spent: 60k - 6k - 35k pays for 1
            with pytest.raises(ToolError, match="Send at most 1 task now"):
                await run.ask_analysts(**plan("a", "b", "c"))
        # the first call keeps a task back, to follow up on what the first round finds
        with meter.metered() as run.usage:
            with pytest.raises(ToolError, match="Send at most 3 tasks now \\(in your first call, keeping room"):
                await run.ask_analysts(**plan("a", "b", "c", "d"))


async def test_the_plan_is_kept_and_its_later_tasks_come_back_with_the_findings(release_path, routed):
    """The lead plans like an analyst: what first, what next depending on it."""
    routed({"lead": [calls(("profile", {"names": ["UR_annual_2025"]})),
                     calls(("ask_analysts", plan(GAP, later=["Check the gap by state, if it is large (waits for: the gap)"]))),
                     call("json", answer="I could not establish it.", findings=[], cannot_tell="Why.")],
            f"analyst {GAP}": [call("gap", table="ur_annual", field="gender", a="female", b="male"),
                               lambda r: call("json", claim="Women's unemployment is above men's.",
                                              results=[result_id(r, "gap")])]})
    async with connect(server=create(release_path)) as analyses:
        _, run = await investigate(analyses, "Do women find work harder than men?")
    assert run.record.plan == [{"approach": "Test what the question asks, then follow up.", "tasks": {"f1": GAP},
                                "later": ["Check the gap by state, if it is large (waits for: the gap)"]}]
    came_back = next(m["content"] for m in run.transcripts["lead"] if m["role"] == "tool" and "-- task:" in m["content"])
    assert came_back.endswith("Or hand in the brief if the question is answered.") and "Your plan's later tasks:" in came_back


async def test_the_lead_plans_from_what_is_published_together(release_path):
    """The list of profiles says which fields MoSPI splits together (span tests: without it 2
    of 3 plans asked for young people by education, which isn't published; and the lead read
    every profile to find which table holds the question's groups), which measure a split is
    limited to (span test: a lead's task asked for education on CWS), and what the groups mean
    (live run: "cities" taken to be unpublished). Said once for tables split alike."""
    async with connect(server=create(release_path)) as analyses:
        listed = profile_list(await analyses.call("tables"))
    assert "sector urban = towns and cities" in listed
    together = next(line for line in listed.split("\n") if line.startswith("published together: "))
    assert together.startswith("published together: ur_annual, wpr_annual: age group")
    # the test release has CWS only in its totals: every split is on PS+SS only
    assert "education: only by gender, sector (ages 15 years and above, measure PS+SS only)" in together
    assert "age group 15-29 years: only by gender, sector, state (measure PS+SS only)" in together
    assert together.count("education: only by") == 1
    monthly = next(line for line in listed.split("\n") if "from table ur_monthly" in line)
    assert "measure CWS;" in monthly


async def test_a_task_whose_group_isnt_published_together_is_refused_and_a_comparison_isnt(release_path, model):
    """The same names, put together two ways: one group (religion x state isn't published) or
    two groups set against each other (both are). The grouping call reads which; code asks the
    server. A reply that can't be read refuses nothing."""
    one = "Find the UR for Muslim women in Kerala for 2025."
    two = "Compare the UR of Muslim women with that of women in Kerala for 2025."
    model.steps += [call("groups", groups=[{"religion": "Islam", "gender": "female", "state": "Kerala"}]),
                    call("groups", groups=[{"religion": "Islam", "gender": "female"}, {"gender": "female", "state": "Kerala"}]),
                    say("no idea")]
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        run.tables = await analyses.call("tables")
        before = set(analyses.results)
        with pytest.raises(ToolError, match=r"task 1: Islam \(religion\), female \(gender\), Kerala \(state\) as one group: "
                                            r"ur_annual: gender x religion x state is not published together.*"
                                            "the nearest groups that are published, each on its own"):
            await run.check_tasks([one])
        await run.check_tasks([two])
        await run.check_tasks([one])                     # unreadable reply: let through
        assert set(analyses.results) == before           # the probes are not kept as results
    assert "Kerala" in model.requests[0]["messages"][1]["content"]
    # a task naming one field only needs no call
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        run.tables = await analyses.call("tables")
        await run.check_tasks(["Find the UR in Kerala and Tamil Nadu for 2025."])
    assert len(model.requests) == 3


async def test_the_lead_is_told_up_front_which_of_the_questions_groups_arent_published(release_path, model):
    """Before the lead's first call: the question's groups, read by the same small call, checked
    with the server, and the nearest published ones for any that aren't, in the lead's context.
    Nothing is said, and no call made, when the question names one field only."""
    model.steps += [call("groups", groups=[{"religion": "Islam", "gender": "female", "state": "Kerala"}])]
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "Is unemployment higher among Muslim women in Kerala?")
        run.tables = await analyses.call("tables")
        before = set(analyses.results)
        await run.read_question()
        assert set(analyses.results) == before              # the probes are not kept as results
        note = run.question_note
        assert note.startswith("The question's groups, as code checked them with the server:")
        assert ("- Islam (religion), female (gender), Kerala (state): not published together. The nearest that are, "
                "each on its own: female (gender), Kerala (state); Islam (religion), female (gender)") in note
        assert note in run.context() and run._refused_as_unpublished()      # its refusal is the brief's evidence
        # one field only: no call, no note
        other = Investigation(analyses, "Is unemployment higher among women than men?")
        other.tables = run.tables
        await other.read_question()
        assert other.question_note == "" and len(model.requests) == 1


async def test_a_task_naming_a_setting_its_groups_arent_published_on_is_caught(release_path):
    """Span test: told which measure each split is on, a lead still asked for graduates on CWS and
    all ages. Code asks the server, no tokens, and the probes are not kept as results."""
    from stats_desk.agents.lead import unpublished_settings
    async with connect(server=create(release_path)) as analyses:
        tables = await analyses.call("tables")
        found = await unpublished_settings(analyses, tables, "Find the UR for 7.graduate on CWS, all ages, for 2025")
        assert found.startswith("7.graduate (education) on measure CWS and age All ages isn't published together in ur_annual")
        assert "the nearest that is: measure PS+SS, age 15 years and above" in found
        assert await unpublished_settings(analyses, tables, "Find the UR for 7.graduate for 2025") is None
        assert await unpublished_settings(analyses, tables, "Find the UR for graduates on PS+SS, 2025") is None
        assert not [i for i in analyses.results if not i.startswith("tables")]


async def test_the_task_checks_refusal_names_the_settings_the_task_named(release_path):
    """Its wording comes from the task's own settings, not from a dataset's (it named "the
    measure or age group" whatever the task said)."""
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        run.tables = await analyses.call("tables")
        with pytest.raises(ToolError, match="without the age group the question doesn't ask about"):
            await run.check_tasks(["Find the UR for 7.graduate, age 15-29 years, for 2025"])
        with pytest.raises(ToolError, match="without the measure or age group the question"):
            await run.check_tasks(["Find the UR for 7.graduate on CWS, all ages, for 2025"])


async def test_the_eval_weak_spots_are_caught_by_code(release_path):
    """The eval's misses, each now a check: a link true by definition said even when
    the brief cites another finding (Q5); an outlier named with an odd space (Q7); a month MoSPI
    skipped (Q2); "fell" against a trend's "no clear trend" (Q2). (Q1's two lookups for one
    comparison are headed off in `value`'s description instead.)"""
    from stats_desk.agents.analyst import Finding
    from stats_desk.agents.checks import flags, outliers_in
    from stats_desk.agents.lead import BriefForm, Rewrite
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        rel = await analyses.call("relate", x_table="ur_annual", y_table="wpr_annual")
        cmp = await analyses.call("compare", table="ur_annual", by="state")
        ser = await analyses.call("series", table="ur_monthly")
        trd = await analyses.call("trend", table="ur_monthly")
        rid = lambda text: text.split("\n")[0].removeprefix("result ")
        linked = Finding("f1", "t", "They go together.", [rid(rel)])
        linked.flags = flags(analyses, linked)
        ranks = Finding("f2", "t", "Top and bottom states match.", [rid(cmp)])
        run.findings = {"f1": linked, "f2": ranks}
        with pytest.raises(Rewrite, match="true by definition: WPR is computed from LFPR and UR"):
            run._check_brief(BriefForm(answer="Yes: states rank the same on both.", findings=["f2"], cannot_tell="Why."))
        analyses.results["compare-0a0a0a"] = ("result compare-0a0a0a\n36 groups; statistical outliers (possibly "
                                              "small samples): Andaman & Nicobar Islands 16.4")
        assert outliers_in(analyses.results["compare-0a0a0a"]) == ["Andaman & Nicobar Islands"]
        odd = "Andaman & Nicobar Islands"                       # gpt-oss's narrow no-break space
        ranks = Finding("f2", "t", "Top and bottom states match.", [rid(cmp), "compare-0a0a0a"])
        run.findings = {"f2": ranks}
        with pytest.raises(Rewrite, match="statistical outlier"):
            run._check_brief(BriefForm(answer=f"It is highest in {odd}.", findings=["f2"], cannot_tell="Why."))
        monthly = Finding("f3", "t", "It fell over the months.", [rid(ser), rid(trd)])
        run.findings = {"f3": monthly}
        with pytest.raises(Rewrite, match="The monthly figures skip July 2025, August 2025"):
            run._check_brief(BriefForm(answer="It stayed about the same.", findings=["f3"], cannot_tell="Why."))
        assert any("no clear trend" in w and "'fell'" in w for w in flags(analyses, monthly)) == \
            ("no clear trend" in trd)


async def test_a_brief_saying_something_isnt_published_needs_a_refusal_behind_it(release_path):
    """Live run: "unemployment by gender for urban areas is not published" -- it is; nobody had
    asked. An unproven claim goes to the repair call (a refusal to the lead looped in the eval)."""
    from stats_desk.agents.analyst import Finding
    from stats_desk.agents.lead import BriefForm, Rewrite
    async with connect(server=create(release_path)) as analyses:
        text = await analyses.call("value", table="ur_annual")
        rid = text.split("\n")[0].removeprefix("result ")
        run = Investigation(analyses, "q")
        run.findings = {"f1": Finding("f1", "t", "c", [rid])}
        run.transcripts["analyst f1"] = [{"role": "tool", "kind": "ok", "content": text}]
        brief = BriefForm(answer="It is published overall; it isn’t published by gender for cities.",
                          findings=["f1"], cannot_tell="Why.")
        with pytest.raises(Rewrite, match="isn't published.*nothing in this question shows.*Take that claim out"):
            run._check_brief(brief)
        # an honest cannot_tell is not such a claim (eval: "the figures do not provide the reasons why")
        honest = BriefForm(answer="It is published overall.", findings=["f1"],
                           cannot_tell="The figures do not provide the reasons why.")
        assert run._check_brief(honest) is honest
        # "published only for <another age group>" says it IS published, there: no evidence (eval Q8)
        run.transcripts["analyst f1"].append({"role": "tool", "kind": "refused",
                                              "content": "ur_annual: this split is published only for age 15 years and above"})
        with pytest.raises(Rewrite):
            run._check_brief(brief)
        run.transcripts["analyst f1"].pop()
        # once a request was refused as not published, saying so is backed
        run.transcripts["analyst f1"].append({"role": "tool", "kind": "refused",
                                              "content": "ur_annual: gender x religion x state is not published together"})
        assert run._check_brief(brief) is brief
        # so is a result listing what wasn't published (the eval's Q2: "March 2026 is not published")
        run.transcripts["analyst f1"].pop()
        series = await analyses.call("series", table="ur_monthly")
        assert "not published: " in series and run._check_brief(brief) is brief


async def test_the_groups_a_question_asked_for_together_are_probed_as_one_combination(release_path, release):
    """The eval's Q4: Muslim women in all of India, and all women in one state, were asked for
    one by one because the combination isn't published; code asks the server about the
    combination itself, and its refusal backs the brief's "not published"."""
    async with connect(server=create(release_path)) as analyses:
        await analyses.call("value", table="ur_annual", where={"gender": "female", "religion": "Islam"})
        state = next(s for s in release.tables["ur_annual"].values["state"] if s != "All India")
        try:
            await analyses.call("value", table="ur_annual", where={"gender": "female", "state": state})
        except ToolError:
            pass
        run = Investigation(analyses, "q")
        await run._probe_combination()
    assert "not published together" in run.probed_unpublished and run._refused_as_unpublished()


async def test_a_task_that_came_back_empty_says_nothing_about_what_is_published(release_path):
    """Live run: three analysts stopped, and the lead wrote that unemployment by
    education isn't published -- its own profile showed 11.2% for graduates."""
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        run.transcripts["analyst f1"] = [
            {"role": "tool", "kind": "refused", "content": "ur_annual: this split is published only for age 15 years and above"},
            {"role": "tool", "kind": "ok", "content": "result value-123456\n..."}]
        said = run._unfinished("f1", Stopped(agent="analyst f1", reason="reached its token budget of 12,000"))
    assert "says nothing about whether the figures are published" in said
    assert "- ur_annual: this split is published only for age 15 years and above" in said and "value-123456" not in said
    assert "give the task again" in said


async def test_a_brief_number_from_nowhere_and_an_empty_answer_are_refused(release_path, routed):
    routed({"lead": [call("json", answer="Unemployment is 4.2%.", findings=[], cannot_tell="Why."),
                     call("json", answer="", findings=[], cannot_tell="Why."),
                     call("json", answer="I could not establish the rate.", findings=[]),
                     call("json", answer="I could not establish the rate.", findings=[], cannot_tell="The rate.")]})
    async with connect(server=create(release_path)) as analyses:
        brief, run = await investigate(analyses, "What is unemployment?")
    lead = json.dumps(run.transcripts["lead"])
    assert "You have no findings yet" in lead                       # a number before any task: give tasks
    assert "The answer is empty" in lead
    assert "cannot_tell: Field required" in lead                    # it must be filled in
    assert brief.answer == "I could not establish the rate."


async def test_code_flags_what_it_can_see_before_the_lead_judges(release_path):
    from stats_desk.agents.analyst import Finding
    from stats_desk.agents.checks import flags
    async with connect(server=create(release_path)) as analyses:
        value = analyses.tool("value").function
        young_urban = (await value(table="ur_annual", where={"age": "15-29 years", "gender": "female",
                                                             "sector": "urban"})).split("\n")[0][7:]
        all_women = (await value(table="ur_annual", where={"gender": "female"})).split("\n")[0][7:]
        finding = Finding("f1", "t", "Young urban women: 18.9% against 3.1% for all women, because of jobs.",
                          [young_urban, all_women])
        found = flags(analyses, finding)
    assert any("differ in age, sector: not like with like" in f for f in found)
    assert any("'because'" in f for f in found)


def test_a_long_result_is_written_short_for_the_lead():
    from stats_desk.agents.analyst import compact
    states = ", ".join(f"S{i} {i}" for i in range(36))
    text = (f"result compare-aaaaaa\nUR, annual, 2025 | by state\ntotal (All India): 3.1\n"
            f"highest first: {states}\n36 groups; lowest S0 0, highest S35 35")
    short = compact(text)
    assert short.startswith("UR, annual, 2025") and "total (All India): 3.1" in short
    assert "highest first: S0 0, S1 1, S2 2, ... 30 more ..., S33 33, S34 34, S35 35" in short


async def test_an_analyst_gets_only_the_tables_of_the_indicator_its_task_names(release_path):
    from stats_desk.agents.analyst import tables_for
    from stats_desk.core.llm import count_tokens
    async with connect(server=create(release_path)) as analyses:
        tables = await analyses.call("tables")
    unemployment = tables_for(tables, "How did unemployment of young women change since 2024?")
    assert "ur_annual:" in unemployment and "wpr_annual:" not in unemployment and "profiles:" not in unemployment
    # a long list as a count (the analyses match the name); a short one spelled out
    assert "state values: 37, All India the total; name one as the task does" in unemployment
    assert "Kerala" not in unemployment and "education values: 1.not literate" in unemployment
    # "unemployed" is not "employed": only the unemployment tables
    assert "wpr_annual:" not in tables_for(tables, "Which groups are most often unemployed?")
    assert "wpr_annual:" in tables_for(tables, "How many women are employed in cities?")
    # a task naming no indicator gets every table
    anything = tables_for(tables, "Compare rural and urban women in 2025.")
    assert all(f"{t}_annual:" in anything for t in ("ur", "wpr"))     # the test release has no lfpr table
    assert count_tokens(unemployment) < count_tokens(anything)


def test_a_rerun_that_reverses_or_halves_is_said_outright_and_a_smaller_one_is_left_to_the_lead():
    from stats_desk.agents.checks import held as held_with_advice

    def held(*args):          # the facts; each warning ends with what to do, checked once below
        return [re.sub(r"; (the finding doesn't hold there, so say so|say it is weaker there)$", "", w)
                for w in held_with_advice(*args)]
    result = ("result gap-aaaaaa\nunemployment rate (UR), annual, 2025 | usual status (PS+SS) | by gender\n"
              "overall: female 8.4, male 5.3; female minus male = +3.1 points")

    def rerun(year_gap, measure_line):
        return (f"result recheck-bbbbbb\nin 2024:\nUR, 2024 | by gender\n"
                f"overall: female 1, male 1; female minus male = {year_gap} points\n"
                f"on current weekly status (CWS):\n{measure_line}")
    refused = "ur_annual: this split is published only for measure PS+SS"
    assert held("gap", result, rerun("-1.7", refused)) == [
        "reversed in 2024: female minus male was +3.1 points in the finding (2025, usual status (PS+SS)), -1.7 points in 2024"]
    assert held("gap", result, rerun("+1.2", refused)) == [
        "half as large or less in 2024: female minus male was +3.1 points in the finding (2025, usual status (PS+SS)), +1.2 points in 2024"]
    assert held("gap", result, rerun("+2.0", refused)) == []          # smaller, same way: the lead's call
    smaller_on_cws = "UR, CWS | by gender\noverall: female 1, male 1; female minus male = +0.5 points"
    assert held("gap", result, rerun("+2.9", smaller_on_cws)) == [
        "half as large or less on current weekly status (CWS): female minus male was +3.1 points in the finding "
        "(2025, usual status (PS+SS)), +0.5 points on current weekly status (CWS)"]
    assert held("value", "result value-cccccc\nvalue: 3.1%", rerun("-1.7", refused)) == []    # no direction
    assert held_with_advice("gap", result, rerun("-1.7", refused))[0].endswith("the finding doesn't hold there, so say so")


def _finding(analyses, fid, claim, result_ids, flags=()):
    from stats_desk.agents.analyst import Finding
    return Finding(fid, "t", claim, list(result_ids), flags=list(flags))


async def test_a_brief_cannot_rest_on_a_link_true_by_definition_or_name_an_outlier_silently(release_path):
    from stats_desk.agents.checks import check
    from stats_desk.agents.lead import BriefForm
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        rel = (await analyses.call("relate", x_table="ur_annual", y_table="wpr_annual")).split("\n")[0][7:]
        built = await check(analyses, _finding(analyses, "f1", "They go together.", [rel]))
        assert any("TRUE BY DEFINITION" in w for w in built.flags)
        run.findings = {"f1": built}
        with pytest.raises(ToolError, match="f1 is a correlation true by definition"):
            run._check_brief(BriefForm(answer="They go together.", findings=["f1"], cannot_tell="why"))
        # the same link reported as true by definition, not as a finding: fine
        run._check_brief(BriefForm(answer="The two move together by definition.", findings=[], cannot_tell="why"))

        cmp = (await analyses.call("compare", table="ur_annual", by="state", where={"gender": "female"})).split("\n")[0][7:]
        run.findings = {"f2": _finding(analyses, "f2", "Chandigarh is highest.", [cmp])}
        named = "Chandigarh has the highest unemployment among women."
        from stats_desk.agents.checks import outliers_in
        outlier = outliers_in(analyses.results[cmp])
        if outlier:                       # the trimmed test data may set a different state apart
            named = f"{outlier[0]} has the highest unemployment among women."
            with pytest.raises(ToolError, match="set apart as a statistical outlier"):
                run._check_brief(BriefForm(answer=named, findings=["f2"], cannot_tell="why"))
            run._check_brief(BriefForm(answer=named + " Its figure may rest on a small sample.",
                                             findings=["f2"], cannot_tell="why"))


def test_a_correlation_says_how_strong_it_is_and_a_small_one_is_no_link():
    from stats_desk.analysis.compute import strength
    assert strength(-0.19, 36).startswith("no clear link")              # the span test's "associated"
    assert strength(0.51, 36) == "a moderate link: across states the two tend to go the same way"
    assert strength(-0.8, 36) == "a strong link: across states the two tend to go opposite ways"


def test_cause_words_that_deny_a_cause_or_give_a_statistical_reason_are_not_claims():
    from stats_desk.agents.checks import cause_claims
    # the four sentences of the span tests' briefs the word match flagged
    assert cause_claims("These figures show the gap, but the data cannot explain why this is the case.") == []
    assert cause_claims("They move together by definition, not because higher participation causes more work.") == []
    assert cause_claims("This state is flagged as a statistical outlier due to its small sample size;") == []
    assert cause_claims("A degree is associated with it, but the figures show a correlation, not a causal effect.") == []
    # and real claims still are
    assert cause_claims("Women are unemployed because of a lack of jobs.") == ["because"]
    assert cause_claims("A degree causes unemployment.") == ["causes"]


async def test_a_brief_before_any_task_is_sent_back_to_give_tasks(release_path):
    from stats_desk.agents.lead import BriefForm
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        with pytest.raises(ToolError, match="no findings yet.*Give the analysts their tasks first"):
            run._check_brief(BriefForm(answer="It was 5.0% in September.", findings=[], cannot_tell="why"))


async def test_the_brief_says_which_tables_periods_and_measures_its_figures_come_from(release_path):
    from stats_desk.agents.lead import figures_from
    async with connect(server=create(release_path)) as analyses:
        annual = await analyses.call("value", table="ur_annual")
        cws = await analyses.call("value", table="ur_annual", where={"measure": "CWS"})
        monthly = await analyses.call("series", table="ur_monthly")
        linked = await analyses.call("relate", x_table="ur_annual", y_table="wpr_annual")
    assert figures_from([annual, cws, monthly, linked]) == [
        "unemployment rate (UR), annual, 2025, usual status (PS+SS)",
        "unemployment rate (UR), annual, 2025, current weekly status (CWS)",
        "unemployment rate (UR), monthly, 2025-04 to 2026-04, current weekly status (CWS)",
        "worker population ratio (WPR), annual, 2025, usual status (PS+SS)"]

async def test_a_brief_its_own_words_can_fix_is_fixed_by_a_small_call_with_just_the_brief(release_path, model):
    from stats_desk.agents.analyst import Finding
    from stats_desk.agents.lead import BriefForm
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        annual = (await analyses.call("value", table="ur_annual")).split("\n")[0][7:]
        run.findings = {"f1": Finding("f1", "t", "c", [annual])}
        model.steps += [call("json", answer="Unemployment was 3.1% in 2025.", findings=["f1"],
                             cannot_tell="Why it is at that level.")]
        fixed = await run._build_brief(BriefForm(answer="Unemployment was 3.1% in 2025.", findings=["f1"],
                                                 cannot_tell=""))
    assert fixed.cannot_tell == "Why it is at that level."
    sent = model.requests[0]["messages"]
    assert len(sent) == 2 and sum(len(m["content"]) for m in sent) < 1500          # just the brief and the fix
    assert "cannot_tell is empty" in sent[1]["content"]                             # what to fix
    assert "Profiles you can read" not in sent[1]["content"] and "-- task:" not in sent[1]["content"]


async def test_a_repair_that_still_fails_goes_back_to_the_lead(release_path, model):
    from stats_desk.agents.analyst import Finding
    from stats_desk.agents.lead import BriefForm
    async with connect(server=create(release_path)) as analyses:
        run = Investigation(analyses, "q")
        annual = (await analyses.call("value", table="ur_annual")).split("\n")[0][7:]
        run.findings = {"f1": Finding("f1", "t", "c", [annual])}
        unfixed = dict(answer="Unemployment was 3.1% in 2025.", findings=["f1"], cannot_tell="")
        model.steps += [call("json", **unfixed), call("json", **unfixed)]
        with pytest.raises(ToolError, match="cannot_tell is empty"):
            await run._build_brief(BriefForm(**unfixed))


async def test_a_brief_mixing_measures_or_annual_with_monthly_says_so_in_its_answer(release_path):
    """The first live question: "fell 0.4 (annual, usual status)... However, the
    monthly series shows a rising trend (weekly status)", as if they disagreed. Code adds the
    note; same-measure annual findings get none."""
    from stats_desk.agents.analyst import Finding
    from stats_desk.agents.lead import not_comparable
    async with connect(server=create(release_path)) as analyses:
        change = (await analyses.call("change", table="ur_annual")).split("\n")[0].removeprefix("result ")
        monthly = (await analyses.call("series", table="ur_monthly")).split("\n")[0].removeprefix("result ")
        compare = (await analyses.call("compare", table="ur_annual", by="gender")).split("\n")[0].removeprefix("result ")
        results = analyses.results
    f1, f2, f3 = Finding("f1", "t", "c", [change]), Finding("f2", "t", "c", [monthly]), Finding("f3", "t", "c", [compare])
    note = not_comparable([f1, f2], results)
    assert note.startswith("Note: the annual figures (") and "usual status (PS+SS)" in note
    assert "and the monthly figures (" in note and "current weekly status (CWS)" in note
    assert note.endswith("are not directly comparable: different measures and periods.")
    assert "f1" not in note and "f2" not in note            # figures named by what they are, not by id
    assert not_comparable([f1, f3], results) == ""          # both annual, both usual status
    assert not_comparable([f2], results) == ""              # one finding: nothing to compare


def test_finding_ids_are_left_out_of_what_the_reader_sees():
    from stats_desk.agents.lead import without_ids
    assert without_ids("It fell 0.4 points (f1). However, it rose (f2), and (f1, e2) agree [f3].") == \
        "It fell 0.4 points. However, it rose, and agree."
    assert without_ids("Rates differ (see f1 and f2); fifteen (15) states.") == "Rates differ; fifteen (15) states."
