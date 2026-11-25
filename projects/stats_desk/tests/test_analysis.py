"""The analysis server, on the test release file (conftest.py): real PLFS unemployment numbers,
a stand-in second indicator, and a small made-up monthly table."""

import re

import pytest

from stats_desk.analysis import analyses as A
from stats_desk.analysis.client import connect
from stats_desk.analysis.data import Refusal, Table
from stats_desk.analysis.server import create
from stats_desk.config import RELEASE_PATH
from stats_desk.core.tools import invoke


def body(result):
    return result.split("\n", 1)[1]


def test_a_request_reads_mospis_published_row_and_states_every_default(release):
    out = A.value(release, "ur_annual")
    assert body(out) == ("unemployment rate (UR), annual, 2025 | usual status (PS+SS) | "
                         "age 15 years and above | everyone, All India\nvalue: 3.1%")
    assert A.value(release, "ur_annual", {"gender": "female", "sector": "urban", "age": "15-29 years"}).endswith("value: 18.9%")
    assert A.value(release, "ur_annual", {"period": 2024}).endswith("value: 3.2%")      # a number is fine


def test_wrong_fields_and_values_are_refused_with_the_valid_ones(release):
    with pytest.raises(Refusal, match="no field 'gendr'. Its fields: measure, state, gender"):
        A.value(release, "ur_annual", {"gendr": "female"})
    with pytest.raises(Refusal, match="education 'grad' could be any of 7.graduate, 8.post graduate"):
        A.value(release, "ur_annual", {"education": "grad"})           # two matches: not guessed
    with pytest.raises(Refusal, match="no education 'doctorate'; its total is 'all' \\(or write total\\). Values: .*7.graduate"):
        A.value(release, "ur_annual", {"education": "doctorate"})
    with pytest.raises(Refusal, match="no period 2019. Stored: 2024, 2025"):
        A.value(release, "ur_annual", {"period": "2019"})


def test_a_group_asked_for_without_mospis_number_is_the_one_it_names(release):
    """Live run: two analysts asked for "graduate", were refused (it is also inside
    "8.post graduate & above"), repeated it, and stopped."""
    def figures(asked):                       # the result without its id line (the id hashes the request)
        return A.value(release, "ur_annual", {"education": asked}).split("\n", 1)[1]
    assert figures("graduate") == figures("7.graduate") and "education = 7.graduate" in figures("graduate")
    assert figures("Post Graduate") == figures("8.post graduate & above")


def test_what_is_published_only_for_another_age_group_comes_on_that_one_said_so(release):
    """Eval: told "published only for age 15 years and above", an analyst reported the figures by
    education as not published, and the brief said MoSPI doesn't publish them."""
    asked = {"table": "ur_annual", "by": "education", "where": {"age": "All ages"}}
    with pytest.raises(Refusal, match="published only for age 15 years and above"):
        A.compare(release, **asked)
    out = A.nearest(release, A.compare, asked)
    # the note first, under the id: at the end, claims left it out
    assert out.split("\n")[1] == ("note: these figures are for age 15 years and above, not All ages as asked "
                                  "(All ages isn't published for this)")
    assert "| age 15 years and above |" in out.split("\n")[2] and "7.graduate" in out
    assert "age 15 years and above" in A.recheck(release, "compare", asked)            # code's reruns too
    with pytest.raises(Refusal, match="no field"):                                       # other refusals stand
        A.nearest(release, A.compare, {"table": "ur_annual", "by": "education", "where": {"agee": "x"}})


def test_several_groups_and_periods_come_in_one_lookup_with_the_total(release):
    """An eval analyst made five single lookups at ~3k tokens a call, ran out of its 12,000, and
    its figures were lost: listing the groups gets them all in one call."""
    out = A.value(release, "ur_annual", {"gender": ["female", "male"], "period": ["2024", "2025"]})
    lines = out.split("\n")
    assert lines[1].endswith("| by gender") and len([x for x in lines if x.startswith(("female, ", "male, "))]) == 4
    assert any(x.startswith("total (person), 2024: ") for x in lines) and any(x.startswith("total (person), 2025: ") for x in lines)
    with pytest.raises(Refusal, match="too many"):
        A.value(release, "ur_annual", {"state": list(release.tables["ur_annual"].values["state"])[:25], "period": ["2024", "2025"]})
    # code's rerun of it in the other measure reads the lists too, and stays on that measure
    assert "on current weekly status" in A.recheck(release, "value", {"table": "ur_annual", "where": {"gender": ["female", "male"]}})


def test_an_age_group_is_not_a_figure_to_back():
    from stats_desk.analysis.client import AGE, NUMBER
    said = "For ages 15‑29 and 15-29 year olds, aged 15 and above or 15+, it was 11.2% in 2025; 5.0-5.2% monthly."
    assert NUMBER.findall(AGE.sub("", said)) == ["11.2", "2025", "5.0", "5.2"]


def test_an_unpublished_combination_says_what_is_published(release):
    # (the trimmed fixture holds fewer splits than the real table; "gender x religion x sector" is in both)
    with pytest.raises(Refusal, match="religion x state is not published together. Published together: "
                                      ".*gender x religion x sector"):
        A.value(release, "ur_annual", {"state": "Bihar", "religion": "Islam"})
    with pytest.raises(Refusal, match="published only for age 15 years and above"):
        A.value(release, "ur_annual", {"education": "7.graduate", "age": "15-29 years"})


def test_compare_ranks_groups_but_never_one_that_contains_others(release):
    out = A.compare(release, "ur_annual", "education")
    assert "total (all): 3.1" in out
    assert re.search(r"highest first: 7\.graduate 11\.2, 8\.post graduate & above 10,", out)
    assert "not ranked: 9.secondary & above" in out
    assert "highest first:" in out and "9.secondary & above 6.5," not in out.split("highest first:")[1].split("\n")[0]


def test_overlapping_age_bands_are_side_by_side_never_ranked(release):
    out = A.compare(release, "ur_annual", "age")
    assert "15-29 years 9.9" in out and "side by side, not ranked" in out and "highest first" not in out


def test_outliers_are_set_apart_across_states_only(release):
    states = A.gap(release, "ur_annual", "gender", "female", "male", across="state")
    assert "statistical outliers (possibly small samples): Chandigarh +12.2" in states
    # across education the far-off groups are graduates: a finding, kept in the ranking
    education = A.gap(release, "ur_annual", "gender", "female", "male", across="education")
    assert "outliers" not in education
    assert "highest 8.post graduate & above +10.6" in education


def test_a_gap_across_another_field_is_computed_from_published_rows(release):
    out = A.gap(release, "ur_annual", "gender", "female", "male", across="sector", where={"age": "15-29 years"})
    assert "overall: female 11.3, male 9.3; female minus male = +2.0 points" in out
    assert "urban 18.9 vs 11.8 = +7.1; rural 8.6 vs 8.2 = +0.4" in out


def test_change_since_the_previous_period(release):
    assert "previous year: 2024 3.2 -> 2025 3.1, -0.1 points" in A.change(release, "ur_annual")
    monthly = A.change(release, "ur_monthly")
    assert "previous published month: 2026-02 5.1 -> 2026-04 5.2, +0.1 points" in monthly
    assert "a year earlier: 2025-04 5.1 -> 2026-04 5.2, +0.1 points" in monthly


def test_a_series_names_the_months_not_published_and_never_averages(release):
    out = A.series(release, "ur_monthly", {"gender": "female"})
    assert "values: 2025-04 5.5, 2025-05 6," in out
    assert "not published: 2025-07, 2025-08" in out and "2026-03" in out
    assert "average" not in out and "mean" not in out


def test_two_groups_over_time_come_as_two_lines_and_their_gap(release):
    """An analyst can ask for two groups' trends at once, like two lines on a
    chart. (The test release's women sit 0.6 points above its men every month.)"""
    out = A.series(release, "ur_monthly", by="gender")
    assert "female: 2025-04 5.5," in out and "male: 2025-04 4.9," in out and "person:" not in out
    assert "female minus male: 2025-04 +0.6," in out and "not published: 2025-07" in out
    line = A.trend(release, "ur_monthly", by="gender")
    assert "one straight line per gender" in line
    # the gap's direction from the line through every month, never from two months (a live
    # run); trend judges it, series only describes
    assert "female minus male from 2025-04 to 2026-04: +0.6 -> +0.6 points (+0.0), but compare the line below" in line
    gap_line = next(x for x in line.split("\n") if x.startswith("the gap's straight line"))
    assert gap_line.startswith("the gap's straight line: +0.00 points a month") and gap_line.endswith("no clear trend")
    assert "straight line" not in A.series(release, "ur_monthly", by="gender")
    with pytest.raises(Refusal, match="leave it out of where"):
        A.trend(release, "ur_monthly", where={"gender": "female"}, by="gender")


def test_a_gap_that_moves_along_its_line_is_called_widening_or_narrowing():
    from stats_desk.analysis.analyses import _gap_direction
    months = ["2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
    growing = dict(zip(months, [-2.0, -2.4, -2.9, -3.3, -3.8, -4.2]))            # a minus b, more negative
    assert _gap_direction("male", "female", growing, "month")[-1].endswith(": widening")
    shrinking = dict(zip(months, [3.0, 2.6, 2.1, 1.7, 1.2, 0.8]))
    assert _gap_direction("female", "male", shrinking, "month")[-1].endswith(": narrowing")


def test_other_analyses_give_one_answer_per_group_where_it_means_something(release):
    """Wherever meaningful, an analysis can
    take `by`. (compare, gap's `across` and change already did; average and relate now do.)"""
    avg = A.average(release, "ur_monthly", "2025-04", "2025-06", by="gender",
                    vs_start="2026-01", vs_end="2026-04")
    assert "female: 2025-04 to 2025-06 5.63" in avg and "male: 2025-04 to 2025-06 5.03" in avg
    assert "female minus male: 2025-04 to 2025-06 +0.6 points; 2026-01 to 2026-04 +0.6 points" in avg
    linked = A.relate(release, "ur_annual", "wpr_annual", by="gender")
    assert "by state x gender" in linked.split("\n")[1] and "gender = female |" not in linked.split("\n")[1]
    assert "\nfemale: " in linked and "\nmale: " in linked and "BUILT IN" in linked
    with pytest.raises(Refusal, match="relate compares states already"):
        A.relate(release, "ur_annual", by="state")


def test_total_names_any_fields_total_and_a_value_of_another_field_is_named_as_such(release):
    """Every field's total has its own name (person, rural + urban, All India, all). Span test: an
    analyst asked for education "All ages", an age group, where education's total was meant."""
    t = release.tables["ur_annual"]
    assert [t._check(f, "total") for f in ("gender", "sector", "state", "education")] == \
        ["person", "rural + urban", "All India", "all"]
    with pytest.raises(Refusal, match="'All ages' is a value of age, not of education; its total is 'all'"):
        A.gap(release, "ur_annual", "education", "7.graduate", "All ages")
    assert "7.graduate minus all = +8.1 points" in A.gap(release, "ur_annual", "education", "7.graduate", "total")
    with pytest.raises(Refusal, match="age has no total"):
        t._check("age", "total")


def test_several_groups_unpublished_for_a_setting_are_answered_on_the_nearest_published_one(release):
    """Span test A7: the task said "all ages"; education is published for 15 years and above only.
    One group was answered on the nearest age with a note; several groups at once came back
    "not published for these settings", with no reason, and the analyst reported no figures."""
    both = A.nearest(release, A.gap, {"table": "ur_annual", "field": "education", "b": "all",
                                      "a": ["7.graduate", "8.post graduate & above"], "where": {"age": "All ages"}})
    assert "7.graduate minus all = +8.1 points" in both and "note: these figures are for age 15 years and above, not All ages" in both
    # each group over two years, in one call (the model's own call, refused before)
    years = ["2024", "2025"]
    out = A.value(release, "ur_annual", [{"education": "7.graduate", "period": years},
                                         {"education": "8.post graduate & above", "period": years}])
    assert "\n7.graduate, 2024: 12.9%" in out and "\n8.post graduate & above, 2025: 10%" in out
    assert "total (everyone), 2024: 3.2%" in out and "total (everyone), 2025: 3.1%" in out
    # one group unpublished among others: its line says why
    mixed = A.value(release, "ur_annual", [{"education": "7.graduate"}, {"education": "all", "age": "15-29 years"},
                                           {"education": "7.graduate", "age": "15-29 years"}])
    assert "7.graduate, 15-29 years: not published (" in mixed and "not published for these settings" not in mixed


def test_groups_are_chosen_the_same_way_in_every_analysis(release):
    """`by` a field: every group and the total; a list in `where`: only those groups and the
    total. An eval lead asked for graduates and post-graduates against everyone in two years, and
    gap took two groups only."""
    # a subset, with the total, in the analyses that show groups side by side
    some = A.compare(release, "ur_annual", "religion", {"religion": ["Islam", "Hinduism"]})
    assert "total (all): 3.1" in some and "highest first: Islam 3.2, Hinduism 3" in some
    moved = A.change(release, "ur_annual", {"religion": ["Islam", "Hinduism"]})
    assert "Islam 3.2 -> 3.2 (+0.0); Hinduism 3.1 -> 3 (-0.1)" in moved and "total (all): 3.2 -> 3.1 (-0.1)" in moved
    one = A.series(release, "ur_monthly", {"gender": ["female"]})
    assert "\nfemale: 2025-04 5.5," in one and "\nmale:" not in one and "\ntotal (person): 2025-04 5.1," in one
    avg = A.average(release, "ur_monthly", "2025-04", "2025-06", {"gender": ["female", "male"]})
    assert "total (person): 2025-04 to 2025-06 5.23" in avg
    assert "female minus male: 2025-04 to 2025-06 +0.6 points" in avg      # the gap ignores the total
    lines = A.trend(release, "ur_monthly", by="gender").split("\n")
    rows = [x.strip().split(":")[0] for x in lines if x.startswith("  ")]
    assert rows[-1] == "total (person)"                                     # after the groups, not ranked
    linked = A.relate(release, "ur_annual", "wpr_annual", x_where={"gender": ["female"]})
    assert "\nfemale: 36 states" in linked and "\nmale:" not in linked and "\ntotal (person): 36 states" in linked
    every = A.value(release, "ur_annual", by="sector")
    assert "rural: 2.4%" in every and "urban: 4.8%" in every and "total (rural + urban): 3.1%" in every
    # gap: several groups against one, each with its own earlier gap; "*" for every group
    both = A.gap(release, "ur_annual", "education", ["8.post graduate & above", "7.graduate"], "all")
    assert "7.graduate 11.2, all 3.1; 7.graduate minus all = +8.1 points" in both
    assert both.count("previous year (2024)") == 2
    every = A.gap(release, "ur_annual", "gender", "*", "person")
    assert "female minus person" in every and "male minus person" in every
    with pytest.raises(Refusal, match="across takes one pair"):
        A.gap(release, "ur_annual", "gender", "*", "person", across="sector")
    # several groups each set on its own, in one call: `where` a list of settings
    rural_kerala, urban = {"state": "Kerala", "sector": "rural", "gender": "female"}, {"sector": "urban", "gender": "female"}
    both = A.value(release, "ur_annual", [rural_kerala, urban])
    assert "\nKerala, rural, female: 6.2%" in both and "\nurban, female: 6.4%" in both
    assert "total (female, the rest at their totals): 3.1%" in both
    assert "Kerala, rural, female minus urban, female = -0.2 points; they differ in sector, state: not like with like" in both
    # one not published: said on its line, with what is; the rest still answered
    one = A.value(release, "ur_annual", [rural_kerala, {"religion": "Islam", "education": "7.graduate"}])
    assert "Islam, 7.graduate: not published (education x religion is not published together" in one
    moved = A.change(release, "ur_annual", [rural_kerala, urban])
    assert "Kerala, rural, female 10.5 -> 6.2 (-4.3)" in moved and "the gap reversed" in moved
    lines = A.trend(release, "ur_monthly", [{"gender": "female"}, {"gender": "male"}])
    assert "one straight line per group" in lines and "total (everyone): " in lines and "female minus male from" in lines
    within = A.gap(release, "ur_annual", "gender", "female", "male", where=[{"sector": "rural"}, {"state": "Kerala"}])
    assert "within rural:" in within and "within Kerala:" in within and "female minus male = +3.3 points" in within
    with pytest.raises(Refusal, match="not both"):
        A.series(release, "ur_monthly", [{"gender": "female"}, {"sector": "urban"}], by="gender")
    with pytest.raises(Refusal, match="no lists inside \\(only value takes them\\)"):
        A.compare(release, "ur_annual", "religion", [{"gender": ["female", "male"]}, urban])
    # code's reruns carry the groups to the other period
    assert "Kerala, rural, female, 2024: 10.5%" in A.recheck(release, "value", {"table": "ur_annual", "where": [rural_kerala, urban]})
    # a list is for the field compared: one field, not periods
    with pytest.raises(Refusal, match="one field only"):
        A.compare(release, "ur_annual", "religion", {"gender": ["female", "male"]})
    with pytest.raises(Refusal, match="several periods"):
        A.compare(release, "ur_annual", "religion", {"period": ["2024", "2025"]})


def test_correlations_true_by_definition_say_so(release):
    part = A.relate(release, "ur_annual", x_where={"gender": "female"})            # y: everyone
    assert "BUILT IN, not a finding: female is part of person" in part
    wpr = A.relate(release, "ur_annual", "wpr_annual")
    assert "BUILT IN, not a finding: WPR is computed from LFPR and UR" in wpr
    real = A.relate(release, "ur_annual", x_where={"gender": "female"}, y_where={"gender": "male"})
    assert "BUILT IN" not in real and "not causes" in real
    assert re.search(r"\d+ states: ordinary correlation r = -?\d\.\d\d, rank correlation rho = ", real)


def test_drill_ranks_the_fields_to_split_next_and_says_what_cant_be(release):
    out = A.drill(release, "ur_annual", {"religion": "Islam"})
    widths = [float(w) for w in re.findall(r"range (\d+\.\d) points", out)]
    assert widths == sorted(widths, reverse=True) and widths
    assert "not published split with these settings: state" in out


def test_the_same_request_gets_the_same_id(release):
    first = A.compare(release, "ur_annual", "state", {"gender": "female"})
    again = A.compare(release, "ur_annual", "state", {"gender": "female"})
    other = A.compare(release, "ur_annual", "state", {"gender": "male"})
    assert first.split("\n")[0] == again.split("\n")[0] != other.split("\n")[0]
    assert re.fullmatch(r"result compare-[0-9a-f]{6}", first.split("\n")[0])


async def test_over_mcp_results_are_kept_and_every_ending_sorted(release_path, monkeypatch):
    async with connect(server=create(release_path)) as analyses:
        tools = {t.name: t for t in analyses.tools}
        assert set(tools) == {f.__name__ for f in A.ANALYSES}

        ok = await invoke(tools["value"], {"table": "ur_annual", "where": {"gender": "female"}})
        assert ok.kind == "ok" and ok.content.endswith("value: 3.1%")
        rid = ok.content.split("\n")[0].removeprefix("result ")
        assert analyses.results[rid] == ok.content

        refused = await invoke(tools["value"], {"table": "ur_annual", "where": {"state": "Atlantis"}})
        assert refused.kind == "refused" and "no state 'Atlantis'" in refused.as_seen

        # MCPServer would drop the misspelt argument and answer; our side refuses the call
        typo = await invoke(tools["value"], {"table": "ur_annual", "wher": {"gender": "female"}})
        assert typo.kind == "bad_arguments" and "wher" in typo.content

        monkeypatch.setattr(Table, "value", lambda self, s: 1 / 0)
        broken = await invoke(tools["value"], {"table": "ur_annual"})
        assert broken.kind == "broken" and "Division" not in broken.as_seen


async def test_a_hand_in_number_not_in_its_cited_results_is_caught(release_path):
    async with connect(server=create(release_path)) as analyses:
        tools = {t.name: t for t in analyses.tools}
        gap = await invoke(tools["gap"], {"table": "ur_annual", "field": "gender", "a": "female",
                                          "b": "male", "across": "sector", "where": {"age": "15-29 years"}})
        rid = gap.content.split("\n")[0].removeprefix("result ")
        good = f"Young urban women: 18.9% vs men 11.8%, a gap of 7.1 points ({rid}), against 0.4 in villages."
        assert analyses.unbacked(good, [rid]) == []
        assert analyses.unbacked("Women 18.9% vs men 11.8%, a gap of 7.0 points.", [rid]) == ["7.0"]
        with pytest.raises(Exception, match="no result value-000000"):
            analyses.unbacked("3.1%", ["value-000000"])


@pytest.mark.skipif(not RELEASE_PATH.exists(), reason="no release file built")
async def test_the_server_runs_as_a_child_process_on_the_real_release():
    async with connect() as analyses:
        tools = {t.name: t for t in analyses.tools}
        out = await invoke(tools["value"], {"table": "lfpr_annual"})
        assert out.kind == "ok" and out.content.endswith("value: 59.3%")


def test_a_recheck_reruns_the_analysis_in_the_other_period_and_on_the_other_measure(release):
    out = A.recheck(release, "value", {"table": "ur_annual"})
    assert "in 2024:\n" in out and "value: 3.2%" in out
    assert "on current weekly status (CWS):\n" in out and "value: 5.3%" in out
    # monthly: a year earlier; there is no other measure
    monthly = A.recheck(release, "value", {"table": "ur_monthly"})
    assert "in 2025-04:\n" in monthly and "value: 5.1%" in monthly and "CWS):" not in monthly
    # a combination not published on the other measure says so, not an error
    young = A.recheck(release, "gap", {"table": "ur_annual", "field": "gender", "a": "female", "b": "male",
                                       "where": {"age": "15-29 years", "sector": "urban"}})
    assert "in 2024:" in young


def test_a_series_gives_the_latest_streak_and_the_latest_against_its_own_history():
    from stats_desk.analysis.compute import against_history, streak
    rising = {"2025-04": 5.0, "2025-05": 5.1, "2025-06": 5.3, "2025-07": 5.6}
    assert streak(rising, monthly=True) == {"direction": "rose", "moves": 3, "from": "2025-04"}
    # a month not published breaks a streak: what happened across it is not known
    assert streak({"2026-01": 5.0, "2026-02": 5.1, "2026-04": 5.3}, monthly=True) is None
    assert streak({"2026-01": 5.0, "2026-02": 5.1, "2026-04": 5.3, "2026-05": 5.4, "2026-06": 5.6},
                  monthly=True) == {"direction": "rose", "moves": 2, "from": "2026-04"}
    assert against_history({"a": 6.0, "b": 4.0, "c": 5.0, "d": 5.5}) == "the highest since a (6)"
    assert against_history({"a": 6.0, "b": 7.0, "c": 5.0}) == "the lowest of the series"
    assert against_history({"a": 6.0, "b": 6.0}) is None


def test_an_average_is_over_months_of_one_figure_and_says_it_is_not_published(release):
    out = A.average(release, "ur_monthly", "2025-04", "2025-06", vs_start="2026-01", vs_end="2026-04")
    assert "average 2025-04 to 2025-06: 5.23 (3 periods)" in out                 # (5.1 + 5.6 + 5.0) / 3
    assert "average 2026-01 to 2026-04: 5.1 (3 periods; not published, left out: 2026-03)" in out
    assert "2025-04 to 2025-06 minus 2026-01 to 2026-04: +0.1 points" in out
    assert "not a published figure" in out
    with pytest.raises(Refusal, match="leave the period out of where"):
        A.average(release, "ur_monthly", "2025-04", "2025-06", where={"period": "2025-05"})
    with pytest.raises(Refusal, match="fewer than 2 published periods"):
        A.average(release, "ur_monthly", "2025-04", "2025-04")


def test_a_trend_is_judged_against_the_scatter_around_its_line_and_never_called_significant(release):
    from stats_desk.analysis.compute import trend_line
    flat = trend_line({"2025-01": 5.0, "2025-02": 5.4, "2025-03": 4.8, "2025-04": 5.3, "2025-05": 4.9})
    assert abs(flat["over"]) <= flat["scatter"]
    steep = trend_line({"2025-01": 5.0, "2025-02": 5.5, "2025-03": 6.1, "2025-04": 6.4, "2025-05": 7.0})
    assert steep["over"] > 2 * steep["scatter"]
    # a month not published keeps its place on the line: 2 months apart, not 1
    assert trend_line({"2026-01": 5.0, "2026-02": 5.1, "2026-04": 5.3})["per"] == pytest.approx(0.1, abs=0.01)
    out = A.trend(release, "ur_monthly")
    assert "points a month" in out and "not a test of statistical significance" in out
    with pytest.raises(Refusal, match="annual tables hold 2 years, so use change"):
        A.trend(release, "ur_annual")
