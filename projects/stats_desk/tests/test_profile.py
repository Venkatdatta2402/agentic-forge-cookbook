"""The release job's checks on a raw download, and the profiles built from the stored tables:
real PLFS numbers (a trimmed copy of the 2025 and 2024 tables, conftest.py) and small made-up
tables where a rule needs a case the real data doesn't have."""

import pytest
from conftest import REAL

from stats_desk.core.llm import count_tokens
from stats_desk.release.build import build_profiles, infer_monthly_measure, interactions, monthly_text
from stats_desk.release.cells import (Cells, ProfileShapeError, guard, unstated_age_band,
                                      verify_totals)
from stats_desk.release.stats import correlate, outliers, spearman


@pytest.fixture(scope="module")
def ur2025():
    return Cells(REAL["2025"], pinned={"status": "PS+SS", "year": "PLFS 2025"})


def test_raw_cells_read_published_values_with_other_fields_at_their_totals(ur2025):
    # every expected value is MoSPI's own published row, checked by hand on 2026-11-09
    assert ur2025.get() == 3.1
    assert ur2025.get(age="15-29 years") == 9.9
    assert ur2025.get(age="15-29 years", gender="female", sector="urban") == 18.9
    assert ur2025.get(state="Gujarat") == 0.9
    assert ur2025.get(religion="Sikhism") == 5.1
    assert ur2025.get(state="Atlantis") is None
    assert Cells(REAL["2025"], pinned={"status": "CWS", "year": "PLFS 2025"}).get() == 5.3


def test_totals_lie_between_their_parts_in_the_real_data(ur2025):
    checks = verify_totals(ur2025)
    assert all(c["holds"] for c in checks.values()), checks


def test_a_total_outside_its_parts_is_caught():
    rows = [{"gender": g, "state": "All India", "value": v} for g, v in
            (("male", "4"), ("female", "6"), ("person", "9"))]
    rows += [{"gender": g, "state": "Kerala", "value": v} for g, v in
             (("male", "4"), ("female", "6"), ("person", "5"))]
    checks = verify_totals(Cells(rows))
    assert checks["gender"] == {"groups": 2, "inside": 1, "holds": False}


def _table(values_by_age, education_total):
    rows = [{"AgeGroup": a, "General_Education": None, "gender": "person", "value": str(v)}
            for a, v in values_by_age.items()]
    rows += [{"AgeGroup": a, "General_Education": None, "gender": "male", "value": str(v - 1)}
             for a, v in values_by_age.items()]
    rows += [{"AgeGroup": None, "General_Education": "all", "gender": "person", "value": str(education_total)},
             {"AgeGroup": None, "General_Education": "7.graduate", "gender": "person", "value": "11.2"}]
    return Cells(rows)


def test_the_age_range_of_tables_without_one_is_worked_out_from_their_totals():
    # the real case: UR alone can't tell 15+ from All ages (both 3.1); LFPR can (59.3 vs 44.9)
    ur = _table({"15 years and above": 3.1, "All ages": 3.1, "15-59 years": 3.4}, 3.1)
    lfpr = _table({"15 years and above": 59.3, "All ages": 44.9, "15-59 years": 63.3}, 59.3)
    band, evidence = unstated_age_band({"UR": ur, "LFPR": lfpr})
    assert band == "15 years and above"
    assert evidence["UR"]["matching"] == ["15 years and above", "All ages"]
    assert unstated_age_band({"UR": ur})[0] is None              # ambiguous alone: stays unsaid


def test_a_release_with_a_new_shape_stops_the_build_instead_of_publishing():
    rows = [{"gender": g, "state": "All India", "value": v} for g, v in
            (("male", "4"), ("female", "6"), ("person", "5"))]
    assert guard(rows, Cells(rows))["gender"]["holds"]            # the known shape: fine
    with pytest.raises(ProfileShapeError, match="don't know: disability"):
        guard([{**r, "disability": "none"} for r in rows], Cells(rows))
    broken = [{**r, "value": "9"} if r["gender"] == "person" else r for r in rows]
    with pytest.raises(ProfileShapeError, match="total of gender is outside its parts"):
        guard(broken, Cells(broken))


def test_outliers_are_set_apart_and_named():
    values = {f"s{i}": v for i, v in enumerate([3, 3.2, 2.9, 3.1, 3.4, 2.8, 3.0, 12.2])}
    assert outliers(values) == ["s7"]


def test_ordinary_and_rank_correlation_disagree_when_one_point_drives_it():
    a = {f"s{i}": float(i) for i in range(12)}
    b = {f"s{i}": float(v) for i, v in enumerate([5, 2, 9, 0, 11, 7, 3, 10, 1, 8, 4, 6])}
    assert correlate(a, b)["agree"] is True                          # unrelated: both near 0.07
    a["island"], b["island"] = 60.0, 60.0                            # one extreme point
    c = correlate(a, b)
    assert c["pearson"] > 0.9 and c["spearman"] < 0.3                # 0.95 vs 0.27
    assert c["agree"] is False and c["driven_by"] == ["island"]


def test_spearman_handles_ties():
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_interactions_rank_without_outlier_states_and_name_them(release):
    t = release.table("ur_annual")
    by_state = next(i for i in interactions(t, t.settings())
                    if i["label"] == "women minus men" and i["across"] == "state")
    assert "Chandigarh" in by_state["spread"].outliers                # +12.2: set apart
    assert by_state["spread"].hi != "Chandigarh"


def test_a_profile_is_built_from_the_stored_tables_in_a_few_hundred_tokens(release):
    text = build_profiles(release, unstated_age="15 years and above")["UR_annual_2025"]
    assert "headline (All India, everyone): 3.1%" in text and "(CWS): 5.3%" in text
    assert "previous year (2024): 3.2%" in text
    assert "education = 7.graduate: 11.2%" in text
    assert "not findings (true by definition)" in text and "because the age bands overlap" in text
    assert "UR vs WPR r = -1.00, because WPR is computed from LFPR and UR" in text
    assert count_tokens(text) < 800


def test_a_monthly_profile_names_the_months_not_published(release):
    text = monthly_text(release.table("ur_monthly"))
    assert "not published: 2025-07, 2025-08" in text and "2026-03" in text
    assert "vs a year earlier (2025-04): 5.1% -> 5.2%" in text


def test_the_monthly_measure_is_inferred_from_the_numbers_not_assumed():
    measure, how = infer_monthly_measure({"April": 5.1, "May": 5.6, "June": None},
                                         {"PS+SS": 3.1, "CWS": 5.3}, "2025")
    assert measure == "CWS" and "not stated by MoSPI" in how


def test_the_monthly_profile_says_how_the_series_moves_with_the_trend_tools_own_verdict(release):
    """The profile had the series but not whether it was rising, so a lead could miss
    a steady rise unless the question asked. Same rule as the trend tool, so they never disagree."""
    from stats_desk.analysis.analyses import trend
    t = release.table("ur_monthly")
    text = monthly_text(t)
    tool = trend(release, "ur_monthly").split("\n")
    sits = next(i for i, line in enumerate(tool) if "usually sit" in line)
    verdict = tool[sits + 1].split(":")[0]               # the line after the scatter is the verdict
    assert f"trend {t.periods[0]} to {t.periods[-1]}: {verdict} (" in text
    assert "groups with a clear trend: " in text and "groups on a streak of 3+ months: " in text
    # 2026-03 isn't published: the latest streak can't run across it
    assert "latest streak" not in text
