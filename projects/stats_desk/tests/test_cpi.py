"""CPI beside PLFS: its source (release/sources/cpi.py), its tables as the analyses read them, and
the two datasets' files read as one release. Built from a slice of the real downloads (conftest)."""

import pytest

from stats_desk.analysis import analyses as A
from stats_desk.analysis.data import Refusal
from stats_desk.datasets import DATASETS
from stats_desk.release.sources import cpi
from stats_desk.release.store import StoreError, write


def body(text):
    return text.split("\n", 1)[1]


def test_both_datasets_files_are_read_as_one_release(both):
    assert {"ur_annual", "ur_monthly", "inflation_monthly", "priceindex_monthly", "inflation2012_monthly"} <= set(both.tables)
    assert both.released.startswith("PLFS annual 2025") and "; CPI monthly to 2026-10 (2012 base to 2025-12)" in both.released
    assert "INFLATION_monthly_2026-10" in both.profiles and "UR_annual_2025" in both.profiles
    # an analysis on one dataset's table, and one relating the two across states
    assert A.value(both, "ur_annual").endswith("value: 3.1%")
    related = A.relate(both, "inflation2012_monthly", "ur_annual")
    assert "x: consumer price inflation (% on a year earlier), 2012 base" in related and "y: unemployment rate" in related
    # lined up as join lines them up: the states only one side has are named, not dropped unsaid
    assert "not in y (ur_annual): Dadra & Nagar Haveli; Daman & Diu" in related


def test_cpi_names_are_plfs_names_and_a_new_spelling_stops_the_build(both, releases_path, tmp_path):
    t = both.tables["inflation_monthly"]
    assert {"Delhi", "Jammu & Kashmir", "Andaman & Nicobar Islands"} <= set(t.values["state"])
    assert t.values["sector"] == ["rural", "rural + urban", "urban"]
    # the 2012 base keeps the two territories merged in 2020 apart: its own, kept as they are
    assert {"Dadra & Nagar Haveli", "Daman & Diu"} <= set(both.tables["inflation2012_monthly"].values["state"])
    # its group and subgroup are the 2024 base's division and group; "X-Overall" is the group's total
    assert "Vegetables" in both.tables["inflation2012_monthly"].values["group"]
    assert "CPI (General)" in both.tables["inflation2012_monthly"].values["division"]
    rows = {"2024": [{"year": "2026", "month": "October", "state": "Odissa", "sector": "Rural",
                      "division": "CPI (General)", "inflation": "1.0", "index": "100"}], "2012": []}
    with pytest.raises(StoreError, match=r"states not spelt as in PLFS: Odissa; add them to \[source.conform.state\]"):
        write(tmp_path / "cpi.duckdb", DATASETS["cpi"], cpi.tables(rows), reference=releases_path / "plfs.duckdb")
    assert not (tmp_path / "cpi.duckdb").exists()          # stopped before anything was written


def test_an_index_is_no_percentage_and_inflation_is(both):
    assert body(A.value(both, "priceindex_monthly")).split("\n")[-1] == "value: 110.05"
    assert body(A.value(both, "inflation_monthly")) == ("consumer price inflation (% on a year earlier), 2024 base, "
                                                        "monthly, 2026-10 | all items, All India\nvalue: 6.09%")


def test_an_item_named_alone_is_in_its_own_division_group_and_class(both):
    out = A.value(both, "inflation_monthly", {"item": "onion"})
    assert "division = Food and beverages, group = Food" in out and "item = Onion" in out
    assert out.endswith("value: 116.22%")


def test_a_level_compared_with_its_parents_left_out_is_each_in_its_own(both):
    """Every group, each in its own division, with the one overall total; and the groups of one
    division, with that division's total."""
    every = A.compare(both, "inflation_monthly", by="group")
    assert "total (all): 6.09" in every and "Food 8.94" in every and "Beverages 0.61" in every
    food = A.compare(both, "inflation_monthly", by="group", where={"division": "food"})
    assert "division = Food and beverages | by group" in food and "total (all): 8.51" in food


def test_what_is_published_together_says_a_hierarchy_once(both):
    t = both.tables["inflation_monthly"]
    assert t.combinations() == ("state: only by sector; division > group > class > sub_class > item "
                                "(each inside the one before): only by sector")
    with pytest.raises(Refusal, match="division x state is not published together"):
        A.value(both, "inflation_monthly", {"state": "Kerala", "division": "Food and beverages"})


def test_a_result_on_both_sides_of_the_change_of_base_says_so(both):
    """Before 2025 the 2024 base's tables hold MoSPI's back series: the 2012 base's figures,
    rescaled. A result naming periods on both sides gets the note; one inside 2026 doesn't."""
    across = A.value(both, "inflation_monthly", {"period": ["2024-10", "2026-10"]})
    assert A.across_join(both, {"table": "inflation_monthly"}, across).endswith(
        "Part of a change across 2025 can come from the change of list, not of prices")
    inside = A.value(both, "inflation_monthly", {"period": ["2026-09", "2026-10"]})
    assert A.across_join(both, {"table": "inflation_monthly"}, inside) == inside
    # the 2012 base's table has no join of its own
    old = A.value(both, "inflation2012_monthly", {"period": ["2025-11", "2025-12"]})
    assert A.across_join(both, {"table": "inflation2012_monthly"}, old) == old


def test_join_puts_two_datasets_side_by_side_by_what_they_share(both):
    """One row per group both tables have, each table's own period named, and said when they differ."""
    out = A.join(both, ["ur_annual", "inflation_monthly"], "state", where={"state": ["Kerala", "Bihar", "All India"]})
    lines = body(out).split("\n")
    assert lines[0].startswith("A: ur_annual: unemployment rate (UR), annual, 2025 |")
    assert lines[1].startswith("B: inflation_monthly: consumer price inflation (% on a year earlier), 2024 base, monthly, 2026-10 |")
    assert lines[2:6] == ["by state: A, B", "Bihar: A 3.8%, B 5.71%", "Kerala: A 4.4%, B 6.22%", "All India: A 3.1%, B 6.09%"]
    assert lines[-1] == "note: the tables are for different periods (A 2025, B 2026-10): side by side, not one moment"
    # every state: each column's lowest and highest written by code, as compare writes them
    every = body(A.join(both, ["ur_annual", "inflation_monthly"], "state")).split("\n")
    assert [line[:3] for line in every if " groups; " in line] == ["A: ", "B: "]
    # a field one table lacks: the fields they all have
    with pytest.raises(Refusal, match="inflation_monthly has no field 'gender'. Fields the tables all have: state, sector, period, year"):
        A.join(both, ["ur_annual", "inflation_monthly"], "gender")


def test_join_over_time_lines_up_like_periods_only(both):
    """Months with months, years with years; a year against a monthly table takes one month of
    each year, said; rows one table lacks are listed apart."""
    with pytest.raises(Refusal, match="ur_annual: years; inflation_monthly: months; no period is in both"):
        A.join(both, ["ur_annual", "inflation_monthly"], "period")
    with pytest.raises(Refusal, match="inflation_monthly is monthly: say which month of each year"):
        A.join(both, ["ur_annual", "inflation_monthly"], ["year", "state"])
    december = A.join(both, ["ur_annual", "inflation2012_monthly"], ["year", "state"],
                      where={"state": ["Kerala", "All India"]}, each_where=[{}, {"month": "December"}])
    assert "2012 base, to 2025, monthly, 2025-12 |" in december
    assert "2025, Kerala: A 4.4%, B 9.49%\n2025, All India: A 3.1%, B 1.33%" in december
    october = A.join(both, ["ur_annual", "inflation_monthly"], ["year", "state"],
                     where={"state": ["Kerala", "All India"]}, each_where=[{}, {"month": 10}])
    assert "2024, All India: A 3.2%, B 6.21%" in october
    assert october.endswith("not in B (inflation_monthly): 2024, Kerala; 2025")       # 2025 missing for every group: once
    # "month" for every month when every table is monthly (span test: written so twice, and refused)
    monthly = A.join(both, ["inflation_monthly", "priceindex_monthly"], ["month"], start="2026-09")
    assert "by period: A, B\n2026-09: A " in monthly
    # a period one table lacks: its stored runs, and the table with the same figure that has it
    with pytest.raises(Refusal, match=r"inflation_monthly has no period 2025-12. Stored: 2024-09, 2024-10, "
                                      r"2026-09, 2026-10; inflation2012_monthly \(the same figure on another base\) has 2025-12$"):
        A.value(both, "inflation_monthly", {"period": "2025-12"})
    # annual with annual, by year and a field
    annual = A.join(both, ["ur_annual", "wpr_annual"], ["period", "sector"])
    assert "2024, urban: A 5%, B 95%" in annual and "2025, rural + urban: A 3.1%, B 96.9%" in annual


def test_join_across_the_change_of_base_says_so(both):
    out = A.join(both, ["inflation_monthly", "priceindex_monthly"], "period")
    assert "2024-10: A 6.21%, B " in out and "not in A (inflation_monthly): 2025-09; 2025-10" in out
    assert A.across_join(both, {"tables": ["inflation_monthly", "priceindex_monthly"]}, out).endswith(
        "Part of a change across 2025 can come from the change of list, not of prices")
    from stats_desk.core.events import label
    assert (label("join", {"tables": ["ur_monthly", "inflation_monthly"], "by": ["period", "sector"]})
            == "join on ur_monthly + inflation_monthly by period, sector")


def test_cpi_profiles_say_the_join_and_keep_long_fields_short(both):
    from stats_desk.core.llm import count_tokens
    profile = both.profiles["INFLATION_monthly_2026-10"]
    assert profile.startswith("PROFILE consumer price inflation (% on a year earlier), 2024 base, monthly, latest 2026-10\n"
                              "headline (All India, all items): 6.09%")
    assert "note: across 2025, figures before 2025 are MoSPI's back series" in profile
    assert "age" not in profile.split("\n")[0]
    assert count_tokens(profile) < 1500


def test_a_claim_or_brief_on_both_sides_of_the_change_of_base_must_say_so(both):
    """Span test: an analyst wrote "6.09% versus 6.21%" for October 2026 against October 2024,
    the result's note on the change of base passed over."""
    from types import SimpleNamespace
    from stats_desk.agents.analyst import check_join
    from stats_desk.core.tools import ToolError
    text = A.across_join(both, {"table": "inflation_monthly"},
                         A.value(both, "inflation_monthly", {"period": ["2024-10", "2026-10"]}))
    analyses = SimpleNamespace(results={"value-1": text})
    with pytest.raises(ToolError, match="both sides of 2025, where the list of items and weights changed"):
        check_join(analyses, ["value-1"], "Inflation was 6.09% in October 2026 against 6.21% in October 2024.")
    check_join(analyses, ["value-1"], "Inflation was 6.09% against 6.21%; the 2024 figure is on the 2012 base, rescaled.")
    check_join(analyses, ["value-2"], "Inflation was 6.09% against 5.54%.")      # no note: nothing to say
    # any way of saying it (span test: refused under a narrower list of words)
    check_join(analyses, ["value-1"], "6.09% against 6.21%; the figures span a change of CPI list in 2025.")
