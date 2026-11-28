"""Time as data (analysis/periods.py): every frequency's periods written, stepped and told apart
the same way, so a quarterly dataset needs nothing new."""

import pytest

from stats_desk.analysis import periods as P


@pytest.mark.parametrize("f, period, next_, year_ago", [
    (P.MONTHLY, "2025-12", "2026-01", "2024-12"),
    (P.QUARTERLY, "2025-Q4", "2026-Q1", "2024-Q4"),
    (P.ANNUAL, "2025", "2026", None),
])
def test_each_frequency_steps_through_its_own_periods(f, period, next_, year_ago):
    assert P.of(period) is f
    assert f.next(period) == next_ and f.year_ago(period) == year_ago
    assert f.label(f.step(period)) == period
    assert f.step(next_) - f.step(period) == 1


def test_frequency_from_columns_and_from_words():
    assert P.of_columns(["year", "month", "period", "value"]) is P.MONTHLY
    assert P.of_columns(["year", "quarter", "period", "value"]) is P.QUARTERLY
    assert P.of_columns(["year", "state", "value"]) is P.ANNUAL
    assert P.of_words("inflation in October 2026") is P.MONTHLY
    assert P.of_words("GDP growth in Q3 of 2025") is P.QUARTERLY
    assert P.of_words("unemployment in 2025") is P.ANNUAL
    assert P.of_words("unemployment of women") is None


def test_missing_periods_and_a_period_as_asked():
    assert P.missing(["2025-11", "2026-02"]) == ["2025-12", "2026-01"]
    assert P.missing(["2025-Q3", "2026-Q2"]) == ["2025-Q4", "2026-Q1"]
    assert P.missing(["2022", "2025"]) == ["2023", "2024"]
    assert P.MONTHLY.normal("2025-10-01") == "2025-10" and P.ANNUAL.normal("2025-10") == "2025"
    assert P.QUARTERLY.normal("2025q3") == "2025-Q3"
    assert P.MONTHLY.part_value("Dec") == "12" and P.MONTHLY.part_value(3) == "03" and P.MONTHLY.part_value(13) is None
    assert P.QUARTERLY.part_value("Q2") == "2" and P.QUARTERLY.part_of("2025-Q2") == "2"
