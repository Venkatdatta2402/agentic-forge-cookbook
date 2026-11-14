"""The daily release check, on a fake MoSPI: what counts as new, what's fetched, and that a
figure MoSPI revised replaces ours and is logged."""

import json

import pytest
from test_mospi import mospi as fake_mospi

from stats_desk.release.update import Stored, check, compare, fetch, newer


def row(month, value, gender="person"):
    return {"year": "2026", "month": month, "gender": gender, "value": value, "unit": "%"}


def save(raw, indicator, freq, year, rows):
    names = {1: "LFPR", 2: "WPR", 3: "UR"}
    (raw / f"{names[indicator]}_f{freq}_{year.replace(' ', '')}.json").write_text(json.dumps(
        {"dataset": "PLFS", "indicator_code": indicator, "frequency_code": freq, "year": year, "rows": rows}))


@pytest.fixture
def raw(tmp_path):
    """What we keep: annual PLFS 2025 and 2024, monthly 2026 to September (two rows)."""
    for code in (1, 2, 3):
        for y in ("PLFS 2025", "PLFS 2024"):
            save(tmp_path, code, 1, y, [{"year": y, "value": "1.0"}])
        save(tmp_path, code, 3, "2026", [row("August", "5.0"), row("September", "5.0")])
    return tmp_path


def moSPI(annual_years, monthly_years, monthly_rows):
    """A fake MoSPI: its metadata lists these years; get_data serves the monthly 2026 rows."""
    def answer(tool, args):
        if tool == "get_metadata":
            years = annual_years if args["frequency_code"] == 1 else monthly_years
            return {"filter_values": {"data": {"year": [{"year": y} for y in years]}}}
        f = args["filters"]
        if f["frequency_code"] == 3 and f["year"] == "2026":
            return {"data": monthly_rows, "meta_data": {"totalRecords": len(monthly_rows), "totalPages": 1}}
        return {"msg": "No Data Found", "statusCode": True}
    return fake_mospi(answer)


def test_only_years_newer_than_ours_are_new():
    assert newer(["PLFS 2026", "PLFS 2025", "PLFS 2024", "2023-24"], ["PLFS 2025", "PLFS 2024"]) == ["PLFS 2026"]
    assert newer(["2026", "2025"], ["2026", "2025"]) == []


async def test_a_quiet_day_costs_three_small_requests_and_fetches_nothing(raw):
    stored = Stored.read(raw)
    m, fake = moSPI(["PLFS 2025", "PLFS 2024", "2023-24"], ["2026", "2025"], [row("August", "5.0"), row("September", "5.0")])
    assert await check(m, stored) == []
    assert len(fake.calls) == 3


async def test_a_new_month_fetches_its_year_and_keeps_every_figure_we_had(raw):
    stored = Stored.read(raw)
    now = [row("August", "5.0"), row("September", "5.0"), row("October", "5.2")]
    m, _ = moSPI(["PLFS 2025", "PLFS 2024"], ["2026", "2025"], now)
    jobs = await check(m, stored)
    assert [(j.frequency, j.year) for j in jobs] == [(3, "2026")] and "new month" in jobs[0].why
    got = await fetch(m, stored, jobs, pace=0)
    assert got.added == [f"{n} 2026 (monthly): 1 rows added (October)" for n in ("LFPR", "WPR", "UR")]
    assert got.revised == [] and not (raw.parent / "REVISIONS.md").exists()
    assert len(json.loads((raw / "UR_f3_2026.json").read_text())["rows"]) == 3


async def test_a_new_year_is_seen_whatever_mospi_calls_it(raw):
    stored = Stored.read(raw)
    m, _ = moSPI(["PLFS 2026 (calendar)", "PLFS 2025", "PLFS 2024"], ["2026", "2025"], [row("August", "5.0"), row("September", "5.0")])
    assert [(j.frequency, j.year) for j in await check(m, stored)] == [(1, "PLFS 2026 (calendar)")]


async def test_a_figure_mospi_revised_replaces_ours_and_is_logged(raw):
    """User, 2026-11-14: MoSPI's figures are the official ones -- take the revision, record it."""
    stored = Stored.read(raw)
    now = [row("August", "5.1"), row("September", "5.0"), row("October", "5.2")]   # August revised
    m, _ = moSPI(["PLFS 2025", "PLFS 2024"], ["2026", "2025"], now)
    got = await fetch(m, stored, await check(m, stored), pace=0)
    assert "UR 2026 (monthly): 1 figures revised, 0 rows dropped" in got.revised
    assert any("month=August" in d and "5.0 -> 5.1" in d for d in got.details)
    kept = json.loads((raw / "UR_f3_2026.json").read_text())["rows"]
    assert {r["month"]: r["value"] for r in kept} == {"August": "5.1", "September": "5.0", "October": "5.2"}
    log = (raw.parent / "REVISIONS.md").read_text(encoding="utf-8")
    assert "5.0 -> 5.1" in log and "UR 2026 (monthly)" in log


async def test_an_outage_is_an_error_never_nothing_new(raw):
    from stats_desk.release.mospi import MospiError
    m, _ = fake_mospi(lambda tool, args: ConnectionError("down"))
    with pytest.raises(MospiError):
        await check(m, Stored.read(raw))


def test_rows_are_matched_on_everything_but_their_value():
    added, gone, changed = compare([row("Aug", "5.0"), row("Sep", "5.0")], [row("Aug", "5.0"), row("Oct", "4.9")])
    assert [r["month"] for r in added] == ["Oct"] and [r["month"] for r in gone] == ["Sep"] and changed == []
