"""The daily release check, on a fake MoSPI: what counts as new for each dataset (its source's
`check`), what's fetched, and that a figure MoSPI revised replaces ours and is logged (the same
for every dataset: release/update.py)."""

import json
import sys
from pathlib import Path

import pytest
from test_mospi import mospi as fake_mospi

from stats_desk.datasets import DATASETS
from stats_desk.release.sources import cpi, plfs
from stats_desk.release.update import Raw, compare, download, fetch, newer

PLFS = DATASETS["plfs"]


def row(month, value, gender="person"):
    return {"year": "2026", "month": month, "gender": gender, "value": value, "unit": "%"}


def save(raw, indicator, freq, year, rows):
    names = {1: "LFPR", 2: "WPR", 3: "UR"}
    (raw / f"{names[indicator]}_f{freq}_{year.replace(' ', '')}.json").write_text(json.dumps(
        {"dataset": "PLFS", "indicator_code": indicator, "frequency_code": freq, "year": year, "rows": rows}))


@pytest.fixture
def raw(tmp_path):
    """What we keep: annual PLFS 2025 and 2024, monthly 2026 to September (two rows)."""
    folder = tmp_path / "plfs"
    folder.mkdir()
    for code in (1, 2, 3):
        for y in ("PLFS 2025", "PLFS 2024"):
            save(folder, code, 1, y, [{"year": y, "value": "1.0"}])
        save(folder, code, 3, "2026", [row("August", "5.0"), row("September", "5.0")])
    return folder


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
    m, fake = moSPI(["PLFS 2025", "PLFS 2024", "2023-24"], ["2026", "2025"], [row("August", "5.0"), row("September", "5.0")])
    assert await plfs.check(m, Raw.read(raw)) == []
    assert len(fake.calls) == 3


async def test_a_new_month_fetches_its_year_and_keeps_every_figure_we_had(raw):
    stored = Raw.read(raw)
    now = [row("August", "5.0"), row("September", "5.0"), row("October", "5.2")]
    m, _ = moSPI(["PLFS 2025", "PLFS 2024"], ["2026", "2025"], now)
    jobs = await plfs.check(m, stored)
    assert [j.name for j in jobs] == ["LFPR_f3_2026", "WPR_f3_2026", "UR_f3_2026"] and "new month" in jobs[0].why
    got = await fetch(m, PLFS, stored, jobs, pace=0, log=lambda *_: None)
    assert got.added == [f"{n} 2026 (monthly): 1 rows added (October)" for n in ("LFPR", "WPR", "UR")]
    assert got.revised == [] and not (raw.parent / "REVISIONS.md").exists()
    kept = json.loads((raw / "UR_f3_2026.json").read_text())
    # the file keeps PLFS's shape: MoSPI's name, each filter, the rows
    assert set(kept) == {"dataset", "indicator_code", "frequency_code", "year", "rows"} and len(kept["rows"]) == 3


async def test_a_new_year_is_seen_whatever_mospi_calls_it(raw):
    m, _ = moSPI(["PLFS 2026 (calendar)", "PLFS 2025", "PLFS 2024"], ["2026", "2025"], [row("August", "5.0"), row("September", "5.0")])
    jobs = await plfs.check(m, Raw.read(raw))
    assert {(j.filters["frequency_code"], j.filters["year"]) for j in jobs} == {(1, "PLFS 2026 (calendar)")}


async def test_a_figure_mospi_revised_replaces_ours_and_is_logged(raw):
    """MoSPI's figures are the official ones: take the revision, record it."""
    stored = Raw.read(raw)
    now = [row("August", "5.1"), row("September", "5.0"), row("October", "5.2")]   # August revised
    m, _ = moSPI(["PLFS 2025", "PLFS 2024"], ["2026", "2025"], now)
    got = await fetch(m, PLFS, stored, await plfs.check(m, stored), pace=0, log=lambda *_: None)
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
        await plfs.check(m, Raw.read(raw))


def test_rows_are_matched_on_everything_but_their_value():
    added, gone, changed = compare([row("Aug", "5.0"), row("Sep", "5.0")], [row("Aug", "5.0"), row("Oct", "4.9")])
    assert [r["month"] for r in added] == ["Oct"] and [r["month"] for r in gone] == ["Sep"] and changed == []


# ---- CPI: its own check, the same fetching

def cpi_mospi(rows_by_year):
    """A fake MoSPI for CPI: the 2024 base's All India rows of each year; nothing else published."""
    def answer(tool, args):
        f = args["filters"]
        rows = rows_by_year.get(f.get("year")) if f.get("state_code") == "1" else None
        if f.get("division_code") == "0":
            rows = [{"state": "Delhi", "month": "October", "value": "1"}]
        if rows:
            return {"data": rows, "meta_data": {"totalRecords": len(rows), "totalPages": 1}}
        return {"msg": "No Data Found", "statusCode": True}
    return fake_mospi(answer)


@pytest.fixture
def cpi_raw(tmp_path):
    folder = tmp_path / "cpi"
    stored = Raw(folder)
    stored.save(DATASETS["cpi"], cpi.all_india(2026), [{"month": "September", "value": "1"}])
    return folder


async def test_cpi_a_quiet_day_costs_two_small_requests(cpi_raw):
    m, fake = cpi_mospi({"2026": [{"month": "September", "value": "1"}]})
    assert await cpi.check(m, Raw.read(cpi_raw)) == []
    assert len(fake.calls) == 2


async def test_cpi_a_new_month_fetches_its_year_and_the_states_again(cpi_raw):
    m, _ = cpi_mospi({"2026": [{"month": "September", "value": "1"}, {"month": "October", "value": "2"}]})
    stored = Raw.read(cpi_raw)
    jobs = await cpi.check(m, stored)
    assert [j.name for j in jobs] == ["all_india_2026", "general_states"] and "a new month" in jobs[0].why
    got = await fetch(m, DATASETS["cpi"], stored, jobs, pace=0, log=lambda *_: None)
    assert "CPI 2024 base, All India, 2026: 1 rows added (October)" in got.added
    assert Raw.read(cpi_raw).filters["general_states"] == {"base_year": "2024", "series": "Current", "division_code": "0"}


async def test_cpi_a_new_year_is_seen(cpi_raw):
    m, _ = cpi_mospi({"2026": [{"month": "September", "value": "1"}], "2027": [{"month": "January", "value": "1"}]})
    assert [j.name for j in await cpi.check(m, Raw.read(cpi_raw))] == ["all_india_2027", "general_states"]


async def test_a_first_download_keeps_each_job_as_it_comes_and_skips_what_is_kept(cpi_raw):
    m, fake = cpi_mospi({"2026": [{"month": "September", "value": "1"}]})
    jobs = [cpi.all_india(2026), cpi.general_states()]
    assert await download(m, DATASETS["cpi"], Raw.read(cpi_raw), jobs, pace=0, log=lambda *_: None) == ["general_states"]
    assert len(fake.calls) == 1


def test_every_dataset_is_built_after_the_one_its_names_are_conformed_to():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    from release import in_build_order
    assert in_build_order(["cpi", "plfs"]) == ["plfs", "cpi"]
    assert in_build_order(["cpi"]) == ["cpi"]
