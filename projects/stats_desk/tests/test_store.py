"""The release file: cleaned tables, cards, profiles; built from a trimmed copy of real PLFS."""

import duckdb
import pytest

from conftest import REAL

from stats_desk.release.store import StoreError, add_profiles, build_release, clean_annual


def row(value, age="15 years and above", education=None, state="Bihar"):
    return {"year": "PLFS 2025", "weekly_status": "PS+SS", "state": state, "gender": "person",
            "sector": "rural", "AgeGroup": age, "religion": None, "socialGroup": None,
            "General_Education": education, "value": str(value)}


def test_copies_of_one_combination_become_one_row_and_the_stated_age_wins():
    # LFPR 2025, rural Bihar: 50.1 in the main table, 50.2 in the education table (age empty)
    rows, notes = clean_annual([row(50.2, age=None, education="all"), row(50.1)], "15 years and above")
    assert [r["value"] for r in rows] == [50.1]
    assert rows[0]["education"] == "all" and rows[0]["religion"] == "all"
    assert "0.1-0.1 points apart" in notes[0]


def test_copies_further_apart_than_rounding_stop_the_build():
    with pytest.raises(StoreError, match="differ by 0.90 points"):
        clean_annual([row(50.1), row(51.0, age=None, education="all")], "15 years and above")


def test_rows_without_an_age_and_no_band_worked_out_stop_the_build():
    with pytest.raises(StoreError, match="no band"):
        clean_annual([row(3.0, age=None, education="7.graduate")], None)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    path = tmp_path_factory.mktemp("release") / "plfs.duckdb"
    build_release(path, annual={"ur_annual": ("unemployment rate (UR), annual", REAL["2025"] + REAL["2024"])},
                  monthly={}, unstated_age="15 years and above")
    add_profiles(path, {"UR_annual": "PROFILE ..."})
    con = duckdb.connect(str(path), read_only=True)
    yield con
    con.close()


def test_stored_values_are_mospis_published_ones(release):
    q = "SELECT value FROM ur_annual WHERE year = ? AND measure = 'PS+SS' AND state = ? AND gender = ? " \
        "AND sector = ? AND age = ? AND education = ? AND religion = 'all' AND social = 'all'"
    assert release.execute(q, [2025, "All India", "person", "rural + urban", "15 years and above", "all"]).fetchone() == (3.1,)
    assert release.execute(q, [2025, "All India", "female", "urban", "15-29 years", "all"]).fetchone() == (18.9,)
    assert release.execute(q, [2025, "All India", "person", "rural + urban", "15 years and above", "7.graduate"]).fetchone() == (11.2,)
    assert release.execute(q, [2024, "All India", "person", "rural + urban", "15 years and above", "all"]).fetchone() == (3.2,)


def test_every_combination_is_stored_once(release):
    dupes = release.execute("SELECT count(*) FROM (SELECT year, measure, state, gender, sector, age, education, "
                            "religion, social FROM ur_annual GROUP BY ALL HAVING count(*) > 1)").fetchone()[0]
    assert dupes == 0


def test_the_card_says_what_a_table_holds_from_the_table_itself(release):
    card = release.execute("SELECT card FROM _cards WHERE name = 'ur_annual'").fetchone()[0]
    assert "gender: female, male, person   [total = 'person']" in card
    assert "measure: CWS, PS+SS   [no total: two different measures" in card
    assert "state x religion (NOT published)" in card
    assert "stored as '15 years and above', the band their totals match" in card
    assert "each total lies between its parts (2025, usual status): gender " in card
    assert release.execute("SELECT released FROM _release").fetchone() == ("annual 2025 (and 2024)",)
    assert release.execute("SELECT profile FROM _profiles WHERE name = 'UR_annual'").fetchone() == ("PROFILE ...",)


def test_the_release_file_opens_read_only(release):
    with pytest.raises(duckdb.Error):
        release.execute("DELETE FROM ur_annual")
