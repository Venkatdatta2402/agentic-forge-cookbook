"""Build the release file from the downloaded PLFS tables. No model calls, no tokens.

    uv run python scripts/fetch_release_data.py      # the data (when MoSPI releases)
    uv run python scripts/build_release.py           # the release file

1. checks on the raw download: the guard, which age band the age-less tables cover, which
   measure the monthly series uses
2. the stored tables and their cards
3. the profiles, built from the stored tables with the analysis tools' own calculations

Writes data/release/plfs.duckdb, and data/profiles/plfs/*.md so the profiles can be read and
diffed in git.
"""

import sys
from pathlib import Path

from stats_desk.analysis.data import MEASURES, Release
from stats_desk.core.llm import count_tokens
from stats_desk.release.build import NAMES, SHORT, build_profiles, infer_monthly_measure
from stats_desk.release.cells import Cells, unstated_age_band
from stats_desk.release.store import MONTHS, add_profiles, build_release
from stats_desk.release.update import Stored

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "plfs"
RELEASE = ROOT / "data" / "release" / "plfs.duckdb"
TEXTS = ROOT / "data" / "profiles" / "plfs"


def main():
    # the years come from what data/raw holds (the daily job adds new ones), not from this file:
    # annual, the newest and the one before it; monthly, every year kept, oldest first
    stored = Stored.read(DATA)
    YEAR, PREVIOUS = stored.years[1][:2]
    MONTHLY_YEARS = tuple(reversed(stored.years[3]))
    print(f"annual {YEAR} (and {PREVIOUS}); monthly {', '.join(MONTHLY_YEARS)}")
    annual_rows = {code: {y: stored.rows[(code, 1, y)] for y in (YEAR, PREVIOUS)} for code in SHORT}
    monthly_rows = {code: {y: stored.rows[(code, 3, y)] for y in MONTHLY_YEARS} for code in SHORT}

    # 1. checks on the raw download
    band, evidence = unstated_age_band({SHORT[c]: Cells(r[YEAR], pinned={"status": "PS+SS", "year": YEAR})
                                        for c, r in annual_rows.items()})
    print(f"tables without an age cover: {band} ("
          + "; ".join(f"{k} {v['unstated_total']:g} matches {v['matching']}" for k, v in evidence.items()) + ")")
    monthly, notes = {}, {}
    for code, r in monthly_rows.items():
        # the monthly measure is inferred against an annual year both series have
        first = next(y for y in reversed(MONTHLY_YEARS) if f"PLFS {y}" in annual_rows[code])
        headlines = {m: Cells(r[first], pinned={"year": first, "month": m}).get() for m in MONTHS}
        annual = {m: Cells(annual_rows[code][f"PLFS {first}"], pinned={"status": m, "year": f"PLFS {first}"}).get()
                  for m in MEASURES}
        measure, how = infer_monthly_measure(headlines, annual, first)
        name = f"{SHORT[code].lower()}_monthly"
        monthly[name] = (f"{NAMES[code]}, monthly", [row for y in MONTHLY_YEARS for row in r[y]], measure, how)
        notes[name] = f"{MEASURES[measure]} -- {how}"

    # 2. the stored tables and their cards
    RELEASE.parent.mkdir(parents=True, exist_ok=True)
    build_release(RELEASE, annual={f"{SHORT[c].lower()}_annual": (f"{NAMES[c]}, annual", r[YEAR] + r[PREVIOUS])
                                   for c, r in annual_rows.items()},
                  monthly=monthly, unstated_age=band)

    # 3. the profiles, from the stored tables
    release = Release(RELEASE)
    texts = build_profiles(release, unstated_age=band, measure_notes=notes)
    cards = dict(release.cards)
    release.close()
    add_profiles(RELEASE, texts)

    TEXTS.mkdir(parents=True, exist_ok=True)
    for old in TEXTS.glob("*"):
        old.unlink()
    for name, text in texts.items():
        (TEXTS / f"{name}.md").write_text(text + "\n", encoding="utf-8")
        print(f"profile {name}: {count_tokens(text)} tokens")
    for name, text in cards.items():
        print(f"\n{text}\n[{count_tokens(text)} tokens]")
    print(f"\n{RELEASE} ({RELEASE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
