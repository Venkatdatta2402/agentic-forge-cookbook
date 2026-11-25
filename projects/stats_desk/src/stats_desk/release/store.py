"""The release file: one DuckDB file with the stored tables, a card for each, and the profiles.

Built by code from what the release job downloaded; opened read-only by the analysis MCP server.
Agents never call MoSPI, so every number they use comes from here, from one release, and the
profiles and the drill-downs always agree.

Cleaning, each step from a rule verified on the data (cells.py has the evidence):
  - readable column names: state, gender, sector, age, education, religion, social, measure, value
  - empty religion/social/education -> 'all' (empty and 'all' carried the same value 18 of 18)
  - empty age -> the band the empty-age tables cover, worked out from their totals
  - the same total published in several tables (the 3.1% headline appeared three times) becomes
    one row; two copies that disagree stop the build
  - annual 'PLFS 2025' -> year 2025; monthly gets year, month and a period date
"""

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb

from ..datasets import DATASET
from .cells import RENAME, TOTALS, Cells, guard

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"], start=1)}
ANNUAL_COLUMNS = DATASET.tables["annual"]["columns"]
MONTHLY_COLUMNS = DATASET.tables["monthly"]["columns"]
NO_TOTAL = {"year": "one year per row", "month": "one month per row", "period": "one month per row",
            **DATASET.no_total}
TIME_COLUMNS = ("year", "month", "period")


class StoreError(Exception):
    """The downloaded rows can't be stored as they are; the build stops."""


def _year(text):
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    return int(digits[:4]) if len(digits) >= 4 else None


COPY_TOLERANCE = 0.5     # points; beyond rounding, two copies disagreeing is a real mismatch


def clean_annual(rows, unstated_age):
    """Raw annual rows of one indicator (any years, both measures) -> (unique clean rows, notes).

    A combination can be published in several tables. Measured on PLFS: 339 combinations appear
    twice per indicator-year; copies agree except a few between the main table and the
    education table (LFPR 2025: 4, by 0.1-0.3; WPR 2025: 2; none in UR or 2024). The table that
    states its age range wins over the education table (whose age is only inferred); each
    difference is reported in the card; a difference above COPY_TOLERANCE stops the build.
    """
    seen, rank, differ = {}, {}, []
    for raw in rows:
        r = {RENAME[k]: v for k, v in raw.items() if k in RENAME}
        try:
            value = float(raw["value"])
        except (TypeError, ValueError, KeyError):
            continue
        row = {"year": _year(r.get("year"))}
        for c in ANNUAL_COLUMNS:
            if c in TIME_COLUMNS:
                continue
            row[c] = r.get(c)
            if not row[c] and None in TOTALS.get(c, ()):         # an empty religion is "all"
                row[c] = DATASET.totals[c]
        band = DATASET.band
        row[band] = row[band] or unstated_age
        if row[band] is None:
            raise StoreError(f"rows without an {band}, and no band could be worked out for them")
        key = tuple(row[c] for c in ANNUAL_COLUMNS)
        stated = 0 if r.get(band) is None else 1          # a table that states its age wins
        if key in seen:
            gap = abs(seen[key] - value)
            if gap > COPY_TOLERANCE:
                raise StoreError(f"two published copies of {dict(zip(ANNUAL_COLUMNS, key))} differ by "
                                 f"{gap:.2f} points ({seen[key]} and {value})")
            if gap > 1e-9:
                differ.append(gap)
            if stated <= rank[key]:
                continue
        seen[key], rank[key] = value, stated
    notes = []
    if differ:
        notes.append(f"{len(differ)} combinations are published in both the main table and the "
                     f"education table with values {min(differ):.1f}-{max(differ):.1f} points apart; "
                     "the main table's value (age stated) is stored")
    return [{**dict(zip(ANNUAL_COLUMNS, k)), "value": v} for k, v in seen.items()], notes


def clean_monthly(rows, measure):
    seen = {}
    for raw in rows:
        try:
            value = float(raw["value"])
        except (TypeError, ValueError, KeyError):
            continue
        r = {RENAME[k]: v for k, v in raw.items() if k in RENAME}
        year, month = _year(r.get("year")), MONTHS.get(str(r.get("month")))
        row = {"year": year, "month": month, "period": date(year, month, 1),
               **{c: r.get(c) for c in MONTHLY_COLUMNS if c not in TIME_COLUMNS}, DATASET.measure: measure}
        key = tuple(row[c] for c in MONTHLY_COLUMNS)
        if key in seen and abs(seen[key] - value) > 1e-9:
            raise StoreError(f"two published copies of {dict(zip(MONTHLY_COLUMNS, key))} disagree")
        seen[key] = value
    return [{**dict(zip(MONTHLY_COLUMNS, k)), "value": v} for k, v in seen.items()]


def _create(con, name, columns, rows, types):
    """Load rows through a temporary JSON-lines file. Measured: inserting 5,124 rows with
    parameters took ~60 s (row by row or 500 at a time alike); DuckDB reads the file at once."""
    spec = {**{c: types[c] for c in columns}, "value": "DOUBLE"}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({c: (r[c].isoformat() if isinstance(r[c], date) else r[c])
                                for c in [*columns, "value"]}, ensure_ascii=False) + "\n")
        path = f.name
    try:
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM read_json(?, format = \'newline_delimited\', '
                    f'columns = {json.dumps(spec)})', [path])
    finally:
        os.unlink(path)


def card(con, name, title, source, notes=()):
    """What a table holds, written from the table itself."""
    columns = [c for c, in con.execute(
        f"SELECT column_name FROM duckdb_columns() WHERE table_name = '{name}' ORDER BY column_index").fetchall()]
    rows = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
    lines = [f"table {name}: {title}", f"source: {source}; {rows:,} rows, one per published combination"]
    for c in columns:
        if c == "value":
            continue
        values = [v for v, in con.execute(f'SELECT DISTINCT {c} FROM "{name}" ORDER BY 1').fetchall()]
        shown = ", ".join(map(str, values[:12])) + (f", ... ({len(values)} in all)" if len(values) > 12 else "")
        total = next((t for t in sorted(TOTALS.get(c, set()) - {None}) if t in values), None)
        rule = (f"total = '{total}'" if total else f"no total: {NO_TOTAL[c]}" if c in NO_TOTAL else "")
        lines.append(f"  {c}: {shown}" + (f"   [{rule}]" if rule else ""))
    split = [c for c in columns if c in TOTALS]
    if len(split) > 1:
        pairs = []
        for i, a in enumerate(split):
            for b in split[i + 1:]:
                ta, tb = sorted(TOTALS[a] - {None})[0], sorted(TOTALS[b] - {None})[0]
                n = con.execute(f'SELECT count(*) FROM "{name}" WHERE {a} <> ? AND {b} <> ?', [ta, tb]).fetchone()[0]
                pairs.append(f"{a} x {b}" + ("" if n else " (NOT published)"))
        lines.append("  split together: " + "; ".join(pairs))
    lines += [f"  note: {n}" for n in notes]
    lines.append(f"  value: {DATASET.value_note}")
    return "\n".join(lines)


def build_release(path, annual: dict, monthly: dict, unstated_age):
    """annual: {table: (title, raw rows of all years)}; monthly: {table: (title, raw rows, measure,
    how the measure was inferred)}. Writes a fresh DuckDB file with the tables and their cards;
    the profiles are built from it next and added with `add_profiles`."""
    # a fresh file every build: DuckDB keeps the space of dropped tables, so rebuilding in place
    # grew the file from 2.9 to 5.3 MB
    Path(path).unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE _cards (name VARCHAR, card VARCHAR)")
        con.execute("CREATE TABLE _profiles (name VARCHAR, profile VARCHAR)")
        con.execute("CREATE TABLE _release (built DATE, released VARCHAR)")
        types = defaultdict(lambda: "VARCHAR", year="INTEGER", month="INTEGER", period="DATE")
        measure, band = DATASET.measure, DATASET.band
        default = DATASET.defaults[measure]
        for name, (title, rows) in annual.items():
            years = sorted({r.get("year") for r in rows})
            for year in years:
                checks = guard([r for r in rows if r.get("year") == year],
                               Cells([r for r in rows if r.get("year") == year],
                                     pinned={measure: default, "year": year}))
            cleaned, notes = clean_annual(rows, unstated_age)
            _create(con, name, ANNUAL_COLUMNS, cleaned, types)
            if unstated_age:
                missing_in = DATASET.dimensions[band]["missing_in"]
                notes.insert(0, f"{missing_in} rows give no {band} in {DATASET.publisher}'s table; stored as "
                                f"'{unstated_age}', the band their totals match")
            short = DATASET.measure_names.get(default, default).split(" (")[0]
            notes.append(f"each total lies between its parts ({_year(years[-1])}, {short}): " + ", ".join(
                f"{f} {c['inside']}/{c['groups']}" for f, c in checks.items()))
            con.execute("INSERT INTO _cards VALUES (?, ?)", [name, card(con, name, title, DATASET.tables["annual"]["source"], notes)])
        for name, (title, rows, used, how) in monthly.items():
            _create(con, name, MONTHLY_COLUMNS, clean_monthly(rows, used), types)
            con.execute("INSERT INTO _cards VALUES (?, ?)", [name, card(
                con, name, title, DATASET.tables["monthly"]["source"], [f"{measure} {used}: {how}"])])
        con.execute("INSERT INTO _release VALUES (?, ?)", [date.today(), _released(con, annual, monthly)])
    finally:
        con.close()


def _released(con, annual, monthly):
    """What this release covers, from the tables themselves, and a fingerprint of every figure in
    them, e.g. "annual 2025 (and 2024); monthly to 2026-09 · data 3f2a91". The fingerprint changes
    when MoSPI corrects a figure even though no new period came: a follow-up then
    knows the earlier brief's figures may differ from today's tables."""
    parts = []
    if annual:
        years = sorted({y for t in annual for y, in con.execute(f'SELECT DISTINCT year FROM "{t}"').fetchall()})
        parts.append(f"annual {years[-1]}" + (f" (and {', '.join(map(str, years[:-1]))})" if len(years) > 1 else ""))
    if monthly:
        last = max(con.execute(f'SELECT max(period) FROM "{t}"').fetchone()[0] for t in monthly)
        parts.append(f"monthly to {last:%Y-%m}")
    digest = hashlib.sha256()
    for t in sorted([*annual, *monthly]):
        for row in con.execute(f'SELECT * FROM "{t}" ORDER BY ALL').fetchall():
            digest.update(repr(row).encode())
    return "; ".join(parts) + f" · data {digest.hexdigest()[:6]}"


def add_profiles(path, profiles: dict):
    """{name: text}, built from the stored tables (release/build.py), into the release file."""
    con = duckdb.connect(str(path))
    try:
        con.execute("DELETE FROM _profiles")
        for name, text in profiles.items():
            con.execute("INSERT INTO _profiles VALUES (?, ?)", [name, text])
    finally:
        con.close()
