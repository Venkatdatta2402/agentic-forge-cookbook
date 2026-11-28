"""The release file: one DuckDB file per dataset, with its stored tables, a card for each, the
profiles, and a line saying what it covers. The same for every dataset: what is particular to
one is its source module's (release/sources/), which hands this module clean tables.

Built by code from what the release job downloaded; opened read-only by the analysis MCP server.
Agents never call MoSPI, so every number they use comes from here, from one release, and the
profiles and the drill-downs always agree.

What every dataset's build gets here:
  - each table stored as its source cleaned it, the time columns typed (year, month, quarter:
    integers; period: a date)
  - every total checked to lie between its parts (`check_totals`): a row wrongly taken for a
    total stops the build
  - a card per table, written from the table itself: its values, totals, which fields are
    published together, the source's notes
  - names that must match another dataset's (its [source] conform_to): a value that dataset
    doesn't have stops the build, rather than become a second name for one state
  - a line saying what the release covers, and a fingerprint of every figure
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb

from ..analysis.periods import FREQUENCIES, TIME, of_columns
from .cells import Rows, totals_problems


class StoreError(Exception):
    """The downloaded rows can't be stored as they are; the build stops."""


@dataclass
class Table:
    """One table as a source hands it over: clean rows, one per published combination."""
    name: str                   # <indicator>_<frequency>: ur_annual, inflation_monthly
    title: str                  # what it holds: "unemployment rate (UR), annual"
    columns: list               # in order, time first; each row also has "value"
    rows: list
    notes: list = field(default_factory=list)       # for its card: what the source found or did


def _types(columns) -> dict:
    return {c: ("DATE" if c == "period" else "INTEGER" if c in TIME else "VARCHAR") for c in columns}


def _create(con, name, columns, rows):
    """Load rows through a temporary JSON-lines file. Measured: inserting 5,124 rows with
    parameters took ~60 s (row by row or 500 at a time alike); DuckDB reads the file at once."""
    spec = {**_types(columns), "value": "DOUBLE"}
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


def no_total(dataset, freq) -> dict:
    """Why a field has no total, for a card: the time columns' (a period is one of the table's
    frequency), then the dataset's own."""
    time = {"year": "one year per row", **{f.part: f"one {f.unit} per row" for f in FREQUENCIES if f.part},
            "period": f"one {freq.unit} per row"}
    return {**time, **dataset.no_total}


def card(con, name, title, source, dataset, notes=()):
    """What a table holds, written from the table itself, in its dataset's terms."""
    columns = [c for c, in con.execute(
        f"SELECT column_name FROM duckdb_columns() WHERE table_name = '{name}' ORDER BY column_index").fetchall()]
    totals, why = dataset.total_values, no_total(dataset, of_columns(columns))
    rows = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
    lines = [f"table {name}: {title}", f"source: {source}; {rows:,} rows, one per published combination"]
    for c in columns:
        if c == "value":
            continue
        values = [v for v, in con.execute(f'SELECT DISTINCT "{c}" FROM "{name}" ORDER BY 1').fetchall()]
        shown = ", ".join(map(str, values[:12])) + (f", ... ({len(values)} in all)" if len(values) > 12 else "")
        total = next((t for t in sorted(totals.get(c, set()) - {None}) if t in values), None)
        rule = (f"total = '{total}'" if total else f"no total: {why[c]}" if c in why else "")
        lines.append(f"  {c}: {shown}" + (f"   [{rule}]" if rule else ""))
    split = [c for c in columns if c in totals]
    if len(split) > 1:
        pairs = []
        for i, a in enumerate(split):
            for b in split[i + 1:]:
                ta, tb = sorted(totals[a] - {None})[0], sorted(totals[b] - {None})[0]
                n = con.execute(f'SELECT count(*) FROM "{name}" WHERE "{a}" <> ? AND "{b}" <> ?', [ta, tb]).fetchone()[0]
                pairs.append(f"{a} x {b}" + ("" if n else " (NOT published)"))
        lines.append("  split together: " + "; ".join(pairs))
    lines += [f"  note: {n}" for n in notes]
    lines.append(f"  value: {dataset.value_note}")
    return "\n".join(lines)


def check_conformed(dataset, tables, reference):
    """Every value of a field this dataset shares with the one it is conformed to (its [source]
    conform_to: CPI's states and sectors are PLFS's) is one that dataset's release names, or one
    of its own ([source.own_values]): a new spelling at MoSPI stops the build rather than become
    a second name for one state. `reference`: that dataset's release file, or None to skip."""
    other = dataset.source.get("conform_to")
    if not other or reference is None or not Path(reference).exists():
        return
    con = duckdb.connect(str(reference), read_only=True)
    try:
        theirs = {}
        for (name,) in con.execute("SELECT name FROM _cards").fetchall():
            cols = [c for c, in con.execute(
                "SELECT column_name FROM duckdb_columns() WHERE table_name = ?", [name]).fetchall()]
            for c in cols:
                if c in dataset.dimensions and c not in TIME and c != "value":
                    theirs.setdefault(c, set()).update(
                        v for v, in con.execute(f'SELECT DISTINCT "{c}" FROM "{name}"').fetchall())
    finally:
        con.close()
    own = dataset.source.get("own_values", {})
    for f, known in theirs.items():
        unknown = sorted({r[f] for t in tables if f in t.columns for r in t.rows} - known - set(own.get(f, {})))
        if unknown:
            plural = dataset.dimensions[f].get("plural", f"{f} values")
            raise StoreError(f"{plural} not spelt as in {other.upper()}: {', '.join(map(str, unknown))}; add them to "
                             f"[source.conform.{f}] in datasets/{dataset.key}.toml")


def check_totals(dataset, t: Table) -> str:
    """Every total of the table lies between its parts, in every group of the other fields (each
    period, each measure apart): a row wrongly taken for a total (a renamed total, a shifted
    column, a blank read as 'all') stops the build, beyond what rounding can do (the description's
    `rounding`: its indicator's, or its source's). Returns the line for the table's card."""
    indicator = dataset.indicators.get(t.name.rsplit("_", 1)[0], {})
    rounding = indicator.get("rounding", dataset.source["rounding"])
    checks, problems = totals_problems(Rows(t.rows, t.columns, dataset), rounding)
    if problems:
        raise StoreError(f"{t.name}: " + "; ".join(problems))
    checked = {f: c for f, c in checks.items() if c["groups"]}
    if not checked:
        return ""
    return "each total lies between its parts: " + ", ".join(
        f"{f} {c['inside']}/{c['groups']}" for f, c in checked.items())


def write(path, dataset, tables: list[Table], reference=None) -> dict:
    """A fresh release file for one dataset: its tables, their cards, what it covers. The
    profiles are built from it next and added with `add_profiles`. {table: rows stored}
    Every check runs before the old file is replaced, so a failed build leaves it in use."""
    check_conformed(dataset, tables, reference)
    checked = {t.name: check_totals(dataset, t) for t in tables}
    # a fresh file every build: DuckDB keeps the space of dropped tables, so rebuilding in place
    # grew the file from 2.9 to 5.3 MB
    Path(path).unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE _cards (name VARCHAR, card VARCHAR)")
        con.execute("CREATE TABLE _profiles (name VARCHAR, profile VARCHAR)")
        con.execute("CREATE TABLE _release (built DATE, released VARCHAR)")
        for t in tables:
            _create(con, t.name, t.columns, t.rows)
            source = dataset.tables[of_columns(t.columns).name]["source"]
            notes = [*t.notes, checked[t.name]] if checked[t.name] else t.notes
            con.execute("INSERT INTO _cards VALUES (?, ?)", [t.name, card(con, t.name, t.title, source, dataset, notes)])
        con.execute("INSERT INTO _release VALUES (?, ?)", [date.today(), released(con, dataset, [t.name for t in tables])])
    finally:
        con.close()
    return {t.name: len(t.rows) for t in tables}


def released(con, dataset, names) -> str:
    """What this release covers, from the tables themselves, and a fingerprint of every figure in
    them, e.g. "PLFS annual 2025 (and 2024); monthly to 2026-10 · data 3f2a91", "CPI monthly to
    2026-10 (2012 base to 2025-12) · data 1c1399": per frequency its latest period, and tables
    ending sooner (a base no longer published) named by their base. The fingerprint changes when
    MoSPI corrects a figure even though no new period came: a follow-up then knows the earlier
    brief's figures may differ from today's tables."""
    parts = []
    for f in sorted(FREQUENCIES, key=lambda f: f.per_year):
        mine = []
        for name in names:
            cols = [c for c, in con.execute(
                "SELECT column_name FROM duckdb_columns() WHERE table_name = ?", [name]).fetchall()]
            if of_columns(cols) is f:
                mine.append((name, [p for p, in con.execute(f'SELECT DISTINCT {f.sql()} FROM "{name}" ORDER BY 1').fetchall()]))
        if not mine:
            continue
        if not f.within_year:
            years = sorted({p for _, ps in mine for p in ps})
            parts.append(f"{f.name} {years[-1]}" + (f" (and {', '.join(years[:-1])})" if len(years) > 1 else ""))
            continue
        last = max(ps[-1] for _, ps in mine)
        sooner = {}
        for name, ps in mine:
            if ps[-1] < last:
                base = (dataset.indicators.get(name.rsplit("_", 1)[0]) or {}).get("base")
                sooner.setdefault((f"{base} base" if base else name, ps[-1]), None)
        parts.append(f"{f.name} to {last}" + "".join(f" ({what} to {end})" for what, end in sooner))
    digest = hashlib.sha256()
    for t in sorted(names):
        for row in con.execute(f'SELECT * FROM "{t}" ORDER BY ALL').fetchall():
            digest.update(repr(row).encode())
    return f"{dataset.name} " + "; ".join(parts) + f" · data {digest.hexdigest()[:6]}"


def add_profiles(path, profiles: dict):
    """{name: text}, built from the stored tables (release/build.py), into the release file."""
    con = duckdb.connect(str(path))
    try:
        con.execute("DELETE FROM _profiles")
        for name, text in profiles.items():
            con.execute("INSERT INTO _profiles VALUES (?, ?)", [name, text])
    finally:
        con.close()
