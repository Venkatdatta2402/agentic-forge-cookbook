"""PLFS's source: its downloaded rows (data/raw/plfs) as clean tables. Only what is PLFS's own is
here; storing them, their cards and the release line are every dataset's (release/store.py).

Cleaning, each step from a rule verified on the data (release/cells.py has the evidence):
  - readable column names (datasets/plfs.toml [source] columns): state, gender, sector, age,
    education, religion, social, measure, value
  - empty religion/social/education -> 'all' (empty and 'all' carried the same value 18 of 18)
  - empty age -> the band the empty-age tables cover, worked out from their totals
    (`unstated_age_band`)
  - the same total published in several tables (the 3.1% headline appeared three times) becomes
    one row; two copies that disagree stop the build
  - annual 'PLFS 2025' -> year 2025; monthly gets year, month and a period date
  - the monthly series' measure, which MoSPI doesn't say, worked out from the numbers
    (`infer_monthly_measure`)
"""

from datetime import date
from pathlib import Path

from ...analysis.data import MEASURES
from ...analysis.periods import MONTH_NAMES as MONTHS
from ...datasets import DATASETS
from ..cells import Cells, guard
from ..store import StoreError, Table
from ..update import Job, Raw, newer, year_order

PLFS = DATASETS["plfs"]
MCP_NAME = PLFS.source["mcp"]                       # the dataset's name at MoSPI
ANNUAL, MONTHLY = PLFS.source["frequencies"]["annual"], PLFS.source["frequencies"]["monthly"]
FREQUENCY = {ANNUAL: "annual", MONTHLY: "monthly"}
PROBE = PLFS.indicators[PLFS.source["probe"]]["code"]   # all indicators are released together
RENAME = PLFS.source["columns"]
ANNUAL_COLUMNS = PLFS.tables["annual"]["columns"]
MONTHLY_COLUMNS = PLFS.tables["monthly"]["columns"]
TIME_COLUMNS = ("year", "month", "period")
(MEASURE,), (BAND,) = PLFS.measures, PLFS.bands      # PLFS's own: one measure, one band (age)
# MoSPI's indicator codes -> names, in code order (datasets/plfs.toml)
_BY_CODE = sorted(PLFS.indicators.items(), key=lambda kv: kv[1]["code"])
NAMES = {i["code"]: i["name"] for _, i in _BY_CODE}
SHORT = {i["code"]: k.upper() for k, i in _BY_CODE}
COPY_TOLERANCE = 0.5     # points; beyond rounding, two copies disagreeing is a real mismatch


# ---- what is downloaded, and what is new at MoSPI (release/update.py fetches and keeps it)
#
# One download per (indicator, frequency, year): MoSPI serves a year as one table, so a new
# month means its year's monthly tables again, a new year its tables for the first time; the
# 1.9 MB annual tables are fetched once. Kept: the newest two annual years and every monthly one.


def job(code, freq, year, why="") -> Job:
    return Job(f"{SHORT[code]}_f{freq}_{year.replace(' ', '')}",
               {"indicator_code": code, "frequency_code": freq, "year": year},
               f"{SHORT[code]} {year} ({FREQUENCY[freq]})", why)


class Stored:
    """data/raw/plfs as PLFS's build reads it: the years of each frequency, newest first, and
    the rows of each (indicator, frequency, year)."""

    def __init__(self, raw: Raw):
        self.rows = {(f["indicator_code"], f["frequency_code"], f["year"]): raw.rows[name]
                     for name, f in raw.filters.items()}
        self.years = {freq: sorted({y for (_, fr, y) in self.rows if fr == freq}, key=year_order, reverse=True)
                      for freq in (ANNUAL, MONTHLY)}

    @classmethod
    def read(cls, folder: Path) -> "Stored":
        return cls(Raw.read(folder))


async def jobs(mospi) -> list[Job]:
    """A first download: the newest two annual years and the newest two monthly years MoSPI
    lists (its metadata, two small requests), for every indicator."""
    out = []
    for freq, keep in ((ANNUAL, 2), (MONTHLY, 2)):
        listed = await mospi.years(MCP_NAME, indicator_code=PROBE, frequency_code=freq)
        out += [job(code, freq, y) for y in listed[:keep] for code in SHORT]
    return out


async def check(mospi, raw: Raw) -> list[Job]:
    """What is new at MoSPI since data/raw/plfs, in three small requests whatever the answer:
      1. the annual years MoSPI lists (its metadata, newest first: "PLFS 2025", "PLFS 2024"...); a
         year listed above our newest is new, whatever it's called
      2. the monthly years it lists ("2026", "2025"): likewise
      3. how many rows our newest monthly year has now: more than we keep means a new month (the
         metadata's months are pooled across years, so they can't show it)
    Each is fetched for every indicator."""
    stored = Stored(raw)
    new = []
    for freq in (ANNUAL, MONTHLY):
        listed = await mospi.years(MCP_NAME, indicator_code=PROBE, frequency_code=freq)
        new += [(freq, y, f"new {FREQUENCY[freq]} year {y}") for y in newer(listed, stored.years[freq])]
    if stored.years[MONTHLY]:
        latest = stored.years[MONTHLY][0]
        have = len(stored.rows.get((PROBE, MONTHLY, latest), []))
        now = await mospi.count(MCP_NAME, {"indicator_code": PROBE, "frequency_code": MONTHLY, "year": latest})
        if now > have:
            new.append((MONTHLY, latest, f"monthly {latest}: {now - have} more rows than we keep (a new month)"))
    return [job(code, freq, y, why) for freq, y, why in new for code in SHORT]


def _year(text):
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    return int(digits[:4]) if len(digits) >= 4 else None


def cells(rows, pinned=None) -> Cells:
    return Cells(rows, PLFS, pinned)


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
            if not row[c] and None in PLFS.total_values.get(c, ()):         # an empty religion is "all"
                row[c] = PLFS.totals[c]
        row[BAND] = row[BAND] or unstated_age
        if row[BAND] is None:
            raise StoreError(f"rows without an {BAND}, and no band could be worked out for them")
        key = tuple(row[c] for c in ANNUAL_COLUMNS)
        stated = 0 if r.get(BAND) is None else 1          # a table that states its age wins
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
               **{c: r.get(c) for c in MONTHLY_COLUMNS if c not in TIME_COLUMNS}, MEASURE: measure}
        key = tuple(row[c] for c in MONTHLY_COLUMNS)
        if key in seen and abs(seen[key] - value) > 1e-9:
            raise StoreError(f"two published copies of {dict(zip(MONTHLY_COLUMNS, key))} disagree")
        seen[key] = value
    return [{**dict(zip(MONTHLY_COLUMNS, k)), "value": v} for k, v in seen.items()]


def unstated_age_band(cells_by_indicator: dict, tolerance=0.05):
    """Which age band do the tables that leave age empty cover? Decided from the numbers.

    For each indicator, the empty-age table's own total (every field at its total) is compared
    with the main table's figure for each age band; a band matches when the two are equal. One
    indicator can be ambiguous (UR 2025: 15+ and All ages are both 3.1%), so the answer is the
    band that matches in EVERY indicator -- or None, and the profile keeps saying "no age range".
    Measured 2026-11-09: LFPR 59.3 = 15+ (All ages 44.9, 15-59 63.3); WPR 57.4 = 15+.
    """
    candidates = None
    evidence = {}
    for name, c in cells_by_indicator.items():
        if BAND not in c.fields:
            continue
        bands = {b: c.get(**{BAND: b}) for b in c.levels(BAND)}
        unstated = None
        # the age-less table's own total: every field at its total, the band not stated
        if any(None in c.totals[f] for f in c.totals if f in c.fields):
            key = tuple((f, "TOTAL" if f in c.totals else None) for f in c.fields)
            values = c._index.get(key)
            if values:
                unstated = values[0]
        if unstated is None:
            continue
        matching = {b for b, v in bands.items() if v is not None and abs(v - unstated) <= tolerance}
        evidence[name] = {"unstated_total": unstated, "bands": bands, "matching": sorted(matching)}
        candidates = matching if candidates is None else candidates & matching
    band = next(iter(candidates)) if candidates and len(candidates) == 1 else None
    return band, evidence


def infer_monthly_measure(monthly_headlines: dict, annual_headlines: dict, year):
    """Which measure the monthly series uses -- MoSPI's API doesn't say, and its definition text
    for UR is a copy of the LFPR definition, so the text can't be trusted for this. The monthly
    headlines of `year` are compared with that year's annual figure on each measure; the nearer
    one wins. Returns (measure code, how it was inferred)."""
    values = [v for v in monthly_headlines.values() if v is not None]
    mean = sum(values) / len(values)          # only to locate the series; never reported as a rate
    nearest = min(annual_headlines, key=lambda m: abs(annual_headlines[m] - mean))
    return nearest, (f"the {len(values)} monthly headlines of {year} sit around {mean:.1f}%; the annual "
                     f"{year} figure is " + ", ".join(f"{v:g}% on {MEASURES.get(m, m)}"
                                                      for m, v in annual_headlines.items())
                     + "; the monthly series matches the nearer one (inferred from the numbers, "
                       "not stated by MoSPI)")


def tables(annual: dict, monthly: dict, unstated_age) -> list[Table]:
    """annual: {table: (title, raw rows of all years)}; monthly: {table: (title, raw rows, measure,
    how the measure was inferred)}. A download with a field the description doesn't map stops
    the build (cells.guard); the totals are checked by the store, as for every dataset."""
    out = []
    for name, (title, rows) in annual.items():
        guard(rows, cells(rows))
        cleaned, notes = clean_annual(rows, unstated_age)
        if unstated_age:
            missing_in = PLFS.dimensions[BAND]["missing_in"]
            notes.insert(0, f"{missing_in} rows give no {BAND} in {PLFS.publisher}'s table; stored as "
                            f"'{unstated_age}', the band their totals match")
        out.append(Table(name, title, ANNUAL_COLUMNS, cleaned, notes))
    for name, (title, rows, used, how) in monthly.items():
        guard(rows, cells(rows))
        out.append(Table(name, title, MONTHLY_COLUMNS, clean_monthly(rows, used), [f"{MEASURE} {used}: {how}"]))
    return out


def build(raw: Path):
    """PLFS's tables from data/raw/plfs, and what its profiles must say (the band the age-less
    tables cover, how the monthly measure was found). The years come from what data/raw holds (the
    daily job adds new ones): annual, the newest and the one before it; monthly, every year kept."""
    from . import Built
    stored = Stored.read(raw)
    year, previous = stored.years[ANNUAL][:2]
    monthly_years = tuple(reversed(stored.years[MONTHLY]))
    log = [f"annual {year} (and {previous}); monthly {', '.join(monthly_years)}"]
    annual_rows = {code: {y: stored.rows[(code, ANNUAL, y)] for y in (year, previous)} for code in SHORT}
    monthly_rows = {code: {y: stored.rows[(code, MONTHLY, y)] for y in monthly_years} for code in SHORT}

    default = PLFS.defaults[MEASURE]
    band, evidence = unstated_age_band({SHORT[c]: cells(r[year], pinned={MEASURE: default, "year": year})
                                        for c, r in annual_rows.items()})
    log.append(f"tables without an age cover: {band} ("
               + "; ".join(f"{k} {v['unstated_total']:g} matches {v['matching']}" for k, v in evidence.items()) + ")")
    monthly, notes = {}, {}
    for code, r in monthly_rows.items():
        # the monthly measure is inferred against an annual year both series have; MoSPI labels
        # an annual year "PLFS 2025", a monthly one "2025"
        label = f"{PLFS.source['mcp']} {{}}".format
        first = next(y for y in reversed(monthly_years) if label(y) in annual_rows[code])
        headlines = {m: cells(r[first], pinned={"year": first, "month": m}).get() for m in MONTHS}
        annual = {m: cells(annual_rows[code][label(first)], pinned={MEASURE: m, "year": label(first)}).get()
                  for m in MEASURES}
        measure, how = infer_monthly_measure(headlines, annual, first)
        name = f"{SHORT[code].lower()}_monthly"
        monthly[name] = (f"{NAMES[code]}, monthly", [row for y in monthly_years for row in r[y]], measure, how)
        notes[name] = f"{MEASURES[measure]} -- {how}"
    stored_tables = tables({f"{SHORT[c].lower()}_annual": (f"{NAMES[c]}, annual", r[year] + r[previous])
                            for c, r in annual_rows.items()}, monthly, band)
    return Built(stored_tables, notes=profile_notes(stored_tables, band, notes), log=log)


def profile_notes(stored_tables, unstated_age, measures=None) -> dict:
    """{table: [line]}: what PLFS's profiles must say that only its source knows. The tables whose
    rows give no age (education's) were read as the band their totals match; the monthly
    series' measure, which MoSPI doesn't state, was worked out from the numbers (`measures`:
    {table: how})."""
    missing_in = PLFS.dimensions[BAND]["missing_in"]
    notes = {}
    for t in stored_tables:
        if unstated_age and missing_in in t.columns:
            notes[t.name] = [f"{missing_in} figures: {PLFS.publisher}'s table gives no {BAND}; they match the "
                             f"{unstated_age} figures, so they are read as that band"]
    for name, how in (measures or {}).items():
        notes.setdefault(name, []).append(f"{MEASURE}: {how}")
    return notes
