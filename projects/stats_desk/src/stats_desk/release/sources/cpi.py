"""CPI's source: MoSPI's rows (data/raw/cpi; what is downloaded and the daily check below) as clean tables, one
per base and indicator. Only what is CPI's own is here; storing them, their cards, the release
line and the check that its states and sectors are PLFS's are every dataset's (release/store.py).

  inflation_monthly        2024 base: the current series (2025 on; inflation from 2026), and
  priceindex_monthly       before 2025 MoSPI's back series (2013-2024, general index, All India):
                           the 2012 base's figures rescaled to 2024 = 100, so one line runs
                           across the change of base; a result on both sides says so
  inflation2012_monthly    2012 base, to Dec 2025: the general index by state, and every group
  priceindex2012_monthly   and subgroup All India

Cleaning, each from what the rows showed (2026-11-26):
  - names conformed with PLFS's (datasets/cpi.toml): four states spelt otherwise, sectors
    Rural/Urban/Combined -> rural/urban/rural + urban
  - the 2024 base's hierarchy as five fields, an empty level "all" (each name has one parent)
  - the 2012 base's group as division and its subgroup as group, "X-Overall" (the group's own
    total) as "all", its "General" as "CPI (General)": the same fields as the 2024 base's, its
    own values; the two bases are never in one table
  - a row without the figure (inflation before a base's second year; rural housing in the 2012
    base) is not stored; the same row published twice (All India in two downloads) becomes one,
    and two copies that disagree stop the build
"""

from datetime import date
from pathlib import Path

from ...analysis.periods import MONTH_NAMES as MONTHS
from ...datasets import CATALOG, DATASETS
from ..store import StoreError, Table
from ..update import Job, Raw

CPI = DATASETS["cpi"]
CONFORM = CPI.source["conform"]
HIERARCHY = ["division", "group", "class", "sub_class", "item"]
COLUMNS_2024 = CPI.tables["monthly"]["columns"]
COLUMNS_2012 = [c for c in COLUMNS_2024 if c not in ("class", "sub_class", "item")]
TABLES = {"inflation_monthly": ("inflation", "2024"), "priceindex_monthly": ("priceindex", "2024"),
          "inflation2012_monthly": ("inflation2012", "2012"), "priceindex2012_monthly": ("priceindex2012", "2012")}
FIGURE = {"inflation": "inflation", "priceindex": "index", "inflation2012": "inflation", "priceindex2012": "index"}


def conform(field, value):
    return CONFORM.get(field, {}).get(value, value)


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _row_2024(raw):
    year, month = int(raw["year"]), MONTHS[raw["month"]]
    row = {"year": year, "month": month, "period": date(year, month, 1),
           "state": conform("state", raw["state"]), "sector": conform("sector", raw["sector"])}
    for f in HIERARCHY:
        row[f] = raw.get(f) or CATALOG.totals[f]
    return row


def _row_2012(raw):
    year, month = int(raw["year"]), MONTHS[raw["month"]]
    sub = raw.get("subgroup") or ""
    return {"year": year, "month": month, "period": date(year, month, 1),
            "state": conform("state", raw["state"]), "sector": conform("sector", raw["sector"]),
            "division": conform("division", raw["group"]),
            "group": CATALOG.totals["group"] if not sub or sub.endswith("-Overall") else sub}


def clean(raw_rows, base) -> dict:
    """{indicator: [clean rows]} for one base's raw rows: each figure its own table's value."""
    make, columns = (_row_2024, COLUMNS_2024) if base == "2024" else (_row_2012, COLUMNS_2012)
    out = {}
    for indicator, b in TABLES.values():
        if b != base:
            continue
        seen = {}
        for raw in raw_rows:
            value = _number(raw.get(FIGURE[indicator]))
            if value is None:
                continue
            row = make(raw)
            key = tuple(row[c] for c in columns)
            if key in seen and abs(seen[key] - value) > 1e-9:
                raise StoreError(f"two published copies of {dict(zip(columns, key))} disagree "
                                 f"({seen[key]} and {value})")
            seen[key] = value
        out[indicator] = [{**dict(zip(columns, k)), "value": v} for k, v in seen.items()]
    return out


def read_raw(folder: Path) -> dict:
    """{base: raw rows} from the downloads: the 2024 base's current and back series together
    (MoSPI's back series: the 2012 base's general index, All India, rescaled to 2024 = 100)."""
    raw = Raw.read(folder)
    rows = {"2024": [], "2012": []}
    for name in raw.rows:
        rows[raw.filters[name]["base_year"]] += raw.rows[name]
    return rows


# ---- what is downloaded, and what is new at MoSPI (release/update.py fetches and keeps it)
#
# MoSPI's get_data has no way to ask for one level of the item hierarchy, so a division asked
# for in every state returns every item in every state (all of the 2024 base: 1.5 million rows;
# probed 2026-11-26). Downloaded instead:
#   2024 base (the current series, which MoSPI updates monthly; Jan 2025 on, inflation from 2026)
#     all_india_<year>        every level of the hierarchy, All India (state code 1, not the 99
#                             MoSPI's own parameter help says; 99 returns nothing)
#     general_states          the general index in every state, every month
#     back_general            MoSPI's back series onto the 2024 basket: general index, All India
#   2012 base (ended Dec 2025: fetched once; its own groups, never put with the 2024 base's)
#     b2012_states_<year>     the general index in every state
#     b2012_all_india_<year>  every group and subgroup, All India (code 99 here)

B2024, BACK = {"base_year": "2024", "series": "Current"}, {"base_year": "2024", "series": "Back"}
B2012 = {"base_year": "2012", "series": "Current"}
FIRST_2024, YEARS_2012 = 2025, range(2013, 2026)


def all_india(year, why="") -> Job:
    return Job(f"all_india_{year}", {**B2024, "state_code": "1", "year": str(year)},
               f"CPI 2024 base, All India, {year}", why)


def general_states(why="") -> Job:
    return Job("general_states", {**B2024, "division_code": "0"}, "CPI 2024 base, general index by state", why)


async def jobs(mospi) -> list[Job]:
    """A first download: the 2024 base's years from 2025 to the newest MoSPI has (one small
    request a year, and one for the year after), its states and back series, and the 2012 base."""
    years, year = [], FIRST_2024
    while await mospi.count(CPI.source["mcp"], all_india(year).filters):
        years.append(year)
        year += 1
    return ([all_india(y) for y in years] + [general_states(),
             Job("back_general", {**BACK, "division_code": "0", "state_code": "1"}, "CPI 2024 base, back series")]
            + [Job(f"b2012_states_{y}", {**B2012, "group_code": "0", "year": str(y)}, f"CPI 2012 base, by state, {y}")
               for y in YEARS_2012]
            + [Job(f"b2012_all_india_{y}", {**B2012, "state_code": "99", "year": str(y)}, f"CPI 2012 base, All India, {y}")
               for y in YEARS_2012])


async def check(mospi, raw: Raw) -> list[Job]:
    """What is new at MoSPI since data/raw/cpi, in two small requests whatever the answer: how
    many All India rows our newest 2024-base year has now (more than we keep: a new month), and
    whether the year after it has any. Either fetches that year's All India rows again, and the
    states' general index (one download of every month). The 2012 base and the back series
    no longer change."""
    years = sorted(int(n.rsplit("_", 1)[1]) for n in raw.rows if n.startswith("all_india_"))
    if not years:
        return []
    latest, mcp = years[-1], CPI.source["mcp"]
    new = []
    have, now = len(raw.rows[f"all_india_{latest}"]), await mospi.count(mcp, all_india(latest).filters)
    if now > have:
        new.append(all_india(latest, f"{latest}: {now - have} more All India rows than we keep (a new month)"))
    if await mospi.count(mcp, all_india(latest + 1).filters):
        new.append(all_india(latest + 1, f"new year {latest + 1}"))
    if new:
        new.append(general_states("the states' general index, every month: fetched with each new month"))
    return new


def tables(raw: dict) -> list[Table]:
    """{base: raw rows} -> the four tables, each with its card's notes."""
    cleaned = {}
    for base, rows in raw.items():
        cleaned.update(clean(rows, base))
    years = {ind: {r["year"] for r in rows} for ind, rows in cleaned.items()}
    out = []
    for name, (indicator, base) in TABLES.items():
        notes = [f"{base} base: MoSPI's list of items and weights of that base"]
        join = CPI.indicators[indicator].get("join")
        if join:
            notes.append(f"across {join['at'][:4]}: {join['note']}")
        # years with an index but no inflation, past the base's first (which has no year before
        # it to compare with): the 2024 base publishes 2025's index only
        index = next(i for i, b in TABLES.values() if b == base and FIGURE[i] == "index")
        if FIGURE[indicator] == "inflation" and years[index]:
            gone = sorted(years[index] - years[indicator] - {min(years[index])})
            if gone:
                notes.append(f"{', '.join(map(str, gone))} has an index but no inflation published on this base")
        columns = COLUMNS_2024 if base == "2024" else COLUMNS_2012
        out.append(Table(name, CPI.indicators[indicator]["name"] + ", monthly", columns, cleaned[indicator], notes))
    return out


def build(raw: Path):
    from . import Built
    return Built(tables(read_raw(raw)))
