"""The daily release check: is anything newer at MoSPI than the raw tables we keep, and if so,
fetch just that -- never changing a figure we already published without a person looking.

The raw downloads live in the repository (data/raw/plfs, decided 2026-11-14), so a day with
nothing new costs MoSPI three small requests, measured that day:

  1. the annual years MoSPI lists (its metadata, newest first: "PLFS 2025", "PLFS 2024", ...);
     a year listed above our newest is new, whatever it's called
  2. the monthly years it lists ("2026", "2025"): likewise
  3. how many rows our newest monthly year has now: more than we keep means a new month (the
     metadata's months are pooled across years, so they can't show it)

Not published yet comes back as "No Data Found", which counts as nothing; an outage raises, and
the job fails rather than reading it as "nothing new".

What's new is fetched for all three indicators (LFPR, WPR, UR), one table per (indicator,
frequency, year): a new month means its year's monthly table again (MoSPI serves a year as one
table), a new year its tables for the first time. The 1.9 MB annual tables are fetched once.

**Revisions: MoSPI's figure wins, and it's logged** (user, 2026-11-14). MoSPI publishes no
revision policy we could find; re-downloading the 2025 and 2026 monthly unemployment tables on
2026-11-14 matched all 459 rows saved on 11-09 (nothing had been released in between, so that
proves little). A re-fetched table replaces ours whole -- MoSPI's figures are the official
ones -- and every figure it changed or dropped is written to data/raw/REVISIONS.md (date,
table, row, old -> new) and counted in the commit message. The release's fingerprint changes
with it (release/store.py), so a follow-up to a brief made before knows not to mix the two.
Revisions are seen when a table is fetched again: a year's monthly table with each new month;
the annual tables only once, when their year first appears.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

INDICATORS = {1: "LFPR", 2: "WPR", 3: "UR"}
PROBE = 3                    # the indicator asked about; all three are released together
ANNUAL, MONTHLY = 1, 3       # MoSPI's frequency codes


@dataclass
class Job:
    frequency: int
    year: str
    why: str


@dataclass
class Stored:
    """What data/raw holds: the years of each frequency, and the rows of each table."""
    raw: Path
    years: dict = field(default_factory=dict)       # frequency -> [year], newest first
    rows: dict = field(default_factory=dict)        # (indicator, frequency, year) -> rows

    @classmethod
    def read(cls, raw: Path) -> "Stored":
        stored = cls(raw)
        for path in sorted(raw.glob("*_f*_*.json")):
            body = json.loads(path.read_text(encoding="utf-8"))
            key = (body["indicator_code"], body["frequency_code"], body["year"])
            stored.rows[key] = body["rows"]
        for freq in (ANNUAL, MONTHLY):
            years = {y for (_, f, y) in stored.rows if f == freq}
            stored.years[freq] = sorted(years, key=year_order, reverse=True)
        return stored

    def path(self, indicator, frequency, year) -> Path:
        return self.raw / f"{INDICATORS[indicator]}_f{frequency}_{year.replace(' ', '')}.json"


def year_order(label: str):
    """"PLFS 2025" -> 2025, "2026" -> 2026, "2023-24" -> 2023: newest sorts last."""
    digits = "".join(c if c.isdigit() else " " for c in label).split()
    return int(digits[0]) if digits else 0


def newer(listed: list[str], ours: list[str]) -> list[str]:
    """Years MoSPI lists that are newer than every one we keep."""
    newest = max((year_order(y) for y in ours), default=0)
    return [y for y in listed if year_order(y) > newest and y not in ours]


async def check(mospi, stored: Stored) -> list[Job]:
    """What to fetch: three small requests to MoSPI, whatever the answer."""
    jobs = []
    for freq, label in ((ANNUAL, "annual"), (MONTHLY, "monthly")):
        listed = await mospi.years("PLFS", indicator_code=PROBE, frequency_code=freq)
        jobs += [Job(freq, y, f"new {label} year {y}") for y in newer(listed, stored.years[freq])]
    if stored.years[MONTHLY]:
        latest = stored.years[MONTHLY][0]
        have = len(stored.rows.get((PROBE, MONTHLY, latest), []))
        now = await mospi.count("PLFS", {"indicator_code": PROBE, "frequency_code": MONTHLY, "year": latest})
        if now > have:
            jobs.append(Job(MONTHLY, latest, f"monthly {latest}: {now - have} more rows than we keep (a new month)"))
    return jobs


def _key(row) -> str:
    return json.dumps({k: v for k, v in row.items() if k != "value"}, sort_keys=True)


def compare(old: list[dict], new: list[dict]) -> tuple[list, list, list]:
    """(rows added, rows gone, (row, old value, new value) changed), matching rows on every
    field but the value."""
    before, after = {_key(r): r for r in old}, {_key(r): r for r in new}
    added = [after[k] for k in after.keys() - before.keys()]
    gone = [before[k] for k in before.keys() - after.keys()]
    changed = [(after[k], before[k]["value"], after[k]["value"]) for k in before.keys() & after.keys()
               if before[k]["value"] != after[k]["value"]]
    return added, gone, changed


@dataclass
class Fetched:
    added: list = field(default_factory=list)      # "UR 2026 (monthly): 459 rows added (October)"
    revised: list = field(default_factory=list)    # "UR 2026 (monthly): 1 figures revised, 0 rows dropped"
    details: list = field(default_factory=list)    # each revised or dropped figure, old -> new


async def fetch(mospi, stored: Stored, jobs: list[Job], pace=0.4) -> Fetched:
    """Fetch each job's tables for all three indicators and write them: MoSPI's figures replace
    ours whole. What it added, and what it revised or dropped, is returned; revisions are also
    appended to REVISIONS.md beside the raw tables."""
    out, tables = Fetched(), {}
    for job in jobs:
        for code in INDICATORS:
            rows = await mospi.data("PLFS", {"indicator_code": code, "frequency_code": job.frequency, "year": job.year},
                                    max_rows=20_000, pace=pace)
            tables[(code, job.frequency, job.year)] = rows
            added, gone, changed = compare(stored.rows.get((code, job.frequency, job.year)) or [], rows)
            name = f"{INDICATORS[code]} {job.year} ({'annual' if job.frequency == ANNUAL else 'monthly'})"
            months = sorted({r.get("month") for r in added if r.get("month")})
            if added:
                out.added.append(f"{name}: {len(added)} rows added" + (f" ({', '.join(months)})" if months else ""))
            if gone or changed:
                out.revised.append(f"{name}: {len(changed)} figures revised, {len(gone)} rows dropped")
                out.details += [f"{name}: {_short(r)}: {a} -> {b}" for r, a, b in changed]
                out.details += [f"{name}: {_short(r)}: {r['value']} -> dropped" for r in gone]
    for (code, freq, year), rows in tables.items():          # only once every fetch has succeeded
        stored.path(code, freq, year).write_text(json.dumps(
            {"dataset": "PLFS", "indicator_code": code, "frequency_code": freq, "year": year, "rows": rows},
            ensure_ascii=False), encoding="utf-8")
    if out.details:
        log = stored.raw.parent / "REVISIONS.md"
        head = "" if log.exists() else "# Figures MoSPI revised or dropped after we had them\n\nOld -> new; MoSPI's figure replaced ours.\n"
        with log.open("a", encoding="utf-8") as f:
            f.write(head + f"\n## {date.today().isoformat()}\n\n" + "".join(f"- {d}\n" for d in out.details))
    return out


def _short(row) -> str:
    return ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("value", "unit", "frequency", "indicator"))
