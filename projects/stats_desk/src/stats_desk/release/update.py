"""The release job's MoSPI side, the same for every dataset: what a release is built from is a
list of downloads (`Job`), each one get_data request (all its pages) kept as one file in
data/raw/<key>/, MoSPI's rows as they came:

    data/raw/<key>/<job name>.json   {"dataset": <MoSPI's name>, <each filter>: ..., "rows": [...]}

What each dataset downloads, and what is new at MoSPI since, is its source's to say
(release/sources/<key>.py: `jobs`, `check`); fetching them, writing them and logging revisions
is here. A day with nothing new costs MoSPI a few small requests per dataset (PLFS three, CPI
two); "No Data Found" (not published yet) counts as nothing, and an outage raises: the job fails
rather than read it as "nothing new".

**Revisions: MoSPI's figure wins, and it's logged**. MoSPI publishes no revision policy we
could find; re-downloading the 2025 and 2026 monthly unemployment tables on 2026-11-14 matched
all 459 rows saved on 11-09 (nothing had been released in between, so that proves little). A
re-fetched table replaces ours whole -- MoSPI's figures are the official ones -- and every
figure it changed or dropped is written to data/raw/REVISIONS.md (date, table, row, old -> new)
and counted in the commit message. The release's fingerprint changes with it
(release/store.py), so a follow-up to a brief made before knows not to mix the two. Revisions
are seen when a table is fetched again: each source fetches again what a new period touches.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# a safety net, not a budget: CPI's largest download is ~30k rows; a wrong filter (no level of
# CPI's hierarchy can be asked for) would ask for millions
MAX_ROWS = 50_000


@dataclass
class Job:
    """One download: get_data with these filters, kept as data/raw/<key>/<name>.json."""
    name: str
    filters: dict
    label: str = ""             # how messages name it: "UR 2026 (monthly)"
    why: str = ""               # why it is fetched now: "new monthly year 2026"


@dataclass
class Raw:
    """What data/raw/<key> holds: each download's filters and rows, by its name."""
    folder: Path
    filters: dict = field(default_factory=dict)     # name -> {filter: value}
    rows: dict = field(default_factory=dict)        # name -> [row]

    @classmethod
    def read(cls, folder: Path) -> "Raw":
        raw = cls(Path(folder))
        for path in sorted(raw.folder.glob("*.json")):
            body = json.loads(path.read_text(encoding="utf-8"))
            raw.rows[path.stem] = body.pop("rows")
            body.pop("dataset", None)
            raw.filters[path.stem] = body
        return raw

    def save(self, dataset, job: Job, rows):
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / f"{job.name}.json").write_text(json.dumps(
            {"dataset": dataset.source["mcp"], **job.filters, "rows": rows}, ensure_ascii=False), encoding="utf-8")
        self.filters[job.name], self.rows[job.name] = dict(job.filters), rows


def year_order(label: str):
    """"PLFS 2025" -> 2025, "2026" -> 2026, "2023-24" -> 2023: newest sorts last."""
    digits = "".join(c if c.isdigit() else " " for c in label).split()
    return int(digits[0]) if digits else 0


def newer(listed: list[str], ours: list[str]) -> list[str]:
    """Years MoSPI lists that are newer than every one we keep."""
    newest = max((year_order(y) for y in ours), default=0)
    return [y for y in listed if year_order(y) > newest and y not in ours]


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


async def fetch(mospi, dataset, raw: Raw, jobs: list[Job], pace=0.4, log=print) -> Fetched:
    """Fetch each job and keep it: MoSPI's figures replace ours whole, written only once every
    job has been fetched. What it added, and what it revised or dropped, is returned; revisions
    are also appended to REVISIONS.md beside the datasets' raw folders."""
    out, got = Fetched(), {}
    for job in jobs:
        rows = await mospi.data(dataset.source["mcp"], job.filters, max_rows=MAX_ROWS, pace=pace)
        got[job.name] = (job, rows)
        log(f"fetched {job.name}: {len(rows):,} rows")
        added, gone, changed = compare(raw.rows.get(job.name) or [], rows)
        name = job.label or job.name
        months = sorted({r.get("month") for r in added if r.get("month")})
        if added:
            out.added.append(f"{name}: {len(added)} rows added" + (f" ({', '.join(months)})" if months else ""))
        if gone or changed:
            out.revised.append(f"{name}: {len(changed)} figures revised, {len(gone)} rows dropped")
            out.details += [f"{name}: {_short(r)}: {a} -> {b}" for r, a, b in changed]
            out.details += [f"{name}: {_short(r)}: {r['value']} -> dropped" for r in gone]
    for job, rows in got.values():
        raw.save(dataset, job, rows)
    if out.details:
        path = raw.folder.parent / "REVISIONS.md"
        head = "" if path.exists() else "# Figures MoSPI revised or dropped after we had them\n\nOld -> new; MoSPI's figure replaced ours.\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(head + f"\n## {date.today().isoformat()}\n\n" + "".join(f"- {d}\n" for d in out.details))
    return out


async def download(mospi, dataset, raw: Raw, jobs: list[Job], pace=0.4, log=print) -> list[str]:
    """A first download: each job not kept yet, kept as it comes (CPI's ~30 take a while; a
    failure keeps what came before it). The names fetched."""
    done = []
    for job in jobs:
        if job.name in raw.rows:
            log(f"have  {job.name}")
            continue
        rows = await mospi.data(dataset.source["mcp"], job.filters, max_rows=MAX_ROWS, pace=pace)
        raw.save(dataset, job, rows)
        done.append(job.name)
        log(f"saved {job.name}: {len(rows):,} rows")
    return done


def _short(row) -> str:
    return ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("value", "unit", "frequency", "indicator"))
