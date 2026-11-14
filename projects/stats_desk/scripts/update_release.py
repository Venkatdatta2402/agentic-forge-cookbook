"""The daily release job: anything new at MoSPI? If so, fetch just that and rebuild the release.
No model calls, no tokens. Run by .github/workflows/stats-desk-release.yml, or by hand:

    uv run python scripts/update_release.py

Exit codes, for the workflow:
    0   nothing new, or new figures fetched and the release rebuilt (see "changed" below).
        Figures MoSPI revised are taken as MoSPI now gives them, listed here and in
        data/raw/REVISIONS.md, and counted in the summary (release/update.py)
    1   anything else (MoSPI unreachable, a release with a new shape: release/cells.py's guard)

On GitHub Actions it also writes `changed=true|false` and a one-line `summary` to $GITHUB_OUTPUT.
"""

import asyncio
import os
import runpy
import sys
from pathlib import Path

from stats_desk import config
from stats_desk.release.mospi import LiveTransport, Mospi
from stats_desk.release.update import Fetched, Stored, check, fetch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "plfs"


def output(**values):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            for k, v in values.items():
                f.write(f"{k}={v}\n")


async def update() -> Fetched | None:
    stored = Stored.read(RAW)
    print("kept: annual " + ", ".join(stored.years[1]) + "; monthly " + ", ".join(stored.years[3]))
    transport = LiveTransport(config.MOSPI_URL)
    try:
        mospi = Mospi(transport)
        jobs = await check(mospi, stored)
        if not jobs:
            return None
        for job in jobs:
            print(f"new at MoSPI: {job.why}")
        return await fetch(mospi, stored, jobs)
    finally:
        await transport.close()


def main():
    got = asyncio.run(update())
    if got is None or not (got.added or got.revised):
        print("nothing new at MoSPI")
        output(changed="false", summary="nothing new")
        return 0
    print("\n".join(got.added))
    if got.revised:
        print("MoSPI revised figures we had; its new figures replace ours (logged in data/raw/REVISIONS.md):")
        print("\n".join(got.revised + [f"   {d}" for d in got.details]))
    print("--- rebuilding the release")
    runpy.run_path(str(ROOT / "scripts" / "build_release.py"), run_name="__main__")
    summary = "; ".join(got.added) + (f"; {len(got.details)} earlier figures revised by MoSPI" if got.details else "")
    output(changed="true", summary=summary[:300])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
