"""Fetch the finest published PLFS tables a profile is built from. No model calls, no tokens.

    uv run python scripts/fetch_release_data.py

Saves one JSON file of rows per (indicator, frequency, year) under data/raw/plfs/. This is the
"MoSPI released data" step: run when a new period appears. It pages through MoSPI at 100 rows
a page, two pages at a time (several hundred requests in all), so it is run rarely and cached.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from stats_desk import config
from stats_desk.datasets import DATASET
from stats_desk.release.mospi import LiveTransport, Mospi
from stats_desk.release.update import ANNUAL, INDICATORS, MCP_NAME, MONTHLY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / DATASET.key
# the years first fetched; after that, the daily release check adds new ones (update.py)
JOBS = ([(code, ANNUAL, year) for code in INDICATORS for year in (f"{MCP_NAME} 2025", f"{MCP_NAME} 2024")]
        + [(code, MONTHLY, year) for code in INDICATORS for year in ("2026", "2025")])


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    transport = LiveTransport(config.MOSPI_URL)
    mospi = Mospi(transport)
    try:
        for code, freq, year in JOBS:
            path = OUT / f"{INDICATORS[code]}_f{freq}_{year.replace(' ', '')}.json"
            if path.exists():
                print(f"have  {path.name}")
                continue
            started = time.monotonic()
            rows = await mospi.data(MCP_NAME, {"indicator_code": code, "frequency_code": freq,
                                               "year": year}, max_rows=20_000, pace=0.4)
            path.write_text(json.dumps({"dataset": MCP_NAME, "indicator_code": code,
                                        "frequency_code": freq, "year": year, "rows": rows},
                                       ensure_ascii=False), encoding="utf-8")
            print(f"saved {path.name}: {len(rows):,} rows in {time.monotonic() - started:.0f}s")
    finally:
        await transport.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
