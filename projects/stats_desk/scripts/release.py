"""The release job, the same for every dataset. No model calls, no tokens.

    uv run python scripts/release.py update  [--dataset plfs|cpi]   the daily job
    uv run python scripts/release.py check   [--dataset ...]        what is new at MoSPI; fetches nothing
    uv run python scripts/release.py fetch   [--dataset ...]        a first download: what isn't kept yet
    uv run python scripts/release.py build   [--dataset ...]        rebuild from what is kept

Every dataset unless one is named, in build order (a dataset whose names are conformed to
another's after it). Per dataset: its source (release/sources/<key>.py) says what it downloads
and what is new; the downloads are kept in data/raw/<key>/ (release/update.py); the build
(release/builder.py) writes data/release/<key>.duckdb and data/profiles/<key>/*.md.

`update`: ask MoSPI what is new (a few small requests per dataset), fetch just that, and rebuild
that dataset's release. Figures MoSPI revised are taken as MoSPI now gives them, listed here and
in data/raw/REVISIONS.md, and counted in the summary. Each dataset on its own: one that fails
(MoSPI unreachable, a release with a new shape: release/cells.py's guard, release/store.py's
checks) doesn't stop the others. Exit codes, for the workflow (.github/workflows/stats-desk-release.yml):
    0   every dataset checked: nothing new, or new figures fetched and the release rebuilt
    1   a dataset failed (the others' updates still stand, and are reported)
On GitHub Actions it also writes `changed=true|false` and a one-line `summary` to $GITHUB_OUTPUT.
"""

import argparse
import asyncio
import os
import sys
import traceback

from stats_desk import config
from stats_desk.core.llm import count_tokens
from stats_desk.datasets import DATASETS
from stats_desk.release import builder
from stats_desk.release.mospi import LiveTransport, Mospi
from stats_desk.release.sources import source
from stats_desk.release.update import Raw, download, fetch


def in_build_order(keys) -> list:
    """A dataset conformed to another is built after it (CPI's names are checked against PLFS's)."""
    out = []
    for key in keys:
        other = DATASETS[key].source.get("conform_to")
        if other in keys and other not in out:
            out.append(other)
        if key not in out:
            out.append(key)
    return out


def raw_of(key) -> Raw:
    return Raw.read(builder.ROOT / "data" / "raw" / key)


def output(**values):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            for k, v in values.items():
                f.write(f"{k}={v}\n")


def build(key, cards=True):
    report = builder.build(key)
    for line in report.log:
        print(line)
    print("stored: " + ", ".join(f"{t} {n:,}" for t, n in report.stored.items()))
    print(f"release: {report.released}")
    for name, text in report.profiles.items():
        print(f"profile {name}: {count_tokens(text)} tokens")
    if cards:
        for name, text in report.cards.items():
            print(f"\n{text}\n[{count_tokens(text)} tokens]")


async def with_mospi(work):
    transport = LiveTransport(config.MOSPI_URL)
    try:
        return await work(Mospi(transport))
    finally:
        await transport.close()


async def check(key, mospi):
    raw = raw_of(key)
    if not raw.rows:
        # never "nothing new": with nothing kept (data/raw/<key> not in the repository) the check
        # can't tell what is new, and the dataset would quietly never update
        raise RuntimeError(f"{key}: no downloads kept in {raw.folder}; run `release.py fetch --dataset {key}` "
                           "and commit them")
    print(f"{key}: {len(raw.rows)} downloads kept")
    jobs = await source(key).check(mospi, raw)
    for why in dict.fromkeys(j.why for j in jobs):
        print(f"{key}: new at MoSPI: {why}")
    if not jobs:
        print(f"{key}: nothing new at MoSPI")
    return raw, jobs


async def update(key, mospi):
    """(added, revised details) for one dataset, its release rebuilt if anything came."""
    raw, jobs = await check(key, mospi)
    if not jobs:
        return [], []
    got = await fetch(mospi, DATASETS[key], raw, jobs)
    if not (got.added or got.revised):
        print(f"{key}: fetched, but nothing new in the rows")
        return [], []
    print("\n".join(got.added))
    if got.revised:
        print("MoSPI revised figures we had; its new figures replace ours (logged in data/raw/REVISIONS.md):")
        print("\n".join(got.revised + [f"   {d}" for d in got.details]))
    print(f"--- rebuilding the {DATASETS[key].name} release")
    build(key, cards=False)
    return got.added, got.details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=["update", "check", "fetch", "build"])
    parser.add_argument("--dataset", choices=list(DATASETS))
    args = parser.parse_args(argv)
    keys = in_build_order([args.dataset] if args.dataset else list(DATASETS))

    if args.command == "build":
        for key in keys:
            build(key)
        return 0
    if args.command == "fetch":
        async def first(mospi):
            for key in keys:
                await download(mospi, DATASETS[key], raw_of(key), await source(key).jobs(mospi))
        asyncio.run(with_mospi(first))
        return 0
    if args.command == "check":
        async def look(mospi):
            for key in keys:
                await check(key, mospi)
        asyncio.run(with_mospi(look))
        return 0

    added, revised, failed = [], [], []

    async def daily(mospi):
        for key in keys:
            try:
                a, r = await update(key, mospi)
                added.extend(a)
                revised.extend(r)
            except Exception:  # noqa: BLE001 -- one dataset's failure mustn't stop the others
                traceback.print_exc()
                failed.append(key)
                print(f"{key}: FAILED; its previous release stays in use")

    asyncio.run(with_mospi(daily))
    if added or revised:
        summary = "; ".join(added) + (f"; {len(revised)} earlier figures revised by MoSPI" if revised else "")
        output(changed="true", summary=summary[:300])
    else:
        output(changed="false", summary="nothing new")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
