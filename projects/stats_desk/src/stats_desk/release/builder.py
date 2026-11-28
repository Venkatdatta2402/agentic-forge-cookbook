"""One dataset's release, built the same way for every dataset. No model calls, no tokens.

  1. its source (release/sources/<key>.py) turns the downloaded rows into clean tables
  2. they are stored with their cards and the release line; names it shares with the dataset it
     is conformed to are checked against that dataset's release (release/store.py)
  3. the profiles, built from the stored tables with the analysis tools' own calculations
     (release/build.py), into the file and as text files to read and diff in git

    data/raw/<key>/          what the release job downloaded
    data/release/<key>.duckdb
    data/profiles/<key>/*.md
"""

from dataclasses import dataclass
from pathlib import Path

from ..analysis.data import Release
from ..datasets import DATASETS
from . import store
from .build import build_profiles
from .sources import source

ROOT = Path(__file__).resolve().parents[3]          # projects/stats_desk


@dataclass
class Report:
    stored: dict            # table -> rows
    released: str
    profiles: dict          # name -> text
    cards: dict             # table -> card
    log: list


def build(key, raw=None, release=None, profiles=None) -> Report:
    """Build dataset `key`'s release from its downloads. The paths default to data/raw/<key>,
    data/release/<key>.duckdb and data/profiles/<key>; the dataset it is conformed to is read
    from beside its release file."""
    dataset = DATASETS[key]
    raw = Path(raw or ROOT / "data" / "raw" / key)
    release = Path(release or ROOT / "data" / "release" / f"{key}.duckdb")
    profiles = Path(profiles or ROOT / "data" / "profiles" / key)
    built = source(key).build(raw)
    other = dataset.source.get("conform_to")
    reference = release.parent / f"{other}.duckdb" if other else None
    release.parent.mkdir(parents=True, exist_ok=True)
    stored = store.write(release, dataset, built.tables, reference)
    opened = Release(release)
    texts = build_profiles(opened, built.notes)
    released, cards = opened.released, dict(opened.cards)
    opened.close()
    store.add_profiles(release, texts)
    profiles.mkdir(parents=True, exist_ok=True)
    for old in profiles.glob("*"):
        old.unlink()
    for name, text in texts.items():
        (profiles / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    return Report(stored, released, texts, cards, built.log)
