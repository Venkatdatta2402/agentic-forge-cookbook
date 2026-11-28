"""Each dataset's source: the one module with what only that dataset needs to become stored
tables. Every source has the same shape:

    build(raw: Path) -> Built      its downloaded rows (data/raw/<key>) as clean tables
                                   (release/store.Table), with what its profiles must say

and everything after that is every dataset's: storing the tables, their cards, the release line,
the check that shared names match (release/store.py), the profiles (release/build.py), the
whole build (release/builder.py). A source holds only its dataset's quirks: PLFS's copies of one
figure in two tables and its unstated ages; CPI's two bases and MoSPI's spellings.
"""

from dataclasses import dataclass, field
from importlib import import_module


@dataclass
class Built:
    tables: list                                       # release/store.Table, in the order stored
    # {table: [line]}: what only the source knows that a table's profile must say (PLFS: the band
    # its age-less rows were read as, how its monthly measure was found); none for CPI
    notes: dict = field(default_factory=dict)
    log: list = field(default_factory=list)            # what the build found, for whoever runs it


def source(key):
    """The source module of a dataset: release/sources/<key>.py."""
    return import_module(f"{__name__}.{key}")
