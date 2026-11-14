import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ["STATS_DESK_LANGFUSE"] = "off"       # the project's .env has real keys: tests send nothing

from stats_desk.analysis.data import Release  # noqa: E402
from stats_desk.config import Limits  # noqa: E402
from stats_desk.core import llm  # noqa: E402
from stats_desk.core.governor import Governor  # noqa: E402
from stats_desk.release.build import build_profiles  # noqa: E402
from stats_desk.release.store import add_profiles, build_release  # noqa: E402

# real PLFS unemployment rows: a trimmed copy of the 2025 and 2024 annual tables
REAL = json.loads((Path(__file__).parent / "fixtures" / "plfs_ur_annual_trimmed.json").read_text(encoding="utf-8"))
# a small made-up monthly table: a gap in 2025 (July-December) and 2026 (March), like MoSPI's
MONTHLY = {("2025", "April"): 5.1, ("2025", "May"): 5.6, ("2025", "June"): 5.0,
           ("2026", "January"): 5.0, ("2026", "February"): 5.1, ("2026", "April"): 5.2}


def monthly_rows():
    return [{"year": y, "month": m, "AgeGroup": "15 years and above", "gender": g, "sector": "rural + urban",
             "value": str(round(v + d, 1))} for (y, m), v in MONTHLY.items()
            for g, d in (("person", 0), ("female", 0.4), ("male", -0.2))]


@pytest.fixture(scope="session")
def release_path(tmp_path_factory):
    """A release file built from the real rows, a stand-in second indicator (for the WPR rule)
    and the made-up monthly table."""
    path = tmp_path_factory.mktemp("release") / "plfs.duckdb"
    rows = REAL["2025"] + REAL["2024"]
    # any order-reversing change keeps every total between its parts
    wpr = [{**r, "value": str(round(100 - float(r["value"]), 1))} for r in rows]
    build_release(path, annual={"ur_annual": ("unemployment rate (UR), annual", rows),
                                "wpr_annual": ("worker population ratio (WPR), annual", wpr)},
                  monthly={"ur_monthly": ("unemployment rate (UR), monthly", monthly_rows(), "CWS", "test")},
                  unstated_age="15 years and above")
    release = Release(path)
    profiles = build_profiles(release, unstated_age="15 years and above")
    release.close()
    add_profiles(path, profiles)
    return path


@pytest.fixture(scope="session")
def release(release_path):
    r = Release(release_path)
    yield r
    r.close()

@pytest.fixture
def model():
    """Install a scripted model and a fresh governor for one test; restore both afterwards."""
    from stubs import Scripted

    scripted = Scripted()
    previous = llm.set_backend(scripted)
    previous_gov = llm.set_governor(Governor(limits={}))
    yield scripted
    llm.set_backend(previous)
    llm.set_governor(previous_gov)


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def small_limits():
    return Limits(tpm=1000, tpd=3000, per_request=800)
