"""Settings, read from the environment. Locally every `.env` from this project up to the repo
root is read, nearest first, and none overrides another: this project's has its Langfuse keys,
the repo root's the Groq key. On Azure Container Apps the variables (and secrets) are set on the app."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[2]
for folder in (PROJECT, *PROJECT.parents):
    if (folder / ".env").is_file():
        load_dotenv(folder / ".env")          # a variable already set is kept

LEAD_MODEL = os.environ.get("STATS_DESK_LEAD_MODEL", "openai/gpt-oss-120b")
# 120b, not 20b: on day 1's live runs gpt-oss-20b failed the analyst workflow 4 times out of 4
# (ignored refusals, never passed the codes it was shown); 120b followed it 5 out of 5.
ANALYST_MODEL = os.environ.get("STATS_DESK_ANALYST_MODEL", "openai/gpt-oss-120b")

# set only for load tests, which point the app at a stand-in for Groq
GROQ_BASE_URL = os.environ.get("STATS_DESK_GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Free questions, on the desk's own keys, for visitors without one (decided 2026-11-13). Each key
# should be on a different Groq organisation: Groq's limits are per organisation, so two keys of
# one organisation share one daily allowance.
FREE_KEYS = [k.strip() for k in os.environ.get("STATS_DESK_FREE_GROQ_KEYS", "").split(",") if k.strip()]
# No fixed count per key (user, 2026-11-13): a key is used until Groq says its day is spent. The
# only count is per visitor (by a hash of their internet address): 3 a day, new questions and
# follow-ups together (user, 2026-11-13).
FREE_PER_VISITOR_PER_DAY = int(os.environ.get("STATS_DESK_FREE_PER_VISITOR_PER_DAY", 3))
# The free keys are used by the desk alone (user, 2026-11-13), so the desk can count each key's
# last 24 hours itself: a question starts on the key with the most left, and not on a key with
# less than FREE_MIN_LEFT (it would run dry almost at once). Groq's own "daily limit" refusal
# stays the backstop when the count is off.
FREE_TOKENS_PER_DAY = int(os.environ.get("STATS_DESK_FREE_TOKENS_PER_DAY", 200_000))
FREE_MIN_LEFT = int(os.environ.get("STATS_DESK_FREE_MIN_LEFT", 20_000))
MOSPI_URL = os.environ.get("STATS_DESK_MOSPI_URL", "https://mcp.mospi.gov.in/mcp")
# the release file the analysis server reads (built by scripts/build_release.py)
RELEASE_PATH = Path(os.environ.get("STATS_DESK_RELEASE", PROJECT / "data" / "release" / "plfs.duckdb"))


@dataclass(frozen=True)
class Limits:
    """One model's limits on this Groq account. The free tier's, unless the environment says otherwise."""

    tpm: int = 8_000            # tokens per minute
    tpd: int = 200_000          # tokens per day (Groq measures a rolling 24 hours)
    per_request: int = 8_000    # prompt + max_tokens in a single request


def limits_for(model: str) -> Limits:
    key = model.split("/")[-1].upper().replace("-", "_").replace(".", "_")
    get = lambda name, default: int(os.environ.get(f"STATS_DESK_{name}_{key}", default))
    base = Limits()
    return Limits(tpm=get("TPM", base.tpm), tpd=get("TPD", base.tpd),
                  per_request=get("PER_REQUEST", base.per_request))
