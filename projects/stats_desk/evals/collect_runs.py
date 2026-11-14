"""SPENDS TOKENS. The desk's runs on the eval questions, everything kept, for score.py.

Run by hand, never in CI. Each run is one whole investigation (lead, analysts, code's checks) on
the Groq key in .env, traced (environment "eval" in Langfuse). About 40-60k tokens a run, so
about 4 questions a day on Groq's free 200k: runs are appended as they finish, a run stopped by
the daily limit is not saved, and the next start carries on in the same file.

One run per question is one sample of the desk's behaviour, not its average (`--runs N` for more).

    python -m evals.collect_runs [Q2 Q5 ...] [--runs N] [--name X | --into FILE]
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "data" / "golden.json"
RUNS = HERE / "runs"


def out_path(name=None):
    """The newest runs file, so a collection stopped by the daily limit carries on in it; a new one
    dated today with `name` (runs_<date>_<name>.jsonl)."""
    if name:
        return RUNS / f"runs_{date.today().isoformat()}_{name}.jsonl"
    return max(RUNS.glob("runs_*.jsonl"), default=RUNS / f"runs_{date.today().isoformat()}.jsonl")


def done(path):
    if not path.exists():
        return set()
    return {(r["id"], r["run"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def _daily_limit(error, kind):
    """The daily-limit error inside `error`, or None. The MCP client's task group wraps an error
    raised inside it in an ExceptionGroup (live, 2026-11-11: the limit crashed the collection)."""
    if isinstance(error, kind):
        return error
    for inner in getattr(error, "exceptions", ()):
        found = _daily_limit(inner, kind)
        if found is not None:
            return found
    return None


def plain(value):
    return value.model_dump() if hasattr(value, "model_dump") else str(value)


async def collect(question, qid=None):
    """One investigation, with everything score.py reads."""
    from stats_desk.agents.lead import investigate
    from stats_desk.analysis.client import connect
    began = time.monotonic()
    async with connect() as analyses:
        brief, run = await investigate(analyses, question, via="eval", tags=[qid] if qid else ())
    record = asdict(brief)
    return {"question": question, "brief": {k: v for k, v in record.items() if k != "trace"},
            "trace": brief.trace, "results": analyses.results,
            "calls": {i: list(c) for i, c in analyses.calls.items()},
            "transcripts": run.transcripts, "seconds": round(time.monotonic() - began, 1)}


def main(argv):
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "eval")      # eval runs apart in Langfuse
    from stats_desk.core import trace
    from stats_desk.core.governor import DailyLimitReached
    runs = int(argv[argv.index("--runs") + 1]) if "--runs" in argv else 1
    wanted = {a for a in argv if a.startswith("Q")}
    questions = [q for q in json.loads(GOLDEN.read_text(encoding="utf-8"))["questions"]
                 if not wanted or q["id"] in wanted]
    RUNS.mkdir(exist_ok=True)
    path = (Path(argv[argv.index("--into") + 1]) if "--into" in argv else
            out_path(argv[argv.index("--name") + 1] if "--name" in argv else None))
    have = done(path)
    print(f"runs go to {path}; {len(have)} already there", flush=True)
    for n in range(1, runs + 1):
        for q in questions:
            if (q["id"], n) in have:
                continue
            try:
                record = asyncio.run(collect(q["question"], q["id"]))
            except BaseException as error:
                stop = _daily_limit(error, DailyLimitReached)
                if stop is None:
                    raise
                print(f"{q['id']} run {n}: Groq's daily limit ({stop}); not saved. Run again after it frees up.")
                trace.flush()
                return 1
            with path.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"id": q["id"], "run": n, **record}, default=plain, ensure_ascii=False) + "\n")
            t = record["trace"]
            print(f"{q['id']} run {n}: {t['outcome']}, {t['tokens']['total']:,} tokens, {t['seconds']} s "
                  f"({t['waited seconds']} s waiting)", flush=True)
            trace.flush()
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
