"""Ask the desk one question, live (spends Groq tokens). Saves the whole run before anything
judges it: runs/<time>.json holds the brief, every agent's transcript, the evidence and usage.

    uv run python scripts/ask.py "Why do young women in cities struggle to find work?"
"""

import asyncio
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from stats_desk.agents.lead import investigate
from stats_desk.analysis.client import connect
from stats_desk.core import events, trace

RUNS = Path(__file__).resolve().parents[1] / "runs"


def show(event):
    detail = {k: v for k, v in event.items() if k not in ("kind", "agent")}
    text = ", ".join(f"{k}={str(v)[:120]}" for k, v in detail.items())
    print(f"  [{time.strftime('%H:%M:%S')}] {event['agent'] or '-'}: {event['kind']} {text}", file=sys.stderr)


def plain(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


async def main(question):
    started = time.strftime("%Y%m%d-%H%M%S")
    with events.listening(show):
        async with connect() as analyses:
            brief, run = await investigate(analyses, question)
    print("\n" + brief.text())
    print(f"\nusage: {json.dumps(brief.usage)}")
    trace.flush()
    if trace.LANGFUSE.client is not None and brief.trace.get("langfuse trace"):
        print("langfuse: " + trace.LANGFUSE.client.get_trace_url(trace_id=brief.trace["langfuse trace"]))
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{started}.json"
    path.write_text(json.dumps({"question": question, "brief": plain(brief), "transcripts": run.transcripts,
                                "results": analyses.results}, default=plain, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    asyncio.run(main(" ".join(sys.argv[1:]) or "Why do young women in cities struggle to find work?"))
