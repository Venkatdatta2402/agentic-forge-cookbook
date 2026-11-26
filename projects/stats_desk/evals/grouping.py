"""SPENDS TOKENS (a little, on the grouping model). The task check's grouping call
(agents/grouping.py) on its own, against hand labels (data/grouping.json): for each task, the
groups of two fields or more it gives against the labelled ones, and whether the task check would
decide the same (code asks the server whether each group is published together; no tokens). A
check that refuses a task the labels pass is the costly mistake: it blocks a valid task.

    python -m evals.grouping [--only G1,C2] [--repeat N] [--model openai/gpt-oss-120b] [--effort medium]

--model and --effort set the call's model and reasoning effort for this run (else config's), to
compare them on the same cases; each row records which it ran on.
"""

import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _key(groups):
    """The groups that combine fields, as a set: what the check probes."""
    return {frozenset(g.items()) for g in groups if len(g) > 1}


async def _refuses(analyses, tables, task, groups):
    from stats_desk.agents.lead import unpublished_group
    out = []
    for g in groups:
        if len(g) > 1:
            found = await unpublished_group(analyses, tables, task, g)
            if found:
                out.append(found)
    return out


async def run(cases, repeat):
    from stats_desk import config
    from stats_desk.agents import grouping, lead
    from stats_desk.analysis.client import connect
    from stats_desk.core import meter
    rows = []
    async with connect() as analyses:
        tables = await analyses.call("tables")
        _, values = lead.catalogue(tables)
        for case in cases:
            names = lead.names_in(case["task"], values)
            expected = await _refuses(analyses, tables, case["task"], case["groups"])
            for rep in range(1, repeat + 1):
                raw = []
                with meter.metered() as usage:
                    got = await grouping.groups_in(case["task"], names, keep=raw)
                refused = await _refuses(analyses, tables, case["task"], got or [])
                row = {"id": case["id"], "rep": rep, "model": config.GROUPING_MODEL, "effort": config.GROUPING_EFFORT,
                       "task": case["task"], "names": [f"{v} ({f})" for f, v in names],
                       "reply": raw, "got": got, "expected": case["groups"], "tokens": usage.total,
                       "same groups": got is not None and _key(got) == _key(case["groups"]),
                       "should refuse": bool(expected), "would refuse": bool(refused),
                       "refusal": refused}
                rows.append(row)
                mark = "ok" if row["same groups"] else "DIFFERS"
                print(f"{case['id']} #{rep}: {usage.total} tokens, {mark}, refuse {row['would refuse']} "
                      f"(should {row['should refuse']})", flush=True)
    return rows


def main(argv):
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "span")
    data = json.loads((HERE / "data" / "grouping.json").read_text(encoding="utf-8"))
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 1
    from stats_desk import config
    if "--model" in argv:
        config.GROUPING_MODEL = argv[argv.index("--model") + 1]
    if "--effort" in argv:
        config.GROUPING_EFFORT = argv[argv.index("--effort") + 1]
    cases = [c for c in data["cases"] if not only or c["id"] in only]
    rows = asyncio.run(run(cases, repeat))
    tag = f"{config.GROUPING_MODEL.split('/')[-1]}_{config.GROUPING_EFFORT}"
    out = HERE / "runs" / f"grouping_{date.today().isoformat()}_{tag}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\n{config.GROUPING_MODEL}, effort {config.GROUPING_EFFORT}")
    for name, part in (("all", rows), *((s, [r for r in rows if r["id"].startswith(s)]) for s in ("G", "C", "H"))):
        if not part:
            continue
        n = len(part)
        print(f"  {name}: {n} calls; same groups {sum(r['same groups'] for r in part)}/{n}; "
              f"valid task refused {sum(r['would refuse'] and not r['should refuse'] for r in part)}; "
              f"unpublished missed {sum(r['should refuse'] and not r['would refuse'] for r in part)}; "
              f"~{sum(r['tokens'] for r in part) // n} tokens a call")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
