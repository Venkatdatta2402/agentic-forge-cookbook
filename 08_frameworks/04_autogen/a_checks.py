"""What building the analysis team found out about AutoGen, each one reproducible offline.

    event_loops        AutoGen's code executor under the event loop a script runs and the one Jupyter runs
    secrets            what model-written code can read: AutoGen's executor against this project's
    gate_examples      what `brightwater.gate` stops -- and one way straight past it
    termination        a reviewer that never approves: which of the four conditions ends the chat
    traps              what the three questions give on the export as it stands, and once cleaned
"""

import asyncio
import json
import tempfile
from pathlib import Path

import a_stubs as S
import analysis_team as AT
import brightwater as B

_PEEK = "import os\nprint(sorted(k for k in os.environ if k.endswith('API_KEY')))"


async def _run(executor_class, code):
    from autogen_core import CancellationToken
    from autogen_core.code_executor import CodeBlock
    result = await executor_class(work_dir=tempfile.mkdtemp(), timeout=30).execute_code_blocks(
        [CodeBlock(code=code, language="python")], CancellationToken())
    return result.output.strip()


def _on_thread(fn, timeout=120):
    """`fn()` on a thread of its own. A Jupyter kernel's thread already runs an event loop, and one loop
    cannot be run inside another -- the same reason `run_team` gives each team a thread."""
    import threading
    box = {}

    def worker():
        try:
            box["value"] = fn()
        except Exception as failed:     # noqa: BLE001 -- what it raises is the finding
            box["value"] = f"{type(failed).__name__}: {failed}"

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    return box.get("value", f"did not finish within {timeout}s")


def event_loops():
    return _on_thread(_event_loops)


def _event_loops():
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    found = {}
    for name, make in [("ProactorEventLoop (a script's default)", asyncio.ProactorEventLoop),
                       ("SelectorEventLoop (what a Jupyter kernel runs on Windows)", asyncio.SelectorEventLoop)]:
        loop = make()
        try:
            found[name] = "ran: " + loop.run_until_complete(_run(LocalCommandLineCodeExecutor, "print('hello')"))
        except Exception as failed:     # noqa: BLE001 -- what it raises is the finding
            found[name] = f"{type(failed).__name__}: {failed}"
        finally:
            loop.close()
    return found


def secrets():
    return _on_thread(_secrets)


def _secrets():
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    loop = asyncio.ProactorEventLoop()
    try:
        return {"AutoGen's executor -- the code can see": loop.run_until_complete(_run(LocalCommandLineCodeExecutor, _PEEK)),
                "this project's executor -- the code can see": loop.run_until_complete(_run(AT._executor_class(), _PEEK))}
    finally:
        loop.close()


def gate_examples():
    examples = {
        "reads the CSV": 'import pandas as pd\nprint(pd.read_csv("orders.csv").shape)',
        "prints an API key": 'import os\nprint(os.environ["GROQ_API_KEY"])',
        "calls a web service": 'import requests\nrequests.get("https://example.com")',
        "deletes a file": 'import os\nos.remove("orders.csv")',
        "the same key, written differently": 'from os import environ\nprint(environ.get("GROQ_API_KEY"))',
    }
    return {name: B.gate(code)[1] or "allowed" for name, code in examples.items()}


def termination():
    """A reviewer that never approves, against a replayed model: which condition stops it, and when."""
    import threading
    box = {}

    async def go():
        work_dir = tempfile.mkdtemp()
        B.write_orders(work_dir)
        team = AT.build_team(S.replay("Q1", approve=False), work_dir, max_messages=12)
        return await team.run(task=B.QUESTIONS["Q1"])

    def worker():
        loop = asyncio.ProactorEventLoop()
        try:
            box["result"] = loop.run_until_complete(go())
        except Exception as failed:     # noqa: BLE001
            box["error"] = f"{type(failed).__name__}: {failed}"
        finally:
            loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(120)
    if "error" in box:
        return {"error": box["error"]}
    result = box["result"]
    return {"messages": len(result.messages), "stop_reason": result.stop_reason,
            "speakers": [m.source for m in result.messages if type(m).__name__ in ("TextMessage",)]}


def traps():
    clean, naive = B.truth(), B.naive()
    return {qid: {"the export as it stands": (naive["duplicates kept"][qid]["entity"], naive["duplicates kept"][qid]["value"]),
                  "duplicates kept AND returns counted": (naive["duplicates kept, returns counted"][qid]["entity"],
                                                          naive["duplicates kept, returns counted"][qid]["value"]),
                  "cleaned (the truth)": (clean[qid]["entity"], clean[qid]["value"])}
            for qid in B.QUESTIONS}


def show_run(qid, out, width=320):
    """One run, read the way a reviewer would: who said what, what the code printed, the finding, the grade."""
    if out.get("error"):
        print("error:", out["error"])
    for turn in out["turns"][1:]:
        if turn.get("kind") == "CodeGenerationEvent":
            continue                                    # the code is long; what it printed is below
        text = " ".join(turn["text"].split())
        print(f"--- {turn['source']}{' (code ran)' if turn.get('kind') == 'CodeExecutionEvent' else ''}: "
              f"{text[:width]}{'...' if len(text) > width else ''}")
    graded = B.grade(qid, out["finding"])
    print(f"\n{'RIGHT' if graded['right'] else 'WRONG'}: answered {graded['entity']} {graded['value']}, "
          f"the truth is {graded['want'][0]} {graded['want'][1]}")
    return graded["right"]


def checked_duplicates(out):
    """Whether any program the analyst ran looked for repeated rows -- the step Q1 needs."""
    return any(".duplicated(" in t["text"] or "drop_duplicates(" in t["text"] for t in out.get("turns", []))


def skills_report(qid, out):
    """What the planner read, against what the question needed, and whether the analysis then did it."""
    read, needed = out.get("skills read", []), B.RELEVANT.get(qid, set())
    return {"read": read, "needed and read": sorted(needed & set(read)), "needed, not read": sorted(needed - set(read)),
            "read, not needed": sorted(set(read) - needed), "the analysis checked for duplicates": checked_duplicates(out)}


STORE = Path(__file__).resolve().parent / "store"


def save(build, qid, out, cost=None):
    """One live run, kept on disk so the side-by-side does not depend on one kernel having run everything."""
    STORE.mkdir(exist_ok=True)
    finding = out.get("finding")
    record = {"build": build, "qid": qid, "finding": finding.model_dump() if finding else None,
              "usage": cost or out.get("usage") or {}, "usage on messages": out.get("usage on messages"),
              "skills read": out.get("skills read", []), "checked duplicates": checked_duplicates(out),
              # in full, so what a plan asked for can be read afterwards rather than guessed from a cut line
              "plans": [t["text"] for t in out.get("turns", []) if t["source"] == "planner"
                        and t.get("kind", "TextMessage") == "TextMessage"],
              "transcript": [{"source": t["source"], "kind": t.get("kind", ""), "text": t["text"]}
                             for t in out.get("turns", [])],
              "turns": len(out.get("turns", [])),
              "speakers": out.get("speakers", []), "stop_reason": out.get("stop_reason"), "error": out.get("error")}
    (STORE / f"{build}_{qid}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def load():
    """Every saved run: {(build, qid): record}, with the finding back as a Finding."""
    runs = {}
    for path in sorted(STORE.glob("*_Q*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["finding"] = B.Finding(**record["finding"]) if record["finding"] else None
        runs[(record["build"], record["qid"])] = record
    return runs
