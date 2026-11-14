"""The runs scored by code, from what collect_runs.py saved: outcome, path and cost. Free.

    outcome   a brief or not; which of the answer key's points it makes (all of a point's numbers
              in the brief); the measure named; the question's own checks (the traps); numbers
              in the brief that no result of the run shows, and numbers only a profile shows
              (run 3 slipped invented numbers through that way); cause words in the answer
    path      each reference step covered, by the first result holding its numbers (matched by
              what came back, not by the arguments: the same figure can come from several
              analyses); recall over the steps; precision over the analyses the agents chose;
              wasted analyses (refused, repeated, never cited and covering nothing); findings
              built from two or more single figures where one comparison would do; the lead's
              tasks, listed for marking by hand
    cost      tokens (by agent), model calls, seconds, seconds waiting for Groq's minute limit,
              Groq's errors, tasks stopped, tasks refused for budget -- from the run's trace

Hand marks on the lead's tasks (data/marks_<runs file>.json), e.g. {"Q1#1": {"tasks": {"f1":
{"task": "<the task's words>", "mark": "lookup", "why": "asks for two figures, tests nothing"}}}},
are shown beside the tasks.

    python -m evals.score evals/runs/runs_<date>.jsonl      the report, to evals/reports/<name>.md
"""

import json
import re
import sys
from pathlib import Path

from stats_desk.agents.checks import cause_claims
from stats_desk.agents.lead import plain

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "data" / "golden.json"
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
RESULT_ID = re.compile(r"\b[a-z]+-[0-9a-f]{6}\b")
ANALYSES = {"value", "compare", "gap", "change", "series", "relate", "drill", "tables"}


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def golden():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return data, {q["id"]: q for q in data["questions"]}


def numbers(text):
    """The numbers in a text, signs aside, result ids left out ("gap-83c1ba" is not 83)."""
    return {abs(float(n)) for n in NUMBER.findall(RESULT_ID.sub("", text or ""))}


def has(text, wanted):
    found = numbers(text)
    return all(abs(float(n)) in found for n in wanted)


def brief_text(brief):
    """The brief as a reader gets it, the line code writes about which figures it rests on included."""
    return "\n".join([*(brief.get(k) or "" for k in ("answer", "ruled_out", "cannot_tell")),
                      "; ".join(brief.get("figures_from") or [])])


def _table(call):
    """The table a result came from: its analysis's `table` (or `x_table`); a recheck's, from the
    analysis it reran."""
    if not call:
        return None
    tool, args = call[0], call[1] or {}
    if tool == "recheck":
        args = args.get("args") or {}
    return args.get("table") or args.get("x_table")


# --------------------------------------------------------------------------- outcome

def outcome(record, q, measures):
    brief = record["brief"]
    # gpt-oss writes non-breaking hyphens and spaces ("built‑in"): read them as plain ones
    text = brief_text(brief).replace("\u2011", "-").replace("\u2010", "-").replace("\u00a0", " ")
    made = [p["id"] for p in q["points"] if has(text, p["numbers"])]
    results = record["results"]
    shown = numbers("\n".join(t.split("\n", 1)[-1] for i, t in results.items() if not i.startswith("profile-")))
    profiles = numbers("\n".join(t for i, t in results.items() if i.startswith("profile-")))
    in_brief = numbers(text)
    checks = {}
    for c in q["checks"]:
        if c["kind"] == "says_any":
            checks[c["id"]] = any(w.lower() in text.lower() for w in c["words"])
        elif c["kind"] == "no_cause_words":
            checks[c["id"]] = not cause_claims(brief.get("answer"))
        elif c["kind"] == "no_bare_verdict":
            # span tests, 2026-11-12: asked whether a degree MAKES people unemployed, the lead
            # answered "Yes." -- agreeing to a cause without one cause word in it
            checks[c["id"]] = not re.match(r"\s*(yes|no)\b", brief.get("answer") or "", re.I)
        elif c["kind"] == "cannot_tell":
            checks[c["id"]] = bool((brief.get("cannot_tell") or "").strip())
        elif c["kind"] == "points":
            checks[c["id"]] = all(p in made for p in c["points"])
    return {"brief": not brief.get("stopped") and bool(brief.get("answer")),
            "stopped": brief.get("stopped"),
            "points": made, "of": len(q["points"]),
            # monthly figures come on one measure only: naming it is asked of annual figures
            "measure named": q["measure"] is None or any(plain(w) in plain(text) for w in measures[q["measure"]]),
            "checks": checks,
            "numbers not in any result": sorted(in_brief - shown - profiles),
            "numbers only in a profile": sorted((in_brief & profiles) - shown),
            "cause words": cause_claims(brief.get("answer"))}


# --------------------------------------------------------------------------- path

def path(record, q):
    results, calls, tools = record["results"], record["calls"], record["trace"]["tools"]
    steps = {s["id"]: None for s in q["steps"]}
    seen, rows = set(), []
    cited = {i for f in (record["brief"].get("all_findings") or {}).values() if isinstance(f, dict)
             for i in (f.get("results") or []) + (f.get("checked_with") or [])}
    for t in tools:
        if t["tool"] not in ANALYSES and t["tool"] != "recheck":
            continue
        rid = t.get("result")
        text = results.get(rid, "") if rid else ""
        covered = []
        for s in q["steps"]:
            if steps[s["id"]] is not None:
                continue
            if "refused_with" in s:
                hit = t["kind"] != "ok" and s["refused_with"] in (t.get("said") or "")
            else:
                hit = (bool(text) and has(text.split("\n", 1)[-1], s["numbers"])
                       and ("table" not in s or _table(calls.get(rid)) == s["table"]))
            if hit:
                steps[s["id"]] = t["n"]
                covered.append(s["id"])
        repeated = bool(rid) and rid in seen
        seen.add(rid)
        wasted = (t["by"] == "model" and t["tool"] in ANALYSES and not covered and
                  (t["kind"] != "ok" or repeated or rid not in cited))
        rows.append({"n": t["n"], "by": t["by"], "path": t["path"], "tool": t["tool"],
                     "args": t["args"], "kind": t["kind"], "result": rid, "said": t.get("said"),
                     "covered": covered, "repeated": repeated, "wasted": wasted})
    chosen = [r for r in rows if r["by"] == "model" and r["tool"] in ANALYSES]
    found = [s for s, n in steps.items() if n is not None]
    chains = []
    for fid, f in (record["brief"].get("all_findings") or {}).items():
        if isinstance(f, dict):
            values = [calls.get(i) for i in f.get("results") or [] if (calls.get(i) or [None])[0] == "value"]
            tables = [_table(c) for c in values]
            if any(tables.count(t) >= 2 for t in set(tables)):
                chains.append(fid)
    tasks = {}
    for t in tools:
        if t["tool"] == "ask_analysts":
            try:
                tasks[f"call {t['n']}"] = json.loads(t["args"]).get("tasks") if isinstance(t["args"], str) else t["args"]
            except ValueError:
                tasks[f"call {t['n']}"] = t["args"]
    return {"steps": steps, "recall": round(len(found) / len(steps), 2) if steps else None,
            "missed": [s for s, n in steps.items() if n is None],
            "analyses chosen": len(chosen),
            "precision": round(sum(bool(r["covered"]) for r in chosen) / len(chosen), 2) if chosen else None,
            "wasted": sum(r["wasted"] for r in rows),
            "refused": sum(r["kind"] != "ok" for r in chosen),
            "single figures chained": chains, "tasks": tasks, "calls": rows}


# --------------------------------------------------------------------------- cost

def cost(record):
    t = record["trace"]
    by_agent = {}
    for c in t["model calls"]:
        agent = (c.get("agent") or "?").split(" f")[0]          # "analyst f2" -> "analyst"
        by_agent[agent] = by_agent.get(agent, 0) + (c["in"] or 0) + (c["out"] or 0)
    findings = (record["brief"].get("all_findings") or {}).values()
    stopped = [f.get("reason") for f in findings if isinstance(f, dict) and "reason" in f and "task" not in f]
    return {"tokens": t["tokens"]["total"], "by agent": by_agent, "model calls": len(t["model calls"]),
            "seconds": t["seconds"], "waited": t["waited seconds"], "cost if paid": t["cost if paid"],
            "groq errors": [c["error"] for c in t["model calls"] if c.get("error")],
            "tasks stopped": stopped,
            "tasks refused": sum(1 for x in t["tools"] if x["tool"] == "ask_analysts" and x["kind"] != "ok")}


def score(records):
    data, by_id = golden()
    return [{"id": r["id"], "run": r["run"], "question": r["question"],
             "outcome": outcome(r, by_id[r["id"]], data["measures"]), "path": path(r, by_id[r["id"]]),
             "cost": cost(r), "answer": r["brief"].get("answer")} for r in records]


# --------------------------------------------------------------------------- the report

def markdown(rows, name, marks=None):
    _, by_id = golden()
    yes = lambda b: "yes" if b else "**no**"
    key = lambda r: f"{r['id']}#{r['run']}"
    lines = [f"# Eval: {name}", "", "One run per question: one sample of the desk's behaviour, not its average "
             "(Groq's free tier, 200k tokens a day).", "", "## Outcome: was the answer right?", "",
             "| | kind | brief | points made | measure named | checks | numbers in no result | only in a profile | cause words |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o, q = r["outcome"], by_id[r["id"]]
        checks = "; ".join(f"{k}: {yes(v)}" for k, v in o["checks"].items()) or "-"
        lines.append(f"| {key(r)} | {q['kind']}{' (seen in development)' if q.get('seen_in_development') else ''} | "
                     f"{yes(o['brief'])} | {len(o['points'])}/{o['of']} {', '.join(o['points'])} | "
                     f"{yes(o['measure named'])} | {checks} | {', '.join(map(str, o['numbers not in any result'])) or '-'} | "
                     f"{', '.join(map(str, o['numbers only in a profile'])) or '-'} | {', '.join(o['cause words']) or '-'} |")
    lines += ["", "## Path: were the steps right?", "",
              "A step is covered by the first result holding its numbers (code's reruns count, marked). Precision: "
              "analyses the agents chose that covered a step, of all they chose.", "",
              "| | steps covered | missed | analyses chosen | precision | wasted | refused | single figures chained |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["path"]
        lines.append(f"| {key(r)} | {p['recall']} | {', '.join(p['missed']) or '-'} | {p['analyses chosen']} | "
                     f"{p['precision']} | {p['wasted']} | {p['refused']} | {', '.join(p['single figures chained']) or '-'} |")
    lines += ["", "## Cost: what did it take?", "",
              "| | tokens | by agent | model calls | s | waiting s | Groq errors | tasks stopped | tasks refused |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["cost"]
        agents = ", ".join(f"{a} {n:,}" for a, n in sorted(c["by agent"].items(), key=lambda kv: -kv[1]))
        lines.append(f"| {key(r)} | {c['tokens']:,} | {agents} | {c['model calls']} | {c['seconds']} | {c['waited']} | "
                     f"{len(c['groq errors'])} | {len(c['tasks stopped'])} | {c['tasks refused']} |")
    for r in rows:
        hand = (marks or {}).get(key(r), {}).get("tasks", {})
        lines += ["", f"### {key(r)}: {r['question']}", "", f"Answer: {r['answer'] or '(no brief: ' + str(r['outcome']['stopped']) + ')'}",
                  "", "The lead's tasks:"]
        for call, tasks in r["path"]["tasks"].items():
            for i, task in enumerate(tasks or []):
                mark = next((m for m in hand.values() if m.get("task") == task), {})
                lines.append(f"- ({call}) {task}" + (f" -- **{mark['mark']}**: {mark.get('why', '')}" if mark else ""))
        lines += ["", "| n | by | who | analysis | arguments | came back | steps |", "|---|---|---|---|---|---|---|"]
        for c in r["path"]["calls"]:
            back = c["result"] or f"{c['kind']}: {(c['said'] or '')[:100]}"
            flag = " (wasted)" if c["wasted"] else " (repeat)" if c["repeated"] else ""
            lines.append(f"| {c['n']} | {c['by']} | {c['path'].split(' > ')[-1]} | {c['tool']} | "
                         f"`{str(c['args'])[:90]}` | {back}{flag} | {', '.join(c['covered']) or '-'} |")
    return "\n".join(lines) + "\n"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    path_ = Path(argv[0])
    rows = score(load(path_))
    marks_file = HERE / "data" / f"marks_{path_.stem}.json"
    marks = json.loads(marks_file.read_text(encoding="utf-8")) if marks_file.exists() else None
    text = markdown(rows, path_.name, marks)
    out = HERE / "reports" / f"{path_.stem}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
