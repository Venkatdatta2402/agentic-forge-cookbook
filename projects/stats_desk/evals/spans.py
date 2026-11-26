"""SPENDS TOKENS (a little). Each step of the desk tested on its own: a fixed input, only that
step run live, its output checked by code. A whole question costs 30-50k tokens and a failure
in it could be any step's; a step costs one to a few model calls, a failure points at the step,
and a step can be run several times to see whether it behaves the same way.

    pick       the lead's first call: which profiles it asks for, given the question and the
               list of profiles. Checked: the needed ones, not all six
    tasks      the lead's next call: its plan (agents/plan.py), given the profiles it needed.
               Checked: it hands out tasks, at most FIRST_ROUND now, none pointing the analyst at
               a profile; how many later tasks it planned; each listed for marking by hand as a
               hypothesis or a lookup; whether code's task check refuses them, and if so the
               lead's second plan
    retask     the lead's reply to code refusing a fixed plan (one a lead really wrote) for a
               measure or age group not published for its groups. Checked: it gives the tasks
               again, and code passes them
    analyst    one analyst on one task, in full. Checked: it handed in, used an analysis that
               fits the task, didn't chain single figures, how many analyses and refusals
    brief      the lead from the point its findings came back, given fixed findings with code's
               reruns and warnings (the traps among them), through its real hand-in loop
               (refusals handed back, up to 3 tries). Checked: a brief handed in, whether the
               first try passed the checks, and the eval's checks on it (golden.json: points, traps)

Fixed inputs: data/spans.json and the eval questions. Every step runs inside a trace (Langfuse,
environment "span"). Results go to runs/spans_<date>.jsonl and reports/spans_<date>.md.

    python -m evals.spans [pick tasks analyst brief] [--only Q2,Q5,A1] [--repeat N]
"""

import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPANS = ("pick", "tasks", "retask", "analyst", "brief")


def cases():
    return json.loads((HERE / "data" / "spans.json").read_text(encoding="utf-8"))


def _decision_calls(decision):
    return [(c["name"], json.loads(c["arguments"] or "{}")) for c in decision.get("calls", [])] \
        if decision.get("type") == "tool_calls" else []


def _call(name, args, result, kind="ok"):
    """A call and its result, as they sit in an agent's conversation."""
    cid = f"span_{uuid.uuid4().hex[:8]}"
    return [{"role": "decision", "type": "tool_calls", "reasoning": "", "applied": True,
             "calls": [{"id": cid, "name": name, "arguments": json.dumps(args, ensure_ascii=False)}]},
            {"role": "tool", "tool_call_id": cid, "name": name, "kind": kind, "content": result}]


async def _current(analyses, names):
    """Profile names as the release now has them: a monthly profile is named for its latest month,
    so spans.json writes it without one ("UR_monthly")."""
    tables = await analyses.call("tables")
    have = next(line for line in tables.split("\n") if line.startswith("profiles: ")).removeprefix("profiles: ").split(", ")
    return [n if n in have else next((p for p in have if p.startswith("_".join(n.split("_")[:2]) + "_")), n)
            for n in names]


async def _profiles(analyses, names):
    """The lead's profile call and its reply."""
    return _call("profile", {"names": names}, await analyses.call("profile", names=names))


async def _lead_turn(analyses, question, history):
    """One call of the lead, as the investigation would make it at that point."""
    from stats_desk.agents.agent import form_tool, system
    from stats_desk.agents.lead import BriefForm, Investigation
    from stats_desk.core import trace
    from stats_desk.core.runtime import Conversation, Think
    from stats_desk.core.tools import Registry
    run = Investigation(analyses, question)
    run.tables = await analyses.call("tables")
    await run.read_question()
    lead = run.lead()
    holder = []
    form = form_tool(BriefForm, holder, build=lead.form_build)
    think = Think(Registry([*lead.tools, form]), lead.model, lead.max_tokens, lead.reasoning_effort,
                  shorten=lead.shorten, keep_recent=lead.keep_recent, require_tool=True, choose=lead.choose)
    conversation = Conversation(system=system(lead.persona), messages=[
        {"role": "user", "content": run.context()}, {"role": "user", "content": question}, *history])
    with trace.span("lead"):
        decision = await think.run(conversation)
    return decision, form, run


async def pick(analyses, case, question):
    decision, _, _ = await _lead_turn(analyses, question, [])
    calls = _decision_calls(decision)
    names = [n for name, args in calls if name == "profile" for n in args.get("names", [])]
    return {"decided": [name for name, _ in calls] or ["wrote text"], "said": decision.get("content"), "profiles": names,
            "checks": {"asked for profiles": bool(names),
                       "has the needed ones": set(case["needed"]) <= set(names),
                       "not all six": len(set(names)) < 6,
                       "extra": sorted(set(names) - set(case["needed"]))}}


async def tasks(analyses, case, question):
    """The lead's plan; and when code's task check refuses it (a measure or age group its groups
    aren't published on), the refusal handed back as ask_analysts would, and the lead's second
    plan. No analyst runs."""
    from stats_desk.agents.lead import FIRST_ROUND
    from stats_desk.core.tools import ToolError
    history = await _profiles(analyses, case["needed"])
    decision, _, run = await _lead_turn(analyses, question, history)
    calls = _decision_calls(decision)
    asked = [args for name, args in calls if name == "ask_analysts"]
    given = [str(t) for args in asked for t in args.get("tasks", [])]
    later = [str(t) for args in asked for t in args.get("later", [])]
    named = {"names groups as published": any(n in t for n in case["names"] for t in given)} if case.get("names") else {}
    refused, again, again_refused = None, [], None
    if given:
        try:
            await run.check_tasks(given)
        except ToolError as e:
            refused = str(e)
    if refused:
        decision2, _, _ = await _lead_turn(analyses, question, history + _call("ask_analysts", asked[0], refused, "refused"))
        asked2 = [args for name, args in _decision_calls(decision2) if name == "ask_analysts"]
        again = [str(t) for args in asked2 for t in args.get("tasks", [])]
        if again:
            try:
                await run.check_tasks(again)
            except ToolError as e:
                again_refused = str(e)
    return {"decided": [name for name, _ in calls] or ["wrote text"], "said": decision.get("content"), "tasks": given,
            "approach": [args.get("approach") for args in asked], "later": later,
            "refused by code": refused, "tasks again": again, "refused again": again_refused,
            "checks": {"handed out tasks": bool(given),
                       "within the first round": 0 < len(given) <= FIRST_ROUND, **named,
                       "planned later tasks": len(later),
                       "none points at a profile": not any("profile" in t.lower() for t in given),
                       "no unpublished setting": not refused,
                       "rewrote after the refusal": (bool(again) and not again_refused) if refused else None}}


async def retask(analyses, case, question):
    """The lead's reply to code refusing its plan: the conversation as it stands after the lead
    sent a fixed plan (spans.json, a plan a lead really wrote) and code's task check refused it,
    the refusal written by that check now. Checked: it gives the tasks again, and code passes them."""
    from stats_desk.core.tools import ToolError
    history = await _profiles(analyses, case["needed"])
    from stats_desk.agents.lead import Investigation
    run = Investigation(analyses, question)
    run.tables = await analyses.call("tables")
    try:
        await run.check_tasks(case["plan"]["tasks"])
        raise RuntimeError(f"{case['id']}: code no longer refuses this plan; the case needs another")
    except ToolError as e:
        refused = str(e)
    decision, _, _ = await _lead_turn(analyses, question, history + _call("ask_analysts", case["plan"], refused, "refused"))
    calls = _decision_calls(decision)
    asked = [args for name, args in calls if name == "ask_analysts"]
    again = [str(t) for args in asked for t in args.get("tasks", [])]
    again_refused = None
    if again:
        try:
            await run.check_tasks(again)
        except ToolError as e:
            again_refused = str(e)
    return {"decided": [name for name, _ in calls] or ["wrote text"], "said": decision.get("content"),
            "refused by code": refused, "tasks": again, "approach": [args.get("approach") for args in asked],
            "refused again": again_refused,
            "checks": {"gave the tasks again": bool(again), "code passes them": bool(again) and not again_refused,
                       "same number of tasks": len(again) == len(case["plan"]["tasks"])}}


async def analyst(analyses, case, question=None):
    from stats_desk.agents.analyst import Finding, run_analyst, tables_for
    transcript = []
    tables = await analyses.call("tables")
    before = set(analyses.results)
    found = await run_analyst(analyses, "f1", case["task"], tables_for(tables, case["task"]), transcript=transcript)
    used = [c["name"] for m in transcript if m.get("role") == "decision" and m.get("type") == "tool_calls"
            for c in m["calls"] if c["name"] != "json"]
    refused = sum(1 for m in transcript if m.get("role") == "tool" and m.get("kind") not in (None, "ok"))
    cited = found.results if isinstance(found, Finding) else []
    values = [analyses.calls[i][1].get("table") for i in cited if analyses.calls.get(i, ("",))[0] == "value"]
    # what it called and what came back, so a failed run can be read without its trace
    steps = [{"call": c["name"], "args": c["arguments"]} for m in transcript
             if m.get("role") == "decision" and m.get("type") == "tool_calls" for c in m["calls"]]
    replies = [{"kind": m.get("kind"), "content": (m.get("content") or "")[:600]} for m in transcript if m.get("role") == "tool"]
    return {"claim": getattr(found, "claim", None), "stopped": getattr(found, "reason", None), "used": used,
            "steps": steps, "replies": replies,
            "new results": len(set(analyses.results) - before),
            "checks": {"handed in": isinstance(found, Finding) and not found.claim.startswith("no claim:"),
                       "handed in by code": isinstance(found, Finding) and found.claim.startswith("no claim:"),
                       "used an analysis that fits": any(u in case["fits"] for u in used),
                       "no chained single figures": not any(values.count(t) >= 2 for t in set(values)),
                       "analyses": len(used), "refused": refused}}


async def brief(analyses, case, question):
    from evals import score as S
    from stats_desk.agents.agent import answer
    from stats_desk.agents.analyst import Finding, compact
    from stats_desk.agents.checks import check
    from stats_desk.agents.lead import (MAX_ANALYSTS, BriefForm, Investigation, figures_from, not_comparable,
                                        without_ids)
    findings = []
    for n, spec in enumerate(case["findings"], 1):
        ids = []
        for c in spec["calls"]:
            text = await analyses.call(c["tool"], **c["args"])
            ids.append(text.split("\n")[0].removeprefix("result "))
        finding = Finding(f"f{n}", spec["task"], spec["claim"], ids,
                          figures={i: compact(analyses.results[i]) for i in ids})
        findings.append(await check(analyses, finding))
    needed = await _current(analyses, next(p for p in cases()["pick"] if p["id"] == case["id"])["needed"])
    history = (await _profiles(analyses, needed)
               + _call("ask_analysts", {"approach": "Test what the question asks.", "tasks": [f.task for f in findings]},
                       "\n\n".join(f.text() for f in findings)))
    # the lead's real hand-in loop from here: refusals handed back, a brief written as text read
    # as one, up to 3 tries. Its tasks are all used, so it can't start analysts in a test.
    run = Investigation(analyses, question)
    run.tables = await analyses.call("tables")
    await run.read_question()
    run.findings = {f.id: f for f in findings}
    run.findings.update({f"f{n}": None for n in range(len(findings) + 1, MAX_ANALYSTS + 1)})
    transcript = []
    result = await answer(run.lead(), question, BriefForm, context=run.context(), transcript=transcript,
                          history=history)
    handed = result.model_dump() if isinstance(result, BriefForm) else None
    if handed is not None:              # as Investigation.run writes it into the brief
        rests_on = [f for f in findings if f.id in handed["findings"]] or findings
        handed["figures_from"] = figures_from(analyses.results[i] for f in rests_on for i in f.results)
        for part in ("answer", "ruled_out", "cannot_tell"):
            handed[part] = without_ids(handed[part])
        note = not_comparable([f for f in findings if f.id in handed["findings"]], analyses.results)
        if note:
            handed["answer"] = f"{handed['answer'].rstrip()} {note}"
    refusals = [f"{m['kind']}: {m['content'][:300]}" for m in transcript
                if m.get("role") == "tool" and m.get("kind") not in (None, "ok")]
    _, golden = S.golden()
    record = {"brief": {**(handed or {}), "stopped": None if handed else result.reason}, "results": analyses.results}
    outcome = S.outcome(record, golden[case["id"]], S.golden()[0]["measures"])
    called = [c["name"] for m in transcript[len(history) + 2:] if m.get("role") == "decision"
              for c in m.get("calls", [])]
    return {"brief": handed, "stopped": None if handed else result.reason, "refusals": refusals, "called": called,
            "warnings shown": [w for f in findings for w in f.flags],
            "checks": {"handed in a brief": handed is not None, "first try passed": not refusals,
                       "refused": len(refusals), "points": f"{len(outcome['points'])}/{outcome['of']}",
                       **outcome["checks"], "measure named": outcome["measure named"],
                       "cause words": outcome["cause words"]}}


STEPS = {"pick": pick, "tasks": tasks, "retask": retask, "analyst": analyst, "brief": brief}


async def run_case(span, case, question):
    from stats_desk.analysis.client import connect
    from stats_desk.core import meter, trace
    began = time.monotonic()
    async with connect() as analyses:
        if case.get("needed"):
            case = {**case, "needed": await _current(analyses, case["needed"])}
        with meter.metered() as usage, trace.Trace(f"span {span}: {case['id']}", via="span",
                                                   tags=[span, case["id"]]) as recorded:
            out = await STEPS[span](analyses, case, question)
    calls = [{k: c.get(k) for k in ("agent", "in", "out", "decided", "finish")} for c in recorded.record["model calls"]]
    return {**out, "tokens": usage.total, "model calls": len(usage.calls), "calls": calls,
            "groq errors": [c.error for c in usage.rejected], "seconds": round(time.monotonic() - began, 1)}


def markdown(rows):
    lines = [f"# Span tests, {date.today().isoformat()}", ""]
    for span in SPANS:
        mine = [r for r in rows if r["span"] == span]
        if not mine:
            continue
        lines += [f"## {span}", "", "| case | rep | checks | tokens | output |", "|---|---|---|---|---|"]
        for r in mine:
            checks = "; ".join(f"{k}: {'yes' if v is True else '**no**' if v is False else v}"
                               for k, v in r["checks"].items() if v not in (None, [], ""))
            output = (r.get("profiles") or r.get("tasks") or r.get("claim") or (r.get("brief") or {}).get("answer")
                      or r.get("stopped") or r.get("decided"))
            lines.append(f"| {r['id']} | {r['rep']} | {checks} | {r['tokens']:,} | {str(output)[:400]} |")
        lines.append("")
    return "\n".join(lines)


def main(argv):
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "span")
    from evals import score as S
    from stats_desk.core import trace
    from stats_desk.core.governor import DailyLimitReached
    from evals.collect_runs import _daily_limit
    wanted = [a for a in argv if a in SPANS] or list(SPANS)
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    repeat = int(argv[argv.index("--repeat") + 1]) if "--repeat" in argv else 1
    data, (_, golden) = cases(), S.golden()
    out = HERE / "runs" / f"spans_{date.today().isoformat()}.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for span in wanted:
        for case in data["analyst" if span == "analyst" else span if span in data else "pick"]:
            if only and case["id"] not in only:
                continue
            for rep in range(1, repeat + 1):
                qid = case.get("question_of", case["id"])
                question = golden[qid]["question"] if qid in golden else case.get("question")
                try:
                    row = {"span": span, "id": case["id"], "rep": rep,
                           **asyncio.run(run_case(span, case, question))}
                except BaseException as error:
                    if _daily_limit(error, DailyLimitReached) is None:
                        raise
                    print(f"{span} {case['id']}: Groq's daily limit; stopping. Saved so far: {out}")
                    trace.flush()
                    return 1
                rows.append(row)
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                print(f"{span} {case['id']} #{rep}: {row['tokens']:,} tokens, "
                      + ", ".join(f"{k}={v}" for k, v in row["checks"].items() if isinstance(v, bool)), flush=True)
                trace.flush()
    report = HERE / "reports" / f"spans_{date.today().isoformat()}.md"
    report.write_text(markdown(rows), encoding="utf-8")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
