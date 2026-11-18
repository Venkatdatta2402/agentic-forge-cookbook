"""What a question's progress stream says: the agents' events (core/events.py), each as
`update(event)`, a small dict for the page:

    who     lead | f1, f2... (an analyst) | code (code's checks) | desk
    kind    thinking | thought | waiting           a model call starts, ends (tokens), waits on
                                                   Groq's per-minute limit
            call | result                          an analysis on the analysis server (mcp: true),
                                                   or the lead's own tools; its result (refused?)
            plan                                   the lead's plan: approach, later tasks
            task | finding                         the lead hands a task out; its finding comes in
            check                                  code reruns a finding's analysis elsewhere
    step    the plain line for a reader, or None, e.g.

        The lead is reading profiles: UR_annual_2025, LFPR_annual_2025
        Analyst 1 started: compare urban young women's and men's unemployment
        Analyst 1: gap on ur_annual by gender (15-29 years)
        Analyst 1 finished: Young urban women's unemployment is 7.1 points above young men's (1 warning)
        The lead is writing the brief

Finding ids (f1) stay in `who` and `finding`, for the page to match things up; the lines say
"Analyst 1" (the ids are for the backend). Events with nothing to show a
reader (a recovered call) give None.
"""

import json

from ..agents.agent import FORM
from ..core.events import label

LEAD_TOOLS = {FORM, "ask_analysts"}          # the lead's own; every other tool is an analysis


def _args(event):
    try:
        args = json.loads(event.get("arguments") or "{}") if isinstance(event.get("arguments"), str) \
            else event.get("arguments") or {}
        return args if isinstance(args, dict) else {}
    except ValueError:
        return {}


def who_of(agent: str) -> str:
    if agent.startswith("analyst "):
        return agent.removeprefix("analyst ")
    return "lead" if agent.startswith("lead") else "desk"


def analyst(fid: str) -> str:
    return f"Analyst {fid.removeprefix('f')}" if fid[1:].isdigit() else fid


def step(event) -> str | None:
    kind, agent = event.get("kind"), event.get("agent") or ""
    who = who_of(agent)
    if kind == "task":
        return f"{analyst(event['finding'])} started: {event['task']}"
    if kind == "finding":
        name = analyst(event["finding"])
        if event.get("stopped"):
            return f"{name} could not finish: {event['stopped']}"
        warnings = event.get("warnings") or 0
        return (f"{name} finished: {event['claim']}"
                + (f" ({warnings} warning{'s' if warnings > 1 else ''})" if warnings else ""))
    if kind == "plan":
        later = event.get("later") or []
        return f"The lead's plan: {event.get('approach', '')}" + (f" (later: {'; '.join(later)})" if later else "")
    if kind == "check":
        return f"Code reruns {analyst(event['finding']).replace('Analyst', 'finding')}'s {event.get('analysis')} in another period"
    if kind == "code_call" and event.get("tool") == "tables":
        return "Listing the published tables for the lead"
    if kind == "waiting":
        seconds = event.get("seconds")
        whom = "The lead" if who == "lead" else analyst(who)
        return f"{whom} waits{f' {seconds} s' if seconds else ''} for room in Groq's per-minute limit"
    if kind == "tool_call":
        tool, args = event.get("tool"), _args(event)
        if who == "lead":
            if tool == "profile":
                return f"The lead is reading profiles: {', '.join(args.get('names') or [])}"
            if tool == FORM:
                return "The lead is writing the brief"
            return None                                  # ask_analysts: each task says it started
        if who != "desk" and tool != FORM:
            return f"{analyst(who)}: {label(tool, args)}"
        return None
    if kind == "tool_result" and event.get("result_kind") not in (None, "ok"):
        if event.get("tool") == FORM:                    # code's checks sent the form back
            return ("The brief was sent back to fix" if who == "lead"
                    else f"{analyst(who)}'s finding was sent back to fix") + _first_line(event)
        if who not in ("lead", "desk"):
            return f"{analyst(who)}: {event.get('tool')} was refused; trying another way"
    return None


def _first_line(event) -> str:
    line = (event.get("preview") or "").strip().split("\n")[0][:200]
    return f": {line}" if line else ""


def update(event) -> dict | None:
    """The event as the page gets it (see the top of this file), or None if it shows nothing."""
    kind = event.get("kind")
    who = who_of(event.get("agent") or "")
    out = {"who": who, "kind": kind, "step": step(event)}
    if kind in ("thinking", "thought", "waiting"):
        if who == "desk":
            return None
        out |= {k: event[k] for k in ("tokens", "seconds", "error") if k in event}
    elif kind == "tool_call":
        tool, args = event.get("tool"), _args(event)
        if tool == "ask_analysts":
            return None                                  # each task comes as its own event
        out |= {"kind": "call", "tool": tool, "mcp": tool not in LEAD_TOOLS,
                "label": label(tool, args) if tool not in LEAD_TOOLS else tool}
        if tool == "profile":
            out["label"] = f"profile: {', '.join(args.get('names') or [])}"
    elif kind == "tool_result":
        tool, refused = event.get("tool"), event.get("result_kind") not in (None, "ok")
        if tool == "ask_analysts" or (tool == FORM and not refused):
            return None
        out |= {"kind": "result", "tool": tool, "mcp": tool not in LEAD_TOOLS, "refused": refused}
        if refused:
            out["message"] = (event.get("preview") or "")[:300]
    elif kind == "code_call":
        out |= {"who": "code", "kind": "call", "tool": event.get("tool"), "mcp": True, "label": event.get("tool")}
    elif kind == "check":
        out |= {"who": "code", "kind": "call", "tool": "recheck", "mcp": True, "finding": event.get("finding"),
                "label": f"recheck: {label(event.get('analysis') or '', event.get('arguments'))}"}
    elif kind == "plan":
        out |= {"who": "lead", "approach": event.get("approach", ""), "later": event.get("later") or []}
    elif kind == "task":
        out |= {"finding": event["finding"], "task": event["task"]}
    elif kind == "finding":
        out |= {k: event[k] for k in ("finding", "claim", "stopped", "flags", "figures", "elsewhere") if k in event}
    else:
        return None
    return out
