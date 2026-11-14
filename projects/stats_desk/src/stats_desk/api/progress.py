"""What a question's progress stream says: the agents' events (core/events.py) as one plain line
each, for someone watching the page, e.g.

    The lead is reading profiles: UR_annual_2025, LFPR_annual_2025
    f1 started: compare urban young women's and men's unemployment
    f1: gap on ur_annual
    f1 finished: Young urban women's unemployment is 7.1 points above young men's (1 warning)
    The lead is writing the brief

Events with nothing to show a reader (a result's full text, a recovered call) give None.
"""

import json

from ..agents.agent import FORM


def _args(event):
    try:
        args = json.loads(event.get("arguments") or "{}")
        return args if isinstance(args, dict) else {}
    except ValueError:
        return {}


def step(event) -> str | None:
    kind, agent = event.get("kind"), event.get("agent") or ""
    who = agent.removeprefix("analyst ")              # "f1"
    if kind == "task":
        return f"{event['finding']} started: {event['task']}"
    if kind == "finding":
        if event.get("stopped"):
            return f"{event['finding']} could not finish: {event['stopped']}"
        warnings = event.get("warnings") or 0
        return (f"{event['finding']} finished: {event['claim']}"
                + (f" ({warnings} warning{'s' if warnings > 1 else ''})" if warnings else ""))
    if kind == "tool_call":
        tool, args = event.get("tool"), _args(event)
        if agent == "lead":
            if tool == "profile":
                return f"The lead is reading profiles: {', '.join(args.get('names') or [])}"
            if tool == FORM:
                return "The lead is writing the brief"
            return None                                  # ask_analysts: each task says it started
        if agent.startswith("analyst ") and tool != FORM:
            table = args.get("table") or args.get("x_table") or ""
            return f"{who}: {tool}" + (f" on {table}" if isinstance(table, str) and table else "")
        return None
    if kind == "tool_result" and event.get("result_kind") not in (None, "ok") and agent.startswith("analyst "):
        return f"{who}: {event.get('tool')} was refused; trying another way"
    return None
