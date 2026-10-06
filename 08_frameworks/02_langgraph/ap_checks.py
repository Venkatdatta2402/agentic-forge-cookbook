"""What LangGraph does that you have to know before production -- each one reproduced, offline.

Every function here runs against `ap_stubs`, costs nothing, and returns what it measured:

    double_send          a resumed node runs again from its first line -- and a case-file step does not
    reordered_answers    answers matched to questions by position -- and by field, in the case file
    wrong_person         an approval answered by somebody else: accepted by the graph, refused by the case file
    runaway_agent        the default step limit, and what each build does with an agent that never stops
    endless_vendor_loop  a loop through interrupt() is never stopped by recursion_limit
    flaky_provider       RetryPolicy recovering a 503
    refused_tool_call    the provider refusing a tool call: the run parked, then the fix
    misread_iban         rewinding a run to the step after reading, and correcting a field
    finish_elsewhere     resuming a paused thread from a different Python process
"""

import operator
import subprocess
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
import openai
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import ap_graph as G
import ap_scratch as A
import ap_stubs as S
import case as CF
import payables as P

HERE = Path(__file__).resolve().parent


def _desk(**overrides):
    parts = {"extract": S.extract, "model": S.Scripted(), "draft": S.draft} | overrides
    return G.build_desk(InMemorySaver(), **parts)


def double_send():
    """A node that sends an email and then waits for the reply, resumed once."""
    sent = []

    class Query(TypedDict, total=False):
        reply: str

    def query_and_wait(state):
        sent.append("Please send a corrected INV-88251.")       # the send ...
        return {"reply": interrupt("waiting for the vendor")}   # ... and the wait, in one node

    g = StateGraph(Query)
    g.add_node("query_and_wait", query_and_wait)
    g.add_edge(START, "query_and_wait")
    g.add_edge("query_and_wait", END)
    graph = g.compile(checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "double-send"}}
    graph.invoke({}, thread)
    while_waiting = len(sent)
    graph.invoke(Command(resume="corrected invoice attached"), thread)
    graph_sent = {"emails sent while waiting": while_waiting, "emails sent after the reply": len(sent)}

    # the same step in the case file: it sends, and its Wait ends it -- the reply starts `read_reply`
    sent.clear()

    def query(case):
        sent.append("Please send a corrected INV-88251.")
        return CF.Wait("vendor_reply", "vendor", then="read_reply")

    desk = CF.Desk({"query": query, "read_reply": lambda case: CF.END}, start="query", fields={},
                   owner={"next": "desk", "waiting": "desk", "vendor_reply": "vendor"})
    desk.open("double-send", {})
    while_waiting = len(sent)
    desk.answer("double-send", "vendor", "vendor_reply", "corrected invoice attached")
    return {"LangGraph": graph_sent,
            "case file": {"emails sent while waiting": while_waiting, "emails sent after the reply": len(sent)}}


def reordered_answers():
    """Tomasz approves; a deploy swaps the order of the two questions; Ines refuses."""
    class Signatures(TypedDict, total=False):
        record: dict

    def v1(state):
        tomasz = interrupt({"ask": "tomasz"})
        ines = interrupt({"ask": "ines"})
        return {"record": {"tomasz": tomasz, "ines": ines}}

    def v2(state):                                 # the same node after a tidy-up
        ines = interrupt({"ask": "ines"})
        tomasz = interrupt({"ask": "tomasz"})
        return {"record": {"tomasz": tomasz, "ines": ines}}

    saver = InMemorySaver()

    def deploy(node):
        g = StateGraph(Signatures)
        g.add_node("sign", node)
        g.add_edge(START, "sign")
        g.add_edge("sign", END)
        return g.compile(checkpointer=saver)

    thread = {"configurable": {"thread_id": "signatures"}}
    deploy(v1).invoke({}, thread)
    deploy(v1).invoke(Command(resume="approved"), thread)
    after = deploy(v2)
    after.invoke(Command(resume="REFUSED"), thread)
    graph_record = after.get_state(thread).values["record"]

    # the same story in the case file, where each answer is a field named for, and owned by, its approver
    def record(case):
        case.set("desk", "record", {"tomasz": case.values["approval:tomasz"], "ines": case.values["approval:ines"]})
        return CF.END

    v1 = {"first": lambda case: CF.Wait("approval:tomasz", "tomasz", then="second"),
          "second": lambda case: CF.Wait("approval:ines", "ines", then="record"), "record": record}
    v2 = {"first": lambda case: CF.Wait("approval:ines", "ines", then="second"),
          "second": lambda case: CF.Wait("approval:tomasz", "tomasz", then="record"), "record": record}
    owner = {"next": "desk", "waiting": "desk", "record": "desk", "approval:tomasz": "tomasz", "approval:ines": "ines"}
    desk = CF.Desk(v1, start="first", fields={}, owner=owner)
    desk.open("s", {})
    desk.answer("s", "tomasz", "approval:tomasz", "approved")
    desk.steps = v2                                  # the deploy: same records, new code
    desk.answer("s", "ines", "approval:ines", "REFUSED")
    return {"what happened": {"tomasz": "approved", "ines": "REFUSED"},
            "LangGraph recorded": graph_record, "case file recorded": desk.state("s")["record"]}


def wrong_person():
    """Invoice B waits for Tomasz's signature. Mira answers it -- in each build."""
    mira = {"approved": True, "by": "mira", "note": "Looks fine to me."}
    P.reset()
    graph = _desk()
    G.start(graph, "B")
    G.answer(graph, "B", mira)
    graph_said = P.outcome(graph.get_state(G.config("B")).values)

    P.reset()
    desk = A.build_desk(extract=S.extract, investigate=S.investigate, draft=S.draft)
    waiting = desk.open("invoice-B", {"key": "B", "email": P.INBOX["B"], "vendor_rounds": 0})
    try:
        desk.answer("invoice-B", "mira", waiting["field"], mira)
        refused = None
    except CF.Denied as denied:
        refused = str(denied)
    return {"waiting on": waiting["field"],
            "LangGraph, after Mira answered": graph_said,
            "case file, when Mira answered": f"Denied: {refused}",
            "case file, still waiting on": desk.waiting("invoice-B")["field"]}


def runaway_agent():
    """An investigator that calls lookup_po with the same argument forever, in both builds."""
    from langgraph._internal._config import DEFAULT_RECURSION_LIMIT
    import llm

    model = S.Forever()
    P.reset()
    graph = G.settle(_desk(model=model), "D")

    calls = {"n": 0}
    real = llm.client.chat.completions.create
    llm.client.chat.completions.create = S.forever_client(calls)
    try:
        P.reset()
        scratch = A.settle(A.build_desk(extract=S.extract, draft=S.draft), "D")
    finally:
        llm.client.chat.completions.create = real
    return {"LangGraph default recursion_limit": DEFAULT_RECURSION_LIMIT,
            "LangGraph, limit 17": {**P.outcome(graph), "model calls": model.calls, "why": graph["trail"][-2]},
            "scratch": {**P.outcome(scratch), "model calls": calls["n"],
                        "why": scratch["recommendation"]["reason"][:160]}}


def endless_vendor_loop(rounds=20):
    """A vendor who keeps sending the same wrong invoice, with the round counter taken away."""
    kept = P.MAX_VENDOR_ROUNDS
    P.MAX_VENDOR_ROUNDS = 1_000
    try:
        P.reset()
        desk = _desk(extract=lambda email: dict(P.GOLD["C"]))
        G.start(desk, "C")
        for _ in range(rounds):
            G.answer(desk, "C", "the same wrong invoice again")
        state = desk.get_state(G.config("C")).values
        return {"vendor rounds": state["vendor_rounds"], "emails sent": len(state["outbox"]),
                "still waiting for the vendor": bool(G.waiting_on(desk, G.config("C")))}
    finally:
        P.MAX_VENDOR_ROUNDS = kept


def flaky_provider():
    """Reading invoice A, where the provider answers the first attempt with a 503."""
    attempts = {"n": 0}

    def extract(email):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise openai.InternalServerError("503 from the provider", body=None,
                                             response=httpx.Response(503, request=httpx.Request("POST", "https://groq")))
        return S.extract(email)

    P.reset()
    state = G.settle(_desk(extract=extract), "A")
    return {**P.outcome(state), "attempts at reading": attempts["n"]}


def refused_tool_call():
    """Groq refusing the investigator's call to 'recomment': without the fix, then with it."""
    kept = G.MAX_REJECTIONS
    G.MAX_REJECTIONS = 0                             # as on the first live run: not handled
    try:
        P.reset()
        desk = _desk(model=S.RefusesOnce())
        try:
            G.start(desk, "C")
            ended = None
        except openai.BadRequestError as refused:
            ended = type(refused).__name__
        parked_at = desk.get_state(G.config("C")).next
    finally:
        G.MAX_REJECTIONS = kept
    desk.invoke(None, G.config("C"))                 # carry on from the last good checkpoint
    resumed = [a["ask"] for a in G.waiting_on(desk, G.config("C"))]

    told = G.build_investigator(S.RefusesOnce()).invoke({"messages": [
        ("user", "Invoice INV-88251 from Brandt Optics against PO-4493 failed the match: price 90.30 against 86.00.")]})
    conversation = [(type(m).__name__, (m.content or " ".join(f"{c['name']}({c['args']})" for c in m.tool_calls))[:140])
                    for m in told["messages"]]
    return {"without the fix, the run ended with": ended, "thread parked before": parked_at,
            "invoke(None) then reached": resumed, "with the fix, the investigator's conversation": conversation,
            "recommendation": told["recommendation"]["action"]}


def misread_iban():
    """Invoice A read with its IBAN's last two digits swapped; the clerk rewinds and fixes the field."""
    def misread(email):
        invoice = S.extract(email)
        if invoice["invoice_number"] == "INV-88240":
            invoice["remit_to"] = "IBAN DE44 5001 0517 5407 3249 13"     # 31 -> 13
        return invoice

    P.reset()
    desk = _desk(extract=misread)
    waiting = [a["ask"] for a in G.start(desk, "A")]
    because = desk.get_state(G.config("A")).values["findings"][0]["detail"]
    history = list(desk.get_state_history(G.config("A")))
    steps = [(s.metadata["step"], s.next) for s in reversed(history)]

    after_reading = next(s for s in history if s.next == ("check_policy",))
    fixed = dict(after_reading.values["invoice"], remit_to="IBAN DE44 5001 0517 5407 3249 31")
    branch = desk.update_state(after_reading.config, {"invoice": fixed, "trail": ["clerk corrected a misread IBAN"]})
    desk.invoke(None, branch)
    final = desk.get_state(G.config("A")).values
    return {"waiting on": waiting, "because": because, "steps before the fix": steps,
            "after the fix": P.outcome(final), "trail": final["trail"],
            "checkpoints in the thread": len(list(desk.get_state_history(G.config("A"))))}


CHILD = r'''
import os, sqlite3, sys
from langgraph.checkpoint.sqlite import SqliteSaver
import ap_graph as G, payables as P

desk = G.build_desk(SqliteSaver(sqlite3.connect(sys.argv[1], check_same_thread=False)))
key = sys.argv[2]
asks = G.waiting_on(desk, G.config(key))
print(f"pid {os.getpid()} found invoice-{key} waiting on: {asks[0]['to']} ({asks[0]['ask']})")
G.answer(desk, key, {"approved": True, "by": asks[0]["approver"], "note": "Checked against the PO."})
print(f"pid {os.getpid()} after answering:", P.outcome(desk.get_state(G.config(key)).values))
print(f"pid {os.getpid()} payments in its ledger:", [(p["invoice"], p["amount"]) for p in P.PAYMENTS])
'''


def finish_elsewhere(db_path, key):
    """Answer a paused invoice's approval from a separate Python process, sharing only the SQLite file."""
    child = subprocess.run([sys.executable, "-W", "ignore", "-c", CHILD, str(db_path), key], cwd=HERE,
                           capture_output=True, text=True, encoding="utf-8")
    return child.stdout or child.stderr[-2000:]
