"""Kestrel's accounts-payable desk as a LangGraph graph.

Two graphs, one inside the other:

    desk            the workflow. Mostly code: read, check, route, wait for people, pay.
    investigator    an agent. A model with tools, going round until it hands in a
                    `Recommendation`. The desk calls it for any invoice that fails the match.

The division of labour is the one `payables.py` states: a model reads, investigates and writes;
the policy decides. Every conditional edge below is a plain function over the state, and none of
them asks a model anything.

Nothing that calls a model is created here at import time. `build_desk(extract=, model=,
draft=)` takes them as arguments, so the notebook can run the whole graph against stubs first
and spend tokens only once the wiring is known to be right.
"""

import operator
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, RetryPolicy, interrupt

import payables as P

_REPO = Path(__file__).resolve().parents[2]
load_dotenv(_REPO / ".env")
# one thing is borrowed from this repo: chapter 1's `tool_call_failure`, see `build_investigator`
if str(_REPO / "01_foundations") not in sys.path:
    sys.path.append(str(_REPO / "01_foundations"))

MODEL = "openai/gpt-oss-120b"
GROQ = "https://api.groq.com/openai/v1"


def chat_model(max_retries=8):
    """The same model the rest of the repo uses, through LangChain's OpenAI class pointed at Groq.

    `max_retries` is the SDK's own retry on 429/5xx, set to what `llm.py` uses so that both
    builds in notebook 1 retry alike -- Groq's free tier allows 8,000 tokens a minute, and reading
    seven emails in a row goes past it. The graph has a second layer, `RetryPolicy` on the nodes
    that call a model, and notebook 1 multiplies the two.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=MODEL, base_url=GROQ, api_key=os.environ["GROQ_API_KEY"],
                      temperature=0, max_retries=max_retries)


def live_extract(model):
    structured = model.with_structured_output(P.Invoice, method="json_schema")
    return lambda email: structured.invoke(P.EXTRACT_PROMPT.format(email=email)).model_dump()


def live_draft(model):
    return lambda **fields: model.invoke(P.VENDOR_QUERY_PROMPT.format(**fields)).content


def _transient(error):
    """What is worth retrying: the network and the provider's bad moments, not our own bugs."""
    import openai
    return isinstance(error, (openai.APIConnectionError, openai.APITimeoutError,
                              openai.RateLimitError, openai.InternalServerError))


RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, retry_on=_transient)

# The investigator's step budget. LangGraph's own default is 10,007 steps -- measured in notebook 1,
# an agent that never hands in ran 5,004 model calls before it tripped -- so every agent loop gets
# a limit of its own. 17 is eight turns of agent -> tools and one `finish`, which is what chapter
# 6's `Agent(max_iterations=24)` allows its loop of three components.
INVESTIGATOR_STEPS = 17


# --------------------------------------------------------------------------- the investigator

class Investigation(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    recommendation: dict
    nudged: bool
    rejected: int                 # tool calls the provider refused, and the model was told about

MAX_REJECTIONS = 3


def build_investigator(model):
    """A tool-using agent as a graph: `agent` -> `tools` -> `agent` ... -> `finish`.

    The loop is an edge that points backwards, which is the whole of what makes this an agent
    rather than a pipeline. It ends when the model calls `recommend`, a tool whose arguments ARE
    the answer -- chapter 6's `submit` form, rebuilt here as routing: nothing executes it, the
    router sees its name and leaves the loop.
    """
    tools = [StructuredTool.from_function(f) for f in P.INVESTIGATOR_TOOLS]
    recommend = StructuredTool.from_function(
        lambda **kw: "Recommendation recorded.", name="recommend", args_schema=P.Recommendation,
        description="Hand in your recommendation. Call this exactly once, when you are done.")
    bound = model.bind_tools(tools + [recommend])

    def agent(state):
        # Groq validates a tool call before returning it, and refuses one naming a tool that does
        # not exist with a 400. The first live run of notebook 1 met exactly that:
        #     attempted to call tool 'recomment' which was not in request.tools
        # LangChain raises it, `RetryPolicy` rightly will not retry a 400, and nothing in LangGraph's
        # agent pieces turns it into something the model can correct -- so it ended the invoice's
        # run. Chapter 1's `tool_call_failure()` does exactly that, and is borrowed here: the
        # refusal goes back into the conversation and the loop goes round again.
        import openai
        try:
            return {"messages": [bound.invoke(state["messages"])]}
        except openai.BadRequestError as refused:
            from llm import tool_call_failure
            told = tool_call_failure(refused)
            if told is None or state.get("rejected", 0) >= MAX_REJECTIONS:
                raise
            return {"messages": [HumanMessage(told)], "rejected": state.get("rejected", 0) + 1}

    def nudge(state):
        # it answered in prose. One reminder, then give up and hold -- chapter 6 learned that an
        # agent which will not use the form usually will not the second time either
        return {"messages": [HumanMessage("Call `recommend` with your recommendation. Do not answer as text.")],
                "nudged": True}

    def finish(state):
        last = state["messages"][-1]
        call = next((c for c in getattr(last, "tool_calls", []) if c["name"] == "recommend"), None)
        if call is None:
            return {"recommendation": {"action": "hold", "amount": 0.0,
                                       "reason": "The investigator did not hand in a recommendation."}}
        return {"recommendation": P.Recommendation(**call["args"]).model_dump()}

    def route(state):
        last = state["messages"][-1]
        if isinstance(last, HumanMessage):
            return "agent"                # the provider refused its call; it has been told why
        calls = getattr(last, "tool_calls", None) or []
        if any(c["name"] == "recommend" for c in calls):
            return "finish"
        if calls:
            return "tools"
        return "finish" if state.get("nudged") else "nudge"

    graph = StateGraph(Investigation)
    graph.add_node("agent", agent, retry_policy=RETRY)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("nudge", nudge)
    graph.add_node("finish", finish)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["agent", "tools", "nudge", "finish"])
    graph.add_edge("tools", "agent")
    graph.add_edge("nudge", "agent")
    graph.add_edge("finish", END)
    return graph.compile()


# --------------------------------------------------------------------------- the desk

class Case(TypedDict, total=False):
    key: str                      # which inbox email this thread is about
    email: str                    # the email being read -- replaced by the vendor's reply
    invoice: dict
    findings: list                # [{"kind": ..., "detail": ...}] from payables.check
    recommendation: dict
    amount: float                 # what will be paid, which is not always what was billed
    qty: float
    owner: str                    # a non-PO invoice's signer, as the clerk found them
    signers: list
    approvals: Annotated[list, operator.add]
    vendor_rounds: int
    outbox: Annotated[list, operator.add]
    trail: Annotated[list, operator.add]
    status: str


def _packet(state):
    """What an approver is shown: enough to sign or refuse without opening anything else."""
    inv = state["invoice"]
    lines = [f"{P.VENDORS[P.find_vendor(inv)]['name']} {inv['invoice_number']} against "
             f"{inv.get('po_number') or 'no PO'}: billed {inv['total']:,.2f}, to pay {state['amount']:,.2f}."]
    if state.get("recommendation"):
        lines.append(f"Investigation: {state['recommendation']['reason']}")
    if state.get("owner"):
        lines.append(f"Non-PO spend; the AP clerk found the requester: {P.PEOPLE[state['owner']]}.")
    return " ".join(lines)


def build_desk(checkpointer=None, extract=None, model=None, draft=None):
    """The desk, compiled. `extract`, `model` and `draft` default to the live model."""
    if model is None and (extract is None or draft is None):
        model = chat_model()
    extract = extract or live_extract(model)
    draft = draft or live_draft(model)
    investigator = build_investigator(model) if model is not None else None

    # ------------------------------------------------------------ nodes: each returns an update
    def read_invoice(state):
        invoice = extract(state["email"])
        # a vendor's corrected invoice comes back through here, and the last round's
        # investigation was about the invoice it replaces -- it must not reach the approver
        return {"invoice": invoice, "recommendation": None,
                "trail": [f"read {invoice['invoice_number']} from {invoice['vendor_name']}"]}

    def check_policy(state):
        invoice = state["invoice"]
        findings = [{"kind": k, "detail": d} for k, d in P.check(invoice)]
        update = {"findings": findings,
                  "trail": [f"check: {', '.join(f['kind'] for f in findings) or 'matched'}"]}
        if not findings:
            update |= {"amount": invoice["total"], "signers": P.approvers(invoice)}
        return update

    def answer_duplicate(state):
        paid = state["findings"][0]["detail"]
        return {"outbox": [{"to": state["invoice"]["sender_email"],
                            "body": f"Thank you for your reminder. {paid}. Please check your records."}],
                "status": "duplicate", "amount": 0.0, "trail": ["duplicate: replied with the payment"]}

    def verify_bank(state):
        vendor = P.VENDORS[P.find_vendor(state["invoice"])]
        answer = interrupt({"ask": "callback", "to": "dev",
                            "question": f"Call {vendor['name']} on {vendor['phone']} (the number ON FILE, "
                                        f"not one from the email) and confirm this change.",
                            "findings": [f["detail"] for f in state["findings"]]})
        if not answer.get("verified"):
            # routing from inside a node, with Command, because a person just decided it
            return Command(goto=END, update={"status": "fraud", "amount": 0.0,
                                             "trail": [f"bank change refused: {answer.get('note', '')}"]})
        return Command(goto="hold", update={"trail": ["bank change confirmed by phone; vendor master "
                                                      "must be updated before this is paid"]})

    def ask_clerk(state):
        invoice = state["invoice"]
        if invoice["total"] > P.NON_PO_LIMIT:
            return Command(goto="hold", update={"trail": ["no PO and over the non-PO limit"]})
        answer = interrupt({"ask": "clerk", "to": "dev",
                            "question": f"Who ordered {invoice['invoice_number']} from {invoice['vendor_name']} "
                                        f"({invoice['total']:,.2f}, no PO)?"})
        return Command(goto="request_approval",
                       update={"owner": answer["owner"], "amount": invoice["total"],
                               "signers": P.approvers(invoice, owner=answer["owner"]),
                               "trail": [f"clerk: {answer.get('note', '')}"]})

    def investigate(state):
        invoice = state["invoice"]
        found = "\n".join(f"- {f['kind']}: {f['detail']}" for f in state["findings"])
        try:
            result = investigator.invoke({"messages": [
                SystemMessage(P.INVESTIGATE_PROMPT.format(policy=P.POLICY)),
                HumanMessage(f"Invoice {invoice['invoice_number']} from {invoice['vendor_name']} against "
                             f"{invoice['po_number']} failed the match:\n{found}\n\nInvoice fields: {invoice}")]},
                {"recursion_limit": INVESTIGATOR_STEPS})
            rec = result["recommendation"]
        except GraphRecursionError:
            rec = {"action": "hold", "amount": 0.0,
                   "reason": f"The investigator did not finish within {INVESTIGATOR_STEPS} steps."}
        update = {"recommendation": rec, "trail": [f"investigator: {rec['action']} -- {rec['reason']}"]}
        if rec["action"] == "approve_partial":
            # the model proposes, the policy disposes: the amount is recomputed, never trusted
            allowed, qty = P.partial_amount(invoice)
            if abs(rec["amount"] - allowed) > 0.01:
                update["trail"].append(f"investigator proposed {rec['amount']:,.2f}; policy allows {allowed:,.2f}")
            update |= {"amount": allowed, "qty": qty, "signers": P.approvers(invoice, allowed)}
        return update

    def query_vendor(state):
        invoice, rec = state["invoice"], state["recommendation"]
        body = draft(vendor=invoice["vendor_name"], invoice_number=invoice["invoice_number"],
                     po=invoice["po_number"], reason=rec["reason"])
        return {"outbox": [{"to": invoice["sender_email"], "body": body}],
                "vendor_rounds": state.get("vendor_rounds", 0) + 1,
                "trail": [f"wrote to {invoice['sender_email']}"]}

    def await_vendor(state):
        # Its own node, separate from the one that sends. A node that is resumed runs again from
        # its first line, so a send placed before this interrupt would go out twice.
        reply = interrupt({"ask": "vendor", "to": state["invoice"]["sender_email"],
                           "question": "Waiting for a corrected invoice."})
        return {"email": reply, "trail": ["vendor replied"]}

    def request_approval(state):
        signed = [a["by"] for a in state.get("approvals", [])]
        signer = next(s for s in state["signers"] if s not in signed)
        answer = interrupt({"ask": "approval", "approver": signer, "to": signer, "packet": _packet(state)})
        return {"approvals": [{"by": signer, "approved": bool(answer.get("approved")),
                               "note": answer.get("note", "")}],
                "trail": [f"{signer} {'approved' if answer.get('approved') else 'refused'}"]}

    def pay(state):
        row = P.schedule_payment(state["key"], state["invoice"], state["amount"], state.get("qty"))
        return {"status": "scheduled", "trail": [f"payment {row['ref']} scheduled: {row['amount']:,.2f}"]}

    def hold(state):
        return {"status": "held", "amount": 0.0, "trail": ["held for a person"]}

    def refused(state):
        return {"status": "refused", "amount": 0.0}

    # ------------------------------------------------------------ routing: code, never a model
    def after_check(state):
        kinds = {f["kind"] for f in state["findings"]}
        if not kinds:
            return "request_approval" if state["signers"] else "pay"
        for kind, node in [("duplicate", "answer_duplicate"), ("bank", "verify_bank"),
                           ("no_po", "ask_clerk"), ("unknown_vendor", "hold"), ("wrong_po", "hold")]:
            if kind in kinds:
                return node
        return "investigate"          # price or quantity

    def after_investigation(state):
        action = state["recommendation"]["action"]
        if action == "query_vendor":
            return "query_vendor" if state.get("vendor_rounds", 0) < P.MAX_VENDOR_ROUNDS else "hold"
        if action == "approve_partial":
            return "request_approval" if state["signers"] else "pay"
        return "hold"

    def after_approval(state):
        if not state["approvals"][-1]["approved"]:
            return "refused"
        signed = {a["by"] for a in state["approvals"] if a["approved"]}
        return "pay" if signed >= set(state["signers"]) else "request_approval"

    graph = StateGraph(Case)
    graph.add_node("read_invoice", read_invoice, retry_policy=RETRY)
    graph.add_node("check_policy", check_policy)
    graph.add_node("answer_duplicate", answer_duplicate)
    graph.add_node("verify_bank", verify_bank, destinations=("hold", END))
    graph.add_node("ask_clerk", ask_clerk, destinations=("request_approval", "hold"))
    graph.add_node("investigate", investigate)
    graph.add_node("query_vendor", query_vendor, retry_policy=RETRY)
    graph.add_node("await_vendor", await_vendor)
    graph.add_node("request_approval", request_approval)
    graph.add_node("pay", pay)
    graph.add_node("hold", hold)
    graph.add_node("refused", refused)

    graph.add_edge(START, "read_invoice")
    graph.add_edge("read_invoice", "check_policy")
    graph.add_conditional_edges("check_policy", after_check,
                                ["request_approval", "pay", "answer_duplicate", "verify_bank",
                                 "ask_clerk", "hold", "investigate"])
    graph.add_conditional_edges("investigate", after_investigation,
                                ["query_vendor", "request_approval", "pay", "hold"])
    graph.add_edge("query_vendor", "await_vendor")
    graph.add_edge("await_vendor", "read_invoice")          # the vendor loop: read the reply, check again
    graph.add_conditional_edges("request_approval", after_approval, ["pay", "request_approval", "refused"])
    for node in ("answer_duplicate", "pay", "hold", "refused"):
        graph.add_edge(node, END)
    return graph.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- driving it

def config(key, run=""):
    return {"configurable": {"thread_id": f"invoice-{key}{run}"}}


def waiting_on(graph, cfg):
    """The questions a paused thread is waiting on, or [] if it is not paused."""
    return [i.value for i in graph.get_state(cfg).interrupts]


def start(graph, key, run=""):
    """Begin one invoice. Returns what it is now waiting for, or [] if it finished."""
    graph.invoke({"key": key, "email": P.INBOX[key], "vendor_rounds": 0}, config(key, run))
    return waiting_on(graph, config(key, run))


def answer(graph, key, reply, run=""):
    """Resume a paused invoice with a person's (or a vendor's) reply."""
    graph.invoke(Command(resume=reply), config(key, run))
    return waiting_on(graph, config(key, run))


def settle(graph, key, run="", respond=P.human, max_answers=6):
    """Run one invoice to the end, answering every question from `payables.HUMANS`."""
    asks = start(graph, key, run)
    answered = 0
    while asks and answered < max_answers:
        ask = asks[0]
        asks = answer(graph, key, respond(key, ask["ask"], ask.get("approver")), run)
        answered += 1
    return graph.get_state(config(key, run)).values
