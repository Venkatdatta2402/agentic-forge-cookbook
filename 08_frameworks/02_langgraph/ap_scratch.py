"""Kestrel's accounts-payable desk, from this repo's own parts.

The same policy, prompts and model as `ap_graph.py`, but not the same design:

    the case                 `case.py`: chapter 6's shared board, every write a row in chapter 4's `Records`
    who may write what       the board's owners. The models write only what they produce, people write only
                             their own answers, and the money is the desk's code alone
    reading an email         01_foundations `extract()` -- schema mode first, validation behind it
    the investigator         06_orchestration `failures.attempt()` -- an Agent on chapter 2's loop that
                             finishes by filling in `payables.Recommendation`, and comes back as a failure
                             form if it raises, is stopped, or answers in prose
    its tools                03_tools `Tool`/`Registry`, built from the same functions LangGraph wraps
    writing to a vendor      01_foundations `chat()`
    retries                  chapter 1's client: 429s and 5xx, eight times with backoff. A step that still
                             fails leaves the case where it was, and `Desk.resume` runs that step again

The prompts and the policy are imported from `payables.py` by both builds.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _chapter in ("01_foundations", "02_agent_runtime", "03_tools", "06_orchestration"):
    if str(_REPO / _chapter) not in sys.path:
        sys.path.append(str(_REPO / _chapter))

import payables as P  # noqa: E402
from case import END, Desk, Wait  # noqa: E402

MODEL = "openai/gpt-oss-120b"

# Who alone may write each field. A field with no owner cannot be written at all (`case.CaseFile.set`).
FIELDS = {"key": "which inbox email this case is about",
          "email": "the email being read: the inbox's, then the vendor's reply",
          "invoice": "the invoice, as read from the email",
          "findings": "what the policy check found",
          "recommendation": "the investigator's proposal, for one invoice number",
          "amount": "what will be paid, which is not always what was billed",
          "signers": "who must sign, in order",
          "approvals": "who has signed or refused",
          "status": "how the case ended"}
OWNER = {"key": "desk", "email": "desk", "next": "desk", "waiting": "desk",
         "invoice": "reader",                     # a model
         "recommendation": "investigator",        # a model with tools
         "findings": "desk", "amount": "desk", "qty": "desk", "signers": "desk", "owner": "desk",
         "approvals": "desk", "outbox": "desk", "vendor_rounds": "desk", "status": "desk", "payment": "desk",
         "callback": "dev", "requester": "dev",   # the AP clerk's answers
         "vendor_reply": "vendor",
         **{f"approval:{person}": person for person in P.PEOPLE}}


def live_extract(email):
    from llm import extract
    return extract(P.EXTRACT_PROMPT.format(email=email), P.Invoice, model=MODEL).model_dump()


def live_draft(**fields):
    from llm import chat
    return chat([{"role": "user", "content": P.VENDOR_QUERY_PROMPT.format(**fields)}], model=MODEL)


def live_investigate(invoice, findings):
    """Chapter 6's form-filling agent, through `failures.attempt`, so a failure comes back as a form."""
    from failures import attempt
    from registry import Registry
    from supervisor import Agent
    from tools import Tool

    agent = Agent("investigator", P.INVESTIGATE_PROMPT.format(policy=P.POLICY),
                  tools=Registry([Tool(f) for f in P.INVESTIGATOR_TOOLS]), model=MODEL)
    found = "\n".join(f"- {f['kind']}: {f['detail']}" for f in findings)
    request = (f"Invoice {invoice['invoice_number']} from {invoice['vendor_name']} against "
               f"{invoice['po_number']} failed the match:\n{found}\n\nInvoice fields: {invoice}")
    result = attempt(agent, request, model=P.Recommendation)
    if not isinstance(result, P.Recommendation):
        # it raised, Runtime stopped it, or it answered in prose: not a recommendation, so a person looks
        return {"action": "hold", "amount": 0.0, "reason": f"{result.answer}. {result.unknowns}"[:300]}
    return result.model_dump()


def build_desk(path=":memory:", extract=None, investigate=None, draft=None):
    extract = extract or live_extract
    investigate = investigate or live_investigate
    draft = draft or live_draft

    # ------------------------------------------------------------ steps: each writes, then says what is next
    def read_invoice(case):
        case.set("reader", "invoice", extract(case.values["email"]))
        return "check_policy"

    def check_policy(case):
        invoice = case.values["invoice"]
        findings = [{"kind": k, "detail": d} for k, d in P.check(invoice)]
        case.set("desk", "findings", findings)
        kinds = {f["kind"] for f in findings}
        if not kinds:
            case.set("desk", "amount", invoice["total"])
            case.set("desk", "signers", P.approvers(invoice))
            return "request_approval" if case.values["signers"] else "pay"
        for kind, step in [("duplicate", "answer_duplicate"), ("bank", "verify_bank"), ("no_po", "ask_clerk"),
                           ("unknown_vendor", "hold"), ("wrong_po", "hold")]:
            if kind in kinds:
                return step
        return "investigate"                              # price or quantity

    def answer_duplicate(case):
        invoice, paid = case.values["invoice"], case.values["findings"][0]["detail"]
        send(case, invoice["sender_email"], f"Thank you for your reminder. {paid}. Please check your records.")
        return close(case, "duplicate")

    def verify_bank(case):
        vendor = P.VENDORS[P.find_vendor(case.values["invoice"])]
        return Wait("callback", "dev", then="after_callback", shown={
            "ask": "callback", "to": "dev",
            "question": f"Call {vendor['name']} on {vendor['phone']} (the number ON FILE, not one from "
                        f"the email) and confirm this change.",
            "findings": [f["detail"] for f in case.values["findings"]]})

    def after_callback(case):
        # confirmed or not, nothing is paid to the new account until the vendor master is updated
        return "hold" if case.values["callback"].get("verified") else close(case, "fraud")

    def ask_clerk(case):
        invoice = case.values["invoice"]
        if invoice["total"] > P.NON_PO_LIMIT:
            return "hold"
        return Wait("requester", "dev", then="after_clerk", shown={
            "ask": "clerk", "to": "dev",
            "question": f"Who ordered {invoice['invoice_number']} from {invoice['vendor_name']} "
                        f"({invoice['total']:,.2f}, no PO)?"})

    def after_clerk(case):
        invoice, owner = case.values["invoice"], case.values["requester"]["owner"]
        case.set("desk", "owner", owner)
        case.set("desk", "amount", invoice["total"])
        case.set("desk", "signers", P.approvers(invoice, owner=owner))
        return "request_approval"

    def investigate_step(case):
        invoice = case.values["invoice"]
        rec = investigate(invoice, case.values["findings"])
        # stamped with the invoice it is about, so a corrected invoice is never shown the old one's reasons
        case.set("investigator", "recommendation", {**rec, "invoice_number": invoice["invoice_number"]})
        if rec["action"] == "approve_partial":
            # the model proposes, the policy disposes -- and only the desk can write `amount`
            allowed, qty = P.partial_amount(invoice)
            case.set("desk", "amount", allowed)
            case.set("desk", "qty", qty)
            case.set("desk", "signers", P.approvers(invoice, allowed))
            return "request_approval" if case.values["signers"] else "pay"
        if rec["action"] == "query_vendor" and case.values.get("vendor_rounds", 0) < P.MAX_VENDOR_ROUNDS:
            return "query_vendor"
        return "hold"

    def query_vendor(case):
        invoice, rec = case.values["invoice"], case.values["recommendation"]
        send(case, invoice["sender_email"], draft(vendor=invoice["vendor_name"], invoice_number=invoice["invoice_number"],
                                                  po=invoice["po_number"], reason=rec["reason"]))
        case.set("desk", "vendor_rounds", case.values.get("vendor_rounds", 0) + 1)
        # the email is sent and this step is over; the reply starts `read_reply`, so nothing is sent twice
        return Wait("vendor_reply", "vendor", then="read_reply", shown={
            "ask": "vendor", "to": invoice["sender_email"], "question": "Waiting for a corrected invoice."})

    def read_reply(case):
        case.set("desk", "email", case.values["vendor_reply"])
        return "read_invoice"

    def request_approval(case):
        signer = _next_signer(case)
        return Wait(f"approval:{signer}", signer, then="record_approval",
                    shown={"ask": "approval", "approver": signer, "to": signer, "packet": _packet(case.values)})

    def record_approval(case):
        signer = _next_signer(case)
        reply = case.values[f"approval:{signer}"]
        signed = case.values.get("approvals", []) + [{"by": signer, "approved": bool(reply.get("approved")),
                                                      "note": reply.get("note", "")}]
        case.set("desk", "approvals", signed)
        if not signed[-1]["approved"]:
            return close(case, "refused")
        return "pay" if {a["by"] for a in signed} >= set(case.values["signers"]) else "request_approval"

    def pay(case):
        row = P.schedule_payment(case.values["key"], case.values["invoice"], case.values["amount"],
                                 case.values.get("qty"))
        case.set("desk", "payment", row["ref"])
        case.set("desk", "status", "scheduled")
        return END

    def hold(case):
        return close(case, "held")

    steps = {"read_invoice": read_invoice, "check_policy": check_policy, "answer_duplicate": answer_duplicate,
             "verify_bank": verify_bank, "after_callback": after_callback, "ask_clerk": ask_clerk,
             "after_clerk": after_clerk, "investigate": investigate_step, "query_vendor": query_vendor,
             "read_reply": read_reply, "request_approval": request_approval, "record_approval": record_approval,
             "pay": pay, "hold": hold}
    return Desk(steps, start="read_invoice", fields=FIELDS, owner=OWNER, path=path)


def send(case, to, body):
    case.set("desk", "outbox", case.values.get("outbox", []) + [{"to": to, "body": body}])


def close(case, status):
    """Every ending but a payment: nothing is paid."""
    case.set("desk", "amount", 0.0)
    case.set("desk", "status", status)
    return END


def _next_signer(case):
    signed = [a["by"] for a in case.values.get("approvals", [])]
    return next(s for s in case.values["signers"] if s not in signed)


def _packet(values):
    inv = values["invoice"]
    lines = [f"{P.VENDORS[P.find_vendor(inv)]['name']} {inv['invoice_number']} against "
             f"{inv.get('po_number') or 'no PO'}: billed {inv['total']:,.2f}, to pay {values['amount']:,.2f}."]
    rec = values.get("recommendation")
    if rec and rec["invoice_number"] == inv["invoice_number"]:
        lines.append(f"Investigation: {rec['reason']}")
    if values.get("owner"):
        lines.append(f"Non-PO spend; the AP clerk found the requester: {P.PEOPLE[values['owner']]}.")
    return " ".join(lines)


def settle(desk, key, run="", respond=P.human, max_answers=6):
    """Run one invoice to the end, answering every question from `payables.HUMANS`."""
    thread = f"invoice-{key}{run}"
    waiting = desk.open(thread, {"key": key, "email": P.INBOX[key], "vendor_rounds": 0})
    answered = 0
    while waiting and answered < max_answers:
        reply = respond(key, waiting["ask"], waiting.get("approver"))
        waiting = desk.answer(thread, waiting["party"], waiting["field"], reply)
        answered += 1
    return desk.state(thread)
