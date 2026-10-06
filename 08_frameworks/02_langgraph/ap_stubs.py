"""Stand-ins for every model call in the desk, so both builds can run without spending a token.

    extract        an email -> the fields `payables.GOLD` says it holds
    draft          a vendor email, from a template
    Scripted       the graph's investigator model: plays back the tool calls a sensible agent makes
    investigate    the scratch build's investigator: hands back the same recommendations
    RefusesOnce    Scripted, except its first try at invoice C is refused the way Groq refused it
    Forever        an investigator that never hands in
    forever_client chapter 2's model client, doing the same -- for the scratch build
"""

import json
from types import SimpleNamespace as NS

import httpx
from langchain_core.messages import AIMessage, HumanMessage

import payables as P

EMAILS = {P.INBOX[k]: P.GOLD[k] for k in P.INBOX} | {P.HUMANS[("C", "vendor")]: P.GOLD["C-reply"]}


def extract(email):
    return dict(EMAILS[email])


def draft(**fields):
    return f"Dear {fields['vendor']}, please send a corrected {fields['invoice_number']}. {fields['reason']}"


def call(name, args, n):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{n}", "type": "tool_call"}])


RECOMMENDATIONS = {
    "HM-5521": {"action": "approve_partial", "amount": 6525.0,
                "reason": "45 of 60 housings received (GRN-7750); 15 backordered."},
    "INV-88251": {"action": "query_vendor", "amount": 0.0,
                  "reason": "Billed 90.30 per LA-50; PO-4493 agreed 86.00."},
}

PLANS = {
    "HM-5521": [call("lookup_po", {"po_number": "PO-4485"}, 1), call("receipts", {"po_number": "PO-4485"}, 2),
                call("recommend", RECOMMENDATIONS["HM-5521"], 3)],
    "INV-88251": [call("lookup_po", {"po_number": "PO-4493"}, 1),
                  call("recommend", RECOMMENDATIONS["INV-88251"], 2)],
}


class Scripted:
    """A chat model that plays back a fixed list of replies per invoice, keyed by invoice number."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, *args, **kwargs):
        self.calls += 1
        asked = next(m.content for m in messages if isinstance(m, HumanMessage))
        plan = next(p for number, p in PLANS.items() if number in asked)
        return plan[sum(isinstance(m, AIMessage) for m in messages)]


def investigate(invoice, findings):
    return dict(RECOMMENDATIONS[invoice["invoice_number"]])


# the body Groq returned on notebook 1's first live run, word for word
REFUSED = {"error": {"message": "Tool call validation failed: tool call validation failed: attempted to call tool "
                                "'recomment' which was not in request.tools",
                     "type": "invalid_request_error", "code": "tool_use_failed",
                     "failed_generation": '{"name": "recomment", "arguments": {"action": "query_vendor", "amount": 0, '
                                          '"reason": "The invoice charges $90.30 each, which exceeds the PO price by '
                                          'more than the allowed 1% margin."}}'}}


class RefusesOnce(Scripted):
    def __init__(self):
        super().__init__()
        self.refused = False

    def invoke(self, messages, *args, **kwargs):
        from langchain_openai.chat_models.base import OpenAIInvalidRequestError
        if not self.refused and any("INV-88251" in str(m.content) for m in messages):
            self.refused = True
            self.calls += 1
            raise OpenAIInvalidRequestError(message=REFUSED["error"]["message"], body=REFUSED,
                                            response=httpx.Response(400, request=httpx.Request("POST", "https://groq")))
        return super().invoke(messages, *args, **kwargs)


class Forever(Scripted):
    def invoke(self, messages, *args, **kwargs):
        self.calls += 1
        return call("lookup_po", {"po_number": "PO-4485"}, self.calls)


def forever_client(counter):
    """A replacement for `llm.client.chat.completions.create`: the same tool call, forever."""
    def create(model=None, messages=None, tools=None, **kwargs):
        counter["n"] += 1
        if tools:
            arguments = json.dumps({"po_number": "PO-4485"})
            tool_call = NS(id=f"c{counter['n']}", type="function", function=NS(name="lookup_po", arguments=arguments))
            dumped = {"role": "assistant", "content": None, "tool_calls": [
                {"id": tool_call.id, "type": "function", "function": {"name": "lookup_po", "arguments": arguments}}]}
            message = NS(content=None, tool_calls=[tool_call], model_dump=lambda **k: dumped)
        else:
            message = NS(content="PO-4485 lists 60 housings at 145.00.", tool_calls=None,
                         model_dump=lambda **k: {"role": "assistant", "content": "PO-4485 lists 60 housings."})
        return NS(choices=[NS(message=message, finish_reason="stop")],
                  usage=NS(prompt_tokens=10, completion_tokens=1, total_tokens=11))
    return create
