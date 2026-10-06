"""Meridian Travel Insurance: the policy, the customers, and what the claims assistant should say.

    WORDING         the policy wording, as numbered clauses -- what retrieval searches
    LIMITS, EXCESS  each plan's money, as data the payout calculator reads
    POLICIES        who holds what: plan, trip, add-ons, medical conditions declared and accepted
    CLAIMS          claims already open or settled
    lookup_policy, claims_for, claim_status, estimate_payout
                    plain functions; each build wraps them as its own kind of tool
    Lookups, account_facts
                    what the model extracts from a question, and the code that turns it into lookups
    QUESTIONS       eight messages from three policyholders, in order -- two of them follow-ups
    EXPECTED        what a correct reply says: covered or not, the payout, the clauses it rests on

The calculator applies limits and excess and nothing else. Whether something is covered at all
is a reading of the wording -- which is the assistant's job, and the reason it needs retrieval.

Everything here is invented.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

TODAY = "2026-10-03"

WORDING = {
    "1.1": ("Definitions", "Close relative: your spouse or partner, parent, child, sibling, grandparent or grandchild."),
    "1.2": ("Definitions", "Pre-existing medical condition: any illness or injury you were diagnosed with or treated "
                           "for in the 12 months before you bought this policy."),
    "1.3": ("Definitions", "Unattended: when you are not in full view of your belongings and not in a position to stop "
                           "them being taken. Belongings locked in a hotel room safe, or in your locked accommodation, "
                           "are not unattended."),
    "1.4": ("Definitions", "Valuables: laptops, tablets, phones, cameras, jewellery and watches."),
    "2.1": ("Cancellation", "We pay your unused, non-refundable travel and accommodation costs, up to the cancellation "
                            "limit for your plan, if you have to cancel your trip because you, a travelling companion or "
                            "a close relative die, are seriously injured or fall seriously ill; or because you are "
                            "called for jury service."),
    "2.2": ("Cancellation", "We do not pay if you cancel because you change your mind, or because of a reason you knew "
                            "about when you bought the policy or booked the trip."),
    "3.1": ("Travel delay", "If your outbound or return departure is delayed by strike, bad weather or mechanical "
                            "breakdown, we pay the travel delay benefit for your plan for each full 6 hours of delay, up "
                            "to the plan maximum. You must get written confirmation of the delay from the carrier. "
                            "Travel delay is not available on the Essential plan."),
    "3.2": ("Travel delay", "We only pay travel delay once for the same delay, whoever claims it."),
    "4.1": ("Medical", "We pay the cost of emergency medical treatment abroad, including hospital stays, up to the "
                       "medical limit for your plan."),
    "4.2": ("Medical", "Pre-existing medical conditions are covered only if you declared them and we accepted them. "
                       "Accepted conditions are listed on your policy schedule."),
    "4.3": ("Medical", "Injuries from winter sports are covered only if your policy schedule shows the Winter Sports "
                       "add-on."),
    "5.1": ("Baggage", "We pay for your belongings lost, stolen or damaged during your trip, up to the baggage limit "
                       "for your plan. No single item, pair or set is paid above the single-item limit; valuables are "
                       "single items."),
    "5.2": ("Baggage", "We do not pay for belongings left unattended, or for theft not reported to the police within "
                       "24 hours. A police report is required for every theft claim."),
    "6.1": ("General exclusions", "We do not pay claims arising from your being under the influence of alcohol or "
                                  "drugs, unless the drugs were prescribed to you."),
    "6.2": ("General exclusions", "We do not cover skiing, snowboarding or any other winter sport unless your policy "
                                  "schedule shows the Winter Sports add-on, or skydiving, motor racing or mountaineering "
                                  "above 4,000 metres under any plan."),
    "7.1": ("Excess", "An excess is deducted from each claim, per person, per section of cover. The excess for your plan "
                      "is in the limits table. No excess applies to travel delay."),
    "8.1": ("Claims", "Tell us about a claim within 31 days of returning home."),
    "8.2": ("Claims", "Send us original receipts and invoices; for cancellation, the cancellation confirmation and a "
                      "medical certificate from the treating doctor; for theft, the police report."),
    "8.3": ("Claims", "When we ask you for a document, you have 30 days from the date of our request to send it. After "
                      "that we may close the claim."),
}

PLANS = ("essential", "plus", "premier")

LIMITS = {   # dollars, per person
    "cancellation": {"essential": 1_500, "plus": 5_000, "premier": 10_000},
    "medical": {"essential": 2_000_000, "plus": 5_000_000, "premier": 10_000_000},
    "baggage": {"essential": 750, "plus": 1_500, "premier": 3_000},
    "single_item": {"essential": 200, "plus": 500, "premier": 1_000},
    "delay_per_6h": {"essential": 0, "plus": 25, "premier": 50},
    "delay_max": {"essential": 0, "plus": 250, "premier": 500},
}
EXCESS = {"essential": 150, "plus": 100, "premier": 0}


def limits_table():
    """The limits table as a clause of its own, so retrieval can find it like any other."""
    rows = [f"{benefit.replace('_', ' ')}: " + ", ".join(f"{plan} ${LIMITS[benefit][plan]:,}" for plan in PLANS)
            for benefit in LIMITS]
    rows.append("excess per claim: " + ", ".join(f"{plan} ${EXCESS[plan]}" for plan in PLANS))
    return "Limits per person, in dollars. " + "; ".join(rows) + "."


WORDING["9.1"] = ("Limits table", limits_table())

POLICIES = {
    "MT-48213": {"holder": "Priya Nair", "plan": "plus", "trip": ("2026-09-10", "2026-09-24"),
                 "destination": "Portugal", "add_ons": [], "accepted_conditions": ["asthma"]},
    "MT-51177": {"holder": "Daniel Okoro", "plan": "essential", "trip": ("2026-12-20", "2027-01-03"),
                 "destination": "Austria", "add_ons": [], "accepted_conditions": []},
    "MT-50392": {"holder": "Sofia Lindqvist", "plan": "premier", "trip": ("2026-08-02", "2026-08-16"),
                 "destination": "Japan", "add_ons": [], "accepted_conditions": []},
}

CLAIMS = {
    "CLM-20931": {"policy": "MT-48213", "section": "baggage", "opened": "2026-09-26",
                  "about": "laptop stolen from hotel room, Lisbon", "claimed": 1800.0,
                  "status": "on hold", "waiting_for": "police report", "requested": "2026-09-29"},
    "CLM-20877": {"policy": "MT-50392", "section": "travel delay", "opened": "2026-08-18",
                  "about": "return flight Tokyo-Stockholm delayed 14 hours", "claimed": 100.0,
                  "status": "paid", "paid": 100.0, "paid_on": "2026-09-02"},
}


# --------------------------------------------------------------------------- the tools

def lookup_policy(policy_number: Annotated[str, "The policy number, e.g. MT-48213"]) -> str:
    """A policy's schedule: holder, plan, trip dates and destination, add-ons, and accepted medical conditions."""
    p = POLICIES.get(policy_number.strip().upper())
    if p is None:
        return f"No policy {policy_number}."
    return (f"{policy_number}: {p['holder']}, {p['plan'].title()} plan, {p['destination']} {p['trip'][0]} to "
            f"{p['trip'][1]}. Add-ons: {', '.join(p['add_ons']) or 'none'}. Accepted medical conditions: "
            f"{', '.join(p['accepted_conditions']) or 'none'}.")


def claims_for(policy_number: Annotated[str, "The policy number"]) -> str:
    """Every claim made on a policy, with its status."""
    rows = [(cid, c) for cid, c in CLAIMS.items() if c["policy"] == policy_number.strip().upper()]
    if not rows:
        return f"No claims on {policy_number}."
    return "\n".join(f"{cid}: {c['section']}, {c['about']}, status {c['status']}"
                     + (f", ${c['paid']:.2f} paid on {c['paid_on']}" if c.get("paid") else "") for cid, c in rows)


def claim_status(claim_id: Annotated[str, "A claim number, e.g. CLM-20931"]) -> str:
    """Where one claim stands, and anything we are waiting for."""
    c = CLAIMS.get(claim_id.strip().upper())
    if c is None:
        return f"No claim {claim_id}."
    line = f"{claim_id}: {c['section']} claim, {c['about']}, opened {c['opened']}, status: {c['status']}."
    if c.get("waiting_for"):
        line += f" Waiting for: {c['waiting_for']}, requested on {c['requested']}."
    if c.get("paid"):
        line += f" ${c['paid']:.2f} paid on {c['paid_on']}."
    return line


def estimate_payout(
        policy_number: Annotated[str, "The policy number"],
        benefit: Annotated[Literal["baggage_item", "medical", "cancellation", "travel_delay"], "Which benefit"],
        amount: Annotated[float, "What is being claimed, in dollars. 0 for travel delay."] = 0.0,
        hours: Annotated[float, "For travel delay only: how many hours the departure was delayed."] = 0.0) -> str:
    """What the plan would pay, after limits and excess, IF the claim is covered. Does not decide coverage."""
    p = POLICIES.get(policy_number.strip().upper())
    if p is None:
        return f"No policy {policy_number}."
    plan = p["plan"]
    if benefit == "travel_delay":
        blocks = int(hours // 6)
        pay = min(blocks * LIMITS["delay_per_6h"][plan], LIMITS["delay_max"][plan])
        return (f"{plan.title()} plan, {hours:g} hours delayed: {blocks} full 6-hour periods at "
                f"${LIMITS['delay_per_6h'][plan]} = ${pay:.2f}. No excess on travel delay.")
    cap = {"baggage_item": LIMITS["single_item"][plan], "medical": LIMITS["medical"][plan],
           "cancellation": LIMITS["cancellation"][plan]}[benefit]
    pay = max(0.0, min(amount, cap) - EXCESS[plan])
    return (f"{plan.title()} plan, ${amount:,.2f} claimed under {benefit.replace('_', ' ')}: capped at ${cap:,} "
            f"less ${EXCESS[plan]} excess = ${pay:,.2f}.")


TOOLS = [lookup_policy, claims_for, claim_status, estimate_payout]

# the first design's prompt, kept so claims_checks can measure what it did
TOOLS_SYSTEM = """You look up the account facts needed to answer a travel-insurance customer. The customer's policy
number is {policy_number}. Always look up the policy. Look up their claims if they mention a claim, an earlier
claim, or something that may already have been claimed. If they give an amount or a delay in hours, estimate the
payout for the benefit it falls under. Call every tool you need at once."""


# --------------------------------------------------------------------------- what the assistant hands back

class Reply(BaseModel):
    """The assistant's answer, as fields a claims system can act on."""

    covered: Literal["yes", "no", "partly", "depends", "not_a_coverage_question"] = Field(
        description="Whether what the customer asks about is covered, according to the wording and their schedule.")
    answer: str = Field(description="The reply to the customer: plain, warm, specific to their policy.")
    payout: float | None = Field(
        default=None, description="The dollar amount they can expect, only if a calculator result gives it. "
                                  "Otherwise null.")
    clauses: list[str] = Field(description="The clause numbers the answer rests on, e.g. ['5.1', '5.2'].")
    next_steps: list[str] = Field(default_factory=list, description="What the customer has to do next, if anything.")


ANSWER_SYSTEM = """You are Meridian Travel Insurance's claims assistant, writing to a policyholder.

Answer only from the policy clauses and the account facts below. Say clearly whether what they ask about is
covered, and why, citing clause numbers. A calculator figure applies limits and excess only -- it does not mean
the claim is covered. If the wording excludes it, it is not covered whatever the calculator says. Never invent
figures, dates or documents: if something you need is not below, say what you would need.

Policy clauses:
{clauses}

Account facts for policy {policy_number}:
{facts}"""

REWRITE_SYSTEM = """Rewrite the customer's latest message as one standalone question that can be understood without
the conversation, keeping every specific detail (amounts, places, items, claim numbers). If it already stands on
its own, return it unchanged. Return only the question."""

# The account facts. Two lookups every answer needs, made by code whatever the question; two that
# depend on it, made by code from fields the model extracts. The first version let the model pick
# its tools in one tool-calling turn -- asked to "call every tool you need at once", it called
# `lookup_policy` alone in 15 of 15 trials across three questions, and never the calculator or
# the claims list (claims_checks.one_shot_tools). A one-turn chain gives a model no second turn.
class Lookups(BaseModel):
    """What the account lookups need from the customer's question."""

    claim_id: str = Field(default="", description="A claim number the customer mentions, e.g. CLM-20931. "
                                                  "Empty if none.")
    benefit: Literal["none", "baggage_item", "medical", "cancellation", "travel_delay"] = Field(
        description="The benefit a money question falls under; none if they are not asking about an amount "
                    "or a delay.")
    amount: float = Field(default=0.0, description="The dollar amount involved, exactly as stated. 0 if none.")
    hours: float = Field(default=0.0, description="For a travel delay, how many hours. 0 otherwise.")


LOOKUPS_SYSTEM = """From a travel-insurance customer's question, pull out what the account lookups need: a claim
number if they mention one, and -- if they ask what they would be paid or mention a cost or a delay -- the benefit
it falls under, the dollar amount and the hours of delay. Copy figures exactly; never guess one."""


def already_paid(policy_number, lookups):
    """Clause 3.2 as code: a travel delay already paid on this policy. The earlier claim, or None.

    Written after both builds' replies to message 6 said, in prose, that Sofia's delay was already
    paid -- and, in their fields, `covered: yes, payout: 100.0`. A system acting on the fields pays
    twice. Code cannot tell whether a second delay on the same trip is the SAME delay, so this does
    not refuse anything: it blocks the payout and holds the reply for a person.
    """
    if lookups is None or lookups.benefit != "travel_delay":
        return None
    for claim_id, claim in CLAIMS.items():
        if claim["policy"] == policy_number and claim["section"] == "travel delay" and claim.get("paid"):
            return f"{claim_id} already paid ${claim['paid']:.2f} for a travel delay on {claim['paid_on']}"
    return None


def enforce(reply, policy_number, lookups):
    """The reply as it may be acted on, and why it is held -- or the reply unchanged and None."""
    earlier = already_paid(policy_number, lookups)
    if earlier and reply.payout:
        return reply.model_copy(update={"payout": None}), f"payout blocked for review: {earlier}"
    return reply, None


def account_facts(policy_number, lookups):
    """Run the lookups, in code: always the schedule and the claims, then whatever the question needs."""
    lines = [lookup_policy(policy_number), claims_for(policy_number)]
    if lookups.claim_id:
        lines.append(claim_status(lookups.claim_id))
    if lookups.benefit != "none":
        lines.append(estimate_payout(policy_number, lookups.benefit, lookups.amount, lookups.hours))
    return "\n".join(lines)


# --------------------------------------------------------------------------- the conversation and the answer key

QUESTIONS = [   # (session, policy, message) -- in order, because two of them are follow-ups
    ("priya", "MT-48213", "My laptop was stolen from my hotel room in Lisbon. It was locked in the room safe, and it "
                          "cost $1,800. How much will I get back?"),
    ("priya", "MT-48213", "What's happening with my claim CLM-20931?"),
    ("priya", "MT-48213", "How long do I have to send it?"),
    ("daniel", "MT-51177", "I'm going skiing in Austria in December. If I break my leg on the slopes, am I covered?"),
    ("daniel", "MT-51177", "My mum has just been taken into hospital and I might have to cancel. Would that be covered, "
                           "and what would you need from me?"),
    ("sofia", "MT-50392", "My flight home from Tokyo was delayed 14 hours. What can I claim?"),
    ("priya", "MT-48213", "Separately: I had a bad asthma attack in Porto and spent a night in hospital. The bill was "
                          "$2,300. Is that covered?"),
    ("sofia", "MT-50392", "Also, my backpack was stolen on the beach while I was in the sea. Can I claim for it?"),
]

EXPECTED = [   # one per question: covered, the payout if one should be stated, and what it must rest on
    {"covered": "yes", "payout": 400.0, "clauses": {"5.1", "1.3"}, "must": "single-item limit $500 less $100 excess"},
    {"covered": "not_a_coverage_question", "payout": None, "clauses": set(), "must": "on hold, waiting for the police report requested 29 September"},
    {"covered": "not_a_coverage_question", "payout": None, "clauses": {"8.3"}, "must": "30 days from 29 September: by 29 October 2026"},
    {"covered": "no", "payout": None, "clauses": {"4.3", "6.2"}, "must": "needs the Winter Sports add-on"},
    {"covered": "yes", "payout": None, "clauses": {"2.1", "8.2"}, "must": "mother is a close relative; up to $1,500 less $150; medical certificate, cancellation confirmation, receipts"},
    {"covered": "no", "payout": None, "clauses": {"3.2"}, "must": "already paid $100 on 2 September under CLM-20877"},
    {"covered": "yes", "payout": 2200.0, "clauses": {"4.1", "4.2"}, "must": "asthma was declared and accepted; $2,300 less $100 excess"},
    {"covered": "no", "payout": None, "clauses": {"5.2", "1.3"}, "must": "left unattended"},
]



# For JSON mode, where the provider does not enforce the shape and the prompt has to state it.
JSON_INSTRUCTION = """

Return only a JSON object with these keys: covered, answer, payout, clauses, next_steps -- as described by this
schema: {schema}"""


# For streaming. A JSON reply from gpt-oss on Groq arrives in one piece (claims_checks.json_mode_streaming),
# so the customer would wait for all of it. Prose streams. So the reply is written prose first, and the
# fields a program acts on come after a marker, as JSON -- one call, the answer streamed, the fields
# checked by code when it is finished.
FIELDS_MARKER = "FIELDS:"

PROSE_THEN_FIELDS = """

Write your reply to the customer first, as plain text. Then, on a line of its own, write FIELDS: followed by one
JSON object with the keys covered, payout, clauses and next_steps, as described by this schema: {schema}"""


def fields_schema():
    """Reply's schema without `answer`, which is the prose before the marker."""
    schema = Reply.model_json_schema()
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k != "answer"}
    schema["required"] = [k for k in schema.get("required", []) if k != "answer"]
    return schema


def visible_prose(text):
    """What of a reply still arriving is safe to show: everything before the marker, holding back a
    tail that might be the marker's first letters."""
    cut = text.find(FIELDS_MARKER)
    return text[:cut] if cut >= 0 else text[:max(0, len(text) - len(FIELDS_MARKER))]


def reply_from(text):
    """A finished prose-then-fields reply -> Reply. ValueError if the fields are missing or malformed."""
    import json
    prose, marker, rest = text.partition(FIELDS_MARKER)
    if not marker:
        raise ValueError("the reply has no FIELDS block")
    body = rest.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        fields = json.loads(body)
    except json.JSONDecodeError as bad:
        raise ValueError(f"FIELDS is not JSON: {bad}") from bad
    fields.pop("answer", None)
    return Reply(answer=prose.strip(), **fields)
