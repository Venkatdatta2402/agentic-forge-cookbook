"""The Aria 2 battery recall, which every notebook in this chapter works from.

Notebook 1 wrote the recall's first five pieces. From notebook 2 on, the recall is under way and
messages keep arriving: customers, journalists, the regulator, new reports from the field. This
file holds what stays fixed while that happens:

    BRIEF         the facts, including a section marked internal only
    PERSONAS      who each specialist is, in their own words (their system prompts)
    DESCRIPTIONS  what everyone else is told about each specialist -- a different thing
    lookup_serial, cell_lot, test_results
                  a small made-up database behind the specialists' tools

The database is written so that a report from outside the recalled batch can go either way, and
only looking it up tells you which. Batch AR2-2409 used the same faulty cell lot for its first
1,800 units; batch AR2-2407 did not use it at all.
"""

import re
from typing import Annotated

from tools import tool

BRIEF = """INCIDENT BRIEF: Aria 2 wireless headphones, battery overheating

What happened
- 14 customers have reported the headphones getting hot while charging; in 6 cases the battery visibly swelled.
- 2 customers suffered minor burns (one to a finger, one to the ear). No fires, no hospital stays.
- All 14 units come from one production batch: serial numbers AR2-2408-00001 to AR2-2408-03200, shipped 5-19 August. 3,200 units in total, about 2,900 already with customers.

What we know
- 11 of the returned units have been opened. In 9, the battery cell has a separator defect that lets it overheat while charging. In the other 2 the cause is not yet determined.
- 400 units from other batches were stress-tested. None showed the fault.
- Headphones are safe to use on battery; the risk is while charging.

What we are doing
- Voluntary recall of the affected batch, starting Monday 22 September.
- Customers should stop charging immediately, and choose a free replacement or a full refund at aria.example.com/recall. Prepaid return packaging is sent to them.
- The product-safety regulator has been notified.

INTERNAL ONLY -- never to appear in anything outside the company
- The cells came from our supplier Voltcell (lot VC-7731). Voltcell has not yet accepted responsibility.
- The 2 undetermined units could mean the fault is wider than the batch. Engineering puts this at low but not zero.
- One of the burned customers has a lawyer and has threatened to sue.
- Estimated recall cost: $410,000.

Use the facts in this brief exactly as they are given -- numbers, names, dates and the web address included -- wherever your piece needs them. Do not add any fact that is not here or that your tools did not return: no other names of people or companies, no other dates, years or numbers, no quotes, no phone numbers or email addresses, no causes or risks beyond those stated. If a piece would normally include something you do not have, leave it out."""


PERSONAS = {
    "support": """You are ANA, the customer support lead. You write to customers who own the product, and you write for the person reading at the kitchen table who is a bit scared. Your only goal is that they are safe and know exactly what to do next. You are warm and plain-spoken; you say sorry like a person, not like a company. You never hide a risk from a customer, and you never bury the instruction they need under explanations. You put the one action that matters first. You avoid jargon, legal phrasing and PR phrasing entirely. When a customer gives a serial number, you look it up before telling them anything about it.""",

    "pr": """You are MARCUS, the PR director. You write for journalists and the public, and your job is to protect the trust people have in the brand while being honest. You take responsibility in tone -- no defensive language, no blaming anyone -- and you keep things in proportion. You write in confident, measured sentences that a journalist can quote. You never speculate, never give details that invite follow-up stories, and you always end by pointing people to where they can get help.""",

    "lawyer": """You are DEBORAH, the company's product lawyer. You write to the product-safety regulator, and everything you write could be read out in court. You are formal, exact and complete: the regulator must get every material fact about the hazard, the affected units and the corrective action, because an incomplete notification is itself a violation. At the same time you never admit fault or legal liability and never speculate about causes that are not established. You state facts, numbers, dates and actions, in numbered paragraphs. Emotion and marketing have no place in what you write.""",

    "engineer": """You are RAJ, the lead hardware engineer. You assess technical reports for the rest of the company, and you are blunt. You say what is confirmed, what is not, and what could make this worse than it looks. You never guess at a unit's history: you look up which cell lot it has, and what testing found for that lot, before you conclude anything. You name suppliers, lots and failure modes precisely. You write in terse bullet points. You never soften a finding to make it sound better, and you never pad.""",
}

# What the supervisor is told about each specialist. Not the persona: a persona says how to be
# the agent, and this says when to ask it. They answer different questions for different readers.
DESCRIPTIONS = {
    "support": "Customer support. Writes replies to customers who own the product: whether their unit is affected, "
               "what to do, how to get a refund or replacement. Can look up any serial number.",
    "pr": "PR. Writes statements and replies for journalists and the public.",
    "lawyer": "Product lawyer. Writes to the product-safety regulator, including updates to the recall notification.",
    "engineer": "Hardware engineering. Assesses technical reports and field failures: whether a unit could have the "
                "battery fault, and whether the fault reaches beyond the recalled batch. Can look up a unit's cell lot "
                "and the test results for any lot.",
}


# --------------------------------------------------------------------------- the database

_BATCHES = {
    "2407": {"shipped": "8-22 July", "units": 3000, "recalled": False},
    "2408": {"shipped": "5-19 August", "units": 3200, "recalled": True},
    "2409": {"shipped": "2-16 September", "units": 2600, "recalled": False},
}

_TESTS = {
    "VC-7731": "11 returned units opened: 9 with a separator defect that lets the cell overheat while charging, "
               "2 cause undetermined. This is the lot behind the recall.",
    "VC-7740": "200 units stress-tested at 45 C while charging: no overheating, no swelling, separators within spec.",
    "VC-7702": "150 units stress-tested at 45 C while charging: no overheating, no swelling, separators within spec.",
}

SERIAL = re.compile(r"^AR2-(\d{4})-(\d{5})$")


def _parse(serial):
    match = SERIAL.match(serial.strip().upper().replace("‑", "-"))
    if not match or match.group(1) not in _BATCHES:
        return None
    return match.group(1), int(match.group(2))


def _lot(batch, number):
    if batch == "2408":
        return "VC-7731"
    if batch == "2409":
        # the first 1,800 units of September's batch were built from the tail of the same cell lot
        return "VC-7731" if number <= 1800 else "VC-7740"
    return "VC-7702"


@tool
def lookup_serial(serial: Annotated[str, "Serial number, e.g. AR2-2408-01422"]) -> str:
    """Look up an Aria 2 serial number: its batch, when it shipped, and whether it is in the recall."""
    parsed = _parse(serial)
    if parsed is None:
        return f"No Aria 2 unit has the serial number {serial}."
    batch, number = parsed
    info = _BATCHES[batch]
    if number > info["units"]:
        return f"No Aria 2 unit has the serial number {serial}."
    status = "IN THE RECALL" if info["recalled"] else "not in the recall"
    return f"{serial}: batch AR2-{batch}, shipped {info['shipped']}. {status}."


@tool
def cell_lot(serial: Annotated[str, "Serial number, e.g. AR2-2409-01544"]) -> str:
    """Which battery cell lot a unit was built with."""
    parsed = _parse(serial)
    if parsed is None or parsed[1] > _BATCHES[parsed[0]]["units"]:
        return f"No Aria 2 unit has the serial number {serial}."
    return f"{serial} was built with battery cells from lot {_lot(*parsed)}."


@tool
def test_results(lot: Annotated[str, "Cell lot, e.g. VC-7731"]) -> str:
    """What testing has found for a battery cell lot."""
    return _TESTS.get(lot.strip().upper(), f"No test results on file for lot {lot}.")
