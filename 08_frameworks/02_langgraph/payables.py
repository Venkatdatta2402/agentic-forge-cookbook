"""Kestrel Labs' accounts-payable desk, which notebook 1 builds twice -- once in LangGraph, once
from this repo's own parts.

Kestrel makes lab instruments. Invoices arrive by email, and each one has to be paid, held, or
refused according to a written policy. Most of that policy is arithmetic and lookups, and is
code. A model is needed for three things only: reading an email into fields, working out *why*
an invoice does not match, and writing to a vendor.

    VENDORS, PURCHASE_ORDERS, RECEIPTS, PAID
                    the records every check reads. PAYMENTS is where approved invoices go.
    INBOX           seven emails, each written to take a different path through the policy
    EXPECTED        what the policy says should happen to each, and who should be asked
    HUMANS          what the people in the loop answer when they are asked, so a run can be
                    repeated without anyone at the keyboard
    check()         the policy, as code: vendor, duplicate, bank details, three-way match
    approvers()     who has to sign, from the amount and the PO
    lookup_po, receipts, billed_against, vendor_record
                    plain functions the investigator agent is given as tools. Neither build
                    owns them: LangGraph wraps them as LangChain tools, the scratch build as
                    03_tools `Tool`s -- the same function objects both times

Everything is invented. Domains end in `.example`, which is reserved and cannot belong to anyone.
"""

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field

TODAY = "2026-09-28"

AUTO_APPROVE_LIMIT = 5_000.00     # matched invoices at or under this are paid without a signature
CONTROLLER_LIMIT = 25_000.00      # above this the finance controller signs as well as the owner
PRICE_TOLERANCE = 0.01            # a unit price may differ from the PO by 1% (rounding, FX)
NON_PO_LIMIT = 1_000.00           # invoices with no PO are allowed under this, with a signature
MAX_VENDOR_ROUNDS = 2             # how many times we write back before a person takes over

POLICY = """Kestrel Labs accounts-payable policy (summary)

1. Every invoice is matched three ways before it is paid: the invoice, the purchase order (price
   and quantity), and the goods receipt (what actually arrived). We never pay for more than was
   received, and never above the PO price by more than 1%.
2. An invoice that does not match is investigated. If the vendor made a mistake, we ask them for
   a corrected invoice. If only part of an order has arrived, we may pay for what arrived and
   leave the rest to be invoiced on delivery.
3. Matched invoices up to $5,000 are paid automatically. Above that the PO's budget owner signs;
   above $25,000 the finance controller signs as well.
4. An invoice already paid is never paid again. Reminders for paid invoices get a reply with the
   payment date and reference.
5. Bank details are never changed from an email. An invoice asking to be paid to an account that
   is not on file is frozen until someone calls the vendor on the phone number ON FILE -- never a
   number from the email -- and confirms the change.
6. Invoices with no PO are allowed under $1,000 if a budget owner signs for them. The AP clerk
   finds out who ordered it."""

PEOPLE = {
    "mira": "Mira Okafor (optics lead, budget owner)",
    "tomasz": "Tomasz Wierzbicki (electronics and mechanics lead, budget owner)",
    "ines": "Ines Albrecht (finance controller)",
    "dev": "Dev Raman (AP clerk)",
}

VENDORS = {
    "V-101": {"name": "Brandt Optics GmbH", "domain": "brandt-optics.example",
              "iban": "DE44 5001 0517 5407 3249 31", "phone": "+49 30 5550 1200", "terms": "net 30"},
    "V-102": {"name": "Corvid Electronics Ltd", "domain": "corvid-electronics.example",
              "iban": "GB33 BUKB 2020 1555 5555 55", "phone": "+44 161 555 0147", "terms": "net 45"},
    "V-103": {"name": "Halvorsen Metals AS", "domain": "halvorsen-metals.example",
              "iban": "NO93 8601 1117 947", "phone": "+47 22 55 50 90", "terms": "net 30"},
    "V-104": {"name": "Pinecrest Office Supply", "domain": "pinecrest-supply.example",
              "iban": "US 021000021 5550123456", "phone": "+1 503 555 0199", "terms": "net 15"},
}

PURCHASE_ORDERS = {
    "PO-4471": {"vendor": "V-101", "owner": "mira", "raised": "2026-07-28",
                "lines": [{"sku": "LA-50", "item": "lens assembly, 50 mm", "qty": 80, "price": 86.00}]},
    "PO-4480": {"vendor": "V-102", "owner": "tomasz", "raised": "2026-08-12",
                "lines": [{"sku": "CB-7", "item": "controller board rev 7", "qty": 25, "price": 412.00}]},
    "PO-4485": {"vendor": "V-103", "owner": "tomasz", "raised": "2026-08-14",
                "lines": [{"sku": "H-220", "item": "machined aluminium housing", "qty": 60, "price": 145.00}]},
    "PO-4490": {"vendor": "V-102", "owner": "tomasz", "raised": "2026-08-25",
                "lines": [{"sku": "CB-7", "item": "controller board rev 7", "qty": 70, "price": 412.00}]},
    "PO-4493": {"vendor": "V-101", "owner": "mira", "raised": "2026-08-27",
                "lines": [{"sku": "LA-50", "item": "lens assembly, 50 mm", "qty": 120, "price": 86.00}]},
}

RECEIPTS = [
    {"grn": "GRN-7702", "po": "PO-4471", "sku": "LA-50", "qty": 40, "date": "2026-08-20", "note": ""},
    {"grn": "GRN-7741", "po": "PO-4471", "sku": "LA-50", "qty": 40, "date": "2026-09-11", "note": ""},
    {"grn": "GRN-7745", "po": "PO-4480", "sku": "CB-7", "qty": 25, "date": "2026-09-08", "note": ""},
    {"grn": "GRN-7750", "po": "PO-4485", "sku": "H-220", "qty": 45, "date": "2026-09-10",
     "note": "Short shipment: 15 housings backordered by vendor, ETA 2026-10-03."},
    {"grn": "GRN-7758", "po": "PO-4490", "sku": "CB-7", "qty": 70, "date": "2026-09-15", "note": ""},
    {"grn": "GRN-7760", "po": "PO-4493", "sku": "LA-50", "qty": 120, "date": "2026-09-12", "note": ""},
]

# invoices already paid: what a duplicate is checked against, and what a reminder is answered from
PAID = [
    {"vendor": "V-101", "invoice": "INV-88213", "po": "PO-4471", "sku": "LA-50", "qty": 40,
     "amount": 3440.00, "paid": "2026-09-04", "ref": "PAY-30118"},
]

# where approved invoices go. A list, so a double payment shows up as two rows -- which is the
# point of keeping it: notebook 1 pays one invoice twice on purpose to show why that happens.
PAYMENTS = []


# --------------------------------------------------------------------------- the inbox

INBOX = {
    "A": """From: Brandt Optics Accounts <ar@brandt-optics.example>
Subject: Invoice INV-88240

Please find our invoice below.

BRANDT OPTICS GMBH -- INVOICE INV-88240
Date: 2026-09-15        Your PO: PO-4471
LA-50  Lens assembly, 50 mm      40 x 86.00 EUR-equivalent USD   3,440.00
TOTAL DUE (USD)                                                   3,440.00
Terms: net 30. Remit to: Brandt Optics GmbH, IBAN DE44 5001 0517 5407 3249 31""",

    "B": """From: Corvid Electronics <billing@corvid-electronics.example>
Subject: INV-2026-0917 / PO-4480

Invoice INV-2026-0917, dated 2026-09-10, against purchase order PO-4480.

  CB-7 controller board rev 7    qty 25   unit 412.00   10,300.00
  Total: USD 10,300.00

Payment terms net 45 to Corvid Electronics Ltd, IBAN GB33 BUKB 2020 1555 5555 55.""",

    "C": """From: Brandt Optics Accounts <ar@brandt-optics.example>
Subject: Invoice INV-88251

BRANDT OPTICS GMBH -- INVOICE INV-88251
Date: 2026-09-18        Your PO: PO-4493
LA-50  Lens assembly, 50 mm     120 x 90.30                      10,836.00
TOTAL DUE (USD)                                                  10,836.00
Remit to: IBAN DE44 5001 0517 5407 3249 31""",

    "D": """From: Halvorsen Metals <faktura@halvorsen-metals.example>
Subject: Faktura / Invoice HM-5521

Invoice HM-5521   Date 2026-09-12   Customer order: PO-4485
H-220 machined aluminium housing   60 pcs @ 145.00   = 8,700.00
Amount due USD 8,700.00 -- please pay to NO93 8601 1117 947 within 30 days.""",

    "E": """From: Brandt Optics Accounts <ar@brandt-optics.example>
Subject: PAYMENT REMINDER -- INV-88213 overdue

Dear Kestrel Labs,

Our records show invoice INV-88213 (PO-4471, 40 x LA-50 lens assembly at 86.00, total 3,440.00 USD)
as unpaid. Please arrange payment at your earliest convenience to IBAN DE44 5001 0517 5407 3249 31.""",

    "F": """From: Corvid Electronics Accounts <accounts@corvid-electronlcs.example>
Subject: Updated remittance details -- INV-2026-0931

Dear customer,

Following an audit we have moved our banking to a new provider. Please update your records and
remit invoice INV-2026-0931 to the account below. Our old account will be closed at the end of
the month, so please do not use it. For any questions call our accounts team on +44 20 3555 0192.

Invoice INV-2026-0931, 2026-09-20, PO-4490
CB-7 controller board rev 7, 70 x 412.00 = 28,840.00 USD
NEW remit-to: Corvid Electronics Ltd, IBAN GB71 MIDL 4005 1555 5555 12""",

    "G": """From: Pinecrest Office Supply <orders@pinecrest-supply.example>
Subject: Invoice PS-3310

Invoice PS-3310, 2026-09-22
Nitrile gloves, box of 100           12 x 12.95     155.40
Lint-free lab wipes, case             6 x 38.10     228.60
Pipette tips, 1000 uL, rack          10 x 30.00     300.00
Total: 684.00 USD. Pay to routing 021000021, account 5550123456.""",
}

# what should become of each, according to the policy -- the scorecard both builds are run against
EXPECTED = {
    "A": {"status": "scheduled", "amount": 3440.00, "signed_by": []},
    "B": {"status": "scheduled", "amount": 10300.00, "signed_by": ["tomasz"]},
    "C": {"status": "scheduled", "amount": 10320.00, "signed_by": ["mira"], "vendor_rounds": 1},
    "D": {"status": "scheduled", "amount": 6525.00, "signed_by": ["tomasz"]},
    "E": {"status": "duplicate", "amount": 0.0, "signed_by": []},
    "F": {"status": "fraud", "amount": 0.0, "signed_by": []},
    "G": {"status": "scheduled", "amount": 684.00, "signed_by": ["mira"]},
}

# The replies each person gives when asked, keyed by what they are asked about. People answer
# through a form -- a dict with named fields -- as they would in any approval screen, so nothing
# downstream has to guess what "yeah fine, Mira's" meant. The vendor is the exception: its
# corrected invoice for C arrives as an email, and to the workflow a vendor is one more party it
# waits for.
HUMANS = {
    ("C", "vendor"): """From: Brandt Optics Accounts <ar@brandt-optics.example>
Subject: RE: Invoice INV-88251

Apologies -- our system applied next year's price list early. Corrected invoice below.

BRANDT OPTICS GMBH -- INVOICE INV-88251-R (replaces INV-88251)
Date: 2026-09-24        Your PO: PO-4493
LA-50  Lens assembly, 50 mm     120 x 86.00                      10,320.00
TOTAL DUE (USD)                                                  10,320.00
Remit to: IBAN DE44 5001 0517 5407 3249 31""",
    ("F", "callback"): {"verified": False,
                        "note": "Called Corvid on +44 161 555 0147, the number on file. They have not "
                                "changed banks and did not send this email."},
    ("G", "clerk"): {"owner": "mira", "note": "Mira ordered these for the optics lab; no PO was raised."},
}


def human(key, subject, approver=None):
    """What a person answers. Approvers approve unless a test says otherwise."""
    if subject == "approval":
        return {"approved": True, "by": approver, "note": ""}
    return HUMANS[(key, subject)]


# Invoice fields as each build's extraction should produce them, for testing both builds without
# a model. Written by hand from the emails above; a live run is checked against these too, since
# an extraction that drops a digit of an IBAN turns an honest invoice into a fraud alert.
def _inv(vendor, email, number, when, po, lines, total, remit, replaces=""):
    return {"vendor_name": vendor, "sender_email": email, "invoice_number": number, "invoice_date": when,
            "po_number": po, "lines": [{"sku": s, "description": d, "qty": q, "unit_price": p}
                                       for s, d, q, p in lines],
            "total": total, "remit_to": remit, "replaces": replaces}


GOLD = {
    "A": _inv("Brandt Optics GmbH", "ar@brandt-optics.example", "INV-88240", "2026-09-15", "PO-4471",
              [("LA-50", "Lens assembly, 50 mm", 40, 86.00)], 3440.00, "IBAN DE44 5001 0517 5407 3249 31"),
    "B": _inv("Corvid Electronics Ltd", "billing@corvid-electronics.example", "INV-2026-0917", "2026-09-10",
              "PO-4480", [("CB-7", "controller board rev 7", 25, 412.00)], 10300.00,
              "IBAN GB33 BUKB 2020 1555 5555 55"),
    "C": _inv("Brandt Optics GmbH", "ar@brandt-optics.example", "INV-88251", "2026-09-18", "PO-4493",
              [("LA-50", "Lens assembly, 50 mm", 120, 90.30)], 10836.00, "IBAN DE44 5001 0517 5407 3249 31"),
    "C-reply": _inv("Brandt Optics GmbH", "ar@brandt-optics.example", "INV-88251-R", "2026-09-24", "PO-4493",
                    [("LA-50", "Lens assembly, 50 mm", 120, 86.00)], 10320.00,
                    "IBAN DE44 5001 0517 5407 3249 31", replaces="INV-88251"),
    "D": _inv("Halvorsen Metals", "faktura@halvorsen-metals.example", "HM-5521", "2026-09-12", "PO-4485",
              [("H-220", "machined aluminium housing", 60, 145.00)], 8700.00, "NO93 8601 1117 947"),
    "E": _inv("Brandt Optics", "ar@brandt-optics.example", "INV-88213", "", "PO-4471",
              [("LA-50", "lens assembly", 40, 86.00)], 3440.00, "IBAN DE44 5001 0517 5407 3249 31"),
    "F": _inv("Corvid Electronics Ltd", "accounts@corvid-electronlcs.example", "INV-2026-0931", "2026-09-20",
              "PO-4490", [("CB-7", "controller board rev 7", 70, 412.00)], 28840.00,
              "Corvid Electronics Ltd, IBAN GB71 MIDL 4005 1555 5555 12"),
    "G": _inv("Pinecrest Office Supply", "orders@pinecrest-supply.example", "PS-3310", "2026-09-22", "",
              [("", "Nitrile gloves, box of 100", 12, 12.95), ("", "Lint-free lab wipes, case", 6, 38.10),
               ("", "Pipette tips, 1000 uL, rack", 10, 30.00)], 684.00,
              "routing 021000021, account 5550123456"),
}


# --------------------------------------------------------------------------- what a model extracts

class Line(BaseModel):
    sku: str = Field(description="The item code as printed, e.g. LA-50. Empty if none is printed.")
    description: str
    qty: float
    unit_price: float


class Invoice(BaseModel):
    """An invoice email, read into fields. Copied, never corrected."""

    vendor_name: str
    sender_email: str = Field(description="The address in the From: line, exactly.")
    invoice_number: str
    invoice_date: str = Field(description="YYYY-MM-DD")
    po_number: str = Field(default="", description="The purchase order it refers to, e.g. PO-4471. "
                                                    "Empty if the email names none.")
    lines: list[Line]
    total: float
    remit_to: str = Field(default="", description="The bank account or routing details the email asks "
                                                  "to be paid to, copied character for character. Empty if none.")
    replaces: str = Field(default="", description="If this invoice says it replaces or corrects an "
                                                  "earlier one, that invoice's number. Empty otherwise.")


EXTRACT_PROMPT = """Read this invoice email into fields. Copy every value exactly as printed -- numbers,
codes, account details -- and do not correct, convert or complete anything. If a field is not in the
email, leave it empty.

{email}"""


class Recommendation(BaseModel):
    """What the investigator hands back. The workflow acts on `action`; people read `reason`."""

    action: Literal["query_vendor", "approve_partial", "hold"] = Field(
        description="query_vendor: the vendor billed wrongly and must send a corrected invoice. "
                    "approve_partial: pay for what was received now, the rest on delivery. "
                    "hold: something is wrong that a person must look at.")
    amount: float = Field(description="For approve_partial, the amount to pay now. Otherwise 0.")
    reason: str = Field(description="Two or three sentences a budget owner can act on, naming the "
                                    "records you checked.")


INVESTIGATE_PROMPT = """You investigate invoices that failed Kestrel Labs' three-way match. Look up the purchase
order, what was received and what has already been billed against it before you conclude anything, and base
your recommendation only on what those records say.

{policy}"""


VENDOR_QUERY_PROMPT = """Write a short, polite email to {vendor} asking for a corrected invoice.

Invoice: {invoice_number} against {po}
The problem, as our investigation found it: {reason}

State the problem with the exact figures, say what a correct invoice would show, and ask them to reply with
it. Sign it "Kestrel Labs Accounts Payable". No subject line, no placeholders in brackets."""


# --------------------------------------------------------------------------- the policy, as code

def _digits(text):
    # digits only: an account can be written "IBAN DE44 5001...", "DE4450010517..." or "routing
    # 021000021, account 5550123456", and the letters and punctuation are what differ between them
    return re.sub(r"\D", "", text or "")


def _key(invoice_number):
    # "INV-88213", "INV 88213" and "inv88213" are one invoice; "INV-88251-R" is not "INV-88251"
    return re.sub(r"[^A-Z0-9]", "", (invoice_number or "").upper())


def _domain(email):
    match = re.search(r"@([\w.-]+)", email or "")
    return match.group(1).lower() if match else ""


def find_vendor(invoice):
    """The vendor on file this invoice claims to be from, by name. None if nobody matches."""
    name = invoice["vendor_name"].lower()
    for vendor_id, vendor in VENDORS.items():
        first = vendor["name"].split()[0].lower()
        if first in name:
            return vendor_id
    return None


def _po_line(po, line):
    """The PO line an invoice line is billing for: by item code, or by the code inside its text.

    The fallback is there because a model reading "CB-7 controller board rev 7" may put CB-7 in
    `sku` or leave `sku` empty and keep it in the description -- both are fair readings of the
    email, measured in notebook 1, and only one of them would match on code alone. An invoice
    should not be held because of which of two correct readings the model chose.
    """
    for po_line in po["lines"]:
        if line["sku"] == po_line["sku"] or (not line["sku"] and po_line["sku"] in line["description"]):
            return po_line
    return None


def billed_qty(po, sku, exclude=None):
    return sum(p["qty"] for p in PAID + PAYMENTS
               if p["po"] == po and p["sku"] == sku and p["invoice"] != exclude)


def received_qty(po, sku):
    return sum(r["qty"] for r in RECEIPTS if r["po"] == po and r["sku"] == sku)


def check(invoice):
    """Every rule the policy can settle without judgement. Returns a list of findings.

    Each finding is a `(kind, detail)` pair and the kinds are what the workflow routes on:

        unknown_vendor   no vendor on file by that name
        duplicate        already paid -- by invoice number, or same PO, item, quantity and amount
        bank             the remit-to account is not the one on file (or the sender's domain is not)
        no_po            no purchase order named
        price            a unit price above the PO's by more than the tolerance
        quantity         billing for more than has been received, net of what is already billed
        wrong_po         a PO that does not exist, or belongs to another vendor

    An empty list means the invoice matches and only the signatures are left.
    """
    findings = []
    vendor_id = find_vendor(invoice)
    if vendor_id is None:
        return [("unknown_vendor", f"no vendor on file matches {invoice['vendor_name']!r}")]
    vendor = VENDORS[vendor_id]

    # Duplicates first: a paid invoice is answered, not investigated, whatever else is wrong with it.
    #
    # By invoice number only. The obvious extra rule -- same PO, same quantity, same amount -- is
    # wrong here, and invoice A is why: PO-4471 was delivered in two halves of 40, so its second
    # invoice is identical in every figure to the first one, which is already paid. Billing twice
    # for the same goods under a NEW number is still caught, by the quantity check below, because
    # it compares against what has been received minus what has been billed.
    for paid in PAID + PAYMENTS:
        if paid["vendor"] == vendor_id and _key(paid["invoice"]) == _key(invoice["invoice_number"]):
            return [("duplicate", f"{paid['invoice']} was paid on {paid['paid']}, ref {paid['ref']}")]

    # Bank details and sender, before any matching: a fraudulent invoice can match perfectly.
    if invoice.get("remit_to") and _digits(vendor["iban"]) not in _digits(invoice["remit_to"]):
        findings.append(("bank", f"asks for payment to {invoice['remit_to']!r}; on file: {vendor['iban']}"))
    if _domain(invoice["sender_email"]) != vendor["domain"]:
        findings.append(("bank", f"sent from {_domain(invoice['sender_email'])!r}; "
                                 f"{vendor['name']} writes from {vendor['domain']!r}"))
    if any(kind == "bank" for kind, _ in findings):
        return findings

    po_number = invoice.get("po_number", "")
    if not po_number:
        return [("no_po", f"no purchase order named; total {invoice['total']:.2f}")]
    po = PURCHASE_ORDERS.get(po_number)
    if po is None or po["vendor"] != vendor_id:
        return [("wrong_po", f"{po_number} is not a {vendor['name']} purchase order")]

    for line in invoice["lines"]:
        po_line = _po_line(po, line)
        if po_line is None:
            findings.append(("wrong_po", f"{line['sku'] or line['description']} is not on {po_number}"))
            continue
        sku = po_line["sku"]
        if line["unit_price"] > po_line["price"] * (1 + PRICE_TOLERANCE):
            findings.append(("price", f"{sku} billed at {line['unit_price']:.2f}; "
                                      f"{po_number} says {po_line['price']:.2f}"))
        available = received_qty(po_number, sku) - billed_qty(po_number, sku, exclude=invoice.get("replaces"))
        if line["qty"] > available:
            findings.append(("quantity", f"{sku}: billed {line['qty']:g}, received and not yet "
                                         f"billed {available:g}"))
    return findings


def approvers(invoice, amount=None, owner=None):
    """Who must sign, in order. Empty for a matched invoice under the automatic limit.

    A non-PO invoice has no PO to name its owner, so `owner` is whoever the clerk found, and they
    sign whatever the amount -- the automatic limit is for invoices that have been matched, and
    there is nothing to match these against.
    """
    amount = invoice["total"] if amount is None else amount
    po = PURCHASE_ORDERS.get(invoice.get("po_number", ""))
    if po is None:
        return [owner] if owner else []
    if amount <= AUTO_APPROVE_LIMIT:
        return []
    signers = [po["owner"]]
    if amount > CONTROLLER_LIMIT:
        signers.append("ines")
    return signers


def partial_amount(invoice):
    """What policy allows paying now: each line's received-and-unbilled quantity at the PO price.

    The investigator proposes an amount; this is what it is checked against. A model doing
    arithmetic on money is the kind of thing chapter 3 put a `calculate` tool in front of.
    """
    po = PURCHASE_ORDERS[invoice["po_number"]]
    total, qty = 0.0, 0.0
    for line in invoice["lines"]:
        po_line = _po_line(po, line)
        available = (received_qty(invoice["po_number"], po_line["sku"])
                     - billed_qty(invoice["po_number"], po_line["sku"]))
        payable = max(0.0, min(line["qty"], available))
        total += payable * min(line["unit_price"], po_line["price"])
        qty += payable
    return round(total, 2), qty


def schedule_payment(key, invoice, amount, qty=None):
    """The one side effect that matters: money leaves. Appends to PAYMENTS and returns the row.

    `qty` is what is being paid for, which for a partial payment is less than was billed. It is
    recorded because the next invoice against the same PO is checked against it.
    """
    row = {"key": key, "vendor": find_vendor(invoice), "invoice": invoice["invoice_number"],
           "po": invoice.get("po_number", ""),
           "sku": invoice["lines"][0]["sku"] if invoice["lines"] else "",
           "qty": sum(l["qty"] for l in invoice["lines"]) if qty is None else qty,
           "amount": round(amount, 2), "paid": TODAY, "ref": f"PAY-{30200 + len(PAYMENTS)}"}
    PAYMENTS.append(row)
    return row


def reset():
    """Empty the payments ledger, so a notebook can run the inbox twice from the same start."""
    PAYMENTS.clear()


# --------------------------------------------------------------------------- the investigator's tools

def lookup_po(po_number: Annotated[str, "A purchase order number, e.g. PO-4485"]) -> str:
    """A purchase order: vendor, budget owner, and each line's item, quantity and agreed unit price."""
    po = PURCHASE_ORDERS.get(po_number.strip().upper())
    if po is None:
        return f"No purchase order {po_number}."
    lines = "; ".join(f"{l['sku']} {l['item']}: {l['qty']} ordered at {l['price']:.2f}" for l in po["lines"])
    return (f"{po_number}: {VENDORS[po['vendor']]['name']}, raised {po['raised']}, "
            f"owner {PEOPLE[po['owner']]}. {lines}")


def receipts(po_number: Annotated[str, "A purchase order number"]) -> str:
    """Every goods receipt against a PO: what arrived, when, and any note the warehouse made."""
    rows = [r for r in RECEIPTS if r["po"] == po_number.strip().upper()]
    if not rows:
        return f"Nothing has been received against {po_number}."
    return "\n".join(f"{r['grn']} {r['date']}: {r['qty']} x {r['sku']} received."
                     + (f" Note: {r['note']}" if r["note"] else "") for r in rows)


def billed_against(po_number: Annotated[str, "A purchase order number"]) -> str:
    """Invoices already paid against a PO."""
    rows = [p for p in PAID + PAYMENTS if p["po"] == po_number.strip().upper()]
    if not rows:
        return f"Nothing has been paid against {po_number} yet."
    return "\n".join(f"{p['invoice']}: {p['qty']} x {p['sku']}, {p['amount']:.2f}, paid {p['paid']} "
                     f"({p['ref']})" for p in rows)


def vendor_record(vendor_name: Annotated[str, "The vendor's name, e.g. Halvorsen Metals"]) -> str:
    """The vendor master record: payment terms and the contact details on file."""
    for vendor in VENDORS.values():
        if vendor["name"].split()[0].lower() in vendor_name.lower():
            return (f"{vendor['name']}: terms {vendor['terms']}, writes from @{vendor['domain']}, "
                    f"phone on file {vendor['phone']}.")
    return f"No vendor on file matches {vendor_name}."


INVESTIGATOR_TOOLS = [lookup_po, receipts, billed_against, vendor_record]


def outcome(state):
    """The part of a finished run that is scored: status, amount paid, who signed."""
    return {"status": state.get("status"), "amount": round(state.get("amount") or 0.0, 2),
            "signed_by": [a["by"] for a in state.get("approvals", []) if a.get("approved")]}


def days_between(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days
