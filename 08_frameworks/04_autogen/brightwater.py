"""Brightwater Outfitters' order data, the questions asked of it, and the answers that are true.

Brightwater sells outdoor gear online. The commercial team wants three numbers before Monday's review,
and the export they have is the kind every analyst gets: mostly right, with two things wrong in it
that nobody mentions.

    write_orders()   the export, as a CSV: Q2 and Q3 2026, about 1,800 orders. Deterministic.
    QUESTIONS        three questions, as the commercial team asked them
    truth()          the answers, computed carefully -- duplicates dropped, returns excluded
    naive()          the answers a careless analysis gets, to show the traps are real
    Finding          the form an answer is handed back in
    grade()          a finding against the truth
    gate()           what model-written code may not do before it is run -- not a sandbox

The two problems, planted:
    duplicates   a re-export appended 70 orders a second time, all West and all in Q3. Same order_id,
                 same everything. Counted twice, they make West look like the fastest-growing region.
    returns      a faulty batch of tents shipped to North in Q3; most of those tents came back, and North's
                 other Q3 returns rose with them.
                 Revenue that still counts them makes North look healthy.
"""

import csv
import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

FILE = "orders.csv"
COLUMNS = ["order_id", "order_date", "region", "channel", "category", "product", "units", "unit_price",
           "discount", "returned"]

CATALOGUE = {   # category -> [(product, price)]
    "tents": [("Ridgeline 2P tent", 289.0), ("Basecamp 4P tent", 449.0)],
    "packs": [("Switchback 40L pack", 159.0), ("Daytrip 22L pack", 79.0)],
    "apparel": [("Stormshell jacket", 219.0), ("Merino base layer", 89.0)],
    "footwear": [("Talus hiking boot", 189.0), ("Approach shoe", 139.0)],
    "stoves": [("Alpine canister stove", 69.0), ("Group stove kit", 129.0)],
}
REGIONS = ["North", "South", "East", "West"]
PROMOTION = ("2026-07-01", "2026-07-14")      # Trailhead: 15% off tents

QUESTIONS = {
    "Q1": "Which region grew net revenue the most from Q2 to Q3 2026, and by what percentage? Net revenue is "
          "units x unit price after discount, excluding returned orders.",
    "Q2": "Which product category had the highest return rate in Q3 2026 (returned orders as a share of orders)?",
    "Q3": "Did the Trailhead promotion (15% off tents, 1-14 July 2026) increase tent units ordered per day -- every "
          "order counts, including ones later returned -- compared with the 14 days before it? By what percentage?",
}


def _orders(seed=3):
    rng = random.Random(seed)
    # orders per day by region and quarter: South grows most, North grows but loses Q3 to returns
    rate = {("North", 2): 2.2, ("North", 3): 2.6, ("South", 2): 2.0, ("South", 3): 2.5,
            ("East", 2): 2.4, ("East", 3): 2.5, ("West", 2): 2.3, ("West", 3): 2.45}
    rows, number = [], 10000
    day = date(2026, 4, 1)
    while day <= date(2026, 9, 30):
        quarter = 2 if day.month <= 6 else 3
        for region in REGIONS:
            count = int(rate[(region, quarter)] + rng.random())
            for _ in range(count):
                number += 1
                category = rng.choice(list(CATALOGUE))
                promo = category == "tents" and PROMOTION[0] <= day.isoformat() <= PROMOTION[1]
                if promo and rng.random() < 0.35:
                    count_extra = 1                     # the promotion lifts tent orders
                else:
                    count_extra = 0
                for _ in range(1 + count_extra):
                    product, price = rng.choice(CATALOGUE[category])
                    faulty = region == "North" and quarter == 3 and category == "tents"
                    returned = rng.random() < (0.6 if faulty else 0.12 if region == "North" and quarter == 3 else 0.05)
                    rows.append({"order_id": f"BW-{number}", "order_date": day.isoformat(), "region": region,
                                 "channel": rng.choice(["web", "web", "app", "marketplace"]),
                                 "category": category, "product": product, "units": rng.choice([1, 1, 1, 2, 2, 3]),
                                 "unit_price": price, "discount": 0.15 if promo else rng.choice([0, 0, 0, 0.05, 0.1]),
                                 "returned": int(returned)})
                    number += 1
        day += timedelta(days=1)
    # the re-export: 70 West Q3 orders appended a second time, larger ones first
    west_q3 = sorted((r for r in rows if r["region"] == "West" and r["order_date"] >= "2026-07-01"),
                     key=lambda r: -r["units"] * r["unit_price"])
    return rows + [dict(r) for r in west_q3[:70]]


def write_orders(folder):
    """Write the export into `folder` and return its path."""
    path = Path(folder) / FILE
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_orders())
    return path


# --------------------------------------------------------------------------- the answers

def _frame(clean):
    import pandas as pd
    df = pd.DataFrame(_orders())
    if clean:
        df = df.drop_duplicates("order_id")
    df["net"] = df["units"] * df["unit_price"] * (1 - df["discount"])
    df["quarter"] = df["order_date"].str[5:7].astype(int).map(lambda m: 2 if m <= 6 else 3)
    return df


def _answers(df, exclude_returns=True):
    kept = df[df["returned"] == 0] if exclude_returns else df
    by = kept.groupby(["region", "quarter"])["net"].sum().unstack()
    growth = ((by[3] / by[2] - 1) * 100).round(1)
    q3 = df[df["quarter"] == 3]
    returns = (q3.groupby("category")["returned"].mean() * 100).round(1)
    tents = df[df["category"] == "tents"]
    during = tents[(tents["order_date"] >= PROMOTION[0]) & (tents["order_date"] <= PROMOTION[1])]["units"].sum() / 14
    before = tents[(tents["order_date"] >= "2026-06-17") & (tents["order_date"] <= "2026-06-30")]["units"].sum() / 14
    lift = round((during / before - 1) * 100, 1)
    return {"Q1": {"entity": growth.idxmax(), "value": float(growth.max()), "all": growth.to_dict()},
            "Q2": {"entity": returns.idxmax(), "value": float(returns.max()), "all": returns.to_dict()},
            "Q3": {"entity": "yes" if lift > 0 else "no", "value": float(lift)}}


def truth():
    """Duplicates dropped by order_id, returns excluded from revenue."""
    return _answers(_frame(clean=True))


def naive():
    """What the same questions give on the export as it stands: duplicates kept; and returns counted too."""
    return {"duplicates kept": _answers(_frame(clean=False)),
            "duplicates kept, returns counted": _answers(_frame(clean=False), exclude_returns=False)}


# --------------------------------------------------------------------------- the form, and grading

class Finding(BaseModel):
    """One answer, as the commercial team will read it."""

    entity: str = Field(description="The one-word answer: a region, a category, or yes / no.")
    value: float = Field(description="The key figure, as a percentage, rounded to one decimal place.")
    answer: str = Field(description="Two or three sentences for the commercial team.")
    data_issues: list[str] = Field(description="Every problem found in the data and what was done about it. "
                                               "Empty if none was found.")


class Review(BaseModel):
    """The reviewer's verdict, as fields the routing can act on rather than a word to search for."""

    approved: bool = Field(description="True only if the code and its output, shown in the analyst's report, "
                                       "answer the question as it is defined.")
    problems: list[str] = Field(default_factory=list,
                                description="Each problem, specific enough to act on: what is wrong in the code or "
                                            "output, and what must be done about it. Empty if approved.")
    fix_by: Literal["analyst", "planner", "nobody"] = Field(
        description="analyst for a calculation or a data problem, planner for a wrong method, nobody if approved.")

    def render(self):
        if self.approved:
            return "APPROVED"
        return f"NOT APPROVED -- for the {self.fix_by}:\n" + "\n".join(f"- {p}" for p in self.problems)


def grade(qid, finding, tolerance=0.5):
    """Right entity, and a figure within `tolerance` percentage points of the truth."""
    want = truth()[qid]
    if finding is None:
        return {"right": False, "entity": None, "value": None, "want": (want["entity"], want["value"])}
    right = (finding.entity.strip().lower() == str(want["entity"]).lower()
             and abs(finding.value - want["value"]) <= tolerance)
    return {"right": right, "entity": finding.entity, "value": finding.value, "want": (want["entity"], want["value"])}


# --------------------------------------------------------------------------- what code may not do

_FORBIDDEN = {
    r"\bos\.environ\b|\bgetenv\b|\bdotenv\b": "reads environment variables",
    r"\bsubprocess\b|\bos\.system\b|\bos\.popen\b|\bpty\b": "starts other programs",
    r"\bsocket\b|\brequests\b|\burllib\b|\bhttpx\b|\baiohttp\b": "uses the network",
    r"\bshutil\.rmtree\b|\bos\.remove\b|\bos\.unlink\b|\brmdir\b": "deletes files",
    r"\.\./|\.\.\\\\|\b[A-Za-z]:\\\\": "reaches outside its working folder",
    r"\b__import__\b|\bexec\(|\beval\(|\bimportlib\b": "builds code at run time",
}


def gate(code):
    """(True, "") if the code may run; otherwise (False, why). Checked before every execution.

    A list of patterns, which a determined author gets past ("o" + "s"). It stops the accidents --
    a model reaching for an API key to "check the configuration", or deleting a file it made -- and
    says no out loud. It is not a sandbox; without Docker there is none here, which is why the
    executor is also given an environment with the secrets taken out.
    """
    for pattern, why in _FORBIDDEN.items():
        if re.search(pattern, code):
            return False, f"refused: the code {why}"
    return True, ""


# --------------------------------------------------------------------------- the team's procedures

# Procedural memory, in chapter 4's terms: how-to the team has learned, kept for the next job. Nine
# procedures, written generically -- none names this export or its problems -- and only three bear on
# Q1 (deduplicate-records, net-revenue, period-growth). The others are there so that what is measured is
# whether the planner PICKS the right ones from their descriptions, not merely whether it obeys one note.
SKILLS = {
    "deduplicate-records": (
        "any question that totals, counts, averages or compares the rows of an export or a log",
        "1. Find the column that identifies a record (an id such as order_id).\n"
        "2. Count the ids that appear more than once, and print that count.\n"
        "3. If the repeated rows are identical, keep one of each and say how many were dropped.\n"
        "4. If they differ, do not guess which is right: report them."),
    "net-revenue": (
        "a question about revenue, sales value or takings",
        "1. Revenue per row = units x unit_price x (1 - discount).\n"
        "2. Check discount is a fraction between 0 and 1 before using it.\n"
        "3. Apply the question's definition of which rows count (returned orders, cancelled orders) exactly "
        "as stated, and say what was excluded."),
    "period-growth": (
        "comparing one figure between two periods (month on month, quarter on quarter)",
        "1. Compute the figure for each period from the same cleaned data.\n"
        "2. Growth % = (later / earlier - 1) x 100.\n"
        "3. Print both periods' figures as well as the growth, for every group compared."),
    "rate-per-day": (
        "an average per day over a date window",
        "1. Count the days in the window inclusively, including days with no rows.\n"
        "2. Divide the window's total by that count."),
    "currency-conversion": (
        "amounts recorded in more than one currency",
        "1. Find each row's currency.\n2. Convert with the rate for the transaction date, not today's.\n"
        "3. Report in one currency, and say which."),
    "outlier-check": (
        "a numeric column that may hold values far outside the usual range",
        "1. Compute the quartiles and the interquartile range (IQR).\n"
        "2. Flag values below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR.\n3. Report them; drop none without a reason."),
    "cohort-retention": (
        "how many customers come back after their first purchase",
        "1. Group customers by the month of their first order.\n"
        "2. For each later month, count the share of each group that ordered again."),
    "significance-test": (
        "deciding whether a difference between two groups is more than chance",
        "1. State the two groups and the measure.\n2. Use a two-proportion z-test for rates, Welch's t-test for "
        "means.\n3. Report the p-value next to the difference, not instead of it."),
    "timezone-normalisation": (
        "timestamps recorded in more than one time zone",
        "1. Convert every timestamp to UTC before grouping by day.\n2. Say which zone the report's days are in."),
}
RELEVANT = {"Q1": {"deduplicate-records", "net-revenue", "period-growth"}}


def skill_catalogue():
    return "\n".join(f"- {name}: {when}" for name, (when, _) in SKILLS.items())


def read_skill(name: str) -> str:
    """The full text of one of the team's procedures."""
    if name.strip() not in SKILLS:
        return f"No procedure named {name!r}. The procedures are:\n{skill_catalogue()}"
    when, steps = SKILLS[name.strip()]
    return f"{name} -- for {when}.\n{steps}"


READ_SKILL_DESCRIPTION = ("Read one of the team's procedures in full, by name. The procedures, and when each "
                          "applies:\n" + skill_catalogue())
