"""Time as data: how often a table is published, and what its periods are.

A table's frequency comes from its columns: a `month` column makes it monthly, a `quarter`
column quarterly, neither annual (one row per year). Each frequency says how its periods are
written ("2025", "2025-Q3", "2025-10"), the word for one ("year", "quarter", "month"), which
period comes next and which is a year earlier. Code asks the frequency, never "is it monthly?",
so a quarterly dataset (GDP) needs nothing new. Periods are written so that their order as text
is their order in time, and each frequency's are told apart by their shape alone.
"""

import re
from dataclasses import dataclass

# a month as MoSPI's rows name it, -> its number
MONTH_NAMES = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"], start=1)}
MONTHS = [m[:3].lower() for m in MONTH_NAMES]


@dataclass(frozen=True)
class Frequency:
    name: str                   # as the description names its tables: annual, quarterly, monthly
    unit: str                   # one period: year, quarter, month
    per_year: int
    part: str | None            # the column naming the part of a year: month, quarter (None: annual)
    shape: str                  # a period, as a pattern
    words: str                  # what in a task's words means this frequency (checked monthly first)
    window: int | None = None   # periods averaged to smooth a series (3 months), or None

    @property
    def within_year(self) -> bool:
        """More than one period a year: a year earlier is a period of its own."""
        return self.per_year > 1

    def sql(self) -> str:
        """The period of a row as this frequency writes it, in SQL."""
        if self.part == "month":
            return "strftime(period, '%Y-%m')"
        if self.part == "quarter":
            return "CAST(year AS VARCHAR) || '-Q' || CAST(quarter AS VARCHAR)"
        return "CAST(year AS VARCHAR)"

    def index(self, period) -> int:
        """Which part of its year a period is: 1-12 (months), 1-4 (quarters), 1 (years)."""
        if self.part == "month":
            return int(period[5:7])
        if self.part == "quarter":
            return int(period[-1])
        return 1

    def step(self, period) -> int:
        """A period's place on a time line, one apart from the next."""
        return int(period[:4]) * self.per_year + self.index(period) - (1 if self.per_year == 1 else 0)

    def label(self, step) -> str:
        if self.per_year == 1:
            return str(step)
        year, i = divmod(step - 1, self.per_year)
        return f"{year}-{i + 1:02d}" if self.part == "month" else f"{year}-Q{i + 1}"

    def next(self, period) -> str:
        return self.label(self.step(period) + 1)

    def year_ago(self, period) -> str | None:
        """The same period a year earlier; None for annual, where it is the previous period."""
        return self.label(self.step(period) - self.per_year) if self.within_year else None

    def normal(self, period) -> str:
        """A period as asked ("2025-10-01", "2025q3", 2025) written as this frequency writes it,
        or as given when it isn't one."""
        text = str(period).strip()
        if self.part == "quarter":
            found = re.match(r"^(\d{4})\s*-?\s*[Qq]([1-4])", text)
            return f"{found[1]}-Q{found[2]}" if found else text
        return text[:7] if self.part == "month" else text[:4]

    def part_value(self, value) -> str | None:
        """A part of the year as asked (12, "12", "Dec"; 3, "Q3") as this frequency's periods
        write it ("12", "3"), or None when it isn't one."""
        text = str(value).strip().lower()
        if self.part == "month":
            if text.isdigit() and 1 <= int(text) <= 12:
                return f"{int(text):02d}"
            return f"{MONTHS.index(text[:3]) + 1:02d}" if text[:3] in MONTHS else None
        if self.part == "quarter":
            text = text.removeprefix("q")
            return text if text in ("1", "2", "3", "4") else None
        return None

    def part_of(self, period) -> str:
        """A period's part of the year, written as part_value writes it."""
        return period[5:7] if self.part == "month" else period[-1] if self.part == "quarter" else ""


ANNUAL = Frequency("annual", "year", 1, None, r"\d{4}", r"\bannual|\byears?\b|\b20\d\d\b")
QUARTERLY = Frequency("quarterly", "quarter", 4, "quarter", r"\d{4}-Q[1-4]", r"\bquarter|\bQ[1-4]\b")
MONTHLY = Frequency("monthly", "month", 12, "month", r"\d{4}-(?:0[1-9]|1[0-2])",
                    r"\bmonth|\b\d{4}-\d{2}\b|\b(?:" + "|".join(MONTHS) + r")[a-z]*\b", window=3)
FREQUENCIES = (MONTHLY, QUARTERLY, ANNUAL)          # the order a task's words are checked in
# the columns that say when a row is, not who it is about
TIME = ("year", *(f.part for f in FREQUENCIES if f.part), "period")
# a table's name ends in its frequency's: ur_annual, inflation_monthly
NAMES = "|".join(f.name for f in FREQUENCIES)


def of_words(text) -> Frequency | None:
    """The frequency a task's words point to, or None (checked from the finest: "2026" alone is a
    year, "October 2026" a month)."""
    return next((f for f in FREQUENCIES if re.search(f.words, text or "", re.I)), None)


def of_columns(columns) -> Frequency:
    """A table's frequency, from its columns."""
    return next((f for f in FREQUENCIES if f.part and f.part in columns), ANNUAL)


def of(period) -> Frequency:
    """The frequency a period is written in, from its shape."""
    return next(f for f in FREQUENCIES if re.fullmatch(f.shape, str(period)))


def step(period) -> int:
    return of(period).step(period)


def missing(periods) -> list:
    """The periods between the first and last that aren't in `periods` (in order)."""
    if not periods:
        return []
    f = of(periods[0])
    held = set(periods)
    return [f.label(s) for s in range(f.step(periods[0]), f.step(periods[-1])) if f.label(s) not in held]
