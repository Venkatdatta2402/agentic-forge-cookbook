"""The calculations shared by the analysis tools and the profiles: one copy of each rule, so a
profile and an agent's drill-down can never disagree.

Each works on published rows read through data.py and computes differences, ranks, correlations,
and -- for one figure over time only -- averages and a straight-line trend. Rates are never
averaged or added ACROSS GROUPS: women's and men's rates don't average to everyone's, because the
groups differ in size (MoSPI's own total rows are the totals). Over months, for one and the same
group, the labour force barely changes size from month to month, so an average of the monthly
rates is close and standard for smoothing; it is always labelled as such, never as a published
figure.
"""

from dataclasses import dataclass

from ..datasets import DATASET
from ..release.stats import correlate, outliers
from .data import BAND, CONTAINS, TOTAL

# Where a far-off group may be noise: the small territories (Lakshadweep, Andaman & Nicobar,
# Chandigarh) are states surveyed with small samples. In other fields a far-off group is a real,
# large group and a finding: setting outliers apart in every field ranked "women minus men across
# education" without its two graduate groups (+8.4, +10.6), hiding the widest gender gap there is.
NOISY_SMALL_GROUPS = DATASET.small_groups

# indicators computed from others (WPR = LFPR x (1 - UR/100)): across states, they move together
# by construction. {pair of indicators: why}
MECHANICAL = DATASET.identities
ACROSS, ACROSS_PLURAL = DATASET.across, DATASET.across_plural


def num(x):
    return f"{round(x, 2):g}"


def pts(x):
    return f"{x:+.1f}"


@dataclass
class Spread:
    """How far apart the groups of one field are. Groups containing others are left out, and
    across states the statistical outliers are set apart (named, not ranked)."""
    named: dict
    rankable: dict
    outliers: list
    lo: str
    hi: str

    @property
    def core(self):
        return {k: x for k, x in self.rankable.items() if k not in self.outliers} or self.rankable

    @property
    def width(self):
        return self.core[self.hi] - self.core[self.lo]

    def text(self, fmt=num):
        if self.width == 0:
            return f"{len(self.rankable)} groups, all {fmt(self.core[self.lo])}"
        line = (f"{len(self.rankable)} groups; " + ("without outliers: " if self.outliers else "")
                + f"lowest {self.lo} {fmt(self.core[self.lo])}, highest {self.hi} {fmt(self.core[self.hi])}, "
                  f"range {self.width:.1f} points")
        if self.outliers:
            line += "; statistical outliers (possibly small samples): " + ", ".join(
                f"{k} {fmt(self.named[k])}" for k in self.outliers)
        return line


def spread(f, named: dict) -> Spread | None:
    rankable = {k: x for k, x in named.items() if k not in CONTAINS.get(f, {})}
    if len(rankable) < 2:
        return None
    odd = outliers(rankable) if f in NOISY_SMALL_GROUPS else []
    core = {k: x for k, x in rankable.items() if k not in odd} or rankable
    return Spread(named, rankable, odd, min(core, key=core.get), max(core, key=core.get))


def gaps_across(t, s, field, a, b, across) -> dict:
    """{group of `across`: (value for a, value for b)}, where both are published."""
    ga, gb = t.groups(s.with_(**{field: a}), across), t.groups(s.with_(**{field: b}), across)
    return {k: (ga[k], gb[k]) for k in ga if k in gb}


def earlier(t, period) -> list:
    """The periods to compare with: the previous stored one, and a year earlier (monthly)."""
    i = t.periods.index(period)
    refs = []
    if i > 0:
        refs.append(("previous " + ("published month" if t.monthly else "year"), t.periods[i - 1]))
    if t.monthly:
        year_ago = f"{int(period[:4]) - 1}{period[4:]}"
        if year_ago in t.periods and year_ago != t.periods[i - 1]:
            refs.append(("a year earlier", year_ago))
    return refs


def changes(t, s, by, period) -> dict:
    """{group of `by`: (value in `period`, value now)}, where both are published."""
    now, before = t.groups(s, by), t.groups(s.with_(period=period), by)
    return {k: (before[k], now[k]) for k in now if k in before}


def missing_months(periods) -> list:
    """Months between the first and last that aren't in `periods` ("YYYY-MM", in order)."""
    out, (y, m) = [], (int(periods[0][:4]), int(periods[0][5:]))
    while f"{y}-{m:02d}" < periods[-1]:
        if f"{y}-{m:02d}" not in periods:
            out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _next_month(period):
    y, m = int(period[:4]), int(period[5:])
    return f"{y + 1}-01" if m == 12 else f"{y}-{m + 1:02d}"


def streak(values: dict, monthly) -> dict | None:
    """The run of moves in one direction that the latest period ends: {"direction", "moves",
    "from"}. A month not published breaks it: what happened across it is not known."""
    order = list(values)
    moves = 0
    direction = None
    for p, q in zip(reversed(order[:-1]), reversed(order[1:])):
        if monthly and _next_month(p) != q:
            break
        d = values[q] - values[p]
        step = "rose" if d > 0 else "fell" if d < 0 else None
        if step is None or (direction and step != direction):
            break
        direction, moves, start = step, moves + 1, p
    return {"direction": direction, "moves": moves, "from": start} if moves else None


def step(period):
    """A period's place on a time line: months for "YYYY-MM", years for "YYYY"."""
    return int(period[:4]) * 12 + int(period[5:7]) if len(period) > 4 else int(period[:4])


def trend_line(values: dict) -> dict:
    """The straight line through the values (least squares), months or years apart as they are,
    so a month not published leaves its gap. {"per": slope a period, "over": change along the
    line from the first period to the last, "scatter": how far the values usually sit from the
    line (the standard deviation of what's left over)}.

    The change along the line is judged against the scatter, not against the typical move from
    one month to the next: on All-India unemployment (2025-04 to 2026-09) the months sit about
    0.3 from the line, with a seasonal rise each May-June, so a -0.2 change along it is no trend,
    though it is twice a typical monthly move (0.1)."""
    xs, ys = [step(p) for p in values], list(values.values())
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    per = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    left = [y - (my + per * (x - mx)) for x, y in zip(xs, ys)]
    scatter = (sum(e * e for e in left) / (len(left) - 2)) ** 0.5 if len(left) > 2 else 0.0
    return {"per": per, "over": per * (xs[-1] - xs[0]), "scatter": scatter}


def trend_verdict(line: dict) -> str:
    """trend_line's change along the line judged against the scatter around it: "no clear
    trend" (no larger than the scatter), "slightly rising" (up to twice it), or "rising"."""
    over, scatter = line["over"], line["scatter"]
    way = "rising" if over > 0 else "falling"
    if abs(over) <= scatter:
        return "no clear trend"
    return f"slightly {way}" if abs(over) <= 2 * scatter else way


def against_history(values: dict) -> str | None:
    """The latest value against the earlier ones: "the highest since 2025-06", "the lowest of the
    series", or None when it equals the period before."""
    order = list(values)
    if len(order) < 2:
        return None
    latest, before = values[order[-1]], values[order[-2]]
    if latest == before:
        return None
    up = latest > before
    for p in reversed(order[:-1]):
        if (values[p] >= latest) if up else (values[p] <= latest):
            return f"the {'highest' if up else 'lowest'} since {p} ({num(values[p])})"
    return f"the {'highest' if up else 'lowest'} of the series"


def strength(r, n):
    """How strong a correlation over n states is, in words. Below 2 / sqrt(n) (0.33 for 36
    states) a correlation this size often appears by chance, so it is no clear link at all: a
    span test had an analyst call r = -0.19 "associated"."""
    if abs(r) < 2 / n ** 0.5:
        return (f"no clear link: a correlation this small (under {2 / n ** 0.5:.2f} for {n} {ACROSS_PLURAL}) "
                "often appears by chance")
    way = "the same way" if r > 0 else "opposite ways"
    size = "weak" if abs(r) < 0.5 else "moderate" if abs(r) < 0.7 else "strong"
    return f"a {size} link: across {ACROSS_PLURAL} the two tend to go {way}"


def relate(tx, sx, ty, sy) -> dict:
    """The across-state correlation of two figures (stats.correlate), with each side's values
    and why it's true by definition when it is."""
    x, y = tx.groups(sx, ACROSS), ty.groups(sy, ACROSS)
    return {**correlate(x, y), "x": x, "y": y, "built_in": built_in(tx, sx, ty, sy)}


def built_in(tx, sx, ty, sy):
    """Why x and y move together by definition, or None."""
    diff = [f for f in tx.fields if f in ty.fields and sx.values.get(f) != sy.values.get(f)]
    if tx.indicator != ty.indicator:
        pair = frozenset({tx.indicator, ty.indicator})
        if pair in MECHANICAL and not diff:
            return MECHANICAL[pair]
        return None
    if len(diff) != 1:
        return None
    f = diff[0]
    a, b = sx.values[f], sy.values[f]
    if f == BAND:
        return f"the {BAND} bands overlap ({a}, {b})"
    if TOTAL.get(f) in (a, b):
        return f"{b if a == TOTAL[f] else a} is part of {TOTAL[f]}"
    if a in CONTAINS.get(f, {}) or b in CONTAINS.get(f, {}):
        return "one group contains the other"
    return None
