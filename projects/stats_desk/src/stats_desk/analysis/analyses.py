"""The analysis tools: what an agent can ask of the release file. The agent chooses what to
compute; compute.py computes it (the same code the profiles are built with), and this module
says the answer, or why it can't be given.

Every answer starts with `result <id>`. The id is a hash of the request, so the same request
always gets the same id. Analysts hand in findings by these ids, and code checks every number
they report against the results they cite (client.py).

Rules kept by every analysis:
  - a figure is MoSPI's published row (data.py); rates are never averaged or added across
    groups; an average over months of one figure (average, series) is said to be one
  - a trend is a straight line set against the usual month-to-month move, never called
    significant: MoSPI's data carries no margins of error
  - only fields with a total can be left out of a request; age, measure and period always
    have a stated value
  - groups containing other groups (education "9.secondary & above") and fields whose groups
    overlap or differ in kind (age bands, the two measures) are shown, never ranked
  - across states, statistical outliers (the small territories) are set apart and named, never
    headlined; in other fields a far-off group is a finding, so it stays in
  - correlations across states describe states, not causes; those true by definition say so
"""

import hashlib
import json
from typing import Annotated

from pydantic import Field

from . import compute
from .compute import num, pts, spread
from .data import CONTAINS, MEASURES, SIDE_BY_SIDE, TIME, TOTAL, Refusal

# The defaults are said once, in tables() and the analyst's instructions: repeated in each of the
# 8 parameters that take a `where`, they cost ~360 tokens on every call an analyst makes.
DEFAULTS_RULE = ("fields left out of `where` are at their total (person, rural + urban, All India, "
                 "all); age is 15 years and above, measure PS+SS, period the latest, unless set")
Where = Annotated[dict | None, Field(description='e.g. {"gender": "female", "period": "2025"}')]
# what a question's words are in the tables (live run: the lead took "cities" to be
# unpublished, though urban by gender is in both the annual and the monthly tables)
MEANINGS = ("what the groups mean: sector urban = towns and cities, rural = villages; state = states and union "
            "territories; there is no figure for a single city or district")


def _result(kind, args, lines):
    clean = {k: v for k, v in args.items() if v not in (None, {}, "")}
    digest = hashlib.sha1(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:6]
    return "\n".join([f"result {kind}-{digest}", *lines])


def _unranked(f, named, fmt=num):
    return [f"not ranked: {k} {fmt(x)} ({CONTAINS[f][k]})" for k, x in named.items() if k in CONTAINS.get(f, {})]


def _field(t, f, where, role):
    if f in TIME:
        raise Refusal("for periods use series (every stored period) or change (vs earlier periods)")
    if f not in t.fields:
        raise Refusal(f"{t.name} has no field '{f}'. Its fields: {', '.join(t.fields)}")
    if where and f in where:
        raise Refusal(f"'{f}' is the field {role} and also set in where; leave it out of where")


# ---------------------------------------------------------------- what's stored

def tables(release, name: str | None = None) -> str:
    """What is stored, and what is published together? With a table name: its card, every value
    of every field, its totals, and which fields are published together. Use it when a call is
    refused as not published, to see what is."""
    if name:
        if name not in release.cards:
            raise Refusal(f"no table '{name}'. Tables: {', '.join(release.cards)}")
        return _result("tables", {"name": name}, [release.cards[name]])
    lines = [f"release: {release.released}"]
    # measure named per table: live run 1's skeptic asked the monthly table for PS+SS, which only
    # the annual tables have
    lines += [f"{t.name}: {t.title}; periods {t.periods[0]} to {t.periods[-1]}; measure "
              f"{' or '.join(t.values['measure'])}; fields {', '.join(f for f in t.fields if f != 'measure')}"
              + (f"; published together: {t.combinations()}" if t.combinations() else "")
              for t in release.tables.values()]
    lines.append("profiles: " + ", ".join(release.profiles))
    lines.append(MEANINGS)
    lines.append(DEFAULTS_RULE)
    # the exact spelling of every group, once: a wrong guess ("graduate") costs a refused call
    values = {}
    for t in release.tables.values():
        for f in t.fields:
            values.setdefault(f, [])
            values[f] += [v for v in t.values[f] if v not in values[f]]
    lines += [f"{f} values: {', '.join(map(str, vs))}" for f, vs in values.items()]
    return _result("tables", {}, lines)


def profile(release, names: list[str]) -> str:
    """What code found striking in the latest release of one or more indicators: the largest
    gaps from the total, the gaps that change most across another field, correlations across
    states, the biggest changes. Ask for every profile you need at once."""
    unknown = [n for n in names if n not in release.profiles]
    if unknown or not names:
        raise Refusal(f"no profile {', '.join(unknown) or 'named'}. Profiles: {', '.join(release.profiles)}")
    return _result("profile", {"names": names}, ["\n\n".join(release.profiles[n] for n in names)])


# ---------------------------------------------------------------- one figure, one field

def value(release, table: str, where: Where = None) -> str:
    """What is the figure for one group? ONE published figure, e.g. participation of rural men in
    Odisha in 2024. Not for comparing: two groups side by side come from gap (both figures and the
    difference in one call), every group of a field from compare, a figure over time from series
    or trend. Never call it twice to put two groups side by side."""
    t = release.table(table)
    s = t.settings(where)
    return _result("value", {"table": table, "where": where}, [t.describe(s), f"value: {num(t.value(s))}%"])


def compare(release, table: str, by: str, where: Where = None) -> str:
    """Which groups are high or low? Every group of one field (`by`) side by side, highest first,
    with the total, e.g. by state or by social group; `where` fixes the rest (e.g. rural people
    only). States that are statistical outliers are named and set apart."""
    t = release.table(table)
    _field(t, by, where, "compared")
    s = t.settings(where, vary=[by])
    groups = t.groups(s, by)
    if len(groups) < 2:
        raise Refusal(t.missing(s, [by]))
    lines = [t.describe(s, [by])]
    if by in TOTAL:
        lines.append(f"total ({TOTAL[by]}): {num(t.value(s.with_(**{by: TOTAL[by]})))}")
    if by in SIDE_BY_SIDE:
        lines.append(", ".join(f"{k} {num(x)}" for k, x in groups.items()))
        lines.append(f"side by side, not ranked: {SIDE_BY_SIDE[by]}")
        return _result("compare", {"table": table, "by": by, "where": where}, lines)
    sp = spread(by, groups)
    lines.append("highest first: " + ", ".join(
        f"{k} {num(x)}" for k, x in sorted(sp.rankable.items(), key=lambda kv: -kv[1])))
    lines += _unranked(by, groups)
    if len(sp.rankable) >= 3:
        lines.append(sp.text())
    return _result("compare", {"table": table, "by": by, "where": where}, lines)


# ---------------------------------------------------------------- gaps and interactions

def _moved(before, now) -> str:
    """How a gap went from `before` to `now`: widened, narrowed, reversed or unchanged."""
    if before * now < 0 and min(abs(before), abs(now)) >= 0.05:
        return "reversed"
    return "widened" if abs(now) > abs(before) else "narrowed" if abs(now) < abs(before) else "unchanged"


def gap(release, table: str, field: str, a: str, b: str, across: str | None = None,
        where: Where = None) -> str:
    """How far apart are two groups, and is the distance growing? a minus b for two groups of ONE
    field (`field`, `a`, `b`: e.g. field sector, a urban, b rural), everything else the same; and
    the same gap a year earlier (annual tables) or a month and a year earlier (monthly): widened,
    narrowed or reversed. `across` another field: the gap within each of its groups, where it
    widens, narrows or reverses. Use it for every comparison of two groups: it computes the
    difference, so never subtract numbers yourself."""
    t = release.table(table)
    _field(t, field, where, "of the gap")
    a, b = t._check(field, a), t._check(field, b)
    if a == b:
        raise Refusal("a and b are the same group")
    where = dict(where or {})
    s = t.settings(where)
    va, vb = t.value(s.with_(**{field: a})), t.value(s.with_(**{field: b}))
    lines = [t.describe(t.settings(where, vary=[field]), [field]),
             f"overall: {a} {num(va)}, {b} {num(vb)}; {a} minus {b} = {pts(va - vb)} points"]
    # the same gap earlier: "is it narrowing?" in one analysis (live run: an
    # analyst needed two calls, and another wrote that it couldn't be done with one)
    for label, p in compute.earlier(t, s.period):
        before = [t.rows(s.with_(period=p, **{field: g})) for g in (a, b)]
        if not all(before):
            lines.append(f"{label} ({p}): not published for these settings")
            continue
        pa, pb = before[0][0]["value"], before[1][0]["value"]
        lines.append(f"{label} ({p}): {a} {num(pa)}, {b} {num(pb)}; {a} minus {b} = {pts(pa - pb)} points; "
                     f"since then the gap {_moved(pa - pb, va - vb)} ({pts((va - vb) - (pa - pb))} points)")
    if across:
        _field(t, across, where, "across")
        if across == field:
            raise Refusal("across must be a different field from the gap's")
        pairs = compute.gaps_across(t, s, field, a, b, across)
        if len(pairs) < 2:
            raise Refusal(t.missing(t.settings({**where, field: a}, vary=[across]), [across]))
        gaps = {k: x - y for k, (x, y) in pairs.items()}
        lines.append(f"across {across}, widest first: " + "; ".join(
            f"{k} {num(pairs[k][0])} vs {num(pairs[k][1])} = {pts(g)}"
            for k, g in sorted(gaps.items(), key=lambda kv: -kv[1])))
        if across in SIDE_BY_SIDE:
            lines.append(f"not ranked: {SIDE_BY_SIDE[across]}")
        elif len(gaps) >= 3:
            lines.append(spread(across, gaps).text(pts))
        overall = va - vb
        reversed_ = [k for k, g in gaps.items() if g * overall < 0 and abs(g) >= 0.05]
        if abs(overall) >= 0.05 and reversed_:
            lines.append(f"the gap is the other way round in: {', '.join(reversed_)}")
    return _result("gap", {"table": table, "field": field, "a": a, "b": b, "across": across,
                           "where": where}, lines)


# ---------------------------------------------------------------- time

def change(release, table: str, where: Where = None, by: str | None = None) -> str:
    """How much did it move since an earlier point? The latest period against the previous year
    (annual tables), or against the previous month and the same month a year earlier (monthly),
    in points. `by`: the same for every group of a field, biggest rise first. Two points in time
    only: whether it is rising or falling over many months is trend."""
    t = release.table(table)
    if by:
        _field(t, by, where, "compared")
    s = t.settings(where, vary=[by] if by else [])
    refs = compute.earlier(t, s.period)
    if not refs:
        raise Refusal(f"no period stored before {s.period} in {table} (stored: {', '.join(t.periods)})")
    if not (t.groups(s, by) if by else t.rows(s)):
        raise Refusal(t.missing(s, [by] if by else []))
    lines = [t.describe(s, [by] if by else [])]
    for label, p in refs:
        if not by:
            now, before = t.rows(s)[0]["value"], t.rows(s.with_(period=p))
            lines.append(f"{label}: {p} {num(before[0]['value'])} -> {s.period} {num(now)}, "
                         f"{pts(now - before[0]['value'])} points" if before
                         else f"{label}: {p} not published for these settings")
            continue
        pairs = compute.changes(t, s, by, p)
        diffs = {k: now - before for k, (before, now) in pairs.items()}
        lines.append(f"{label} ({p} -> {s.period}), rose most first: " + "; ".join(
            f"{k} {num(pairs[k][0])} -> {num(pairs[k][1])} ({pts(d)})"
            for k, d in sorted(diffs.items(), key=lambda kv: -kv[1])))
        if len(diffs) >= 3 and by not in SIDE_BY_SIDE:
            lines.append("  changes: " + spread(by, diffs).text(pts))
    return _result("change", {"table": table, "where": where, "by": by}, lines)


def _by_period(t, s, by=None) -> dict:
    """{group: {period: value}} over every stored period; one group (None) without `by`."""
    if not by:
        return {None: {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}}
    out = {}
    for r in t.rows(s, vary=[by], all_periods=True):
        if r[by] != TOTAL.get(by):
            out.setdefault(r[by], {})[r["period"]] = r["value"]
    return out


def _gap_between(lines: dict) -> tuple | None:
    """For exactly two lines: (a, b, {period: a minus b}) over the periods both have."""
    if len(lines) != 2:
        return None
    (a, va), (b, vb) = lines.items()
    return a, b, {p: va[p] - vb[p] for p in va if p in vb}


def _gap_direction(a, b, d: dict, unit) -> list[str]:
    """Whether the gap between two lines is widening or narrowing, judged on the straight line
    through all its periods against how far they usually sit from it -- never on two end points
    (live run: a gap swinging between 2.1 and 3.8 points was called "widening" from
    April 2025's 2.9 against September 2026's 3.2)."""
    first, last = list(d)[0], list(d)[-1]
    out = [f"{a} minus {b} from {first} to {last}: {pts(d[first])} -> {pts(d[last])} points "
           f"({pts(d[last] - d[first])}), but compare the line below, not two months"]
    if len(d) < 4:
        return out
    line = compute.trend_line(d)
    verdict = compute.trend_verdict(line)
    if verdict != "no clear trend":
        size = sum(d.values()) / len(d)                  # which way is "wider" depends on the gap's sign
        wider = (line["per"] > 0) == (size > 0)
        verdict = ("slightly " if verdict.startswith("slightly") else "") + ("widening" if wider else "narrowing")
    out.append(f"the gap's straight line: {round(line['per'], 2) + 0.0:+.2f} points a {unit}; its {unit}s usually sit "
               f"{line['scatter']:.1f} points from the line: {verdict}")
    return out


def series(release, table: str, where: Where = None, by: str | None = None) -> str:
    """Describe a figure over time: descriptive statistics of every stored period, with no verdict
    on direction. Every value, months not published, lowest and highest, the biggest move between
    periods, the latest streak (e.g. rose 3 months in a row), how the latest compares with the
    rest, 3-month averages. Use it to spot peaks, dips, seasonality or a one-month blip. It does
    not say whether the figure is rising or falling overall: that is trend. `by`: the same for
    each group of a field side by side, and for two groups their gap in every period."""
    t = release.table(table)
    if where and ({"period", "year"} & set(where)):
        raise Refusal("series covers every stored period; leave the period out of where")
    if by:
        _field(t, by, where, "compared")
        return _series_by(t, t.settings(where, vary=[by]), by, {"table": table, "where": where, "by": by})
    s = t.settings(where)
    values = {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}
    if not values:
        raise Refusal(t.missing(s))
    order = list(values)
    lines = [t.describe(s, period=f"{order[0]} to {order[-1]}"),
             "values: " + ", ".join(f"{p} {num(x)}" for p, x in values.items())]
    if t.monthly and compute.missing_months(order):
        lines.append("not published: " + ", ".join(compute.missing_months(order)))
    lo, hi = min(values, key=values.get), max(values, key=values.get)
    lines.append(f"lowest {num(values[lo])} ({lo}), highest {num(values[hi])} ({hi})")
    if len(order) > 2:
        moves = [(values[q] - values[p], p, q) for p, q in zip(order, order[1:])]
        d, p, q = max(moves, key=lambda m: abs(m[0]))
        lines.append(f"largest move between published periods: {p} -> {q} {pts(d)} points")
    run = compute.streak(values, t.monthly)
    if run and run["moves"] >= 2:
        lines.append(f"latest streak: {run['direction']} {run['moves']} {'months' if t.monthly else 'years'} "
                     f"in a row ({run['from']} to {order[-1]})")
    history = compute.against_history(values)
    lines.append(f"latest ({order[-1]}, {num(values[order[-1]])}): "
                 + (history or f"the same as {order[-2]}" if len(order) > 1 else "the only period"))
    if t.monthly:
        windows = [order[i:i + 3] for i in range(len(order) - 2)
                   if compute.step(order[i + 2]) - compute.step(order[i]) == 2]       # 3 months in a row
        if len(windows) >= 2:
            last, before = windows[-1], next((w for w in reversed(windows) if compute.step(w[-1]) <= compute.step(windows[-1][0]) - 1), None)
            avg = lambda w: sum(values[p] for p in w) / 3
            line = f"3-month average (of the monthly rates, not a published figure): {last[0]} to {last[-1]} {num(avg(last))}"
            if before:
                line += f"; the 3 months before, {before[0]} to {before[-1]}, {num(avg(before))}"
            lines.append(line)
    return _result("series", {"table": table, "where": where}, lines)


def _series_by(t, s, by, args):
    groups = _by_period(t, s, by)
    if not groups:
        raise Refusal(t.missing(s, [by]))
    order = sorted({p for v in groups.values() for p in v})
    lines = [t.describe(s, [by], period=f"{order[0]} to {order[-1]}")]
    for g, values in groups.items():
        periods = list(values)
        lo, hi = min(values, key=values.get), max(values, key=values.get)
        history = compute.against_history(values)
        lines.append(f"{g}: " + ", ".join(f"{p} {num(x)}" for p, x in values.items()))
        lines.append(f"  {g}: lowest {num(values[lo])} ({lo}), highest {num(values[hi])} ({hi}), latest "
                     f"{periods[-1]} {num(values[periods[-1]])}" + (f", {history}" if history else ""))
    if t.monthly and compute.missing_months(order):
        lines.append("not published: " + ", ".join(compute.missing_months(order)))
    gap = _gap_between(groups)
    if gap:
        a, b, d = gap
        wide, narrow = max(d, key=lambda p: abs(d[p])), min(d, key=lambda p: abs(d[p]))
        lines.append(f"{a} minus {b}: " + ", ".join(f"{p} {pts(x)}" for p, x in d.items()))
        lines.append(f"  widest {pts(d[wide])} points ({wide}), narrowest {pts(d[narrow])} ({narrow})")
        # no verdict on whether it is widening here: series describes, trend judges
    return _result("series", args, lines)


def _stretch(t, values, start, end, name):
    """The published periods from `start` to `end`, and the ones not published between them."""
    for p in (start, end):
        if p not in t.periods:
            raise Refusal(f"{t.name} has no period {p}. Stored: {', '.join(t.periods)}")
    if end < start:
        raise Refusal(f"{name}: {start} comes after {end}")
    held = {p: x for p, x in values.items() if start <= p <= end}
    if len(held) < 2:
        raise Refusal(f"{name}: fewer than 2 published periods from {start} to {end}")
    gaps = compute.missing_months([start, *held, end]) if t.monthly else []
    return held, gaps


def average(release, table: str, start: str, end: str, where: Where = None,
            vs_start: str | None = None, vs_end: str | None = None, by: str | None = None) -> str:
    """What was the level over a stretch of months? The average of one figure from `start` to
    `end` (e.g. 2026-01 to 2026-06); with `vs_start` and `vs_end`, against another stretch, and
    the difference. `by`: one average per group of a field, and for two groups their gap. An
    average of one group's monthly rates, not a published figure: never use it to combine groups
    (women and men don't average to everyone)."""
    t = release.table(table)
    if where and ({"period", "year"} & set(where)):
        raise Refusal("the periods are set by start and end; leave the period out of where")
    if (vs_start is None) != (vs_end is None):
        raise Refusal("give both vs_start and vs_end, or neither")
    args = {"table": table, "start": start, "end": end, "where": where, "vs_start": vs_start, "vs_end": vs_end}
    if by:
        _field(t, by, where, "compared")
        return _average_by(t, t.settings(where, vary=[by]), by, [(start, end)] + ([(vs_start, vs_end)] if vs_start else []),
                           {**args, "by": by})
    s = t.settings(where)
    values = {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}
    if not values:
        raise Refusal(t.missing(s))
    lines = [t.describe(s, period="averages over periods")]
    stretches = [(start, end)] + ([(vs_start, vs_end)] if vs_start else [])
    means = []
    for a, b in stretches:
        held, gaps = _stretch(t, values, a, b, "average")
        means.append(sum(held.values()) / len(held))
        lines.append(f"average {a} to {b}: {num(means[-1])} ({len(held)} periods"
                     + (f"; not published, left out: {', '.join(gaps)}" if gaps else "") + ")")
    if len(means) == 2:
        lines.append(f"{start} to {end} minus {vs_start} to {vs_end}: {pts(means[0] - means[1])} points")
    lines.append("an average of the monthly rates for one group, not a published figure")
    return _result("average", args, lines)


def _average_by(t, s, by, stretches, args):
    groups = _by_period(t, s, by)
    if not groups:
        raise Refusal(t.missing(s, [by]))
    lines = [t.describe(s, [by], period="averages over periods")]
    means = {}                                  # group -> [average of each stretch]
    for g, values in groups.items():
        parts = []
        for a, b in stretches:
            held, gaps = _stretch(t, values, a, b, "average")
            means.setdefault(g, []).append(sum(held.values()) / len(held))
            parts.append(f"{a} to {b} {num(means[g][-1])}" + (f" (not published, left out: {', '.join(gaps)})" if gaps else ""))
        line = f"{g}: " + "; ".join(parts)
        if len(stretches) == 2:
            line += f"; difference {pts(means[g][0] - means[g][1])} points"
        lines.append(line)
    gap = _gap_between({g: dict(enumerate(m)) for g, m in means.items()})
    if gap:
        a, b, d = gap
        lines.append(f"{a} minus {b}: " + "; ".join(f"{x} to {y} {pts(d[i])} points" for i, (x, y) in enumerate(stretches)))
    lines.append("averages of the monthly rates for each group, not published figures")
    return _result("average", args, lines)


def trend(release, table: str, where: Where = None, start: str | None = None, end: str | None = None,
          by: str | None = None) -> str:
    """Is it rising or falling overall? The straight line through a figure's months (`start` to
    `end`, or all of them), its change along the line, and a verdict: rising, falling or no clear
    trend, judged against how far the months usually scatter around the line. Monthly tables: the
    annual ones hold 2 years, so use change there. `by`: one line per group of a field, and for
    two groups whether their gap is widening. Not a test of statistical significance: MoSPI
    publishes no margins of error."""
    t = release.table(table)
    if where and ({"period", "year"} & set(where)):
        raise Refusal("the periods are set by start and end; leave the period out of where")
    if by:
        _field(t, by, where, "compared")
        return _trend_by(t, t.settings(where, vary=[by]), by, start, end,
                         {"table": table, "where": where, "start": start, "end": end, "by": by})
    s = t.settings(where)
    values = {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}
    if not values:
        raise Refusal(t.missing(s))
    start, end = start or t.periods[0], end or t.periods[-1]
    held, gaps = _stretch(t, values, start, end, "trend")
    if len(held) < 4:
        raise Refusal(f"a trend needs at least 4 published periods; {start} to {end} has {len(held)}"
                      + ("" if t.monthly else ": the annual tables hold 2 years, so use change"))
    line = compute.trend_line(held)
    unit = "month" if t.monthly else "year"
    over, scatter = line["over"], line["scatter"]
    lines = [t.describe(s, period=f"{start} to {end}"),
             f"straight line: {line['per']:+.2f} points a {unit}, {pts(over)} points from {start} to {end}"
             + (f" ({len(held)} periods; not published: {', '.join(gaps)})" if gaps else f" ({len(held)} periods)"),
             f"the {unit}s usually sit {scatter:.1f} points from the line"]
    verdict = compute.trend_verdict(line)
    if verdict == "no clear trend":
        lines.append(f"{verdict}: the change along the line is no larger than the usual scatter around it")
    elif verdict.startswith("slightly"):
        lines.append(f"{verdict}: the change along the line is larger than the usual scatter, but less than twice it")
    else:
        lines.append(f"{verdict}: the change along the line is more than twice the usual scatter around it")
    lines.append("not a test of statistical significance: MoSPI's data carries no margins of error")
    return _result("trend", {"table": table, "where": where, "start": start, "end": end}, lines)


def _trend_by(t, s, by, start, end, args):
    groups = _by_period(t, s, by)
    if not groups:
        raise Refusal(t.missing(s, [by]))
    start, end = start or t.periods[0], end or t.periods[-1]
    for p in (start, end):
        if p not in t.periods:
            raise Refusal(f"{t.name} has no period {p}. Stored: {', '.join(t.periods)}")
    unit = "month" if t.monthly else "year"
    fits, short = {}, []
    for g, values in groups.items():
        held = {p: x for p, x in values.items() if start <= p <= end}
        if len(held) >= 4:
            fits[g] = (compute.trend_line(held), held)
        else:
            short.append(str(g))
    if not fits:
        raise Refusal(f"a trend needs at least 4 published periods for a group; {start} to {end} has fewer"
                      + ("" if t.monthly else ": the annual tables hold 2 years, so use change"))
    lines = [t.describe(s, [by], period=f"{start} to {end}"), f"one straight line per {by}, steepest rise first:"]
    for g, (line, held) in sorted(fits.items(), key=lambda kv: -kv[1][0]["per"]):
        lines.append(f"  {g}: {line['per']:+.2f} points a {unit}, {pts(line['over'])} points from {start} to {end}; "
                     f"{compute.trend_verdict(line)} (its {unit}s usually sit {line['scatter']:.1f} points from the line)")
    if short:
        lines.append(f"  fewer than 4 published periods, no line: {', '.join(short)}")
    gap = _gap_between({g: held for g, (_, held) in fits.items()})
    if gap:
        a, b, d = gap
        first, last = list(d)[0], list(d)[-1]
        if d[first] * d[last] < 0 and min(abs(d[first]), abs(d[last])) >= 0.05:
            lines.append(f"the lines crossed: {a} minus {b} was {pts(d[first])} in {first}, {pts(d[last])} in {last}")
        lines += _gap_direction(a, b, d, unit)
    lines.append("not a test of statistical significance: MoSPI's data carries no margins of error")
    return _result("trend", args, lines)


# ---------------------------------------------------------------- across states

def _sides(release, x_table, y_table, x_where, y_where):
    sides = []
    for name, where in ((x_table, x_where), (y_table or x_table, y_where)):
        t = release.table(name)
        if "state" not in t.fields:
            raise Refusal(f"{name} has no states; relate works across states (the annual tables)")
        if where and "state" in where:
            raise Refusal("relate compares states; leave state out of where")
        s = t.settings(where, vary=["state"])
        if len(t.groups(s, "state")) < 2:
            raise Refusal(t.missing(s, ["state"]))
        sides.append((t, s))
    (tx, sx), (ty, sy) = sides
    if tx is ty and sx.values == sy.values and sx.period == sy.period:
        raise Refusal("x and y are the same figures; set y_where (or y_table) to something else")
    return tx, sx, ty, sy


def relate(release, x_table: str, y_table: str | None = None, x_where: Where = None,
           y_where: Where = None, by: str | None = None) -> str:
    """Do two figures go together across states? E.g. participation (x_table lfpr_annual) and
    unemployment (y_table ur_annual), each with its own `x_where` and `y_where`: ordinary and rank
    correlation, the states driving any disagreement, and whether the link is true by definition.
    `by`: the same within each group of a field (e.g. for women, and for men). Describes states
    (people's chances there), not causes."""
    if by:
        return _relate_by(release, x_table, y_table, x_where, y_where, by)
    tx, sx, ty, sy = _sides(release, x_table, y_table, x_where, y_where)
    args = {"x_table": x_table, "y_table": y_table, "x_where": x_where, "y_where": y_where}
    lines = [f"x: {tx.describe(sx, ['state'])}", f"y: {ty.describe(sy, ['state'])}"]
    c = compute.relate(tx, sx, ty, sy)
    if c.get("too_few"):
        lines.append(f"only {c['n']} states have both figures: too few for a correlation (10 needed)")
        return _result("relate", args, lines)
    if c.get("pearson") is None or c.get("spearman") is None:
        lines.append("one of the two doesn't vary across states: no correlation")
        return _result("relate", args, lines)
    lines.append(f"{c['n']} states: ordinary correlation r = {c['pearson']:.2f}, "
                 f"rank correlation rho = {c['spearman']:.2f}")
    lines.append(compute.strength(c["pearson"], c["n"]))
    if c["agree"]:
        lines.append("the two agree")
    else:
        lines.append("they disagree: a few states pull the ordinary one; r without each: " + ", ".join(
            f"{m['name']} {m['pearson_without']:.2f}" for m in c["most_influential"]))
    if c["built_in"]:
        lines.append(f"BUILT IN, not a finding: {c['built_in']}")
    lines.append("this describes states (people's chances in them), not causes")
    lines.append("states (x, y): " + "; ".join(
        f"{k} {num(c['x'][k])}, {num(c['y'][k])}" for k in sorted(set(c["x"]) & set(c["y"]))))
    return _result("relate", args, lines)


def _relate_by(release, x_table, y_table, x_where, y_where, by):
    """relate within each group of `by`: one correlation per group, the states' figures left out
    (relate without `by` lists them)."""
    tx = release.table(x_table)
    ty = release.table(y_table or x_table)
    if by == "state":
        raise Refusal("relate compares states already; give by another field, e.g. gender")
    for t, where in ((tx, x_where), (ty, y_where)):
        _field(t, by, where, "compared")
    groups = [g for g in tx.values[by] if g != TOTAL.get(by) and g in ty.values[by]]
    lines = [f"x: {tx.describe(tx.settings(x_where, vary=['state', by]), ['state', by])}",
             f"y: {ty.describe(ty.settings(y_where, vary=['state', by]), ['state', by])}"]
    done = 0
    for g in groups:
        try:
            gx, sx, gy, sy = _sides(release, x_table, y_table, {**(x_where or {}), by: g}, {**(y_where or {}), by: g})
        except Refusal as e:
            lines.append(f"{g}: not published ({str(e).split(': ', 1)[-1][:120]})")
            continue
        c = compute.relate(gx, sx, gy, sy)
        if c.get("too_few") or c.get("pearson") is None or c.get("spearman") is None:
            lines.append(f"{g}: too few states with both figures, or no variation: no correlation")
            continue
        done += 1
        lines.append(f"{g}: {c['n']} states, r = {c['pearson']:.2f}, rho = {c['spearman']:.2f}; "
                     f"{compute.strength(c['pearson'], c['n'])}"
                     + ("" if c["agree"] else "; the two disagree: a few states pull r")
                     + (f"; BUILT IN, not a finding: {c['built_in']}" if c["built_in"] else ""))
    if not done:
        raise Refusal(f"no group of {by} has both figures for enough states")
    lines.append("this describes states (people's chances in them), not causes")
    return _result("relate", {"x_table": x_table, "y_table": y_table, "x_where": x_where, "y_where": y_where,
                              "by": by}, lines)


# ---------------------------------------------------------------- where to look next

def drill(release, table: str, where: Where = None) -> str:
    """Which field matters most? When you don't know where to look: for every field not yet set
    in `where`, how far apart its groups are, widest first. Then compare or gap on that field, or
    drill again with a group of it set."""
    t = release.table(table)
    s = t.settings(where)
    headline = t.value(s)
    found, closed = [], []
    for f in t.fields:
        if f not in TOTAL or f in s.named:
            continue
        sp = spread(f, t.groups(s, f))
        if sp is None:
            closed.append(f)
        else:
            found.append((sp.width, f"  {f}: {sp.text()}"))
    lines = [t.describe(s), f"value: {num(headline)}"]
    if found:
        lines.append("split next by, groups differing most first:")
        lines += [text for _, text in sorted(found, key=lambda x: -x[0])]
    if closed:
        lines.append("not published split with these settings: " + ", ".join(closed))
    return _result("drill", {"table": table, "where": where}, lines)


# ---------------------------------------------------------------- does it hold elsewhere?

def _other_period(t, period):
    """The period to recheck against: a year earlier (monthly), or the other stored year."""
    if t.monthly:
        year_ago = f"{int(period[:4]) - 1}{period[4:]}"
        return year_ago if year_ago in t.periods else None
    i = t.periods.index(period)
    return t.periods[i - 1] if i > 0 else (t.periods[i + 1] if i + 1 < len(t.periods) else None)


def _variants(release, analysis, args):
    """(label, changed args): the same request in the other period and on the other measure."""
    keys = ["where"] if "table" in args else ["x_where", "y_where"]
    first = release.table(args.get("table") or args["x_table"])
    s = first.settings(args.get(keys[0]))
    if analysis not in ("series", "change", "average", "trend"):     # these already span periods
        other = _other_period(first, s.period)
        if other:
            yield f"in {other}", {**args, **{k: {**(args.get(k) or {}), "period": other} for k in keys}}
    measures = first.values.get("measure", [])
    if len(measures) > 1:
        other = next(m for m in measures if m != s.values["measure"])
        yield (f"on {MEASURES.get(other, other)}",
               {**args, **{k: {**(args.get(k) or {}), "measure": other} for k in keys}})


def recheck(release, analysis: str, args: dict) -> str:
    """The same analysis in the other period and on the other measure, to see whether a finding
    holds there. Run by code on every finding (agents/checks.py), not offered to the analysts."""
    if analysis not in RECHECKABLE:
        raise Refusal(f"can't recheck {analysis}; only {', '.join(RECHECKABLE)}")
    lines = []
    for label, changed in _variants(release, analysis, args):
        try:
            lines.append(f"{label}:\n" + RECHECKABLE[analysis](release, **changed).split("\n", 1)[1])
        except Refusal as e:
            lines.append(f"{label}: {e}")
    return _result("recheck", {"analysis": analysis, "args": args},
                   lines or ["nothing to recheck: no other period or measure is stored for this"])


RECHECKABLE = {f.__name__: f for f in (value, compare, gap, change, series, average, trend, relate, drill)}
ANALYSES = [tables, profile, value, compare, gap, change, series, average, trend, relate, drill, recheck]
