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
import itertools
import json
import re
from typing import Annotated

from pydantic import WithJsonSchema

from . import compute
from .compute import num, pts, spread
from .data import CONTAINS, DEFAULTS, MEASURES, SIDE_BY_SIDE, TIME, TOTAL, Refusal

# The defaults are said once, in tables() and the analyst's instructions: repeated in each of the
# 8 parameters that take a `where`, they cost ~360 tokens on every call an analyst makes.
DEFAULTS_RULE = "\n".join([
    "choosing groups, the same in every analysis (the total always comes with them):",
    '- one group: set its fields in where, e.g. {"state": "Odisha", "sector": "rural", "gender": "female"}',
    '- every group of a field: by, e.g. by "state"',
    '- some groups of one field: a list as that field\'s value, e.g. where {"sector": ["rural", "urban"]}',
    '- groups that differ in more than one field: where a list of groups, e.g. '
    '[{"state": "Odisha", "sector": "rural"}, {"religion": "Sikhism", "gender": "male"}]',
    "fields not set are at their total (person, rural + urban, All India, all; for any field, total works "
    "too); age is 15 years and above, measure PS+SS, period the latest, unless set"])
# Sent with 11 parameters on every analyst call: the groups are explained once, in the tables
# list, so this only points there (Pydantic's own schema for the type and an explanation here
# were 68 tokens each, ~450 a call)
Where = Annotated[dict | list[dict] | None,
                  WithJsonSchema({"type": ["object", "array", "null"],
                                  "description": 'the groups: see "choosing groups" in the tables list'})]
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
    """What is stored, what is published together, and how to choose groups? With a table name:
    its card, every value of every field, its totals, and which fields are published together.
    Use it when a call is refused as not published, to see what is."""
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

MAX_LOOKUPS = 40       # every state (36) of one figure fits


def value(release, table: str, where: Where = None, by: str | None = None) -> str:
    """What are the figures? The published figures for every group asked for, all in one call,
    with the total; for two groups, their difference. Never look groups up one call at a time.
    Only here can a list go in several fields at once, periods too, and inside a group of a list
    of groups: where {"sector": ["rural", "urban"], "period": ["2024", "2025"]} gives four
    figures. For how a gap moved, use gap; to rank, compare."""
    t = release.table(table)
    args_where = where
    groups = _custom(t, where, by=by, lists=True)
    if groups:
        return _value_groups(t, groups, {"table": table, "where": where})
    if by:
        _field(t, by, where, "listed")
        where = {**(where or {}), by: [g for g in t.values[by] if g != TOTAL.get(by)]}
    lists = {k: v for k, v in (where or {}).items() if isinstance(v, list)}
    if not lists:
        s = t.settings(where)
        return _result("value", {"table": table, "where": where}, [t.describe(s), f"value: {num(t.value(s))}%"])
    return _values(t, table, where, lists, {"table": table, "where": args_where, "by": by})


def _value_groups(t, groups, args):
    """value for groups each set on its own: one line each, the total on what they share, and for
    two the difference (saying when they differ in more than one field)."""
    common, differ = _common(groups), _differ(groups)
    spans = bool(set(differ) & set(TIME))
    lines = [t.describe(t.settings(common, vary=[f for f in differ if f not in TIME]),
                        [f for f in differ if f not in TIME], period="periods as listed" if spans else None)]
    got, refusals = {}, []
    for label, w in groups:
        try:
            got[label] = t.value(t.settings(w))
            lines.append(f"{label}: {num(got[label])}%")
        except Refusal as refused:
            refusals.append(refused)
            lines.append(f"{label}: not published ({_why(refused)})")
    if not got:
        raise refusals[0]               # none of them: one refusal (and nearest() may step in)
    # the total on what the groups share, in each period they cover
    shared = {k: v for k, v in common.items() if k not in TIME}
    periods = sorted({str(w[k]) for _, w in groups for k in w if k in TIME}) if spans else [None]
    for p in periods:
        try:
            value_ = t.value(t.settings({**shared, **({"period": p} if p else common)}))
            lines.append(f"{_total_of(t, shared)}{', ' + p if p else ''}: {num(value_)}%")
        except Refusal:
            pass
    if len(got) == 2 == len(groups):
        (a, va), (b, vb) = got.items()
        lines.append(f"{a} minus {b} = {pts(va - vb)} points" + (f"; {_not_like(groups)}" if _not_like(groups) else ""))
    return _result("value", args, lines)


def _values(t, table, where, lists, args):
    """value with lists: every combination of the listed groups (and periods), one line each."""
    if any(not v for v in lists.values()):
        raise Refusal("a list in where is empty: give at least one group")
    combos = list(itertools.product(*lists.values()))
    if len(combos) > MAX_LOOKUPS:
        raise Refusal(f"{len(combos)} figures at once is too many (at most {MAX_LOOKUPS}); for every group of a "
                      "field ranked use compare, over time series")
    fixed = {k: v for k, v in where.items() if k not in lists}
    fields = [k for k in lists if k not in TIME]
    periods = [str(p) for p in lists.get("period", lists.get("year", []))]
    base = t.settings(fixed, vary=fields)
    lines = [t.describe(base, fields, period=", ".join(periods) or None)]
    refusals = []
    for combo in combos:
        asked = dict(zip(lists, combo))
        try:
            s = t.settings({**fixed, **asked})
            label = ", ".join(str(s.period if k in TIME else s.values[k]) for k in lists)
            lines.append(f"{label}: {num(t.value(s))}%")
        except Refusal as refused:
            refusals.append(refused)
            why = re.search(r"(published only for .+|not published.*)$", str(refused))
            lines.append(f"{', '.join(str(v) for v in combo)}: not published"
                         + (f" ({why[0]})" if why and why[0] != "not published" else ""))
    if len(refusals) == len(combos):
        raise refusals[0]               # none of them: one refusal (and nearest() may step in)
    # the total for comparison, when one field's groups were listed
    if len(fields) == 1 and fields[0] in TOTAL:
        f = fields[0]
        for p in periods or [None]:
            try:
                s = t.settings({**fixed, f: TOTAL[f], **({"period": p} if p else {})})
                lines.append(f"{_total(f)}{', ' + s.period if p else ''}: {num(t.value(s))}%")
            except Refusal:
                pass
    return _result("value", args, lines)


def _listed(t, where, by=None):
    """(where without its list, the field compared, the chosen groups or None). The same way of
    asking for groups in every analysis: `by` a field is every group of it, with the total; a list
    for a field in `where` is only those groups (and the total); several fields set in `where`
    pick one combination, e.g. a state, a religion and a gender."""
    where = dict(where or {})
    lists = [k for k, v in where.items() if isinstance(v, list)]
    if not lists:
        return where, by, None
    if set(lists) & set(TIME):
        raise Refusal("for several periods use series, trend or average (or value, which takes a list of periods)")
    if len(lists) > 1 or (by and lists != [by]):
        raise Refusal("list the groups of one field only, the one compared" + (f" ({by})" if by else "")
                      + "; for figures of several fields at once use value")
    f = lists[0]
    return where, f, [t._check(f, g) for g in where.pop(f)]


MAX_GROUPS = 8


def _custom(t, where, periods=True, by=None, lists=False):
    """[(label, settings)] when `where` is a list of groups each set on its own (e.g. rural
    women in one state, and urban men of one religion), else None. The label is the group's
    values, e.g. "Odisha, rural". `lists` (value): a list inside a group gives each of its values,
    e.g. a group over two years is two figures (span test: the model asked for three groups,
    each over 2024 and 2025, and was refused)."""
    if not isinstance(where, list):
        return None
    if by:
        raise Refusal("give by or a list of groups, not both")
    if not 2 <= len(where) <= MAX_GROUPS and not (lists and len(where) == 1):
        raise Refusal(f"give 2 to {MAX_GROUPS} groups in the list")
    if any(not isinstance(w, dict) for w in where):
        raise Refusal('each group in the list is one set of settings, e.g. {"sector": "rural", "gender": "female"}')
    expanded = []
    for w in where:
        inside = {k: v for k, v in w.items() if isinstance(v, list)}
        if inside and not lists:
            raise Refusal('each group in the list is one set of settings with no lists inside (only value '
                          'takes them), e.g. {"sector": "rural", "gender": "female"}')
        expanded += [{**w, **dict(zip(inside, combo))} for combo in itertools.product(*inside.values())]
    if len(expanded) > MAX_LOOKUPS:
        raise Refusal(f"{len(expanded)} figures at once is too many (at most {MAX_LOOKUPS})")
    out = []
    for w in expanded:
        if not periods and set(w) & set(TIME):
            raise Refusal("this analysis covers the periods itself; leave the period out of every group")
        s = t.settings(w)
        out.append((", ".join(str(s.period) if k in TIME else str(s.values[k]) for k in w) or "everyone", w))
    if len({label for label, _ in out}) < len(out):
        raise Refusal("two groups in the list are the same")
    return out


def _common(groups) -> dict:
    """The settings every group shares: the total of the groups is taken on these."""
    first, *rest = [w for _, w in groups]
    return {k: v for k, v in first.items() if all(str(w.get(k)) == str(v) for w in rest)}


def _differ(groups) -> list:
    ws = [w for _, w in groups]
    return sorted({k for w in ws for k in w if any(str(o.get(k)) != str(w[k]) for o in ws)})


def _total_of(t, common) -> str:
    named = ", ".join(str(t._check(k, v)) for k, v in common.items() if k not in TIME)
    return f"total ({named}, the rest at their totals)" if named else "total (everyone)"


def _not_like(groups) -> str:
    differ = [f for f in _differ(groups) if f not in TIME]
    return f"they differ in {', '.join(differ)}: not like with like" if len(differ) > 1 else ""


def _within(analysis, release, groups, kwargs, key="where", also=None) -> list:
    """The analysis within each custom group, one block each (compare, gap, drill, relate)."""
    lines, refusals = [], []
    for label, w in groups:
        try:
            extra = {also: {**(kwargs.get(also) or {}), **w}} if also else {}
            text = analysis(release, **{**kwargs, key: w, **extra})
        except Refusal as refused:
            refusals.append(refused)
            lines.append(f"{label}: not published ({_why(refused)})")
            continue
        lines.append(f"within {label}:")
        lines += ["  " + line for line in text.split("\n")[1:]]
    if len(refusals) == len(groups):
        raise refusals[0]                # as for one group: nearest() may answer on the nearest published
    return lines


def _why(refused) -> str:
    """A refusal's reason, for one group's line among others: what is published instead, e.g.
    "published only for age 15 years and above" (span test: "not published for these settings",
    with no reason, left an analyst reporting the figures don't exist)."""
    text = str(refused)
    found = re.search(r"published only for .+", text)
    return found[0] if found else text.split(": ", 1)[-1]


def _keep(groups: dict, chosen) -> dict:
    return groups if chosen is None else {k: v for k, v in groups.items() if k in chosen}


def _total(by) -> str:
    return f"total ({TOTAL[by]})"


def compare(release, table: str, by: str, where: Where = None) -> str:
    """Which groups are high or low? The groups of one field (`by`, e.g. state) ranked highest
    first, with the total; `where` fixes the rest (e.g. rural people only). Groups that differ in
    more than one field (where a list of groups): a ranking within each. Small states that are
    statistical outliers are set apart."""
    t = release.table(table)
    args = {"table": table, "by": by, "where": where}
    groups = _custom(t, where)
    if groups:
        return _result("compare", args, _within(compare, release, groups, {"table": table, "by": by}))
    where, by, chosen = _listed(t, where, by)
    _field(t, by, where, "compared")
    s = t.settings(where, vary=[by])
    groups = _keep(t.groups(s, by), chosen)
    if len(groups) < (1 if chosen else 2):
        raise Refusal(t.missing(s, [by]))
    lines = [t.describe(s, [by])]
    if by in TOTAL:
        lines.append(f"{_total(by)}: {num(t.value(s.with_(**{by: TOTAL[by]})))}")
    if by in SIDE_BY_SIDE:
        lines.append(", ".join(f"{k} {num(x)}" for k, x in groups.items()))
        lines.append(f"side by side, not ranked: {SIDE_BY_SIDE[by]}")
        return _result("compare", args, lines)
    sp = spread(by, groups)
    rankable = sp.rankable if sp else {k: x for k, x in groups.items() if k not in CONTAINS.get(by, {})}
    lines.append("highest first: " + ", ".join(
        f"{k} {num(x)}" for k, x in sorted(rankable.items(), key=lambda kv: -kv[1])))
    lines += _unranked(by, groups)
    if sp and len(sp.rankable) >= 3:
        lines.append(sp.text())
    return _result("compare", args, lines)


# ---------------------------------------------------------------- gaps and interactions

def _moved(before, now) -> str:
    """How a gap went from `before` to `now`: widened, narrowed, reversed or unchanged."""
    if before * now < 0 and min(abs(before), abs(now)) >= 0.05:
        return "reversed"
    return "widened" if abs(now) > abs(before) else "narrowed" if abs(now) < abs(before) else "unchanged"


def gap(release, table: str, field: str, a: str | list[str], b: str, across: str | None = None,
        where: Where = None) -> str:
    """How far apart are groups, and is the gap growing? a minus b for groups of one field
    (`field` sector, `a` urban, `b` rural), everything else the same, and the same gap a year
    (or a month) earlier: widened, narrowed or reversed. `a` a list: each against b; `a` "*":
    every group against b (b may be the total). `across` another field: the gap within each of
    its groups (one a only). Groups that differ in more than one field (where a list of groups):
    the gap within each. It computes every difference: never subtract numbers yourself."""
    t = release.table(table)
    groups = _custom(t, where)
    if groups:
        return _result("gap", {"table": table, "field": field, "a": a, "b": b, "across": across, "where": where},
                       _within(gap, release, groups, {"table": table, "field": field, "a": a, "b": b, "across": across}))
    _field(t, field, where, "of the gap")
    b = t._check(field, b)
    if a == "*":
        a_list = [g for g in t.values[field] if g not in (b, TOTAL.get(field))]
    else:
        a_list = [t._check(field, g) for g in (a if isinstance(a, list) else [a])]
    if b in a_list or not a_list:
        raise Refusal("a and b are the same group" if a_list else "give at least one group as a")
    if across and len(a_list) > 1:
        raise Refusal("across takes one pair: one group as a")
    where = dict(where or {})
    s = t.settings(where)
    vb = t.value(s.with_(**{field: b}))
    lines = [t.describe(t.settings(where, vary=[field]), [field])]
    refusals = []
    for one in a_list:
        try:
            va = t.value(s.with_(**{field: one}))
        except Refusal as refused:
            refusals.append(refused)
            lines.append(f"{one}: not published ({_why(refused)})")
            continue
        lines.append(f"{'overall: ' if len(a_list) == 1 else ''}{one} {num(va)}, {b} {num(vb)}; "
                     f"{one} minus {b} = {pts(va - vb)} points")
        # the same gap earlier: "is it narrowing?" in one analysis (live run: an
        # analyst needed two calls, and another wrote that it couldn't be done with one)
        for label, p in compute.earlier(t, s.period):
            before = [t.rows(s.with_(period=p, **{field: g})) for g in (one, b)]
            if not all(before):
                lines.append(f"  {label} ({p}): not published for these settings")
                continue
            pa, pb = before[0][0]["value"], before[1][0]["value"]
            lines.append(f"  {label} ({p}): {one} {num(pa)}, {b} {num(pb)}; {one} minus {b} = {pts(pa - pb)} points; "
                         f"since then the gap {_moved(pa - pb, va - vb)} ({pts((va - vb) - (pa - pb))} points)")
    if len(refusals) == len(a_list):
        raise refusals[0]                # as for one group: nearest() may answer on the nearest published
    a = a_list[0] if len(a_list) == 1 else a
    if across:
        va = t.value(s.with_(**{field: a}))
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
    """How much did it move since the previous year (annual tables), or the previous month and a
    year earlier (monthly)? In points. Several groups: side by side with the total; for two, how
    their gap moved. Two points in time only: rising or falling over many months is trend."""
    t = release.table(table)
    args = {"table": table, "where": where, "by": by}
    custom = _custom(t, where, periods=False, by=by)
    if custom:
        return _change_groups(t, custom, args)
    where, by, chosen = _listed(t, where, by)
    if by:
        _field(t, by, where, "compared")
    s = t.settings(where, vary=[by] if by else [])
    refs = compute.earlier(t, s.period)
    if not refs:
        raise Refusal(f"no period stored before {s.period} in {table} (stored: {', '.join(t.periods)})")
    if not (_keep(t.groups(s, by), chosen) if by else t.rows(s)):
        raise Refusal(t.missing(s, [by] if by else []))
    lines = [t.describe(s, [by] if by else [])]
    for label, p in refs:
        if not by:
            now, before = t.rows(s)[0]["value"], t.rows(s.with_(period=p))
            lines.append(f"{label}: {p} {num(before[0]['value'])} -> {s.period} {num(now)}, "
                         f"{pts(now - before[0]['value'])} points" if before
                         else f"{label}: {p} not published for these settings")
            continue
        pairs = _keep(compute.changes(t, s, by, p), chosen)
        diffs = {k: now - before for k, (before, now) in pairs.items()}
        lines.append(f"{label} ({p} -> {s.period}), rose most first: " + "; ".join(
            f"{k} {num(pairs[k][0])} -> {num(pairs[k][1])} ({pts(d)})"
            for k, d in sorted(diffs.items(), key=lambda kv: -kv[1])))
        if by in TOTAL:
            try:
                total = [t.value(s.with_(**{by: TOTAL[by]}, **({"period": q} if q else {}))) for q in (p, None)]
                lines.append(f"  {_total(by)}: {num(total[0])} -> {num(total[1])} ({pts(total[1] - total[0])})")
            except Refusal:
                pass
        sp = spread(by, diffs) if len(diffs) >= 3 and by not in SIDE_BY_SIDE else None
        if sp:
            lines.append("  changes: " + sp.text(pts))
    return _result("change", args, lines)


def _change_groups(t, custom, args):
    """change for groups each set on its own, the total on what they share, and for two how the
    gap between them moved."""
    groups, head, _, notes = _lines(t, None, None, custom=custom)
    now = t.periods[-1]
    refs = compute.earlier(t, now)
    if not refs:
        raise Refusal(f"no period stored before {now} in {t.name}")
    lines = [head(None), *notes]
    for label, p in refs:
        moved = {g: (v[p], v[now]) for g, v in groups.items() if p in v and now in v}
        lines.append(f"{label} ({p} -> {now}): " + "; ".join(
            f"{g} {num(b)} -> {num(a)} ({pts(a - b)})" for g, (b, a) in moved.items()) if moved
            else f"{label} ({p} -> {now}): not published for these groups")
        two = [g for g in moved if not _is_total(g)]
        if len(two) == 2 == len(custom):
            (x0, x1), (y0, y1) = moved[two[0]], moved[two[1]]
            lines.append(f"  {two[0]} minus {two[1]}: {pts(x0 - y0)} -> {pts(x1 - y1)} points, "
                         f"the gap {_moved(x0 - y0, x1 - y1)}")
    return _result("change", args, lines)


def _by_period(t, s, by=None, chosen=None) -> dict:
    """{group: {period: value}} over every stored period, the chosen groups only (all if None),
    then the total; one group (None) without `by`."""
    if not by:
        return {None: {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}}
    out, total = {}, {}
    for r in t.rows(s, vary=[by], all_periods=True):
        if r[by] == TOTAL.get(by):
            total[r["period"]] = r["value"]
        elif chosen is None or r[by] in chosen:
            out.setdefault(r[by], {})[r["period"]] = r["value"]
    if out and total:
        out[_total(by)] = total
    return out


def _lines(t, s, by, chosen=None, custom=None):
    """The groups an analysis over time draws, one line each: (groups {name: {period: value}},
    head(period) -> its first line, what a group is called, notes under the head). From `by`
    (with `chosen`), or from custom groups each set on its own, with the total on what they share."""
    if not custom:
        groups = _by_period(t, s, by, chosen)
        if not groups:
            raise Refusal(t.missing(s, [by]))
        return groups, lambda period: t.describe(s, [by], period=period), by, []
    common, differ = _common(custom), [f for f in _differ(custom) if f not in TIME]
    groups, notes = {}, []
    for label, w in custom:
        values = {r["period"]: r["value"] for r in t.rows(t.settings(w), all_periods=True)}
        if values:
            groups[label] = values
        else:
            notes.append(f"{label}: not published ({_why(t.missing(t.settings(w)))})")
    if not groups:
        raise Refusal(t.missing(t.settings(custom[0][1])))
    total = {r["period"]: r["value"] for r in t.rows(t.settings(common), all_periods=True)}
    if total:
        groups[_total_of(t, common)] = total
    if _not_like(custom):
        notes.append("the groups: " + _not_like(custom))
    base = t.settings(common, vary=differ)
    return groups, lambda period: t.describe(base, differ, period=period), "group", notes


def _is_total(g) -> bool:
    return str(g).startswith("total (")


def _gap_between(lines: dict) -> tuple | None:
    """For exactly two groups (the total aside): (a, b, {period: a minus b}) over the periods both
    have."""
    lines = {g: v for g, v in lines.items() if not _is_total(g)}
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
    """What happened period by period? Descriptive statistics, no verdict on direction: every
    value, months not published, lowest and highest, the biggest move, the latest streak, how the
    latest compares with the past, 3-month averages. For peaks, dips, seasonality or a one-off
    blip. Several groups: side by side with the total; for two, their gap each period. Rising or
    falling overall is trend."""
    t = release.table(table)
    args = {"table": table, "where": where, "by": by}
    custom = _custom(t, where, periods=False, by=by)
    if custom:
        return _series_by(t, _lines(t, None, None, custom=custom), args)
    if where and ({"period", "year"} & set(where)):
        raise Refusal("series covers every stored period; leave the period out of where")
    where, by, chosen = _listed(t, where, by)
    if by:
        _field(t, by, where, "compared")
        s = t.settings(where, vary=[by])
        return _series_by(t, _lines(t, s, by, chosen), args)
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


def _series_by(t, drawn, args):
    groups, head, _, notes = drawn
    order = sorted({p for v in groups.values() for p in v})
    lines = [head(f"{order[0]} to {order[-1]}"), *notes]
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
    """What was the level over a stretch of months? The average from `start` to `end` (e.g.
    2026-01 to 2026-06); with `vs_start` and `vs_end`, against another stretch. Several groups:
    one average each, with the total; for two, their gap. An average of a group's own monthly
    rates, not a published figure: never use it to combine groups (women and men don't average to
    everyone)."""
    t = release.table(table)
    if (vs_start is None) != (vs_end is None):
        raise Refusal("give both vs_start and vs_end, or neither")
    args = {"table": table, "start": start, "end": end, "where": where, "vs_start": vs_start, "vs_end": vs_end}
    stretches = [(start, end)] + ([(vs_start, vs_end)] if vs_start else [])
    custom = _custom(t, where, periods=False, by=by)
    if custom:
        return _average_by(t, _lines(t, None, None, custom=custom), stretches, {**args, "by": by})
    if where and ({"period", "year"} & set(where)):
        raise Refusal("the periods are set by start and end; leave the period out of where")
    where, by, chosen = _listed(t, where, by)
    if by:
        _field(t, by, where, "compared")
        return _average_by(t, _lines(t, t.settings(where, vary=[by]), by, chosen), stretches, {**args, "by": by})
    s = t.settings(where)
    values = {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}
    if not values:
        raise Refusal(t.missing(s))
    lines = [t.describe(s, period="averages over periods")]
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


def _average_by(t, drawn, stretches, args):
    groups, head, _, notes = drawn
    lines = [head("averages over periods"), *notes]
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
    """Is it rising or falling overall? A straight line through the months (`start` to `end`, or
    all) and a verdict: rising, falling or no clear trend, judged against how far the months
    usually sit from the line. Monthly tables only: the annual ones hold 2 years, so use change.
    Several groups: one line each, with the total; for two, whether their gap is widening. Not a
    test of statistical significance: MoSPI publishes no margins of error."""
    t = release.table(table)
    args = {"table": table, "where": where, "start": start, "end": end, "by": by}
    custom = _custom(t, where, periods=False, by=by)
    if custom:
        return _trend_by(t, _lines(t, None, None, custom=custom), start, end, args)
    if where and ({"period", "year"} & set(where)):
        raise Refusal("the periods are set by start and end; leave the period out of where")
    where, by, chosen = _listed(t, where, by)
    if by:
        _field(t, by, where, "compared")
        return _trend_by(t, _lines(t, t.settings(where, vary=[by]), by, chosen), start, end, {**args, "by": by})
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


def _trend_by(t, drawn, start, end, args):
    groups, head, noun, notes = drawn
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
    lines = [head(f"{start} to {end}"), *notes, f"one straight line per {noun}, steepest rise first:"]
    # the months MoSPI skipped inside the stretch, as the one-figure trend says them (eval: a
    # brief resting on this result never said March 2026 wasn't published)
    held = sorted({p for _, held in fits.values() for p in held})
    if t.monthly and held and compute.missing_months([start, *held, end]):
        lines.append("not published: " + ", ".join(compute.missing_months([start, *held, end])))
    # the total after the groups, not ranked among them
    for g, (line, held) in sorted(fits.items(), key=lambda kv: (_is_total(kv[0]), -kv[1][0]["per"])):
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
    unemployment (y_table ur_annual), each with its own `x_where` and `y_where`: the correlation,
    the states that drive it, and whether the link is true by definition. Several groups (`by`, or
    the groups in x_where): a correlation within each, with the total; each group's settings
    apply to both figures. Describes states (people's chances there), not causes."""
    args = {"x_table": x_table, "y_table": y_table, "x_where": x_where, "y_where": y_where}
    groups = _custom(release.table(x_table), x_where)
    if groups:
        # like `by`: each group's settings apply to both figures
        if by or isinstance(y_where, list):
            raise Refusal("give the list of groups in x_where only (each applies to both figures), and no by")
        return _result("relate", args, _within(relate, release, groups,
                                               {"x_table": x_table, "y_table": y_table, "y_where": y_where},
                                               key="x_where", also="y_where"))
    x_where, by, cx = _listed(release.table(x_table), x_where, by)
    y_where, by, cy = _listed(release.table(y_table or x_table), y_where, by)
    if cx and cy and set(cx) != set(cy):
        raise Refusal("list the same groups in x_where and y_where, or in one of them only")
    if by:
        return _relate_by(release, x_table, y_table, x_where, y_where, by, cx or cy, {**args, "by": by})
    tx, sx, ty, sy = _sides(release, x_table, y_table, x_where, y_where)
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


def _relate_by(release, x_table, y_table, x_where, y_where, by, chosen, args):
    """relate within each group of `by` (the chosen ones, if listed) and the total: one
    correlation per group, the states' figures left out (relate without `by` lists them)."""
    tx = release.table(x_table)
    ty = release.table(y_table or x_table)
    if by == "state":
        raise Refusal("relate compares states already; give by another field, e.g. gender")
    for t, where in ((tx, x_where), (ty, y_where)):
        _field(t, by, where, "compared")
    groups = [g for g in tx.values[by] if g != TOTAL.get(by) and g in ty.values[by]
              and (chosen is None or g in chosen)]
    if by in TOTAL:
        groups.append(TOTAL[by])
    lines = [f"x: {tx.describe(tx.settings(x_where, vary=['state', by]), ['state', by])}",
             f"y: {ty.describe(ty.settings(y_where, vary=['state', by]), ['state', by])}"]
    done, refusals = 0, []
    for g in groups:
        name = _total(by) if g == TOTAL.get(by) else g
        try:
            gx, sx, gy, sy = _sides(release, x_table, y_table, {**(x_where or {}), by: g}, {**(y_where or {}), by: g})
        except Refusal as e:
            refusals.append(e)
            lines.append(f"{name}: not published ({_why(e)})")
            continue
        c = compute.relate(gx, sx, gy, sy)
        if c.get("too_few") or c.get("pearson") is None or c.get("spearman") is None:
            lines.append(f"{name}: too few states with both figures, or no variation: no correlation")
            continue
        done += 1
        lines.append(f"{name}: {c['n']} states, r = {c['pearson']:.2f}, rho = {c['spearman']:.2f}; "
                     f"{compute.strength(c['pearson'], c['n'])}"
                     + ("" if c["agree"] else "; the two disagree: a few states pull r")
                     + (f"; BUILT IN, not a finding: {c['built_in']}" if c["built_in"] else ""))
    if refusals and len(refusals) == len(groups):
        raise refusals[0]                # as for one group: nearest() may answer on the nearest published
    if not done:
        raise Refusal(f"no group of {by} has both figures for enough states")
    lines.append("this describes states (people's chances in them), not causes")
    return _result("relate", args, lines)


# ---------------------------------------------------------------- where to look next

def drill(release, table: str, where: Where = None) -> str:
    """Which field matters most? When you don't know where to look: for every field not set in
    `where`, how far apart its groups are, widest first; then compare or gap on that field.
    Groups that differ in more than one field (where a list of groups): the same within each."""
    t = release.table(table)
    groups = _custom(t, where)
    if groups:
        return _result("drill", {"table": table, "where": where}, _within(drill, release, groups, {"table": table}))
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
    given = args.get(keys[0]) or {}
    wheres = given if isinstance(given, list) else [given]          # several groups: the first's settings
    # a value with lists (several groups or periods at once): the setting from its single values
    s = first.settings({k: v for k, v in wheres[0].items() if not isinstance(v, list)})
    spans_periods = any(isinstance(w.get("period"), list) or isinstance(w.get("year"), list)
                        or ({"period", "year"} & set(w) and isinstance(given, list)) for w in wheres)
    if analysis not in ("series", "change", "average", "trend") and not spans_periods:   # these span periods
        other = _other_period(first, s.period)
        if other:
            yield f"in {other}", {**args, **{k: _set(args.get(k), period=other) for k in keys}}
    measures = first.values.get("measure", [])
    if len(measures) > 1:
        other = next(m for m in measures if m != s.values["measure"])
        yield f"on {MEASURES.get(other, other)}", {**args, **{k: _set(args.get(k), measure=other) for k in keys}}


def _set(where, **settings):
    """`where` with these settings, in every group when it is a list of groups."""
    if isinstance(where, list):
        return [{**w, **settings} for w in where]
    return {**(where or {}), **settings}


ONLY_FOR = re.compile(r"^(\w+): .*published only for (age|measure) (.+)$")
# the note `nearest` puts under a result's id; the analyst's hand-in is checked against it
NOTE = re.compile(r"^note: these figures are for (age|measure) (.+?), not (.+?) as asked", re.M)
WHERES = (("where", "table"), ("x_where", "x_table"), ("y_where", "y_table"))


def nearest(release, analysis, kwargs: dict, tries=2, allow=("age", "measure")) -> str:
    """The analysis, or, when what was asked is published only for another age group or measure,
    the same analysis on the nearest one that is, saying so in its last line. An eval run: told
    "this split is published only for age 15 years and above", an analyst reported that the
    figures by education "are not published", and the brief said MoSPI doesn't publish them."""
    try:
        return analysis(release, **kwargs)
    except Refusal as refused:
        found = ONLY_FOR.match(str(refused))
        if not found or tries == 0 or found[2] not in allow:
            raise
        table, field, offered = found[1], found[2], [v.strip() for v in found[3].split(", ")]
        used = DEFAULTS[field] if DEFAULTS[field] in offered else offered[0]
        changed, asked = dict(kwargs), None
        for where_key, table_key in WHERES:
            this = kwargs.get(table_key) or (kwargs.get("x_table") if where_key == "y_where" else None)
            if this == table and where_key in analysis.__code__.co_varnames:
                given = kwargs.get(where_key) or {}
                first = given[0] if isinstance(given, list) and given and isinstance(given[0], dict) else given
                asked = asked or (first.get(field, DEFAULTS[field]) if isinstance(first, dict) else None)
                changed[where_key] = _set(given, **{field: used})
        if asked is None or asked == used:
            raise
        text = nearest(release, analysis, changed, tries - 1, allow)
        # first, under the id: at the end, two span-test analysts used the figures and wrote
        # claims that never said they were for another age group and measure than asked
        head, _, rest = text.partition("\n")
        return (f"{head}\nnote: these figures are for {field} {used}, not {asked} as asked "
                f"({asked} isn't published for this)\n{rest}")


def recheck(release, analysis: str, args: dict) -> str:
    """The same analysis in the other period and on the other measure, to see whether a finding
    holds there. Run by code on every finding (agents/checks.py), not offered to the analysts."""
    if analysis not in RECHECKABLE:
        raise Refusal(f"can't recheck {analysis}; only {', '.join(RECHECKABLE)}")
    lines = []
    for label, changed in _variants(release, analysis, args):
        try:
            # the rerun on the other measure may not be moved back to the first one: that is its point
            allow = ("age",) if label.startswith("on ") else ("age", "measure")
            lines.append(f"{label}:\n" + nearest(release, RECHECKABLE[analysis], changed, allow=allow).split("\n", 1)[1])
        except Refusal as e:
            lines.append(f"{label}: {e}")
    return _result("recheck", {"analysis": analysis, "args": args},
                   lines or ["nothing to recheck: no other period or measure is stored for this"])


RECHECKABLE = {f.__name__: f for f in (value, compare, gap, change, series, average, trend, relate, drill)}
ANALYSES = [tables, profile, value, compare, gap, change, series, average, trend, relate, drill, recheck]
