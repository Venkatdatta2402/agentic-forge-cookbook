"""Profiles: what code finds striking in one release. Fixed code, no model, run when MoSPI
publishes. Built from the stored tables with the analysis tools' own calculations
(analysis/compute.py), so a profile and an agent's drill-down can never disagree: the profile is
those analyses run at the default settings, with the most striking results kept.

One writer for every table of every dataset (`profile`), each section written when the table
has what it needs:

  every table    its title and latest period; each setting it offers a choice of, at its
                 default (a measure by name, a band "unless stated"); the headline, every field
                 at its total, in its unit; the same on each other value of a measure (never
                 mix them); what its source found that its figures need said (`notes`: PLFS's
                 band of the age-less rows, its monthly measure worked out from the numbers)
  a series       (several periods a year: monthly, quarterly) the series, periods not published,
                 a change of base inside it, its trend, latest streak and latest period against
                 its own history (the trend and series tools' rules, so the lead sees a steady
                 rise without being asked); groups with a clear trend or a streak of 3+; changes
                 vs each earlier period; every group now; the widest gaps across another field;
                 the widest ranges over the series
  one period     (annual) the previous year, then the 3 most striking of each kind: gaps from
  a year         the total; gaps (women-men, urban-rural, youth-all) that change across another
                 field; correlations across states, within the indicator and between indicators,
                 those true by definition listed as not findings (a table with states); changes
                 since the previous year. Ranked within the kind by its own measure: a 19-point
                 gap, r = 0.41 and a 6.5-point change share no scale, and the first version's
                 single ranking put a correlation true by definition (LFPR vs WPR, r = 0.99) first.
  a hierarchy    only its top and finest levels: the ones between repeat the same movers

What agents read is the text: a few hundred tokens per indicator.
"""

from ..analysis.compute import (against_history, changes, earlier, gaps_across, missing, num, pts,
                                relate, spread, streak, trend_line, trend_verdict)
from ..analysis.data import DEFAULTS, MEASURES, TOTAL
from ..datasets import CATALOG

# a gap is a difference between two groups of one field
CONTRASTS = {f: c for ds in CATALOG.datasets for f, c in ds.contrasts}     # every dataset's, a field's the same in each
ACROSS, ACROSS_PLURAL = CATALOG.across, CATALOG.across_plural
GAPS, INTERACTIONS = "largest gaps from the total", "gaps that change most across another field"
CORRELATIONS, CHANGES = f"strongest correlations across {ACROSS_PLURAL}", "biggest changes since last release"
PER_KIND = 3


def interactions(t, s):
    """For each contrast, how its gap changes across each other field."""
    out = []
    for field, (a, b, label) in CONTRASTS.items():
        if field not in t.fields:
            continue
        ra, rb = t.rows(s.with_(**{field: a})), t.rows(s.with_(**{field: b}))
        if not ra or not rb:
            continue
        for other in t.fields:
            if other == field or other in t.measures:
                continue
            gaps = {k: x - y for k, (x, y) in gaps_across(t, s, field, a, b, other).items()}
            sp = spread(other, gaps, t)
            if sp:
                out.append({"label": label, "across": other, "overall": ra[0]["value"] - rb[0]["value"],
                            "spread": sp})
    return out


def _interaction_text(it):
    sp = it["spread"]
    text = (f"The gap {it['label']} ({pts(it['overall'])} points overall) varies across {it['across']}: "
            f"from {pts(sp.core[sp.lo])} ({sp.lo}) to {pts(sp.core[sp.hi])} ({sp.hi})")
    if sp.outliers:
        text += "; set apart as outliers (possibly small, noisy samples): " + "; ".join(
            f"{k} {pts(sp.named[k])}" for k in sp.outliers)
    return text


def _top(kinds):
    out = {}
    for kind, items in kinds.items():
        top = []
        for _, text in sorted(items, key=lambda i: -i[0]):
            if text not in top:
                top.append(text)
            if len(top) == PER_KIND:
                break
        if top:
            out[kind] = top
    return out


def _correlations(tables):
    """{table name: [(x label, y label, relate result)]}: within each indicator (women vs men,
    urban vs rural, youth vs 15+) and between indicators, each point a state."""
    out = {t.name: [] for t in tables}
    for t in tables:
        s = t.settings()
        for field, (a, b, _) in CONTRASTS.items():
            if field in t.fields:
                out[t.name].append((f"{t.indicator.upper()} {a}", f"{t.indicator.upper()} {b}",
                                    relate(t, s.with_(**{field: a}), t, s.with_(**{field: b}))))
    for i, tx in enumerate(tables):
        for ty in tables[i + 1:]:
            entry = (tx.indicator.upper(), ty.indicator.upper(), relate(tx, tx.settings(), ty, ty.settings()))
            out[tx.name].append(entry)
            out[ty.name].append(entry)
    return out


def _said(t, s) -> list:
    """The settings the table offers a choice of, at the value its figures are on: a measure by
    its name, a band "unless stated" (["usual status (PS+SS)", "ages 15+ unless stated"]). One
    value only (PLFS's monthly measure) is no choice: the source's note says what it is."""
    out = [MEASURES.get(s.values[m], s.values[m]) for m in t.measures if len(t.values[m]) > 1]
    return out + [f"{b}s {CATALOG.dimensions[b].get('short', DEFAULTS[b])} unless stated"
                  for b in t.bands if len(t.values[b]) > 1]


def _fields(t) -> list:
    """The fields a profile goes through: not the measures (never mixed), and of a hierarchy only
    its top and finest levels (CPI's first profile named jewellery at its group, class,
    sub_class and item, 1.7k tokens)."""
    middle = {f for f in t.within if f in t.within.values()}
    return [f for f in t.fields if f not in t.measures and f not in middle]


def profile(t, correlations=(), notes=()) -> str:
    """The profile of one table, of any dataset (the module's docstring lists its sections).
    `correlations`: its relate results across states (`_correlations`), for a table with states;
    `notes`: lines its source asks the profile to say (sources/<key>.py, `Built.notes`)."""
    s = t.settings()
    headline = t.value(s)
    series = t.freq.within_year
    everyone = CATALOG.dataset_of(t.indicator).everyone
    lines = [", ".join([f"PROFILE {t.title}, {'latest ' if series else ''}{s.period}", *_said(t, s)]),
             f"headline ({TOTAL[ACROSS]}, {everyone}): {num(headline)}{t.unit}"]
    for m in t.measures:
        for other in t.values[m]:
            rows = t.rows(s.with_(**{m: other})) if other != s.values[m] else []
            if rows:
                lines.append(f"on the other measure, {MEASURES.get(other, other)}: {num(rows[0]['value'])}{t.unit} "
                             "-- never mix the two")
    lines += list(notes)
    lines += _series(t, s, headline) if series else _latest(t, s, headline, correlations)
    return "\n".join(lines)


def _latest(t, s, headline, correlations) -> list:
    """A table of one period a year: the previous year, and the most striking of each kind."""
    lines = []
    unit = t.unit
    refs = earlier(t, s.period)
    if refs:
        label, before = refs[0]
        rows = t.rows(s.with_(period=before))
        if rows:
            lines.append(f"{label} ({before}): {num(rows[0]['value'])}{unit}")
    fields = _fields(t)
    kinds = {GAPS: [], INTERACTIONS: [], CORRELATIONS: [], CHANGES: []}
    for f in fields:
        sp = spread(f, t.groups(s, f), t) if f in TOTAL else None
        for k, x in (sp.rankable.items() if sp else ()):
            flag = " (statistical outlier: possibly a small, noisy sample)" if k in sp.outliers else ""
            kinds[GAPS].append((abs(x - headline), f"{f} = {k}: {num(x)}{unit} vs {num(headline)}{unit} overall "
                                                   f"({pts(x - headline)} points){flag}"))
    for it in interactions(t, s):
        kinds[INTERACTIONS].append((it["spread"].width, _interaction_text(it)))
    built = []
    for x, y, c in correlations:
        if c.get("too_few") or c.get("pearson") is None:
            continue
        if c["built_in"]:
            built.append(f"{x} vs {y} r = {c['pearson']:.2f}, because {c['built_in']}")
            continue
        if abs(c["pearson"]) < 2 / c["n"] ** 0.5:      # this small often appears by chance: not striking
            continue
        note = ("" if c["agree"] else f" -- rank correlation {c['spearman']:.2f} disagrees; driven by "
                + ", ".join(c.get("driven_by") or [f"a few {ACROSS_PLURAL}"]))
        kinds[CORRELATIONS].append((abs(c["pearson"]) * (1 if c["agree"] else 0.5),
                                    f"{x} and {y}: r = {c['pearson']:.2f} over {c['n']} {ACROSS_PLURAL}{note}"))
    if refs:
        for f in fields:
            if f in TOTAL:
                for k, (b, n) in changes(t, s, f, refs[0][1]).items():
                    kinds[CHANGES].append((abs(n - b), f"{f} = {k}: {pts(n - b)} points since {refs[0][1]}"))

    for kind, items in _top(kinds).items():
        lines.append(f"{kind}:")
        lines += [f"  - {text}" for text in items]
    if built:
        lines.append("not findings (true by definition): " + "; ".join(built))
    if ACROSS in t.fields:
        describes = CATALOG.dataset_of(t.indicator).describes
        lines.append(f"Across-{ACROSS} correlations describe {ACROSS_PLURAL} ({describes}), not causes.")
    return lines


def _over_time(series, unit) -> list:
    """The series over time, with the trend and series tools' own rules: is it rising or falling
    along a straight line (judged against the periods' scatter around it), the streak the latest
    period ends, and the latest period against the series' own history. `unit`: month, quarter."""
    order = list(series)
    if len(order) < 4:
        return []
    line = trend_line(series)
    lines = [f"trend {order[0]} to {order[-1]}: {trend_verdict(line)} ({pts(line['over'])} points along a "
             f"straight line; the {unit}s usually sit {line['scatter']:.1f} from it)"]
    run = streak(series)
    if run and run["moves"] >= 2:
        lines.append(f"latest streak: {run['direction']} {run['moves']} {unit}s in a row ({run['from']} to {order[-1]})")
    history = against_history(series)
    if history:
        lines.append(f"latest ({order[-1]}): {history}")
    return lines


def _groups_over_time(t, s, fields) -> list:
    """Groups (one field at a time) with a clear trend, and groups on a streak of 3+ months: the
    PER_KIND most striking of each (a trend by its change along the line against its scatter)."""
    trends, streaks = [], []
    for f in fields:
        for level in t.groups(s, f):
            values = {r["period"]: r["value"] for r in t.rows(s.with_(**{f: level}), all_periods=True)}
            if len(values) < 4:
                continue
            line = trend_line(values)
            verdict = trend_verdict(line)
            if verdict != "no clear trend":
                trends.append((abs(line["over"]) / max(line["scatter"], 1e-9),
                               f"{f} = {level}: {verdict} ({pts(line['over'])} along the line)"))
            run = streak(values)
            if run and run["moves"] >= 3:
                streaks.append((run["moves"], f"{f} = {level}: {run['direction']} {run['moves']} {t.freq.unit}s in a row"))
    return [f"groups with a clear trend: " + ("; ".join(text for _, text in sorted(trends, key=lambda p: -p[0])[:PER_KIND])
                                              or "none"),
            f"groups on a streak of 3+ {t.freq.unit}s: " + ("; ".join(text for _, text in sorted(streaks, key=lambda p: -p[0])[:PER_KIND])
                                                    or "none")]


SHOWN, SERIES_SHOWN = 12, 24      # a field with more groups is cut to its ends; a series to its last months


def _listed(named: dict, fmt, by=None) -> str:
    """Every group, or a long field's highest and lowest PER_KIND (CPI's 358 items): a profile is
    a few hundred tokens."""
    if len(named) <= SHOWN:
        return ", ".join(fmt(k, x) for k, x in named.items())
    order = sorted(named, key=lambda k: by(named[k]) if by else named[k])
    ends = [*reversed(order[-PER_KIND:]), f"... {len(named) - 2 * PER_KIND} more ...", *reversed(order[:PER_KIND])]
    return ", ".join(k if k.startswith("... ") else fmt(k, named[k]) for k in ends)


def _series(t, s, headline) -> list:
    """A table of several periods a year: how it moves, and how its groups do."""
    lines = []
    series = {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}
    shown = list(series)[-SERIES_SHOWN:]
    lines.append(("series: " if len(shown) == len(series) else f"series (from {list(series)[0]}; the last {len(shown)}): ")
                 + ", ".join(f"{p} {num(series[p])}" for p in shown))
    if missing(list(series)):
        lines.append("not published: " + ", ".join(missing(list(series))))
    join = CATALOG.indicators.get(t.indicator, {}).get("join")
    if join and list(series)[0] < join["at"] <= list(series)[-1]:
        lines.append(f"note: across {join['at'][:4]}, {join['note']}")
    lines += _over_time(series, t.freq.unit)
    fields = _fields(t)
    lines += _groups_over_time(t, s, fields)
    for label, before in earlier(t, s.period):
        lines.append(f"vs {label} ({before}): {num(series[before])}{t.unit} -> {num(headline)}{t.unit}")
        for f in fields:
            moved = changes(t, s, f, before)
            if moved:
                lines.append(f"  {f}: " + _listed({k: n - b for k, (b, n) in moved.items()},
                                                   lambda k, d: f"{k} {pts(d)}", by=abs))
    lines.append("groups now: " + "; ".join(
        f"{f}: " + _listed(t.groups(s, f), lambda k, x: f"{k} {num(x)}") for f in fields if t.groups(s, f)))
    for it in sorted((i for i in interactions(t, s) if i["across"] in fields), key=lambda i: -i["spread"].width)[:PER_KIND]:
        sp = it["spread"]
        lines.append(f"gap {it['label']} ({pts(it['overall'])}) across {it['across']}: "
                     + _listed(sp.named, lambda k, g: f"{k} {pts(g)}"))
    ranges = {}
    for f in fields:
        for level in t.groups(s, f):
            values = [r["value"] for r in t.rows(s.with_(**{f: level}), all_periods=True)]
            if len(values) >= 3:
                ranges[f"{f}={level}"] = max(values) - min(values)
    widest = sorted(ranges.items(), key=lambda kv: -kv[1])[:4]
    lines.append("widest range over the series: " + ", ".join(f"{k} {v:.1f}" for k, v in widest))
    return lines


def build_profiles(release, notes=None) -> dict:
    """{profile name: text} for every stored table: those of one period a year first, then the
    series. `notes`: {table: [line]}, what each table's source asks its profile to say."""
    tables = list(release.tables.values())
    # correlations across states: among the tables of one period a year that have states
    correlations = _correlations([t for t in tables if not t.freq.within_year and ACROSS in t.fields])
    out = {}
    for t in sorted(tables, key=lambda t: t.freq.within_year):
        out[f"{t.indicator.upper()}_{t.freq.name}_{t.periods[-1]}"] = profile(
            t, correlations.get(t.name, ()), (notes or {}).get(t.name, ()))
    return out
