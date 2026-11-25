"""Profiles: what code finds striking in one release. Fixed code, no model, run when MoSPI
publishes. Built from the stored tables with the analysis tools' own calculations
(analysis/compute.py), so a profile and an agent's drill-down can never disagree: the profile is
those analyses run at the default settings, with the most striking results kept.

Annual profile (per indicator; latest year, usual status, age 15+, everyone):
  headline       MoSPI's total; on the other measure; the previous year
  gaps           every group of every field with a total, vs the total
  interactions   how a gap (women-men, urban-rural, youth-all) changes across another field
  correlations   across states, within the indicator and between indicators; those true by
                 definition listed as not findings
  changes        since the previous year
Of each kind, the 3 most striking, ranked within the kind by its own measure: a 19-point gap,
r = 0.41 and a 6.5-point change share no scale, and the first version's single ranking put a
correlation true by definition (LFPR vs WPR, r = 0.99) at the top.

Monthly profile: the series, months not published, its trend, latest streak and latest month
against its own history (the trend and series tools' rules, so the lead sees a steady
rise without being asked); groups with a clear trend or a streak of 3+ months; changes vs the
previous month and a year earlier, the groups now, the widest interactions, the widest ranges
over the series.

What agents read is the text: a few hundred tokens per indicator.
"""

from ..analysis.compute import (against_history, changes, earlier, gaps_across, missing_months, num, pts,
                                relate, spread, streak, trend_line, trend_verdict)
from ..analysis.data import MEASURES, TOTAL
from ..datasets import DATASET

# MoSPI's indicator codes -> names, in code order (datasets/plfs.toml)
_BY_CODE = sorted(DATASET.indicators.items(), key=lambda kv: kv[1]["code"])
NAMES = {i["code"]: i["name"] for _, i in _BY_CODE}
SHORT = {i["code"]: k.upper() for k, i in _BY_CODE}

# a gap is a difference between two groups of one field
CONTRASTS = dict(DATASET.contrasts)
ACROSS, ACROSS_PLURAL = DATASET.across, DATASET.across_plural
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
            if other in (field, DATASET.measure):
                continue
            gaps = {k: x - y for k, (x, y) in gaps_across(t, s, field, a, b, other).items()}
            sp = spread(other, gaps)
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


BAND_SHORT = DATASET.dimensions[DATASET.band]["short"]
HEADLINE = f"headline ({TOTAL[ACROSS]}, everyone)"


def annual_text(t, correlations, unstated_age=None):
    s = t.settings()
    headline = t.value(s)
    measure = s.values[DATASET.measure]
    lines = [f"PROFILE {t.title}, {s.period}, {MEASURES.get(measure, measure)}, {DATASET.band}s {BAND_SHORT} unless stated",
             f"{HEADLINE}: {num(headline)}{DATASET.unit}"]
    for other in t.values[DATASET.measure]:
        rows = t.rows(s.with_(**{DATASET.measure: other})) if other != measure else []
        if rows:
            lines.append(f"on the other measure, {MEASURES.get(other, other)}: {num(rows[0]['value'])}% "
                         "-- never mix the two")
    refs = earlier(t, s.period)
    if refs:
        label, before = refs[0]
        rows = t.rows(s.with_(period=before))
        if rows:
            lines.append(f"{label} ({before}): {num(rows[0]['value'])}%")

    kinds = {GAPS: [], INTERACTIONS: [], CORRELATIONS: [], CHANGES: []}
    for f in t.fields:
        sp = spread(f, t.groups(s, f)) if f in TOTAL else None
        for k, x in (sp.rankable.items() if sp else ()):
            flag = " (statistical outlier: possibly a small, noisy sample)" if k in sp.outliers else ""
            kinds[GAPS].append((abs(x - headline), f"{f} = {k}: {num(x)}% vs {num(headline)}% overall "
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
        for f in t.fields:
            if f in TOTAL:
                for k, (b, n) in changes(t, s, f, refs[0][1]).items():
                    kinds[CHANGES].append((abs(n - b), f"{f} = {k}: {pts(n - b)} points since {refs[0][1]}"))

    for kind, items in _top(kinds).items():
        lines.append(f"{kind}:")
        lines += [f"  - {text}" for text in items]
    if built:
        lines.append("not findings (true by definition): " + "; ".join(built))
    missing_in = DATASET.dimensions[DATASET.band].get("missing_in")
    if unstated_age and missing_in in t.fields:
        lines.append(f"{missing_in} figures: {DATASET.publisher}'s table gives no {DATASET.band}; they match the "
                     f"{unstated_age} figures, so they are read as that band")
    lines.append(f"Across-{ACROSS} correlations describe {ACROSS_PLURAL} ({DATASET.describes}), not causes.")
    return "\n".join(lines)


def _over_time(series) -> list:
    """The series over time, with the trend and series tools' own rules: is it rising or falling
    along a straight line (judged against the months' scatter around it), the streak the latest
    month ends, and the latest month against the series' own history."""
    order = list(series)
    if len(order) < 4:
        return []
    line = trend_line(series)
    lines = [f"trend {order[0]} to {order[-1]}: {trend_verdict(line)} ({pts(line['over'])} points along a "
             f"straight line; the months usually sit {line['scatter']:.1f} from it)"]
    run = streak(series, monthly=True)
    if run and run["moves"] >= 2:
        lines.append(f"latest streak: {run['direction']} {run['moves']} months in a row ({run['from']} to {order[-1]})")
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
            run = streak(values, monthly=True)
            if run and run["moves"] >= 3:
                streaks.append((run["moves"], f"{f} = {level}: {run['direction']} {run['moves']} months in a row"))
    return [f"groups with a clear trend: " + ("; ".join(text for _, text in sorted(trends, key=lambda p: -p[0])[:PER_KIND])
                                              or "none"),
            f"groups on a streak of 3+ months: " + ("; ".join(text for _, text in sorted(streaks, key=lambda p: -p[0])[:PER_KIND])
                                                    or "none")]


def monthly_text(t, measure_note=None):
    s = t.settings()
    headline = t.value(s)
    series = {r["period"]: r["value"] for r in t.rows(s, all_periods=True)}
    lines = [f"PROFILE {t.title}, latest {s.period}, {DATASET.band}s {BAND_SHORT} unless stated",
             f"{HEADLINE}: {num(headline)}{DATASET.unit}"]
    if measure_note:
        lines.append(f"measure: {measure_note}")
    lines.append("series: " + ", ".join(f"{p} {num(x)}" for p, x in series.items()))
    if missing_months(list(series)):
        lines.append("not published: " + ", ".join(missing_months(list(series))))
    lines += _over_time(series)
    fields = [f for f in t.fields if f != DATASET.measure]
    lines += _groups_over_time(t, s, fields)
    for label, before in earlier(t, s.period):
        lines.append(f"vs {label} ({before}): {num(series[before])}% -> {num(headline)}%")
        for f in fields:
            moved = changes(t, s, f, before)
            if moved:
                lines.append(f"  {f}: " + ", ".join(f"{k} {pts(n - b)}" for k, (b, n) in moved.items()))
    lines.append("groups now: " + "; ".join(
        f"{f}: " + ", ".join(f"{k} {num(x)}" for k, x in t.groups(s, f).items()) for f in fields))
    for it in sorted(interactions(t, s), key=lambda i: -i["spread"].width)[:PER_KIND]:
        sp = it["spread"]
        lines.append(f"gap {it['label']} ({pts(it['overall'])}) across {it['across']}: "
                     + ", ".join(f"{k} {pts(g)}" for k, g in sp.named.items()))
    ranges = {}
    for f in fields:
        for level in t.groups(s, f):
            values = [r["value"] for r in t.rows(s.with_(**{f: level}), all_periods=True)]
            if len(values) >= 3:
                ranges[f"{f}={level}"] = max(values) - min(values)
    widest = sorted(ranges.items(), key=lambda kv: -kv[1])[:4]
    lines.append("widest range over the series: " + ", ".join(f"{k} {v:.1f}" for k, v in widest))
    return "\n".join(lines)


def build_profiles(release, unstated_age=None, measure_notes=None) -> dict:
    """{profile name: text} for every stored table."""
    annual = [t for t in release.tables.values() if not t.monthly]
    correlations = _correlations(annual)
    out = {f"{t.indicator.upper()}_annual_{t.periods[-1]}": annual_text(t, correlations[t.name], unstated_age)
           for t in annual}
    for t in release.tables.values():
        if t.monthly:
            out[f"{t.indicator.upper()}_monthly_{t.periods[-1]}"] = monthly_text(
                t, (measure_notes or {}).get(t.name))
    return out


def infer_monthly_measure(monthly_headlines: dict, annual_headlines: dict, year):
    """Which measure the monthly series uses -- MoSPI's API doesn't say, and its definition text
    for UR is a copy of the LFPR definition, so the text can't be trusted for this. The monthly
    headlines of `year` are compared with that year's annual figure on each measure; the nearer
    one wins. Returns (measure code, how it was inferred)."""
    values = [v for v in monthly_headlines.values() if v is not None]
    mean = sum(values) / len(values)          # only to locate the series; never reported as a rate
    nearest = min(annual_headlines, key=lambda m: abs(annual_headlines[m] - mean))
    return nearest, (f"the {len(values)} monthly headlines of {year} sit around {mean:.1f}%; the annual "
                     f"{year} figure is " + ", ".join(f"{v:g}% on {MEASURES.get(m, m)}"
                                                      for m, v in annual_headlines.items())
                     + "; the monthly series matches the nearer one (inferred from the numbers, "
                       "not stated by MoSPI)")
