"""Code's checks on every finding, before the lead reads it. Fired by code, not left to an agent's
choice (chapter 6: agents skip optional steps).

This was the skeptic's job. On live runs 1-2 a skeptic with tools re-fetched figures it had been
shown or re-ran the analyst's own analysis in other words, caught nothing, and cost ~45% of a
question's tokens; with code gathering its evidence it judged in one call, but the lead reads
every finding and decides what to do next anyway. So code gathers the
evidence and the lead judges it:
  - every cited analysis rerun in the other period and on the other measure (`recheck`)
  - warnings: figures compared that differ in more than one field, a state set apart as an
    outlier named in the claim, a correlation true by definition, cause words, and a gap or a
    correlation that reverses in the rerun or is half as large or less there (`held`)
"""

import re
from itertools import combinations

from ..core import events
from ..core.tools import ToolError
from ..datasets import DATASET
from .analyst import Finding, compact

CAUSE_WORDS = re.compile(r"\b(because|due to|caus\w*|leads? to|led to|results? in|as a result|"
                         r"owing to|explains?|drives?)\b", re.I)
NEGATION = re.compile(r"\b(not|no|cannot|can't|don't|doesn't|do not|does not|nor|without)\b", re.I)
STATISTICAL = re.compile(r"\b(sample|definition|defined|rounding|computed|calculated)\b", re.I)
MOVE_WORDS = re.compile(r"\b(fell|falls?|falling|rose|rises?|rising|declin\w*|increas\w*|decreas\w*|drop\w*|"
                        r"grew|grow\w*|widen\w*|narrow\w*|upward|downward)\b", re.I)


def _spaced(text):
    """Every kind of space as one: gpt-oss writes "Andaman &" + a narrow no-break space + "Nicobar"."""
    return re.sub(r"\s+", " ", text or "")


def cause_claims(text):
    """The cause words a text uses to claim a cause: not those it denies ("cannot explain why",
    "not because ...", "not a causal effect") nor those giving a statistical reason ("due to its
    small sample"). Matched word by word, these four sentences of the span tests' briefs
    were all flagged."""
    found = set()
    for clause in re.split(r"[.;:,]\s+|\bbut\b", text or ""):
        for m in CAUSE_WORDS.finditer(clause):
            before, after = clause[:m.start()], clause[m.end():m.end() + 40]
            if NEGATION.search(before) or STATISTICAL.search(after):
                continue
            found.add(m[1].lower())
    return sorted(found)


def flags(analyses, finding: Finding) -> list[str]:
    """What code can see is wrong or worth a look, without a model."""
    out = []
    cited = [(i, *analyses.calls.get(i, ("", {}))) for i in finding.results]
    values = [(i, args) for i, name, args in cited if name == "value"]
    for (i, a), (j, b) in combinations(values, 2):
        wa, wb = a.get("where") or {}, b.get("where") or {}
        # several groups in one value say themselves when they aren't like with like
        if a.get("table") == b.get("table") and isinstance(wa, dict) and isinstance(wb, dict):
            differ = sorted(k for k in set(wa) | set(wb) if str(wa.get(k)) != str(wb.get(k)))
            if len(differ) > 1:
                out.append(f"{i} and {j} differ in {', '.join(differ)}: not like with like; don't present "
                           "them as a gap between two groups")
    for i, _, _ in cited:
        result = analyses.results.get(i, "")
        if "BUILT IN" in result:
            out.append(f"{i} is a correlation TRUE BY DEFINITION ("
                       + result.split("BUILT IN, not a finding: ")[1].split("\n")[0]
                       + "): it is not a finding; say so instead")
        named = [s for s in outliers_in(result) if _spaced(s) in _spaced(finding.claim)]
        if named:
            out.append(f"the claim names {', '.join(named)}, set apart in {i} as a statistical outlier: "
                       "if you name it, say it may rest on a small sample")
        # eval: the trend said "no clear trend", the claim and the brief "fell by 0.2"
        moved = MOVE_WORDS.findall(finding.claim)
        # not when the claim itself says there is none (eval: "neither sector showing a clear
        # rising or falling trend" was flagged)
        says_none = re.search(r"\b(no clear|neither|not (?:a |any )?(?:clear|rising|falling|rise|fall))", finding.claim, re.I)
        if moved and not says_none and re.search(r"(^|: )no clear trend", result, re.M):
            out.append(f"{i} finds no clear trend (the change along its straight line is within the usual "
                       f"scatter), but the claim says {', '.join(repr(w) for w in dict.fromkeys(moved))}: say "
                       "there is no clear trend; a change between two months is not a trend")
    words = cause_claims(finding.claim)
    if words:
        out.append(f"the claim says {', '.join(repr(w) for w in words)}: the figures show what, not why; "
                   "don't repeat it")
    return out


def outliers_in(result):
    """The groups a result set apart as statistical outliers."""
    return [s for line in result.split("\n") if "statistical outliers" in line
            for s in re.findall(r"([A-Z][\w &]+?) [+-]?\d", line.split("small samples): ")[-1])]


# the one number that says which way a result points: a gap's overall size, a correlation;
# with what it is ("women minus men", "the correlation across states")
DIRECTION = {"gap": re.compile(r"overall: .*?; (?P<what>.+? minus .+?) = (?P<n>[+-]?\d+(?:\.\d+)?) points"),
             "relate": re.compile(r"ordinary correlation r = (?P<n>-?\d+(?:\.\d+)?)")}
HEADER = re.compile(r"^((?:in|on) .+):$")


def _setting(result):
    """Which year and measure a result is for, from its description line: "2025, usual status"."""
    parts = result.split("\n")[1].removeprefix("x: ").split(" | ")
    return f"{parts[0].rsplit(', ', 1)[-1]}, {parts[1]}" if len(parts) > 1 else parts[0]


def held(analysis, result, recheck):
    """Where a rerun points the other way, or is half as large or less, said outright, e.g.
    "reversed in 2024: women minus men was +3.1 points in the finding (2025, usual status
    (PS+SS)), -1.7 points in 2024". Whether "smaller but the same way" still holds stays the
    lead's judgement."""
    pattern = DIRECTION.get(analysis)
    found = pattern.search(result) if pattern else None
    if found is None:
        return []
    now = float(found["n"])
    what, unit = (found["what"], " points") if analysis == "gap" else (f"the correlation across {DATASET.across_plural}", "")
    out, label, body = [], None, []
    for line in recheck.split("\n")[1:] + ["in end:"]:
        header = HEADER.match(line)
        if header:
            there = pattern.search("\n".join(body)) if label else None
            if there and abs(now) >= 0.05:
                there = float(there["n"])
                said = (f"{what} was {now:+g}{unit} in the finding ({_setting(result)}), "
                        f"{there:+g}{unit} {label}")
                if there * now < 0 and abs(there) >= 0.05:
                    out.append(f"reversed {label}: {said}; the finding doesn't hold there, so say so")
                elif abs(there) <= abs(now) / 2:
                    out.append(f"half as large or less {label}: {said}; say it is weaker there")
            label, body = header[1], []
        else:
            body.append(line)
    return out


async def check(analyses, finding: Finding) -> Finding:
    """Reruns and warnings, put on the finding."""
    turned = []
    for i in finding.results:
        analysis, args = analyses.calls.get(i, (None, None))
        if analysis is None:
            continue
        events.emit("check", finding=finding.id, analysis=analysis, arguments=args)
        try:
            text = await analyses.call("recheck", analysis=analysis, args=args)
        except ToolError:
            continue          # tables: nothing to recheck
        rid = text.split("\n")[0].removeprefix("result ")
        finding.checked_with.append(rid)
        finding.elsewhere[rid] = compact(text)
        turned += [f"{i}: {w}" for w in held(analysis, analyses.results[i], text)]
    # added to any it came with (a finding handed in by code says so first)
    finding.flags = list(dict.fromkeys(finding.flags + flags(analyses, finding) + turned))
    return finding
