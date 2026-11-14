"""Small statistics, in plain Python so every number can be followed by hand.

Everything here works on published RATES, so nothing averages them: a group's rate is MoSPI's
own row for that group, and these functions only find far-off groups and measure co-movement.
"""

import math
from statistics import median


def outliers(named: dict, threshold=3.5):
    """Names whose value is far from the rest: |robust z| above `threshold`, where
    robust z = 0.6745 * (x - median) / MAD (Iglewicz & Hoaglin). Needs at least 5 values."""
    if len(named) < 5:
        return []
    values = list(named.values())
    mid = median(values)
    mad = median(abs(v - mid) for v in values)
    if mad == 0:
        return []
    return sorted((k for k, v in named.items() if abs(0.6745 * (v - mid) / mad) > threshold),
                  key=lambda k: -abs(named[k] - mid))


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def ranks(values):
    """Ranks starting at 1, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def correlate(a: dict, b: dict, min_n=10, disagree=0.25):
    """Ordinary (Pearson) and rank (Spearman) correlation over the names both share.

    Both are always computed (it's free). When they disagree by more than `disagree`, a few
    names are pulling the ordinary one; they are named by leave-one-out influence: how much
    the Pearson correlation moves when each name is dropped.
    """
    names = sorted(set(a) & set(b))
    out = {"n": len(names)}
    if len(names) < min_n:
        out["too_few"] = True
        return out
    xs, ys = [a[k] for k in names], [b[k] for k in names]
    r, rho = pearson(xs, ys), spearman(xs, ys)
    out.update({"pearson": r, "spearman": rho})
    if r is None or rho is None:
        return out
    influence = []
    for i, name in enumerate(names):
        without = pearson(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
        if without is not None:
            influence.append((abs(r - without), name, without))
    influence.sort(reverse=True)
    out["most_influential"] = [{"name": n, "pearson_without": round(w, 3)} for _, n, w in influence[:3]]
    out["agree"] = abs(r - rho) <= disagree
    if not out["agree"]:
        out["driven_by"] = [n for d, n, _ in influence[:3] if d > 0.1]
    return out
