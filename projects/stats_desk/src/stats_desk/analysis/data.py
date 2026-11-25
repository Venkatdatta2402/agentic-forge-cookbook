"""The release file as the analyses see it: its tables, their fields, and which published row a
request means.

The one rule everything here follows (verified on the data, see release/cells.py): a request
names some fields; every other field with a total is at its total, and the fields without one
get a stated default. So a request always points at MoSPI's own published rows. Nothing here
computes a rate; when a combination isn't published, the answer says what is.
"""

import re
from dataclasses import dataclass, field

import duckdb

from ..release.cells import TOTALS

# the value meaning "every group": person, rural + urban, All India, all
TOTAL = {f: min(v - {None}) for f, v in TOTALS.items()}
# fields with no total, and what a request that doesn't name them gets
DEFAULTS = {"age": "15 years and above", "measure": "PS+SS"}
MEASURES = {"PS+SS": "usual status (PS+SS)", "CWS": "current weekly status (CWS)"}
# groups that contain other groups of their field: shown, never ranked against them
CONTAINS = {"education": {"9.secondary & above": "contains 4.secondary to 8.post graduate"}}
# fields whose groups overlap or measure different things: shown side by side, never ranked
SIDE_BY_SIDE = {"age": "the bands overlap (15-29 is inside 15 years and above)",
                "measure": "two different measures; never mix them"}
TIME = ("year", "month", "period")


class Refusal(Exception):
    """The request can't be answered as asked, for a reason the agent can act on."""


def _period(value):
    return value.strftime("%Y-%m") if hasattr(value, "strftime") else str(value)


def _bare(value) -> str:
    """A value as matched against what was asked: no number in front ("7.graduate" -> "graduate"),
    any case, hyphens as spaces."""
    text = re.sub(r"^\s*\d+\s*\.\s*", "", str(value)).lower()
    return " ".join(re.sub(r"[-‐‑–]", " ", text).split())


@dataclass
class Settings:
    """Which published rows a request means: a value for every field, and the period."""
    values: dict
    period: str
    named: set = field(default_factory=set)      # fields the request set itself

    def with_(self, **changes):
        return Settings({**self.values, **{k: v for k, v in changes.items() if k != "period"}},
                        changes.get("period", self.period), self.named | set(changes))


class Table:
    def __init__(self, con, name, title):
        self.con, self.name, self.title = con, name, title
        self.indicator = name.split("_")[0]
        columns = [c for c, in con.execute(
            "SELECT column_name FROM duckdb_columns() WHERE table_name = ? ORDER BY column_index",
            [name]).fetchall()]
        self.monthly = "month" in columns
        self.fields = [c for c in columns if c not in (*TIME, "value")]
        self.values = {f: [v for v, in con.execute(f'SELECT DISTINCT "{f}" FROM "{name}" ORDER BY 1').fetchall()]
                       for f in self.fields}
        time = "strftime(period, '%Y-%m')" if self.monthly else "CAST(year AS VARCHAR)"
        self._time = time
        self.periods = [p for p, in con.execute(f'SELECT DISTINCT {time} FROM "{name}" ORDER BY 1').fetchall()]
        # which fields MoSPI publishes split together, for which ages and measures
        split = [f for f in self.fields if f in TOTAL]
        flags = ", ".join(f"\"{f}\" <> '{TOTAL[f]}'" for f in split) or "true"
        self.shapes = set()
        for *on, age, measure in con.execute(f'SELECT DISTINCT {flags}, age, measure FROM "{name}"').fetchall():
            self.shapes.add((frozenset(f for f, x in zip(split, on) if x), age, measure))

    # ---- what a request means ----

    def settings(self, where=None, vary=()) -> Settings:
        where = {k: str(v) for k, v in (where or {}).items()}
        period = where.pop("period", None) or (None if self.monthly else where.pop("year", None))
        for key in where:
            if key not in self.fields:
                raise Refusal(f"{self.name} has no field '{key}'. Its fields: {', '.join(self.fields)}; "
                              f"the time is set with 'period' ({self.periods[0]} to {self.periods[-1]})")
        values = {}
        for f in self.fields:
            if f in vary:
                continue
            if f in where:
                values[f] = self._check(f, where[f])
            elif f in TOTAL:
                values[f] = TOTAL[f]
            elif DEFAULTS.get(f) in self.values[f]:
                values[f] = DEFAULTS[f]
            elif len(self.values[f]) == 1:
                values[f] = self.values[f][0]
            else:
                raise Refusal(f"{self.name}: say which {f} ({', '.join(self.values[f])})")
        if period is not None:
            period = period[:7] if self.monthly else period[:4]
            if period not in self.periods:
                raise Refusal(f"{self.name} has no period {period}. Stored: {', '.join(self.periods)}")
        return Settings(values, period or self.periods[-1], set(where))

    def _check(self, f, value):
        """A value as published, or the one it plainly means: MoSPI numbers some groups
        ("7.graduate"), and an analyst asking for "graduate" means that one, not "8.post graduate
        & above" -- a live run lost two analysts to that refusal, each repeating it."""
        if value in self.values[f]:
            return value
        total = f"; its total is '{TOTAL[f]}' (or write total)" if f in TOTAL else "; it has no total"
        # every field's total has its own name (person, rural + urban, All India, all): "total"
        # works for any of them
        if str(value).strip().lower() in ("total", "overall"):
            if f in TOTAL and TOTAL[f] in self.values[f]:
                return TOTAL[f]
            raise Refusal(f"{self.name}: {f} has no total. Values: {', '.join(map(str, self.values[f]))}")
        want = _bare(value)
        exact = [v for v in self.values[f] if _bare(v) == want]
        close = exact or [v for v in self.values[f] if want and want in _bare(v)]
        if len(close) == 1:
            return close[0]
        if close:
            raise Refusal(f"{self.name}: {f} '{value}' could be any of {', '.join(map(str, close))}: "
                          "give one of them exactly as written here")
        listed = self.values[f] if len(self.values[f]) <= 40 else self.values[f][:40] + ["..."]
        # a value of another field put under this one (span test: "All ages", an age group, given
        # as the education to compare against, where education's total "all" was meant)
        other = next((g for g in self.fields if g != f and want
                      and any(_bare(v) == want for v in self.values[g])), None)
        if other:
            raise Refusal(f"{self.name}: '{value}' is a value of {other}, not of {f}{total}. "
                          f"Values of {f}: {', '.join(map(str, listed))}")
        raise Refusal(f"{self.name}: no {f} '{value}'{total}. Values: {', '.join(map(str, listed))}")

    def describe(self, s: Settings, vary=(), period=None):
        """One line saying exactly which figures these are."""
        parts = [f"{self.title}, {period or s.period}"]
        if "measure" not in vary and "measure" in s.values:
            parts.append(MEASURES.get(s.values["measure"], s.values["measure"]))
        if "age" not in vary and "age" in s.values:
            parts.append(f"age {s.values['age']}")
        chosen = [f"{f} = {v}" for f, v in s.values.items()
                  if f not in ("measure", "age") and v != TOTAL.get(f)]
        parts.append(", ".join(chosen) if chosen else "everyone, All India")
        if vary:
            parts.append(f"by {' x '.join(vary)}")
        return " | ".join(parts)

    # ---- reading published rows ----

    def rows(self, s: Settings, vary=(), all_periods=False):
        """Published rows for these settings, each field in `vary` free. [{field..., value}]"""
        where, params = [], []
        for f, v in s.values.items():
            if f not in vary:
                where.append(f'"{f}" = ?')
                params.append(v)
        if not all_periods:
            where.append(f"{self._time} = ?")
            params.append(s.period)
        cols = ", ".join([*(f'"{f}"' for f in vary), f"{self._time} AS period", "value"])
        sql = f'SELECT {cols} FROM "{self.name}" WHERE {" AND ".join(where) or "true"} ORDER BY period'
        cur = self.con.cursor()       # one cursor per query: tool calls may run in threads
        try:
            names = [*vary, "period", "value"]
            return [dict(zip(names, r)) for r in cur.execute(sql, params).fetchall()]
        finally:
            cur.close()

    def value(self, s: Settings):
        rows = self.rows(s)
        if not rows:
            raise Refusal(self.missing(s))
        return rows[0]["value"]

    def groups(self, s: Settings, f):
        """{group: value} for every group of `f` (the total, if any, left out), in MoSPI's order."""
        return {r[f]: r["value"] for r in self.rows(s, vary=[f]) if r[f] != TOTAL.get(f)}

    def combinations(self) -> str:
        """Which fields MoSPI splits together, in one line, or "" if every field goes with every
        other: e.g. "age groups 15-29 years, 15-59 years: only by gender, sector, state (measure
        PS+SS only); education: only by gender, sector, state (ages 15 years and above, measure
        PS+SS only)". In the tables list, which the lead plans from (span tests: without it, 2 of 3
        plans asked for young people by education, which MoSPI doesn't publish; with it, 1 of 3).
        The measure too: without it, a lead's task asked for education on CWS, published on PS+SS
        only."""
        splits = [(sp, age) for sp, age, _ in self.shapes]
        fields = {f for sp, _ in splits for f in sp}
        every = set(self.values.get("measure", []))

        def only(measures):                 # a measure a split is limited to, or ""
            return f"measure {', '.join(sorted(measures))}" if measures and measures != every else ""

        out = []
        by_age = {}
        for age in sorted({a for _, a in splits} - {DEFAULTS["age"]}):
            partners = frozenset(f for sp, a in splits if a == age for f in sp)
            measures = frozenset(m for _, a, m in self.shapes if a == age)
            by_age.setdefault((partners, measures), []).append(age)
        for (partners, measures), ages in by_age.items():
            if partners != fields or only(measures):
                out.append(f"age group{'s' if len(ages) > 1 else ''} {', '.join(ages)}: only by "
                           f"{', '.join(sorted(partners)) or 'nothing else (All India totals)'}"
                           + (f" ({only(measures)} only)" if only(measures) else ""))
        for f in sorted(fields):
            partners = {g for sp, _ in splits if f in sp for g in sp} - {f}
            ages = sorted({a for sp, a in splits if f in sp})
            measure = only({m for sp, _, m in self.shapes if f in sp})
            limits = ([f"ages {ages[0]}"] if ages == [DEFAULTS["age"]] else []) + ([measure] if measure else [])
            if partners != fields - {f} or limits:
                out.append(f"{f}: only by {', '.join(sorted(partners)) or 'itself'}"
                           + (f" ({', '.join(limits)} only)" if limits else ""))
        return "; ".join(out)

    def missing(self, s: Settings, vary=()):
        """Why nothing is published for these settings, and what is."""
        split = frozenset(f for f in self.fields if f in TOTAL and (f in vary or s.values.get(f) != TOTAL[f]))
        same = [(age, m) for sp, age, m in self.shapes if sp == split]
        if not same:
            widest = sorted({sp for sp, _, _ in self.shapes if sp}, key=lambda sp: (-len(sp), sorted(sp)))
            maximal = [sp for sp in widest if not any(sp < other for other in widest)]
            return (f"{self.name}: {' x '.join(sorted(split))} is not published together. Published "
                    "together: " + "; ".join(" x ".join(sorted(sp)) for sp in maximal))
        ages = sorted({a for a, m in same if m == s.values.get("measure")})
        measures = sorted({m for a, m in same})
        if s.values.get("measure") not in measures:
            return f"{self.name}: {self.describe(s, vary)} is published only for measure {', '.join(measures)}"
        if s.values.get("age") not in ages:
            return f"{self.name}: this split is published only for age {', '.join(ages)}"
        return f"{self.name}: nothing published for {self.describe(s, vary)}"


class Release:
    """The release file, opened read-only."""

    def __init__(self, path):
        self.con = duckdb.connect(str(path), read_only=True)
        self.cards = dict(self.con.execute("SELECT name, card FROM _cards ORDER BY name").fetchall())
        self.profiles = dict(self.con.execute("SELECT name, profile FROM _profiles ORDER BY name").fetchall())
        self.released = self.con.execute("SELECT released FROM _release").fetchone()[0]
        self.tables = {name: Table(self.con, name, card.split("\n", 1)[0].split(": ", 1)[1])
                       for name, card in self.cards.items()}

    def table(self, name) -> Table:
        if name not in self.tables:
            raise Refusal(f"no table '{name}'. Tables: {', '.join(self.tables)}")
        return self.tables[name]

    def close(self):
        self.con.close()
