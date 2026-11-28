"""The release file as the analyses see it: its tables, their fields, and which published row a
request means.

The one rule everything here follows (verified on the data, see release/cells.py): a request
names some fields; every other field with a total is at its total, and the fields without one
get a stated default. So a request always points at MoSPI's own published rows. Nothing here
computes a rate; when a combination isn't published, the answer says what is.
"""

import difflib
import re
from dataclasses import dataclass, field

import duckdb

from ..datasets import CATALOG
# the columns saying when a row is (year, month or quarter, period), and a table's frequency
from .periods import TIME, of_columns

# every fact below is the datasets' own (datasets/*.toml), a field the same in each that has it
# the value meaning "every group": person, rural + urban, All India, all
TOTAL = CATALOG.totals
# fields with no total, and what a request that doesn't name them gets
DEFAULTS = CATALOG.defaults
MEASURES = CATALOG.measure_names
# the settings (fields with no total) by kind, any number of each: measures (different measures,
# never mixed) and bands (overlapping values); a table may have none (`Table.measures`, `.bands`)
SETTINGS, MEASURE_FIELDS, BAND_FIELDS = CATALOG.settings, CATALOG.measures, CATALOG.bands
# groups that contain other groups of their field: shown, never ranked against them
CONTAINS = CATALOG.contains
# fields whose groups overlap or measure different things: shown side by side, never ranked
SIDE_BY_SIDE = CATALOG.side_by_side


class Refusal(Exception):
    """The request can't be answered as asked, for a reason the agent can act on."""


def _period(value):
    return value.strftime("%Y-%m") if hasattr(value, "strftime") else str(value)


def _bare(value) -> str:
    """A value as matched against what was asked: no number in front ("7.graduate" -> "graduate"),
    any case, hyphens as spaces, "&" as "and"."""
    text = re.sub(r"^\s*\d+\s*\.\s*", "", str(value)).lower()
    text = re.sub(r"\s*&\s*", " and ", text)
    return " ".join(re.sub(r"[-‐‑–]", " ", text).split())


SHOWN = 12          # a field with more values than this is shown to analysts as a count (analyst.py)


def value_list(text) -> list[str]:
    """A field's values as the tables list writes them: ", " between them, or "; " when a value
    has a comma of its own (CPI's "Arhar, tur"; split on commas, 359 items counted as 474)."""
    return [v.strip() for v in text.split("; " if "; " in text else ", ")]
CLOSE = 0.9         # a misspelling this close to one value only is read as that value


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
    def __init__(self, con, name, title, catalog=None):
        self.con, self.name, self.title = con, name, title
        # its full name when several release files are attached, "cpi"."inflation_monthly": a
        # query cursor doesn't keep a connection's search path
        self.ref = f'"{catalog}"."{name}"' if catalog else f'"{name}"'
        self.indicator = name.split("_")[0]
        # its indicator's unit: "%" for a rate, "" for an index (its description line names it)
        self.unit = CATALOG.units.get(self.indicator, "%")
        # what a figure at every total is about: PLFS's everyone, CPI's all items
        self.everyone = (CATALOG.dataset_of(self.indicator).everyone if self.indicator in CATALOG.indicators
                         else "everyone")
        columns = [c for c, in con.execute(
            "SELECT column_name FROM duckdb_columns() WHERE table_name = ? AND (? IS NULL OR database_name = ?) "
            "ORDER BY column_index", [name, catalog, catalog]).fetchall()]
        # how often it is published, and how its periods are written (periods.py)
        self.freq = of_columns(columns)
        self.fields = [c for c in columns if c not in (*TIME, "value")]
        # the settings this table has, by kind: PLFS's a measure and an age band, CPI's none
        self.measures = [f for f in self.fields if f in MEASURE_FIELDS]
        self.bands = [f for f in self.fields if f in BAND_FIELDS]
        self.values = {f: [v for v, in con.execute(f'SELECT DISTINCT "{f}" FROM {self.ref} ORDER BY 1').fetchall()]
                       for f in self.fields}
        time = self.freq.sql()
        self._time = time
        self.periods = [p for p, in con.execute(f'SELECT DISTINCT {time} FROM {self.ref} ORDER BY 1').fetchall()]
        # which fields MoSPI publishes split together, on which values of the measures and bands:
        # {(fields split, ((setting, value), ...))}
        split = [f for f in self.fields if f in TOTAL]
        kinds = [*self.measures, *self.bands]
        flags = ", ".join([*(f"\"{f}\" <> '{TOTAL[f]}'" for f in split), *(f'"{f}"' for f in kinds)]) or "true"
        self.shapes = set()
        for row in con.execute(f'SELECT DISTINCT {flags} FROM {self.ref}').fetchall():
            on, at = row[:len(split)], row[len(split):]
            self.shapes.add((frozenset(f for f, x in zip(split, on) if x), tuple(zip(kinds, at))))
        # a hierarchy's levels (CPI: division > group > class > sub_class > item): each field
        # `within` another, and every value's own parent, from the rows
        self.within = {f: CATALOG.dimensions[f]["within"] for f in self.fields
                       if CATALOG.dimensions.get(f, {}).get("within") in self.fields}
        self.parent_of = {f: dict(con.execute(f'SELECT DISTINCT "{f}", "{p}" FROM {self.ref} WHERE "{f}" <> ?',
                                              [TOTAL[f]]).fetchall())
                          for f, p in self.within.items()}
        self.same = []          # tables of the same figure on another base (Release fills it)
        # fields whose far-off groups may be small samples, in this table's own dataset's word
        self.small_groups = (CATALOG.dataset_of(self.indicator).small_groups if self.indicator in CATALOG.indicators
                             else CATALOG.small_groups)

    @property
    def figure(self) -> str:
        """What the table's figure is, whatever its base: CPI's inflation2012 is inflation."""
        return (CATALOG.indicators.get(self.indicator) or {}).get("same_as", self.indicator)

    def ancestors(self, f) -> list:
        """The levels above a hierarchy's field, nearest first (item: sub_class, class, ...)."""
        out = []
        while f in self.within:
            f = self.within[f]
            out.append(f)
        return out

    # ---- what a request means ----

    def settings(self, where=None, vary=()) -> Settings:
        where = {k: str(v) for k, v in (where or {}).items()}
        # an annual table's period is its year, asked for either way
        period = where.pop("period", None) or (None if self.freq.within_year else where.pop("year", None))
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
        # a level named with its parents left out has its own parents: "rice" is in its sub_class,
        # class, group and division (a row for rice says them all)
        for f in [f for f in self.within if f in where and values.get(f) != TOTAL.get(f)]:
            child, value = f, values[f]
            for p in self.ancestors(f):
                if p in where or p in vary:
                    break
                value = self.parent_of[child].get(value, values.get(p))
                values[p], child = value, p
        if period is not None:
            period = self.freq.normal(period)
            if period not in self.periods:
                raise Refusal(self.no_period(period))
        return Settings(values, period or self.periods[-1], set(where))

    def stored(self) -> str:
        """The stored periods as runs: "2014-01 to 2024-12, 2026-01 to 2026-10" (span test: CPI's
        142 months listed one by one were ~700 tokens of a refusal)."""
        step = self.freq.step
        runs = []
        for p in self.periods:
            if runs and step(p) == step(runs[-1][1]) + 1:
                runs[-1][1] = p
            else:
                runs.append([p, p])
        return ", ".join(a if a == b else f"{a}, {b}" if step(b) - step(a) == 1 else f"{a} to {b}" for a, b in runs)

    def no_period(self, period) -> str:
        """A period this table doesn't hold, what it does, and a table with the same figure that
        holds it (span test: told only that CPI's 2024 base has no December 2025, an analyst
        answered that the figure isn't published; the 2012 base's table has it)."""
        there = [u.name for u in self.same if period in u.periods]
        return (f"{self.name} has no period {period}. Stored: {self.stored()}"
                + (f"; {', '.join(there)} (the same figure on another base) has {period}" if there else ""))

    def _check(self, f, value):
        """A value as published, or the one it plainly means: MoSPI numbers some groups
        ("7.graduate"), and an analyst asking for "graduate" means that one, not "8.post graduate
        & above" -- a live run lost two analysts to that refusal, each repeating it. Also a value
        in other words the dataset's description gives (a religion's people for the religion, women
        for female), "and" for "&", and a misspelling close to one value only. Analysts are shown
        a long list of values as a count only, so the match is made here."""
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
        exact = [v for v in self.values[f] if want in {_bare(v), *map(_bare, CATALOG.synonyms(f, v))}]
        close = exact or [v for v in self.values[f] if want and want in _bare(v)]
        if len(close) == 1:
            return close[0]
        if close:
            raise Refusal(f"{self.name}: {f} '{value}' could be any of {', '.join(map(str, close))}: "
                          "give one of them exactly as written here")
        bare = {_bare(v): v for v in self.values[f]}
        spelled = difflib.get_close_matches(want, bare, n=2, cutoff=CLOSE)
        if len(spelled) == 1:
            return bare[spelled[0]]
        # the closest first, then the rest (a long list cut, with the closest kept)
        nearest = [bare[b] for b in difflib.get_close_matches(want, bare, n=5, cutoff=0.6)]
        rest = [v for v in self.values[f] if v not in nearest]
        listed = nearest + rest if len(self.values[f]) <= 40 else nearest + rest[:40 - len(nearest)] + ["..."]
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
        # the settings first, always said: a measure by its name, a band as "age 15-29 years"
        for f in [*self.measures, *self.bands]:
            if f not in vary and f in s.values:
                v = s.values[f]
                parts.append(MEASURES.get(v, v) if f in self.measures else f"{f} {v}")
        chosen = [f"{f} = {v}" for f, v in s.values.items()
                  if f not in (*self.measures, *self.bands) and v != TOTAL.get(f)]
        parts.append(", ".join(chosen) if chosen else f"{self.everyone}, {TOTAL[CATALOG.across]}")
        if vary:
            parts.append(f"by {' x '.join(vary)}")
        return " | ".join(parts)

    # ---- reading published rows ----

    def rows(self, s: Settings, vary=(), all_periods=False):
        """Published rows for these settings, each field in `vary` free. [{field..., value}]
        A hierarchy's level varied with its parents left at their totals frees them too, each
        group in its own division: its rows and the one overall total, not each parent's own."""
        where, params, free = [], [], set()
        for f in vary:
            for a in self.ancestors(f):
                if s.values.get(a) == TOTAL.get(a) and a not in s.named:
                    free.add(a)
                    where.append(f'("{f}" <> ? OR "{a}" = ?)')
                    params += [TOTAL[f], TOTAL[a]]
        for f, v in s.values.items():
            if f not in vary and f not in free:
                where.append(f'"{f}" = ?')
                params.append(v)
        if not all_periods:
            where.append(f"{self._time} = ?")
            params.append(s.period)
        cols = ", ".join([*(f'"{f}"' for f in vary), f"{self._time} AS period", "value"])
        sql = f'SELECT {cols} FROM {self.ref} WHERE {" AND ".join(where) or "true"} ORDER BY period'
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
        shapes = [(sp, dict(at)) for sp, at in self.shapes]
        fields = {f for sp, _ in shapes for f in sp}

        def only(chosen) -> list:
            """The measures the shapes chosen are limited to: ["measure PS+SS"], or []."""
            out = []
            for m in self.measures:
                on = {at[m] for _, at in chosen}
                if on and on != set(self.values[m]):
                    out.append(f"{m} {', '.join(sorted(on))}")
            return out

        out = []
        # a band's values other than its default, with the fields each is split by, said for
        # values alike at once ("age groups 15-29 years, 15-59 years: only by ...")
        for b in self.bands:
            alike = {}
            for v in sorted({at[b] for _, at in shapes} - {DEFAULTS.get(b)}):
                chosen = [(sp, at) for sp, at in shapes if at[b] == v]
                partners = frozenset(f for sp, _ in chosen for f in sp)
                measures = tuple(frozenset(at[m] for _, at in chosen) for m in self.measures)
                alike.setdefault((partners, measures), []).append((v, chosen))
            for (partners, _), group in alike.items():
                limits = only(group[0][1])
                if partners != fields or limits:
                    nothing = f"nothing else ({TOTAL[CATALOG.across]} totals)"
                    out.append(f"{b} group{'s' if len(group) > 1 else ''} {', '.join(v for v, _ in group)}: only by "
                               f"{', '.join(sorted(partners)) or nothing}"
                               + (f" ({', '.join(limits)} only)" if limits else ""))
        # a hierarchy's levels go together by what they are (each inside the one above): said once,
        # with what else they go with, not level by level (CPI: six lines naming each other)
        levels = [f for f in self.fields if f in self.within or f in self.within.values()]
        for f in sorted(fields):
            if f in levels:
                continue
            chosen = [(sp, at) for sp, at in shapes if f in sp]
            partners = {g for sp, _ in chosen for g in sp} - {f}
            # a band limits a field only when the field is split at its default alone
            limits = [f"{b}s {DEFAULTS[b]}" for b in self.bands if {at[b] for _, at in chosen} == {DEFAULTS.get(b)}]
            limits += only(chosen)
            if partners != fields - {f} or limits:
                out.append(f"{f}: only by {', '.join(sorted(partners)) or 'itself'}"
                           + (f" ({', '.join(limits)} only)" if limits else ""))
        if levels:
            partners = {g for sp, _ in shapes if set(sp) & set(levels) for g in sp} - set(levels)
            if partners != fields - set(levels):
                out.append(f"{' > '.join(levels)} (each inside the one before): only by "
                           f"{', '.join(sorted(partners)) or 'themselves'}")
        return "; ".join(out)

    def missing(self, s: Settings, vary=()):
        """Why nothing is published for these settings, and what is."""
        split = frozenset(f for f in self.fields if f in TOTAL and (f in vary or s.values.get(f) != TOTAL[f]))
        same = [dict(at) for sp, at in self.shapes if sp == split]
        if not same:
            widest = sorted({sp for sp, _ in self.shapes if sp}, key=lambda sp: (-len(sp), sorted(sp)))
            maximal = [sp for sp in widest if not any(sp < other for other in widest)]
            return (f"{self.name}: {' x '.join(sorted(split))} is not published together. Published "
                    "together: " + "; ".join(" x ".join(sorted(sp)) for sp in maximal))
        # the settings in turn, measures first: the first whose value isn't published with this
        # split, among the shapes with the values before it
        for f in [*self.measures, *self.bands]:
            offered = sorted({at[f] for at in same})
            if s.values.get(f) not in offered:
                if f in self.measures:
                    return f"{self.name}: {self.describe(s, vary)} is published only for {f} {', '.join(offered)}"
                return f"{self.name}: this split is published only for {f} {', '.join(offered)}"
            same = [at for at in same if at[f] == s.values.get(f)]
        return f"{self.name}: nothing published for {self.describe(s, vary)}"


class Release:
    """The release, opened read-only: one dataset's file, or a folder (or list) of them, one file
    per dataset (plfs.duckdb, cpi.duckdb), each built and updated by its own job, since MoSPI
    releases them on different days. Several are attached to one connection and read as one:
    their tables, cards and profiles together; what each release covers, joined."""

    def __init__(self, path):
        files = _release_files(path)
        if len(files) == 1:
            self.con = duckdb.connect(str(files[0]), read_only=True)
            catalogs = [None]
        else:
            self.con = duckdb.connect()
            catalogs = []
            for f in files:
                quoted = str(f).replace("'", "''")          # ATTACH takes no parameter for its path
                self.con.execute(f"ATTACH '{quoted}' AS \"{f.stem}\" (READ_ONLY)")
                catalogs.append(f.stem)
        self.cards, self.profiles, released, home = {}, {}, [], {}
        for c in catalogs:
            at = f'"{c}".' if c else ""
            cards = self.con.execute(f"SELECT name, card FROM {at}_cards ORDER BY name").fetchall()
            self.cards.update(cards)
            home.update({name: c for name, _ in cards})           # which file each table is in
            self.profiles.update(self.con.execute(f"SELECT name, profile FROM {at}_profiles ORDER BY name").fetchall())
            released.append(self.con.execute(f"SELECT released FROM {at}_release").fetchone()[0])
        self.released = "; ".join(released)
        self.tables = {name: Table(self.con, name, card.split("\n", 1)[0].split(": ", 1)[1], home[name])
                       for name, card in self.cards.items()}
        for t in self.tables.values():
            t.same = [u for u in self.tables.values() if u is not t and u.figure == t.figure and u.freq == t.freq]

    def table(self, name) -> Table:
        if name not in self.tables:
            raise Refusal(f"no table '{name}'. Tables: {', '.join(self.tables)}")
        return self.tables[name]

    def close(self):
        self.con.close()


def _release_files(path) -> list:
    """The release files a path means: itself, a list of files, or every .duckdb file in a folder
    (PLFS's first, so its tables come first, as when it was the only one)."""
    from pathlib import Path
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    path = Path(path)
    if path.is_dir():
        found = sorted(path.glob("*.duckdb"), key=lambda p: (p.stem != "plfs", p.stem))
        if not found:
            raise FileNotFoundError(f"no release files in {path}")
        return found
    return [path]
