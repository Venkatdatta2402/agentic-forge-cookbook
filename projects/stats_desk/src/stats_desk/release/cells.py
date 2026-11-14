"""MoSPI's raw rows, and the checks run on them before anything is stored.

MoSPI's rows are published estimates, one per combination of fields (verified 2026-11-09 on
PLFS unemployment: no IDs, nothing computed on request). Reading them after they're stored is
analysis/data.py's job; this module only checks a download: that every total lies between its
parts (`verify_totals`), that the shape is the known one (`guard`), and which age band the
tables without an age cover (`unstated_age_band`).

Field rules, each checked against the numbers rather than assumed:

  field        total             verified how
  gender       person            between male and female in 1,830 of 1,830 groups
  sector       rural + urban     between rural and urban in 1,818 of 1,818
  state        All India         between the lowest and highest state in 144 of 144
  religion,    all (or empty)    between their groups in every group; empty == 'all' (18 of 18)
  social,
  education
  age          no total          overlapping bands (15-29 inside 15+ inside All ages); default 15+
  status       no total          CWS and PS+SS are two measures; pinned per profile, always stated
  year, month  no total          pinned to the period profiled

`verify_totals` repeats the between-parts check on every download, so a new release (or another
dataset) can't quietly break a rule.
"""

from collections import defaultdict

RENAME = {"AgeGroup": "age", "weekly_status": "status", "socialGroup": "social",
          "General_Education": "education", "state": "state", "gender": "gender",
          "sector": "sector", "religion": "religion", "month": "month", "year": "year",
          "quarter": "quarter"}

# fields with a total, and the values that mean "every group of this field"
TOTALS = {"gender": {"person"}, "sector": {"rural + urban"}, "state": {"All India"},
          "religion": {"all", None}, "social": {"all", None}, "education": {"all", None}}
DEFAULTS = {"age": "15 years and above"}           # no total: the broad band
NOT_STATED = None                                   # education tables leave age empty


def is_total(field, value):
    return field in TOTALS and value in TOTALS[field]


class Cells:
    """Raw MoSPI rows of one table, indexed by their fields."""

    def __init__(self, rows, pinned=None):
        """`rows`: raw MoSPI rows. `pinned`: values for the fields with no total, e.g.
        {"status": "PS+SS", "year": "PLFS 2025"}; rows with other values are left out."""
        self.pinned = dict(pinned or {})
        self.fields = []
        self.rows = []
        for raw in rows:
            row = {RENAME[k]: v for k, v in raw.items() if k in RENAME}
            if any(row.get(f) != v for f, v in self.pinned.items()):
                continue
            try:
                row["value"] = float(raw["value"])
            except (TypeError, ValueError, KeyError):
                continue
            self.rows.append(row)
        self.fields = [f for f in RENAME.values()
                       if f not in self.pinned and len({r.get(f) for r in self.rows}) > 1]
        self._index = defaultdict(list)
        for r in self.rows:
            self._index[self._key(r)].append(r["value"])

    def _key(self, row):
        return tuple((f, "TOTAL" if is_total(f, row.get(f)) else row.get(f)) for f in self.fields)

    def levels(self, field):
        """The non-total values of a field, in the order MoSPI lists them."""
        seen = []
        for r in self.rows:
            v = r.get(field)
            if v not in seen and not is_total(field, v) and v is not None:
                seen.append(v)
        return seen

    def get(self, **split):
        """The published value for this combination, or None. Unnamed fields are at their total
        (age: 15+). Copies of one combination in several tables are store.py's to reconcile."""
        want = {f: "TOTAL" if f in TOTALS else DEFAULTS.get(f) for f in self.fields}
        for f, v in split.items():
            want[f] = "TOTAL" if is_total(f, v) else v
        values = self._index.get(tuple((f, want[f]) for f in self.fields))
        return values[0] if values else None


def unstated_age_band(cells_by_indicator: dict, tolerance=0.05):
    """Which age band do the tables that leave age empty cover? Decided from the numbers.

    For each indicator, the empty-age table's own total (every field at its total) is compared
    with the main table's figure for each age band; a band matches when the two are equal. One
    indicator can be ambiguous (UR 2025: 15+ and All ages are both 3.1%), so the answer is the
    band that matches in EVERY indicator -- or None, and the profile keeps saying "no age range".
    Measured 2026-11-09: LFPR 59.3 = 15+ (All ages 44.9, 15-59 63.3); WPR 57.4 = 15+.
    """
    candidates = None
    evidence = {}
    for name, cells in cells_by_indicator.items():
        if "age" not in cells.fields:
            continue
        bands = {b: cells.get(age=b) for b in cells.levels("age")}
        unstated = None
        for field in ("education", "religion", "social"):
            if field in cells.fields:
                key = tuple((f, "TOTAL" if f in TOTALS else (NOT_STATED if f == "age" else None))
                            for f in cells.fields)
                values = cells._index.get(key)
                if values:
                    unstated = values[0]
                    break
        if unstated is None:
            continue
        matching = {b for b, v in bands.items() if v is not None and abs(v - unstated) <= tolerance}
        evidence[name] = {"unstated_total": unstated, "bands": bands, "matching": sorted(matching)}
        candidates = matching if candidates is None else candidates & matching
    band = next(iter(candidates)) if candidates and len(candidates) == 1 else None
    return band, evidence


class ProfileShapeError(Exception):
    """The data no longer has the shape the rules were written for. The build stops instead of
    publishing quietly wrong tables and profiles; the previous release stays in use."""


KNOWN_RAW = set(RENAME) | {"value", "unit", "indicator", "frequency", "year_type"}


def guard(raw_rows, cells: Cells):
    """Stop the build if the download's shape isn't the known one; otherwise return the totals
    check, for the table's card."""
    problems = []
    unknown = sorted({k for r in raw_rows for k in r} - KNOWN_RAW)
    if unknown:
        problems.append(f"fields the profile rules don't know: {', '.join(unknown)}")
    checks = verify_totals(cells)
    for field, check in checks.items():
        if not check["holds"]:
            problems.append(f"the total of {field} is outside its parts in "
                            f"{check['groups'] - check['inside']} of {check['groups']} groups")
    for field in TOTALS:
        if field in cells.fields and not any(is_total(field, r.get(field)) for r in cells.rows):
            problems.append(f"no total row for {field} (expected one of {sorted(map(str, TOTALS[field]))})")
    if problems:
        raise ProfileShapeError("; ".join(problems))
    return checks


def verify_totals(cells: Cells, tolerance=0.05):
    """For each field with a total: in how many groups does the total lie between its parts?"""
    results = {}
    for field in TOTALS:
        if field not in cells.fields:
            continue
        others = [f for f in cells.fields if f != field]
        groups = defaultdict(lambda: {"total": [], "parts": []})
        for r in cells.rows:
            key = tuple(("T" if is_total(f, r.get(f)) else r.get(f)) for f in others)
            groups[key]["total" if is_total(field, r.get(field)) else "parts"].append(r["value"])
        checked = inside = 0
        for g in groups.values():
            if g["total"] and len(g["parts"]) >= 2:
                checked += 1
                lo, hi = min(g["parts"]), max(g["parts"])
                inside += all(lo - tolerance <= t <= hi + tolerance for t in g["total"])
        results[field] = {"groups": checked, "inside": inside,
                          "holds": checked > 0 and inside == checked}
    return results
