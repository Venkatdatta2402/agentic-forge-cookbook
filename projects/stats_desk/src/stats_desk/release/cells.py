"""A dataset's raw rows, and the checks run on them before anything is stored: the same for every
dataset whose description maps the source's columns onto ours ([source] columns).

MoSPI's rows are published estimates, one per combination of fields (verified 2026-11-09 on
PLFS unemployment: no IDs, nothing computed on request). Reading them after they're stored is
analysis/data.py's job; this module only checks a download: that every total lies between its
parts (`verify_totals`) and that the shape is the known one (`guard`). The field rules are each
dataset's description's; for PLFS each was checked against the numbers rather than assumed:

  field        total             verified how
  gender       person            between male and female in 1,830 of 1,830 groups
  sector       rural + urban     between rural and urban in 1,818 of 1,818
  state        All India         between the lowest and highest state in 144 of 144
  religion,    all (or empty)    between their groups in every group; empty == 'all' (18 of 18)
  social,
  education
  age          no total          overlapping bands (15-29 inside 15+ inside All ages); default 15+
  measure      no total          CWS and PS+SS are two measures; pinned per profile, always stated
  year, month  no total          pinned to the period profiled

(Counted on PLFS's raw download, 2025, usual status. The store now runs the check on every
dataset's clean rows, every period and measure: an education-table row with no age and its copy
with age 15+ are then one group, not two, so the counts on the cards differ from these.)

The store (release/store.py) repeats the between-parts check on every table of every build, so
a new release (or another dataset) can't quietly break a rule.
"""

from collections import defaultdict


class Cells:
    """Raw rows of one table, indexed by their fields (ours, from the dataset's column map)."""

    def __init__(self, rows, dataset, pinned=None):
        """`rows`: raw MoSPI rows. `pinned`: values for the fields with no total, e.g.
        {"measure": "PS+SS", "year": "PLFS 2025"}; rows with other values are left out."""
        self.dataset = dataset
        self.rename = dataset.source["columns"]
        self.totals = dataset.total_values
        # a field without a total: its default, when a lookup doesn't name it (PLFS: age 15+)
        self.defaults = {f: v for f, v in dataset.defaults.items() if f in dataset.settings}
        self.pinned = dict(pinned or {})
        self.rows = []
        for raw in rows:
            row = {self.rename[k]: v for k, v in raw.items() if k in self.rename}
            if any(row.get(f) != v for f, v in self.pinned.items()):
                continue
            try:
                row["value"] = float(raw["value"])
            except (TypeError, ValueError, KeyError):
                continue
            self.rows.append(row)
        self.fields = [f for f in self.rename.values()
                       if f not in self.pinned and len({r.get(f) for r in self.rows}) > 1]
        self._index = defaultdict(list)
        for r in self.rows:
            self._index[self._key(r)].append(r["value"])

    def is_total(self, field, value):
        return field in self.totals and value in self.totals[field]

    def _key(self, row):
        return tuple((f, "TOTAL" if self.is_total(f, row.get(f)) else row.get(f)) for f in self.fields)

    def levels(self, field):
        """The non-total values of a field, in the order MoSPI lists them."""
        seen = []
        for r in self.rows:
            v = r.get(field)
            if v not in seen and not self.is_total(field, v) and v is not None:
                seen.append(v)
        return seen

    def get(self, **split):
        """The published value for this combination, or None. Unnamed fields are at their total
        (or default). Copies of one combination in several tables are the source's to reconcile."""
        want = {f: "TOTAL" if f in self.totals else self.defaults.get(f) for f in self.fields}
        for f, v in split.items():
            want[f] = "TOTAL" if self.is_total(f, v) else v
        values = self._index.get(tuple((f, want[f]) for f in self.fields))
        return values[0] if values else None


class ProfileShapeError(Exception):
    """The data no longer has the shape the rules were written for. The build stops instead of
    publishing quietly wrong tables and profiles; the previous release stays in use."""


class Rows(Cells):
    """A table's clean rows, already in our column names, as a source hands them to the store:
    what every dataset's totals are checked on (`totals_problems`)."""

    def __init__(self, rows, columns, dataset):
        self.dataset, self.rows = dataset, rows
        self.totals = {f: v for f, v in dataset.total_values.items() if f in columns}
        self.fields = [c for c in columns if len({r[c] for r in rows}) > 1]


def guard(raw_rows, cells: Cells):
    """Stop the build if the download has fields the dataset's description doesn't map; the
    totals themselves are checked on the clean rows, for every dataset (`totals_problems`)."""
    source = cells.dataset.source
    known = set(source["columns"]) | set(source.get("other_columns", []))
    unknown = sorted({k for r in raw_rows for k in r} - known)
    if unknown:
        raise ProfileShapeError(f"fields the profile rules don't know: {', '.join(unknown)}")


def totals_problems(cells: Cells, tolerance):
    """(checks, problems): every total between its parts, give or take `tolerance` (what rounding
    can do), and a total row wherever a field with a total is split. Any problem stops the build."""
    checks = verify_totals(cells, tolerance)
    problems = [f"the total of {field} is outside its parts in "
                f"{check['groups'] - check['inside']} of {check['groups']} groups"
                for field, check in checks.items() if not check["holds"]]
    for field in cells.totals:
        if field in cells.fields and not any(cells.is_total(field, r.get(field)) for r in cells.rows):
            problems.append(f"no total row for {field} (expected one of {sorted(map(str, cells.totals[field]))})")
    return checks, problems


def verify_totals(cells: Cells, tolerance=0.05):
    """For each field with a total: in how many groups does the total lie between its parts?"""
    results = {}
    for field in cells.totals:
        if field not in cells.fields:
            continue
        others = [f for f in cells.fields if f != field]
        groups = defaultdict(lambda: {"total": [], "parts": []})
        for r in cells.rows:
            key = tuple(("T" if cells.is_total(f, r.get(f)) else r.get(f)) for f in others)
            groups[key]["total" if cells.is_total(field, r.get(field)) else "parts"].append(r["value"])
        checked = inside = 0
        for g in groups.values():
            if g["total"] and len(g["parts"]) >= 2:
                checked += 1
                lo, hi = min(g["parts"]), max(g["parts"])
                inside += all(lo - tolerance <= t <= hi + tolerance for t in g["total"])
        # no group with the total and two parts: nothing to check (a table with one state), not a failure
        results[field] = {"groups": checked, "inside": inside, "holds": inside == checked}
    return results
