"""The datasets, each described as data in its own file (plfs.toml): the one place code and the
agents' instructions learn what a dataset holds. A semantic layer in the manner of SDMX's data
structure definitions: dimensions with their code lists, totals, defaults and names; the
indicators, each with its unit; how the source's rows map onto ours; and the examples the
instructions show. What is published together is not described: it is worked out from the
stored rows.

Several datasets share dimensions (state, sector) under one name, with the same values and the
same total: conformed dimensions, written so at the release (a source's own spelling is mapped
in its file). So the analyses read every field's facts from one place (`CATALOG`), whichever
dataset a table comes from; loading refuses two datasets that describe one field differently.
`DATASET` is PLFS, for what is still written for it alone (its release job; the instructions).
"""

import json
import re
import tomllib
from pathlib import Path
from string import Template

HERE = Path(__file__).parent


class Dataset:
    def __init__(self, data: dict):
        self.data = data
        self.key, self.name, self.publisher = data["key"], data["name"], data["publisher"]
        self.subject, self.holds = data["subject"], data["holds"]
        self.value_note, self.cannot, self.groups_mean = data["value_note"], data["cannot"], data["groups_mean"]
        self.describes, self.describes_in = data["describes"], data["describes_in"]
        # what a figure at every total is about: PLFS's everyone, CPI's all items
        self.everyone = data.get("everyone", "everyone")
        self.source = data["source"]
        self.tables = data["tables"]
        self.indicators = data["indicators"]                  # key -> {code, name, unit, words}
        self.dimensions = data["dimensions"]                  # name -> its description, in file order
        self.examples = data.get("examples", {})
        self.contrasts = [(c["field"], (c["a"], c["b"], c["label"])) for c in data.get("contrasts", [])]
        d = self.dimensions
        # the value meaning "every group" of each dimension that has one: person, rural + urban, ...
        self.totals = {f: x["total"] for f, x in d.items() if "total" in x}
        # the values a source row may give for a total (an empty religion is "all")
        self.total_values = {f: {x["total"], None} if x.get("empty_is_total") else {x["total"]}
                             for f, x in d.items() if "total" in x}
        # dimensions with no total, which a request always sets or gets the default of
        self.settings = tuple(f for f, x in d.items() if "total" not in x)
        self.defaults = {f: x["default"] for f, x in d.items() if "default" in x}
        # the settings by kind (their `role`), any number of each, or none (CPI): a measure's
        # values are different measures (a figure on one is never set against one on another; a
        # finding is rechecked on the others); a band's values overlap (15-29 inside 15+)
        self.measures = tuple(f for f in self.settings if d[f].get("role") == "measure")
        self.bands = tuple(f for f in self.settings if d[f].get("role") == "band")
        self.measure_names = {k: v for f in self.measures for k, v in d[f].get("names", {}).items()}
        self.contains = {f: x["contains"] for f, x in d.items() if "contains" in x}
        self.side_by_side = {f: x["side_by_side"] for f, x in d.items() if "side_by_side" in x}
        self.no_total = {f: x["no_total"] for f, x in d.items() if "no_total" in x}
        self.small_groups = {f for f, x in d.items() if x.get("small_groups")}
        self.across = next(f for f, x in d.items() if x.get("across"))
        self.across_plural = d[self.across]["plural"]
        self.identities = {frozenset(pair): i["because"] for i in data.get("identities", []) for pair in i["pairs"]}
        self.words = {k: i["words"] for k, i in self.indicators.items()}
        self.units = {k: i["unit"] for k, i in self.indicators.items()}
        self.short = {k: k.upper() for k in self.indicators}
        self.by_code = {i["code"]: k for k, i in self.indicators.items()}

    @property
    def unit(self) -> str:
        """The unit all its indicators share (PLFS: %); a dataset whose indicators differ has none."""
        units = set(self.units.values())
        if len(units) != 1:
            raise ValueError(f"{self.name}'s indicators have different units: {sorted(units)}")
        return next(iter(units))

    def synonyms(self, field, value) -> list[str]:
        """Other ways a value is written ("15+" for 15 years and above)."""
        return self.dimensions.get(field, {}).get("synonyms", {}).get(value, [])

    def fill(self, text: str, **more) -> str:
        """A template's $names filled in from this description: tool descriptions are written once
        for every dataset, with this one's words and examples."""
        return Template(text).substitute(self.words_for_text(), **more)

    def words_for_text(self) -> dict:
        return {"publisher": self.publisher, "name": self.name, "across": self.across,
                "across_plural": self.across_plural, "describes": self.describes,
                "describes_in": self.describes_in,
                **{f"ex_{k}": v if isinstance(v, str) else json.dumps(v) for k, v in self.examples.items()}}


# what a dimension shared by two datasets must say the same in both
SHARED = ("total", "empty_is_total", "role", "default", "plural", "across")


class Catalog:
    """Every dataset's dimensions and indicators as one description, for the analyses: a field
    means the same in every dataset that has it (conformed dimensions), an indicator key is one
    dataset's. Built from the datasets; refuses two that describe one field differently."""

    def __init__(self, datasets):
        self.datasets = list(datasets)
        dims, owner = {}, {}
        for ds in self.datasets:
            for f, x in ds.dimensions.items():
                if f in dims:
                    differ = [k for k in SHARED if dims[f].get(k) != x.get(k)]
                    if differ:
                        raise ValueError(f"field '{f}' is described differently by {owner[f]} and {ds.key}: "
                                         f"{', '.join(differ)}")
                    dims[f] = {**dims[f], **{k: v for k, v in x.items() if k not in SHARED},
                               "synonyms": {**dims[f].get("synonyms", {}), **x.get("synonyms", {})}}
                else:
                    dims[f], owner[f] = dict(x), ds.key
            clash = set(ds.indicators) & {k for other in self.datasets if other is not ds for k in other.indicators}
            if clash:
                raise ValueError(f"indicator keys used by two datasets: {sorted(clash)}")
        self.dimensions = dims
        merged = lambda attr: {k: v for ds in self.datasets for k, v in getattr(ds, attr).items()}
        self.totals, self.total_values = merged("totals"), merged("total_values")
        self.defaults, self.measure_names = merged("defaults"), merged("measure_names")
        self.contains, self.side_by_side = merged("contains"), merged("side_by_side")
        self.no_total, self.identities = merged("no_total"), merged("identities")
        self.words, self.units, self.short = merged("words"), merged("units"), merged("short")
        self.indicators = merged("indicators")
        self.settings = tuple(dict.fromkeys(f for ds in self.datasets for f in ds.settings))
        self.small_groups = {f for ds in self.datasets for f in ds.small_groups}
        # a field shared by two datasets has one role in both (SHARED), so each is one kind
        self.measures = tuple(dict.fromkeys(f for ds in self.datasets for f in ds.measures))
        self.bands = tuple(dict.fromkeys(f for ds in self.datasets for f in ds.bands))
        self.across = self.datasets[0].across
        self.across_plural = self.datasets[0].across_plural
        if any(ds.across != self.across for ds in self.datasets):
            raise ValueError("datasets correlate across different fields")

    def dataset_of(self, indicator: str) -> Dataset:
        return next(ds for ds in self.datasets if indicator in ds.indicators)

    def synonyms(self, field, value) -> list[str]:
        return self.dimensions.get(field, {}).get("synonyms", {}).get(value, [])


class Words:
    """What the instructions say of the datasets a desk answers from, in one place: of one dataset,
    its own words (PLFS alone reads as it always did); of several, each named."""

    def __init__(self, datasets):
        self.datasets = list(datasets)
        one = self.one = self.datasets[0] if len(self.datasets) == 1 else None
        self.publisher = self.datasets[0].publisher
        self.examples = self.datasets[0].examples          # the first's (PLFS's): they show how to ask
        self.across, self.across_plural = self.datasets[0].across, self.datasets[0].across_plural
        # what code reruns a finding on, and what a figure is quoted with: the other measure only
        # where a dataset has one (its settings' kinds); a dataset with none never reads of one
        measured = any(d.measures for d in self.datasets)
        self.rerun = "in the other period" + (" and on the other measure" if measured else "")
        self.quoted_with = "group, year and measure" if measured else "group and year"
        if one:
            self.about = f"{one.subject}: {one.publisher}'s {one.name}, with {one.holds}"
            self.analyst_about = f"{one.subject} ({one.publisher}'s {one.name})"
            self.cannot, self.unit = one.cannot, one.unit
            self.describes, self.describes_in, self.groups_mean = one.describes, one.describes_in, one.groups_mean
            return
        names = " and ".join(d.name for d in self.datasets)
        self.about = (f"India's official statistics, {self.publisher}'s: "
                      + "; ".join(f"{d.name} ({d.subject}), with {d.holds}" for d in self.datasets))
        self.analyst_about = f"India's official statistics ({self.publisher}'s {names})"
        self.cannot = ", ".join(f"{d.cannot} ({d.name})" for d in self.datasets)
        units = {u for d in self.datasets for u in d.units.values()}
        self.unit = "%" if units == {"%"} else "%, none for an index,"
        self.describes = ", or ".join(dict.fromkeys(d.describes for d in self.datasets))
        self.describes_in = ", or ".join(dict.fromkeys(d.describes_in for d in self.datasets))
        self.groups_mean = "; ".join(dict.fromkeys(d.groups_mean for d in self.datasets))

    def fill(self, text: str, **more) -> str:
        """A template's $names filled in: tool descriptions are written once for every dataset."""
        return Template(text).substitute(
            {"publisher": self.publisher, "across": self.across, "across_plural": self.across_plural,
             "describes": self.describes, "describes_in": self.describes_in,
             **{f"ex_{k}": v if isinstance(v, str) else json.dumps(v) for k, v in self.examples.items()}}, **more)


def load(key: str) -> Dataset:
    return Dataset(tomllib.loads((HERE / f"{key}.toml").read_text(encoding="utf-8")))


DATASETS = {key: load(key) for key in ("plfs", "cpi")}
DATASET = DATASETS["plfs"]
CATALOG = Catalog(DATASETS.values())
# what the instructions say: of every dataset the desk answers from
WORDS = Words(DATASETS.values())


def words_for(tables) -> Words:
    """The words for the datasets these tables (names, or Table objects) come from."""
    keys = {getattr(t, 'indicator', str(t).split('_')[0]) for t in tables}
    found = [d for d in DATASETS.values() if keys & set(d.indicators)]
    return Words(found or [DATASET])


def indicator_pattern(dataset=CATALOG) -> str:
    """The indicators' table-name prefixes as one alternation: "lfpr|wpr|ur"."""
    return "|".join(map(re.escape, dataset.indicators))
