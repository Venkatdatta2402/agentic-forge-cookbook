"""The datasets, each described as data in its own file (plfs.toml): the one place code and the
agents' instructions learn what a dataset holds. A semantic layer in the manner of SDMX's data
structure definitions: dimensions with their code lists, totals, defaults and names; the
indicators; how the source's rows map onto ours; and the examples the instructions show.
What is published together is not described: it is worked out from the stored rows.

Adding a dataset means writing its file. Today there is one, PLFS (`DATASET`).
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
        self.subject, self.holds, self.unit = data["subject"], data["holds"], data["unit"]
        self.value_note, self.cannot, self.groups_mean = data["value_note"], data["cannot"], data["groups_mean"]
        self.describes, self.describes_in = data["describes"], data["describes_in"]
        self.source = data["source"]
        self.tables = data["tables"]
        self.indicators = data["indicators"]                  # key -> {code, name, words}
        self.dimensions = data["dimensions"]                  # name -> its description, in file order
        self.examples = data["examples"]
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
        self.measure = next(f for f, x in d.items() if x.get("role") == "measure")
        self.band = next(f for f, x in d.items() if x.get("role") == "band")
        self.measure_names = d[self.measure].get("names", {})
        self.contains = {f: x["contains"] for f, x in d.items() if "contains" in x}
        self.side_by_side = {f: x["side_by_side"] for f, x in d.items() if "side_by_side" in x}
        self.no_total = {f: x["no_total"] for f, x in d.items() if "no_total" in x}
        self.small_groups = {f for f, x in d.items() if x.get("small_groups")}
        self.across = next(f for f, x in d.items() if x.get("across"))
        self.across_plural = d[self.across]["plural"]
        self.identities = {frozenset(pair): i["because"] for i in data.get("identities", []) for pair in i["pairs"]}
        self.words = {k: i["words"] for k, i in self.indicators.items()}
        self.short = {k: k.upper() for k in self.indicators}
        self.by_code = {i["code"]: k for k, i in self.indicators.items()}

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


def load(key: str) -> Dataset:
    return Dataset(tomllib.loads((HERE / f"{key}.toml").read_text(encoding="utf-8")))


DATASET = load("plfs")


def indicator_pattern(dataset: Dataset = DATASET) -> str:
    """The indicators' table-name prefixes as one alternation: "lfpr|wpr|ur"."""
    return "|".join(map(re.escape, dataset.indicators))
