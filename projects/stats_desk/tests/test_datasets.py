"""Datasets as one description (datasets/__init__.py): a field two datasets share means the same in
both, and one each describes differently is refused when loading."""

import copy

import pytest

from stats_desk.datasets import DATASET, Catalog, Dataset


def other(**changes):
    """A second dataset sharing PLFS's state and sector, with fields and an indicator of its own."""
    data = copy.deepcopy(DATASET.data)
    data.update(key="other", name="OTHER")
    data["indicators"] = {"price": {"code": 1, "unit": "", "name": "a price index", "words": "price"}}
    data["identities"] = []
    data["dimensions"] = {"state": dict(DATASET.dimensions["state"]), "sector": dict(DATASET.dimensions["sector"]),
                          "item": {"total": "all", "empty_is_total": True, "synonyms": {"rice": ["paddy"]}}}
    for field, change in changes.items():
        data["dimensions"][field].update(change)
    return Dataset(data)


def test_datasets_sharing_a_field_are_one_description():
    both = Catalog([DATASET, other()])
    assert both.totals["state"] == "All India" and both.totals["item"] == "all"
    assert both.units == {"lfpr": "%", "wpr": "%", "ur": "%", "price": ""}
    assert both.measures == ("measure",) and both.bands == ("age",)       # PLFS's; the other has none
    assert both.dataset_of("price").key == "other" and both.synonyms("item", "rice") == ["paddy"]
    assert both.synonyms("gender", "female") == DATASET.synonyms("gender", "female")
    assert other().measures == () and other().bands == () and other().settings == ()


def test_settings_are_a_list_any_number_of_each_kind():
    """A dataset with two measures and a band of its own sits beside PLFS's measure and age."""
    two = other()
    two.data["dimensions"].update({
        "basis": {"role": "measure", "default": "current", "names": {"current": "at current prices"}},
        "price_kind": {"role": "measure", "default": "retail"},
        "duration": {"role": "band", "default": "1 year"}})
    two = Dataset(two.data)
    assert two.settings == ("basis", "price_kind", "duration")
    assert two.measures == ("basis", "price_kind") and two.bands == ("duration",)
    both = Catalog([DATASET, two])
    assert both.measures == ("measure", "basis", "price_kind") and both.bands == ("age", "duration")
    assert both.measure_names["current"] == "at current prices" and both.measure_names["CWS"]


def test_a_field_with_one_role_in_one_dataset_and_another_in_the_next_is_refused():
    clash = copy.deepcopy(other().data)
    clash["dimensions"]["age"] = {**DATASET.dimensions["age"], "role": "measure"}
    with pytest.raises(ValueError, match="field 'age' is described differently by plfs and other: role"):
        Catalog([DATASET, Dataset(clash)])


def test_a_field_two_datasets_describe_differently_is_refused():
    with pytest.raises(ValueError, match="field 'sector' is described differently by plfs and other: total"):
        Catalog([DATASET, other(sector={"total": "Combined"})])


def test_a_dataset_whose_indicators_differ_in_unit_has_no_one_unit():
    mixed = other()
    mixed.units["index"] = "%"
    with pytest.raises(ValueError, match="different units"):
        mixed.unit
    assert DATASET.unit == "%"
