"""The eval: its answer keys are right (checked against the real release file), kept out of
every prompt, and the scoring reads a run the way it should. No tokens."""

import json

import pytest

from evals import score as S
from stats_desk.config import RELEASE_PATH

DATA, BY_ID = S.golden()
needs_release = pytest.mark.skipif(not RELEASE_PATH.exists(), reason="no release file built")


@needs_release
def test_every_number_in_the_answer_keys_is_what_the_release_file_says():
    from stats_desk.analysis import analyses as A
    from stats_desk.analysis.data import Refusal, Release
    release = Release(RELEASE_PATH)
    try:
        for q in DATA["questions"]:
            shown = ""
            for p in q["points"]:
                out = getattr(A, p["call"]["tool"])(release, **p["call"]["args"])
                assert S.has(out.split("\n", 1)[1], p["numbers"]), (q["id"], p["id"], out)
                shown += out
            for s in q["steps"]:
                if "numbers" in s:
                    assert S.has(shown, s["numbers"]), (q["id"], s["id"])
        # Q4's combination really is unpublished
        with pytest.raises(Refusal, match="not published together"):
            A.value(release, "ur_annual", {"state": "Kerala", "religion": "Islam", "gender": "female"})
    finally:
        release.close()


def test_no_eval_question_or_answer_is_in_what_the_agents_are_told():
    import json as _json

    from stats_desk.agents import analyst, lead, plan
    from stats_desk.analysis.analyses import ANALYSES, DEFAULTS_RULE, MEANINGS
    # everything an agent reads that isn't the data: prompts, tool descriptions, the plan's field
    # descriptions, the groups' meanings and defaults in the tables list
    told = " ".join([lead.PERSONA, analyst.PERSONA, lead.Investigation.ask_analysts.__doc__,
                     _json.dumps(plan.Plan.model_json_schema()), MEANINGS, DEFAULTS_RULE,
                     *(f.__doc__ for f in ANALYSES)]).lower()
    for q in DATA["questions"]:
        assert q["question"].lower() not in told, q["id"]
    for place in ("delhi", "haryana", "bihar", "nagaland", "kerala", "andaman", "chandigarh", "graduate", "muslim"):
        assert place not in told, place


def run(answer, cannot_tell="", stopped=None, tools=(), results=None, calls=None, findings=None):
    """A saved run, as collect_runs writes it, cut down to what the scoring reads."""
    return {"id": "Q8", "run": 1, "question": BY_ID["Q8"]["question"],
            "brief": {"answer": answer, "ruled_out": "", "cannot_tell": cannot_tell, "stopped": stopped,
                      "all_findings": findings or {}},
            "results": results or {}, "calls": calls or {},
            "trace": {"tools": list(tools), "tokens": {"total": 1000}, "seconds": 9.0, "waited seconds": 4.0,
                      "cost if paid": 0.0002,
                      "model calls": [{"agent": "lead", "in": 400, "out": 100, "error": None},
                                      {"agent": "analyst f1", "in": 400, "out": 100, "error": "413 too large"}]}}


EDUCATION = ("result compare-aaaaaa\nunemployment rate (UR), annual, 2025 | usual status (PS+SS) | by education\n"
             "total (all): 3.1\nhighest first: 7.graduate 11.2, 1.not literate 0.3")


def tool(n, name, result=None, kind="ok", by="model", said=None):
    return {"n": n, "path": "lead > ask_analysts > analyst f1", "tool": name, "args": "{}", "by": by,
            "kind": kind, "result": result, "said": said, "seconds": 0.1}


def test_a_run_is_scored_on_its_answer_its_path_and_its_cost():
    record = run("Graduates' unemployment is 11.2% against 3.1% for everyone (usual status, 2025); 9.9 for "
                 "some reason.", cannot_tell="Why graduates are more often unemployed.",
                 tools=[tool(1, "value", kind="refused", said="no education 'graduate'"),
                        tool(2, "compare", "compare-aaaaaa"), tool(3, "compare", "compare-aaaaaa"),
                        tool(4, "recheck", "recheck-bbbbbb", by="code")],
                 results={"compare-aaaaaa": EDUCATION, "recheck-bbbbbb": "result recheck-bbbbbb\nin 2024:\n..."},
                 calls={"compare-aaaaaa": ["compare", {"table": "ur_annual", "by": "education"}]},
                 findings={"f1": {"task": "t", "results": ["compare-aaaaaa"], "checked_with": []}})
    [row] = S.score([record])
    o, p, c = row["outcome"], row["path"], row["cost"]
    assert o["brief"] and o["points"] == ["P1"] and o["measure named"]
    assert o["checks"] == {"no cause read in": True, "no bare yes or no to a cause": True,
                           "says the figures can't show cause": True}
    assert o["numbers not in any result"] == [9.9]                     # invented: caught
    assert p["steps"] == {"S1": 2} and p["recall"] == 1.0
    assert p["analyses chosen"] == 3 and p["precision"] == 0.33 and p["refused"] == 1
    assert [x["wasted"] for x in p["calls"]] == [True, False, True, False]     # refused; covered; a repeat
    assert c["by agent"] == {"lead": 500, "analyst": 500} and len(c["groq errors"]) == 1


def test_cause_words_and_no_brief_are_caught():
    [row] = S.score([run("A degree causes unemployment, because graduates wait for good jobs.")])
    assert row["outcome"]["cause words"] == ["because", "causes"]
    assert S.score([run("Yes. Graduates: 11.2%.")])[0]["outcome"]["checks"]["no bare yes or no to a cause"] is False
    assert row["outcome"]["checks"]["no cause read in"] is False
    assert row["outcome"]["checks"]["says the figures can't show cause"] is False
    [none] = S.score([run("", stopped="answered in text instead of submitting")])
    assert none["outcome"]["brief"] is False and none["outcome"]["points"] == []


def test_two_single_figures_from_one_table_are_flagged_as_a_chain():
    calls = {"value-aaaaaa": ["value", {"table": "ur_annual", "where": {"gender": "female"}}],
             "value-bbbbbb": ["value", {"table": "ur_annual", "where": {"gender": "male"}}]}
    [row] = S.score([run("x", calls=calls, findings={"f1": {"task": "t", "results": list(calls), "checked_with": []}})])
    assert row["path"]["single figures chained"] == ["f1"]


def test_the_report_has_the_three_parts_and_every_run():
    record = run("Graduates: 11.2% against 3.1%.", tools=[tool(1, "compare", "compare-aaaaaa")],
                 results={"compare-aaaaaa": EDUCATION}, calls={"compare-aaaaaa": ["compare", {"table": "ur_annual"}]})
    text = S.markdown(S.score([record]), "test.jsonl")
    for part in ("## Outcome", "## Path", "## Cost", "### Q8#1"):
        assert part in text
    assert json.dumps(S.score([record]), default=str)                   # the saved scores are plain JSON
