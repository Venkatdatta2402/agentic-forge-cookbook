"""Offline tests for the analysis team, both builds.   python test_analysis.py   (costs nothing)"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")

import a_checks as K  # noqa: E402
import a_stubs as S  # noqa: E402
import analysis_team as AT  # noqa: E402
import brightwater as B  # noqa: E402
import team_scratch as TS  # noqa: E402


def test_the_traps_are_real():
    for qid, seen in K.traps().items():
        assert seen["cleaned (the truth)"] == (B.truth()[qid]["entity"], B.truth()[qid]["value"])
    assert K.traps()["Q1"]["the export as it stands"][0] != B.truth()["Q1"]["entity"]     # duplicates mislead Q1
    assert K.traps()["Q3"]["the export as it stands"][1] > B.truth()["Q3"]["value"] + 10   # and inflate Q3


def test_autogen_team_offline():
    for qid in B.QUESTIONS:
        out = AT.run_team(B.QUESTIONS[qid], client_factory=lambda q=qid: S.replay(q), timeout=120)
        assert out["error"] is None and out["stop_reason"] == "'reporter' answered", out
        assert out["code_runs"] == 1 and B.grade(qid, out["finding"])["right"]
        ran = next(t["text"] for t in out["turns"] if t["kind"] == "CodeExecutionEvent")
        assert ran.startswith("exit 0"), ran                                     # the code really ran


def test_scratch_team_offline():
    for qid in B.QUESTIONS:
        team = TS.Team()
        team.answer = S.scratch(team, qid)
        out = team.run(B.QUESTIONS[qid])
        assert out["stop_reason"] == "reporter answered", out["stop_reason"]
        assert out["speakers"] == ["planner", "analyst", "reviewer", "reporter"]
        assert [t["field"] for t in out["turns"][1:]] == ["plan", "evidence", "report", "review", "finding"]
        assert out["turns"][2]["source"] == "runner" and "exit 0" in out["turns"][2]["text"]   # the code really ran
        assert B.grade(qid, out["finding"])["right"]


def test_the_case_file():
    team = TS.Team()
    team.answer = S.scratch(team, "Q1", approve=False)
    out = team.run(B.QUESTIONS["Q1"])
    assert out["stop_reason"] == "needs a person: the second review did not approve" and out["rejections"] == 2
    assert "evidence" not in team.board.view("planner") and "report" not in team.board.view("analyst")
    from state import Denied
    try:
        team.board.set("analyst", "evidence", "it printed 37.6")
        assert False, "the analyst wrote the evidence"
    except Denied:
        pass
    printed = team.board.values["evidence"]
    assert TS.grounded(S.finding("Q1"), printed) is None
    made_up = S.finding("Q1").model_copy(update={"value": 41.2})
    assert TS.grounded(made_up, printed).startswith("held for a person")


def test_routing():
    ok, back = B.Review(approved=True, fix_by="nobody"), B.Review(approved=False, problems=["x"], fix_by="planner")
    assert [AT.route(s, c) for s, c in [("user", ""), ("planner", ""), ("analyst", ""), ("reviewer", ok),
                                        ("reviewer", back), ("reviewer", "prose, not a form")]] == \
        ["planner", "analyst", "reviewer", "reporter", "planner", None]


def test_containment():
    loops = K.event_loops()
    assert any(v.startswith("NotImplementedError") for v in loops.values())     # why run_team needs its own loop
    seen = K.secrets()
    assert "GROQ_API_KEY" in seen["AutoGen's executor -- the code can see"]
    assert seen["this project's executor -- the code can see"] == "[]"
    gate = K.gate_examples()
    assert gate["reads the CSV"] == "allowed" and gate["prints an API key"].startswith("refused")
    assert gate["the same key, written differently"] == "allowed"                # a gate is not a sandbox


def test_budgets_stop_before_the_next_call():
    import asyncio

    import llm
    from autogen_core.models import UserMessage
    client = AT.model_client(budget=0, label="autogen test")          # already at its budget
    try:
        asyncio.run(client.create([UserMessage(content="hi", source="user")]))
        assert False, "the AutoGen client made a call past its budget"
    except AT.BudgetExceeded:
        pass
    with TS.Budget(limit=0, label="scratch test"):
        try:
            llm.client.chat.completions.create(model="x", messages=[])
            assert False, "the scratch client made a call past its budget"
        except AT.BudgetExceeded:
            pass
    assert "BUDGET REACHED" in AT.PROGRESS.read_text(encoding="utf-8")


def test_termination():
    stopped = K.termination()
    assert stopped["stop_reason"] == "Functional termination condition met", stopped    # the second rejection


def test_skills_offline():
    out = AT.run_team(B.QUESTIONS["Q1"], client_factory=lambda: S.replay("Q1", skills=True), timeout=120, skills=True)
    assert out["error"] is None and out["skills read"] == ["deduplicate-records"], out["skills read"]
    assert B.grade("Q1", out["finding"])["right"] and K.checked_duplicates(out)
    team = TS.Team(skills=True)
    team.answer = S.scratch(team, "Q1")
    out = team.run(B.QUESTIONS["Q1"])
    assert out["skills read"] == ["deduplicate-records"] and B.grade("Q1", out["finding"])["right"]
    assert K.skills_report("Q1", out)["needed, not read"] == ["net-revenue", "period-growth"]
    assert B.read_skill("no-such-skill").startswith("No procedure named")
    assert "read_skill" in TS.Team(skills=True).agents["planner"].tools.names


if __name__ == "__main__":
    for test in (test_the_traps_are_real, test_autogen_team_offline, test_scratch_team_offline, test_the_case_file,
                 test_routing,
                 test_containment, test_budgets_stop_before_the_next_call, test_termination,
                 test_skills_offline):
        test()
        print("ok ", test.__name__)
