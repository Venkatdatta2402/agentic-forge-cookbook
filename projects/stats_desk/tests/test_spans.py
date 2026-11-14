"""The span tests' own code, on a scripted model and the test release: each step is run as the
investigation would run it, and checked. No tokens."""

import json

from stubs import call, calls

from evals import spans as SP
from stats_desk.analysis.client import connect
from stats_desk.analysis.server import create

Q8 = "Does getting a degree make people more likely to be unemployed?"


async def test_the_pick_step_is_the_leads_first_call_and_is_checked(release_path, model):
    model.steps += [calls(("profile", {"names": ["UR_annual_2025"]}))]
    async with connect(server=create(release_path)) as analyses:
        out = await SP.pick(analyses, {"id": "Q8", "needed": ["UR_annual_2025"]}, Q8)
    assert out["checks"]["has the needed ones"] and out["checks"]["not all six"] and out["checks"]["extra"] == []
    sent = model.requests[0]["messages"]
    assert "Profiles you can read:" in sent[1]["content"] and sent[2]["content"] == Q8   # as a real run sends it


async def test_the_tasks_step_starts_from_the_profiles_read(release_path, model):
    model.steps += [calls(("ask_analysts", {"tasks": ["Find unemployment of graduates in the UR_annual_2025 profile",
                                                       "How does unemployment differ by education?"]}))]
    async with connect(server=create(release_path)) as analyses:
        out = await SP.tasks(analyses, {"id": "Q8", "needed": ["UR_annual_2025"]}, Q8)
    assert out["tasks"][1] == "How does unemployment differ by education?"
    assert out["checks"]["handed out tasks"] and out["checks"]["none points at a profile"] is False
    assert any(m["role"] == "tool" and "PROFILE unemployment rate" in m["content"]
               for m in model.requests[0]["messages"])


async def test_the_analyst_step_runs_one_analyst_on_one_task(release_path, model):
    model.steps += [call("compare", table="ur_annual", by="education"),
                    lambda r: call("json", claim="Graduates' unemployment is the highest, 11.2%.",
                                   results=[next(m["content"] for m in r["messages"] if m["role"] == "tool")
                                            .split("\n")[0][7:]])]
    async with connect(server=create(release_path)) as analyses:
        out = await SP.analyst(analyses, {"id": "A9", "task": "How does unemployment differ by education?",
                                          "fits": ["compare"]})
    assert out["checks"]["handed in"] and out["checks"]["used an analysis that fits"]
    assert out["used"] == ["compare"] and out["checks"]["no chained single figures"]


async def test_the_brief_step_gets_fixed_findings_with_codes_checks_and_is_scored(release_path, model):
    case = next(c for c in SP.cases()["brief"] if c["id"] == "Q8")
    model.steps += [call("json", answer="Graduates' unemployment is 11.2% in 2025 (usual status), against 3.1% "
                         "for everyone.", findings=["f1"], cannot_tell="Whether a degree causes it.")]
    async with connect(server=create(release_path)) as analyses:
        out = await SP.brief(analyses, case, Q8)
    c = out["checks"]
    assert c["handed in a brief"] and c["first try passed"] and c["points"] == "1/1"
    assert c["no cause read in"] and c["says the figures can't show cause"] and c["measure named"]
    finding = next(m["content"] for m in model.requests[0]["messages"] if m["role"] == "tool" and "-- task:" in m["content"])
    assert "rerun by code" in finding and "claim: Graduates have the highest unemployment" in finding


async def test_a_brief_refused_on_the_first_try_is_handed_back_as_in_a_real_run(release_path, model):
    case = next(c for c in SP.cases()["brief"] if c["id"] == "Q8")
    model.steps += [call("json", answer="Graduates' unemployment is 12.5%.", findings=["f1"], cannot_tell="Why."),
                    call("json", answer="Graduates' unemployment is 11.2% (usual status).", findings=["f1"], cannot_tell="Why.")]
    async with connect(server=create(release_path)) as analyses:
        out = await SP.brief(analyses, case, Q8)
    assert out["checks"]["handed in a brief"] and out["checks"]["first try passed"] is False
    assert out["checks"]["refused"] == 1 and "12.5" in out["refusals"][0]


def test_the_span_cases_name_real_eval_questions_and_their_figures_hold():
    from evals import score as S
    _, golden = S.golden()
    data = SP.cases()
    assert {c["id"] for c in data["pick"]} == set(golden) == {c["id"] for c in data["brief"]}
    assert json.dumps(data)                                          # plain JSON
