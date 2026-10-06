"""Offline tests for the questionnaire, both builds.   python test_questionnaire.py   (costs nothing)"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")

import crew_questionnaire as CQ  # noqa: E402
import q_checks as K  # noqa: E402
import q_stubs as S  # noqa: E402
import questionnaire_scratch as QS  # noqa: E402
import tallyfold as T  # noqa: E402


def test_the_key_is_consistent():
    # every statement the key rests on exists, and the gold answers pass the guardrail and the key
    assert all(sid in T.STATEMENTS for key in T.KEY.values() for sid in key["rests_on"])
    assert not T.check(S.questionnaire())
    assert all(row["right"] and not row["missing evidence"] for row in T.grade(S.questionnaire()))


def test_check_catches_what_it_should():
    missing = S.questionnaire(drop=("Q9",))
    assert any("Q9" in p for p in T.check(missing))
    invented = S.questionnaire().model_copy(deep=True)      # GOLD's answers are shared; never edit them
    invented.answers[0].evidence = ["C9"]
    assert any("C9" in p for p in T.check(invented))


def test_both_builds_offline():
    out = CQ.run(CQ.sequential_crew(llm=S.ScriptedLLM()))
    assert out["questionnaire"] is not None and not T.check(out["questionnaire"])
    assert [t["tools used"] for t in out["tasks"][:3]] == [1, 1, 1]
    scratch = QS.Desk(answer=S.answer).run()
    assert scratch["questionnaire"] is not None and not T.check(scratch["questionnaire"])


def test_guardrails():
    assert K.guardrail_retry() == {"lead attempts": 2, "finished questionnaire passes the check": True}
    calls = []

    def drops_once(agent, request, model):
        if model is T.Questionnaire:
            calls.append(request)
            return S.questionnaire(drop=("Q9",) if len(calls) == 1 else ())
        return S.answer(agent, request, model)

    desk = QS.Desk(answer=drops_once)
    assert desk.run()["questionnaire"] is not None and len(calls) == 2
    assert "Q9 is answered 0 times" in calls[1]                     # the feedback reached the second attempt


def test_a_guardrail_that_never_passes_is_reported():
    # CrewAI raises when the guardrail still fails after its retries; `run` reports it instead
    class AlwaysDrops(S.ScriptedLLM):
        def call(self, messages, *args, **kwargs):
            self.drop_first, self.calls = True, []
            return super().call(messages, *args, **kwargs)
    out = CQ.run(CQ.sequential_crew(llm=AlwaysDrops()))
    assert out["questionnaire"] is None and "guardrail" in out["error"]


def test_a_failed_parallel_task_does_not_hang_the_caller():
    class Fails(S.ScriptedLLM):
        def call(self, messages, *args, **kwargs):
            if "You are Security engineer" in "\n".join(str(m.get("content", "")) for m in messages):
                raise RuntimeError("provider refused the call")
            return super().call(messages, *args, **kwargs)
    out = CQ.run(CQ.sequential_crew(llm=Fails()), timeout=10)
    assert out["questionnaire"] is None and "did not finish" in out["error"]


def test_a_failed_specialist_is_reported_by_the_scratch_desk():
    seen = K.failed_specialist(timeout=10)
    assert "did not finish" in seen["CrewAI"]["ended with"] and seen["CrewAI"]["seconds"] >= 10
    scratch = seen["scratch"]
    assert scratch["questionnaire"] is None and list(scratch["failed"]) == ["security"]
    assert "security raised RuntimeError" in scratch["failed"]["security"] and not scratch["the lead was asked"]
    assert scratch["seconds"] < 5


def test_a_refused_native_tool_call_is_retried():
    import litellm

    class RefusedOnce(CQ.GroqLLM):
        def _call_once(self, messages, *args, **kwargs):
            if not any("was refused" in str(m.get("content", "")) for m in messages):
                raise litellm.BadRequestError(
                    message='GroqException - {"error":{"message":"Tool choice is none, but model called a tool",'
                            '"code":"tool_use_failed","failed_generation":"{\\"name\\": \\"repo_browser.print_tree\\"}"}}',
                    model="groq/openai/gpt-oss-120b", llm_provider="groq")
            return "Thought: I now know the final answer\nFinal Answer: done"

    CQ.GroqLLM.refusals.clear()
    assert RefusedOnce(model=CQ.MODEL, api_key="stub").call([{"role": "user", "content": "go"}]).endswith("done")
    assert len(CQ.GroqLLM.refusals) == 1


def test_a_rate_limit_is_waited_out():
    import litellm
    assert CQ._retry_after("Please try again in 13.8s.") == 14.8
    assert CQ._retry_after("Please try again in 1m2.5s.") == 63.5

    class LimitedOnce(CQ.GroqLLM):
        def _call_once(self, messages, *args, **kwargs):
            if not CQ.GroqLLM.waits:
                raise litellm.RateLimitError(message="rate_limit_exceeded ... Please try again in 0.01s.",
                                             model="groq/qwen/qwen3.8-27b", llm_provider="groq")
            return "Final Answer: done"

    CQ.GroqLLM.waits.clear()
    assert LimitedOnce(model=CQ.MODEL, api_key="stub").call("go").endswith("done")
    assert len(CQ.GroqLLM.waits) == 1


def test_hierarchical_wires_up():
    assert CQ.run(CQ.hierarchical_crew(llm=S.ScriptedLLM()))["questionnaire"] is not None


if __name__ == "__main__":
    for test in (test_the_key_is_consistent, test_check_catches_what_it_should, test_both_builds_offline,
                 test_guardrails, test_a_guardrail_that_never_passes_is_reported,
                 test_a_failed_parallel_task_does_not_hang_the_caller,
                 test_a_failed_specialist_is_reported_by_the_scratch_desk, test_a_refused_native_tool_call_is_retried,
                 test_a_rate_limit_is_waited_out, test_hierarchical_wires_up):
        test()
        print("ok ", test.__name__)
