"""Offline tests for the claims assistant, both builds.   python test_claims.py   (costs nothing)"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")

import httpx  # noqa: E402
import openai  # noqa: E402
from langchain_core.messages import SystemMessage  # noqa: E402

import claims_chain as C  # noqa: E402
import claims_scratch as X  # noqa: E402
import claims_stubs as S  # noqa: E402
import meridian as M  # noqa: E402


def chain_assistant(model=None, fallback=None):
    return C.Assistant(model=model or S.Scripted(), retriever=C.build_retriever(S.WordEmbeddings()),
                       fallback=fallback)


def scratch_assistant():
    return X.Assistant(index=X.ClauseIndex(embed=S.embed), rewrite=S.rewrite, lookups=S.lookups, respond=S.respond)


def refusal():
    return openai.BadRequestError("tool_use_failed", body={"error": {"code": "tool_use_failed"}},
                                  response=httpx.Response(400, request=httpx.Request("POST", "https://groq")))


class RefusesExtraction(S.Scripted):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(isinstance(m, SystemMessage) and m.content.startswith("From a travel-insurance") for m in messages):
            raise refusal()
        return super()._generate(messages, stop, run_manager, **kwargs)


class Down(S.Scripted):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(isinstance(m, SystemMessage) and m.content.startswith("You are Meridian") for m in messages):
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://groq"))
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_both_builds_answer_the_inbox():
    for assistant in (chain_assistant(), scratch_assistant()):
        turns = [assistant.ask(*q) for q in M.QUESTIONS]
        for turn, want in zip(turns, M.EXPECTED):
            assert turn["reply"].covered == want["covered"], (turn["message"], turn["reply"])
            assert turn["reply"].payout == want["payout"]
        # memory: the follow-up was rewritten using the conversation before it
        assert "police report" in turns[2]["question"] and "CLM-20931" in turns[2]["question"]
        assert "$400.00" in turns[0]["facts"]                                       # the calculator ran
        assert "CLM-20877" in turns[5]["facts"]                                     # the earlier claim is in view


def test_fallbacks():
    turn = chain_assistant(model=RefusesExtraction()).ask(*M.QUESTIONS[0])
    assert "Plus plan" in turn["facts"] and "capped" not in turn["facts"], turn["facts"]   # degraded, not failed
    turn = chain_assistant(model=Down(), fallback=S.Scripted()).ask(*M.QUESTIONS[0])
    assert turn["reply"].payout == 400.0                                         # the second model answered


def test_streaming():
    want = S.SCRIPT[M.QUESTIONS[0][2]][1]
    parts = list(scratch_assistant().stream(*M.QUESTIONS[0], chunks=S.chunks))
    answers = [p["answer"] for p in parts if "answer" in p]
    assert len(answers) > 3 and all(len(x) < len(y) for x, y in zip(answers, answers[1:]))   # it grew
    assert "FIELDS" not in answers[-1]                                           # the marker is never shown
    assert parts[-1]["reply"].model_dump() == want
    assert list(chain_assistant().stream(*M.QUESTIONS[0]))[-1]["reply"].payout == 400.0
    # a reply whose fields do not parse falls back to the enforced call
    broken = lambda *args: ["The laptop is covered.", f"\n{M.FIELDS_MARKER} {{not json"]
    assert list(scratch_assistant().stream(*M.QUESTIONS[0], chunks=broken))[-1]["reply"].payout == 400.0


def test_double_payment_is_held():
    # both builds' live reply to message 6: right in prose, and fields that would pay twice
    live = {"covered": "yes", "answer": "Your $100 was already paid on 2 September.", "payout": 100.0,
            "clauses": ["3.1", "3.2"], "next_steps": []}

    class Repeats(S.Scripted):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if any(isinstance(m, SystemMessage) and m.content.startswith("You are Meridian") for m in messages):
                from langchain_core.messages import AIMessage
                from langchain_core.outputs import ChatGeneration, ChatResult
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content="", tool_calls=[{"name": "Reply", "args": live, "id": "r", "type": "tool_call"}]))])
            return super()._generate(messages, stop, run_manager, **kwargs)

    scratch = X.Assistant(index=X.ClauseIndex(embed=S.embed), rewrite=S.rewrite, lookups=S.lookups,
                          respond=lambda *a: M.Reply(**live))
    for turn in (chain_assistant(model=Repeats()).ask(*M.QUESTIONS[5]), scratch.ask(*M.QUESTIONS[5])):
        assert turn["reply"].payout is None and "CLM-20877" in turn["held"], turn
    # and a payout that repeats nothing is left alone
    for turn in (chain_assistant().ask(*M.QUESTIONS[0]), scratch_assistant().ask(*M.QUESTIONS[0])):
        assert turn["reply"].payout == 400.0 and turn["held"] is None


def test_batch_and_callbacks():
    assistant, steps = chain_assistant(), C.Steps()
    turns = assistant.batch([M.QUESTIONS[0], M.QUESTIONS[3], M.QUESTIONS[5]], callbacks=[steps])
    assert [t["reply"].covered for t in turns] == ["yes", "no", "no"]
    assert {"look_up", "retrieve", "respond", "check"} <= {row["step"] for row in steps.rows}


def test_the_retrieval_floor():
    # a stand-in embedding scores on its own scale, so it gets no floor unless it asks for one
    assert X.ClauseIndex(embed=S.embed).floor is None and len(X.ClauseIndex(embed=S.embed).search("stolen laptop")) == 4
    assert X.ClauseIndex(embed=S.embed, floor=1.01).search("stolen laptop") == []      # nothing clears it: nothing
    scratch = X.Assistant(index=X.ClauseIndex(embed=S.embed, floor=1.01), rewrite=S.rewrite, lookups=S.lookups,
                          respond=S.respond)
    assert scratch.ask("t", "MT-48213", M.QUESTIONS[0][2])["clauses"] == []          # and the assistant still answers


if __name__ == "__main__":
    for test in (test_both_builds_answer_the_inbox, test_fallbacks, test_streaming, test_double_payment_is_held,
                 test_batch_and_callbacks, test_the_retrieval_floor):
        test()
        print("ok ", test.__name__)
