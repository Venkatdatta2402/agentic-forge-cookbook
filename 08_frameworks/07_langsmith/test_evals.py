"""Offline tests for 07.   python test_evals.py   (costs nothing: no Groq, no Gemini, nothing sent to LangSmith)"""

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")

import eval_set as E  # noqa: E402 -- first: puts 05 on the path
import ask_index as A  # noqa: E402
import ask_stubs as S05  # noqa: E402
import claim_judge as J  # noqa: E402
import cookbook as CB  # noqa: E402
import cookbook_evals as C  # noqa: E402
import eval_stubs as S  # noqa: E402
import traces_checks as K  # noqa: E402
import tracing as T  # noqa: E402

BOARD = E.QUESTIONS[7]          # "How is chapter 6's shared board different from chapter 4's memory stores? ..."


def small_index():
    index = A.open_index(path=tempfile.mkdtemp(), embed=S05.WordEmbedding())
    A.refresh(index, [d for d in CB.documents() if d.id_ in ("06_orchestration/state.py", "04_memory/stores.py",
                                                            "05_context/compression.py")], path=None)
    return index


def offline(test):
    """No tracing, and answers saved to a scratch folder rather than store/."""
    def run():
        from langsmith import tracing_context
        store, C.STORE = C.STORE, Path(tempfile.mkdtemp())
        try:
            with tracing_context(enabled=False):
                test()
        finally:
            C.STORE = store
    run.__name__ = test.__name__
    return run


def test_every_point_rests_on_words_in_its_evidence():
    assert [q["kind"] for q in E.QUESTIONS].count("one-shot") == 7 and len(E.QUESTIONS) == 10
    for q in E.QUESTIONS:
        assert not E.unquoted(q), (q["question"], E.unquoted(q))
        assert all(quotes for _, quotes in q["points"])                  # no point without a quote
        assert set(q["files"]) <= {d.id_ for d in CB.documents()}, q["question"]   # files the assistant can read
    assert len({q["question"] for q in E.QUESTIONS}) == 10


def test_the_examples_carry_the_key():
    ex = C.examples()
    assert len(ex) == 10 and ex[7]["inputs"] == {"question": BOARD["question"], "kind": "agent"}
    assert ex[7]["outputs"]["points"] == [p for p, _ in BOARD["points"]] and ex[0]["outputs"]["word"] == "sense"


@offline
def test_tracing_changes_nothing_that_is_sent():
    # the same question through 05's engine and agent, with LangSmith's wrappers and without: the requests that
    # reach Groq must be identical, or the traced system is not the system being evaluated
    index, A.FLOOR = small_index(), 0.0
    try:
        same = asyncio.run(K.tracing_is_invisible(index, "what does compaction leave?", BOARD["question"]))
        assert same["requests, untraced"] == 3 and same["identical"]
        assert same["tools offered, traced"] == ["search_cookbook['query']",
                                                 "open_file['line_end', 'line_start', 'path']"]
    finally:
        A.FLOOR = 0.66


@offline
def test_an_answer_is_saved_before_it_is_returned():
    index, A.FLOOR = small_index(), 0.0
    try:
        chat_model, _ = S.fake_groq()
        answer = C.target(index, "test-exp", chat_model=chat_model)
        out = asyncio.run(answer({"question": BOARD["question"], "kind": "agent"}))
        kept = C.saved("test-exp")[BOARD["question"]]
        assert kept == out and kept["cited"] == ["06_orchestration/state.py"]
        assert kept["calls"][0]["tool"] == "search_cookbook" and kept["usage"]["calls"] == 2
        one = asyncio.run(answer({"question": E.QUESTIONS[4]["question"], "kind": "one-shot"}))
        assert one["answer"] == "An answer [1]." and one["cited"] and one["usage"]["calls"] == 1
        assert len(C.saved("test-exp")) == 2
    finally:
        A.FLOOR = 0.66


@offline
def test_groqs_daily_limit_is_not_an_answer():
    # a question cut off by Groq's daily limit is not saved, so the next run answers it rather than keeping a stop
    import httpx
    import openai
    index = small_index()
    body = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
                                 "Limit 200000, Used 199800, Requested 900.", "type": "tokens",
                      "code": "rate_limit_exceeded"}}

    def chat_model(budget=None, wrap=None, **kwargs):
        def through(client):
            transport = httpx.MockTransport(lambda r: httpx.Response(429, json=body, headers={"retry-after-ms": "1"}))
            http = (httpx.AsyncClient if isinstance(client, openai.AsyncOpenAI) else httpx.Client)(transport=transport)
            return client.with_options(http_client=http, max_retries=0)
        return A.chat_model(budget=budget, wrap=through, **kwargs)

    try:
        asyncio.run(C.target(index, "test-exp", chat_model=chat_model)({"question": BOARD["question"], "kind": "agent"}))
        assert False, "a day-limit stop was returned as an answer"
    except C.GroqDayLimit:
        pass
    assert C.saved("test-exp") == {}


def test_the_judge_waits_a_minute_and_stops_for_the_day():
    folder = Path(tempfile.mkdtemp())
    one, two, three = E.QUESTIONS[0]["question"], E.QUESTIONS[1]["question"], E.QUESTIONS[2]["question"]
    answers = {one: {"answer": "Shape, not sense."}, two: {"answer": None, "stopped": "APIStatusError: 413"},
               three: {"answer": "Refused goes to the model."}}
    claims = {"claims": [{"claim": "shape only", "quote": "Shape"}, {"claim": "sense not", "quote": "not sense"}]}
    marks = {"marks": [{"n": 1, "verdict": "supported", "source": "llm.py 92", "why": ""}], "points_made": [1, 9]}
    waits = []
    gemini = S.FakeGemini([S.rate_limited(delay="7s"), claims, marks, S.rate_limited(per_day=True)])
    verdicts, finished = J.judge_all(answers, folder / "v.json", client=gemini, sleep=waits.append, log=lambda *a: None)
    assert not finished and waits == [9.0]                     # waited as long as Gemini asked, plus a margin
    assert set(verdicts) == {one, two} and verdicts[two] == {"no answer": "APIStatusError: 413"}
    assert verdicts[one]["points made"] == [1]                 # a point number the question does not have is dropped
    assert verdicts[one]["unmarked"] == 1 and verdicts[one]["claims"][1]["verdict"] == "unmarked"
    assert len(gemini.prompts) == 4                            # nothing was sent for the answer that never came

    again = S.FakeGemini([claims, marks])                      # the next day: only what is missing
    verdicts, finished = J.judge_all(answers, folder / "v.json", client=again, sleep=waits.append, log=lambda *a: None)
    assert finished and len(again.prompts) == 2 and set(verdicts) == {one, two, three}


def test_the_judge_reads_the_answer_before_it_sees_the_key():
    claims = [{"claim": "the board lives on disk", "quote": "lives on **disk**"}]
    extract, check = J.prompts(BOARD, "The board lives on disk.", claims)
    assert "The board lives on disk." in extract
    for point, _ in BOARD["points"]:
        assert point not in extract and point in check            # the key only once the claims are fixed
    assert "neither of them can trust" not in extract and "[06_orchestration/state.py, lines 1-20]" in check
    assert '1. the board lives on disk\n   words: "lives on **disk**"' in check
    assert J._unanchored(claims + [{"claim": "x", "quote": "lives in memory"}], "The board lives on disk.") == [
        {"claim": "x", "quote": "lives in memory"}]


def test_scores():
    marks = ["supported", "contradicted", "contradicted", "not in the sources"]
    verdict = {"claims": [{"verdict": m} for m in marks], "points made": [1, 2]}
    assert J.scores(verdict, E.QUESTIONS[4]) == {"claims wrong": 2, "claims not in the sources": 1, "claims": 4,
                                                 "points made": 0.5}
    assert J.scores({"no answer": "stopped"}, E.QUESTIONS[4]) == {}


@offline
def test_nothing_is_published_until_every_answer_is_judged():
    C._save("test-exp", BOARD["question"], {"answer": "x"})
    assert C.publish(client=None, name="test-exp", log=lambda *a: None) is False      # never reaches LangSmith


def test_cited_files():
    assert C.cited("see [04_memory/recall.py] and 【06_orchestration/state.py†L1-L7】.") == [
        "04_memory/recall.py", "06_orchestration/state.py"]


def test_checks():
    ids = K.otel_ids()
    assert not ids["the same run"] and ids["the run id the server makes of it"].startswith("00000000-0000-0000")
    sig = K.traceable_signature()
    assert "config" in sig["traceable"]["the model is offered"]
    assert sig["tracing.traced_tool"] == sig["plain"]


if __name__ == "__main__":
    for test in (test_every_point_rests_on_words_in_its_evidence, test_the_examples_carry_the_key,
                 test_tracing_changes_nothing_that_is_sent, test_an_answer_is_saved_before_it_is_returned,
                 test_groqs_daily_limit_is_not_an_answer, test_the_judge_waits_a_minute_and_stops_for_the_day,
                 test_the_judge_reads_the_answer_before_it_sees_the_key, test_scores,
                 test_nothing_is_published_until_every_answer_is_judged, test_cited_files, test_checks):
        test()
        print("ok ", test.__name__)
