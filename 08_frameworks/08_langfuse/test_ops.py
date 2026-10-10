"""Offline tests for 08.   python test_ops.py   (costs nothing: no Groq, no Gemini, nothing sent to Langfuse)"""

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace as NS

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")

import cookbook_ops as O  # noqa: E402 -- first: puts 05 and 07 on the path
import ask_index as A  # noqa: E402

sys.path.insert(0, str(O.HERE.parent / "07_langsmith"))
import eval_stubs as S  # noqa: E402 -- 07's scripted Groq and Gemini
import test_evals as TE  # noqa: E402


class FakeLangfuse:
    """The few client calls cookbook_ops makes, recorded rather than sent."""

    def __init__(self):
        self.scores, self.queries, self.spans = [], [], []
        self.api = NS(metrics=NS(metrics=self._metrics))

    def get_prompt(self, name, version=None, **kwargs):
        return NS(name=name, version=version, prompt=O.PROMPTS[version])

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs):
        span = NS(updates=[], update=lambda **u: span.updates.append(u))
        self.spans.append((kwargs, span))
        yield span

    def get_current_trace_id(self):
        return f"trace-{len(self.spans)}"

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        pass

    def _metrics(self, query):
        self.queries.append(json.loads(query))
        return NS(data=[{"promptVersion": 1, "sum_totalCost": 0.01}])


def scratch(test):
    def run():
        store, O.STORE = O.STORE, Path(tempfile.mkdtemp())
        try:
            test()
        finally:
            O.STORE = store
    run.__name__ = test.__name__
    return run


def test_the_two_prompt_versions():
    assert O.PROMPTS[1] == A.AGENT_PROMPT                      # version 1 is 05's prompt, unchanged
    assert O.PROMPTS[2].startswith(A.AGENT_PROMPT) and "gives up" in O.PROMPTS[2] and "LangGraph" not in O.PROMPTS[2]
    assert [q["kind"] for q in O.AGENT_QUESTIONS] == ["agent"] * 3


@scratch
def test_each_version_is_the_prompt_the_agent_is_given_and_each_answer_is_saved():
    client, index = FakeLangfuse(), TE.small_index()
    chat_model, sent = S.fake_groq()
    asyncio.run(O.run_version(client, index, 2, chat_model=chat_model, log=lambda *a: None))
    assert all(body["messages"][0]["content"] == O.PROMPTS[2] for body in sent)        # every request: version 2
    answers = O.saved(2)
    assert len(answers) == 3 and {a["trace"] for a in answers.values()} == {"trace-1", "trace-2", "trace-3"}
    assert all(span.updates[0]["output"]["answer"] for _, span in client.spans)
    before = len(sent)
    asyncio.run(O.run_version(client, index, 2, chat_model=chat_model, log=lambda *a: None))
    assert len(sent) == before                                  # nothing asked twice


@scratch
def test_groqs_daily_limit_is_not_an_answer():
    import httpx
    import openai
    body = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
                                 "Limit 200000, Used 199800, Requested 900.", "type": "tokens",
                      "code": "rate_limit_exceeded"}}

    def chat_model(budget=None, wrap=None, **kwargs):
        def through(client):
            transport = httpx.MockTransport(lambda r: httpx.Response(429, json=body, headers={"retry-after-ms": "1"}))
            kind = httpx.AsyncClient if isinstance(client, openai.AsyncOpenAI) else httpx.Client
            return client.with_options(http_client=kind(transport=transport), max_retries=0)
        return A.chat_model(budget=budget, wrap=through, **kwargs)

    try:
        asyncio.run(O.run_version(FakeLangfuse(), TE.small_index(), 1, chat_model=chat_model, log=lambda *a: None))
        assert False, "a day-limit stop was saved as an answer"
    except O.GroqDayLimit:
        assert O.saved(1) == {}


@scratch
def test_points_made_go_on_each_trace_once():
    import claim_judge as J
    client, q = FakeLangfuse(), O.AGENT_QUESTIONS[0]
    O._save(1, q["question"], {"kind": "agent", "answer": "The board is not memory.", "stopped": None, "trace": "t-1"})
    judged = {"claims": [{"claim": "not memory", "quote": "not memory"}]}
    marks = {"marks": [{"n": 1, "verdict": "supported", "source": "point 1", "why": ""}], "points_made": [1, 3]}
    real = J.judge
    J.judge = lambda question, answer, client=None, **kw: real(question, answer, client=S.FakeGemini([judged, marks]),
                                                               **kw)
    try:
        O.judge(client, 1, log=lambda *a: None)
        O.judge(client, 1, log=lambda *a: None)                # again: nothing new to judge, nothing scored twice
    finally:
        J.judge = real
    assert len(client.scores) == 1 and client.scores[0]["trace_id"] == "t-1"
    assert client.scores[0]["value"] == round(2 / len(q["points"]), 3) and client.scores[0]["name"] == "points made"


def test_a_metrics_query():
    client = FakeLangfuse()
    rows = O.metrics(client, ["promptVersion"], [("totalCost", "sum")], filters=[O.tagged(2)])
    query = client.queries[0]
    assert rows == [{"promptVersion": 1, "sum_totalCost": 0.01}]
    assert query["view"] == "observations" and query["dimensions"] == [{"field": "promptVersion"}]
    assert {"column": "tags", "operator": "any of", "value": ["prompt v2"], "type": "arrayOptions"} in query["filters"]
    assert {"column": "environment", "operator": "=", "value": "default", "type": "string"} in query["filters"]


def test_the_price_is_groqs():
    assert O.GROQ_PRICE == {"input": 0.15e-6, "output": 0.60e-6}
    assert round(900 * O.GROQ_PRICE["input"] + 60 * O.GROQ_PRICE["output"], 6) == 0.000171   # as Langfuse priced it


if __name__ == "__main__":
    for test in (test_the_two_prompt_versions,
                 test_each_version_is_the_prompt_the_agent_is_given_and_each_answer_is_saved,
                 test_groqs_daily_limit_is_not_an_answer, test_points_made_go_on_each_trace_once,
                 test_a_metrics_query, test_the_price_is_groqs):
        test()
        print("ok ", test.__name__)
