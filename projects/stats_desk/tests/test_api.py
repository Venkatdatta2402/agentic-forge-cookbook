"""The web API with scripted agents: the real app, lead, analysts, analysis server and checks;
a stand-in for Groq's key check. No tokens."""

import asyncio
import json
import socket

import httpx
import pytest
from stubs import call, calls, plan
from test_team import GAP, findings_by_task, result_id

from stats_desk.agents.agent import Stopped
from stats_desk.agents.analyst import Finding
from stats_desk.agents.lead import follow_up_context
from stats_desk.agents.record import Record
from stats_desk.analysis.client import connect
from stats_desk.analysis.server import create
from stats_desk.api.analysis_server import SharedAnalysisServer
from stats_desk.api.app import create_app
from stats_desk.api.desk import Ask, KeyRefused, today
from stats_desk.agents.agent import FORM
from stats_desk.api.progress import step, update
from stats_desk.api.store import MemoryStore
from stats_desk.core import llm
from stats_desk.core.governor import DailyLimitReached, Governor, account_of

KEY = "gsk_test_key_never_stored"
QUESTION = "Why do young women in cities struggle to find work?"


async def accept(key):
    pass


def make(release_path, store=None, check_key=accept, max_running=3, free_keys=()):
    store = store or MemoryStore()
    app = create_app(store=store, connect=lambda: connect(server=create(release_path)), check_key=check_key,
                     release_path=release_path, max_running=max_running, free_keys=free_keys,
                     free_per_visitor=3)
    return app, store


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://desk")


def events_of(body):
    """[(event, data)] from a server-sent-events body."""
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n") if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def routed():
    """test_team's routed scripted model, for the API's tests."""
    from test_team import Routed
    holder = {}

    def install(scripts):
        holder["model"] = Routed(scripts)
        holder.setdefault("previous", llm.set_backend(holder["model"]))     # restored once, at the end
        holder.setdefault("governor", llm.set_governor(Governor(limits={})))
        llm.set_backend(holder["model"])
        return holder["model"]
    yield install
    if "model" in holder:
        llm.set_backend(holder["previous"])
        llm.set_governor(holder["governor"])


def gap_question():
    def brief(request):
        return call("json", answer="Among urban 15-29 year olds, women's unemployment is 18.9% against "
                    "men's 11.8% in 2025, a gap of 7.1 points.", findings=[findings_by_task(request)[GAP]],
                    cannot_tell="Why the gap exists.")
    return {
        "lead": [calls(("profile", {"names": ["UR_annual_2025"]})), calls(("ask_analysts", plan(GAP))), brief],
        f"analyst {GAP}": [
            call("gap", table="ur_annual", field="gender", a="female", b="male", across="sector",
                 where={"age": "15-29 years"}),
            lambda r: call("json", claim="Young urban women's unemployment is 18.9%, 7.1 points above young "
                           "urban men's.", results=[result_id(r, "gap")])],
    }


def no_numbers(answer="The figures can't answer that."):      # "not published" would need a refusal
    return {"lead": [call("json", answer=answer, findings=[], cannot_tell="Anything about why.")]}


async def test_a_question_streams_its_steps_and_saves_its_whole_record(release_path, routed):
    routed(gap_question())
    app, store = make(release_path)
    async with client(app) as c:
        r = await c.post("/questions", json={"question": QUESTION}, headers={"X-Groq-Key": KEY})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        got = events_of(r.text)
        assert [e for e, _ in got][0] == "started" and got[-1][0] == "brief"
        steps = [d["step"] for e, d in got if e == "step" and d["step"]]
        assert steps[:2] == ["Listing the published tables for the lead", "The lead is reading profiles: UR_annual_2025"]
        assert f"Analyst 1 started: {GAP}" in steps
        assert "Analyst 1: gap on ur_annual, female against male (15-29 years)" in steps
        assert any(s.startswith("Analyst 1 finished: Young urban women's") for s in steps)
        assert steps[-1] == "The lead is writing the brief"
        # what the page draws: every model call starts and ends, code's reruns, the finding's figures
        updates = [d for e, d in got if e == "step"]
        thoughts = [d for d in updates if d["kind"] == "thought"]
        assert len(thoughts) == len([d for d in updates if d["kind"] == "thinking"]) == 5
        assert {d["who"] for d in thoughts} == {"lead", "f1"}
        assert any(d["kind"] == "call" and d["who"] == "code" and d["tool"] == "recheck" for d in updates)
        found = next(d for d in updates if d["kind"] == "finding")
        assert found["finding"] == "f1" and any(k.startswith("gap on ur_annual") for k in found["figures"])
        assert all(k.startswith("rerun by code: gap on ur_annual") for k in found["elsewhere"]) and found["elsewhere"]
        brief_id = got[0][1]["id"]
        shown = (await c.get(f"/briefs/{brief_id}")).json()
    assert sum(d["tokens"] for d in thoughts) == Record.from_doc(store.docs[brief_id]).usage["total"]
    assert shown["status"] == "done" and "18.9%" in shown["answer"] and shown["figures_from"]
    assert shown["findings"][0]["finding"]["claim"].startswith("Young urban women") and "transcripts" not in shown
    # the record keeps everything: every agent's messages, every task, every result
    record = Record.from_doc(store.docs[brief_id])
    assert set(record.transcripts) == {"lead", "analyst f1"} and record.tasks == {"f1": GAP}
    assert record.plan[0]["approach"] and record.plan[0]["tasks"] == {"f1": GAP}           # the plan, kept
    assert any(i.startswith("gap-") for i in record.results) and record.usage["calls"] == 5
    assert KEY not in json.dumps(store.docs)                     # the key is never kept
    assert app.state.desk.active == 0 and store.leases == {}     # its slot and its key are free again


async def test_a_follow_up_can_cite_the_earlier_findings_without_asking_again(release_path, routed):
    routed(gap_question())
    app, store = make(release_path)
    async with client(app) as c:
        first = events_of((await c.post("/questions", json={"question": QUESTION},
                                         headers={"X-Groq-Key": KEY})).text)[0][1]["id"]
        model = routed({"lead": [call("json", answer="For young urban women it was 18.9% in 2025.",
                                      findings=["e1"], cannot_tell="Why.")]})
        r = await c.post("/questions", json={"question": "And what was it for women exactly?", "follow_up_of": first},
                         headers={"X-Groq-Key": KEY})
    got = events_of(r.text)
    assert got[-1][0] == "brief", got[-1]
    seen = model.asked[0][1]["messages"][1]["content"]
    assert "This is a follow-up" in seen and f"last question: {QUESTION}" in seen
    assert "e1 -- task: " + GAP in seen and "cite them as e1" in seen
    record = Record.from_doc(store.docs[got[0][1]["id"]])
    assert record.brief["findings"] == ["e1"] and record.parent == first and record.conversation == first
    assert record.thread == [QUESTION]


def test_a_follow_up_after_a_new_release_cannot_cite_the_old_figures():
    parent = Record("p", "How high is it?", release="old", status="done",
                    findings={"f1": Finding("f1", "t", "It is 18.9%.", ["gap-aaaaaa"], {"gap-aaaaaa": "18.9"}),
                              "f2": Stopped(agent="analyst f2", reason="budget")},
                    results={"gap-aaaaaa": "result gap-aaaaaa\n..."},
                    brief={"answer": "18.9%.", "findings": ["f1"], "cannot_tell": "Why."})
    record = Record.follow_up("n", "And now?", parent, release="new")
    assert list(record.earlier) == ["e1"] and record.results == {} and not record.earlier_citable
    text = follow_up_context(record)
    assert "re-released" in text and "[gap-aaaaaa]" not in text and "e1 -- task: t" in text
    same = Record.follow_up("n", "And now?", parent, release="old")
    assert same.earlier_citable and "gap-aaaaaa" in same.results and same.earlier_brief["findings"] == ["e1"]


def test_a_follow_ups_context_stays_under_its_cap():
    findings = {f"f{i}": Finding(f"f{i}", f"task {i} " + "words " * 40, "claim " * 30, [f"r-{i:06d}"],
                                 {f"r-{i:06d}": "figure line " * 60}) for i in range(1, 5)}
    parent = Record("p", "q " * 50, release="r", status="done", findings=findings, thread=["older " * 40] * 5,
                    brief={"answer": "a " * 100, "findings": ["f1"], "cannot_tell": "c"})
    text = follow_up_context(Record.follow_up("n", "next", parent, "r"))
    assert llm.count_tokens(text) <= 1_500 and "e1 -- task" in text


def test_a_record_saves_and_loads_whole():
    record = Record("x", "q", release="r", findings={"f1": Finding("f1", "t", "c", ["a-000001"]),
                                                    "f2": Stopped(agent="analyst f2", reason="budget"), "f3": None},
                    transcripts={"lead": [{"role": "user", "content": "q"}]})
    back = Record.from_doc(record.to_doc())
    assert back.findings["f1"] == record.findings["f1"] and back.findings["f2"].reason == "budget"
    assert back.findings["f3"] is None and back.transcripts == record.transcripts


class FakeGroq:
    """Stands in for Groq per key: a key in `dry` answers "daily limit"; the rest answer from
    the routed script. Lets the real QuestionKeys switch keys mid-question."""
    dry: set = set()
    script = None
    calls: list = []           # the key each call went out on

    def __init__(self, api_key=None):
        self.key = api_key

    async def create(self, **request):
        FakeGroq.calls.append(self.key)
        if self.key in FakeGroq.dry:
            raise DailyLimitReached("Groq's daily limit: tokens per day", retry_after=600)
        return await FakeGroq.script.create(**request)

    async def close(self):
        pass


@pytest.fixture
def groq(monkeypatch, routed):
    """Questions go through the real QuestionKeys to FakeGroq clients (free keys with the same
    tokens left are tried in list order)."""
    def install(scripts, dry=()):
        FakeGroq.script, FakeGroq.dry, FakeGroq.calls = routed(scripts), set(dry), []
        llm.set_backend(None)                    # no global scripted model: each key's own client
        return FakeGroq.script
    monkeypatch.setattr(llm, "GroqBackend", FakeGroq)
    return install


async def test_free_questions_start_on_the_key_with_most_left_and_count_what_they_use(release_path, groq):
    free = ["gsk_free_org_a", "gsk_free_org_b"]
    groq(no_numbers())
    app, store = make(release_path, free_keys=free)
    await store.add_usage(f"tokens-{account_of(free[0])}", 150_000)     # a has 50k left, b 200k
    async with client(app) as c:
        got = events_of((await c.post("/questions", json={"question": QUESTION})).text)
        assert got[-1][0] == "brief"
        left = (await c.get("/health")).json()["free_keys_tokens_left"]
    # it ran on b (most left), and b's count went up by what its calls used (300 + 40 a call)
    assert left == [50_000, 200_000 - 340]
    # a key under 20,000 left isn't started on; with both so, the free questions are used up
    await store.add_usage(f"tokens-{account_of(free[0])}", 40_000)
    await store.add_usage(f"tokens-{account_of(free[1])}", 190_000)
    async with client(app) as c:
        r = await c.post("/questions", json={"question": QUESTION}, headers={"X-Forwarded-For": "5.5.5.5"})
    assert r.status_code == 429 and "used up" in r.json()["detail"]


def test_a_keys_count_is_its_last_24_hours():
    clock = type("C", (), {"t": 1_000_000.0, "__call__": lambda self: self.t})()
    store = MemoryStore(clock=clock)

    async def go():
        await store.add_usage("k", 100)
        clock.t += 23 * 3600
        await store.add_usage("k", 10)
        assert await store.usage_24h("k") == 110
        clock.t += 3 * 3600                     # the first 100 is now 26 hours old
        assert await store.usage_24h("k") == 10
    asyncio.run(go())


async def test_a_free_key_that_runs_dry_midway_hands_the_question_to_the_other(release_path, groq):
    free = ["gsk_free_org_a", "gsk_free_org_b"]
    groq(gap_question(), dry={"gsk_free_org_a"})
    app, store = make(release_path, free_keys=free)
    async with client(app) as c:
        got = events_of((await c.post("/questions", json={"question": QUESTION})).text)
    assert got[-1][0] == "brief", got[-1]
    assert any("carrying on with the other free key" in (d.get("step") or "") for _, d in got)
    record = Record.from_doc(store.docs[got[0][1]["id"]])
    assert record.status == "done" and record.free and record.key_switches == 1
    assert await store.blocked(f"spent-{account_of('gsk_free_org_a')}")       # every machine skips it now
    assert store.leases == {}


async def test_a_lone_free_question_borrows_the_idle_free_key_call_by_call(release_path, groq):
    """Analysts shouldn't wait on one key's minute while the other sits idle."""
    free = ["gsk_free_org_a", "gsk_free_org_b"]
    groq(gap_question())
    app, store = make(release_path, free_keys=free)
    async with client(app) as c:
        got = events_of((await c.post("/questions", json={"question": QUESTION})).text)
    assert got[-1][0] == "brief", got[-1]
    assert set(FakeGroq.calls) == set(free)                       # both keys' minutes used
    used = [await store.usage_24h(f"tokens-{account_of(k)}") for k in free]
    assert all(used) and sum(used) == Record.from_doc(store.docs[got[0][1]["id"]]).usage["total"]
    assert store.leases == {}                                     # only its own key was ever leased


async def test_a_free_key_another_question_holds_is_not_borrowed(release_path, groq):
    free = ["gsk_free_org_a", "gsk_free_org_b"]
    groq(gap_question())
    app, store = make(release_path, free_keys=free)
    await store.claim(account_of(free[1]), "someone else's question")
    async with client(app) as c:
        got = events_of((await c.post("/questions", json={"question": QUESTION})).text)
    assert got[-1][0] == "brief" and set(FakeGroq.calls) == {free[0]}


async def test_a_borrowed_key_that_runs_dry_is_dropped_and_the_question_goes_on(release_path, groq):
    free = ["gsk_free_org_a", "gsk_free_org_b"]
    groq(gap_question(), dry={"gsk_free_org_b"})
    app, store = make(release_path, free_keys=free)
    async with client(app) as c:
        got = events_of((await c.post("/questions", json={"question": QUESTION})).text)
    assert got[-1][0] == "brief", got[-1]
    record = Record.from_doc(store.docs[got[0][1]["id"]])
    assert record.key_switches == 0 and FakeGroq.calls.count(free[1]) == 1     # tried once, then left
    assert await store.blocked(f"spent-{account_of(free[1])}")


async def test_with_every_free_key_dry_the_question_pauses_and_carries_on_with_the_users_own_key(release_path, groq):
    free = ["gsk_free_org_a", "gsk_free_org_b"]
    groq(gap_question(), dry=set(free))
    app, store = make(release_path, free_keys=free)
    async with client(app) as c:
        got = events_of((await c.post("/questions", json={"question": QUESTION})).text)
        assert got[-1][0] == "paused" and "own Groq key" in got[-1][1]["message"]
        brief_id = got[0][1]["id"]
        assert (await c.get(f"/briefs/{brief_id}")).json()["resumable"] is True
        # the next visitor is told at once: no free key left today
        r = await c.post("/questions", json={"question": QUESTION}, headers={"X-Forwarded-For": "9.9.9.9"})
        assert r.status_code == 429 and "used up" in r.json()["detail"]
        # the user's decision: carry it on with their own key
        groq(gap_question())
        got = events_of((await c.post(got[-1][1]["carry_on"], headers={"X-Groq-Key": KEY})).text)
    assert got[-1][0] == "brief", got[-1]
    record = Record.from_doc(store.docs[brief_id])
    assert record.status == "done" and record.resumes == 1 and not record.free
    assert {"lead", "lead, run 1"} <= set(record.transcripts)          # the paused run's messages kept


async def test_a_resumed_question_starts_from_what_it_had(release_path, routed):
    """Carried on: its findings are shown to the lead, unfinished tasks listed, numbering goes on."""
    routed(gap_question())
    app, store = make(release_path)
    async with client(app) as c:
        brief_id = events_of((await c.post("/questions", json={"question": QUESTION},
                                           headers={"X-Groq-Key": KEY})).text)[0][1]["id"]
    record = Record.from_doc(store.docs[brief_id])
    record.status, record.findings["f2"], record.tasks["f2"] = "stopped", None, "how did it change since 2024"
    await store.save(brief_id, record.to_doc())
    model = routed({"lead": [lambda r: call("json", answer="Young urban women's unemployment is 18.9%.",
                                            findings=["f1"], cannot_tell="Why.")]})
    async with client(app) as c:
        got = events_of((await c.post(f"/briefs/{brief_id}/resume", headers={"X-Groq-Key": KEY})).text)
    assert got[-1][0] == "brief", got[-1]
    seen = model.asked[0][1]["messages"][1]["content"]
    assert "had to stop partway" in seen and "f1 -- task: " + GAP in seen
    assert "- how did it change since 2024" in seen                   # unfinished: listed, not a finding
    async with client(app) as c:                                     # done now: nothing to carry on
        assert (await c.post(f"/briefs/{brief_id}/resume", headers={"X-Groq-Key": KEY})).status_code == 409


async def test_a_failed_question_says_why_in_plain_words_and_is_saved(release_path, model):
    async def broken(**request):
        raise RuntimeError("an outage")
    model.create = broken
    app, store = make(release_path)
    async with client(app) as c:
        r = await c.post("/questions", json={"question": "How high is unemployment?"}, headers={"X-Groq-Key": KEY})
    got = events_of(r.text)
    assert got[-1][0] == "failed" and "went wrong on our side" in got[-1][1]["error"]
    assert store.docs[got[0][1]["id"]]["status"] == "failed" and app.state.desk.active == 0


async def test_closing_the_page_stops_the_question_and_keeps_what_it_had(release_path, model):
    async def slow(**request):
        await asyncio.sleep(3600)
    model.create = slow
    app, store = make(release_path)
    desk = app.state.desk
    record, _, task = await desk.ask(Ask(question="How high is unemployment?"), KEY, "v")
    await asyncio.sleep(0.3)
    task.cancel()                                   # what the stream does when the page closes
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = Record.from_doc(store.docs[record.id])
    assert saved.status == "stopped" and "page was closed" in saved.stopped and saved.transcripts["lead"]
    assert desk.active == 0 and store.leases == {}


async def test_refusals_before_anything_runs(release_path):
    async def refuse(key):
        raise KeyRefused("Groq didn't accept this key.")
    ask = {"question": "How high is unemployment?"}
    app, store = make(release_path, check_key=refuse)
    async with client(app) as c:
        r = await c.post("/questions", json=ask)                   # no key, and no free questions
        assert r.status_code == 401 and "X-Groq-Key" in r.json()["detail"]
        r = await c.post("/questions", json=ask, headers={"X-Groq-Key": "bad"})
        assert r.status_code == 401 and "didn't accept" in r.json()["detail"]
    assert app.state.desk.active == 0 and store.docs == {}

    app, store = make(release_path)
    await store.claim(account_of(KEY), "abc")                       # a question on this key, on any machine
    async with client(app) as c:
        r = await c.post("/questions", json=ask, headers={"X-Groq-Key": KEY})
        assert r.status_code == 409 and "/briefs/abc" in r.json()["detail"]
    assert app.state.desk.active == 0

    app, _ = make(release_path, max_running=0)
    async with client(app) as c:
        r = await c.post("/questions", json=ask, headers={"X-Groq-Key": KEY})
        assert r.status_code == 503 and r.headers["retry-after"] == "120"


async def test_free_questions_on_the_desks_keys(release_path, routed):
    free = ["gsk_free_org_a", "gsk_free_org_b"]
    app, store = make(release_path, free_keys=free)
    ask = {"question": "How high is unemployment among the young?"}
    alice, bob = {"X-Forwarded-For": "1.1.1.1"}, {"X-Forwarded-For": "2.2.2.2"}
    async with client(app) as c:
        # 3 a day each, questions and follow-ups together: a question, a follow-up, a question
        routed(no_numbers())
        got = events_of((await c.post("/questions", json=ask, headers=alice)).text)
        assert got[0][1]["free"] is True and got[-1][0] == "brief"
        routed(no_numbers())
        follow = {"question": "And for women?", "follow_up_of": got[0][1]["id"]}
        assert events_of((await c.post("/questions", json=follow, headers=alice)).text)[-1][0] == "brief"
        routed(no_numbers())
        assert events_of((await c.post("/questions", json=ask, headers=alice)).text)[-1][0] == "brief"
        assert (await c.get("/free", headers=alice)).json()["left"] == 0         # the page's count
        assert (await c.get("/free", headers=bob)).json()["left"] == 3
        r = await c.post("/questions", json=follow, headers=alice)    # a 4th, even a follow-up: no
        assert r.status_code == 429 and "3 free questions (follow-ups included)" in r.json()["detail"]
        # both keys busy right now (on any machine): try again soon
        for k in free:
            await store.claim(account_of(k), "someone")
        r = await c.post("/questions", json=ask, headers=bob)
        assert r.status_code == 503
        # Groq saying a key's day is spent closes it too, whatever the count says
        store.leases.clear()
        for k in free:
            await store.block(f"spent-{account_of(k)}", 600)
        r = await c.post("/questions", json=ask, headers=bob)
        assert r.status_code == 429 and "used up" in r.json()["detail"]
    assert list(store.counters.values()) == [3]          # alice's three; refused asks don't count
    assert not any("gsk_free" in json.dumps(d) for d in store.docs.values())
    assert not any("1.1.1.1" in name for name in store.counters)       # visitors by a hash only


async def test_pages_that_spend_no_tokens(release_path):
    store = MemoryStore()
    pub = Record("pub", "An example", release="r", status="done", brief={"answer": "A."}, public=True)
    await store.save("pub", pub.to_doc())
    await store.save("priv", Record("priv", "Someone's question", release="r", status="done",
                                    brief={"answer": "B."}).to_doc())
    app, _ = make(release_path, store)
    async with client(app) as c:
        assert [e["id"] for e in (await c.get("/examples")).json()] == ["pub"]
        assert (await c.get("/briefs/priv")).json()["answer"] == "B."       # by its link only
        assert (await c.get("/briefs/nope")).status_code == 404
        assert "UR_annual_2025" in (await c.get("/profiles")).json()["profiles"]
        assert (await c.get("/profiles/UR_annual_2025")).json()["text"].startswith("PROFILE unemployment rate")
        assert (await c.get("/profiles/nope")).status_code == 404
        assert (await c.get("/health")).json()["ok"] is True
        page = await c.get("/")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert "<title>Stats Desk</title>" in page.text and 'fetch("/free")' in page.text


async def test_one_shared_analysis_server_over_http(release_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    shared = SharedAnalysisServer(release_path, port)
    try:
        async with shared.connect() as a, shared.connect() as b:      # two questions, one server
            first, second = await asyncio.gather(a.call("tables"), b.call("tables"))
        assert first == second and "ur_annual" in first and shared.alive()
        assert not b.results.keys() - {first.split("\n")[0].removeprefix("result ")}
    finally:
        shared.stop()


def test_progress_steps_are_plain_lines_without_finding_ids():
    assert step({"kind": "tool_call", "agent": "lead", "tool": "profile",
                 "arguments": '{"names": ["UR_annual_2025"]}'}) == "The lead is reading profiles: UR_annual_2025"
    assert step({"kind": "tool_call", "agent": "analyst f2", "tool": "relate",
                 "arguments": '{"x_table": "lfpr_annual"}'}) == "Analyst 2: relate on lfpr_annual"
    assert step({"kind": "tool_call", "agent": "analyst f1", "tool": "change",
                 "arguments": '{"table": "ur_monthly", "by": "gender", "where": {"period": "2026-09"}}'}) == \
        "Analyst 1: change on ur_monthly by gender (2026-09)"
    assert step({"kind": "finding", "agent": "lead", "finding": "f2", "claim": "C.", "warnings": 2}) == \
        "Analyst 2 finished: C. (2 warnings)"
    assert step({"kind": "tool_result", "agent": "analyst f1", "tool": "value", "result_kind": "ok"}) is None
    assert step({"kind": "waiting", "agent": "analyst f1", "seconds": 22}) == \
        "Analyst 1 waits 22 s for room in Groq's per-minute limit"


def test_the_page_gets_who_did_what():
    """What the page draws from (api/progress.py): each event says who, what, and on what."""
    call = update({"kind": "tool_call", "agent": "analyst f1", "tool": "gap",
                   "arguments": '{"table": "ur_annual", "field": "gender", "a": "female", "b": "male"}'})
    assert call == {"who": "f1", "kind": "call", "tool": "gap", "mcp": True, "label": "gap on ur_annual, female against male",
                    "step": "Analyst 1: gap on ur_annual, female against male"}
    assert update({"kind": "tool_call", "agent": "lead", "tool": "ask_analysts", "arguments": "{}"}) is None
    assert update({"kind": "tool_call", "agent": "lead", "tool": FORM, "arguments": "{}"})["mcp"] is False
    sent_back = update({"kind": "tool_result", "agent": "analyst f2", "tool": FORM, "result_kind": "refused",
                        "preview": "These numbers are not in the results you cited: 3.0\nFix them."})
    assert sent_back["mcp"] is False and sent_back["step"] == \
        "Analyst 2's finding was sent back to fix: These numbers are not in the results you cited: 3.0"
    assert update({"kind": "tool_result", "agent": "analyst f2", "tool": FORM, "result_kind": "ok"}) is None
    refused = update({"kind": "tool_result", "agent": "analyst f2", "tool": "value", "result_kind": "refused",
                      "preview": "ur_annual: gender x religion x state is not published together."})
    assert refused["refused"] and refused["message"].startswith("ur_annual: gender x religion")
    assert update({"kind": "check", "agent": "lead", "finding": "f1", "analysis": "trend",
                   "arguments": {"table": "ur_monthly"}})["who"] == "code"
    assert update({"kind": "thought", "agent": "lead, run 2", "tokens": 900}) == \
        {"who": "lead", "kind": "thought", "step": None, "tokens": 900}
    assert update({"kind": "recovered", "agent": "lead", "called": "x", "tool": "profile"}) is None
    shown = update({"kind": "plan", "agent": "lead", "approach": "Graduates first, then by gender.",
                    "tasks": {"f1": "t"}, "later": ["By gender, if graduates are high"]})
    assert shown["later"] == ["By gender, if graduates are high"] and shown["step"] == \
        "The lead's plan: Graduates first, then by gender. (later: By gender, if graduates are high)"


async def test_a_key_is_used_for_its_question_only():
    assert llm._backend is None
    async with llm.using_key("gsk_one") as keys:
        assert llm.backend() is keys and keys.client._client.api_key == "gsk_one"
        assert keys.account == account_of("gsk_one")
    assert llm._question.get() is None


async def test_a_rejected_call_still_goes_on_the_keys_count(monkeypatch):
    """Groq charges a rejected call (its context and what the model wrote) but says no number:
    the count gets our estimate -- the visible rejected text counted exactly, or a typical
    reply for the agent, whichever is larger."""
    from test_runtime import rejected

    from stats_desk.core import meter
    long_hand_in = "{" + '"answer": "' + "words " * 600 + '"}'
    for generated, agent, least in (("short", "analyst f1", 200), (long_hand_in, "lead", 600)):
        step = rejected(generated)

        class Rejecting(FakeGroq):
            async def create(self, **request):
                step(request)
        monkeypatch.setattr(llm, "GroqBackend", Rejecting)
        counted = []

        async def used(key, tokens):
            counted.append((key, tokens))
        previous = llm.set_governor(Governor(limits={}))
        messages = [{"role": "user", "content": "hi"}]
        try:
            with meter.acting_as(agent):
                async with llm.using_key("k", on_usage=used):
                    with pytest.raises(llm.BadRequestError):
                        await llm.complete("m", messages, max_tokens=1500)
        finally:
            llm.set_governor(previous)
        context = llm.estimate(messages)
        assert counted[0][0] == "k" and counted[0][1] - context >= least
        assert counted[0][1] - context <= 1500                         # never past max_tokens


async def test_analysts_hitting_a_dry_key_at_once_switch_it_only_once(monkeypatch):
    from stubs import say
    monkeypatch.setattr(llm, "GroqBackend", FakeGroq)
    FakeGroq.dry, FakeGroq.script = {"a"}, type("S", (), {"create": staticmethod(
        lambda **r: asyncio.sleep(0, (say("ok"), {})))})()
    asked = []

    async def spare(spent, retry_after):
        asked.append(spent)
        return "b" if spent == "a" else None
    previous = llm.set_governor(Governor(limits={}))
    try:
        async with llm.using_key("a", spare) as keys:
            replies = await asyncio.gather(*(llm.complete("m", [{"role": "user", "content": "hi"}]) for _ in range(3)))
    finally:
        llm.set_governor(previous)
    assert [r.content for r in replies] == ["ok"] * 3 and asked == ["a"] and keys.key == "b"
