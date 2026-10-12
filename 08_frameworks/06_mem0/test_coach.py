"""Offline tests for both coaches, and for chapter 4's QdrantJournal.   python test_coach.py   (costs nothing)"""

import os
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")
os.environ["MEM0_DIR"] = tempfile.mkdtemp()           # Mem0's id file and default history, kept out of the repo

import coach_checks as K  # noqa: E402
import coach_mem0 as M  # noqa: E402
import coach_scratch as S  # noqa: E402
import coach_stubs as T  # noqa: E402
import runners as R  # noqa: E402


def test_the_scenario():
    dates = [s["date"] for s in R.SESSIONS]
    assert dates == sorted(dates) and len(set(dates)) == len(dates)          # the stand-ins find check-ins by date
    assert set(R.CHECKS) <= {s["id"] for s in R.SESSIONS} and set(R.CURRENT) == set(R.RUNNERS)
    got = R.leads("maya-6", "Your race is on 1 November (it was 18 Oct); aim for 1:45. Run Tue, Friday, Sun mornings.")
    assert "the race on 1 November" in got["needs, mentioned"] and "the goal of 1:45" in got["needs, mentioned"]
    assert got["old facts mentioned"] == ["the old race date: 18 Oct"]
    assert R.found("finished at 11:45", ["1:45"]) == []                        # whole times only


def test_the_two_journals_agree():
    import stores
    from pipeline import _chroma_where
    from recall import Recall
    entries = [("2026-08-12", "billing-api", "concluded", "PGRST204 came from a stale schema cache after the migration"),
               ("2026-09-02", "checkout-api", "concluded", "The pool ran dry under peak traffic; max_connections was 20"),
               ("2026-11-06", "checkout-api", "concluded", "The backfill locked the orders table and checkout slowed"),
               ("2026-11-07", "checkout-api", "observed", "Backfill batches over 500 rows hold the lock too long")]
    root, seen = Path(tempfile.mkdtemp()), {}
    with T.embeddings():
        for name, cls in (("sqlite+chroma", stores.Journal), ("qdrant", stores.QdrantJournal)):
            j = cls(root / name)
            ids = j.write(entries)
            r = {"exact": [[x["id"] for x in j.find_exact(t)] for t in ("PGRST204", "PGRST20", "pgrst204", "zebra")],
                 "like": [(i, round(s, 3)) for s, i, _ in j.find_like("checkout slowed by the backfill", k=3)],
                 "like where": [i for _, i, _ in j.find_like("the backfill and the lock", k=3, where=_chroma_where(
                     {"about": ["checkout-api"], "since": "2026-11-01"}))],
                 "recent": [x["id"] for x in j.recent(about="checkout-api", limit=2)],
                 "count": j.count(about="checkout-api"), "values": j.values("about"),
                 "candidates": j.candidates("about", "anything about billing")}
            j.update(ids[2], "The backfill held a row lock per batch", now="2026-11-08")
            j.pin(ids[0])
            j.used([ids[1]], "2026-11-10")
            r.update(updated=([x["id"] for x in j.find_exact("orders")], j.get(ids[2])["written"]),
                     kept=(j.is_pinned(ids[0]), j.accesses(ids[1])),
                     recall=[i for _, i, _, _ in Recall(j, today="2026-11-20", floor=0.0)("checkout backfill lock", k=2)],
                     stale=Recall(j, today="2027-06-01", evict_below=0.05, record_use=False).stale())
            j.forget(ids[3])
            r["forgotten"] = (len(j), j.get(ids[3]))
            seen[name] = r
    assert seen["sqlite+chroma"] == seen["qdrant"], seen
    assert seen["qdrant"]["exact"] == [[1], [], [1], []]                        # a near-miss code finds nothing
    with T.embeddings():                                                       # where the two differ: stemming
        assert [x["id"] for x in stores.QdrantJournal(root / "qdrant").find_exact("batches")] == [3]


def test_the_scratch_coach_offline():
    root = Path(tempfile.mkdtemp())
    with T.embeddings(), T.groq():
        done, ok = S.run(root=root, path=root / "scratch.json", log=lambda *_: None)
    assert ok and all(done[s["id"]]["reply"] for s in R.SESSIONS)
    maya = S.holdings("maya", root)
    assert {k: v["value"] for k, v in maya["profile"].items()} == {
        "race": "Riverside Half Marathon, Sunday 1 November (moved from 18 October)", "goal": "1:45",
        "days": "Tuesday, Friday and Sunday (new job)", "time_of_day": "mornings, before work",
        "injuries": "none: knee cleared by a physio on 22 September (tight IT band)"}
    assert maya["profile"]["goal"]["was"] == "under 1:50"                      # replaced, not lost
    assert [r["km"] for r in maya["runs"] if r["kind"] == "long"] == [14, 16, 17, 18]   # events never merged
    assert len(maya["notes"]) == 1
    shown = done["maya-6"]["shown"]
    assert "1 November" in shown and "Thursday" not in shown and "evenings" not in shown
    tom = S.holdings("tom", root)
    assert "Maya" not in str(tom) and "Riverside" not in str(tom) and not done["tom-3"]["kept"]


def test_the_mem0_coach_offline():
    m, _ = K.offline_memory()
    root = Path(tempfile.mkdtemp())
    with T.groq() as seen:
        done, ok = M.run(m, path=root / "mem0.json", log=lambda *_: None)
    assert ok and all("added" in done[s["id"]] for s in R.SESSIONS)
    assert K.isolation(done)["from the other runner"] == []
    assert all(r["parts"].keys() == {"meaning", "keywords", "entities"} for r in done["maya-6"]["recalled"])
    races = [r["memory"] for r in M.holdings(m, "maya") if "Riverside" in r["memory"]]
    assert len(races) == 2 and any("18 October" in r for r in races)           # the old date is still held
    system = next(s["messages"][0]["content"] for s in seen if s["session"] == "maya-6")
    assert "What you remember about Maya" in system and "(2026-09-22)" in system   # memories carry their check-in


def test_round_two_offline():
    root = Path(tempfile.mkdtemp())
    with T.embeddings(), T.groq() as seen:
        done, ok = S.run(root=root, path=root / "s.json", log=lambda *_: None, write="step")
    assert ok and not any(s["tools"] for s in seen)                              # the coach only replies now
    assert "Tuesday 2026-09-01" in str(seen[1]["messages"])                     # the write step knows the weekday
    maya = S.holdings("maya", root)
    assert maya["profile"]["race"]["value"].startswith("Riverside Half Marathon, Sunday 1 November")
    assert [r["km"] for r in maya["runs"] if r["kind"] == "long"] == [14, 16, 17, 18]
    assert done["maya-1"]["kept"][0]["said"].startswith("Kept: race =")
    m, llm = K.offline_memory()
    with T.groq():
        done, ok = M.run(m, path=root / "m.json", log=lambda *_: None, instructions=M.RUNNER_ONLY)
    assert ok and all("Do not extract the coach's recommendations" in p[-1]["content"] for p in llm.prompts)


def test_round_three_offline():
    assert S.unquoted("Monday, Tuesday, Wednesday, Thursday, Friday", "I run five days a week",
                      "I run five days a week and do my long run on Saturdays") == \
        "not in the quote: monday, tuesday, wednesday, thursday, friday"                # a made-up value
    assert S.unquoted("10", "a 10k race", "I ran a 10k race on Saturday") is None              # 10k is 10
    assert S.unquoted("1:45", "go for 1:45", "I've decided to go for 1:45") is None
    assert S.unquoted("x", "words never said", "I ran a 10k race") == "the quote is not in the message"
    root = Path(tempfile.mkdtemp())
    with T.embeddings(), T.groq():
        done, ok = S.run(root=root, path=root / "s.json", log=lambda *_: None, write="checked")
    assert ok
    # the stand-in's careful values add dates the message never says ("moved from 18 October", the knee "on 10
    # September" where the runner said "on Thursday"): both refused
    refused = {r["field"]: r["why"] for r in done["maya-3"]["kept"][0]["refused"]}
    assert refused == {"race": "not in the quote: 18, october", "injuries": "not in the quote: 10, september"}
    maya = S.holdings("maya", root)
    assert maya["profile"]["goal"]["was"] == "under 1:50" and maya["profile"]["goal"]["was_since"] == "2026-09-01"
    assert "before: under 1:50 (since 2026-09-01)" in done["maya-6"]["shown"]          # the old value, dated
    assert sum(n["source"] == "runner said" for n in maya["notes"]) == 6               # every check-in's words, dated
    assert [r["km"] for r in maya["runs"] if r["kind"] == "long"] == [14, 16, 17, 18]


def test_chapter4_writing():
    import stores
    import writing as W
    assert W.unsupported("Monday, Tuesday, Wednesday, Thursday, Friday", "I run five days a week") == \
        ["monday", "tuesday", "wednesday", "thursday", "friday"]                       # a made-up value
    record = "- race: Riverside Half Marathon, Sunday 18 October"
    assert W.unsupported("Riverside Half Marathon, Sunday 1 November",
                         "The Riverside Half has been moved to Sunday 1 November\n" + record) == []   # names on record
    assert W.unsupported("vegetarian", "I'm vegetarian") == [] and W.unsupported("knee cleared", "my knee is fine") == []
    assert W.unsupported("Riverside Half, Sunday 1 November", "moved to Sunday 1 November") == ["half"]
    assert [W.day_of(t, "2026-09-08") for t in ("Sunday's long run", "Tuesday", "Thursday", "yesterday",
                                                "26 September", "3 days ago", "a while back")] == \
        ["2026-09-06", "2026-09-01", "2026-09-03", "2026-09-07", "2025-09-26", "2026-09-05", None]
    root = Path(tempfile.mkdtemp())
    p = stores.Profile(root / "p.json", {"diet": "what they eat"})
    p.write("diet", "vegetarian", "2026-09-02")
    p.write("diet", "eats meat", "2026-10-10")
    p.add_field("shoes", "which")
    again = stores.Profile(root / "p.json")                                             # it persists
    assert again.read("diet") == "eats meat" and set(again.fields()) == {"diet", "shoes"}
    assert again.lines() == ["- diet: eats meat (since 2026-10-10; before: vegetarian (since 2026-09-02))"]
    calls = []

    def extract(prompt, form):                       # first fills in a made-up value, then corrects it when told
        calls.append(prompt)
        days = "Monday to Friday" if len(calls) == 1 else "five days a week"
        return form.model_validate({"profile": {"days": days}, "new_fields": []})
    fields = {"days": "the days they run"}
    form = W.profile_form(fields)
    filled, still, sent_back = W.fill("x", form, lambda f: W.profile_problems(f, fields, "I run five days a week"),
                                      extract=extract)
    assert still == [] and filled.profile.days == "five days a week" and "not said: monday, friday" in sent_back[0]
    calls.clear()                                     # a problem marked "drop" is removed without a second call
    filled, still, sent_back = W.fill("x", form, lambda f: [{"key": ("run", 0), "what": "an old run", "why": "copied",
                                                             "drop": True}], extract=extract)
    assert len(calls) == 1 and sent_back == [] and still[0]["drop"]
    assert W.day_of("2026-08-30", "2026-09-15") == "2026-08-30"                       # a written date is a date
    tom = "the City Marathon on Sunday 8 November. I run five days a week"         # meaning, not spelling
    assert W.unsupported("City Marathon on 2026-11-08", tom) == [] and W.unsupported("Nov 8, 2026", tom) == []
    assert W.unsupported("5 days a week", tom) == [] and W.unsupported("City Marathon on 2026-11-09", tom) == ["2026-11-09"]
    assert W.unsupported("knee cleared on 22 September", "my knee is fine") == ["22 september"]


def test_round_four_offline():
    import writing as W
    assert S.minutes_of("1:38") == 98 and S.minutes_of("49:30") == 49.5 and S.minutes_of("") is None
    root = Path(tempfile.mkdtemp())
    with T.embeddings(), T.groq() as seen:
        done, ok = S.run(root=root, path=root / "s.json", log=lambda *_: None, write="card")
    assert ok
    tom = S.holdings("tom", root)
    assert tom["profile"]["diet"]["value"] == "vegetarian"                          # fuel written, diet untouched
    assert tom["profile"]["fuel"]["value"].startswith("gels") and tom["profile"]["fuel"]["was"].startswith("water")
    assert "pre_run_eating" in tom["fields"]                                        # a field added where none fit
    assert "pre_run_eating" in str(next(s for s in seen if s["session"] == "tom-3" and "new_fields:" in str(s["messages"])))
    maya = S.holdings("maya", root)
    assert [(r["on_day"], r["km"], r["minutes"]) for r in maya["runs"]] == [
        ("2026-08-30", 14, 86), ("2026-09-01", 6, None), ("2026-09-03", 6, None), ("2026-09-06", 16, 98),
        ("2026-09-13", 17, 105), ("2026-09-20", 18, 111), ("2026-09-26", 10, 49.5)]          # dates worked out in code
    assert not any(n["source"] == "runner said" for n in maya["notes"])               # no whole messages
    # the stand-in dates the knee "on 22 September", which nothing says: sent back once, then refused
    step = done["maya-4"]["kept"][0]
    assert step["sent back"] and step["refused"][0]["why"] == "not said: 22 september"
    assert W.day_of("Sunday's long run", "2026-09-15") == "2026-09-13"


def test_chapter4_profile_history_and_retired_notes():
    import stores
    root = Path(tempfile.mkdtemp())
    p = stores.Profile(root / "p.json")
    assert p.write("race", "Riverside Half, 18 October", "2026-09-01")
    assert p.write("race", "Riverside Half, 1 November", "2026-09-15")
    assert not p.write("race", "riverside half -- 1 November", "2026-09-29")          # unchanged: ignored
    fact = stores.Profile(root / "p.json").facts["race"]
    assert fact["since"] == "2026-09-15" and fact["was"] == "Riverside Half, 18 October"
    assert fact["history"] == [{"value": "Riverside Half, 18 October", "since": "2026-09-01"}]
    with T.embeddings():
        j = stores.QdrantJournal(root / "j")
        ids = j.write([("2026-09-01", "maya", "fact", "Runs Tuesday, Thursday, Saturday and Sunday evenings",
                        ["days", "time_of_day"]),
                       ("2026-09-29", "maya", "habit", "Goes out too fast at race starts")])
        assert j.retire("days", "2026-09-22") == [ids[0]] and j.retire("days", "2026-09-23") == []
        assert [i for _, i, _ in j.find_like("which evenings does she run", k=5)] == [ids[1]]   # retired: not found
        assert [r["id"] for r in j.find_exact("Thursday")] == [] and len(j) == 2          # but still stored
        assert {r["id"]: r["retired"] for r in j.recent(limit=10, live=False)} == {ids[0]: True, ids[1]: False}


def test_round_five_offline():
    root = Path(tempfile.mkdtemp())
    with T.embeddings(), T.groq():
        done, ok = S.run(root=root, path=root / "s.json", log=lambda *_: None, write="card5")
    assert ok
    maya = S.holdings("maya", root)
    assert "longest_run" not in maya["fields"]                                         # the run log holds that
    assert any("records can work this out" in r["why"] for r in done["maya-1"]["kept"][0]["refused"])
    notes = {n["text"]: n for n in maya["notes"]}
    assert notes["Her long run is on Sundays, after the three shorter days"]["retired"]    # days changed 22 Sep
    assert notes["Goes out far too fast in the first kilometre at race starts"]["source"] == "habit"
    assert "Her long run is on Sundays" not in done["maya-6"]["shown"]
    assert [(r["on_day"], r["km"]) for r in maya["runs"] if r["kind"] == "long"] == [
        ("2026-08-30", 14), ("2026-09-06", 16), ("2026-09-13", 17), ("2026-09-20", 18)]


def test_a_stopped_run_carries_on():
    root = Path(tempfile.mkdtemp())
    with T.embeddings():
        with T.groq(fail_on="maya-3"):
            done, ok = S.run(root=root, path=root / "s.json", log=lambda *_: None)
        assert not ok and list(done) == ["maya-1", "tom-1", "maya-2"]
        with T.groq():
            done, ok = S.run(root=root, path=root / "s.json", log=lambda *_: None)
    assert ok and len(S.holdings("maya", root)["runs"]) == 7                   # nothing written twice
    m, _ = K.offline_memory(root=root / "m")
    m.llm.llm.fail_on = "maya-3"
    with T.groq():
        done, ok = M.run(m, path=root / "m.json", log=lambda *_: None)
    assert not ok and "reply" in done["maya-3"] and "added" not in done["maya-3"]
    m.llm.llm.fail_on = None
    with T.groq() as seen:
        done, ok = M.run(m, path=root / "m.json", log=lambda *_: None)
    assert ok and "added" in done["maya-3"]
    assert "maya-3" not in [s["session"] for s in seen]                       # its reply was not asked for again


def test_what_you_have_to_know():
    assert K.adds_only()["events in history"] == ["ADD"] and len(K.adds_only()["race memories held"]) == 2
    assert K.dates()["add(timestamp=...)"].startswith("The timestamp parameter is not supported")
    size = K.prompt_size()
    assert size["extraction prompt alone"] > 7000 and size["largest add request seen here"] > K.GROQ_REQUEST_LIMIT
    found = K.search()
    assert found["Chroma's keyword search"] is None and found["Qdrant, one result's parts"]["max_possible_score"] == 2.5
    scope = K.scoping()
    assert scope["memories stored for one user"] == 25 and scope["get_all with its default top_k"] == 20
    assert scope["search(user_id=...)"].startswith("Top-level entity parameters")
    assert K.links() == {"the prompt asks for linked_memory_ids": True, "add reads them from the model's reply": False}
    assert K.defaults()["Gemini embedder uses memory_action"] is False


if __name__ == "__main__":
    for test in (test_the_scenario, test_the_two_journals_agree, test_the_scratch_coach_offline,
                 test_the_mem0_coach_offline, test_round_two_offline, test_round_three_offline,
                 test_chapter4_writing, test_round_four_offline,
                 test_chapter4_profile_history_and_retired_notes, test_round_five_offline,
                 test_a_stopped_run_carries_on,
                 test_what_you_have_to_know):
        test()
        print("ok ", test.__name__)
