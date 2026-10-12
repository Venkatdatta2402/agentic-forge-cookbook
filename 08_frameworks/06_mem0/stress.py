"""Retrieval with a year behind it: the same two last check-ins, after 120 older memories per runner.

In the nine check-ins each runner has 8-27 memories, and Mem0's search returns up to 20, so it returns nearly
everything and cannot miss; the scratch coach is shown its whole profile and run log. Retrieval was hardly tested.
Here each build's best store is copied (Mem0 round 2, scratch round 7) and given a year of earlier check-ins --
old runs and races, old goals and schedules, old niggles, preferences -- dated August 2025 to July 2026, before
anything in the scenario. Then the two check-ins whose stores are complete as of their date (Maya's 13 October, Tom's
30 September) search again, and each fact `answer_key.NEEDED` lists is looked for in what the coach would be shown.

No model is called. Mem0's memories go in with `infer=False` (stored as written, one embedding each) and their names
are linked as Mem0 links them, so they compete in all three of its scores; the scratch build gets the same texts as
runs in its run log and notes in its journal.
"""

import json
import random
import re
import shutil
from datetime import date, timedelta

import coach_mem0 as M
import coach_scratch as S
import runners as R

ROOT = R.STORE / "stress"
RESULTS = R.STORE / "answers" / "stress.json"
SESSIONS = {s["id"]: s for s in R.SESSIONS}
CHECKINS = ("maya-6", "tom-3")

# How each needed fact can be recognised in a context, for counting -- every context is also printed for reading
MARKERS = {
    "race on 1 November": [r"1 november", r"november 1\b", r"2026-11-01"],
    "goal 1:45": [r"1:45(?!\s*on)", r"1 hour 45"],
    "runs Tue, Fri, Sun": [r"tuesday, friday,? (and )?sunday"],
    "runs in the mornings": [r"\bmornings?\b"],
    "14 km on 30 August": [r"14 km.*(30 august|august 30|2026-08-30)|(2026-08-30).*14 km"],
    "16 km on 6 September": [r"16 km.*(6 september|september 6|2026-09-06)|(2026-09-06).*16 km"],
    "17 km on 13 September": [r"17 km.*(13 september|september 13|2026-09-13)|(2026-09-13).*17 km"],
    "18 km on 20 September": [r"18 km.*(20 september|september 20|2026-09-20)|(2026-09-20).*18 km"],
    "starts races too fast": [r"too fast"],
    "longest run, 28 km": [r"28 km"],
    "race on 8 November": [r"8 november", r"november 8\b", r"2026-11-08"],
    "vegetarian": [r"vegetarian"],
    "gels": [r"\bgels?\b"],
    "cramps when he eats under 2 hours before": [r"cramps.*(two hours|2 hours)|(two hours|2 hours).*cramps"],
}

_FELT = ["felt strong", "legs heavy", "windy on the river path", "hot and humid", "easy and relaxed",
         "tired from work", "rain the whole way", "good rhythm", "cold start, warmed up", "a bit flat"]


def _days(start="2025-08-03", end="2026-07-26"):
    day, last = date.fromisoformat(start), date.fromisoformat(end)
    while day <= last:
        yield day
        day += timedelta(days=7)


def _say(day):
    return f"{day:%A}, {day.day} {day:%B} {day.year}"


def filler(runner, seed=7):
    """About 120 earlier memories for one runner: ~60 runs and races, ~60 notes, none from the scenario's weeks."""
    rng, name = random.Random(f"{runner}-{seed}"), R.RUNNERS[runner]
    he = {"maya": "she", "tom": "he"}[runner]
    out = []
    for day in _days():
        for kind, km, pace in rng.sample([("long", rng.choice([12, 13, 15, 19, 21]), 6.3),
                                          ("easy", rng.choice([5, 7, 8, 9]), 6.6),
                                          ("tempo", rng.choice([6, 8, 10]), 5.4)], 1):
            minutes = round(km * pace + rng.uniform(-4, 4))
            felt = rng.choice(_FELT)
            out.append({"on": day.isoformat(), "kind": kind, "km": km, "minutes": minutes, "felt": felt,
                        "text": f"{name} ran a {kind} {km} km on {_say(day)} in {minutes} minutes; {felt}"})
    for when, km, race, time in ({"maya": [("2025-10-12", 10, "Harbour 10k", "53:40"), ("2026-03-15", 21.1,
                                            "Spring Half Marathon", "2:01:30"), ("2026-05-17", 10, "Lakeside 10k",
                                            "51:05")],
                                  "tom": [("2025-11-09", 21.1, "Hilltop Half Marathon", "1:58:20"), ("2026-04-19",
                                           21.1, "River Half Marathon", "1:52:10"), ("2026-06-14", 10, "Park 10k",
                                           "46:30")]}[runner]):
        out.append({"on": when, "kind": "race", "km": km, "minutes": None, "felt": f"{race}, {time}",
                    "text": f"{name} ran the {race} on {_say(date.fromisoformat(when))} in {time}"})
    notes = {
        "maya": ["In the 2025-26 winter {n} ran on Monday, Wednesday and Saturday, in the evenings",
                 "{n}'s goal for the Spring Half Marathon was to finish under two hours",
                 "{n} had a sore right Achilles in February 2026 that settled after a week off",
                 "{n} prefers flat routes along the river to hilly ones",
                 "{n} runs in Brooks Ghost shoes and replaced her pair in May 2026",
                 "{n} struggles to run well in heat above 25 degrees",
                 "Before the Lakeside 10k {n} had porridge two hours before the start",
                 "{n} ran even splits at the Lakeside 10k and finished strongly",
                 "{n} sometimes runs with her sister Priya on Saturdays",
                 "{n} does a yoga class on Wednesday evenings",
                 "{n} uses a running watch to keep her easy runs easy",
                 "{n} found that eating a banana an hour before a run sits fine with her",
                 "{n}'s left knee was fine through the whole of the spring",
                 "{n} likes a podcast on her easy runs and music on hard ones",
                 "{n} took two weeks off in December 2025 with a chest cold"],
        "tom": ["Last winter {n} did his long runs on Sundays with a club",
                "{n}'s goal for the River Half Marathon was to run under 1:55",
                "{n} had a side stitch at 15 km on a long run in June 2026",
                "{n} tried a sports drink in April 2026 and found it too sweet",
                "{n} drinks a coffee before every morning run",
                "{n} runs with the Riverside Striders club on Thursday evenings",
                "{n}'s previous shoes were Saucony and he moved to Hoka in 2026",
                "{n} likes to do strides after easy runs",
                "{n} prefers to run his long runs on trails when he can",
                "{n} had a sore left calf in January 2026 for about ten days",
                "{n} eats a large pasta dinner the night before a race",
                "{n} cycles to work twice a week",
                "{n} once ran a half marathon on no breakfast and felt fine",
                "{n} takes a rest day every Monday",
                "{n} finds he sleeps badly before races"]}[runner]
    for i, text in enumerate(notes * 4):              # each repeated with a different date and a different detail
        day = date(2025, 8, 3) + timedelta(days=(i * 23) % 360)
        detail = rng.choice(["", " (noted at a check-in)", f", {he} said", " -- mentioned in passing",
                             f", {he} told the coach", ""])
        out.append({"on": day.isoformat(), "kind": "note", "text": text.format(n=name) + detail})
    unique = {o["text"]: o for o in out}
    return list(unique.values())


def fresh_copies(root=ROOT):
    """Copies of the two best stores, so the recorded rounds stay as they were."""
    for source, name in ((M.ROUND2, "mem0"), (S.ROUND7, "scratch")):
        target = root / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def fill_mem0(m, log=print):
    for runner in R.RUNNERS:
        items = filler(runner)
        for item in items:
            got = m.add(item["text"], user_id=runner, metadata={"session": item["on"]}, infer=False)["results"]
            for row in got:                           # link its names, as Mem0 does for what its model extracts
                m._link_entities_for_memory(row["id"], item["text"], {"user_id": runner})
        log(f"  mem0: {runner} +{len(items)} memories")


def fill_scratch(root=ROOT, log=print):
    for runner in R.RUNNERS:
        memory = S.Profiled5(runner, root / "scratch")
        items = filler(runner)
        runs = [S.Run(on=i["on"], kind=i["kind"] if i["kind"] != "race" else "race", km=i["km"],
                      minutes=i["minutes"], felt=i["felt"]) for i in items if i["kind"] != "note"]
        for run in runs:
            memory.runs.write(run)
        known = {row["text"] for row in memory.notes.recent(limit=10000, live=False)}
        notes = [i for i in items if i["kind"] == "note" and i["text"] not in known]     # a resumed fill adds none twice
        if notes:
            memory.notes.write([(n["on"], runner, "fact", n["text"]) for n in notes], now=notes[-1]["on"])
        log(f"  scratch: {runner} +{len(runs)} runs, +{len(notes)} notes")


def found(context, what):
    text = context.lower()
    return any(re.search(p, text) for p in MARKERS[what])


def score(contexts):
    """Per check-in: each needed fact found in the context or not."""
    import answer_key as K
    return {sid: {what: found(contexts[sid], what) for what, _ in K.NEEDED[sid]} for sid in CHECKINS}


def run(root=ROOT, results=RESULTS, embedder=None, log=print):
    """Copy, fill, search again, score, save. Returns the contexts and the scores for both builds."""
    fresh_copies(root)
    m = M.memory(root / "mem0", embedder=embedder)
    fill_mem0(m, log)
    fill_scratch(root, log)
    return measure(m, root, results)


def measure(m, root=ROOT, results=RESULTS):
    """The two check-ins' searches on the filled copies, scored and saved. Separate from filling, so a fill that
    stops on a network error can be finished and measured without filling again."""
    out = {"memories per runner": {r: len(filler(r)) for r in R.RUNNERS}, "mem0": {}, "scratch": {},
           "scratch, notes by share": {}, "scratch, hybrid per question + habits": {}}
    for sid in CHECKINS:
        session = SESSIONS[sid]
        recalled = M.recall(m, session)
        out["mem0"][sid] = {"context": "\n".join(f"({r['session']}) {r['memory']}" for r in recalled),
                            "returned": len(recalled),
                            "older memories among them": sum(r["session"] < "2026-08-01" for r in recalled)}
        memory = S.Profiled5(session["runner"], root / "scratch")
        out["scratch"][sid] = {"context": memory.render(session["date"], session["message"])}
        out["scratch, notes by share"][sid] = {"context": memory.render(
            session["date"], session["message"], share=S.NOTES_SHARE, cap=S.NOTES_CAP)}
        out["scratch, hybrid per question + habits"][sid] = {"context": memory.render(
            session["date"], session["message"], share=S.NOTES_SHARE, cap=S.NOTES_CAP, search="questions")}
    for build in ("mem0", "scratch", "scratch, notes by share", "scratch, hybrid per question + habits"):
        marks = score({sid: out[build][sid]["context"] for sid in CHECKINS})
        for sid in CHECKINS:
            out[build][sid]["needed"] = marks[sid]
    out["embedded"] = m.embedding_model.embedded
    results.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
