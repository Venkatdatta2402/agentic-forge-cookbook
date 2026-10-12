"""Two runners, nine check-ins over six weeks, and what a coach with a memory should know at each one.

Every check-in is one message from a runner and one reply. The messages are scripted, so both coaches -- Mem0's and
the one built from this repo's parts -- are told exactly the same things in the same order, and any difference in
what they remember comes from the memory.

What makes it hard is that the facts move. Maya's race is put back two weeks, she switches from evening runs to
mornings, a sore knee comes and goes, a new job cuts her from four running days to three, and a personal best
moves her goal. A memory that only ever adds will hold the old version of each of those beside the new one. A
second runner, Tom, is there to check that one runner's memories never reach the other.

    SESSIONS   the check-ins, in the order they happen
    CHECKS     for the replies that need memory: the facts they need, the old facts that would be wrong now,
               and (for Tom) Maya's facts that must not appear at all
    CURRENT    what each memory should hold as true by the end
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STORE = HERE / "store"
for _chapter in ("01_foundations", "02_agent_runtime", "03_tools", "04_memory", "06_orchestration"):
    if str(REPO / _chapter) not in sys.path:
        sys.path.append(str(REPO / _chapter))

COACH_MODEL = "openai/gpt-oss-120b"        # the coach, in both builds, on Groq
REPLY_TOKENS = 2000                        # gpt-oss's hidden reasoning counts against this, so it is set explicitly

RUNNERS = {"maya": "Maya", "tom": "Tom"}

# One persona for both coaches. Each build adds only what its memory supplies and how it writes.
COACH = """You are a running coach. You coach several runners through check-ins: each check-in is one message from a
runner, and you reply once. Make every reply specific to this runner -- their race, their goal, the days they run,
how their training has gone and anything that hurts. Keep replies short and practical: a few short paragraphs or a
short list. Today is {date}, and you are talking to {name}."""

SESSIONS = [
    {"id": "maya-1", "runner": "maya", "date": "2026-09-01", "message": (
        "Hi! I'm Maya. I've signed up for the Riverside Half Marathon on Sunday 18 October and I'd love to finish "
        "under 1:50. I run four days a week -- Tuesday, Thursday, Saturday, and a long run on Sunday -- always in the "
        "evening after work. My longest run so far is 14 km, last Sunday, in 1:26. One thing: my left knee gets sore "
        "after long downhill stretches. What should my long runs look like between now and the race?")},
    {"id": "tom-1", "runner": "tom", "date": "2026-09-02", "message": (
        "Hey coach, Tom here. I'm running my first marathon, the City Marathon on Sunday 8 November. I run five days "
        "a week and do my long run on Saturdays -- 24 km last Saturday, which felt fine. I'm vegetarian, and so far "
        "I've done my long runs on water alone. What should I be eating and drinking on long runs?")},
    {"id": "maya-2", "runner": "maya", "date": "2026-09-08", "message": (
        "Sunday's long run was 16 km in 1:38 -- felt good, and the knee was fine. Tuesday and Thursday were easy "
        "6 km runs. Should I add a tempo run, and if so, on which day?")},
    {"id": "maya-3", "runner": "maya", "date": "2026-09-15", "message": (
        "Two changes. The Riverside Half has been moved to Sunday 1 November because of roadworks on the course. "
        "And I've switched to running in the mornings before work -- evenings were getting too busy. Sunday's long "
        "run was 17 km in 1:45. My knee was sore after the hill repeats on Thursday, though. How many weeks do I "
        "have now, and should I change anything?")},
    {"id": "tom-2", "runner": "tom", "date": "2026-09-16", "message": (
        "Saturday was 28 km. I took gels like you suggested and it went much better than on water alone -- though "
        "I'd eaten breakfast only an hour before and got stomach cramps for the first 5 km, which always happens "
        "when I eat less than two hours before a run. My right hamstring has been tight since, too. Should I still "
        "do 30 km this weekend?")},
    {"id": "maya-4", "runner": "maya", "date": "2026-09-22", "message": (
        "Good news: I saw a physio and my knee is fine -- it was a tight IT band, and I've been cleared to run "
        "normally, hills included. Less good: I've started a new job, so I can only run three days a week now -- "
        "Tuesday, Friday and Sunday. Sunday's long run was 18 km in 1:51. Can you rework my week around the new "
        "days?")},
    {"id": "maya-5", "runner": "maya", "date": "2026-09-29", "message": (
        "I ran a 10k race on Saturday in 49:30 -- a personal best! I went out far too fast in the first kilometre, "
        "as I always seem to at race starts, and paid for it at the end. Because of that result I've decided to go "
        "for 1:45 at the half instead of 1:50. Is that realistic, and how should my next few weeks look?")},
    {"id": "tom-3", "runner": "tom", "date": "2026-09-30", "message": (
        "Quick one: what's my longest run been so far, and how many weeks until my race? I want to plan my taper. "
        "And what should I eat on race morning, and when?")},
    {"id": "maya-6", "runner": "maya", "date": "2026-10-13", "message": (
        "Final check before I start tapering. Can you remind me: when is my race, what's my goal time, which days "
        "do I run now and at what time of day, and how have my long runs built up so far? And any advice for race "
        "morning?")},
]

# Ways of writing the same fact, for finding it in a reply. These only point at what to read: every reply is read
# in full, and the reading is what the notebook reports.
_RACE_NOW = ["1 November", "November 1", "1 Nov", "Nov 1", "1st November", "November 1st", "1/11", "11/1"]
_RACE_BEFORE = ["18 October", "October 18", "18 Oct", "Oct 18", "18th October", "October 18th"]
_EVENINGS = ["evening", "after work"]
_MORNINGS = ["morning", "before work"]

CHECKS = {
    "maya-2": {"needs": {"her running days": ["Tuesday", "Thursday", "Saturday"],
                         "her goal of 1:50, or her race on 18 October": ["1:50", "sub-1:50"] + _RACE_BEFORE},
               "stale": {}},
    "maya-4": {"needs": {"the race on 1 November": _RACE_NOW + ["weeks"],
                         "running in the mornings": _MORNINGS},
               "stale": {"the old race date": _RACE_BEFORE, "evening runs": _EVENINGS}},
    "maya-5": {"needs": {"her three days: Tuesday, Friday, Sunday": ["Friday"],
                         "the race on 1 November": _RACE_NOW + ["weeks"]},
               "stale": {"the old race date": _RACE_BEFORE, "Thursday or Saturday runs": ["Thursday", "Saturday"]}},
    "tom-3": {"needs": {"his longest run, 28 km": ["28 km", "28km", "28-km", "28 kilometres", "28 kilometers"],
                        "his race on 8 November": ["8 November", "November 8", "8 Nov", "Nov 8", "weeks"],
                        # said in passing on 16 September, in a message about something else
                        "eating at least two hours before, because of his cramps": [
                            "cramp", "cramps", "two hours", "2 hours", "two-hour", "2-hour", "2 h", "2h",
                            "three hours", "3 hours", "3-hour", "three-hour"]},
              "stale": {},
              "not his": {"anything of Maya's": ["Maya", "Riverside", "knee", "IT band", "1:45", "1:50"]}},
    "maya-6": {"needs": {"the race on 1 November": _RACE_NOW,
                         "the goal of 1:45": ["1:45"],
                         "Tuesday, Friday and Sunday": ["Friday"],
                         "mornings": _MORNINGS,
                         "the long runs building up": ["14", "16", "17", "18"],
                         "not going out too fast at the start": ["too fast", "too quick", "start", "first kilometre",
                                                                 "first km", "conservative"]},
               "stale": {"the old race date": _RACE_BEFORE, "the old goal of 1:50": ["1:50"],
                         "Thursday or Saturday runs": ["Thursday", "Saturday"], "evening runs": _EVENINGS,
                         "the knee as a current problem": ["knee"]}},
}

# What each memory should hold as true at the end -- and what used to be true and no longer is
CURRENT = {
    "maya": {"race": "Riverside Half Marathon, Sunday 1 November (moved from 18 October)",
             "goal": "1:45 (was 1:50)",
             "days": "Tuesday, Friday and Sunday (was Tuesday, Thursday, Saturday, Sunday)",
             "time of day": "mornings (was evenings)",
             "knee": "cleared by a physio on 22 September -- a tight IT band (was sore after downhills and hills)",
             "long runs": "14, 16, 17, 18 km",
             "race starts": "goes out too fast"},
    "tom": {"race": "City Marathon, Sunday 8 November", "diet": "vegetarian", "long runs": "24, 28 km, on Saturdays",
            "fuel": "gels worked better than water alone", "hamstring": "right hamstring tight since the 28 km",
            "stomach": "cramps if he eats less than two hours before a run"},
}


def found(text, spellings):
    """Which of the spellings appear in the text, as whole words (case ignored)."""
    return [s for s in spellings if re.search(rf"(?<![\w:]){re.escape(s)}(?![\w:])", text, flags=re.IGNORECASE)]


def leads(session_id, reply):
    """What a reply mentions of the facts its check lists: places to look when reading it, not a score."""
    check = CHECKS[session_id]
    out = {"needs, mentioned": [], "needs, not mentioned": [], "old facts mentioned": [], "other runner's": []}
    for fact, spellings in check["needs"].items():
        out["needs, mentioned" if found(reply, spellings) else "needs, not mentioned"].append(fact)
    for fact, spellings in check["stale"].items():
        if found(reply, spellings):
            out["old facts mentioned"].append(f"{fact}: {', '.join(found(reply, spellings))}")
    for fact, spellings in check.get("not his", {}).items():
        if found(reply, spellings):
            out["other runner's"].append(f"{fact}: {', '.join(found(reply, spellings))}")
    return out
