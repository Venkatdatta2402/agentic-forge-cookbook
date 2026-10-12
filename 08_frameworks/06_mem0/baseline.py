"""The six runs so far, marked by hand against `answer_key` -- before any change to either build.

Three measures, kept apart, because a system can save a fact right and later show it wrong, or keep an old fact
without ever presenting it as current:

    EXTRACTION   per fact, what that check-in's write kept: right, wrong, or not saved. Each check-in is marked
                 only on its own message -- a later correction does not mend an earlier write. Anything not listed
                 for a run was saved right. "Saved" means what was KEPT: a scratch change the check refused counts
                 as not saved. Both halves of a two-part fact must be right.
    INVENTED     facts saved as the runner's that the runner never said
    CURRENT      at the two check-ins that ask about the present, each state asked for: right in the reply or not
    STALE        at the check-ins after a change was told, each item in the coach's context that states a value
                 already replaced as if it were current. Not counted: a change stated as a change ("moved from 18
                 October to..."), a value labelled "before:", a dated one-off event ("knee soreness after hill
                 repeats on 10 September"). Counted even with a date beside it: "(2026-09-01) signed up for the
                 race on 18 October" is a fact, and a dated fact reads as still true
    ADVICE       memories that are the coach's own advice (Mem0 round 1), counted apart: not wrong, but not the
                 runner's either
    RETRIEVAL    at the five check-ins that need memory, each fact `answer_key.NEEDED` lists: did it reach the
                 coach's context in its current version? Listed here only when it did not, labelled by why:
                   write      never saved right (missing or wrong in EXTRACTION) -- retrieval had nothing to find
                   retrieval  saved right, but not in the context
                 Found in the context in any form counts -- a kept message that says it, say -- because that is
                 what the coach had in front of it
"""

import answer_key as K

RUNS = ["mem0 r1", "mem0 r2", "scratch r1", "scratch r2", "scratch r3", "scratch r4", "scratch r5", "scratch r6", "scratch r7"]

# what each run got wrong or did not save; every fact not listed was saved right
EXTRACTION = {
    "mem0 r1": {},
    "mem0 r2": {
        "M16": "wrong: the easy runs dated 8 and 10 September",
        "M34": "wrong: kept as a one-off in the 10k, not as a habit",
    },
    "scratch r1": {
        "M05": "missing", "M08": "wrong: dated 27 August", "M09": "missing",
        "T02": "missing", "T03": "missing", "T05": "missing", "T07": "wrong: dated 27 August", "T08": "missing",
        "M12": "wrong: dated 3 September", "M13": "missing", "M14": "missing: 'felt good' not kept",
        "M16": "wrong: dated 5 and 7 September",
        "M17": "missing: the move was never written", "M20": "wrong: dated 12 September", "M21": "missing",
        "T12": "wrong: dated 14 September", "T13": "missing",
        "M26": "missing", "M28": "missing",
        "M31": "wrong: dated 28 September", "M32": "missing", "M34": "missing",
    },
    "scratch r2": {
        "M05": "missing", "M08": "wrong: dated 29 August",
        "T02": "missing", "T04": "wrong: saved as Monday to Friday", "T05": "missing", "T07": "wrong: dated 27 August",
        "M12": "wrong: dated 4 September", "M16": "wrong: dated 8 and 3 September",
        "M22": "missing: 'knee sore', without the hill repeats",
        "T14": "wrong: kept as a one-off on 12 September, not as a habit",
        "M24": "missing: 'hills included' not kept",
        "M34": "wrong: kept as a one-off in the 10k, not as a habit",
    },
    "scratch r3": {
        "M05": "missing", "M08": "wrong: dated 28 August",
        "T02": "missing", "T05": "missing", "T07": "wrong: dated 27 August",
        "T09": "missing: refused (quote one word off)", "T10": "missing: refused (quote one word off)",
        "M12": "wrong: dated 5 September", "M16": "wrong: dated 8 and 3 September",
        "T14": "missing: 'cramps for first 5 km', without the cause",
        "M28": "wrong: dated 19 September", "M33": "missing",
    },
    "scratch r4": {
        "M05": "missing", "T02": "missing: sent back for a year it added, and dropped",
        "T14": "wrong: kept as a one-off (breakfast an hour before), not as a habit",
        "M24": "missing: 'hills included' not kept", "M34": "missing",
    },
    "scratch r5": {
        "M05": "missing", "T02": "missing", "T05": "missing",
        "M23": "missing: 'a tight IT band' not kept", "M26": "missing",
        "M33": "missing: refused as a field the records could work out, and not kept on the run",
    },
    "scratch r6": {
        "M05": "missing",
        "T01": "missing: refused -- written '2026-11-08', which the check did not match to '8 November'",
        "T02": "missing", "T03": "missing: refused with T01",
        "T04": "missing: refused -- written '5 days a week', which the check did not match to 'five'",
        "T05": "missing", "T13": "missing: gels never written; fuel still says water alone",
        "M24": "missing: 'hills included' not kept", "M26": "missing", "M33": "missing",
    },
    "scratch r7": {"M05": "missing", "M26": "missing", "M33": "missing"},
}

INVENTED = {
    "mem0 r1": ["Tom's 'training schedule includes upcoming long runs of 28 km on 9 September and 30 km on "
                "16 September' -- the coach's plan, stored as Tom's"],
    "mem0 r2": [], "scratch r1": [], "scratch r3": [], "scratch r4": [],
    "scratch r2": ["Tom's days as Monday, Tuesday, Wednesday, Thursday, Friday"],
    "scratch r6": [], "scratch r7": [],
    "scratch r5": ["Maya's knee soreness after Thursday's hill repeats saved as a habit ('on Thursdays'); it "
                   "happened once"],
}

ADVICE = {"mem0 r1": 14, "mem0 r2": 0}

# Groq tokens per round, all calls (the coach, and the write step with its second tries)
TOKENS = {"scratch r4": 40471, "scratch r5": 64818, "scratch r6": 41351, "scratch r7": 43879}

# the states each asking check-in asks about: True if the reply had the value that was true that day
CURRENT = {
    "mem0 r1": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True,
                           "long runs": False},          # a 20 km on 27 September she never ran; 14 km left out
                "tom-3": {"longest run": True, "race date": True, "diet": True, "fuel": True, "cramps": True}},
    "mem0 r2": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True, "long runs": True},
                "tom-3": {"longest run": True, "race date": True, "diet": True, "fuel": True, "cramps": True}},
    "scratch r1": {"maya-6": {"race date": False, "goal": True, "days": False, "time of day": True,
                              "long runs": False},       # dates wrong
                   "tom-3": {"longest run": True, "race date": False,     # "the second Sunday of October"
                             "diet": True, "fuel": False, "cramps": True}},   # fuel never known
    "scratch r2": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True,
                              "long runs": False},       # easy runs and the 10k listed as long runs; dates wrong
                   "tom-3": {"longest run": True, "race date": True, "diet": True,
                             "fuel": False, "cramps": True}},             # "you've done well with water only"
    "scratch r3": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True,
                              "long runs": False},       # dates a day or two off
                   "tom-3": {"longest run": True, "race date": True, "diet": True, "fuel": True, "cramps": True}},
    "scratch r4": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True, "long runs": True},
                   "tom-3": {"longest run": True, "race date": True, "diet": True, "fuel": True,
                             "cramps": False}},          # a snack 30-60 minutes before; cramps not reflected
    "scratch r5": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True, "long runs": True},
                   "tom-3": {"longest run": True, "race date": True, "diet": True,
                             "fuel": False,              # a gel before the start, but "stick to water only on the run"
                             "cramps": True}},           # ties race-morning food to cramps "within 2 h of the start"
    "scratch r6": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True, "long runs": True},
                   "tom-3": {"longest run": True,
                             "race date": False,         # never stored (refused); the coach made up "13 October"
                             "diet": True, "fuel": False,  # "water-only fueling during runs"
                             "cramps": True}},           # "respecting your 2-hour gut-comfort window"
    "scratch r7": {"maya-6": {"race date": True, "goal": True, "days": True, "time of day": True, "long runs": True},
                   "tom-3": {"longest run": True, "race date": True, "diet": True, "fuel": True,
                             "cramps": True}},           # "eat at least 2 hours before the start"
}

# the weeks to Tom's race (5 weeks 4 days), as each coach counted them from a date it had right -- arithmetic
WEEKS = {"mem0 r1": "5 weeks 3 days", "mem0 r2": "5 weeks 2 days", "scratch r1": "no date known",
         "scratch r2": "5 weeks 2 days", "scratch r3": "39 days (right)", "scratch r4": "5 weeks 4 days (right)",
         "scratch r5": "5 weeks 4 days (right)", "scratch r6": "no date known (it made up 13 October)",
         "scratch r7": "39 days, about 5 and a half weeks (right)"}

# per check-in: (items stating a replaced value as current, items in the context, what they were)
STALE = {
    "mem0 r1": {"maya-4": (3, 18, ["four days, in the evening", "a long-run plan to the 18 October race",
                                   "signed up for the race on 18 October"]),
                "maya-5": (7, 20, ["a long-run plan to the 18 October race", "a plan with Thursday and Saturday runs",
                                   "four days, in the evening", "a Saturday recovery run",
                                   "signed up for the race on 18 October", "keep Thursday easy",
                                   "trim Thursday's hills"]),
                "maya-6": (6, 20, ["a long-run plan to the 18 October race", "four days, in the evening",
                                   "signed up for 18 October, under 1:50", "a plan with Thursday and Saturday runs",
                                   "a Saturday recovery run", "keep Thursday easy"]),
                "tom-3": (0, 11, [])},
    "mem0 r2": {"maya-4": (2, 10, ["four days, in the evening", "signed up for the race on 18 October"]),
                "maya-5": (3, 13, ["four days, in the evening", "signed up for the race on 18 October",
                                   "knee sore after long downhills (cleared on 22 September)"]),
                "maya-6": (3, 15, ["signed up for 18 October, under 1:50", "four days, in the evening",
                                   "knee sore after long downhills"]),
                "tom-3": (0, 8, [])},
    "scratch r1": {"maya-4": (1, 12, ["note: race on 2026-10-18"]),
                   "maya-5": (1, 12, ["days: Tue, Thu, Sat, Sun"]),
                   "maya-6": (2, 14, ["days: Tue, Thu, Sat, Sun", "note: race on 2026-10-18"]),
                   "tom-3": (1, 8, ["note: currently fueling long runs with water only"])},
    "scratch r2": {"maya-4": (0, 11, []), "maya-5": (0, 13, []), "maya-6": (0, 14, []),
                   "tom-3": (1, 7, ["food: drinks water only on long runs"])},
    "scratch r3": {"maya-4": (1, 13, ["her 1 September message: 18 October, evenings"]),
                   "maya-5": (1, 14, ["her 1 September message"]),
                   "maya-6": (1, 15, ["her 1 September message"]),
                   "tom-3": (1, 9, ["his 2 September message: long runs on water alone"])},
    "scratch r4": {"maya-4": (1, 14, ["note: runs Tue, Thu, Sat and Sun evenings"]),
                   "maya-5": (0, 16, []),
                   "maya-6": (1, 18, ["note: runs Tue, Thu, Sat and Sun evenings"]),
                   "tom-3": (0, 11, [])},
    "scratch r5": {"maya-4": (0, 11, []), "maya-5": (0, 12, []), "maya-6": (0, 13, []),
                   "tom-3": (2, 10, ["fuel: water alone (gels went into a note, not the field)",
                                     "note: only drinks water on long runs"])},
    "scratch r6": {"maya-4": (0, 10, []), "maya-5": (0, 11, []), "maya-6": (0, 13, []),
                   "tom-3": (1, 6, ["fuel: water alone (gels never written)"])},
    "scratch r7": {"maya-4": (0, 10, []), "maya-5": (0, 11, []), "maya-6": (0, 13, []), "tom-3": (0, 10, [])},
}


def table():
    """One row per run: the three measures, totalled."""
    rows = []
    total = len(K.FACTS)
    for run in RUNS:
        marks = EXTRACTION[run]
        wrong = sum(v.startswith("wrong") for v in marks.values())
        missing = sum(v.startswith("missing") for v in marks.values())
        asked = [ok for checks in CURRENT[run].values() for ok in checks.values()]
        stale = sum(s for s, _, _ in STALE[run].values())
        shown = sum(n for _, n, _ in STALE[run].values())
        rows.append({"run": run, "saved right": f"{total - wrong - missing}/{total}", "saved wrong": wrong,
                     "not saved": missing, "invented": len(INVENTED[run]),
                     "current state right": f"{sum(asked)}/{len(asked)}",
                     "stale items shown": f"{stale}/{shown} ({stale / shown:.0%})"})
    return rows


def check():
    ids = {f[0] for f in K.FACTS}
    for run in RUNS:
        assert set(EXTRACTION[run]) <= ids, run
        assert set(CURRENT[run]) == set(K.ASKED) and all(set(CURRENT[run][c]) == set(K.ASKED[c]) for c in K.ASKED)
    return True


# what did not reach the coach, per run and check-in: {needed: "write" | "retrieval"}
RETRIEVAL = {
    "mem0 r1": {},
    "mem0 r2": {"maya-6": {"starts races too fast": "write"}},          # kept as a one-off in the 10k
    "scratch r1": {"maya-2": {"long run on Sunday": "write",
                              "race on 18 October": "retrieval"},      # in a note Recall did not bring
                   "maya-4": {"race on 1 November": "write"}, "maya-5": {"race on 1 November": "write"},
                   "tom-3": {"race on 8 November": "write", "gels": "write"},
                   "maya-6": {"race on 1 November": "write", "14 km on 30 August": "write",
                              "16 km on 6 September": "write", "17 km on 13 September": "write",
                              "18 km on 20 September": "write", "starts races too fast": "write"}},
    "scratch r2": {"maya-2": {"long run on Sunday": "write"},
                   "tom-3": {"cramps when he eats under 2 hours before": "write"},
                   "maya-6": {"14 km on 30 August": "write", "16 km on 6 September": "write",
                              "starts races too fast": "write"}},
    "scratch r3": {"maya-6": {"14 km on 30 August": "write", "16 km on 6 September": "write",
                              "18 km on 20 September": "write",
                              "starts races too fast": "retrieval"}},  # three old messages took Recall's places
    "scratch r4": {"maya-2": {"long run on Sunday": "write"},
                   "tom-3": {"cramps when he eats under 2 hours before": "write"},
                   "maya-6": {"starts races too fast": "write"}},
    "scratch r5": {"maya-2": {"long run on Sunday": "write"}},
    "scratch r6": {"maya-2": {"long run on Sunday": "write"},
                   "tom-3": {"race on 8 November": "write", "gels": "write"}},
    "scratch r7": {"maya-2": {"long run on Sunday": "write"}},
}


def retrieval_table():
    """Per run: needed facts that reached the coach, and the misses by cause."""
    total = sum(len(items) for items in K.NEEDED.values())
    rows = []
    for run in RUNS:
        misses = [why for per in RETRIEVAL[run].values() for why in per.values()]
        rows.append({"run": run, "reached the coach": f"{total - len(misses)}/{total}",
                     "never saved (write)": misses.count("write"),
                     "saved, not retrieved (retrieval)": misses.count("retrieval")})
    return rows


def check_retrieval():
    needed = {s: {w for w, _ in items} for s, items in K.NEEDED.items()}
    for run in RUNS:
        for session, per in RETRIEVAL[run].items():
            assert set(per) <= needed[session], (run, session)
            for what, why in per.items():
                ids = dict(K.NEEDED[session])[what]
                saved_right = not any(fid in EXTRACTION[run] for fid in ids)
                # write failures must be facts marked missing or wrong; retrieval failures, facts saved right
                assert (why == "retrieval") == saved_right, (run, session, what, why)
    return True
