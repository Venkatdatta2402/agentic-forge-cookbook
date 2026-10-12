"""The answer key: every important fact in the nine check-ins, written by hand from the messages -- never shown to
either memory system. It is what their writes are scored against afterwards.

    FACTS     one entry per fact that can be right or wrong on its own: a run's distance, its date and its time are
              three facts, so one wrong date does not hide two right ones. Each carries the exact words of the
              message it comes from, and `check()` confirms those words are there -- the key is checked against the
              transcript, not against anyone's memory of it.
              kind:  state   true from one date until another fact replaces it (race date, goal, days)
                     event   something that happened, once (a run)
                     habit   something that keeps happening (cramps when eating late, starting races too fast)
    STATES    for each state, its values and the dates each was true -- what "current" and "stale" mean on a day
    ASKED     the check-ins that ask about the current state, and which states each asks about. Nothing else is
              scored for current-state accuracy: a reply is not marked down for not mentioning what it was not
              asked
"""

import runners as R

MESSAGES = {s["id"]: s["message"] for s in R.SESSIONS}

FACTS = [
    # Maya, 1 September
    ("M01", "maya-1", "state", "race", "the race is the Riverside Half Marathon", "Riverside Half Marathon"),
    ("M02", "maya-1", "state", "race date", "the race is on Sunday 18 October", "Sunday 18 October"),
    ("M03", "maya-1", "state", "goal", "goal: under 1:50", "under 1:50"),
    ("M04", "maya-1", "state", "days", "runs four days: Tuesday, Thursday, Saturday, Sunday",
     "Tuesday, Thursday, Saturday, and a long run on Sunday"),
    ("M05", "maya-1", "state", "long run day", "the long run is on Sunday", "a long run on Sunday"),
    ("M06", "maya-1", "state", "time of day", "runs in the evening, after work", "always in the evening after work"),
    ("M07", "maya-1", "event", "run 1", "a 14 km run", "My longest run so far is 14 km"),
    ("M08", "maya-1", "event", "run 1", "...on Sunday 30 August", "last Sunday"),
    ("M09", "maya-1", "event", "run 1", "...in 1:26", "in 1:26"),
    ("M10", "maya-1", "state", "knee", "left knee gets sore after long downhill stretches",
     "my left knee gets sore after long downhill stretches"),
    # Tom, 2 September
    ("T01", "tom-1", "state", "race", "the race is the City Marathon", "the City Marathon"),
    ("T02", "tom-1", "state", "first marathon", "it is his first marathon", "my first marathon"),
    ("T03", "tom-1", "state", "race date", "the race is on Sunday 8 November", "Sunday 8 November"),
    ("T04", "tom-1", "state", "days", "runs five days a week", "I run five days a week"),
    ("T05", "tom-1", "state", "long run day", "long runs on Saturdays", "do my long run on Saturdays"),
    ("T06", "tom-1", "event", "run 1", "a 24 km run", "24 km last Saturday"),
    ("T07", "tom-1", "event", "run 1", "...on Saturday 29 August", "last Saturday"),
    ("T08", "tom-1", "event", "run 1", "...which felt fine", "which felt fine"),
    ("T09", "tom-1", "state", "diet", "vegetarian", "I'm vegetarian"),
    ("T10", "tom-1", "state", "fuel", "long runs on water alone so far", "done my long runs on water alone"),
    # Maya, 8 September
    ("M11", "maya-2", "event", "run 2", "a 16 km long run", "Sunday's long run was 16 km"),
    ("M12", "maya-2", "event", "run 2", "...on Sunday 6 September", "Sunday's long run"),
    ("M13", "maya-2", "event", "run 2", "...in 1:38", "in 1:38"),
    ("M14", "maya-2", "event", "run 2", "...felt good, knee fine", "felt good, and the knee was fine"),
    ("M15", "maya-2", "event", "easy runs", "two easy 6 km runs", "were easy 6 km runs"),
    ("M16", "maya-2", "event", "easy runs", "...on Tuesday 1 and Thursday 3 September", "Tuesday and Thursday"),
    # Maya, 15 September
    ("M17", "maya-3", "state", "race date", "the race moved to Sunday 1 November", "moved to Sunday 1 November"),
    ("M18", "maya-3", "state", "time of day", "now runs in the mornings, before work",
     "switched to running in the mornings before work"),
    ("M19", "maya-3", "event", "run 3", "a 17 km long run", "Sunday's long\nrun was 17 km"),
    ("M20", "maya-3", "event", "run 3", "...on Sunday 13 September", "Sunday's long"),
    ("M21", "maya-3", "event", "run 3", "...in 1:45", "in 1:45"),
    ("M22", "maya-3", "state", "knee", "knee sore after hill repeats on Thursday 10 September",
     "My knee was sore after the hill repeats on Thursday"),
    # Tom, 16 September
    ("T11", "tom-2", "event", "run 2", "a 28 km run", "Saturday was 28 km"),
    ("T12", "tom-2", "event", "run 2", "...on Saturday 12 September", "Saturday was"),
    ("T13", "tom-2", "state", "fuel", "gels on long runs, much better than water alone",
     "I took gels like you suggested and it went much better than on water alone"),
    ("T14", "tom-2", "habit", "cramps", "stomach cramps whenever he eats less than two hours before a run",
     "which always happens\nwhen I eat less than two hours before a run"),
    ("T15", "tom-2", "state", "injury", "right hamstring tight since that run", "My right hamstring has been tight"),
    # Maya, 22 September
    ("M23", "maya-4", "state", "knee", "knee cleared by a physio: a tight IT band", "it was a tight IT band"),
    ("M24", "maya-4", "state", "knee", "...cleared to run normally, hills included",
     "cleared to run normally, hills included"),
    ("M25", "maya-4", "state", "days", "now three days: Tuesday, Friday, Sunday", "Tuesday, Friday and\nSunday"),
    ("M26", "maya-4", "state", "job", "a new job is why", "I've started a new job"),
    ("M27", "maya-4", "event", "run 4", "an 18 km long run", "Sunday's long run was 18 km"),
    ("M28", "maya-4", "event", "run 4", "...on Sunday 20 September", "Sunday's long run"),
    ("M29", "maya-4", "event", "run 4", "...in 1:51", "in 1:51"),
    # Maya, 29 September
    ("M30", "maya-5", "event", "10k race", "a 10k race", "I ran a 10k race"),
    ("M31", "maya-5", "event", "10k race", "...on Saturday 26 September", "on Saturday"),
    ("M32", "maya-5", "event", "10k race", "...in 49:30", "in 49:30"),
    ("M33", "maya-5", "event", "10k race", "...a personal best", "a personal best"),
    ("M34", "maya-5", "habit", "race starts", "habitually goes out too fast at race starts",
     "as I always seem to at race starts"),
    ("M35", "maya-5", "state", "goal", "goal now 1:45", "go\nfor 1:45 at the half instead of 1:50"),
]

# For each state: its values and the date each became true (the next one ends it). What was true on a day is the
# last value that had started by then.
STATES = {
    ("maya", "race date"): [("2026-09-01", "18 October"), ("2026-09-15", "1 November")],
    ("maya", "goal"): [("2026-09-01", "under 1:50"), ("2026-09-29", "1:45")],
    ("maya", "days"): [("2026-09-01", "Tue, Thu, Sat, Sun"), ("2026-09-22", "Tue, Fri, Sun")],
    ("maya", "time of day"): [("2026-09-01", "evenings"), ("2026-09-15", "mornings")],
    ("maya", "knee"): [("2026-09-01", "sore after long downhills"), ("2026-09-15", "sore after hill repeats"),
                       ("2026-09-22", "cleared: a tight IT band")],
    ("tom", "race date"): [("2026-09-02", "8 November")],
    ("tom", "fuel"): [("2026-09-02", "water alone"), ("2026-09-16", "gels")],
    ("tom", "diet"): [("2026-09-02", "vegetarian")],
}

# The check-ins that ask about the current state, and what each asks. Maya's last check-in asks for her race, goal,
# days, time of day and long runs; Tom's asks his longest run and the weeks to his race, and what to eat on race
# morning -- where his diet, his fuel and his cramps all bear on the answer.
ASKED = {
    "maya-6": {"race date": "1 November", "goal": "1:45", "days": "Tue, Fri, Sun", "time of day": "mornings",
               "long runs": "14, 16, 17, 18 km on 30 Aug, 6, 13, 20 Sep"},
    # the race date, not the weeks to it: a coach with the right date can still count the weeks wrong, and that is
    # arithmetic, not memory (reported apart)
    "tom-3": {"longest run": "28 km", "race date": "8 November", "diet": "vegetarian",
              "fuel": "gels", "cramps": "eat at least two hours before"},
}


def current(runner, state, day):
    """What a state was on a day, or None if the runner had not said yet."""
    values = [v for since, v in STATES[(runner, state)] if since <= day]
    return values[-1] if values else None


def check():
    """Every fact's words are in its message (spacing aside), ids are unique,
    and every state's dates run in order."""
    ids = [f[0] for f in FACTS]
    assert len(ids) == len(set(ids))
    for fid, session, kind, topic, fact, words in FACTS:
        assert kind in ("state", "event", "habit"), fid
        assert " ".join(words.split()) in " ".join(MESSAGES[session].split()), (fid, words)
    for key, values in STATES.items():
        assert [d for d, _ in values] == sorted(d for d, _ in values), key
    return len(FACTS)
