"""The baseline's answers, marked by hand against the repo BEFORE the judge saw them -- what the judge is checked
against.

For each question: the claims that are wrong (with what the repo says instead), the reference points the answer
makes, and the calls that could fairly go either way, kept apart rather than forced. Every wrong claim was checked
against the file it contradicts; figures the evidence lines do not hold (the handoff's 4,935 against 12,178 tokens,
the "no weather station for Narnia" refusal) were checked in their notebooks.
"""

import eval_set as E

Q = [q["question"] for q in E.QUESTIONS]

BASELINE = {
    Q[0]: {"wrong": [], "points": [1, 2, 3, 4, 5],
           "note": "'a schema cannot require values in a specified range' is not in the repo, and not true of JSON "
                   "Schema, which has minimum and maximum: at most 'not in the sources'"},
    Q[1]: {"wrong": [], "points": [1, 2, 3, 4],
           "borderline wrong": ["the runtime aborts the run and surfaces the detection message as the stop reason -- "
                                "true as far as it goes (the run ends, and its stop reason is that message), but it "
                                "leaves out the turn the Think is given to answer (runtime.py 223-226)"],
           "revised": "first marked wrong; moved to borderline AFTER the judge marked it supported, to keep to the "
                      "rule set before judging began -- an omission is not a false statement (05's durability "
                      "case). Discount it accordingly."},
    Q[2]: {"wrong": [], "points": [1, 2, 3, 4], "borderline points": [4]},
    Q[3]: {"wrong": [], "points": [1], "borderline points": [4],
           "note": "gives 'immediate context without extra calls' as the reason for injecting recall; misses the "
                   "repo's reason -- offered as a tool, recall is skipped on questions asking for a decision"},
    Q[4]: {"wrong": [], "points": [1, 2, 3, 4]},
    Q[5]: {"wrong": [], "points": [1, 2, 3, 4]},
    Q[6]: {"wrong": [], "points": [1, 2],
           "borderline wrong": ["the server inherits the parent's environment and all of the credentials in it -- the "
                                "SDK passes only an allowlist without secrets by default, and it is env={**os.environ} "
                                "that passes everything (07_mcp/README.md 112-116); but the README also says the "
                                "server 'starts holding whatever we were holding' (110-111)"]},
    Q[7]: {"wrong": [], "points": [1, 2, 3, 4, 5, 6],
           "note": "cites '【open_file†L1-L7】': the tool's name and lines, never the file -- so 'cites an answer "
                   "file' fails although the answer is right"},
    Q[8]: {"no answer": True},
    Q[9]: {"wrong": [], "points": [1, 2, 3, 4], "borderline points": [5]},
}

# 05's saved answers to the same questions, from its own live runs: its engine on the seven one-shot questions, and
# its two agents on the board and LangGraph questions. Marked before any judge saw them, and NOT used to tune the
# judge -- the judge's agreement is reported on these.
LLAMA = "08_frameworks/05_llamaindex/store/"
HELD_OUT = {
    ("engine", Q[0]): {"wrong": [], "points": [1, 2, 4, 5],
                       "note": "'a schema cannot express scores between 0 and 10' -- not in the repo, and not true of "
                               "JSON Schema"},
    ("engine", Q[1]): {"wrong": [], "points": [1, 2, 3, 4],
                       "borderline wrong": ["the runtime aborts the loop and emits a stop event with the reason -- "
                                            "leaves out the answer turn, as the baseline's does"]},
    ("engine", Q[2]): {"wrong": [], "points": [1, 2, 3], "borderline points": [4]},
    ("engine", Q[3]): {"wrong": [], "points": [1], "borderline points": [4]},
    ("engine", Q[4]): {"wrong": [], "points": [1, 2, 3, 4]},
    ("engine", Q[5]): {"wrong": ["after the transfer, the NEXT turn is answered by the new owner -- the new owner "
                                 "answers the SAME turn, 'so the person never waits for a second message' "
                                 "(supervisor.py 120-121)",
                                 "delegation is implemented by adding the specialist's tools to the supervisor's "
                                 "runtime -- the supervisor's tools are the specialist AGENTS, each wrapped by "
                                 "as_tool (supervisor.py 6-11, 85-107)"],
                       "points": [1, 2, 3]},
    ("engine", Q[6]): {"wrong": [], "points": [1, 2],
                       "borderline wrong": ["the server inherits the parent's environment and therefore any "
                                            "credentials in it -- as the baseline's MCP answer"]},
    ("llamaindex agent", Q[7]): {"wrong": [], "points": [1, 2, 3, 4, 5, 6]},
    ("scratch agent", Q[7]): {"wrong": [], "points": [1, 2, 3, 4, 5, 6],
                              "borderline wrong": ["procedural knowledge does not need a store -- the notebook says "
                                                   "MOSTLY not; what the agent works out for itself does "
                                                   "(04_types_of_memory, prose lines 80-94)"]},
    ("scratch agent", Q[8]): {"wrong": ["a LangGraph node returns a Wait (the case file's steps do)",
                                        "LangGraph lacks parallelism, streaming, rewinding (it has them; the case file "
                                        "lacks them)",
                                        "case.py's _go code labelled as ap_graph.py"],
                              "points": [1, 2, 3, 4, 5, 7], "borderline points": [6],
                              "note": "durability listed under LangGraph alone: an omission, not marked wrong"},
}


# Round 3's answer to the LangGraph question -- the first answer to it in any round -- checked against
# ap_desk.ipynb and ap_graph.py (its interrupt() line is ap_graph.py 292, shown in the notebook at prose line 97).
ROUND3 = {
    Q[8]: {"wrong": [], "points": [1, 2, 3, 4, 5],
           "borderline wrong": ["the case file's only failure mode is a Denied when the wrong person writes -- the "
                                "notebook says durability stops at the workflow in both designs (prose line 541)"]},
}

# Each judge tried on the held-out answers. "caught" is read, not counted: each wrong claim marked by hand was looked
# for in the judge's list of contradicted claims. The two that could not run are here too.
JUDGE_TRIALS = [
    {"judge": "gemini-3.8-flash", "ran": False,
     "why not": "503 'high demand' on every try, twice, an hour apart"},
    {"judge": "qwen/qwen3.8-27b (Groq)", "ran": False,
     "why not": "judged 3 of the 10 answers, then was refused on its limit of 1,000 output tokens a minute; "
                "a claim list is about that long, so a round would be mostly waiting"},
    {"judge": "gemini-2.5-flash", "ran": True, "verdicts": "held-out.gemini-2.5-flash.json", "requests": 20,
     "caught": ["the LangGraph answer's Wait credited to LangGraph",
                "the LangGraph answer's features LangGraph 'lacks' (as five claims, one a feature)"],
     "missed": ["handoff: the NEXT turn answered by the new owner", "delegation adds the specialist's tools",
                "case.py's code labelled ap_graph.py"],
     "false alarms": "7: six citation markers (【4†L1-L5】) read as file line numbers; 'only the refused outcome' "
                     "read outside its comparison",
     "points": "7 of 10 answers exactly as by hand; the other 3 differ by one borderline point"},
    {"judge": "openai/gpt-oss-20b (Groq)", "ran": True, "verdicts": "held-out.openai_gpt-oss-20b.json", "requests": 21,
     "caught": [],
     "missed": ["handoff: the NEXT turn answered by the new owner", "delegation adds the specialist's tools",
                "the LangGraph answer's Wait credited to LangGraph",
                "the LangGraph answer's features LangGraph 'lacks'", "case.py's code labelled ap_graph.py"],
     "false alarms": "1: 'started as a child of the notebook' marked wrong",
     "points": "4 of 10 exactly as by hand; on the compaction answer it found no claims at all; on the scratch "
               "board answer it found 3 of 6 points"},
]


def held_out_answers():
    """The held-out answers' text, from 05's saved runs, keyed as HELD_OUT is."""
    import json
    root = E.CB.REPO / LLAMA
    engine = {a["question"]: a["answer"] for a in json.loads((root / "answers.json").read_text(encoding="utf-8"))["answers"]}
    llama = json.loads((root / "agent.json").read_text(encoding="utf-8"))["runs"]
    scratch = json.loads((root / "agent_scratch.json").read_text(encoding="utf-8"))["runs"]
    found = {("engine", q): engine[q] for q in Q[:7]}
    found[("llamaindex agent", Q[7])] = llama[0]["answer"]
    found[("scratch agent", Q[7])] = scratch[0]["answer"]
    found[("scratch agent", Q[8])] = scratch[1]["answer"]
    assert set(found) == set(HELD_OUT) and all(found.values())
    return found


def judge_held_out(path, model=None, log=print):
    """The judge over the held-out answers, each verdict saved as it arrives (resumable, like judge_all)."""
    import json

    import claim_judge as J
    model = model or J.JUDGE_MODEL
    questions = {q["question"]: q for q in E.QUESTIONS}
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for (who, question), answer in held_out_answers().items():
        key = f"{who} :: {question}"
        if key in done:
            continue
        try:
            done[key] = J.judge(questions[question], answer, model=model, log=log)
        except J.JudgeQuotaSpent as spent:
            log(f"    stopped: {spent}")
            return done, False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"    judged {len(done)} of {len(HELD_OUT)}: {key[:70]}")
    return done, True


def side_by_side(labels, verdicts, questions=None):
    """Per answer: what I marked, and what the judge marked -- wrong claims and points made -- for reading together."""
    import claim_judge as J
    questions = questions or {q["question"]: q for q in E.QUESTIONS}
    rows = []
    for key, mine in labels.items():
        question = key[1] if isinstance(key, tuple) else key
        verdict = verdicts.get(f"{key[0]} :: {key[1]}" if isinstance(key, tuple) else key)
        if mine.get("no answer") or verdict is None or verdict.get("no answer"):
            continue
        judged = J.scores(verdict, questions[question])
        rows.append({"answer": key, "wrong, by hand": len(mine["wrong"]),
                     "borderline, by hand": len(mine.get("borderline wrong", [])),
                     "wrong, by the judge": judged["claims wrong"],
                     "judge's wrong claims": [f"{c['claim']} -- {c['why']}" for c in verdict["claims"]
                                              if c["verdict"] == "contradicted"],
                     "points, by hand": mine["points"], "borderline points": mine.get("borderline points", []),
                     "points, by the judge": verdict["points made"]})
    return rows
