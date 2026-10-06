"""What building the questionnaire crew found out about CrewAI, each one reproducible.

    prompt_seen          offline. The full prompt CrewAI sends a specialist for its first step.
    form_as_shown        offline. How the `Section` form is described to the model, by CrewAI and by
                         chapter 3's `Tool` -- which is how the scratch build's agents see it.
    guardrail_retry      offline. The lead leaves out a question; the guardrail sends the task back.
    prose_instead_of_form offline. An agent answers a form-shaped task in prose.
    telemetry            offline. What CrewAI does on this machine unless told not to.
    overlap              from a run: did the "parallel" tasks actually overlap?
"""

import os

import crew_questionnaire as CQ
import q_stubs as S
import questionnaire_scratch as QS
import tallyfold as T


class _Spy(S.ScriptedLLM):
    def call(self, messages, *args, **kwargs):
        if not hasattr(self, "first"):
            self.first = [dict(m) for m in messages]          # a copy: CrewAI appends to the same list
        return super().call(messages, *args, **kwargs)


def prompt_seen(role="compliance"):
    llm = _Spy()
    crew = CQ.sequential_crew(llm=llm)
    index = T.ROLES.index(role)
    crew.tasks = [crew.tasks[index]]
    crew.agents = [crew.agents[index]]
    crew.tasks[0].async_execution = False
    crew.kickoff()
    return {m["role"]: m["content"] for m in llm.first}


def form_as_shown():
    """The part of each build's prompt that tells the model what shape to answer in."""
    import json
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2] / "03_tools"))
    from tools import Tool
    user = prompt_seen()["user"]
    crewai_part = user[user.index("Ensure your final answer contains only"):user.index("Ensure the final output does not")]
    ours = Tool(lambda **kw: None, name="submit", description="Hand in your answer.", args=T.Section).schema
    answer = ours["function"]["parameters"]["$defs"]["Answer"]["properties"]
    return {"CrewAI shows the model": crewai_part.strip(),
            "chapter 3's Tool shows the model (Answer's fields)": json.dumps(answer, indent=1)}


def guardrail_retry():
    llm = S.ScriptedLLM(drop_first=True)
    result = CQ.run(CQ.sequential_crew(llm=llm))
    return {"lead attempts": llm.calls.count("lead"),
            "finished questionnaire passes the check": result["questionnaire"] is not None
            and not T.check(result["questionnaire"])}


def prose_instead_of_form():
    """With an LLM CrewAI treats as not supporting function calling (the stand-in), which takes the
    converter's text path. Groq's model takes a different path (`instructor`), not reproduced here."""
    from crewai import Agent, Crew, Task

    class Prose(S.ScriptedLLM):
        def call(self, messages, *args, **kwargs):
            return "Thought: done\nFinal Answer: Q1 no, Q2 no, Q3 partly -- see the certifications document."

    agent = Agent(role="Compliance manager", goal="g", backstory="b", llm=Prose(), verbose=False)
    task = Task(description="Answer Q1-Q3.", expected_output="answers", agent=agent, output_pydantic=T.Section)
    try:
        Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        return {"result": "no error", "pydantic": type(task.output.pydantic).__name__}
    except Exception as error:       # noqa: BLE001 -- what it raises is the finding
        return {"the crew ended with": type(error).__name__, "message": str(error)[:200]}


def failed_specialist(timeout=10):
    """The security engineer's model call fails. CrewAI's parallel task, and the scratch build's `attempt`."""
    import time

    import llm
    from failures import attempt

    class Fails(S.ScriptedLLM):
        def call(self, messages, *args, **kwargs):
            if "You are Security engineer" in "\n".join(str(m.get("content", "")) for m in messages):
                raise RuntimeError("provider refused the call")
            return super().call(messages, *args, **kwargs)

    began = time.monotonic()
    crew = CQ.run(CQ.sequential_crew(llm=Fails()), timeout=timeout)
    crew_seconds = round(time.monotonic() - began, 1)

    # the scratch desk: the security engineer is a real chapter 6 Agent, run through `attempt`, whose
    # model call fails; the other two specialists and the lead are stand-ins
    security, asked = QS.make_agents()["security"], []

    def answer(agent, request, model):
        asked.append(agent)
        return attempt(security, request, model=model) if agent == "security" else S.answer(agent, request, model)

    real = llm.client.chat.completions.create

    def refuse(*args, **kwargs):
        raise RuntimeError("provider refused the call")

    llm.client.chat.completions.create = refuse
    try:
        began = time.monotonic()
        desk = QS.Desk(answer=answer)
        out = desk.run()
    finally:
        llm.client.chat.completions.create = real
    return {"CrewAI": {"ended with": crew["error"], "seconds": crew_seconds},
            "scratch": {"questionnaire": out["questionnaire"], "failed": out["failed"],
                        "seconds": round(time.monotonic() - began, 1), "the lead was asked": "lead" in asked,
                        "log": desk.log}}


def telemetry():
    from crewai.events.listeners.tracing import utils
    stored = utils._load_user_data()
    return {"CREWAI_DISABLE_TELEMETRY": os.environ.get("CREWAI_DISABLE_TELEMETRY"),
            "CREWAI_TESTING (stops the first-run trace prompt)": os.environ.get("CREWAI_TESTING"),
            "OTEL_SDK_DISABLED (deliberately not set)": os.environ.get("OTEL_SDK_DISABLED"),
            "what CrewAI keeps about this machine (keys only)": sorted(stored)}


def overlap(tasks):
    """Each task's start and end, relative to the first start, and which specialist tasks overlapped."""
    timed = [t for t in tasks if t.get("start") and t.get("end")]
    if not timed:
        return {}
    zero = min(t["start"] for t in timed)
    spans = {t["task"]: (round((t["start"] - zero).total_seconds(), 1), round((t["end"] - zero).total_seconds(), 1))
             for t in timed}
    specialists = [spans[r] for r in T.ROLES if r in spans]
    together = all(a[0] < b[1] and b[0] < a[1] for i, a in enumerate(specialists) for b in specialists[i + 1:])
    return {"seconds from start": spans, "the three specialist tasks overlapped": together}


def report(questionnaire, width=150):
    """A questionnaire read against the key: each answer, what the pack supports, and the trap."""
    if questionnaire is None:
        print("no questionnaire")
        return 0
    rows = {row["id"]: row for row in T.grade(questionnaire)}
    by_id = {a.id: a for a in questionnaire.answers}
    for qid, (_, question) in T.QUESTIONS.items():
        row, answer = rows[qid], by_id.get(qid)
        mark = "ok " if row["right"] else "BAD"
        print(f"{qid} {mark} {row['status'] or '-':8} (key: {row['want']:8}) evidence {answer.evidence if answer else '-'}"
              + (f"  missing {row['missing evidence']}" if row["missing evidence"] else "")
              + ("  [needs review]" if answer and answer.needs_review else ""))
        if answer:
            print(f"     {answer.answer[:width]}")
    right = sum(row["right"] for row in rows.values())
    print(f"\n{right} of {len(rows)} with the status the policy pack supports")
    return right
