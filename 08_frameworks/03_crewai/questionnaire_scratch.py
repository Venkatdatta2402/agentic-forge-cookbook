"""Tallyfold's security questionnaire, from this repo's own parts.

Almost all of it is chapter 6, which built exactly these pieces for the Aria 2 recall:

    an agent        `supervisor.Agent`: a name, a persona (its system prompt) and its own tools,
                    running chapter 2's Think -> Act -> Observe loop
    a task          `failures.attempt(agent, request, model=...)`: chapter 6's `answer` -- run the agent
                    until it fills in a form, here `Section`/`Questionnaire` -- with a failure coming back
                    as a form too, whether the agent raised, was stopped, or answered in prose
    in parallel     a thread pool, as `aggregate.opinions` asks three specialists at once. A specialist
                    that fails does not take the others down, and the lead is not asked to assemble
                    around a hole: the desk stops and says which section failed and why
    passing work on the lead's request carries the specialists' forms, as JSON
    the guardrails  `tallyfold.check_section` on each specialist's form and `tallyfold.check` on the
                    lead's, the problems sent back, at most twice each

CrewAI's role / goal / backstory become one persona string; its task description and expected
output become the request; `context=` is a string the code builds; `async_execution` is a pool.
"""

import concurrent.futures
import json
import sys
from pathlib import Path

import tallyfold as T

_REPO = Path(__file__).resolve().parents[2]
for _chapter in ("01_foundations", "02_agent_runtime", "03_tools", "06_orchestration"):
    if str(_REPO / _chapter) not in sys.path:
        sys.path.append(str(_REPO / _chapter))

MODEL = "qwen/qwen3.8-27b"           # as the crew uses -- see crew_questionnaire.MODEL


def persona(role):
    title, goal, backstory = T.PERSONAS[role]
    return f"You are the {title} at Tallyfold. {backstory}\n\nYour goal: {goal}"


def make_agents():
    from registry import Registry
    from supervisor import Agent
    from tools import Tool
    pack = Registry([Tool(f) for f in T.TOOLS])
    agents = {role: Agent(role, persona(role), tools=pack, model=MODEL) for role in T.ROLES}
    agents["lead"] = Agent("lead", persona("lead"), model=MODEL)
    return agents


def live_answer(agent, request, model):
    from failures import attempt
    return attempt(agent, request, model=model)


def failed(form):
    """The reason, if this is chapter 6's failure form rather than the form that was asked for."""
    from messages import Response
    return f"{form.answer} -- {form.unknowns}"[:300] if isinstance(form, Response) else None


class Desk:
    """The specialists and the lead. `answer` is chapter 6's `attempt`, injectable so it can be stubbed."""

    def __init__(self, answer=None, agents=None, retries=2):
        self.answer = answer or live_answer
        self.agents = agents or (make_agents() if answer is None else {r: r for r in (*T.ROLES, "lead")})
        self.retries = retries
        self.log = []

    def section(self, role):
        """One specialist's answers, checked by `tallyfold.check_section` and sent back if they fail."""
        request = f"{T.INSTRUCTIONS}\n\nYour questions:\n{T.questions_for(role)}"
        feedback = ""
        for attempt in range(self.retries + 1):
            form = self.answer(self.agents[role], request + (f"\n\n{feedback}" if feedback else ""), T.Section)
            if failed(form):
                # it raised, was stopped, or would not use the form: another whole run is unlikely to
                # go differently, so it is reported, not retried (chapter 6's `attempt`, retries=0)
                self.log.append({"step": f"{role}, attempt {attempt + 1}", "failed": failed(form)})
                return form
            problems = (T.check_section(form, role) if isinstance(form, T.Section)
                        else ["the specialist did not return its answers as the form"])
            self.log.append({"step": f"{role}, attempt {attempt + 1}", "problems": problems})
            if not problems:
                return form
            feedback = "Fix these and return all your answers again:\n- " + "\n- ".join(problems)
        return form

    def assemble(self, sections, feedback=""):
        answers = json.dumps({role: s.model_dump() if isinstance(s, T.Section) else str(s)
                              for role, s in sections.items()}, indent=1)
        request = ("Assemble Harrow & Pike's security questionnaire from the specialists' answers below. Keep every "
                   "fact, status and statement id exactly as the specialist gave it; make the wording consistent; "
                   "answer every question exactly once; and write notes for sales on what will disappoint the "
                   f"customer.\n\nThe questionnaire:\n" + "\n".join(f"{q}: {t}" for q, (_, t) in T.QUESTIONS.items())
                   + f"\n\nThe specialists' answers:\n{answers}" + (f"\n\n{feedback}" if feedback else ""))
        return self.answer(self.agents["lead"], request, T.Questionnaire)

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            sections = dict(zip(T.ROLES, pool.map(self.section, T.ROLES)))
        holes = {role: failed(form) for role, form in sections.items() if failed(form)}
        if holes:
            # a lead handed a missing section would answer those questions itself, from nothing
            return {"questionnaire": None, "sections": sections, "failed": holes}
        feedback = ""
        for attempt in range(self.retries + 1):
            questionnaire = self.assemble(sections, feedback)
            if failed(questionnaire):
                self.log.append({"step": f"assemble, attempt {attempt + 1}", "failed": failed(questionnaire)})
                return {"questionnaire": None, "sections": sections, "failed": {"lead": failed(questionnaire)}}
            problems = (T.check(questionnaire) if isinstance(questionnaire, T.Questionnaire)
                        else ["the lead did not return a questionnaire"])
            self.log.append({"step": f"assemble, attempt {attempt + 1}", "problems": problems})
            if not problems:
                return {"questionnaire": questionnaire, "sections": sections}
            feedback = "Fix these and return the whole questionnaire again:\n- " + "\n- ".join(problems)
        return {"questionnaire": None, "sections": sections}
