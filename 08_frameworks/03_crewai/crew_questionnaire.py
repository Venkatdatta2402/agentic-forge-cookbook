"""Tallyfold's security questionnaire, answered by a CrewAI crew.

Two crews over the same agents, because CrewAI offers two ways to run them:

    sequential     the crew this project uses. Three specialist tasks, each answering its own
                   questions, run AT THE SAME TIME (`async_execution=True`); the lead's task waits
                   for them through `context=` and assembles one questionnaire. Its output is a
                   pydantic form, and a guardrail -- `tallyfold.check`, plain code -- sends it back
                   with the problems if it does not pass.
    hierarchical   one task, the whole questionnaire, and a manager CrewAI creates from `manager_llm`
                   that decides who does what, delegating through CrewAI's built-in coworker tools.

What CrewAI asks for is a description of people and work: an agent is a role, a goal and a
backstory; a task is a description and an expected output. Everything between those -- the prompt
each agent is given, the tool-calling format, passing one task's output to the next, turning text
into the pydantic form -- is CrewAI's.

Two environment variables are set before CrewAI is imported, and both matter. CrewAI sends
telemetry to its own servers by default; and on the first run on a machine it records a machine
fingerprint and stops for 20 seconds with an interactive "view your execution traces? [y/N]"
prompt, which `CREWAI_TESTING` is the only switch for in this version. `OTEL_SDK_DISABLED`, the
switch most write-ups suggest, is deliberately NOT used: it turns off OpenTelemetry for the whole
process, which would also silence Langfuse (notebook 8) in the same kernel.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TESTING", "true")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import tallyfold as T  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# qwen3.8-27b, not the gpt-oss-120b the rest of the chapter uses. Under CrewAI's text tool format,
# gpt-oss made native tool calls Groq refused (one to repo_browser.print_tree, a tool from its own
# training), then answered without reading a document and cited ids it invented. qwen follows the
# format. Both builds use it, so the comparison stays like for like.
MODEL = "groq/qwen/qwen3.8-27b"           # LiteLLM's name for it: provider/model


class GroqLLM(LLM):
    """CrewAI's LLM, plus the one repair this model needs under CrewAI's prompt.

    CrewAI describes tools in prose and expects the model to write "Action: read_document" as TEXT;
    it sends no `tools` to the API. gpt-oss is trained to call tools natively, and on the first live
    run it did -- a native call to `repo_browser.print_tree`, a tool from its own training that exists
    nowhere here. Groq refuses that with a 400 ("Tool choice is none, but model called a tool"), and
    CrewAI has no step that turns a refusal into something the model can correct. This is that step,
    in the spirit of chapter 1's `tool_call_failure()`: say what was refused and what format is
    wanted, and try again. Every refusal is kept in `refusals`, so a run can report how often.
    """

    refusals: list = []
    waits: list = []

    def call(self, messages, *args, **kwargs):
        messages = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
        refused_times = 0
        for _ in range(15):
            try:
                return self._call_once(messages, *args, **kwargs)
            except Exception as failed:           # noqa: BLE001 -- two cases handled, everything else raised
                said = str(failed)
                if "rate_limit_exceeded" in said or "RateLimitError" in type(failed).__name__:
                    # Wait as long as Groq says. qwen3.8-27b allows 1,000 OUTPUT tokens a minute, and three
                    # specialists writing at once pass it within seconds. LiteLLM's own retries came back too
                    # fast and gave up -- and a task that gives up inside `async_execution` hangs the crew.
                    # Chapter 1's OpenAI client honours Retry-After, which is why the scratch build never
                    # noticed this limit.
                    wait = _retry_after(said)
                    GroqLLM.waits.append(wait)
                    time.sleep(wait)
                    continue
                if "tool_use_failed" not in said or refused_times == 2:
                    raise
                refused_times += 1
                GroqLLM.refusals.append(said[:300])
                messages = messages + [{"role": "user", "content": (
                    "Your last reply was refused: you tried to call a tool directly, and no tool can be called "
                    "that way here. Write your next step as plain text instead -- 'Thought:', then 'Action:' "
                    "with one of the tool names you were given and 'Action Input:' with its JSON arguments -- "
                    "or give your 'Final Answer:'.")}]

    def _call_once(self, messages, *args, **kwargs):
        return super().call(messages, *args, **kwargs)


def _retry_after(message, default=15.0):
    """Seconds to wait, from Groq's "Please try again in 13.8s" (or "1m2.5s"), plus a margin."""
    import re
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    if not match:
        return default
    return float(match.group(1) or 0) * 60 + float(match.group(2)) + 1.0


def crew_llm(model=None):
    # 429s are waited out in GroqLLM.call, as long as Groq asks, so LiteLLM's own quick retries are off
    return GroqLLM(model=model or MODEL, api_key=os.environ["GROQ_API_KEY"], temperature=0, num_retries=0)


TOOLS = [tool(f.__name__)(f) for f in T.TOOLS]


def make_agents(llm, delegation=False):
    """The four people. Specialists get the policy pack; the lead gets their answers and nothing else."""
    agents = {}
    for role in T.ROLES:
        title, goal, backstory = T.PERSONAS[role]
        agents[role] = Agent(role=title, goal=goal, backstory=backstory, tools=TOOLS, llm=llm,
                             allow_delegation=False, max_iter=8, verbose=False)
    title, goal, backstory = T.PERSONAS["lead"]
    agents["lead"] = Agent(role=title, goal=goal, backstory=backstory, llm=llm,
                           allow_delegation=delegation, max_iter=8, verbose=False)
    return agents


def guardrail(output):
    """CrewAI calls this on the lead's output: (True, output) to accept, (False, feedback) to redo the task."""
    questionnaire = output.pydantic
    if questionnaire is None:
        return False, "The output was not a questionnaire. Return the questionnaire as the JSON form asked for."
    problems = T.check(questionnaire)
    if problems:
        return False, "Fix these and return the whole questionnaire again:\n- " + "\n- ".join(problems)
    return True, output


def section_guardrail(role):
    """The same, for one specialist's task: `tallyfold.check_section`, as a CrewAI guardrail."""
    def check(output):
        section = output.pydantic
        if section is None:
            return False, "The output was not a Section. Return your answers as the JSON form asked for."
        problems = T.check_section(section, role)
        if problems:
            return False, "Fix these and return all your answers again:\n- " + "\n- ".join(problems)
        return True, output
    return check


ALL_QUESTIONS = "\n".join(f"{qid}: {text}" for qid, (_, text) in T.QUESTIONS.items())


def sequential_crew(llm=None, delegation=False):
    llm = llm or crew_llm()
    agents = make_agents(llm, delegation)
    sections = [Task(description=f"{T.INSTRUCTIONS}\n\nYour questions:\n{T.questions_for(role)}",
                     expected_output="One answer per question, with its status, the answer text, the statement "
                                     "ids it rests on, and needs_review.",
                     agent=agents[role], output_pydantic=T.Section, async_execution=True, name=role,
                     guardrail=section_guardrail(role), guardrail_max_retries=2)
                for role in T.ROLES]
    assemble = Task(description="Assemble Harrow & Pike's security questionnaire from the specialists' answers. "
                                "Keep every fact, status and statement id exactly as the specialist gave it; make the "
                                "wording consistent; answer every question exactly once; and write notes for sales "
                                f"on what will disappoint the customer.\n\nThe questionnaire:\n{ALL_QUESTIONS}",
                    expected_output="The full questionnaire: all nine answers, and notes for sales.",
                    agent=agents["lead"], context=sections, output_pydantic=T.Questionnaire,
                    guardrail=guardrail, guardrail_max_retries=2, name="assemble")
    return Crew(agents=list(agents.values()), tasks=[*sections, assemble], process=Process.sequential, verbose=False)


def hierarchical_crew(llm=None):
    """The same people with a manager deciding who answers what. No agent is assigned to the task."""
    llm = llm or crew_llm()
    agents = make_agents(llm)
    task = Task(description=f"Complete Harrow & Pike's security questionnaire.\n\n{T.INSTRUCTIONS}\n\n"
                            f"The questionnaire:\n{ALL_QUESTIONS}\n\nEach question belongs to the specialist whose "
                            "expertise it is: certifications, audits and people to compliance; technical controls "
                            "to security; data location and use to privacy. Delegate accordingly.",
                expected_output="The full questionnaire: all nine answers, and notes for sales.",
                output_pydantic=T.Questionnaire, guardrail=guardrail, guardrail_max_retries=2, name="questionnaire")
    return Crew(agents=[agents[r] for r in T.ROLES], tasks=[task], process=Process.hierarchical,
                manager_llm=llm, verbose=False)


class Counted:
    """Every completion LiteLLM makes while the block runs, counted from the responses themselves.

    CrewAI keeps its own count (`crew.usage_metrics`); this is a second one, taken one layer down,
    to compare against -- the same idea as chapter 6's `Meter`.
    """

    def __enter__(self):
        import litellm
        self.calls, self.prompt, self.completion = 0, 0, 0

        def record(kwargs, response, start, end):
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.calls += 1
                self.prompt += usage.prompt_tokens or 0
                self.completion += usage.completion_tokens or 0

        self._record = record
        litellm.success_callback.append(record)
        return self

    def __exit__(self, *exc):
        import litellm
        litellm.success_callback.remove(self._record)

    def summary(self):
        return {"calls": self.calls, "prompt": self.prompt, "completion": self.completion,
                "total": self.prompt + self.completion}


def run(crew, timeout=900):
    """Kick the crew off; return the questionnaire, each task's output, and two counts of what it cost.

    Two ways a crew ends badly, and neither may take the caller down with it:
      - it RAISES: a guardrail still failing after its retries ends the whole crew with an exception
      - it HANGS: an exception inside an `async_execution` task is printed from its worker thread and
        the crew then waits for that task forever -- no error, no timeout. Measured: the first live
        run sat for 15 minutes on one refused model call. So the crew runs in a thread and is given
        `timeout` seconds. A hung thread cannot be cancelled in Python; it is left behind, as a daemon
        so it does not hold the process open, and the run is reported as not having finished.
    """
    import threading
    error, result, box = None, None, {}

    def kick():
        try:
            box["result"] = crew.kickoff()
        except Exception as failed:        # noqa: BLE001 -- reported, not raised
            box["error"] = f"{type(failed).__name__}: {failed}"

    with Counted() as counted:
        worker = threading.Thread(target=kick, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            error = f"the crew did not finish within {timeout}s (a task failed in a worker thread, or it is stuck)"
        else:
            result, error = box.get("result"), box.get("error")
    questionnaire = result.pydantic if result is not None and isinstance(result.pydantic, T.Questionnaire) else None
    tasks = [{"task": t.name, "agent": t.agent.role if t.agent else "(manager's choice)",
              "tools used": t.used_tools, "tool errors": t.tools_errors, "delegations": t.delegations,
              "start": t.start_time, "end": t.end_time,
              "output": t.output.pydantic if t.output and t.output.pydantic else (t.output.raw if t.output else None)}
             for t in crew.tasks]
    return {"questionnaire": questionnaire, "raw": result.raw if result else None, "tasks": tasks, "error": error,
            "usage": crew.usage_metrics.model_dump() if crew.usage_metrics else {},
            "counted": counted.summary()}
