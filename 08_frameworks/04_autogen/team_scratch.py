"""Brightwater's analysis team, from this repo's own parts.

The same four people, routing rules and stopping rules as `analysis_team.py` -- but not the same
design. AutoGen's team is a chat that everybody reads. This one is a case file:

    the case      chapter 6's `state.Shared`: one board, a field per piece of work, each field written by
                  one owner and read only by the people who need it. Nobody reads a transcript.
    evidence      written to the board by the code runner itself. The analyst cannot write it -- the board
                  refuses anyone but the owner -- so the reviewer judges what the program printed, never
                  somebody's account of it. AutoGen needed a subclass to get the same thing into the chat.
    agents        chapter 6's `supervisor.Agent`: a persona, its own tools, chapter 2's loop
    running code  a `run_python` TOOL (chapter 3's `Tool`) rather than a separate executor agent. Same gate as
                  the AutoGen build, and a subprocess started with a deliberately small environment --
                  chapter 7's lesson that a process you launch starts with whatever you hand it
    forms         the review and the finding are filled in through chapter 1's `extract()`: the provider
                  enforces the schema, so the reply is the form. A form still unusable after extract's
                  retries comes back as chapter 6's failure form
    who's next    `analysis_team.route`, the same rules as the AutoGen team
    going out     the finding's figure must be one the program printed -- chapter 6's `sendable`, for a Finding
    the budget    chapter 6's `Meter`, with a limit
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import analysis_team as AT
import brightwater as B

_REPO = Path(__file__).resolve().parents[2]
for _chapter in ("01_foundations", "02_agent_runtime", "03_tools", "06_orchestration"):
    if str(_REPO / _chapter) not in sys.path:
        sys.path.append(str(_REPO / _chapter))

from meter import Meter  # noqa: E402 -- chapter 6, on the path from the lines above

MODEL = AT.MODEL

# what the child process is given: enough to start Python on this machine, and nothing else
_KEEP = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "HOME", "LANG", "PYTHONIOENCODING")

# the case file: what each field holds, who writes it, who reads it
FIELDS = {"plan": "how to answer: the steps, and how to check the data first",
          "evidence": "the last program that ran, and what it printed",
          "report": "the analyst's account of what the program found",
          "review": "the reviewer's verdict, and anything that must be fixed",
          "finding": "the answer, as it goes to the commercial team"}
OWNER = {"plan": "planner", "evidence": "runner", "report": "analyst", "review": "reviewer", "finding": "reporter"}
READERS = {"plan": ["planner", "analyst", "reviewer"],
           "evidence": ["analyst", "reviewer", "reporter"],
           "report": ["reviewer", "reporter"],
           "review": ["planner", "analyst", "reviewer"],
           "finding": []}
WRITES = {who: field for field, who in OWNER.items()}
FORMS = {"reviewer": B.Review, "reporter": B.Finding}


# the AutoGen analyst writes a code block that its executor finds and runs; this one calls a tool
ANALYST = """You are Brightwater's data analyst. Answer the question by writing a Python (pandas) program that reads
orders.csv from the current folder and prints every number the answer needs, and running it with the run_python
tool. Check the data before you use it -- print anything suspicious you find, and say what you did about it. Print
only what the answer needs: no decoration, no full tables. When it has run, report the figures it printed, exactly."""

# the AutoGen reviewer's prompt, told where the evidence is: a field of its own, not the end of a report
REVIEWER = AT.REVIEWER.replace(
    "The analyst's report ends with the code that\nran and what it printed: judge THAT",
    "The evidence on the case is the code that ran\nand what it printed: judge THAT")
assert REVIEWER != AT.REVIEWER


class Budget(Meter):
    """Chapter 6's `Meter`, with a limit: every call counted and logged, none made past the limit.

    The scratch team's turns are whole chapter 2 loops -- the analyst's can be several calls -- so a
    check between turns is not enough, for the same reason it was not enough for AutoGen.
    `over` says why it stopped, because a form turn reports the stop as its own failure.

        with Budget(35_000, "scratch Q1") as budget:
            team.run(question)
        budget.summary()
    """

    def __init__(self, limit=AT.SCRATCH_BUDGET, label="scratch ?"):
        self.limit, self.label, self.over = limit, label, None

    def spent(self):
        return sum(call["prompt"] + call["completion"] for call in self.calls)

    def __enter__(self):
        import llm
        super().__enter__()
        metered = llm.client.chat.completions.create

        def create(*args, **kwargs):
            if self.spent() >= self.limit:
                self.over = f"token budget of {self.limit:,} reached ({self.spent():,} spent)"
                AT.progress(*self.label.split(" ", 1), self.spent(), "BUDGET REACHED -- stopping")
                raise AT.BudgetExceeded(self.over)
            response = metered(*args, **kwargs)
            AT.progress(*self.label.split(" ", 1), self.spent())
            return response

        llm.client.chat.completions.create = create
        return self                       # Meter's __exit__ puts the real client back


def python_runner(work_dir, board, timeout=60, limit=AT.OUTPUT_LIMIT, label="scratch ?", refused=None):
    """A `run_python(code)` tool bound to one working folder. What ran, and what it printed, goes on the board."""
    def run_python(code: Annotated[str, "A complete Python program. It runs in a folder holding orders.csv; "
                                        "print everything you need to see."]) -> str:
        """Run a Python program and return what it printed (stdout and stderr)."""
        ok, why = B.gate(code)
        if not ok:
            if refused is not None:
                refused.append({"why": why, "code": code[:300]})
            return f"Not run -- {why}. Rewrite the program without it."
        script = Path(work_dir) / f"step_{len(list(Path(work_dir).glob('step_*.py')))}.py"
        script.write_text(code, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k in _KEEP}
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            done = subprocess.run([sys.executable, script.name], cwd=work_dir, env=env, capture_output=True,
                                  text=True, encoding="utf-8", timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"Stopped after {timeout}s without finishing."
        output = (done.stdout + done.stderr).strip() or "(it printed nothing)"
        output = output[:limit] + ("\n... (cut)" if len(output) > limit else "")
        errors = [l for l in output.splitlines() if "Error" in l or "Traceback" in l]
        AT.progress(*label.split(" ", 1), "-", f"code exit={done.returncode} {errors[-1][:120] if errors else ''}")
        # the runner, not the analyst, puts the evidence on the case
        board.set("runner", "evidence", f"The program:\n```python\n{code}\n```\n"
                                        f"What it printed (exit {done.returncode}):\n```\n{output}\n```")
        return f"exit {done.returncode}:\n{output}"
    return run_python


def grounded(finding, evidence, tolerance=0.051):
    """Whether the finding's figure is one the program printed (to rounding). None if so, else why not.

    Chapter 6's `sendable`, for a Finding: a dull check, run before anything leaves the team. It
    catches a figure the reporter made up or worked out for itself; it cannot catch an analysis
    that printed the wrong figure.
    """
    printed = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", evidence or "")]
    if any(abs(n - finding.value) <= tolerance or abs(n * 100 - finding.value) <= tolerance for n in printed):
        return None
    return f"held for a person: {finding.value} is not a figure the program printed"


class Team:
    """The four people and the case file. `answer(name, request, form)` replaces every model call offline."""

    def __init__(self, work_dir=None, answer=None, max_turns=AT.MAX_MESSAGES, qid="?", skills=False):
        from state import Shared
        self.label = f"scratch {qid}"
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="brightwater-")
        B.write_orders(self.work_dir)
        self.board = Shared(fields=FIELDS, owner=OWNER, readers=READERS)
        self.refused = []
        self.reviews = []
        self.max_turns = max_turns
        self.answer = answer
        self.run_python = python_runner(self.work_dir, self.board, label=self.label, refused=self.refused)
        self.skills, self.skills_read = skills, []
        self.agents = self._agents()

    def read_skill(self, name: Annotated[str, "The procedure's name, exactly as listed"]) -> str:
        """The full text of one of the team's procedures."""
        self.skills_read.append(name)
        return B.read_skill(name)

    def _agents(self):
        from registry import Registry
        from supervisor import Agent
        from tools import Tool
        # with `skills`, the planner may read the team's procedures through a chapter 3 tool -- the same
        # text and the same catalogue the AutoGen planner gets
        planner = (Agent("planner", AT.PLANNER_WITH_SKILLS, model=MODEL,
                         tools=Registry([Tool(self.read_skill, name="read_skill",
                                              description=B.READ_SKILL_DESCRIPTION)]))
                   if self.skills else Agent("planner", AT.PLANNER, model=MODEL))
        return {"planner": planner,
                "analyst": Agent("analyst", ANALYST, tools=Registry([Tool(self.run_python)]), model=MODEL),
                "reviewer": Agent("reviewer", REVIEWER, model=MODEL),
                "reporter": Agent("reporter", AT.REPORTER, model=MODEL)}

    def _ask(self, name, request, form=None):
        if self.answer is not None:
            return self.answer(name, request, form)
        if form is None:
            return self.agents[name].run(request)
        # The reviewer and the reporter have no tools: they read and fill in a form. An agent loop that
        # finishes by calling a `submit` tool would let the model answer in prose instead. Chapter 1's
        # `extract()` asks the provider to enforce the form's schema, so the reply IS the form, and validates
        # and retries behind that -- what AutoGen's `output_content_type` does.
        from failures import failure
        from llm import extract
        agent = self.agents[name]
        try:
            return extract(f"{agent.persona}\n\n{request}", form, model=agent.model)
        except ValueError as unusable:          # still not the form after extract's retries
            return failure(f"{name} did not fill in its form", str(unusable)[:200])

    def _turn(self, name, question):
        """One person's turn: they read their view of the case, and what they produce goes in their field."""
        request = f"The question: {question}\n\n{self.board.view(name)}\n\nYour turn."
        form = FORMS.get(name)
        said = self._ask(name, request, form)
        if form is not None and not isinstance(said, form):
            # a failure form from `attempt`: it raised, was stopped, or answered in prose
            return None, f"needs a person: {said.answer} -- {said.unknowns}"[:300]
        text = said.render() if name == "reviewer" else said.model_dump_json() if name == "reporter" else said
        self.board.set(name, WRITES[name], text)
        return said, None

    def run(self, question, budget=AT.SCRATCH_BUDGET, qid="?"):
        """Answer one question, inside a `Budget`: every model call is counted, none made past `budget`."""
        finding, stop = None, None
        with Budget(budget, self.label) as spent:
            try:
                finding, stop = self._work(question)
            except AT.BudgetExceeded:
                pass
        stop = spent.over or stop          # a form turn reports the budget as its own failure
        turns = [{"source": "user", "text": question}] + [
            {"source": w["author"], "text": w["to"], "field": w["field"],
             **({"kind": "CodeExecutionEvent"} if w["author"] == "runner" else {})} for w in self.board.history]
        return {"finding": finding, "stop_reason": stop, "turns": turns,
                "speakers": [t["source"] for t in turns[1:] if t["source"] != "runner"],
                "refused": list(self.refused), "rejections": sum(not r.approved for r in self.reviews),
                "skills read": list(self.skills_read),
                "usage": spent.summary()}

    def _work(self, question):
        last, said = "user", question
        for _ in range(self.max_turns - 1):
            name = AT.route(last, said)
            said, failed = self._turn(name, question)
            if failed:
                return None, failed
            if name == "reporter":
                return said, grounded(said, self.board.values.get("evidence")) or "reporter answered"
            if name == "reviewer":
                self.reviews.append(said)
                if sum(not r.approved for r in self.reviews) >= AT.MAX_REJECTIONS:
                    return None, "needs a person: the second review did not approve"
            last = name
        return None, "too many turns"
