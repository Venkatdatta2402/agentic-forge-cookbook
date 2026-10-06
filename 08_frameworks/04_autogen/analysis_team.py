"""Brightwater's analysis team, as an AutoGen group chat.

Four agents in one conversation, each seeing what the others said:

    planner    reads the question and says how to answer it
    analyst    writes pandas code and RUNS it -- a `CodeExecutorAgent`, which asks the model for code,
               executes the code blocks it gets back, and reports what they printed
    reviewer   checks the method and the figures, and replies APPROVED or says what to fix
    reporter   writes the answer as a `Finding` -- structured output -- once the reviewer approves

Who speaks next is AutoGen's `SelectorGroupChat`: a model reads the conversation and picks. Where the
choice should not be a judgement it is code -- `next_speaker` sends the question to the planner, any
analysis to the reviewer, and an approval to the reporter -- and returns None everywhere else, which
hands that turn back to the model. The chat stops when the reporter has spoken, or after too many
messages, too many tokens or too long: four termination conditions, combined with `|`.

Three pieces of plumbing, each forced by something measured while building this:

    run_team      AutoGen is asyncio, and its local code executor starts Python with
                  `asyncio.create_subprocess_exec` -- which a Windows SelectorEventLoop does not
                  implement, and a Jupyter kernel on Windows runs a SelectorEventLoop. Measured:
                  NotImplementedError. So the team runs on its own thread with a ProactorEventLoop,
                  the same bridge chapter 7's `Session` built for MCP.
    Executor      the executor builds the child's environment from `os.environ.copy()`, so code a model
                  wrote can read every API key the notebook loaded. Measured: True. This one removes
                  anything that looks like a secret for the length of each execution.
    approve       AutoGen's `approval_func`, wired to `brightwater.gate`: code is checked before it runs.
"""

import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

import brightwater as B

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "openai/gpt-oss-120b"
GROQ = "https://api.groq.com/openai/v1"
# 12, not 20: with 20 the scratch team spent about 100,000 tokens on one question, its reviewer
# rejecting the same analysis five times. A run that is not done by 12 stops and says so.
MAX_MESSAGES = 12

# Token budgets per run, enforced in code: a run that reaches its budget stops and says so. Measured
# before they existed: about 10k tokens for a run the reviewer approves at once, about 100k for one
# where the reviewer kept rejecting.
AUTOGEN_BUDGET = 20_000
SCRATCH_BUDGET = 35_000
# For the procedures experiment, AutoGen's budget is the scratch team's: its Q1 run without them already
# came to 21,856, and reading procedures adds the planner's tool round. A budget that stopped this run
# before its write-up, as it stopped Q3's, would measure the budget rather than the procedures.
SKILLS_BUDGET = 35_000

PROGRESS = Path(__file__).resolve().parent / "store" / "progress.log"


class BudgetExceeded(RuntimeError):
    """A run reached its token budget. Raised before the next model call, wherever in the chat it comes."""


def progress(build, qid, spent, note=""):
    """One line per model call, so a long run can be watched -- and stopped -- while it runs."""
    PROGRESS.parent.mkdir(exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as log:
        log.write(f"{time.strftime('%H:%M:%S')} {build} {qid} total={spent} {note}\n")


def model_client(model=MODEL, label="autogen ?", budget=AUTOGEN_BUDGET):
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    class Logged(OpenAIChatCompletionClient):
        """Every call's tokens added to a running total and written to the progress log -- and no call
        made once the total has reached the budget.

        Checked HERE, per model call, because AutoGen's own `TokenUsageTermination` is checked between
        agent turns and counts the usage attached to the messages a turn emits. The analyst is a
        `CodeExecutorAgent`, whose one turn can be many calls -- write code, run it, decide whether to
        retry, rewrite, reflect -- and the first live run with a 15,000-token termination reached
        44,971 tokens before it was stopped by hand.
        """

        spent = calls = prompt = completion = 0

        async def create(self, *args, **kwargs):
            if Logged.spent >= budget:
                progress(*label.split(" ", 1), Logged.spent, "BUDGET REACHED -- stopping")
                raise BudgetExceeded(f"token budget of {budget:,} reached ({Logged.spent:,} spent)")
            result = await super().create(*args, **kwargs)
            Logged.calls += 1
            Logged.prompt += result.usage.prompt_tokens
            Logged.completion += result.usage.completion_tokens
            Logged.spent = Logged.prompt + Logged.completion
            progress(*label.split(" ", 1), Logged.spent)
            return result

        @classmethod
        def billed(cls):
            """What the API was charged for -- every call, including the ones no message carries."""
            return {"calls": cls.calls, "prompt": cls.prompt, "completion": cls.completion, "total": cls.spent}
    # AutoGen needs to be told what a model it has never heard of can do
    info = ModelInfo(vision=False, function_calling=True, json_output=True, structured_output=True,
                     family="unknown")
    return Logged(model=model, base_url=GROQ, api_key=os.environ["GROQ_API_KEY"],
                  model_info=info, temperature=0, max_retries=8)


# --------------------------------------------------------------------------- executing model-written code

_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_ENV_LOCK = threading.Lock()


def _executor_class():
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

    class Executor(LocalCommandLineCodeExecutor):
        """AutoGen's local executor, with secrets taken out of the environment while code runs.

        The parent copies `os.environ` into the child at the moment it starts it, so the secrets are
        removed for exactly that long and put back after. `os.environ` belongs to the whole process,
        which is why the lock: this is a workaround for a missing parameter, not a sandbox.
        """

        label = "autogen ?"

        async def execute_code_blocks(self, code_blocks, cancellation_token):
            with _ENV_LOCK:
                hidden = {k: os.environ.pop(k) for k in list(os.environ) if _SECRET.search(k)}
                # and printed as UTF-8: on Windows a child's output is otherwise cp1252, and a program
                # printing the model's favourite non-breaking hyphen died of UnicodeEncodeError (live,
                # Q1 with procedures). The scratch build's runner has always set this.
                encoding = os.environ.get("PYTHONIOENCODING")
                os.environ["PYTHONIOENCODING"] = "utf-8"
                try:
                    result = await super().execute_code_blocks(code_blocks, cancellation_token)
                finally:
                    os.environ.update(hidden)
                    if encoding is None:
                        os.environ.pop("PYTHONIOENCODING", None)
                    else:
                        os.environ["PYTHONIOENCODING"] = encoding
            # what a program printed goes back into the conversation; cut it as chapter 3's invoke()
            # cuts a tool result. Uncut, one long printout pushed a request to 8,770 tokens and Groq
            # refused it (413: the free tier's limit is 8,000 tokens a minute)
            if len(result.output) > OUTPUT_LIMIT:
                result.output = result.output[:OUTPUT_LIMIT] + "\n... (output cut)"
            # one line per execution, so a run whose code keeps failing can be seen failing
            errors = [l for l in result.output.splitlines() if "Error" in l or "Traceback" in l]
            progress(*self.label.split(" ", 1), "-", f"code exit={result.exit_code} {errors[-1][:120] if errors else ''}")
            return result

    return Executor


OUTPUT_LIMIT = 4000   # characters of a program's output that go back into the chat

REFUSED = []          # every block the gate turned down, for the notebook to show


def _analyst_class():
    from autogen_agentchat.agents import CodeExecutorAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import CodeExecutionEvent, CodeGenerationEvent, TextMessage

    class EvidenceAnalyst(CodeExecutorAgent):
        """AutoGen's code-executing agent, whose final report carries the code it ran and what it printed."""

        async def on_messages_stream(self, messages, cancellation_token):
            code = output = None
            async for item in super().on_messages_stream(messages, cancellation_token):
                if isinstance(item, CodeGenerationEvent) and item.code_blocks:
                    code = "\n\n".join(block.code for block in item.code_blocks)
                elif isinstance(item, CodeExecutionEvent):
                    output = item.result.output
                elif isinstance(item, Response) and code is not None and isinstance(item.chat_message, TextMessage):
                    report = item.chat_message
                    item = Response(chat_message=TextMessage(source=report.source, models_usage=report.models_usage,
                                                             content=with_evidence(report.content, code, output or "")),
                                    inner_messages=item.inner_messages)
                yield item

    return EvidenceAnalyst


def approve(request):
    """AutoGen calls this before running any code block. The decision is `brightwater.gate`'s."""
    from autogen_agentchat.agents import ApprovalResponse
    ok, why = B.gate(request.code)
    if not ok:
        REFUSED.append({"why": why, "code": request.code[:300]})
    return ApprovalResponse(approved=ok, reason=why or "allowed")


# --------------------------------------------------------------------------- the team

PLANNER = """You plan data analyses for Brightwater Outfitters' commercial team. Given a question about the order
export orders.csv (columns: order_id, order_date, region, channel, category, product, units, unit_price, discount,
returned), say in a few numbered steps how to answer it -- including how to check the data before trusting it.
Do not write code. Keep it short."""

ANALYST = """You are Brightwater's data analyst. Answer the question by writing Python (pandas) in ONE ```python
code block that reads orders.csv from the current folder and prints every number the answer needs. Check the data
before you use it -- print anything suspicious you find, and say what you did about it. Print only what the answer
needs: no decoration, no full tables. When the code has run, report the figures it printed, exactly."""

REVIEWER = """You review analyses before they reach the commercial team. The analyst's report ends with the code that
ran and what it printed: judge THAT, not the analyst's description of it. Check the method and the figures against
the question: is the data checked and cleaned, is every figure computed the way the question defines it, is the
arithmetic right? Fill in your review. If you do not approve, list each problem specifically enough to fix, and say
who must fix it. There is one revision: if a second review still finds problems, a person takes over."""

PLANNER_WITH_SKILLS = PLANNER + """

The team keeps the procedures it has learned; the read_skill tool lists them. Before you plan, read the ones that
apply to this question, and build their steps into your plan."""

REPORTER = """You write the final answer for Brightwater's commercial team from the approved analysis: the one-word
answer, the key figure as a percentage to one decimal place, two or three plain sentences, and every data problem
that was found and handled."""


EVIDENCE_LIMIT = 2500    # characters of printed output carried in the report


def with_evidence(report, code, output):
    """The analyst's report, with the code that ran and what it printed appended -- by code, every time.

    Added after both builds' reviewers kept rejecting: they were asked to check figures they could not
    see. In AutoGen the code and its output are EVENTS, which other agents never receive; the reviewer
    read only the analyst's summary of them, and the scratch build's reviewer said so in as many words:
    "no verifiable evidence was provided". A check needs the artifact, not a claim about it -- the same
    reason chapter 3's `verified()` looks at the world rather than at what the tool reported.
    """
    printed = output if len(output) <= EVIDENCE_LIMIT else output[:EVIDENCE_LIMIT] + "\n... (cut)"
    return f"{report}\n\nThe code that ran:\n```python\n{code}\n```\nWhat it printed:\n```\n{printed}\n```"


def route(source, content):
    """Who speaks after `source`. Shared with the scratch build, so both teams are routed by the same rules.

    A review is a `Review` form, so who fixes what is a field the reviewer filled in, not a word to look
    for in its prose. None only if the reviewer answered in prose, when the selector model reads it.
    """
    if source == "user":
        return "planner"
    if source == "planner":
        return "analyst"
    if source == "analyst":
        return "reviewer"
    if source == "reviewer":
        if isinstance(content, B.Review):
            return "reporter" if content.approved else ("planner" if content.fix_by == "planner" else "analyst")
        return None
    return None


def next_speaker(messages):
    """AutoGen's `selector_func`: the routing rules, applied to the last message in the chat."""
    last = messages[-1]
    return route(last.source, getattr(last, "content", ""))


def rejections(messages):
    """How many reviews in a chat did not approve."""
    return sum(1 for m in messages if isinstance(getattr(m, "content", None), B.Review) and not m.content.approved)


MAX_REJECTIONS = 2       # the first sends the work back once; the second hands it to a person


def build_team(client, work_dir, max_messages=MAX_MESSAGES, max_tokens=AUTOGEN_BUDGET, seconds=600, label="autogen ?",
               skills=False):
    from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
    from autogen_agentchat.conditions import (FunctionalTermination, MaxMessageTermination,
                                              SourceMatchTermination, TimeoutTermination, TokenUsageTermination)
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_core.model_context import TokenLimitedChatCompletionContext
    from autogen_core.tools import FunctionTool

    # every agent sees as much of the chat as fits in 4,500 tokens, newest first: Groq's free tier
    # refuses any single request above 8,000 tokens a minute, and a group chat's history only grows.
    # The first version kept the last 10 MESSAGES -- one of which was a long program and its printout,
    # and the request came to 8,770 tokens. A count of messages says nothing about their size.
    recent = lambda: TokenLimitedChatCompletionContext(client, token_limit=4500)
    # With `skills`, the planner may read the team's procedures, through an ordinary AutoGen tool. Not
    # AutoGen's `memory=`: that puts memory into the context for the agent (all of it, with `ListMemory`),
    # and the point here is whether the agent picks the right procedures itself.
    tools = ([FunctionTool(B.read_skill, name="read_skill", description=B.READ_SKILL_DESCRIPTION)]
             if skills else [])
    planner = AssistantAgent("planner", client, system_message=PLANNER_WITH_SKILLS if skills else PLANNER,
                             model_context=recent(), tools=tools, reflect_on_tool_use=bool(skills),
                             max_tool_iterations=3,
                             description="Plans how to answer a data question, including checks on the data.")
    executor = _executor_class()(work_dir=work_dir, timeout=60, cleanup_temp_files=False)   # keep what ran
    executor.label = label
    analyst = _analyst_class()("analyst", executor, model_client=client,
                                system_message=ANALYST, model_context=recent(), approval_func=approve,
                                max_retries_on_error=2, supported_languages=["python"],
                                description="Writes and runs pandas code against orders.csv.")
    reviewer = AssistantAgent("reviewer", client, system_message=REVIEWER, model_context=recent(),
                              output_content_type=B.Review,
                              description="Checks the code and its output; approves or lists what to fix.")
    reporter = AssistantAgent("reporter", client, system_message=REPORTER, model_context=recent(),
                              output_content_type=B.Finding,
                              description="Writes the final answer once the reviewer has approved.")
    seen = []

    def second_rejection(delta):
        # a person takes over after the second review that does not approve: one revision, not a loop
        seen.extend(delta)
        return rejections(seen) >= MAX_REJECTIONS

    stop = (SourceMatchTermination(["reporter"]) | MaxMessageTermination(max_messages)
            | TokenUsageTermination(max_total_token=max_tokens) | TimeoutTermination(seconds)
            | FunctionalTermination(second_rejection))
    # a structured reply is a message type the team has to be told about, or it refuses to pass it on
    from autogen_agentchat.messages import StructuredMessage
    return SelectorGroupChat([planner, analyst, reviewer, reporter], model_client=client,
                             selector_func=next_speaker, termination_condition=stop,
                             custom_message_types=[StructuredMessage[B.Finding], StructuredMessage[B.Review]])


# --------------------------------------------------------------------------- running it

def run_team(question, client_factory=None, timeout=900, qid="?", skills=False, budget=AUTOGEN_BUDGET):
    """Answer one question with a fresh team, on its own thread and event loop; return what happened.

    Everything asyncio -- the client, the team, the executor -- is created inside the thread's loop,
    because an async client belongs to the loop it was made in.
    """
    work_dir = tempfile.mkdtemp(prefix="brightwater-")
    B.write_orders(work_dir)
    box = {}

    async def go():
        from autogen_agentchat.base import TaskResult
        client = client_factory() if client_factory else model_client(label=f"autogen {qid}", budget=budget)
        box["client"] = client
        team = build_team(client, work_dir, max_tokens=budget, label=f"autogen {qid}", skills=skills)
        try:
            # streamed rather than `run`, so the messages so far survive a run that is stopped. The stream
            # is read to its end: returning from inside it abandons AutoGen's own shutdown task
            result = None
            async for item in team.run_stream(task=question):
                if isinstance(item, TaskResult):
                    result = item
                else:
                    box.setdefault("messages", []).append(item)
            return result
        finally:
            await client.close()

    def worker():
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        try:
            box["result"] = loop.run_until_complete(go())
        except Exception as failed:     # noqa: BLE001 -- reported, not raised
            box["error"] = f"{type(failed).__name__}: {failed}"
        finally:
            loop.close()

    began = time.monotonic()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    seconds = round(time.monotonic() - began, 1)
    if thread.is_alive():
        return {"error": f"did not finish within {timeout}s", "seconds": seconds, "finding": None, "turns": [],
                "code_runs": 0, "speakers": [], "usage": {}, "stop_reason": None}
    if "error" in box:
        from types import SimpleNamespace
        out = summarise(SimpleNamespace(messages=box.get("messages", []), stop_reason=None), seconds)
        out = {**out, "error": box["error"], "work_dir": work_dir}
    else:
        out = summarise(box["result"], seconds)
    if hasattr(box.get("client"), "billed"):
        # The usage AutoGen attaches to messages leaves out every call whose message the team never
        # emits -- the analyst's retries and reflections among them. Measured: 11,041 on messages
        # against 21,856 billed, on the same run. So a run's usage is the client's own count.
        out["usage on messages"], out["usage"] = out["usage"], box["client"].billed()
    return out


def summarise(result, seconds):
    """What the chat produced: the finding, who spoke, what ran, why it stopped, what it cost."""
    finding, turns, code_runs, skills_read = None, [], 0, []
    prompt = completion = calls = 0
    for message in result.messages:
        kind = type(message).__name__
        if kind == "ToolCallRequestEvent":
            skills_read += [json.loads(call.arguments or "{}").get("name", "?") for call in message.content
                            if call.name == "read_skill"]
        usage = getattr(message, "models_usage", None)
        if usage is not None:
            calls += 1
            prompt += usage.prompt_tokens
            completion += usage.completion_tokens
        content = getattr(message, "content", None)
        if kind == "CodeExecutionEvent":
            code_runs += 1
            content = f"exit {message.result.exit_code}: {message.result.output.strip()}"
        if isinstance(content, B.Finding):
            finding = content
        turns.append({"source": message.source, "kind": kind,
                      "text": content.model_dump_json() if hasattr(content, "model_dump_json") else str(content)})
    return {"finding": finding, "stop_reason": result.stop_reason, "turns": turns, "code_runs": code_runs,
            "rejections": rejections(result.messages), "skills read": skills_read,
            "speakers": [t["source"] for t in turns if t["kind"] in ("TextMessage", "CodeGenerationEvent",
                                                                      "StructuredMessage")],
            "usage": {"calls": calls, "prompt": prompt, "completion": completion, "total": prompt + completion},
            "seconds": seconds, "error": None}
