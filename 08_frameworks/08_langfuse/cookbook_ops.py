"""05's cookbook assistant in Langfuse: what 07 did not show -- the full step tree through OpenTelemetry, a cost on
every model call, the agent's prompt as versions in Langfuse, and monitoring read back through the metrics API.

    setup()           Langfuse's client, with LlamaIndex's OpenTelemetry instrumentation sending to it
    ensure_price()    gpt-oss-120b on Groq, at Groq's prices, in the project's model table: Langfuse prices a call
                      from that table, and has no entry for this model
    ensure_prompts()  the agent's instructions as prompt `cookbook-agent`: version 1 is 05's AGENT_PROMPT, version 2
                      a revision made after 07's round 3
    run_version()     the three agent questions through 05's agent with one prompt version, each a traced, tagged
                      span; each answer saved locally before it is returned, as in 07
    judge()           07's claim judge over the saved answers, its "points made" put on each trace as a score
    metrics()         one query to the metrics API: cost, tokens and time, grouped as asked

The answers come from 05 as 07 left it (its four fixes in), on the 07 dataset's three agent questions.
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _folder in ("05_llamaindex", "07_langsmith"):
    if str(HERE.parent / _folder) not in sys.path:
        sys.path.append(str(HERE.parent / _folder))

import eval_set as E            # noqa: E402 -- 07's questions, points and evidence
import ask_index as A           # noqa: E402 -- 05's system
import claim_judge as J         # noqa: E402 -- 07's judge, checked against hand marks there

STORE = HERE / "store"
PROMPT = "cookbook-agent"
AGENT_QUESTIONS = [q for q in E.QUESTIONS if q["kind"] == "agent"]
BUDGET = 45_000                 # tokens a question, as in 05 and 07

# Groq's prices for openai/gpt-oss-120b, from its model page (console.groq.com/docs/model/openai/gpt-oss-120b) on
# 8 October 2026: $0.15 per million input tokens, $0.60 per million output. Langfuse wants USD per unit.
GROQ_PRICE = {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000}

# Version 2, written after 07's round 3. Its LangGraph answer was right but left out what is true of BOTH designs
# (durability stops at the workflow) and what the case file gives up, and called a Denied the case file's "only"
# failure mode. The additions are general -- nothing names LangGraph -- but they were written from misses on these
# very questions, so a better score on them shows the revision does what it was written for, not that it
# generalises. That needs questions it was not written from.
PROMPTS = {
    1: A.AGENT_PROMPT,
    2: A.AGENT_PROMPT + """

When the question compares two things, cover each side in turn: how it works, what can go wrong with it, and what it
gives up. Say plainly when something holds for both. Do not call anything the only one of its kind unless a source
says so.""",
}


# --------------------------------------------------------------------------- tracing

_SET_UP = False


def setup():
    """Langfuse's client, with LlamaIndex's OpenTelemetry instrumentation sending to it. Once per process.

    Langfuse 4 is an OpenTelemetry tracer: its client installs the tracer provider, and LlamaIndex's instrumentation
    (openinference) writes its spans to whatever provider is installed, so every LlamaIndex step lands in Langfuse
    -- query, retrieval, embedding, each model call -- nested under the span that was open when it ran.
    """
    global _SET_UP
    from langfuse import get_client
    client = get_client()
    if not _SET_UP:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        LlamaIndexInstrumentor().instrument()
        _SET_UP = True
    return client


def ensure_price(client):
    """The model's price in the project's model table, added if missing. Returns the entry.

    The table holds hundreds of Langfuse's own entries, so it is read page by page: on the first page alone, this
    entry is missed and creating it again is refused ("already exists in project")."""
    page = 1
    while True:
        listed = client.api.models.list(page=page, limit=100)
        found = [m for m in listed.data if m.model_name == A.MODEL]
        if found:
            return found[0]
        if page >= listed.meta.total_pages:
            break
        page += 1
    return client.api.models.create(model_name=A.MODEL, match_pattern=r"(?i)^(openai/)?gpt-oss-120b$", unit="TOKENS",
                                    input_price=GROQ_PRICE["input"], output_price=GROQ_PRICE["output"])


def ensure_prompts(client):
    """Prompt `cookbook-agent`, versions 1 and 2 as in PROMPTS, created in order if missing. Langfuse numbers
    versions itself, so each is checked to have the text it should."""
    have = {}
    for version in sorted(PROMPTS):
        try:
            have[version] = client.get_prompt(PROMPT, version=version, cache_ttl_seconds=0, max_retries=0)
        except Exception:                                   # noqa: BLE001 -- not created yet
            have[version] = client.create_prompt(name=PROMPT, prompt=PROMPTS[version], labels=[f"v{version}"],
                                                 commit_message="05's AGENT_PROMPT" if version == 1 else
                                                 "after 07's round 3: both sides, what each gives up")
        assert have[version].version == version and have[version].prompt == PROMPTS[version], version
    return have


# --------------------------------------------------------------------------- running

def _path(kind, version):
    return STORE / kind / f"v{version}.json"


def saved(version, kind="answers"):
    path = _path(kind, version)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(version, question, output):
    answers = saved(version)
    answers[question] = output
    _path("answers", version).parent.mkdir(parents=True, exist_ok=True)
    _path("answers", version).write_text(json.dumps(answers, indent=1, ensure_ascii=False), encoding="utf-8")


class GroqDayLimit(RuntimeError):
    """Groq's daily limit, hit mid-question: not an answer, so not saved, and the next run asks it again."""


async def run_version(client, index, version, chat_model=A.chat_model, environment=None, log=print):
    """The agent questions not yet answered with this prompt version, one at a time, each its own trace.

    `propagate_attributes` puts the prompt version, the tag and the trace name on every observation made inside it,
    LlamaIndex's included: that is how each model call is linked to the version that wrote its instructions.
    `environment` keeps stand-in runs apart from real ones in every view and query.
    """
    from langfuse import propagate_attributes
    prompt = client.get_prompt(PROMPT, version=version)
    for q in AGENT_QUESTIONS:
        if q["question"] in saved(version):
            continue
        llm = chat_model(budget=A.Budget(BUDGET, f"08 v{version}"))
        began = time.monotonic()
        with propagate_attributes(prompt=prompt, tags=[f"prompt v{version}"], trace_name="cookbook-agent",
                                  environment=environment), \
                client.start_as_current_observation(as_type="span", name="cookbook-agent",
                                                    input={"question": q["question"]}) as span:
            run = await A.run_agent(A.agent(index, llm, system_prompt=prompt.prompt), q["question"])
            span.update(output={"answer": run["answer"]}, metadata={"stopped": run["stopped"]})
            trace_id = client.get_current_trace_id()
        if run["stopped"] and "tokens per day" in run["stopped"]:
            raise GroqDayLimit(run["stopped"])
        _save(version, q["question"], {"kind": "agent", "answer": run["answer"], "stopped": run["stopped"],
                                       "calls": run["calls"], "usage": llm.budget.summary(), "trace": trace_id,
                                       "seconds": round(time.monotonic() - began, 1)})
        log(f"    v{version}: {q['question'][:60]} -- {llm.budget.spent:,} tokens")
    client.flush()


def set_aside_timeouts(version):
    """Move this version's answers that LlamaIndex's time limit cut off into v<n>.timed-out.json, so the next run
    asks them again. Kept, not deleted: they are a finding, and their traces stay in Langfuse."""
    answers = saved(version)
    cut = {q: a for q, a in answers.items() if a["stopped"] and "WorkflowTimeoutError" in a["stopped"]}
    if cut:
        aside = STORE / "answers" / f"v{version}.timed-out.json"
        kept = json.loads(aside.read_text(encoding="utf-8")) if aside.exists() else {}
        aside.write_text(json.dumps({**kept, **cut}, indent=1, ensure_ascii=False), encoding="utf-8")
        _path("answers", version).write_text(json.dumps({q: a for q, a in answers.items() if q not in cut},
                                                        indent=1, ensure_ascii=False), encoding="utf-8")
    return cut


def timed_out(version):
    aside = STORE / "answers" / f"v{version}.timed-out.json"
    return json.loads(aside.read_text(encoding="utf-8")) if aside.exists() else {}


def set_aside_unanswered(version, note):
    """Like set_aside_timeouts, for any answer that did not come: moved into v<n>.unanswered.json with a note of what
    was changed before asking it again."""
    answers = saved(version)
    cut = {q: {**a, "then": note} for q, a in answers.items() if not a["answer"]}
    if cut:
        aside = STORE / "answers" / f"v{version}.unanswered.json"
        kept = json.loads(aside.read_text(encoding="utf-8")) if aside.exists() else {}
        for q, a in cut.items():                     # every attempt kept, in order: a question can fail twice
            earlier = kept.get(q, [])
            kept[q] = (earlier if isinstance(earlier, list) else [earlier]) + [a]
        aside.write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
        _path("answers", version).write_text(json.dumps({q: a for q, a in answers.items() if q not in cut},
                                                        indent=1, ensure_ascii=False), encoding="utf-8")
    return cut


# --------------------------------------------------------------------------- judging

def judge(client, version, log=print):
    """07's judge over this version's saved answers; "points made" goes onto each answer's trace as a score."""
    verdicts, finished = J.judge_all(saved(version), _path("verdicts", version), log=log)
    questions = {q["question"]: q for q in AGENT_QUESTIONS}
    for question, verdict in verdicts.items():
        trace = saved(version)[question]["trace"]
        if "points made" in verdict and not verdict.get("scored"):
            client.create_score(trace_id=trace, name="points made", data_type="NUMERIC",
                                value=J.scores(verdict, questions[question])["points made"],
                                comment=f"{len(verdict['points made'])} of {len(questions[question]['points'])} "
                                        f"reference points, by {verdict['judge']}")
            verdict["scored"] = True
    _path("verdicts", version).write_text(json.dumps(verdicts, indent=1, ensure_ascii=False), encoding="utf-8")
    client.flush()
    return verdicts, finished


# --------------------------------------------------------------------------- monitoring

def metrics(client, dimensions, measures, filters=(), view="observations", since=timedelta(days=7),
            environment="default"):
    """One metrics-API query: `measures` as (measure, aggregation) pairs, grouped by `dimensions`, in one
    environment (stand-in runs are kept out). Rows as dicts."""
    now = datetime.now(timezone.utc)
    query = {"view": view, "dimensions": [{"field": d} for d in dimensions],
             "metrics": [{"measure": m, "aggregation": a} for m, a in measures],
             "filters": [*filters, {"column": "environment", "operator": "=", "value": environment, "type": "string"}],
             "fromTimestamp": (now - since).isoformat(), "toTimestamp": now.isoformat()}
    return client.api.metrics.metrics(query=json.dumps(query)).data


def tagged(version):
    """A filter: only what ran with this prompt version."""
    return {"column": "tags", "operator": "any of", "value": [f"prompt v{version}"], "type": "arrayOptions"}
