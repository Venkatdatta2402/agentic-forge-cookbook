"""05's cookbook assistant as a LangSmith experiment: a dataset, every answer traced, checks, and a judge's verdicts.

    ensure_dataset()   the ten questions as a LangSmith dataset: the question in, what a right answer says out
    start()            an experiment, named, before anything runs in it -- so a stopped run can be finished in it
    run()              answer every question the experiment has no saved answer for, through 05, traced: the
                       one-shot questions by its cited engine, the cross-chapter ones by its agent. Each answer is
                       saved locally the moment it exists, and the free checks are scored as it goes
    judge()            the claim judge over the saved answers (claim_judge.judge_all), verdicts saved one by one
    publish()          the verdicts into the experiment as feedback, once every answer has one
    results()          one row per question, from what was saved: the checks, the verdict, the cost

Answering and judging are separate on purpose. They spend two different quotas -- Groq for answers, Gemini for
verdicts -- and a judge that hits its rate limit must never mean answering again. Answers go to
`store/answers/<experiment>.json` before any judge sees them, verdicts to `store/verdicts/<experiment>.json`.
"""

import asyncio
import json
import re
import time

import eval_set as E            # puts 05_llamaindex on the path
import ask_index as A           # noqa: E402 -- 05's system, evaluated as it is
import claim_judge as J
import tracing as T

DATASET = "ask-the-cookbook"
STORE = E.HERE / "store"
BUDGETS = {"one-shot": 15_000, "agent": 45_000}       # tokens per question, as in 05


# --------------------------------------------------------------------------- the dataset

def examples(questions=None):
    """The questions as LangSmith examples: the question is the input, what a right answer says is the output."""
    return [{"inputs": {"question": q["question"], "kind": q["kind"]},
             "outputs": {"points": [point for point, _ in q["points"]], "files": q["files"],
                         "word": q.get("word")},
             "metadata": {"evidence": [f"{rel}:{first}-{last}" for rel, first, last in q["evidence"]]}}
            for q in (questions or E.QUESTIONS)]


def ensure_dataset(client, name=DATASET, questions=None):
    """The dataset, created if missing; an example whose points changed is updated, a new one added.

    LangSmith versions a dataset on every change, and an experiment records the version it ran against, so an
    answer key corrected later does not silently re-grade an old experiment.
    """
    wanted = {e["inputs"]["question"]: e for e in examples(questions)}
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description="Questions about agentic-forge-cookbook, with what a right "
                                                          "answer says and the lines it rests on.")
    have = {ex.inputs["question"]: ex for ex in client.list_examples(dataset_id=dataset.id)}
    new = [e for q, e in wanted.items() if q not in have]
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    changed = [(have[q], e) for q, e in wanted.items() if q in have and
               (have[q].outputs != e["outputs"] or have[q].inputs != e["inputs"])]
    for example, e in changed:
        client.update_example(example.id, inputs=e["inputs"], outputs=e["outputs"], metadata=e["metadata"])
    return dataset, {"added": len(new), "updated": len(changed), "unchanged": len(have) - len(changed)}


def start(client, name, description="", metadata=None, dataset=DATASET):
    """An experiment, created empty if it does not exist yet, against the dataset."""
    if client.has_project(name):
        return client.read_project(project_name=name)
    return client.create_project(name, description=description, metadata=metadata or {},
                                 reference_dataset_id=client.read_dataset(dataset_name=dataset).id)


# --------------------------------------------------------------------------- answering

def _path(kind, name):
    return STORE / kind / f"{name}.json"


def saved(name, kind="answers"):
    path = _path(kind, name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(name, question, output):
    answers = saved(name)
    answers[question] = output
    _path("answers", name).parent.mkdir(parents=True, exist_ok=True)
    _path("answers", name).write_text(json.dumps(answers, indent=1, ensure_ascii=False), encoding="utf-8")


def cited(answer):
    """The repo files an answer names, however it brackets them."""
    return sorted(set(re.findall(r"\d\d_[\w./-]+?\.(?:py|md|ipynb)", answer or "")))


class GroqDayLimit(RuntimeError):
    """Groq's daily token limit, hit mid-question. Not an answer, so nothing is saved: a later run retries it."""


def target(index, name, chat_model=A.chat_model):
    """The system under test, as one async function: 05's engine for a one-shot question, 05's agent for a
    cross-chapter one, traced, metered, and saved before it returns."""
    from langsmith.run_helpers import get_current_run_tree

    async def answer(inputs):
        question, kind = inputs["question"], inputs["kind"]
        llm = chat_model(budget=A.Budget(BUDGETS[kind], f"07 {kind}"), wrap=T.wrap_client)
        began = time.monotonic()
        if kind == "one-shot":
            engine = T.traced_engine(A.engine(index, llm))
            try:
                # sync inside, so on a thread; the thread is given the current context, and the trace with it
                result = await asyncio.to_thread(A.ask, engine, question)
            except Exception as error:       # noqa: BLE001 -- a budget or a refused request is a stop, kept
                result = {"answer": None, "sources": [], "stopped": f"{type(error).__name__}: {error}"}
            output = {"answer": result["answer"], "stopped": result.get("stopped"), "calls": [],
                      "cited": sorted({s["where"].split(" § ")[0] for s in result["sources"]})}
        else:
            run = await A.run_agent(A.agent(index, llm, wrap=T.traced_tool), question)
            output = {"answer": run["answer"], "stopped": run["stopped"], "calls": run["calls"],
                      "cited": cited(run["answer"])}
        if output["stopped"] and "tokens per day" in output["stopped"]:
            raise GroqDayLimit(output["stopped"])
        run_tree = get_current_run_tree()
        output.update(kind=kind, usage=llm.budget.summary(), seconds=round(time.monotonic() - began, 1),
                      run=str(run_tree.id) if run_tree else None, trace=str(run_tree.trace_id) if run_tree else None)
        _save(name, question, output)
        return output

    return answer


# The free checks, scored as each answer arrives. LangSmith passes an evaluator what it names: here the run's
# output, the example's expected output, and the input.

def answered(outputs, reference_outputs):
    return {"key": "answered", "score": int(bool(outputs.get("answer")) and not outputs.get("stopped"))}


def cites_an_answer_file(outputs, reference_outputs):
    return {"key": "cites an answer file", "score": int(bool(set(outputs.get("cited", [])) &
                                                              set(reference_outputs["files"])))}


def has_the_word(inputs, outputs, reference_outputs):
    if inputs["kind"] != "one-shot":
        return {"key": "has the word", "score": None}
    import cookbook as CB
    return {"key": "has the word", "score": int(CB.mentions(outputs.get("answer") or "", reference_outputs["word"]))}


CHECKS = [answered, cites_an_answer_file, has_the_word]


def to_answer(client, name, dataset=DATASET, kinds=("one-shot", "agent")):
    """The examples of these kinds that this experiment has no saved answer for."""
    done = saved(name)
    return [ex for ex in client.list_examples(dataset_name=dataset)
            if ex.inputs["question"] not in done and ex.inputs["kind"] in kinds]


async def run(client, index, name, dataset=DATASET, chat_model=A.chat_model, kinds=("one-shot", "agent")):
    """Answer what is missing, into the experiment, one question at a time (Groq's per-minute limit). `kinds` limits
    a round to the questions a change can affect."""
    from langsmith import aevaluate
    missing = to_answer(client, name, dataset, kinds)
    if not missing:
        return None
    return await aevaluate(target(index, name, chat_model=chat_model), data=missing, evaluators=CHECKS,
                           experiment=name, max_concurrency=1, client=client)


# --------------------------------------------------------------------------- judging, and publishing

def judge(name, kinds=("one-shot", "agent"), client=None, sleep=time.sleep, log=print):
    """Verdicts for the experiment's saved answers of these kinds that have none. Returns (verdicts, finished)."""
    answers = {q: a for q, a in saved(name).items() if a.get("kind", "one-shot") in kinds}
    return J.judge_all(answers, _path("verdicts", name), client=client, sleep=sleep, log=log)


def publish(client, name, kinds=("one-shot", "agent"), log=print):
    """The verdicts into the experiment as feedback, once every answer of these kinds has one. Once per experiment."""
    from langsmith import evaluate
    verdicts = saved(name, "verdicts")
    answers = {q for q, a in saved(name).items() if a.get("kind", "one-shot") in kinds}
    if not answers <= set(verdicts):
        log(f"    {len(answers - set(verdicts))} answers have no verdict yet: run judge() first")
        return False
    marker = _path("verdicts", name).with_suffix(".published")
    if marker.exists():
        log("    already published")
        return True
    questions = {q["question"]: q for q in E.QUESTIONS}

    def verdict(inputs):
        if inputs["question"] not in verdicts:
            return {"results": []}                  # a kind not judged in this experiment
        found = verdicts[inputs["question"]]
        scored = J.scores(found, questions[inputs["question"]])
        wrong = [f"{c['claim']} -- {c['why']}" for c in found.get("claims", []) if c["verdict"] == "contradicted"]
        return {"results": [{"key": key, "score": value, **({"comment": "\n".join(wrong)}
                                                              if key == "claims wrong" and wrong else {})}
                            for key, value in scored.items() if key != "claims"]}

    evaluate(name, evaluators=[verdict], client=client)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    return True


def traces(client, name):
    """What LangSmith recorded for each question of an experiment, read back from it, and kept in store/traces/:
    every model call (tokens in and out, seconds), every tool call, and what retrieval returned."""
    runs = list(client.list_runs(project_name=name))
    found = []
    for root in sorted((r for r in runs if r.parent_run_id is None), key=lambda r: r.start_time):
        steps = sorted((r for r in runs if r.trace_id == root.trace_id and r.id != root.id), key=lambda r: r.start_time)
        seconds = lambda r: round((r.end_time - r.start_time).total_seconds(), 1) if r.end_time else None
        found.append({
            "question": root.inputs.get("question"), "seconds": seconds(root), "trace": str(root.trace_id),
            "model calls": [{"in": r.prompt_tokens, "out": r.completion_tokens, "seconds": seconds(r)}
                            for r in steps if r.run_type == "llm"],
            "tool calls": [{"tool": r.name, "args": r.inputs, "returned chars": len(str((r.outputs or {}).get("output", "")))}
                           for r in steps if r.run_type == "tool"],
            "retrieved": [d["metadata"]["file"] for r in steps if r.run_type == "retriever"
                          for d in (r.outputs or {}).get("documents", [])]})
    _path("traces", name).parent.mkdir(parents=True, exist_ok=True)
    _path("traces", name).write_text(json.dumps(found, indent=1, ensure_ascii=False), encoding="utf-8")
    return found


def results(name):
    """One row per question, from the saved answers and verdicts: no LangSmith call, no model call."""
    import cookbook as CB
    answers, verdicts = saved(name), saved(name, "verdicts")
    rows = []
    for q in E.QUESTIONS:
        a = answers.get(q["question"])
        if a is None:
            continue
        v = verdicts.get(q["question"])
        row = {"question": q["question"], "kind": q["kind"], "answered": bool(a["answer"]) and not a["stopped"],
               "stopped": a["stopped"], "cites an answer file": bool(set(a["cited"]) & set(q["files"])),
               "has the word": CB.mentions(a["answer"] or "", q["word"]) if q["kind"] == "one-shot" else None,
               "tool calls": len(a["calls"]), "tokens": a["usage"]["total"],
               "largest request": a["usage"]["largest request"], "seconds": a["seconds"]}
        if v is not None:
            row.update(J.scores(v, q))
            row["wrong"] = [f"{c['claim']} -- {c['why']}" for c in v.get("claims", []) if c["verdict"] == "contradicted"]
        rows.append(row)
    return rows
