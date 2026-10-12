"""What you have to know about Mem0 2.2.1 before trusting it with a runner -- each one shown, offline, from Mem0
itself rather than from its documentation.

    adds_only       `add` never updates or deletes: a changed fact becomes a second memory beside the first
    dates           `timestamp` raises in the open-source build; the model is told today's date as the date of
                    the conversation, whatever the conversation's own date was
    prompt_size     the extraction prompt is too big for Groq's free tier on its own
    search          three scores, two of them quietly absent unless the right store and packages are there
    scoping         filters, not ids; get_all's silent limit of 20
    defaults        telemetry on, a shared history file, an embedder that ignores what it is embedding for
    links           links the model is asked for and the code throws away
"""

import inspect
import json
import re
import tempfile
from pathlib import Path

import coach_mem0 as M
import coach_stubs as T
import runners as R

GROQ_REQUEST_LIMIT = 8000


def offline_memory(sessions=None, root=None):
    """A real Mem0 -- Qdrant, spaCy, BM25 -- with the stand-in model and embedder, after the given check-ins."""
    root = Path(root or tempfile.mkdtemp())
    llm = T.FakeMemoryLLM()
    m = M.memory(root / "mem0", llm=llm, embedder=T.FakeEmbedder(), key="stub")
    for session in sessions or []:
        M.remember(m, session, f"Coach's reply for {session['id']}.")
    return m, llm


def _sessions(*ids):
    return [s for s in R.SESSIONS if s["id"] in ids]


def adds_only():
    """Maya's race moves: what Mem0 holds afterwards, and every event its history recorded."""
    from mem0 import Memory
    from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
    m, _ = offline_memory(_sessions("maya-1", "maya-3"))
    held = M.holdings(m, "maya")
    events = sorted({e["event"] for r in held for e in m.history(r["id"])})
    doc = " ".join(inspect.getdoc(Memory.add).split())
    return {"race memories held": [r["memory"] for r in held if "Riverside" in r["memory"]],
            "events in history": events,
            "add's docstring still says": doc[doc.index("infer (bool"):doc.index("If False")].strip(),
            "the prompt says": re.search(r"Your sole operation is ADD[^.]*", ADDITIVE_EXTRACTION_PROMPT).group(0)}


def dates():
    """What happens to a check-in's date on the way to Mem0's model."""
    m, llm = offline_memory(_sessions("maya-1"))
    try:
        m.add("x", user_id="maya", timestamp="2026-09-01")
        raised = "nothing"
    except ValueError as error:
        raised = str(error).splitlines()[0][:160]
    prompt = llm.prompts[0][-1]["content"]
    tail = prompt[prompt.index("## Observation Date"):].replace("\n\n", "\n").strip()
    return {"add(timestamp=...)": raised, "what the model is told": tail.splitlines()}


def prompt_size(llm=None):
    """Mem0's extraction prompt against Groq's free-tier ceiling, alone and as one real add sends it."""
    from chat import count_tokens
    from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
    if llm is None:
        _, llm = offline_memory(_sessions("maya-1", "maya-3", "maya-4"))
    biggest = max(sum(count_tokens(m["content"]) for m in p) for p in llm.prompts)
    return {"extraction prompt alone": count_tokens(ADDITIVE_EXTRACTION_PROMPT),
            "largest add request seen here": biggest,
            "plus Mem0's own output allowance (its default max_tokens)": biggest + 2000,
            "Groq's free-tier limit per request": GROQ_REQUEST_LIMIT}


def search():
    """Which of Mem0's three scores a local store actually gives -- Qdrant (with fastembed and spaCy) and Chroma."""
    from mem0.utils.entity_extraction import extract_entities
    m, _ = offline_memory(_sessions("maya-1", "maya-3"))
    hit = m.search("When is Maya's Riverside race?", filters={"user_id": "maya"}, explain=True)["results"][0]
    chroma_root = Path(tempfile.mkdtemp())
    from mem0 import Memory
    chroma = Memory.from_config({**M.config(chroma_root, "stub"),
                                 "vector_store": {"provider": "chroma",
                                                  "config": {"collection_name": "coach", "path": str(chroma_root)}}})
    return {"Qdrant, one result's parts": {k: round(v, 3) if isinstance(v, float) else v
                                           for k, v in hit["score_details"].items()},
            "Chroma's keyword search": chroma.vector_store.keyword_search("Riverside", top_k=5),
            "spaCy's entities in a question": extract_entities("Is Maya's Riverside Half Marathon on 1 November?")}


def scoping():
    """How one runner's memories are asked for -- and what get_all leaves out without saying so."""
    m, _ = offline_memory(_sessions("maya-1", "maya-3", "maya-4", "maya-5"))
    try:
        m.search("race", user_id="maya")
        top_level = "accepted"
    except ValueError as error:
        top_level = str(error)[:120]
    # 25 plain memories for one user, stored as given (infer=False, no model call)
    m.add([{"role": "user", "content": f"Training note number {i}."} for i in range(25)], user_id="many", infer=False)
    return {"search(user_id=...)": top_level,
            "memories stored for one user": len(m.get_all(filters={"user_id": "many"}, top_k=200)["results"]),
            "get_all with its default top_k": len(m.get_all(filters={"user_id": "many"})["results"]),
            "Tom's search, in a store holding only Maya": m.search("race", filters={"user_id": "tom"})["results"]}


def defaults():
    """Three defaults worth changing, read from Mem0's own code."""
    import mem0.configs.base as base
    import mem0.memory.telemetry as telemetry
    from mem0.embeddings.gemini import GoogleGenAIEmbedding
    body = inspect.getsource(GoogleGenAIEmbedding.embed).split('"""')[-1]
    return {"telemetry, unless MEM0_TELEMETRY says otherwise": "on" if 'os.environ.get("MEM0_TELEMETRY", "True")'
            in inspect.getsource(telemetry) else "off",
            "telemetry here": "on" if telemetry.MEM0_TELEMETRY else "off",
            "where the history database goes by default": [
                line.strip() for line in inspect.getsource(base).splitlines()
                if line.startswith("mem0_dir") or 'os.path.join(mem0_dir, "history.db")' in line],
            "Gemini embedder uses memory_action": "memory_action" in body}


def links():
    """The prompt asks the model to link each new memory to related old ones; does anything keep the links?"""
    from mem0 import Memory
    from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
    source = inspect.getsource(Memory._add_to_vector_store)
    return {"the prompt asks for linked_memory_ids": "linked_memory_ids" in ADDITIVE_EXTRACTION_PROMPT,
            "add reads them from the model's reply": 'mem.get("linked_memory_ids")' in source}


def isolation(done):
    """Across a whole run: did any memory recalled for one runner come from the other's check-ins?"""
    theirs = {s["date"]: s["runner"] for s in R.SESSIONS}
    crossed = [(sid, r["memory"]) for sid, rec in done.items() for r in rec.get("recalled", [])
               if theirs.get(r["session"]) not in (None, rec["runner"])]
    return {"recalled memories": sum(len(rec.get("recalled", [])) for rec in done.values()), "from the other runner": crossed}


def estimate():
    """Both builds offline over every check-in: how many Groq requests each makes and how big the largest is --
    what the live run will cost before it is spent."""
    import coach_scratch as S
    root = Path(tempfile.mkdtemp())
    with T.embeddings(), T.groq() as scratch_seen:
        S.run(root=root / "scratch", path=root / "scratch.json", log=lambda *_: None)
    m = M.memory(root / "mem0", llm=T.FakeMemoryLLM(), embedder=(embedder := T.FakeEmbedder()), key="stub")
    with T.groq() as mem0_seen:
        M.run(m, path=root / "mem0.json", log=lambda *_: None)
    return {"scratch coach": {"requests": len(scratch_seen), "largest": max(s["tokens"] for s in scratch_seen),
                              "sent in all": sum(s["tokens"] for s in scratch_seen)},
            "Mem0 coach": {"requests": len(mem0_seen), "largest": max(s["tokens"] for s in mem0_seen),
                           "sent in all": sum(s["tokens"] for s in mem0_seen)},
            "Mem0's model": {"requests": len(m.llm.llm.prompts)},
            "texts Mem0 embeds": len(set(embedder.texts))}


def show(result):
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))


def side_by_side(session_id, mem0, scratch):
    """One check-in: the runner's message, then each coach's reply with what the check points at in it."""
    import textwrap
    session = next(s for s in R.SESSIONS if s["id"] == session_id)
    print(f"{session_id}, {session['date']} -- {R.RUNNERS[session['runner']]}:")
    print(textwrap.fill(session["message"], 116, initial_indent="  ", subsequent_indent="  "))
    for name, done in (("Mem0", mem0), ("from scratch", scratch)):
        reply = done[session_id]["reply"]
        print(f"\n---- the coach with memory {name} ----\n{reply}\n")
        for key, items in R.leads(session_id, reply).items():
            if items:
                print(f"  {key}: {'; '.join(items)}")


def recall_report(done):
    """For each check-in: how many memories Mem0 brought back, and how alike the closest and the least alike were."""
    rows = []
    for sid, record in done.items():
        meaning = [r["parts"]["meaning"] for r in record.get("recalled", [])]
        rows.append({"check-in": sid, "recalled": len(meaning),
                     "most alike": max(meaning, default=None), "least alike": min(meaning, default=None)})
    return rows


def cost(mem0, scratch):
    """What each build spent: the coach on Groq, and Mem0's own model on Gemini."""
    def groq(done, key):
        usage = [rec["usage"] for rec in done.values() if "usage" in rec]
        return {"calls": sum(u["calls"] for u in usage), "prompt": sum(u["prompt"] for u in usage),
                "completion": sum(u.get("completion", 0) for u in usage)}
    calls = [c for rec in mem0.values() for c in rec.get("added", {}).get("memory model", [])]
    return {"the Mem0 coach, on Groq": groq(mem0, "usage"),
            "Mem0's model, on Gemini": {"calls": len(calls), "prompt": sum(c["prompt"] for c in calls),
                                        "output (thinking included)": sum(c["output"] for c in calls),
                                        "seconds": round(sum(c.get("seconds", 0) for c in calls))},
            "the scratch coach, on Groq": groq(scratch, "usage")}
