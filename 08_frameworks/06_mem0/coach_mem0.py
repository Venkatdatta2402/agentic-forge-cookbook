"""The running coach, with Mem0 as its memory.

Mem0's loop, as its documentation gives it: before replying, `search` the runner's memories with their message and
put what comes back in the prompt; reply; then `add` the exchange, and Mem0's own model decides what in it is worth
keeping. The coach never decides what to remember -- Mem0 does, after every check-in.

    the coach      gpt-oss-120b on Groq through chapter 1's client: the same model and persona as the scratch coach
    Mem0's model   Gemini flash-lite. Mem0's extraction prompt is about 7,600 tokens before a word of conversation
                   is added, and Groq's free tier refuses any request over 8,000 -- on Groq, every `add` would fail
    embeddings     gemini-embedding-001 at 768 numbers, through Mem0's own Gemini embedder
    the store      Qdrant on disk -- Mem0's default, and the one local store whose keyword search Mem0 uses (BM25,
                   from fastembed). Entities for the third score come from spaCy
    one runner     `user_id`. Every search is filtered by it; nothing else keeps two runners' memories apart
    dates          `add(timestamp=...)` raises in the open-source build, so the check-in's date goes in Mem0's
                   `prompt` (custom instructions), and in metadata so it can be shown
"""

import json
import os
import re
import time

import runners as R

# Read by Mem0 when it is imported, so set before that. Telemetry is ON by default and goes to PostHog; MEM0_DIR is
# where Mem0 keeps its anonymous id and, by default, the history database of every project on the machine.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("MEM0_DIR", str(R.STORE / "mem0" / "home"))

MEMORY_MODEL = "gemini-3.5-flash-lite"
MEMORY_TOKENS = 8000                     # Mem0's default is 2,000; the model's thinking counts against it
EMBED_MODEL = "models/gemini-embedding-001"
ROOT = R.STORE / "mem0"
ANSWERS = R.STORE / "answers" / "mem0.json"

DATED = ("This conversation took place on {date}. Resolve every relative date in it ('last Sunday', 'on Saturday') "
         "against {date} -- not against the Observation Date or the Current Date.")

# Round 2. In round 1 Mem0 stored the coach's own advice as memories ("Maya was recommended a 20 km long run on 27
# September"), nearly half of Maya's, and the coach later read a planned run as one she had done. Mem0's custom
# instructions are its own way to say what to keep.
RUNNER_ONLY = (" Extract only what the runner says: their facts, their plans, what they did and how it went. Do not "
               "extract the coach's recommendations, plans or advice -- they are in the coach's replies already, and "
               "a run the coach suggested is not a run that happened.")
ROUND2 = R.STORE / "mem0-round2"
ANSWERS2 = R.STORE / "answers" / "mem0.round2.json"


class MemoryQuotaSpent(RuntimeError):
    """Gemini's daily limit, which no waiting clears today."""


def config(root=ROOT, key=None):
    key = key or os.environ.get("GEMINI_API_KEY")
    return {
        "vector_store": {"provider": "qdrant",
                         "config": {"collection_name": "coach", "path": str(root / "qdrant"),
                                    "embedding_model_dims": 768, "on_disk": True}},
        # the key is passed, not left to Mem0: its Gemini classes fall back to GOOGLE_API_KEY, which on this machine
        # is a different key from the one in .env
        "llm": {"provider": "gemini", "config": {"model": MEMORY_MODEL, "api_key": key, "max_tokens": MEMORY_TOKENS}},
        "embedder": {"provider": "gemini", "config": {"model": EMBED_MODEL, "api_key": key, "embedding_dims": 768}},
        "history_db_path": str(root / "history.db"),
    }


def _patiently(call, *args, **kwargs):
    """Gemini's busy and per-minute refusals are waited out; its daily one stops the run."""
    from google.genai import errors
    for attempt in range(6):
        try:
            return call(*args, **kwargs)
        except errors.APIError as error:
            text = str(error)
            if error.code == 429 and "PerDay" in text:
                raise MemoryQuotaSpent(text[:300]) from error
            if error.code not in (429, 500, 503) or attempt == 5:
                raise
            delay = re.search(r"retryDelay'?:\s*'?(\d+)", text)
            time.sleep(int(delay.group(1)) + 1 if delay else 20 * (attempt + 1))


class Counted:
    """Mem0's model, with every call counted -- and Gemini's busy spells waited out rather than failing an add."""

    def __init__(self, llm):
        self.llm, self.calls = llm, []
        client = getattr(llm, "client", None)
        if client is not None and hasattr(client, "models"):
            real = client.models.generate_content

            def generate_content(*args, **kwargs):
                response = _patiently(real, *args, **kwargs)
                usage = getattr(response, "usage_metadata", None)
                if usage is not None and self.calls:
                    self.calls[-1].update(prompt=usage.prompt_token_count or 0,
                                          output=(usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
                return response

            client.models.generate_content = generate_content

    def generate_response(self, messages, **kwargs):
        self.calls.append({"prompt": 0, "output": 0})
        started = time.monotonic()
        try:
            return self.llm.generate_response(messages, **kwargs)
        finally:
            self.calls[-1]["seconds"] = round(time.monotonic() - started, 1)

    def __getattr__(self, name):
        return getattr(self.llm, name)


class Cached:
    """Mem0's embedder, with each text embedded once.

    Gemini returns the same numbers for the same text every time, so a kept vector is exactly what Mem0 would get
    again -- and the free tier's 1,000 texts a day go on new text only. Kept on disk, so a re-run costs nothing.
    Keyed on the text alone because Mem0's Gemini embedder ignores whether a text is being stored or searched for.
    """

    def __init__(self, embedder, path):
        self.embedder, self.path, self.embedded = embedder, path, 0
        self.vectors = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def embed(self, text, memory_action=None):
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(self, texts, memory_action="add"):
        missing = [t for t in dict.fromkeys(texts) if t not in self.vectors]
        if missing:
            vectors = _patiently(self.embedder.embed_batch, missing, memory_action)
            self.vectors.update({t: list(v) for t, v in zip(missing, vectors)})
            self.embedded += len(missing)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.vectors), encoding="utf-8")
        return [self.vectors[t] for t in texts]

    def __getattr__(self, name):
        return getattr(self.embedder, name)


def memory(root=ROOT, llm=None, embedder=None, key=None):
    """Mem0's `Memory` on this project's store, its model counted and its embeddings kept."""
    from mem0 import Memory
    root.mkdir(parents=True, exist_ok=True)
    m = Memory.from_config(config(root, key))
    m.llm = Counted(llm or m.llm)
    m.embedding_model = Cached(embedder or m.embedding_model, root / "embeddings.json")
    return m


def recall(m, session):
    """Mem0's search for this runner, with the check-in's own words as the query -- its defaults otherwise.

    `explain=True` changes nothing about what comes back; it keeps the three parts of each score. The score itself
    is (meaning + keywords + entities) divided by however many of those three fired for this query, so it is not a
    similarity, and two searches' scores are not on one scale. Mem0's 0.1 threshold is applied to the meaning part
    alone, before the other two are added."""
    found = m.search(session["message"], filters={"user_id": session["runner"]}, explain=True)["results"]
    return [{"memory": r["memory"], "score": round(r["score"], 3), "session": (r.get("metadata") or {}).get("session"),
             "parts": {k: round(r["score_details"][v], 3) for k, v in
                       (("meaning", "semantic_score"), ("keywords", "bm25_score"), ("entities", "entity_boost"))}}
            for r in found]


def system(session, recalled):
    """The coach's persona, and what Mem0 brought back -- each memory with the check-in it came from."""
    lines = [f"- ({r['session']}) {r['memory']}" for r in recalled] or ["- nothing yet"]
    return (R.COACH.format(date=session["date"], name=R.RUNNERS[session["runner"]]) +
            f"\n\nWhat you remember about {R.RUNNERS[session['runner']]} from earlier check-ins:\n" + "\n".join(lines))


def reply(system_text, message, model=R.COACH_MODEL):
    """One reply from the coach, on chapter 1's Groq client. Asked once more if the reasoning used every token."""
    import llm
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": message}]
    usage = {"calls": 0, "prompt": 0, "completion": 0}
    for _ in range(2):
        response = llm.client.chat.completions.create(model=model, messages=messages, max_tokens=R.REPLY_TOKENS)
        usage["calls"] += 1
        usage["prompt"] += response.usage.prompt_tokens
        usage["completion"] += response.usage.completion_tokens
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text, usage
    return "", usage


def remember(m, session, reply_text, instructions=""):
    """Mem0's `add` over the check-in, dated through custom instructions (plus any more). Returns what it stored."""
    exchange = [{"role": "user", "content": session["message"]}, {"role": "assistant", "content": reply_text}]
    calls = len(m.llm.calls)
    added = m.add(exchange, user_id=session["runner"], metadata={"session": session["date"]},
                  prompt=DATED.format(date=session["date"]) + instructions)["results"]
    return {"memories": [a["memory"] for a in added], "events": sorted({a["event"] for a in added}),
            "memory model": m.llm.calls[calls:]}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path, done):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")


def run(m, sessions=R.SESSIONS, path=ANSWERS, log=print, instructions=""):
    """Every check-in in order: recall, reply, remember -- each step saved as it finishes, so a run stopped by a quota
    carries on where it stopped and never asks the coach again for a reply it already has."""
    done = _load(path)
    for session in sessions:
        record = done.setdefault(session["id"], {"runner": session["runner"], "date": session["date"]})
        if "reply" not in record:
            record["recalled"] = recall(m, session)
            record["reply"], record["usage"] = reply(system(session, record["recalled"]), session["message"])
            _save(path, done)
        if "added" not in record:
            try:
                record["added"] = remember(m, session, record["reply"], instructions)
            except Exception as error:      # noqa: BLE001 -- Mem0 wraps the model's error in its own LLMError
                quota = isinstance(error.__cause__, MemoryQuotaSpent) or isinstance(error, MemoryQuotaSpent)
                log(f"  {session['id']}: add failed -- {'daily quota' if quota else type(error).__name__}: "
                    f"{str(error)[:200]}")
                return done, False
            _save(path, done)
        log(f"  {session['id']}: recalled {len(record['recalled'])}, stored {len(record['added']['memories'])}")
    return done, True


def holdings(m, runner):
    """Everything Mem0 holds for a runner, oldest check-in first. get_all's default stops at 20."""
    found = m.get_all(filters={"user_id": runner}, top_k=200)["results"]
    rows = [{"id": r["id"], "session": (r.get("metadata") or {}).get("session"), "memory": r["memory"]} for r in found]
    return sorted(rows, key=lambda r: (r["session"] or "", r["memory"]))
