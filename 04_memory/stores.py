"""Four stores, and how a memory ends up in one rather than another.

One thing decides it, and it is decided when a memory is *written*: **what will be asked of it
later**. The shape it takes follows from that.

    KeyValue   it will be looked up by name              a timezone, a project root
    Records    it will be filtered, ordered, counted     deploys, tickets, metrics
    Journal    it will be searched by literal and by
               resemblance                               what the agent worked out
    Graph      it will be walked                         who owns what, what calls what

`Records` and `Journal` describe the same world and are separated by what gets asked of them:
one is counted and ordered, the other is searched. Authorship is not the criterion, but it does
explain why the data turns up in the shape it does. A deploy pipeline emits rows on a schema
designed before the first deploy ever ran, so filtering and counting are cheap. The agent's
notes about *why* something broke could not have had a schema designed for them in advance, so
what is available is prose. Authorship constrains what a store can be asked cheaply; the asking
is still what decides.

Which index answers which question:

    the question                              where it goes
    ----------------------------------------  ---------------------------------------------
    "what is this user's timezone"            KeyValue. A name, and one value under it.
    "how many failed last week"               Records, SQL. No vector database has COUNT.
    "the most recent failed deploy"           Records, SQL: filter, order, limit.
    "where did PGRST204 come from"            Journal, FTS5 -- which is a boolean MATCH to
                                              select candidates and then BM25 to rank them,
                                              not "exact matching". See Journal.find_exact.
    "has this happened before?"               Journal, Chroma. A symptom names no field to
                                              filter on, and there is no WHERE clause for
                                              "resembles this".
    "...but only about billing, since August" Journal, Chroma: resemblance, restricted to rows
                                              whose metadata qualifies. Neither SQL nor bare
                                              vectors do this alone.
    "both at once, ranked together"           Journal.find_hybrid: reciprocal rank fusion
                                              over BM25 and cosine. Chroma does this
                                              natively, but not in the embedded build.
    "who else touched this service"           Graph. The answer is in no single entry.

The cost of two indexes over the journal is that a write reaches both, and so does a delete;
a delete that reaches one leaves an entry findable half the time. That is what `03_operations`
is about, and dual-writing to a search index is what production systems do here too.

One asymmetry to design around: Chroma's range operators take numbers only, so a date is stored
as `20260812` in the metadata while SQL keeps `"2026-08-12"`. `{"at": {"$gte": "2026-08-10"}}`
raises; `WHERE at >= '2026-08-10'` does not.

**Everything here persists** -- SQLite and Chroma to files, Neo4j to a server. A store that dies
with the process is a cache, not memory. Neo4j has no embedded mode, which makes it the one
thing in this repo needing a service up somewhere; that cost is part of choosing it.
"""

import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def _connect(path):
    # check_same_thread=False because a store is built in one thread and read from another as
    # soon as retrieval runs alongside anything else -- 05_context's gather() does exactly
    # that, and the default raised "SQLite objects created in a thread can only be used in that
    # same thread". Python's guard is stricter than SQLite itself: this build is compiled
    # serialized (`sqlite3.threadsafety == 3`), so one connection is safe to share.
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


# --------------------------------------------------------------------------------------
# 1. a key and a value


class KeyValue:
    """A dict that survives the process.

    This is the whole of long-term memory for most of what an agent needs to remember. A
    timezone, a preferred language, a project root: each is a fact with a name, and a name is
    something you can look up exactly. Reaching past this for anything cleverer makes the
    answer approximate in exchange for nothing.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data = {}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, key, value):
        self.data[key] = value
        self._flush()
        return self

    def read(self, key, default=None):
        return self.data.get(key, default)

    def forget(self, key):
        self.data.pop(key, None)
        self._flush()
        return self

    def _flush(self):
        if self.path:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def __len__(self):
        return len(self.data)


class Profile:
    """Facts with names, like `KeyValue` -- each replaced by its newest value, with the value it replaced kept
    beside it, and when each was written.

    Replacing is the point of a named fact: a race that moves leaves one race date, not two for a model to choose
    between. What plain replacing loses is the record that anything changed. Measured in `08_frameworks/06_mem0`:
    when a write missed a change, the old value was shown to the agent as a plain current fact, with nothing beside it
    to contradict it. So each fact keeps one step of history -- `was`, and `was_since` -- and `lines()` shows it.

    The fields are designed in advance, from what will be asked (a profile has a race, a goal, the days they run),
    and `add_field` lets more be added when something worth keeping fits none of them. Each field has an `about`,
    which is what a form asking for it is told it holds.
    """

    def __init__(self, path, fields=None):
        self.path = Path(path)
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.about = {**(fields or {}), **saved.get("fields", {})}
        self.facts = saved.get("facts", {})

    def fields(self):
        return dict(self.about)

    def add_field(self, name, about):
        self.about[name] = about
        self._flush()
        return self

    @staticmethod
    def same(a, b):
        """Two values that say the same thing: case, spacing and punctuation aside."""
        norm = lambda v: " ".join(re.findall(r"[a-z0-9:]+", (v or "").lower()))
        return norm(a) == norm(b)

    def write(self, name, value, at):
        """Replace a fact -- unless the value is the one already there. Measured in `06_mem0`: a form that re-wrote
        every field with its unchanged value reset every "since" date and pushed the real previous value out of
        `was`. Returns whether anything changed."""
        old = self.facts.get(name, {})
        if old and self.same(old.get("value"), value):
            return False
        history = old.get("history", []) + ([{"value": old["value"], "since": old["since"]}] if old else [])
        self.facts[name] = {"value": value, "since": at, "was": old.get("value"), "was_since": old.get("since"),
                            "history": history}
        self.about.setdefault(name, "")
        self._flush()
        return True

    def read(self, name, default=None):
        return self.facts.get(name, {}).get("value", default)

    def lines(self, before=True):
        """One line per fact: its value, since when, and -- with `before` -- what it said before, and since when."""
        out = []
        for name, f in self.facts.items():
            earlier = (f"; before: {f['was']}" + (f" (since {f['was_since']})" if f.get("was_since") else "")
                       if before and f.get("was") else "")
            out.append(f"- {name}: {f['value']} (since {f['since']}{earlier})")
        return out

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"fields": self.about, "facts": self.facts}, indent=2), encoding="utf-8")

    def __len__(self):
        return len(self.facts)


# --------------------------------------------------------------------------------------
# 2. structured records, written by something other than the agent


class Records:
    """Structured history that already exists, produced by other systems.

    Deploy logs, tickets, metrics, rows out of a CRM. The defining thing about them is not that
    they have columns -- it is that they get **filtered, ordered and counted**. Columns are what
    makes that cheap, and they exist because a process produced these rows on a schema somebody
    designed in advance -- by a process, not by the agent. When something in an agent system does
    write here, it is the harness recording a run, not the model recording a judgement.

    That is why this is relational and `Journal` below is not. A schema has to be decided before
    the first row exists, which is possible when a pipeline is emitting them and impossible when
    an agent has just noticed something.

    The questions this shape answers, and no other store here can:

        recent()  filter on fields, order by time, take n
        count()   aggregate. No vector database has COUNT, at any size

    Nothing is stripped out to make some other store look necessary: the `text` column holds
    whatever the process wrote, in full.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS records (
        id      INTEGER PRIMARY KEY,
        at      TEXT NOT NULL,          -- ISO 8601, so string order is time order
        kind    TEXT NOT NULL,
        status  TEXT NOT NULL,
        service TEXT NOT NULL,
        text    TEXT NOT NULL
    );
    """

    def __init__(self, path=":memory:"):
        self.db = _connect(path)
        self.db.executescript(self.SCHEMA)

    def write(self, at, kind, status, service, text):
        """Present because the notebook has to populate the table -- stand in for a feed."""
        cur = self.db.execute(
            "INSERT INTO records (at, kind, status, service, text) VALUES (?, ?, ?, ?, ?)",
            (at, kind, status, service, text),
        )
        self.db.commit()
        return cur.lastrowid

    def write_many(self, rows):
        for row in rows:
            self.write(*row)
        return self

    def recent(self, kind=None, status=None, service=None, since=None, limit=5):
        """Filter, order, limit -- written once here with parameters, not generated per request.

        A model filling in `kind` and `status` on a tool call is checked by `03_tools`'s
        `args_model()` before anything runs, and a bad argument fails loudly. A model writing
        the query itself can produce a wrong JOIN, and a wrong JOIN returns plausible rows
        rather than an error.
        """
        where, args = [], []
        for column, value in (("kind", kind), ("status", status), ("service", service)):
            if value:
                where.append(f"{column} = ?"); args.append(value)
        if since:
            where.append("at >= ?"); args.append(since)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        args.append(limit)
        return self.db.execute(
            f"SELECT * FROM records {clause} ORDER BY at DESC LIMIT ?", args
        ).fetchall()

    def count(self, **filters):
        where = " AND ".join(f"{k} = ?" for k in filters)
        clause = ("WHERE " + where) if where else ""
        return self.db.execute(
            f"SELECT COUNT(*) AS n FROM records {clause}", list(filters.values())
        ).fetchone()["n"]

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) AS n FROM records").fetchone()["n"]


# --------------------------------------------------------------------------------------
# 3. the agent's own write-ups, reachable by literal and by resemblance


_EMBED_CACHE = {}


def embed(texts, task="RETRIEVAL_DOCUMENT", dim=EMBED_DIM):
    """Turn text into vectors with Gemini.

    Groq serves no embedding model, so this is the one place in the repo that talks to a
    different provider. Deterministic: the same string returns bit-identical values every time,
    which is what makes the saved outputs in the notebooks worth reading.

    `task` matters more than it looks. The same sentence embeds differently depending on whether
    it is going into an index or being used to search one, and Gemini wants to be told which.

    Determinism also makes an in-process cache free of consequences, and it is not an
    optimisation for its own sake: `03_operations.ipynb` sweeps a handful of queries across
    several settings, which is ninety-odd calls for about a dozen distinct strings. Uncached
    that trips the free tier's per-minute limit; cached it is one call per string per session.
    """
    wanted = [t for t in texts if (task, dim, t) not in _EMBED_CACHE]
    if wanted:
        for i, vector in enumerate(_embed_uncached(wanted, task, dim)):
            _EMBED_CACHE[(task, dim, wanted[i])] = vector
    return [_EMBED_CACHE[(task, dim, t)] for t in texts]


def _embed_uncached(texts, task, dim):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Embeddings are the only part of this repo that need "
            "it -- see .env.example. Get one at https://aistudio.google.com/apikey"
        )
    body = {"requests": [
        {"model": f"models/{EMBED_MODEL}",
         "content": {"parts": [{"text": t}]},
         "taskType": task,
         "outputDimensionality": dim}
        for t in texts]}
    # Retried on 429, because the free tier's limit is per MINUTE and a notebook that
    # embeds in several cells will trip it. `llm.py` gets this for free -- the OpenAI SDK
    # is constructed with max_retries=8 -- but this is a raw HTTP call, so it is on us.
    # Honouring Retry-After when the API sends one; exponential backoff when it does not.
    for attempt in range(6):
        response = requests.post(
            f"{_ENDPOINT}/{EMBED_MODEL}:batchEmbedContents",
            headers={"x-goog-api-key": key}, json=body, timeout=60,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return [e["values"] for e in response.json()["embeddings"]]
        wait = float(response.headers.get("Retry-After", 0)) or min(2 ** attempt * 5, 60)
        time.sleep(wait)
    response.raise_for_status()
    return [e["values"] for e in response.json()["embeddings"]]


def cosine(a, b):
    """Plain Python. There is no numpy here, and 768 floats do not need it."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Journal:
    """What the agent itself worked out, in its own words.

    The counterpart to `Records`: same world, different author. A pipeline records *that* a
    release rolled back and how long it took. The agent's journal records what it concluded
    about *why* -- and nobody could have designed a schema for that in advance, which is exactly
    why it is prose with a little metadata rather than columns.

    Two indexes over the same entries, because two different questions get asked of them:

        find_exact()   a literal -- an error code, a path, a flag. FTS5.
        find_like()    a resemblance, optionally narrowed by metadata. Chroma.

    Both matter for agent write-ups specifically. An agent's note about a failure is largely
    *the failure string*: `PGRST204`, `max_connections`, `src/billing/reconcile.py`. Those are
    the parts a person searches for later and the parts dense vectors are worst at, because an
    embedding is built to collapse surface form into meaning and an identifier has no meaning to
    collapse to.

    Chroma reports **distance** (`1 - cosine` in cosine space); everything here converts back to
    similarity so a bigger number means a closer match.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entries (
        id     INTEGER PRIMARY KEY,
        at     TEXT NOT NULL,
        about  TEXT NOT NULL,           -- what it concerns
        source TEXT NOT NULL,           -- how the agent came by it
        text   TEXT NOT NULL
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
        USING fts5(text, content='entries', content_rowid='id');

    -- the distinct values a metadata field has ever taken, indexed for lexical lookup.
    -- Searching the entries would not do: `checkout-api` lives in a column, not in anybody's
    -- prose, so BM25 over `text` finds it never. See `candidates()`.
    CREATE TABLE IF NOT EXISTS field_values (
        field TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (field, value)
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS field_values_fts USING fts5(field, value);

    -- every time an entry was actually USED, not merely written. Ranking asks how recently
    -- and how often a memory has been reinforced, which is a different question from when
    -- it was authored and a different question again from whether it is still true.
    CREATE TABLE IF NOT EXISTS accesses (
        entry_id INTEGER NOT NULL,
        at       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS accesses_entry ON accesses(entry_id);
    """

    def __init__(self, path, summarize=None):
        """`summarize(text) -> str` is called on write, if two-stage recall needs it.

        It lives here rather than in the caller so the ordering below cannot be got wrong:
        the fallible step runs BEFORE the durable one, and both columns land in one
        transaction. Summarize after writing and a timeout leaves new text sitting behind an
        old summary -- a memory that describes itself incorrectly, which is worse than one
        with no summary at all.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.db = _connect(path / "journal.db")
        self.db.executescript(self.SCHEMA)
        # `pinned` arrives in 03_operations, after journals already exist on disk. A memory
        # store outlives the code that created it, so new columns are migrations rather than
        # edits to SCHEMA -- CREATE TABLE IF NOT EXISTS will not add a column to a live table.
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(entries)")}
        for column, spec in (("pinned", "INTEGER NOT NULL DEFAULT 0"), ("summary", "TEXT"),
                             ("written", "TEXT")):
            if column not in columns:
                self.db.execute(f"ALTER TABLE entries ADD COLUMN {column} {spec}")
        # rows that predate this column are backfilled from `at`, which is the best guess
        # available: before the distinction existed, the two were being conflated anyway.
        self.db.execute("UPDATE entries SET written = at WHERE written IS NULL")
        self.db.commit()
        self.chroma = chromadb.PersistentClient(
            path=str(path / "chroma"),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        # cosine, not Chroma's default squared L2 -- these embeddings are not unit-normalised,
        # so L2 would rank partly by vector length rather than by direction
        self.vectors = self.chroma.get_or_create_collection(
            "journal", configuration={"hnsw": {"space": "cosine"}}
        )
        self.summarize = summarize

    # -- writing, which has to reach both indexes ---------------------------------------

    def write(self, entries, now=None):
        """Add entries, indexing each one both ways, with a single embedding call.

        `now` is when the row is being created, as opposed to `at`, which is what the entry is
        about. Writing a note today about an incident in July gives `at="2026-07-02"` and
        `now="2026-12-01"`: you would filter it under July and it is a brand new memory.
        Defaults to `at` when not given, which is right whenever the agent writes things down
        as they happen.
        """
        # every summary first: a model call that fails should leave the journal untouched
        # rather than half-written. Only then does anything reach the database.
        summaries = [self.summarize(text) if self.summarize else None
                     for _, _, _, text in entries]

        ids, texts, metas = [], [], []
        for (at, about, source, text), summary in zip(entries, summaries):
            cur = self.db.execute(
                "INSERT INTO entries (at, about, source, text, summary, written) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (at, about, source, text, summary, now or at),
            )
            self.db.execute(
                "INSERT INTO entries_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, text)
            )
            ids.append(str(cur.lastrowid))
            texts.append(text)
            # `at` becomes an integer because Chroma will not compare strings with $gte
            metas.append({"at": int(at[:10].replace("-", "")), "about": about, "source": source})
            for field, value in (("about", about), ("source", source)):
                added = self.db.execute(
                    "INSERT OR IGNORE INTO field_values (field, value) VALUES (?, ?)",
                    (field, value),
                )
                if added.rowcount:
                    self.db.execute(
                        "INSERT INTO field_values_fts (field, value) VALUES (?, ?)",
                        (field, value),
                    )
        self.db.commit()
        self.vectors.add(ids=ids, documents=texts,
                         embeddings=embed(texts, task="RETRIEVAL_DOCUMENT"),
                         metadatas=metas)
        return [int(i) for i in ids]

    def update(self, entry_id, text, now=None):
        """Change an entry's text. A vector cannot be edited, only replaced.

        Rewriting a memory makes it current, so `written` moves to `now` and its decay starts
        again from zero. `at` does not move: a corrected note about the second of July is still
        a note about the second of July, and still filters under it.

        There is deliberately no setter for `summary` on its own. A summary that can be
        written independently is a summary that can drift from what it describes, and the
        drift is silent -- two-stage recall keeps working, just on a description of something
        the entry no longer says.
        """
        summary = self.summarize(text) if self.summarize else None
        row = self.db.execute("SELECT at FROM entries WHERE id = ?", (entry_id,)).fetchone()
        self.db.execute("UPDATE entries SET text = ?, summary = ?, written = ? WHERE id = ?",
                        (text, summary, now or (row["at"] if row else None), entry_id))
        self.db.execute("INSERT INTO entries_fts (entries_fts) VALUES ('rebuild')")
        self.db.commit()
        self.vectors.update(ids=[str(entry_id)], documents=[text],
                            embeddings=embed([text], task="RETRIEVAL_DOCUMENT"))
        return self

    def forget(self, entry_id):
        """Delete from both indexes. Missing one leaves an entry findable half the time."""
        self.db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self.db.execute("INSERT INTO entries_fts (entries_fts) VALUES ('rebuild')")
        self.db.commit()
        self.vectors.delete(ids=[str(entry_id)])
        return self

    # -- reading ------------------------------------------------------------------------

    def find_exact(self, term, limit=5):
        """Literal search, via FTS5 -- which is two stages, not one.

        **BM25 is the ranking algorithm; FTS5 is the engine that implements it.** Calling this
        "exact match" is half the story:

          1. `MATCH` is boolean, over an inverted index. A row either contains the token or it
             does not, and rows that do not are gone. That stage is genuinely 1/0, and it is
             what lets a search for something absent return *nothing* -- an answer no vector
             index can give.
          2. `bm25()` then scores the survivors on a continuous scale. SQLite exposes it as the
             hidden `rank` column, and the values are **negative**, so `ORDER BY rank` ascending
             puts the best match first. Measured on a four-row table searching PGRST204:

                 -1.8690e-06   3 occurrences,  7 words
                 -1.5725e-06   1 occurrence,   3 words
                 -5.0708e-07   1 occurrence,  92 words

             All three BM25 properties are visible there: term frequency (3 beats 1),
             saturation (3x the occurrences buys only ~1.19x the score, not 3x), and length
             normalisation (the same single occurrence scores ~3x worse in a longer row).

        Contrast Chroma's `where_document={"$contains": ...}`, which is substring matching over
        raw text and has no ranking at all:

            query        FTS5                    Chroma $contains
            PGRST204     1 row                   1 row
            PGRST20      0 -- a different code   1 -- matches as a prefix
            pgrst204     1 -- case-insensitive   0 -- misses it

        FTS5 tokenises; `$contains` looks for characters. For a question whose whole point is
        that a near-miss identifier means something else, that decides it.
        """
        return self.db.execute(
            "SELECT e.* FROM entries_fts f JOIN entries e ON e.id = f.rowid "
            "WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?",
            (term, limit),
        ).fetchall()

    def find_like(self, query, k=3, where=None):
        """Resemblance, optionally narrowed by metadata. No 'not found' -- k results, always.

        `where` filters *before* ranking, which is the combination neither of the others gives:
        "entries that read like this, but only about billing-api, and only since August".
        Narrowing first also means the k results come from the subset, rather than being
        whatever survives filtering a global top-k.
        """
        qv = embed([query], task="RETRIEVAL_QUERY")[0]
        got = self.vectors.query(
            query_embeddings=[qv], n_results=min(k, max(len(self), 1)), where=where
        )
        return [(1.0 - dist, int(i), doc)
                for i, doc, dist in zip(got["ids"][0], got["documents"][0], got["distances"][0])]

    # FTS5 does no stopword removal, so OR-ing every word of a question into a MATCH ranks
    # partly by how often an entry says "the". Measured: "what happened with the rollback"
    # returned the migration entry as its top BM25 hit, matching on `the` -- the entries say
    # "rolled back", which is not the token `rollback` at all. A lexical index expects search
    # terms, not a sentence.
    STOPWORDS = frozenset(
        "a an and are as at be but by did do for from had has have how in is it of on or "
        "that the this to was were what when where which who why with about during".split()
    )

    @classmethod
    def _fts_query(cls, text):
        words = [w.strip("?,.'\"();:") for w in text.lower().split()]
        return " OR ".join(w for w in words if len(w) > 2 and w not in cls.STOPWORDS)

    def find_hybrid(self, query, k=3, rrf_k=60):
        """Both indexes, fused by rank -- BM25 and cosine together, the usual "hybrid search".

        Fused on **ranks** rather than scores, because the two scores share no scale and never
        will: FTS5's BM25 is negative and corpus-dependent, cosine sits in a narrow band off
        zero. Normalising them into comparability runs into the trap `03_operations` covers,
        where min-max promotes the least-bad of a bad batch to 1.0. Ranks have no such failure:

            score(d) = sum over each index of  1 / (rrf_k + rank of d in that index)

        `rrf_k` (conventionally 60) flattens the gap between the top few positions, so one index
        being confidently wrong cannot dominate the other.

        This generalises what an exact-first rule cannot. Answering `PGRST204` by trying FTS5
        and falling back to vectors works *because that query is a label* and you can tell by
        looking. An ordinary question has no such tell, and exact-first would discard the vector
        ranking whenever any single word happened to match.

        Chroma implements this natively -- `Search().rank(Rrf([Knn(...), Knn(...)]))` over a
        BM25 sparse index -- but `Search` raises `NotImplementedError: Search is not implemented
        for Local Chroma`. It works on distributed and hosted Chroma only, which is why the
        fusion is done here by hand.
        """
        lexical = self._fts_query(query)
        try:
            lex_ids = [row["id"] for row in self.find_exact(lexical, limit=10)] if lexical else []
        except sqlite3.OperationalError:
            lex_ids = []          # a query that is all stopwords, or all punctuation
        vec_ids = [i for _s, i, _t in self.find_like(query, k=len(self))]

        fused = {}
        for ranking in (lex_ids, vec_ids):
            for rank, entry_id in enumerate(ranking):
                fused[entry_id] = fused.get(entry_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(score, eid, self.get(eid)["text"]) for eid, score in best], lex_ids, vec_ids

    def recent(self, about=None, source=None, limit=5):
        """The journal has fields too -- they just were not designed before the first entry."""
        where, args = [], []
        for column, value in (("about", about), ("source", source)):
            if value:
                where.append(f"{column} = ?"); args.append(value)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        args.append(limit)
        return self.db.execute(
            f"SELECT * FROM entries {clause} ORDER BY at DESC LIMIT ?", args
        ).fetchall()

    def count(self, **filters):
        where = " AND ".join(f"{k} = ?" for k in filters)
        clause = ("WHERE " + where) if where else ""
        return self.db.execute(
            f"SELECT COUNT(*) AS n FROM entries {clause}", list(filters.values())
        ).fetchone()["n"]

    def values(self, field):
        """Every value a field has taken. Fine to enumerate when there are a handful."""
        return [r["value"] for r in self.db.execute(
            "SELECT value FROM field_values WHERE field = ? ORDER BY value", (field,))]

    def candidates(self, field, text, keep=0.6, limit=20):
        """Which values of `field` a question could plausibly be naming.

        This exists for the case a fixed list cannot cover. When a field has three possible
        values you put them in the type and the model sees all three. When it has forty
        thousand -- every customer, every repository, every SKU -- you cannot, and a model given
        no list will confidently invent one that looks right.

        **Two lexical searches, unioned, because they miss different things.**

        *The field's vocabulary* -- the distinct strings `about` has ever held. Catches a
        question that names the value: "anything about billing" finds `billing-api`. It is the
        only thing that can, since `checkout-api` lives in a column and appears in nobody's
        prose.

        *The entries* -- take the values attached to entries that rank well. Catches a question
        that names the subject without naming the value: "what happened with the schema cache?"
        matches no service name at all, but the entry it hits belongs to `billing-api`.

        Measured on six entries, the two overlap almost nowhere: vocabulary answers the first
        two questions and nothing else, entries answer the last two and nothing else. Either
        alone leaves half the questions unfilterable.

        The cutoff is a *fraction* of what matched rather than a fixed count or a score
        threshold. A count is arbitrary -- five is too few when a question names six services
        and too many when it names one. A threshold does not transfer, because BM25 scores
        depend on the corpus, so a number tuned against eight values is meaningless against
        eight thousand.

        Set generously, and 0.6 rather than tighter for a specific reason: below 0.5 a two-way
        tie gets halved. Asked to "compare billing and checkout" both score identically, and
        `ceil(2 * 0.3) == 1` drops one of the two the question named.

        Returns [] when nothing matched, meaning *do not filter on this field* rather than
        *filter on my best guess*.
        """
        terms = self._fts_query(text)
        if not terms:
            return []
        match = f"value : ({terms})"

        # 1. values whose own text matches
        try:
            rows = self.db.execute(
                "SELECT value FROM field_values_fts "
                "WHERE field_values_fts MATCH ? AND field = ? ORDER BY rank LIMIT ?",
                (match, field, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        from_values = list(dict.fromkeys(r["value"] for r in rows))
        from_values = from_values[:max(1, math.ceil(len(from_values) * keep))] if from_values else []

        # 2. values carried by entries that match
        try:
            hits = self.db.execute(
                "SELECT e.* FROM entries_fts f JOIN entries e ON e.id = f.rowid "
                "WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?",
                (terms, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            hits = []
        from_entries = list(dict.fromkeys(
            h[field] for h in hits if field in h.keys() and h[field] is not None))

        return list(dict.fromkeys(from_values + from_entries))

    # -- what has actually been used -----------------------------------------------------

    def pin(self, entry_id, pinned=True):
        """Exempt an entry from decay-driven deletion.

        `lambda(f)` protects memories by *use*, which is the right default and has one hole in
        it: something read once, long ago, that still matters -- a policy, a postmortem, the
        reason a decision went the way it did. It has no usage to earn a longer half-life and
        no way to acquire one, so it ages out with the noise.

        A boolean set deliberately covers that, and is the reason this chapter asks no model to
        score importance. Measured elsewhere, a model handed "rate this memory 1-10" returns 6
        to 8 for nearly everything: a call per write, and a number that moves no ranking.
        """
        self.db.execute("UPDATE entries SET pinned = ? WHERE id = ?",
                        (1 if pinned else 0, entry_id))
        self.db.commit()
        return self

    def is_pinned(self, entry_id):
        row = self.db.execute("SELECT pinned FROM entries WHERE id = ?", (entry_id,)).fetchone()
        return bool(row["pinned"]) if row else False

    def used(self, entry_ids, at):
        """Record that these entries were used. Called by whatever consumed them, not by search.

        Appearing in a top-k is not use. Being read is. Keeping the two apart is the whole
        value of this table: a memory that keeps surfacing and keeps being ignored should not
        be reinforced by its own popularity.
        """
        self.db.executemany("INSERT INTO accesses (entry_id, at) VALUES (?, ?)",
                            [(int(i), at) for i in entry_ids])
        self.db.commit()
        return self

    def accesses(self, entry_id):
        """Every recorded use of one entry, oldest first."""
        return [r["at"] for r in self.db.execute(
            "SELECT at FROM accesses WHERE entry_id = ? ORDER BY at", (entry_id,))]

    def get(self, entry_id):
        return self.db.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()

    def entries(self):
        """Every entry's id, date and pin -- what a forgetting sweep reads, without knowing how it is stored."""
        return self.db.execute("SELECT id, at, pinned FROM entries").fetchall()

    def meta(self, entry_id):
        got = self.vectors.get(ids=[str(entry_id)])
        return got["metadatas"][0] if got["ids"] else None

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]


# --------------------------------------------------------------------------------------
# 3b. the same journal, on one engine


_QDRANT = {}          # one client per folder: local Qdrant locks its folder against a second client
_BM25 = []


def _bm25():
    if not _BM25:
        from fastembed import SparseTextEmbedding
        _BM25.append(SparseTextEmbedding(model_name="Qdrant/bm25"))
    return _BM25[0]


class QdrantJournal:
    """`Journal`'s methods, on Qdrant alone: one record per entry, holding the text, its meaning vector and its
    keyword vector together.

    `Journal` runs two engines because local Chroma has no keyword search and no fusion -- so every write, update
    and delete has to reach both, and one that reaches only one leaves an entry findable half the time. Qdrant's
    local mode has both searches and fuses them itself, so each of those is a single operation on a single record.
    What each method answers is unchanged, which is why `Recall`, `Retrieve` and the add/update/ignore check run on
    either journal:

        find_exact   BM25 over a sparse keyword vector (fastembed's `Qdrant/bm25`, with Qdrant weighting each word
                     by its rarity). Like FTS5 it can return NOTHING: `PGRST20` does not find `PGRST204`, case is
                     ignored. Unlike FTS5 it STEMS: "batches" finds "batch", "locked" finds "lock" -- better for
                     prose, and the one place the two journals answer differently. FTS5's query syntax (AND,
                     phrases) is not supported: the words of `term` are matched as any-of
        find_like    cosine over Gemini embeddings (`embed`, with the same task types), filtered BEFORE ranking.
                     `where` takes the Chroma form `pipeline.py` writes, so a pipeline step can use either journal
        find_hybrid  reciprocal rank fusion, done by Qdrant -- with `rrf_k` passed, because its own default is 2
        values       the values a field has taken, from Qdrant's facets
        used         when an entry was used, kept on the entry itself

    Dates are stored twice, as `at` ("2026-08-12", what the rows of `Journal` hold) and `day` (20260812), because
    ordering and ranges want a number -- the same split `Journal` makes between SQLite and Chroma.

    Local mode has two limits worth knowing. Only one client may open a folder at a time (so clients are shared
    per folder here). And filters scan every entry rather than using an index, which is nothing at a few thousand
    entries; past that, Qdrant runs as a server.
    """

    STOPWORDS = Journal.STOPWORDS
    _fts_query = Journal._fts_query

    # A note can say which profile fields it depends on -- the fifth, optional element of an entry -- and `retire`
    # takes every note depending on a field out of every search when that field changes. Measured in `06_mem0`: a
    # note saying "runs Tuesday, Thursday, Saturday and Sunday evenings" kept coming back weeks after both had
    # changed, because nothing replaces a note. Retired notes stay stored; `live=False` reaches them.

    def __init__(self, path, summarize=None):
        from qdrant_client import QdrantClient, models
        self.models = models
        folder = str((Path(path) / "qdrant").resolve())
        Path(folder).mkdir(parents=True, exist_ok=True)
        if folder not in _QDRANT:
            _QDRANT[folder] = QdrantClient(path=folder)
        self.client, self.name, self.summarize = _QDRANT[folder], "journal", summarize
        if not self.client.collection_exists(self.name):
            self.client.create_collection(
                self.name,
                vectors_config={"dense": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
        ids = [p.id for p in self._all()]
        self._next = max(ids, default=0) + 1

    # -- the pieces every method shares ------------------------------------------------

    def _all(self, flt=None):
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(self.name, scroll_filter=flt, limit=256, offset=offset,
                                               with_payload=True, with_vectors=False)
            points += batch
            if offset is None:
                return points

    def _sparse(self, text, query=False):
        encoder = _bm25()
        got = next(iter(encoder.query_embed(text) if query else encoder.embed([text])))
        return self.models.SparseVector(indices=got.indices.tolist(), values=got.values.tolist())

    def _point(self, entry_id, payload, text):
        return self.models.PointStruct(
            id=entry_id, payload=payload,
            vector={"dense": embed([text], task="RETRIEVAL_DOCUMENT")[0], "bm25": self._sparse(text)})

    @staticmethod
    def _row(point):
        return {"id": point.id, **{k: v for k, v in point.payload.items() if k not in ("day", "accesses")}}

    def _filter(self, where=None, live=True, **equal):
        """Chroma's `where` (as `pipeline._chroma_where` writes it) and plain equalities, as a Qdrant filter --
        leaving out retired notes unless `live` is False."""
        m = self.models
        must = [m.FieldCondition(key=k, match=m.MatchValue(value=v)) for k, v in equal.items() if v is not None]
        clauses = list(where.get("$and", [where])) if where else []
        for clause in clauses:
            for key, cond in clause.items():
                key = "day" if key == "at" else key
                if not isinstance(cond, dict):
                    must.append(m.FieldCondition(key=key, match=m.MatchValue(value=cond)))
                elif "$in" in cond:
                    must.append(m.FieldCondition(key=key, match=m.MatchAny(any=list(cond["$in"]))))
                else:
                    bounds = {op.strip("$"): v for op, v in cond.items()}
                    must.append(m.FieldCondition(key=key, range=m.Range(**bounds)))
        retired = [m.FieldCondition(key="retired", match=m.MatchValue(value=True))] if live else []
        return m.Filter(must=must, must_not=retired) if must or retired else None

    # -- writing: one record, so nothing can reach one index and miss the other -----------

    def write(self, entries, now=None):
        """Entries are (at, about, source, text), or with a fifth element: the profile fields the entry depends on."""
        summaries = [self.summarize(entry[3]) if self.summarize else None for entry in entries]
        points = []
        for entry, summary in zip(entries, summaries):
            at, about, source, text = entry[:4]
            payload = {"at": at, "day": int(at[:10].replace("-", "")), "about": about, "source": source,
                       "text": text, "summary": summary, "written": now or at, "pinned": 0, "accesses": [],
                       "depends_on": list(entry[4]) if len(entry) > 4 else [], "retired": False}
            points.append(self._point(self._next, payload, text))
            self._next += 1
        self.client.upsert(self.name, points)
        return [p.id for p in points]

    def update(self, entry_id, text, now=None):
        """Text, summary and both vectors replaced in one write. `at` stays; `written` moves to `now`."""
        summary = self.summarize(text) if self.summarize else None
        old = self.client.retrieve(self.name, [entry_id], with_payload=True)
        if not old:
            return self
        payload = {**old[0].payload, "text": text, "summary": summary, "written": now or old[0].payload["at"]}
        self.client.upsert(self.name, [self._point(entry_id, payload, text)])
        return self

    def retire(self, field, at):
        """Take every entry depending on `field` out of the searches, because the field has changed. Returns their
        ids. The entries stay stored, with when they were retired."""
        m = self.models
        hit = self._all(m.Filter(must=[m.FieldCondition(key="depends_on", match=m.MatchAny(any=[field]))],
                                 must_not=[m.FieldCondition(key="retired", match=m.MatchValue(value=True))]))
        for point in hit:
            self._set(point.id, retired=True, retired_on=at)
        return [p.id for p in hit]

    def forget(self, entry_id):
        self.client.delete(self.name, points_selector=self.models.PointIdsList(points=[entry_id]))
        return self

    # -- reading -----------------------------------------------------------------------

    def find_exact(self, term, limit=5):
        """Entries containing any word of `term`, best BM25 first; [] when none does."""
        got = self.client.query_points(self.name, query=self._sparse(term, query=True), using="bm25",
                                       query_filter=self._filter(), limit=limit, with_payload=True)
        return [self._row(p) for p in got.points]

    def find_like(self, query, k=3, where=None):
        """[(similarity, id, text)] -- k results, filtered first. No 'not found' here; `Recall`'s floor gives that."""
        got = self.client.query_points(self.name, query=embed([query], task="RETRIEVAL_QUERY")[0], using="dense",
                                       query_filter=self._filter(where), limit=k, with_payload=True)
        return [(p.score, p.id, p.payload["text"]) for p in got.points]

    def find_hybrid(self, query, k=3, rrf_k=60):
        """Qdrant's own reciprocal rank fusion, given `Journal`'s `rrf_k`. Left to itself Qdrant uses k=2, which
        does the opposite of what 60 is for: the top place in either ranking all but decides the result. Returns
        what `Journal.find_hybrid` does: the fused results, and each index's ranking on its own."""
        m = self.models
        dense, sparse = embed([query], task="RETRIEVAL_QUERY")[0], self._sparse(query, query=True)
        live = self._filter()
        fused = self.client.query_points(self.name, prefetch=[m.Prefetch(query=dense, using="dense", limit=len(self),
                                                                         filter=live),
                                                              m.Prefetch(query=sparse, using="bm25", limit=10,
                                                                         filter=live)],
                                         query=m.RrfQuery(rrf=m.Rrf(k=rrf_k)), limit=k, with_payload=True)
        lex_ids = [p.id for p in self.client.query_points(self.name, query=sparse, using="bm25", query_filter=live,
                                                             limit=10).points]
        vec_ids = [p.id for p in self.client.query_points(self.name, query=dense, using="dense", query_filter=live,
                                                             limit=len(self)).points]
        return [(p.score, p.id, p.payload["text"]) for p in fused.points], lex_ids, vec_ids

    def recent(self, about=None, source=None, limit=5, live=True):
        got, _ = self.client.scroll(self.name, scroll_filter=self._filter(live=live, about=about, source=source),
                                    limit=limit,
                                    order_by=self.models.OrderBy(key="day", direction="desc"), with_payload=True)
        return [self._row(p) for p in got]

    def count(self, **filters):
        return self.client.count(self.name, count_filter=self._filter(**filters)).count

    def values(self, field):
        return sorted(h.value for h in self.client.facet(self.name, key=field, limit=10000).hits)

    def candidates(self, field, text, keep=0.6, limit=20):
        """`Journal.candidates`, with the same two sources unioned: values whose own words the question uses,
        and the values carried by entries the question matches."""
        terms = set(self._fts_query(text).split(" OR ")) - {""}
        if not terms:
            return []
        scored = [(len(terms & set(re.findall(r"[a-z0-9]+", v.lower()))), v) for v in self.values(field)]
        from_values = [v for n, v in sorted(scored, key=lambda s: -s[0]) if n]
        from_values = from_values[:max(1, math.ceil(len(from_values) * keep))] if from_values else []
        from_entries = [r[field] for r in self.find_exact(" ".join(terms), limit=limit) if r.get(field) is not None]
        return list(dict.fromkeys(from_values + from_entries))

    # -- what has been used -------------------------------------------------------------

    def _set(self, entry_id, **payload):
        self.client.set_payload(self.name, payload=payload, points=[entry_id])

    def pin(self, entry_id, pinned=True):
        self._set(entry_id, pinned=1 if pinned else 0)
        return self

    def is_pinned(self, entry_id):
        row = self.get(entry_id)
        return bool(row["pinned"]) if row else False

    def used(self, entry_ids, at):
        for entry_id in entry_ids:
            got = self.client.retrieve(self.name, [int(entry_id)], with_payload=True)
            if got:
                self._set(int(entry_id), accesses=[*got[0].payload.get("accesses", []), at])
        return self

    def accesses(self, entry_id):
        got = self.client.retrieve(self.name, [entry_id], with_payload=True)
        return sorted(got[0].payload.get("accesses", [])) if got else []

    def get(self, entry_id):
        got = self.client.retrieve(self.name, [entry_id], with_payload=True)
        return self._row(got[0]) if got else None

    def entries(self):
        return [{"id": p.id, "at": p.payload["at"], "pinned": p.payload["pinned"]} for p in self._all()]

    def meta(self, entry_id):
        row = self.get(entry_id)
        return {"at": int(row["at"][:10].replace("-", "")), "about": row["about"], "source": row["source"]} if row else None

    def __len__(self):
        return self.client.count(self.name).count


# --------------------------------------------------------------------------------------
# 4. relationships, when the answer is not in any single memory


class Graph:
    """Facts as edges between things, so a question can walk from one to another, on Neo4j.

    Needs a running server -- there is no embedded mode. `NEO4J_URI`, `NEO4J_USER` and
    `NEO4J_PASSWORD` come from `.env`; an AuraDB Free instance is the cheapest way to get one.
    This is the only notebook in the repo that will not run without a service somewhere else,
    which is a real cost and worth knowing before choosing a graph database for anything.

    What it buys is the traversal. `connected()` and `path()` below are single Cypher queries
    with a variable-length pattern -- `-[:REL*1..2]-` -- instead of a breadth-first loop in
    Python pulling one ring of neighbours at a time. At five edges that is a stylistic
    difference. At five million it is the whole ballgame, because the walk happens next to the
    data rather than across a network for every hop.

    Edges carry `since`/`until` instead of being deleted, because "used to own this service" is
    a fact with an end date, not a fact that was ever wrong. Dropping it loses real history.

    **One relationship type, `REL`, with the real name on a property.** Cypher cannot
    parameterize a relationship type -- `-[:$type]-` is not valid -- so typed relationships
    would mean building query strings out of caller input, which is how you get a Cypher
    injection. A property is parameterizable and safe. Real schemas do use typed relationships;
    they also know their types at authoring time, which this does not.
    """

    # database=None means "whatever this server calls home", and that is deliberate. Almost
    # every Neo4j example hardcodes "neo4j", which is right for a self-hosted server and wrong
    # on Aura, where the database is named after the instance -- `f9fdfff8` rather than `neo4j`.
    # Hardcoding it fails with DatabaseNotFound while the credentials are perfectly correct,
    # and the error says the database does not exist rather than that you asked for the wrong one.
    def __init__(self, uri=None, user=None, password=None, database=None, reset=False):
        uri = uri or os.environ.get("NEO4J_URI")
        user = user or os.environ.get("NEO4J_USER", "neo4j")
        password = password or os.environ.get("NEO4J_PASSWORD")
        if not uri or not password:
            raise RuntimeError(
                "NEO4J_URI and NEO4J_PASSWORD are not set. Unlike every other store here, "
                "Neo4j needs a server -- create a free one at https://console.neo4j.io and "
                "put the credentials in .env. See .env.example."
            )
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()
        if reset:
            self.run("MATCH (n:Thing) DETACH DELETE n")
        self.run("CREATE INDEX thing_name IF NOT EXISTS FOR (t:Thing) ON (t.name)")

    def run(self, cypher, **params):
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(cypher, **params)]

    def close(self):
        self.driver.close()

    def write(self, subject, relation, obj, since=None, until=None):
        self.run(
            """
            MERGE (s:Thing {name: $subject})
            MERGE (o:Thing {name: $object})
            CREATE (s)-[:REL {type: $relation, since: $since, until: $until}]->(o)
            """,
            subject=subject, object=obj, relation=relation, since=since, until=until,
        )
        return self

    def expire(self, subject, relation, obj, at):
        """Stop an edge being true, without pretending it never was."""
        self.run(
            """
            MATCH (:Thing {name: $subject})-[r:REL {type: $relation}]->(:Thing {name: $object})
            WHERE r.until IS NULL
            SET r.until = $at
            """,
            subject=subject, object=obj, relation=relation, at=at,
        )
        return self

    # a relationship is live at `at` when it had already started and had not yet ended.
    # Repeated in several queries below rather than factored out, because seeing the same
    # condition in each one is the point -- every question about a graph with time in it has
    # to say WHEN, and forgetting to is how stale edges leak into answers.
    _LIVE = ("($at IS NULL OR ((r.since IS NULL OR r.since <= $at) "
             "AND (r.until IS NULL OR r.until > $at)))")

    def about(self, node, at=None):
        return self.run(
            f"""
            MATCH (n:Thing {{name: $node}})-[r:REL]-(other:Thing)
            WHERE {self._LIVE}
            RETURN startNode(r).name AS subject, r.type AS relation,
                   endNode(r).name AS object, r.since AS since, r.until AS until
            """,
            node=node, at=at,
        )

    def neighbours(self, node, at=None):
        return [e["object"] if e["subject"] == node else e["subject"]
                for e in self.about(node, at)]

    def connected(self, start, hops=2, at=None):
        """Every node within `hops`, with its distance -- one variable-length Cypher pattern."""
        rows = self.run(
            f"""
            MATCH path = (n:Thing {{name: $start}})-[r:REL*1..{int(hops)}]-(other:Thing)
            WHERE all(r IN relationships(path) WHERE {self._LIVE})
            RETURN other.name AS name, min(length(path)) AS hops
            """,
            start=start, at=at,
        )
        return {r["name"]: r["hops"] for r in rows if r["name"] != start}

    def path(self, start, end, hops=3, at=None):
        """The chain of edges linking two things, or None.

        This is the part that answers the question. `connected()` says Meera is two hops from
        checkout-api; `path()` says *why* -- checkout-api calls billing-api, which Meera owns --
        and the chain is the answer, not the node.
        """
        rows = self.run(
            f"""
            MATCH path = shortestPath(
                (a:Thing {{name: $start}})-[r:REL*1..{int(hops)}]-(b:Thing {{name: $end}}))
            WHERE all(r IN relationships(path) WHERE {self._LIVE})
            RETURN [rel IN relationships(path) |
                    {{subject: startNode(rel).name, relation: rel.type,
                      object: endNode(rel).name}}] AS edges
            """,
            start=start, end=end, at=at,
        )
        return rows[0]["edges"] if rows else None

    @staticmethod
    def phrase(edges):
        return " -> ".join(
            f"{e['subject']} {e['relation']} {e['object']}" for e in edges
        )

    def __len__(self):
        return self.run("MATCH ()-[r:REL]->() RETURN count(r) AS n")[0]["n"]
