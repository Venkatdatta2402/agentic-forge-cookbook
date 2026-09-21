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
    db = sqlite3.connect(path)
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

    def meta(self, entry_id):
        got = self.vectors.get(ids=[str(entry_id)])
        return got["metadatas"][0] if got["ids"] else None

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]


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
