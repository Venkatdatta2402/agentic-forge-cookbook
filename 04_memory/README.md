# Memory

**What:** Everything an agent keeps once a turn is over — the message list it
is already carrying, the facts it writes down about a user, the runs it has
done before, and the skills it picked up doing them. Where each of those
lives, how it gets back out, how it is ranked when more comes back than fits,
and what happens when it is wrong.

**Why:** `02_agent_runtime` gave every component the same conversation and
`03_tools` let the agent act on the world. Both forget everything the moment
the loop returns. Start a second session and the agent meets you as a
stranger — no idea it already asked your timezone, already read that file,
already tried the approach that failed.

The usual fix is to reach for a vector database, and that is the wrong first
move. Most of what an agent needs to remember is a fact with a name — a
timezone, a preferred language, a project root — and a dict answers that
exactly, instantly, and correctly. A vector index answers it *approximately*,
which is strictly worse. The database is not the subject; the query is.

**This chapter also absorbs retrieval.** Retrieval and memory run on the same
machinery — index, search, rank. What differs is the write path: memory is
written by the agent from its own experience, a document corpus is bulk-loaded
and the agent was never there. That is a difference in provenance, not in
architecture, so it is a section here rather than a chapter of its own.

**The core idea:** memory is a small set of operations — **write, update,
read, retrieve, forget** — and the interesting decision is made *before* any of
them runs. Where a memory belongs is settled when it is written, by one thing:
**what will be asked of it later.** The shape it takes follows from that.

| Store | What it will be asked | Example |
|---|---|---|
| `KeyValue` | looked up by name | "what is this user's timezone" |
| `Records` | filtered, ordered, counted | "which failed most recently", "how many last week" |
| `Journal` | searched by literal, and by resemblance | "where did `PGRST204` come from", "has this happened before" |
| `Graph` | walked | "who else touched this, and what did they decide" |

The tempting alternative — "structured data over here, prose over there" — gets
the causation backwards. Shape is a consequence of the question, not a criterion
of its own, and treating it as the criterion leads to filing a thing's fields in
one store and its text in another, which makes each store look necessary by
blinding the other.

`Records` and `Journal` describe the same world, and what separates them is that
one gets counted and the other gets searched. Who wrote them is not the deciding
factor, but it does explain why each arrives in the shape it does: a deploy
pipeline emits rows on a schema designed before the first deploy ran, so
filtering is cheap; the agent's notes on *why* something broke could never have
had a schema designed in advance, so what is available is prose. Authorship
constrains what a store can be asked cheaply. The asking still decides.

A second, smaller decision comes after: a store can offer more than one way in.
`Journal` carries both a literal index and a vector index over the same entries,
and which one answers depends on the question:

| The question | The index |
|---|---|
| an exact error code, flag, or identifier | FTS5 — a boolean `MATCH` to select, then **BM25** to rank |
| "has this happened before?" — a symptom naming no field | embeddings; nothing else can do this |
| "…but only about this service, since August" | embeddings **with a metadata filter** |
| both at once, ranked together | reciprocal rank fusion over the two |

Nothing is introduced until a question has failed without it.

## Notebooks

1. `01_what_memory_is.ipynb` — the agent that meets you twice and does not
   know it. Working and short-term memory need **no store at all**: chapter
   2's message list is already memory, and so is a scratchpad. Trimming, and
   what survives a hand-off between sessions. Closes with the list of recall
   questions the rest of the chapter owes an answer to
2. `02_stores.ipynb` — four stores, chosen by what each memory will be asked. `Records` (a deploy pipeline, SQLite) and `Journal` (the
   agent's own write-ups, SQLite FTS5 + Chroma) describe the same world and
   differ by author; neither is stripped down to make the other look necessary.
   What FTS5 actually is gets settled here — a boolean `MATCH` selecting
   candidates, then **BM25** ranking them, with the negative scores shown and
   term frequency, saturation and length normalisation visible in them. The
   question that forces embeddings is a **symptom** — *"checkout is timing out
   and the database seems to be blocking"* — which names no field to filter on;
   all five of its words return zero from full-text search, and filtering to the
   right service still returns three entries with no opinion about which
   explains a timeout. Then reciprocal rank fusion over BM25 and cosine, which
   on this corpus makes an answer **worse** — and the cause turns out to be the
   corpus, not the method: `out` appears in one entry of six, so BM25 reads it
   as a rare and informative term. At 60% document frequency its weight
   collapses to ~1e-06. The stopword list is standing in for what rarity does
   by itself at scale, which is a thing a six-entry demonstration cannot show. Chroma implements
   this fusion natively and raises `NotImplementedError: Search is not
   implemented for Local Chroma`, which is the honest reason two engines are
   running. **No text-to-SQL**: `recent(kind, status, service, since, limit)` is
   written once with parameters, because a bad tool argument fails loudly and a
   wrong generated `JOIN` returns plausible rows
3. `03_operations.ipynb` — the same five operations across every store and index,
   and the fact that only their signatures are portable. `write` on a graph
   needs an entity-extraction call no other store wants; `update` on a vector
   index is a delete and a re-embed because there is nothing to edit in place;
   `retrieve` does not exist on a dict at all. `read` and `retrieve` are two
   operations and not one — a read is a lookup where you know the key and get
   back exactly what you asked for, a retrieve is a ranked *guess* against a
   query you cannot name a key for — and everything hard in this notebook is
   on the retrieve side. Three things inside it worth naming separately:
   - **Scoring.** `score = w1*similarity + w2*exp(-lambda*t)`, where `t` is
     time since last *access*. Opens on the failure that looks fine until you
     print the two terms apart: decay spans its whole 0–1 range while cosine
     lives in a narrow band well off zero, so recency quietly decides the
     ranking on its own. Min-max across candidates to make the weights mean
     something — then threshold on **raw** similarity, because normalizing
     promotes the least-bad of thirty bad candidates to 1.0. Over-fetch where
     it is free and trim before anything expensive: the index's k and the
     model's k are different numbers. Frequency enters as a slower decay for
     often-used memories, with the rich-get-richer runaway shown rather than
     asserted. No LLM-rated importance — models return 6–8 for nearly
     everything, so it costs a call per write and moves no ranking; a `pinned`
     flag covers what it was actually for
   - **Two-stage recall, as tools rather than a pipeline.** `recall(query)`
     returns ids and one-line summaries; `open_memory(id)` returns the full
     text. Chapter 2's loop already runs this, and tool calling already *is*
     the structured output, so the ID-parsing problem never appears. Built
     only at the point where a memory is a whole transcript — for short
     memories the matched line *is* the memory and `open_memory` should not
     exist. Its real second job: being opened is evidence a memory was used,
     which mere top-k membership is not
   - **Forgetting.** Decay and deletion are not the same tool. Decay is a
     ranking change — the row is still there, still costs index and storage,
     and still wins on a query where nothing else matches, which is exactly
     how a contradicted fact resurfaces confidently. Deletion is the only
     thing that fixes *wrong*, and a bounded size is the only thing that keeps
     top-k clean. They belong together because decay is what makes deletion
     safe: you delete what stayed decayed, instead of guessing. A graph does
     neither — an edge that has stopped being true gets a validity range, so
     "used to work at X" stays answerable. That is a fact with an end date,
     not a fact that was ever wrong, and deleting it loses real history
4. `04_types_of_memory.ipynb` — semantic (facts), episodic (events),
   procedural (how-to): what each holds, which store from notebook 2 fits, and
   what breaks when they are paired wrong. Episodic memory in a plain vector
   index that cannot answer which event was *latest*; semantic facts in a
   graph that a dict lookup already answered. The write path is where the
   types diverge hardest — a semantic write has to search for a near-duplicate
   first and decide ADD / UPDATE / IGNORE, while that same rule applied to
   episodic memory destroys a run history, because "deploy failed Aug 3" and
   "deploy failed Aug 12" are two events and not a contradiction
5. `05_memory_in_the_loop.ipynb` — wiring it back into `02_agent_runtime`,
   which turns out to need no modification: a component is anything with
   `.run(conversation)`, so `Recall` slots into the loop exactly like
   `Observe`, and `remember` is just a tool. Recall by injection before
   `Think` (reliable, paid every turn) versus recall as a tool the model
   chooses to call (cheap, and the model often does not think to look). Where
   this lands: **recall by injection, write by tool** — recall has to be
   dependable, and judging what is worth keeping is something the model is
   genuinely good at. The one thing genuinely missing is a place to hang the
   store: nothing on `Conversation` holds a memory handle, so `remember` has
   to close over it

## Keeping the summaries honest

Two-stage recall only works while a summary still describes its memory, and
the ordering that guarantees this is worth stating on its own. The summary is
a **derived** field: the only method that changes the text is the method that
rewrites the summary, and there is no separate setter. Inside that method, the
fallible thing runs before the durable one —

```python
def update(self, id, new_text):
    summary = llm_summarize(new_text)     # fails? nothing has been written yet
    with self.db:                         # both fields, one transaction
        self.db.execute(
            "UPDATE memories SET text=?, summary=? WHERE id=?",
            (new_text, summary, id),
        )
```

Summarize first, write both together, and a timeout leaves the old state
intact rather than leaving new text behind an old summary.

The bill this quietly creates is worth stating: two-stage recall moves cost
onto the **write** path. Every save and every update now carries a model call
that a single-stage store does not, which is one more reason not to build it
until memories are genuinely large.

Rewriting from the delta — old summary, what was added, what was removed —
beats re-summarizing cold, for a reason that is not obvious: it keeps the
phrasing stable, so the summary's own embedding moves only when the fact
moves. Its cost is photocopy drift over many edits, and the repair is a
scheduled **reflection** pass that reads the source: *here is the memory, here
is its summary, what would a search miss?* Rewrite only when it objects. Same
shape as `verified()` in `03_tools` — check the artifact, do not trust the
report that produced it.

## The one axis worth keeping straight

The names for memory types come from three different questions, which is why
any single list of them looks incomplete. Separated, they close:

|  | Semantic (facts) | Episodic (events) | Procedural (how-to) |
|---|---|---|---|
| **Working** (this turn) | facts pasted into the prompt | recent messages | system prompt, tool schemas |
| **Short-term** (this session) | extracted entities | the conversation | the scratchpad, the plan |
| **Long-term** (across sessions) | user profile | run history | learned skills, few-shot examples |

*Scratchpad* and *conversation memory* are not additional types. They are
cells in this table.

## What this chapter does with chapters 1–3

| From earlier chapters | What `04_memory` does with it |
|---|---|
| `trim_to_budget()` and `Conversation.notes` (1) | Where working memory already runs out of room. Notebook 1 starts here rather than from nothing |
| `ConversationMemory.key_facts` (1) | The first long-term memory in the repo, written before there was anywhere to put it. Notebook 2 gives it a store |
| The `messages` list every component reads and writes (2) | Named for what it is: working and short-term memory, with a hard size limit and no persistence |
| `Observe` summarizing to keep the conversation small (2) | The same summary, kept instead of discarded — the cheapest long-term memory there is |
| `Runtime.run(conversation)` (2) | Already takes a conversation the caller built, so loading before and saving after needs no change to it |
| Any object with `.run(conversation)` is a component (2) | **This is the memory hook, and it already exists.** Recall is a component placed before `Think`, exactly like `Observe`. Notebook 5 changes nothing in `02_agent_runtime` |
| `Think(tools=, allow_fork=, allow_replan=)` (2) | The other route in: `remember` and `recall` are ordinary tools, so the model can reach memory without a new power being invented for it |
| `Registry` — one object hiding many implementations (3) | The same shape for stores: one interface, four backends, and the leaks named rather than hidden |
| `Tool` / `@tool` and `args_model()` (3) | `remember`, `recall` and `Records.recent` become ordinary tools. The parameterized query is why no text-to-SQL is needed |
| `invoke()` trimming an oversized tool result (3) | The same pressure one layer earlier: what comes back from `recall` is trimmed by score before it is ever a tool result |
| `verified()` — the call returned, but did it do the job (3) | Applies to writes. A fact was saved, but saved *wrong* is the failure that outlives every other kind |

## Where the ideas come from

- **Generative Agents** (Park et al., 2023) — the retrieval score as a
  weighted sum of similarity and recency, min-max normalized across
  candidates. Notebook 3's scoring section starts from this and drops its
  third term
- **ACT-R base-level activation** — why frequently accessed memories should
  decay more slowly, as a sum of independently decaying access terms rather
  than a patch bolted onto a single decay rate
- **Retrieve-then-rerank / small-to-big retrieval** — the standard names for
  what notebook 3 builds as `recall` plus `open_memory`. The nearest
  everyday example is a coding agent: `grep` returns matching lines, and the
  agent decides which files are worth opening
- **Okapi BM25** — the ranking function behind SQLite's FTS5, and the reason a
  literal search returns an ordering rather than a set. Term frequency,
  saturation and length normalisation are all visible in the scores notebook 2
  prints
- **Reciprocal Rank Fusion** (Cormack et al., 2009) — combining two rankings by
  position instead of by score, which is what makes hybrid search possible at
  all when one side is BM25 (negative, corpus-dependent) and the other is
  cosine (a narrow band off zero)

## What this chapter needs that earlier ones did not

Groq serves no embedding model, so `stores.embed()` calls **Gemini**
(`gemini-embedding-001`, 768 dimensions) — the only place in the repo talking
to a second provider. It is deterministic: the same string returns
bit-identical vectors every time, which is what makes the saved outputs worth
reading when the point of a cell is *which* neighbours came back and in what
order.

Two real databases arrive here as well. **Chroma** installs with pip and runs
in-process — though not at full strength: its native hybrid search, a BM25
sparse index fused with `Rrf`, raises `NotImplementedError: Search is not
implemented for Local Chroma` in the embedded client, and works only on the
distributed and hosted builds. That is why FTS5 sits alongside it doing a job
Chroma can do elsewhere, and it is a more ordinary reason to run two engines
than anything about scale: the product has the feature, the deployment does
not.

**Neo4j** goes further — it has no embedded mode at all, so notebook 2 is the
only notebook in the repo that cannot run without a service being up
somewhere. A free instance at <https://console.neo4j.io> takes a couple of
minutes, and `.env.example` lists all three credentials.

That requirement is worth treating as part of the lesson rather than a
footnote: a graph database is infrastructure, and the cost of running one
belongs in the decision to use one.
