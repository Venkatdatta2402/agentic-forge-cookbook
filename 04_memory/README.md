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
   wrong generated `JOIN` returns plausible rows. Closes by **composing** them,
   because plenty of questions need two: *"why did the latest billing deploy
   fail?"* is a filter-order-limit question and a prose question, and neither
   store does the other's half. `pipeline.py` makes each store take a turn that
   either **answers** or **narrows what the next one may look at**, with a
   narrowing that belongs to none of them -- Chroma wants `{"at": {"$gte":
   20260819}}`, SQL wants a string, Cypher wants a parameter, so each step
   translates. Same four stores, three orders, three different questions. Empty
   narrowing means carry on unnarrowed (`fail_open`), which is right for a hint
   and wrong for a tenant filter, so it is set per step
3. `03_operations.ipynb` — the same five operations across every store and index,
   and the fact that only their signatures are portable. `write` on a graph
   needs an entity-extraction call no other store wants; `update` on a vector
   index is a delete and a re-embed because there is nothing to edit in place;
   `retrieve` does not exist on a dict at all. `read` and `retrieve` are two
   operations and not one — a read is a lookup where you know the key and get
   back exactly what you asked for, a retrieve is a ranked *guess* against a
   query you cannot name a key for — and everything hard in this notebook is
   on the retrieve side. Four things inside it worth naming separately:
   - **Scoring.** `score = w1*similarity + w2*activation`. Opens by separating
     three things that all look like time, because confusing them is the
     expensive mistake: **reinforcement** (has this been used), **validity**
     (is it still true, which is supersession's job and not a ranking's), and
     the **write date** (metadata you filter on). Only the first is a ranking
     signal, so it is measured from an `accesses` table rather than from `at`.
     Then the failure that looks fine until you print the two terms apart:
     cosine moves across 0.115 while decay moves across 0.816, so "fifty-fifty"
     is seven-eighths recency. Min-max across candidates to make a weight mean
     a share of the decision — then threshold on **raw** similarity, *before*
     the rescaling, because min-max manufactures a 1.0 out of whatever it is
     handed. Answerable questions measure 0.72–0.78 here and unanswerable ones
     0.52–0.55, which is where the floor comes from. Frequency goes **inside**
     lambda rather than beside it — `lambda(f) = lambda_0 / (1 + alpha*f)` —
     so it stretches the clock instead of competing with recency: a sum would
     let ten uses last quarter outrank one use yesterday. Both weights are
     measured, not chosen: six questions with known answers rule out the even
     split, and `w_reinforcement=0.5` lets one memory take first place on every
     query and collect every access in the corpus. No LLM-rated importance —
     models return 6–8 for nearly everything, so it costs a call per write and
     moves no ranking; a `pinned` flag covers what it was actually for. Three
     numbers are left explicitly unearned, because five months of history
     cannot separate them: `H₀`, `alpha`, and linear-versus-damped consolidation
   - **Two-stage recall, as tools rather than a pipeline.** `recall(query)`
     returns ids and one-line summaries; `open_memory(id)` returns the full
     text. Chapter 2's loop already runs this, and tool calling already *is*
     the structured output, so the ID-parsing problem never appears. Built
     only at the point where a memory is a whole transcript — for short
     memories the matched line *is* the memory and `open_memory` should not
     exist. Its real second job: being opened is evidence a memory was used,
     which mere top-k membership is not — so **the last stage records**, and
     `record_use` is derived from the store rather than left as a flag. A
     single-stage store with nothing recording has an empty `accesses` table
     and a reinforcement term that has quietly become write-date decay. Four
     things that looked right and were not get fixed in public here: a loop
     missing its trailing `Observe()`, so tool results never reach the model;
     a listing with no dates, so a question about November opens October; a
     docstring that never says what ids are for; and a demo question the
     summaries already answered
   - **Deciding to write.** `update` arrives early as a mechanism with nothing
     that calls it, which is how an agent ends up holding the same fact five
     times. So a write becomes a search, then a decision: **add / update /
     ignore**, with `Literal` putting the three in the schema and a
     `model_validator` rejecting an `update` that names nothing to update.
     Two contradicting memories about one fact is *already* the failure — it
     means an earlier write went in as `add` — so the job is preventing the
     pair, not adjudicating it. Stated plainly: the check is bounded by the
     search, so a memory retrieval fails to surface becomes a duplicate made
     by the procedure meant to prevent it; and it compares a conclusion
     against the store, never against the world
   - **Forgetting.** Decay and deletion are not the same tool. Decay is a
     ranking change — the row is still there, still costs index and storage,
     and still comes back **first** on a query where nothing else is close,
     which the notebook shows at an activation of 0.0351. Deletion is opt-in:
     `evict_below` is `None` by default and nothing is ever removed for having
     decayed, because for most journals keeping everything is the right answer
     and the cost of a bound is paid in dilution rather than disk. What decay
     buys is a defensible basis for one — a score that has *stayed* low is
     evidence — but it only measures **use**, so a sweep cannot tell a
     superseded memory from one nobody has happened to ask about. That is what
     `pin()` and a dry run are for. A graph does neither: an edge that has
     stopped holding gets an end date, so July returns Meera, September returns
     Devi, and an unfiltered query returns both
4. `04_types_of_memory.ipynb` — semantic (facts), episodic (events),
   procedural (how-to): what each holds, which store from notebook 2 fits, and
   what breaks when they are paired wrong. Episodic memory in a plain vector
   index that cannot answer which event was *latest*; semantic facts in a
   graph that a dict lookup already answered. The write path is where the
   types diverge hardest, and notebook 3 already built the mechanism: the
   **add / update / ignore** decision is right for a semantic fact and
   destroys a run history when applied to episodic memory, because "deploy
   failed Aug 3" and "deploy failed Aug 12" look like duplicates to a
   similarity check and are two events. Notebook 3 owns the procedure; this
   one owns the trap
5. `05_memory_in_the_loop.ipynb` — wiring it back into `02_agent_runtime`,
   which turns out to need no modification: a component is anything with
   `.run(conversation)`, so `Retrieve` slots into the loop exactly like
   `Observe`, and `remember` is just a tool. Where this lands: **recall by
   injection, write by tool** — but not for the reason it first looks like.
   Offered `recall` as a tool, the model is not forgetful: over fifteen trials
   on questions that *ask about the past* it called it fifteen times. The split
   is by question shape. Asked to **decide** something — *"the error rate is at
   three percent, what should I do?"* — it consulted memory seven times in
   fifteen, and the misses answer out of general knowledge while the team's own
   rule (*page Priya above two percent*) sits unread. A policy does not resemble
   the situation it governs, which is `04_types_of_memory`'s finding arriving in
   the loop. And the failure is invisible: no empty result, no error, just a
   fluent answer that consulted nothing.
   Two things bite on the way in. **What `Retrieve` injects is derived from the
   store, not configured** — a journal with summaries has a second stage, one
   without does not — and getting it wrong in the generous direction injects
   full bodies (2,136 characters against 363), leaves the model no reason to
   open anything, and stops two-stage recall silently: correct answers, no
   warning, six times the tokens. And **`Observe` rewrites every tool result
   through a model**, which for memory is damage. The listing loses its `#ids`
   in roughly half of live runs, and without an id there is nothing for
   `open_memory` to be called with — so the second stage disappears while the
   loop still looks healthy. An 868-character write-up comes back at 125–250
   characters carrying two of its six facts. `Observe(extractors=...)` switches
   the model call off per tool, and `04_memory` ships the two entries that
   always apply.
   The one thing genuinely missing is a place to hang the store: nothing on
   `Conversation` holds a memory handle, so the tools are built by a function
   that closes over it. Also where the pipelines from notebook 2 become tools --
   **you write several and the model picks one by name**, so the composition
   stays something you tested and the only judgement left to the model is which
   shape of question it is looking at. A model assembling the steps itself is
   the option not taken: it produces orders that are syntactically fine and
   retrieve nothing, and that failure is indistinguishable from a question the
   store cannot answer

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
- **GraphRAG** — the query patterns that composition generalises: vector-first
  (find pivot entities, then traverse) and graph-first (filter entities, then
  rank inside them) are two orderings of the same narrow-then-answer shape. Its
  **global search** is the one thing no arrangement of these four stores gives
  you, because it reads cluster summaries written at *build* time rather than
  retrieving anything — the same move the agent literature calls **reflection**

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
