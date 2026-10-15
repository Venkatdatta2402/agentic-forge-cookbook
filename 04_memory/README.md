# 04 · Memory

> What an agent keeps once a turn is over, where it keeps it, how it gets it back, and what
> happens when it's wrong.

Without memory, the agent from chapters 2 and 3 forgets everything when the loop returns. In a
second session it meets you as a stranger. It doesn't know it already asked your timezone, already
read that file, or already tried the approach that failed.

**Contents:** [The idea](#the-idea-in-one-picture) · [What you'll learn](#what-youll-learn) ·
[Notebooks](#notebooks) · [Key ideas](#key-ideas) · [Modules](#modules) ·
[How it connects](#how-it-connects) · [Running it](#running-it)

---

## The idea in one picture

The usual first move is "add a vector database". That's the wrong first move. Most of what an agent
needs to remember is a fact with a name, such as a timezone, a preferred language or a project
root, and a dict answers that exactly, instantly and correctly. A vector index answers it
*approximately*, which is strictly worse.

**What decides where a memory goes is the question that will be asked of it later.** Its shape
follows from that.

| Store | What it will be asked | Example | Built on |
|---|---|---|---|
| `KeyValue` | looked up by name | "what is this user's timezone?" | a dict saved to a file |
| `Records` | filtered, ordered, counted | "which deploy failed most recently?", "how many last week?" | SQLite |
| `Journal` | searched by exact words, and by resemblance | "where did `PGRST204` come from?", "has this happened before?" | SQLite FTS5 + Chroma |
| `Graph` | walked | "who else touched this, and what did they decide?" | Neo4j |

A store can also offer more than one way in. `Journal` holds a keyword index and a vector index
over the same entries:

| The question | The index that answers it |
|---|---|
| an exact error code, flag or identifier | **FTS5**: a boolean `MATCH` selects, then **BM25** ranks |
| "has this happened before?", a symptom that names no field | **embeddings**; nothing else can do this |
| "…but only about this service, since August" | embeddings **with a metadata filter** |
| both at once, ranked together | **reciprocal rank fusion** over the two |

The rule this chapter follows is that **nothing is introduced until a question has failed
without it.**

## What you'll learn

- Why working and short-term memory need **no store at all**
- How to pick a store by the question it will answer, and how to **compose** stores when one
  question needs two
- The five operations (**write, update, read, retrieve, forget**) and why only their *signatures*
  carry over between stores
- How to **rank** what comes back: similarity, reinforcement, a floor, and weights you measure
  rather than guess
- **Two-stage recall** (short summaries first, open the full memory on demand)
- How to stop an agent storing the same fact five times (**add / update / ignore**)
- Semantic, episodic and procedural memory, and the trap of treating events like facts
- How to wire memory into chapter 2's loop with no change to the loop

## Notebooks

| # | Notebook | What it shows |
|---|---|---|
| 1 | [`01_what_memory_is`](notebooks/01_what_memory_is.ipynb) | The agent that meets you twice and doesn't know it. Chapter 2's message list already *is* working and short-term memory; trimming; what survives a hand-off between sessions |
| 2 | [`02_stores`](notebooks/02_stores.ipynb) | Four stores, each brought in by a question the last couldn't answer. What FTS5 and BM25 really do. A symptom that only embeddings can find. Fusion making one answer *worse*, and why. Composing stores into pipelines. No text-to-SQL |
| 3 | [`03_operations`](notebooks/03_operations.ipynb) | The five operations across every store. Scoring and its weights, measured. Two-stage recall as tools. Deciding to write (add / update / ignore). Forgetting: decay vs deletion |
| 4 | [`04_types_of_memory`](notebooks/04_types_of_memory.ipynb) | Semantic (facts), episodic (events) and procedural (how-to), and what breaks when each is paired with the wrong store |
| 5 | [`05_memory_in_the_loop`](notebooks/05_memory_in_the_loop.ipynb) | Memory wired into chapter 2's runtime, unchanged: recall by injection, writing by tool, and why |

## Key ideas

### Working memory is just the message list
Chapter 2's `messages` already is memory, with a hard size limit and no persistence. A
`Scratchpad` (a dict) is short-term memory the agent can revise. Writing `"plan"` twice replaces
it, where the conversation would keep both versions. Neither needs a database.

### Read and retrieve are different operations
A **read** is a lookup where you know the key and get exactly what you asked for. A **retrieve**
is a ranked *guess* against a query you can't name a key for. Almost everything hard in this
chapter is on the retrieve side.

### Ranking: measure the weights, then put a floor on raw similarity
`score = w1·similarity + w2·activation`, where activation is **reinforcement** (how recently and
how often a memory was actually *used*, from an `accesses` table), not when it was written.

- **A 50/50 weighting isn't 50/50.** Cosine similarity moved across a range of 0.115 while decay
  moved across 0.816, so "fifty-fifty" was really seven-eighths recency. The fix is to min-max
  normalise each term across candidates, so a weight means a share of the decision.
- **Put the floor on raw similarity**, *before* normalising. Min-max turns whatever it's given
  into a 1.0. Questions the memory could answer scored 0.72–0.78, and questions it couldn't
  scored 0.52–0.55, which is where the floor comes from.
- **Frequency slows decay instead of competing with it:** `λ(f) = λ₀ / (1 + α·f)`. A sum would
  let ten uses last quarter outrank one use yesterday.
- **No importance scores rated by a model.** Models return 6–8 for nearly everything. A `pinned`
  flag covers what importance was actually for.

### Two-stage recall
`recall(query)` returns ids and one-line summaries; `open_memory(id)` returns the full text, like
`grep` followed by opening a file. Opening a memory is also the best evidence that it was *used*,
so that's the stage that records an access. This is only worth building when memories are large;
for short ones, the matched line *is* the memory.

### Deciding to write: add, update or ignore
A write first searches for what's already there, then decides **add / update / ignore**. Without
this an agent ends up holding the same fact five times. Two limits come with it. The check can
only find duplicates that retrieval surfaces. And it compares against the store, never against
the world.

### Facts and events need different write rules
"Deploy failed Aug 3" and "deploy failed Aug 12" look like duplicates to a similarity check, but
they're two events. The add / update / ignore rule is right for a **semantic** fact and destroys
an **episodic** history.

|  | Semantic (facts) | Episodic (events) | Procedural (how-to) |
|---|---|---|---|
| **Working** (this turn) | facts pasted into the prompt | recent messages | system prompt, tool schemas |
| **Short-term** (this session) | extracted entities | the conversation | the scratchpad, the plan |
| **Long-term** (across sessions) | user profile | run history | learned skills, few-shot examples |

### Forgetting: decay is not deletion
Decay only changes the ranking. The row is still stored, and it still comes back **first** when
nothing else is close. Deletion is opt-in (`evict_below` is `None` by default), because decay
measures *use*, not truth.

### Recall is injected; remembering is a tool
In the loop, `Retrieve` is a **component** that runs before `Think` on every turn, and
`remember` is a **tool** the model calls when it decides something is worth keeping. The reason
is what the measurements showed.

Offered `recall` as a tool, the model is not forgetful: over fifteen trials on questions that
*ask about the past* it called it fifteen times. The split is by question shape. Asked to
**decide** something (*"the error rate is at three percent, what should I do?"*), it consulted
memory seven times in fifteen, and the misses answer out of general knowledge while the team's own
rule (*page Priya above two percent*) sits unread. A policy doesn't look like the situation it
governs. And the failure is invisible: no empty result, no error, just a fluent answer that
consulted nothing.

Writing is the opposite. Not writing something down costs nothing, and judging what's worth
keeping is a judgement, which models are good at. So the model decides.

Two traps on the way in:
- **Injecting full bodies** when the store has summaries (2,136 characters against 363) leaves
  the model no reason to open anything, and quietly costs six times the tokens.
- **`Observe` rewrites tool results through a model**, which drops the `#ids` that `open_memory`
  needs in about half of live runs. `Observe(extractors=EXTRACTORS)` turns that off for memory
  tools.

## Modules

| File | What's in it |
|---|---|
| [`memory.py`](memory.py) | Working and short-term memory with no store: `Scratchpad`, `save_session()`, `load_session()` |
| [`stores.py`](stores.py) | `KeyValue`, `Profile`, `Records`, `Journal`, `QdrantJournal`, `Graph`, and `embed()` (Gemini embeddings) |
| [`recall.py`](recall.py) | `Recall`: over-fetch, floor, normalise, weight; hybrid (meaning + keywords) by default; `questions()` to search a message one question at a time |
| [`memory_tools.py`](memory_tools.py) | `Retrieve` (the component), `memory_tools()` (`remember`, `recall`, `open_memory`), and `EXTRACTORS` |
| [`pipeline.py`](pipeline.py) | Composing stores: each `Step` either answers or narrows what the next may look at; `Pipeline` |
| [`writing.py`](writing.py) | Checking a write before it's kept: `profile_form`, `unsupported`, `day_of`, `fill`, `rules` |

### Added later, by `08_frameworks/06_mem0`
That project's running coach stored everything correctly and still told a runner the wrong race
date, because the wrong things had been *written*. Its general fixes came back here:

- **`Profile`**: named facts that change. The new value replaces the old, which is kept beside it
  with its dates.
- **`QdrantJournal`**: the journal on a single engine (keywords and vectors on one record).
- **`writing.py`**: a required slot for every field, a check that every day, date, number and
  name in a value appears in the source, and date arithmetic done in code.
- **Hybrid `Recall`**, on by default. This chapter's notebooks pass `hybrid=False`, because that
  is what their numbers were measured on.

## How it connects

**Builds on:** chapter 1's `trim_to_budget()` and `notes` (where working memory runs out of
room), chapter 2's "a component is anything with `.run(conversation)`" (the hook memory plugs
into, so **no change to `02_agent_runtime`**), and chapter 3's `@tool` and `verified()`.

**Used by:** `05_context` (retrieval as one of its operations), `06_orchestration` (contrasted
with a shared board), and `08_frameworks` (LangChain's retrieval, Mem0's from-scratch build).

**Where the ideas come from:** Generative Agents (Park et al., 2023) for weighted retrieval
scores; ACT-R base-level activation for decay slowed by use; retrieve-then-rerank / small-to-big
retrieval; GraphRAG; Okapi BM25; Reciprocal Rank Fusion (Cormack et al., 2009).

## Running it

```bash
conda activate agentic-forge
# .env needs GROQ_API_KEY, GEMINI_API_KEY, and for notebook 2 the NEO4J_* values
```

| Needs | Why |
|---|---|
| **Gemini** (`gemini-embedding-001`, 768 dims) | Groq serves no embedding model. The same text always returns the same vector, so saved outputs can be compared |
| **Chroma** | Installs with pip and runs in-process. Its native hybrid search raises `NotImplementedError` in the local build, which is why FTS5 runs beside it |
| **Neo4j** | Has no embedded mode, so notebook 2 needs a running instance. A free one at <https://console.neo4j.io> takes a couple of minutes. A graph database is infrastructure, and that cost belongs in the decision to use one |

## Going deeper

[`NOTES.md`](NOTES.md) has the full write-up: every measurement behind the scoring weights, how
summaries are kept honest, the composition pipelines, and everything added from `06_mem0`.
