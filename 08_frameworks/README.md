# Frameworks

**What:** The major frameworks for building agentic systems — LangChain, LangGraph, CrewAI, AutoGen,
LlamaIndex, Mem0, LangSmith and Langfuse — each learned by **building one small, working,
production-minded project with it**, not by touring its API.

**Why:** Chapters 1–7 built the parts from scratch: the model call, the agent loop, tools, memory,
context, orchestration, MCP. That is what makes a framework readable. Every abstraction a framework
introduces is something this repo has already built once, so the questions worth asking are
concrete: what did the framework do for us, what did we have to write anyway, what did it hide,
and what did it cost.

**The core idea:** wherever it applies, the same project is built **twice** — once with the
framework, once from this repo's own primitives — on the same problem, the same prompts and the
same answer key, so the difference between the two builds is the framework and nothing else.

## How this chapter is organised

Differently from chapters 1–7. Each framework has its own folder, holding the project's modules
**and** its notebook. The system lives in the modules; the notebook shows it:

1. **Build** — what the project is, with the important functions shown as code (taken from the
   modules, so they cannot drift) and short explanations
2. **Run and show** — the system running offline against stand-ins, then live, the from-scratch
   build on the same inputs, what building it found out, and findings

Every folder also has stand-ins for each model call (`*_stubs.py`), the behaviours worth knowing
reproduced as functions (`*_checks.py`), and offline tests (`test_*.py`) that cost nothing to run.

| Folder | Framework | Project | From scratch | Status |
|---|---|---|---|---|
| `01_langchain` | LangChain | **Meridian's claims assistant** — answers travel-insurance customers from the policy wording and their account: retrieval, structured output, tools, composition, fallbacks, callbacks, streaming, memory | yes | ✅ |
| `02_langgraph` | LangGraph | **Kestrel Labs' accounts-payable desk** — invoices matched, routed, investigated by an agent, and held for people for days: persistence, human-in-the-loop, resume in another process, rewind | yes | ✅ |
| `03_crewai` | CrewAI | **Tallyfold's security questionnaire** — three specialists answering in parallel from a policy pack, a lead assembling, code guardrails sending work back; sequential and hierarchical crews. Runs on qwen3.8-27b (see below) | yes | ✅ |
| `04_autogen` | AutoGen | **Brightwater's analysis team** — a planner, an analyst that writes and runs pandas, a reviewer and a reporter, answering three questions from an order export with two problems planted in it: group chat, code execution, termination, structured messages | yes | ✅ |
| `05_llamaindex` | LlamaIndex | **Ask the cookbook** — answers questions about this repo from its READMEs, notebook prose and modules: splitting prose and code, a vector index refreshed as files change, fused retrieval, a floor that turns off-topic questions away, cited answers, retrieval evaluation, and an agent that searches and opens files | the agent only | ✅ |
| `06_mem0` | Mem0 | **A running coach that remembers** — two runners, nine weekly check-ins, and facts that keep changing (a race moved, new running days, a knee cleared, a new goal): extraction after every conversation, hybrid search, scoping by user, custom instructions, history | yes | ✅ |
| `07_langsmith` | LangSmith | **Evaluating the cookbook assistant**: `05_llamaindex`'s assistant traced, run on a 10-question dataset whose reference points rest on quoted lines of the repo, scored by free checks and a claim-by-claim judge checked against hand marks, and fixed round by round from its traces: datasets, experiments, evaluators, feedback, traces read back | — | ✅ |
| `08_langfuse` | Langfuse | **The cookbook assistant in production**: the same assistant, showing only what `07_langsmith` did not: every LlamaIndex step traced through OpenTelemetry, a cost on every model call, the agent's prompt as versions linked to each call, and monitoring through the metrics API | — | ✅ |

The split the chapter follows: LangChain is **building blocks** for LLM applications, LangGraph is
**stateful orchestration**, CrewAI is **role and task** multi-agent systems, AutoGen is **agents in
conversation**, LlamaIndex is **knowledge and retrieval**, Mem0 is **memory**, and LangSmith and
Langfuse are **evaluation and observability**.

## What the finished projects found

**`01_langchain`** — both builds scored 5 of 8 at the same cost, and made the same real mistake:
replying correctly in prose that a delay was already paid while the fields said to pay it again.
That became a rule in code. Building it found that a chain gives a model one turn and this model
calls one tool per turn (15 of 15); that Groq does not stream JSON replies, so the reply streams as
prose first and fields after; that `with_structured_output(method="json_schema")` silently means a
tool call on non-OpenAI model classes; and that LangChain 1.x has deprecated both of its memory
classes in favour of LangGraph.

**`02_langgraph`** — both builds handled all seven invoices as the policy says (19,778 tokens against
22,874; 267 lines against 263). LangGraph's durability (pause for days, resume in another process,
carry on after a crash, rewind and correct) worked first time. The scratch build is a different
design, not a smaller graph: a **case file** (`case.py`) on chapter 6's shared board, every write a row
in chapter 4's `Records`, each field with one owner, and a step that asks a person *ending* rather
than being re-run. That removes two of the graph's hazards by construction, and refuses an approval
answered by the wrong person, which the graph records as the right person's; rewind-and-fork is what
it gives up. Five behaviours to know
before production, each reproduced offline: a resumed node runs again from its first line; answers
to `interrupt()` are matched by position, so a reordering deploy swaps them; the default step
limit is 10,007; a loop through `interrupt()` is never stopped by it; retry layers multiply. And
one found live: a tool call the provider refuses ends the run.

**`03_crewai`** — the sequential crew scored 9 of 9 on a questionnaire with five traps, at 17,632 tokens;
the hierarchical crew and the scratch build 8 of 9 each, both on the same label. Getting there took three repairs
underneath the roles and tasks: CrewAI 0.193 describes tools in the prompt and parses text back, and gpt-oss-120b —
trained to call tools natively — made a refused native call, then answered without reading anything and cited ids
it invented, so this project runs on **qwen3.8-27b**; a failure inside a parallel (`async_execution`) task makes
the crew wait forever, so runs get a deadline; and LiteLLM's retries gave up on qwen's 1,000-output-tokens-a-minute
limit, so 429s are waited out as Groq asks. CrewAI also describes forms to the model without their allowed values or
field descriptions, and is on telemetry by default.

**`04_autogen`** — without help, both builds got Q1 wrong the same way: the export's 70 duplicated orders
were never dropped and both reviewers approved. Given nine generic **procedures as skills** (three relevant),
both got it right. AutoGen's planner opened exactly the three that apply. The scratch planner never opened the
dedupe procedure, yet put dedupe in its plan from the catalogue line alone, so picking a skill and being helped
by it are separate things to measure. Q2 and Q3 the scratch build got right; AutoGen got Q2 right and was
stopped at its budget on Q3 just after the reviewer approved the right figure. Over Q1–Q3, AutoGen billed
56,624 tokens against the scratch build's 44,350, but AutoGen's own messages report only 36,682 of its
56,624, which is why `TokenUsageTermination` cannot enforce a budget: the limit has to live in the model
client. Its code executor hands model-written code every API key in the environment, and runs it with
Windows' default output encoding. It does not run inside a Jupyter kernel on Windows, and the code and its
output never reach the other agents. The scratch build is a case file on chapter 6's board, where the
evidence is a field the code runner owns.

**`05_llamaindex`**:

- **The one-shot engine.** It answered all 15 README-reader questions with a right word and an answer file cited (48,093 tokens). It turned away all 3 off-topic questions before calling a model. The floor that does this has a thin margin: 0.702 for the lowest question the cookbook answers against 0.624 for the highest it does not. The floor has to be applied to raw vector scores, because after fusion a score is rank arithmetic.
- **The agent, and the from-scratch build.** Only the agent was built twice. LlamaIndex's agent sends everything it has found on every call, so on a two-sided question its 8th request reached 8,172 tokens and Groq refused it. Its memory limit does not trim within a run. The scratch agent (chapter 2's loop, with chapter 5 choosing each call's context) kept every call near 4,300 tokens of context and answered both questions. That cost more tokens overall, not fewer.
- **Checked line by line,** that answer had every hazard right and real citations, but put three facts on the wrong side. That is what `07_langsmith` and `08_langfuse` evaluate claim by claim.
- **What it took to build.** The agent's tools needed four fixes, each found by watching it fail:
  - results sized to fit the request limit;
  - passages that say which lines of their file they are;
  - no floor on the agent's search;
  - a cap of eight tool calls with a final answer.
- **Behaviours to know.** Each is reproduced offline:
  - with no model set, LlamaIndex reaches for OpenAI;
  - a loaded index forgets its splitter;
  - `refresh_ref_docs` never removes a deleted file;
  - a first build resumes only if it was saved along the way;
  - `citation_chunk_size=512` splits passages into extra sources.

**`06_mem0`**:

- **What decided it was the writing, not the store.** Mem0 writes memory in a step of its own, after every check-in, with a separate model. In round 1 it caught every fact, including one mentioned in passing (Tom's stomach cramps). The scratch build (chapter 4's design: a profile whose fields are replaced, a run log, a notes journal) had the coach write as a side job of replying. It saved a race without its date, never wrote the race moving, put changed days in a note, and dated runs wrong. So at the last check-in it told Maya the old race date and the old days. Giving writing its own step (one `extract()` after the reply) fixed every named fact in round 2.
- **Mem0 only adds.** In 2.2.1, `add` never updates or deletes, although its docstring still says it does. Old and new facts stay side by side; the coach still picked the current ones, because changes are written as changes and each memory carries its date. Search returned up to its limit of 20 every check-in, and its 0.1 threshold filtered nothing.
- **By default Mem0 keeps the assistant's advice as memories:** 10 of Maya's 26 in round 1, and the coach twice took a planned run for one she had done. One custom instruction ("only what the runner says") removed them.
- **A form is only as good as its filling.** In round 2 the scratch write step still left a field stale when a change came under another heading ("gels worked" went into a run's note, so `food` still said water only), and invented detail (Tom's days as Monday to Friday). A stale field is shown as a current fact, which is worse than Mem0's two dated versions. Round 3 fixed both:
  - a required slot for every field;
  - every change checked in code against the runner's own words;
  - old values and the runner's messages kept in view.

  Its last replies to both runners had every named fact right; run dates were still a day or two off. The exact-words check also refused one correct fact (a quote with one word changed), and the kept messages crowded `Recall`'s three slots. It cost 38k Groq tokens, against 27k.
- **Round 4 undid those mistakes, and its general parts went into chapter 4:**
  - `Profile`: named facts that keep their previous value and dates;
  - `writing.py`: a slot per field, a check on every day, date, number and name, `day_of` for dates, and one retry instead of dropping.

  Every run was dated right for the first time; diet and race fuel stayed apart; nothing correct was thrown away. Still wrong:
  - a form that re-wrote unchanged values wiped the one step of history;
  - a note repeating a profile fact went stale;
  - one habit was never written down.

  40k Groq tokens.
- **Measured against a hand-written answer key** of 50 facts (`answer_key.py`, never shown to either system), for extraction, current-state accuracy and stale exposure (`baseline.py`).
  - **Round 5:** chapter 4's store rules, habits, a check that every statement is accounted for, notes retired when their field changes. Habits were saved for the first time, but copied old runs cost a second call each, so 65k tokens.
  - **Round 6:** only this message's runs, with copies dropped without a second call. Back to 41k, but the check compared values as written, refused Tom's race written as "2026-11-08", and his coach made a date up.
  - **Round 7:** the check compares what dates and numbers mean. 47 of 50 facts saved right, nothing wrong or invented, every current state right, no stale item shown, 44k tokens.
- **At their best, Mem0 (round 2) and the scratch build (round 7) both answered every current-state question right.**
  - **Mem0** saved 48 facts, but 17% of what it showed the coach was a fact already replaced. The coach sorted it out from the dates.
  - **The scratch build** saved 47 facts and showed nothing stale, because its stores replace and retire. It needed seven measured rounds to learn what to check.
- **Setting it up.** Mem0's extraction prompt is about 7,600 tokens, too big for Groq's free tier (8,000 per request), so Mem0's own model runs on Gemini flash-lite: about 90k input tokens per round. The coach stays on Groq. Full search needs Qdrant, `fastembed` and spaCy, and without them falls back to meaning-only with a single warning line. Other behaviours, each reproduced offline:
  - `timestamp` raises an error in the open-source build;
  - telemetry is on by default;
  - `get_all` stops at 20 without saying so;
  - the links between memories that the prompt asks for are thrown away.
- **Retrieval, measured.** In the nine check-ins, nearly every fact the coach lacked had never been saved right: only 2 of the misses were retrieval's. With about 100 older memories added per runner (`stress.py`, embeddings only):
  - **Mem0's single search** let 8 of 14 needed facts through, down from 13. It dropped Maya's current days, mornings and two runs, and kept her old ones.
  - **The scratch build's profile and run log** were untouched, because they are shown, not searched. Its notes lost both habits until three changes: chapter 4's `Recall` became hybrid by default, it searches one question at a time, and the coach shows habits always. Then 14 of 14 got through.
- **Added to chapter 4:**
  - `QdrantJournal`, the journal on one engine with `Journal`'s methods, so `Recall` runs on either;
  - `Profile`;
  - `writing.py`;
  - hybrid, per-question `Recall`. Chapters 4 and 5's notebooks pass `hybrid=False`, which is what they measured.

**`07_langsmith`**:

- **The loop worked.** Three rounds on the same 10 questions. The baseline answered 9; the LangGraph comparison failed twice, first on a reply Groq refused, then on a request over Groq's size limit. Four fixes in 05, each from a trace:
  - `open_file` names its file;
  - an unparseable reply is asked again;
  - every request is cut to 6,000 tokens;
  - the answer at the step limit is given what the run found. As LlamaIndex ships, that request carries none of it, so the answer is written blind.

  Round 3 answered the question correctly: no claim wrong by hand, and 5 of 7 points, the judge agreeing (71%). Each round ran once.
- **A small test set, because of the free tiers.** Ten questions, each run once a round. One round takes most of a day's Groq tokens, and judging it takes a day's judge requests. A real evaluation needs many more questions first, and, since an agent's runs vary, a few runs of each agent question to measure how consistently it succeeds. Every result here is one sample.
- **The judge, checked first.** Two calls to Gemini Flash: the answer's claims listed without the answer key, then each checked against the key and the quoted lines. Asked in one call, it rewrote the answer's claims into the key's shape and missed a wrong one. Measured on 10 hand-marked answers held out from tuning:
  - points made matched the hand marks;
  - it caught 2 of 5 wrong claims and raised 7 false alarms.

  So it scores coverage, and its error lists are leads for a person.
- **Free tiers decided more than the tools did.** Gemini Flash allows about 20 requests a day per model, and stopped offering the checked judge to new keys. The newest Flash was unreachable, qwen allows about one claim list a minute, and gpt-oss-20b caught none of the errors.
- **LangSmith and LlamaIndex.** LangSmith has no LlamaIndex integration of its own. Through OpenTelemetry, every LlamaIndex step is lost inside an experiment, because client and server derive span ids differently. So 07 traces at 05's seams: `wrap_openai`, and a wrapped `traceable`. A bare `traceable` adds a `config` argument to a tool's schema.

**`08_langfuse`**:

- **No code of ours on the spans.** Langfuse is an OpenTelemetry tracer, so LlamaIndex's own instrumentation shows all 165 steps of one answer, nested where they ran. One entry in the price table (Groq's $0.15 and $0.60 per million tokens) puts dollars on every call. `propagate_attributes(prompt=...)` links each call to its prompt version, and one metrics query answers cost, tokens and time by version. All of 08 came to about $0.055.
- **Two things to know when reading the numbers.** One model call is recorded as four "generations", so counts are four times the calls while sums are right. New organizations must read traces through the v2 observations API.
- **Prompt v2 against v1,** on the three agent questions, one run each: a size set by Groq's free tier (three questions twice is about a day's tokens), so each result is one sample. Points made were the same on two questions; on the LangGraph comparison, the question v2 was written from, it went from 71% to 86%. v2 cost $0.034 against $0.021.
- **Running for real found two more limits, both fixed in 05:**
  - Under Groq's 8,000-tokens-a-minute limit, calls come about 45 seconds apart, so 300 seconds was too short for a long question. It is 900 now.
  - At the step-limit answer, gpt-oss kept calling tools when the run's results came as tool messages, even when told plainly that none were left. As plain text, it answered.

## Versions, and what it took to install them

All eight frameworks share the repo's one environment — `environment.yml` is still the single source
of dependencies. Getting there was a finding of its own:

| | Version | Note |
|---|---|---|
| LangChain (`langchain-core`, `langchain-openai`) | 1.6.x | `langchain` itself is not installed; nothing here needs it |
| LangGraph | 1.2.12 | |
| CrewAI | **0.193.2** | deliberately old — see below |
| AutoGen (`autogen-agentchat`, `autogen-ext`) | 0.7.5 | last released 2025-09-30; Microsoft Agent Framework is its successor |
| LlamaIndex (`llama-index-core`) | 0.14.x | |
| Mem0 (`mem0ai`) | 2.2.1 | |
| LangSmith | 0.14.1 | |
| Langfuse | 4.15.6 | |

**CrewAI is pinned to 0.193.2 (September 2025).** Every CrewAI release from 0.201 pins
`chromadb~=1.1`, and from 1.7 also `mcp~=1.28`. Installing a current one downgrades `mcp` from 2.2
to 1.28, which breaks `07_mcp`, and `chromadb` from 1.5.9 to 1.1, which `04_memory`'s store is built
on. 0.193.2 is the newest release whose pins fit. A framework's dependency pins are part of what it
costs to adopt — here, either an old version or a second environment.

The rest installed together with small moves: `protobuf` 7 → 5.29 (AutoGen pins it), `onnxruntime`
1.29 → 1.22, `websockets` 17 → 16. Chapters 4 and 7 were re-checked afterwards: every module imports,
Chroma works, and an MCP round trip succeeds.

**Added for `07_langsmith` and `08_langfuse`:** `openinference-instrumentation-llama-index` 4.5.4, LlamaIndex's
OpenTelemetry instrumentation, with `openinference-instrumentation`, `openinference-semantic-conventions` and
`opentelemetry-instrumentation` 0.66b0. They are pinned to the OpenTelemetry versions already installed, so nothing
else moved.

**Added for `06_mem0`:** `fastembed` 0.9.0 (BM25 for Mem0's keyword search on Qdrant), and `spacy` 3.8.16 with its
`en_core_web_sm` model (the names Mem0 links memories by). New packages only; nothing installed moved.

## What this chapter needs

- **Groq**, as everywhere — every project runs on `openai/gpt-oss-120b` except `03_crewai`, which runs on
  `qwen/qwen3.8-27b` (its free-tier limit is 1,000 output tokens a minute, so a crew takes minutes)
- **Gemini**, for embeddings, as in `04_memory` (LangChain's retrieval uses chapter 4's `embed()`), and in
  `06_mem0` also for Mem0's own model, `gemini-3.5-flash-lite`
- **LangSmith and Langfuse accounts**, for notebooks 7 and 8: `LANGSMITH_API_KEY`,
  `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env` (`LANGFUSE_HOST` is the
  deprecated name); `GEMINI_API_KEY_JUDGE` optionally, for 07's judge. All are listed in `.env.example`. Both are cloud
  services, so traces — prompts and outputs included — leave the machine. Langfuse can be
  self-hosted, but needs Docker, which this machine does not have

## Running it

Each folder's tests run offline and cost nothing:

```
cd 08_frameworks/02_langgraph && python test_ap.py
cd 08_frameworks/01_langchain && python test_claims.py
cd 08_frameworks/05_llamaindex && python test_ask.py
cd 08_frameworks/06_mem0 && python test_coach.py
```

The notebooks run on the `agentic-forge` kernel from inside their folder. Live sections spend real
tokens — roughly 20k per build per project — against Groq's daily cap, which is why every project
is tested against stand-ins first, and why a fix re-runs only the cells it touches.
