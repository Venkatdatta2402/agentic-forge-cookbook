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
| `06_mem0` | Mem0 | a running coach that remembers each runner across weeks of sessions | yes | planned |
| `07_langsmith` | LangSmith | tracing, a dataset and evaluations for the `05_llamaindex` cookbook assistant, including an answer judge that holds up | — | planned |
| `08_langfuse` | Langfuse | the same assistant, traced and judged with Langfuse, against LangSmith: costs, prompt versions, scores, monitoring | — | planned |

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
- **Checked line by line,** that answer had every hazard right and real citations, but put four facts on the wrong side. That is what `07_langsmith` and `08_langfuse` evaluate claim by claim.
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

## What this chapter needs

- **Groq**, as everywhere — every project runs on `openai/gpt-oss-120b` except `03_crewai`, which runs on
  `qwen/qwen3.8-27b` (its free-tier limit is 1,000 output tokens a minute, so a crew takes minutes)
- **Gemini**, for embeddings, as in `04_memory` (LangChain's retrieval uses chapter 4's `embed()`)
- **LangSmith and Langfuse accounts**, for notebooks 7 and 8: `LANGSMITH_API_KEY`,
  `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` in `.env`. Both are cloud
  services, so traces — prompts and outputs included — leave the machine. Langfuse can be
  self-hosted, but needs Docker, which this machine does not have

## Running it

Each folder's tests run offline and cost nothing:

```
cd 08_frameworks/02_langgraph && python test_ap.py
cd 08_frameworks/01_langchain && python test_claims.py
cd 08_frameworks/05_llamaindex && python test_ask.py
```

The notebooks run on the `agentic-forge` kernel from inside their folder. Live sections spend real
tokens — roughly 20k per build per project — against Groq's daily cap, which is why every project
is tested against stand-ins first, and why a fix re-runs only the cells it touches.
