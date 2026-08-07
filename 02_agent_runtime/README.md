# Agent Runtime

**What:** A single runtime that executes different agentic design patterns —
ReAct, Plan & Execute, ReWOO, Reflexion, Debate, Tree of Thoughts — without
hardcoding any of them. The runtime only walks a declared loop and passes a
conversation from one component to the next; every pattern is a different
loop built from the same few primitives.

**Why:** Agentic patterns get taught as a catalogue of unrelated
architectures. They aren't. Chain of Thought and ReAct differ by one integer.
Self-Consistency and Debate have identical component lists and differ only in
two booleans decided at runtime. Building one runtime that supports all of
them, instead of one hardcoded loop per pattern, is the actual engineering
problem this chapter solves.

**The core idea:** every agent runtime does the same three things, in some
order:
- **Observe** — bring something in (a user event, a tool's raw effect) and
  turn it into something the conversation can read. Never decides.
- **Think** — reason over the conversation and produce a decision, a plan, or
  a critique. The only component allowed to reason.
- **Act** — change the outside world (call a tool, start a sub-agent, finish).
  Never decides — only executes what a Think already decided.

`runtime.py` never knows which pattern it's running. It doesn't import
`Observe`, `Think`, `Act`, or even `Conversation` — only that whatever is in
its loop has a `.run(conversation)`.

## Notebooks

1. `01_workflows_vs_agents.ipynb` — what actually distinguishes an AI
   workflow (predetermined pipeline, each step expects a fixed shape of
   output) from an AI agent (a loop that decides what runs next from the real
   content of what it observes) ✅
2. `02_the_agent_loop.ipynb` — pulls chapter 1's `asend_with_tools()` apart
   into `Runtime`, `Observe`, `Think`, `Act`; why raw tool output never lands
   in `messages`; why translating our own message roles back into the API's
   takes real work; ReAct as a declared loop; then four budgets, doom-loop
   detection, and a graceful stop ✅
3. `03_retries_and_reflection.ipynb` — two ways the loop notices something is
   wrong and goes again: `DebugThink` repairing tool arguments inside `Act`
   where the conversation never sees it, and `ReflectionThink` critiquing an
   answer without writing one. Builds `Loop` ✅
4. `04_planning.ipynb` — `PlannerThink`; `$stepN` references for arguments
   that don't exist yet; decided vs. executed steps; `Loop(until=plan_done)`
   and replanning; then reading the dependency graph straight out of the
   plan's own references and running it through `Graph` ✅
5. `05_sub_agents.ipynb` — delegation as a tool call; why a branch gets a
   self-contained goal and nothing else; splitting a budget across branches;
   `persistent` × `broadcast` as four configurations, including debate;
   self-consistency; `SelectThink` and beam search ✅
6. `06_agent_patterns.ipynb` — eleven named patterns as declared loops over
   everything above, every one constructed for real, and an honest account of
   the two things these components can't express ✅

**A note on notebook count.** The repo's guideline is 2–5 notebooks per
chapter. This one has 6, for the opposite reason `01_foundations` has 9: not a
collection of independent primitives, but one topic that turned out to be
large. Each notebook still builds one runnable thing against one real failure.

## What this chapter does with chapter 1

Almost nothing here is invented from scratch. Most of it is a chapter 1
primitive used again, extended, or deliberately departed from:

| From `01_foundations` | What `02_agent_runtime` does with it |
|---|---|
| `Conversation` (nb 2) | Still the shared state, but gains roles the API has never heard of — `decision`, `plan`, `critique` — and non-message slots: `pending`, `step_outputs`, `remaining_budget`, `branches` |
| `notes`, a non-message slot (nb 9) | The precedent for those slots: not everything the agent knows belongs in what the model is shown |
| `asend_with_tools()` (nb 9) | Split into four components. Its `for _ in range(retries)` becomes `Runtime`'s `while not finished` |
| Native tool-call messages (nb 6) | `Think._build_messages()` has to *reconstruct* them — the `tool_calls`/`tool_call_id` pairing is the signal that work is done, and inventing new roles destroyed it |
| `execute_tool_call()` (nb 6) | Reused directly inside `Act.execute()`, Pydantic validation and all — never reimplemented |
| `resolve_tools()`'s retry (nb 6) | Rethought. It re-asked the same model with the same context; `DebugThink` instead gets the JSON schema and every prior failed attempt, and its repairs never touch the conversation |
| `extract()` + Pydantic (nb 5) | How every non-tool-calling Think returns structure: `Plan`, `ContinueOrFinish`, `Reflection`, `Selection` |
| `count_tokens()` (nb 3) | Same measurement, opposite purpose: chapter 1 used it to decide what to *drop* from history, here it decides when to *stop* a run |
| `temperature` / `top_p` (nb 1) | Per-branch, so several sub-agents can answer one question differently — the knobs used to generate disagreement rather than suppress it |
| `asyncio.gather` (nb 8) | **Deliberately not used.** `Graph` and forking use `ThreadPoolExecutor`, because notebooks already run an event loop and library code starting its own conflicts with it. The work is blocking I/O, so threads lose nothing |
| `Agent` / Loopy (nb 9) | The thing `Runtime` generalizes: one fixed loop becomes a declared list |

Two changes were made *to* chapter 1, both forced by this chapter:

- `llm.py`'s clients pass `max_retries=8`. The SDK already retries HTTP 429s
  with backoff, but its default of 2 assumes one call at a time — and this
  chapter runs branches and tool calls concurrently, which bursts straight
  through a free-tier tokens-per-minute cap.
- `llm.py` grew a provider layer (`llm.use("gemini")`, see the chapter 1
  README). A single free tier isn't enough for a chapter that forks: Groq's
  100k tokens/day can go in one run of notebook 5. Switching providers is now
  a one-liner rather than a reason to stop for the day.

If a run here dies on a rate limit, `llm.use(...)` to a provider with headroom
and re-run the notebook — it is not a code failure, and nothing else changes.

## Modules

`runtime.py`, `observe.py`, `think.py`, and `act.py` were built incrementally
as the notebooks needed them — same as every other chapter, no code before
there's something to prove it against.

- `runtime.py` — `Runtime` (declared loop, `repeat_from`, budgets, doom-loop
  detection, graceful stop), `Loop` (bounded inner cycle, `count` or `until`),
  `Graph` (dependency-scheduled plan execution)
- `observe.py` — `Observe`: raw effect → observation, via a deterministic
  extractor or an LLM fallback; merges forked branch results; resolves
  `$stepN` references for the next planned step
- `think.py` — `Think` (tool call, delegation, or answer), `PlannerThink`,
  `ReflectionThink`, `SelectThink`, and `DebugThink`, which is deliberately
  *not* a loop component
- `act.py` — `Act`: executes `respond`, `tool_call`, `fork`, `advance_branch`,
  and `prune` decisions. `execute()` is pure, so `Graph` can call it from
  several threads at once
