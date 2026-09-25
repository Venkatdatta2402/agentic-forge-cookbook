# Orchestration

**What:** Several agents that are *different from each other* — each with its own role, system
prompt and tools — working on one task. How the work gets handed to the right one, how they
talk and what they share, and how their answers get combined into one.

**Why:** `02_agent_runtime` already has sub-agents. A Think can call `delegate`, the work splits
into branches, each branch runs its own loop with a clean context, and the results come back as
one block. So the obvious question is what this chapter adds, and the answer is not context
isolation — branches have that already.

What branches do not have is **difference**. Every branch is seeded with its parent's system
message (`Act._seed_branch`) and built by the same `branch_runtime` factory, so it has the same
persona, the same instructions and the same tools as the agent that spawned it. It is a clone
handed a narrower goal. The only knobs chapter 2 had for making branches disagree were
`temperature` and `top_p` — disagreement had to be *manufactured*, because nothing about the
branches was actually different.

In a multi-agent system the agents are different because they were designed that way:

| | Sub-agents (`02_agent_runtime`) | Multi-agent (this chapter) |
|---|---|---|
| Who defines them | the model, at runtime, by writing a goal | you, ahead of time, as a roster |
| System prompt | copied from the parent | its own |
| Tools | the parent's | its own — often fewer |
| Context | clean, goal only | clean, goal only |
| Why they disagree | sampling randomness | they know and are allowed different things |
| What choosing means | nothing — they are interchangeable | picking the agent that fits the job |

Context isolation is in both columns, which is exactly why it is not the distinguishing idea.

**The core idea:** an agent is a *definition* — a role, a system prompt, a set of tools — and
orchestration is everything that follows from having more than one definition. Selection
only becomes a real decision once there is something to select between. Communication only
needs a schema once the two ends are not the same agent. Aggregation only needs more than a
vote once the voters know different things.

Worth noticing before starting: the hook for this was already in chapter 2.
`branch_runtime(branch)` receives the branch, so a factory *could* build a different runtime per
branch. Nothing ever did, because the model was inventing branch ids on the spot and there was
no roster for it to name. The gap is small in code and large in design.

## Notebooks

1. `01_single_vs_multi_agent.ipynb` — what makes a system multi-agent, starting from the
   table above. `_seed_branch` shown copying the parent's system prompt, and `delegate` shown
   having no field for a role or tools, both without a model call. An agent as a definition
   (name, persona, tools) on chapter 2's runtime. Then a product recall that needs five people
   (support lead, PR director, lawyer, engineer, social media manager) done three ways on
   gpt-oss-20b: one agent and a chapter 2 fork sharing one persona with per-piece instructions,
   against five specialists. The one agent put the internal-only supplier into the press
   statement and the regulator notice, and told the public "No fire risk". The fork, each
   branch a copy of the parent with one piece and a goal restating its rules, did about as well
   as the specialists, at about twice their cost (1× / 6.5× / 3.3×). Ends on which to use: one
   agent for small jobs, a fork when the job is too big for one context but one persona can do
   every part, specialists when the parts need different people ✅
2. `02_delegation_and_supervision.ipynb` — a supervisor over a fixed roster, built from
   **agents as tools**: a worker is an ordinary chapter 3 `Tool`, so the supervisor is a chapter
   2 loop and nothing new is invented. What a supervisor knows about an agent is its
   *description*, not its persona. Routing five real messages, then the failure that matters: a
   customer reports a bulging unit from a batch that is not recalled, the supervisor sends it to
   support because a customer email goes to support, and support — whose lookup only answers
   "is this serial on the list" — replies "for now, you're safe". The unit was built from the
   recalled cell lot, which only engineering can see. Two fixes measured over 8 trials each:
   sharpening the descriptions changed the first choice **0 of 8** times (p = 1.0), a rule in
   the supervisor's own prompt **8 of 8** (p ≤ 0.001). Then **handoffs** against delegation on
   one customer chat: 2.5× cheaper, the specialist's own voice, but the supervisor is out of it.
   Replies come back as plain strings throughout, which is what notebook 3 takes up ✅
3. `03_communication_state_and_collaboration.ipynb` — the shapes between agents. A reply becomes
   a form the agent fills in by calling `submit`, so `visibility`, `confident`, `unknowns` and
   `ask_next` exist as fields a program can act on — self-reported, so a place to look rather
   than a guarantee. A request becomes typed arguments, so notebook 2's missing serial number is
   refused before any agent runs, which *is* a guarantee. Then a shared board: in memory, not a
   log and not `04_memory`'s store, rendered into each agent's context and written through a
   tool that knows who owns which field. Engineering records the cell lot; support answers the
   customer from it in two calls without asking engineering at all — and forgets to record what
   it sent, which is the difference between reading (the code does it) and writing (the agent
   has to choose to). Two agents write one unowned field and the last one wins silently. Finally
   three specialists answer "widen the recall?" at once: majority says no and is wrong, weights
   tie at 3-2-1 and flip at 4-2-1, and the rule that works asks which agent actually checked
   something ✅
4. `04_orchestration_patterns.ipynb` — the patterns, on two axes rather than as a list of
   architectures: **organisation** (who can talk to whom), **control** (who runs next) and
   **execution shape** (how the work moves). Shared state answers only the first, which is why
   notebook 3's board section turns out to have had no control mechanism at all — the cells were
   its scheduler. Peer-to-peer: support holds its colleagues as tools and asks nobody, **0 of 4
   runs**, until its own task tells it to; the same shape notebook 2 measured, and it survives
   the move to the bigger model. A manager picking speakers through a form, which stops after one
   speaker because the first specialist answered the question. The same board driven by an
   **event** instead, where a write fires the lawyer and nothing spends a call on deciding — plus
   the version of that watcher which tested the text for "7731" and lost a `risk: no`. A debate
   that produces agreement, at 16k tokens for four turns, judged by an agent whose `evidence`
   field reads "none". Ends on hybrids as one choice per column, not a fourth category. Runs on
   **gpt-oss-120b**: on 20b the agents held tools without using them and every pattern looked
   alike ✅
5. `05_build_a_multi_agent_system.ipynb` — one hybrid desk out of the whole chapter: a supervisor
   routing (hierarchical), support asking engineering itself (peer-to-peer), a board holding the
   case (blackboard), an event firing the lawyer when `risk` is written, a fan-out deciding the
   question nobody owns, and a gate in plain code in front of anything that leaves. Run once, it
   answered the customer well and **silently skipped two of its own patterns**: engineering never
   called `record`, so `risk` was never written, so the event and the fan-out never ran. Moving
   those fields into the *form* engineering must fill in — and letting the code do the writing —
   fires all five, for 26% more tokens. Then the failure half, three quarters of it offline
   against stubs: a worker that raises, one that hits its budget and returns "Stopping: reached
   the maximum…", one that will not use the form, all arriving as the same `Response` shape; a
   `sendable` gate whose four checks are four real failures from earlier notebooks; and watchers
   that feed each other, refused at depth 3. Ends on what the system still gets wrong, including
   an event that fired twice for one report ✅

## What this chapter does with earlier chapters

| From earlier chapters | What `06_orchestration` does with it |
|---|---|
| `delegate`, `_fork`, `_seed_branch` (2) | The baseline this chapter is measured against. Branches are clones; everything here starts from making them not |
| `branch_runtime(branch)` (2) | The existing hook for a different runtime per branch — never used for that until now |
| `temperature` / `top_p` per branch (2) | How chapter 2 manufactured disagreement. Here disagreement comes from roles instead |
| Self-consistency, debate, `SelectThink` (2) | Aggregation over identical agents. Notebook 3 covers the case they cannot: agents that know different things, where counting them is the wrong move |
| Budgets split across branches (2) | Still needed, and harder: workers with different jobs need different shares |
| `Tool` / `Registry` (3) | Each agent gets its own subset of tools — a narrower registry is what makes a role real, not just its prompt |
| `@tool` wraps any function (3) | Including one that runs a whole agent — a worker wrapped as a `Tool`, so the supervisor's loop is an ordinary chapter 2 loop |
| Context selection (5) | Deciding what each agent is shown, now that different agents need different slices |

## Modules

Built incrementally as the notebooks need them, same as every other chapter — no code before
there is something to prove it against.

- `recall.py` — the scenario every notebook from 2 on works from: the brief, the specialists'
  personas, what each is *described* as to the others, and a small database behind their tools.
  A report can go either way, and only a lookup says which: batch AR2-2409's first 1,800 units
  carry the recalled cell lot
- `supervisor.py` — `Agent` (moved out of notebook 1), `as_tool`, `supervisor`, and `Chat`, a
  conversation one agent at a time owns and can hand to another. A handoff is an ordinary tool
  whose effect is to stop the current agent through chapter 2's `cancel` hook
- `meter.py` — notebook 1's `Meter`, moved here when notebook 2 needed it too

- `messages.py` — `Response`, the form an agent fills in by calling `submit`; `form_tool`, which
  turns any pydantic model into that form; and `answer`, which runs an agent that finishes by
  filling one in
- `state.py` — `Shared`, the board: fields, an owner per field, readers per field, and the write
  tool bound to one agent's name. `on_board` puts the view into an agent's context
- `aggregate.py` — `Opinion`, asking several agents at once, and the ways of combining what they
  say: `majority`, `weighted`, `decisive`, `disagreement`
- `failures.py` — `attempt` (a worker's failure comes back as a form, not an exception),
  `sendable` (whether a reply may leave the building, and why not), `guard_watchers` (a board
  whose events feed each other is refused rather than left to recurse)
- `coordination.py` — the arrangements: `peers` (peer-to-peer), `pipeline` (fixed order),
  `GroupChat` (a manager picking speakers through a form), `debate` (fixed alternation). Named
  `coordination` because `02_agent_runtime` already has a `patterns.py`, and two modules with one
  name on the path is a bug waiting for whoever imports them in the wrong order

Notebook 1 keeps its own copy of `Agent` and `Meter`, written inline before there was a second
notebook to share them with. Its saved outputs came from that copy, so it stays as it is.

## Cost

This chapter multiplies model calls: a supervisor plus three workers is at least four loops, and
asking three specialists one question in notebook 3 is three whole agent runs. Everything runs on
**`openai/gpt-oss-20b`**, which is cheaper than the 120b model and has its own daily quota; its
output ceiling and rate limits are why `supervisor.Think` sets `max_tokens` and why `Agent`
allows 24 iterations, which is eight turns rather than eight calls.

Schemas, ownership, routing and tie-breaking are all deterministic, and get tested against a
stubbed model before anything is spent. Live calls are for what only a live model settles, such
as whether three specialists actually disagree. Anything claimed as a finding is measured over
repeats: notebook 2's routing rule is 8 trials per version, not one run each.
