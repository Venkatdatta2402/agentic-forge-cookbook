# 06 · Orchestration

> Several agents that are *different from each other*, each with its own role, prompt and tools,
> working on one task: routing work to the right one, letting them share what they know, and
> combining their answers.

**Contents:** [The idea](#the-idea-in-one-picture) · [What you'll learn](#what-youll-learn) ·
[Notebooks](#notebooks) · [Key ideas](#key-ideas) · [Modules](#modules) ·
[How it connects](#how-it-connects) · [Running it](#running-it)

---

## The idea in one picture

Chapter 2 already has sub-agents, so what does this chapter add? It isn't context isolation,
because branches already have that. It's **difference**. A chapter 2 branch is a *clone* of its
parent, with the same persona, instructions and tools, handed a narrower goal. Disagreement had
to be manufactured with `temperature`.

| | Sub-agents (`02_agent_runtime`) | Multi-agent (this chapter) |
|---|---|---|
| Who defines them | the model, at runtime, by writing a goal | you, ahead of time, as a roster |
| System prompt | copied from the parent | its own |
| Tools | the parent's | its own, often fewer |
| Context | clean, goal only | clean, goal only |
| Why they disagree | sampling randomness | they know and are allowed different things |
| What choosing means | nothing; they're interchangeable | picking the agent that fits the job |

**An agent is a definition: a role, a system prompt and a set of tools.** Orchestration is
everything that follows from having more than one:

- **Selection** only becomes a real decision once there's something to choose between.
- **Communication** only needs a schema once the two ends are different agents.
- **Aggregation** only needs more than a vote once the voters know different things.

Every notebook uses one scenario: the **recall of the Aria 2 wireless headphones**, whose battery
can overheat, handled by a support lead, an engineer, a lawyer, a PR director and a social-media
manager. A customer report can go either
way, and only a lookup tells which. Batch AR2-2409's first 1,800 units carry the recalled cell
lot, and only engineering can see that.

## What you'll learn

- When one agent is enough, when a fork is, and when you need specialists
- A **supervisor** over a fixed roster, built from agents wrapped as ordinary tools
- **Delegation vs handoff**: who stays in charge of the conversation
- Structured messages between agents, a **shared board** with an owner per field, and
  aggregation that asks *who actually checked*
- Orchestration patterns on three axes (**organisation, control, execution shape**) instead of a
  list of architectures
- How a multi-agent system fails, and a gate in plain code before anything leaves

## Notebooks

| # | Notebook | What it shows |
|---|---|---|
| 1 | [`01_single_vs_multi_agent`](notebooks/01_single_vs_multi_agent.ipynb) | One agent vs a chapter 2 fork vs five specialists on the recall. The single agent leaked the internal-only supplier into the press statement and said "No fire risk"; the fork matched the specialists at about twice their cost (1× / 6.5× / 3.3×) |
| 2 | [`02_delegation_and_supervision`](notebooks/02_delegation_and_supervision.ipynb) | A supervisor routing to agents-as-tools; a misroute that told a customer "you're safe" when they weren't; handoffs vs delegation |
| 3 | [`03_communication_state_and_collaboration`](notebooks/03_communication_state_and_collaboration.ipynb) | Replies as forms, typed requests, a shared board with owned fields, and three ways of combining opinions |
| 4 | [`04_orchestration_patterns`](notebooks/04_orchestration_patterns.ipynb) | Peer-to-peer, a manager picking speakers, event-driven, debate, on gpt-oss-120b |
| 5 | [`05_build_a_multi_agent_system`](notebooks/05_build_a_multi_agent_system.ipynb) | One hybrid desk built from the whole chapter, then its failure modes tested offline |

## Key ideas

### A supervisor knows an agent by its *description*
Workers are wrapped as ordinary chapter 3 `Tool`s, so the supervisor is just a chapter 2 loop.
What the supervisor knows about each worker is its tool description, not its persona.

That caused the failure that matters. A customer reported a bulging unit from a batch that wasn't
on the recall list. The supervisor sent it to support, because customer emails go to support.
Support's lookup only answers "is this serial on the list?", so it replied "for now, you're safe".
The unit was built from the recalled cell lot, which only engineering could see. Two fixes were
measured over 8 trials each:

| Fix | First choice changed |
|---|---|
| Sharper agent descriptions | **0 of 8** (p = 1.0) |
| A rule in the supervisor's own prompt | **8 of 8** (p ≤ 0.001) |

### Delegation vs handoff
**Delegation** keeps one agent in charge: it asks another agent, relays the answer, and every turn
comes back through it. A **handoff** moves the conversation itself, so the new agent talks to the
person directly from then on. A handoff is an ordinary `transfer_to_<name>` tool that stops the
current agent through chapter 2's `cancel` hook. Notebook 2 measured both. Then **handoffs**
against delegation on one customer chat: 2.5× cheaper, the specialist's own voice, but the
supervisor is out of it.

### Structure between agents
- **Replies as forms.** An agent finishes by calling `submit`, so `visibility`, `confident`,
  `unknowns` and `ask_next` are fields a program can act on. They're self-reported, so treat them
  as a place to look, not a guarantee.
- **Requests as typed arguments.** A request missing its serial number is refused before any
  agent runs, and that *is* a guarantee.
- **A shared board.** Each field has one owner. Reading is injected into each agent's context;
  writing is a tool bound to the agent's name, so ownership is enforced. Support answered a
  customer from engineering's field without asking engineering at all, and then forgot to record
  what it had sent. The code does the reading; writing is something the agent has to choose to do.
- **Combining opinions.** On "widen the recall?", the majority said no and was wrong. Weights tied
  at 3-2-1 and flipped at 4-2-1. The rule that worked asks **which agent actually checked
  something**.

### Patterns are choices on three axes
**Organisation** (who can talk to whom), **control** (who runs next) and **execution shape** (how
the work moves). A "hybrid" is just one choice per axis. Things found on the way:

- Support holding its colleagues as tools asked nobody, **0 of 4 runs**, until its own task told
  it to.
- A board driven by **events** (a write fires the lawyer) spends no calls deciding who's next.
- A debate produced agreement at 16k tokens for four turns, judged by an agent whose `evidence`
  field read "none".

### Make the important steps part of the form
Run once, the full desk answered the customer well and **silently skipped two of its own
patterns**. Engineering never called `record`, so `risk` was never written, so the event and the
fan-out never ran. Moving those fields into the **form** engineering must fill in, and letting the
code do the writing, fired all five for 26% more tokens.

## Modules

| File | What's in it |
|---|---|
| [`recall.py`](recall.py) | The scenario: the brief, each specialist's persona and description, and a small database behind their tools |
| [`supervisor.py`](supervisor.py) | `Agent`, `as_tool`, `supervisor`, and `Chat` (a conversation one agent at a time owns and can hand off) |
| [`messages.py`](messages.py) | `Response` (the form an agent fills in), `form_tool`, `answer` |
| [`state.py`](state.py) | `Shared`, the board: fields, an owner per field, readers per field, and a write tool bound to one agent; `on_board` |
| [`aggregate.py`](aggregate.py) | `Opinion`, asking several agents at once, and `majority`, `weighted`, `decisive`, `disagreement` |
| [`coordination.py`](coordination.py) | `peers`, `pipeline`, `GroupChat` (a manager picking speakers), `debate` |
| [`failures.py`](failures.py) | `attempt` (a failure comes back as a form), `sendable` (may this reply leave, and why not), `guard_watchers` (refuses events that feed each other) |
| [`meter.py`](meter.py) | `Meter`, which counts tokens and calls |

## How it connects

**Builds on:** chapter 2's `delegate` and branches (the baseline everything here is measured
against), chapter 3's `@tool` (an agent wrapped as a tool), and chapter 5's context selection
(what each agent is shown).

**Not to be confused with chapter 4's memory.** The board holds the working state of *one case*
for minutes, in an ordinary object. Chapter 4's stores keep what must outlive a run, on disk.

**Used by** `08_frameworks`: CrewAI, AutoGen and LangGraph are each compared against a scratch
build on this chapter's board.

## Running it

```bash
conda activate agentic-forge
# .env needs GROQ_API_KEY
```

This chapter multiplies model calls: a supervisor plus three workers is at least four loops. Most
of it runs on **`openai/gpt-oss-20b`**, which is cheaper and has its own daily quota; notebook 4
runs on gpt-oss-120b, because on 20b agents held tools without using them. Schemas, ownership,
routing and tie-breaking are tested against a stubbed model first. Findings are measured over
repeated trials, not single runs.

## Going deeper

[`NOTES.md`](NOTES.md) has the full write-up of each notebook and every measurement.
