# Agent Engineering

A learning repo for building agentic AI systems from scratch — one concept at a
time, with code and a notebook for each.

## How it's organized

Each numbered folder is a self-contained chapter. Inside, the same four
questions apply:

- **README.md** — what is it, why does it exist
- **notebooks/** — how it works
- **\*.py** — the implementation

Chapters build roughly in order, but each one should be usable on its own.

| Folder | Covers |
|---|---|
| `01_foundations` | Talking to a model at all — prompting, structured output, function calling, async |
| `02_agent_runtime` | How a single agent executes (Observe/Think/Act) |
| `03_tools` | Giving an agent the ability to act on the world |
| `04_memory` | What an agent keeps across turns, where it keeps it, how it gets it back |
| `05_context` | Context engineering and window management |
| `06_orchestration` | Coordinating multiple agents |
| `07_mcp` | Model Context Protocol — reaching tools that live in someone else's process |
| `08_frameworks` | Building one real project per agent framework — LangChain, LangGraph, CrewAI, AutoGen, LlamaIndex, Mem0, LangSmith, Langfuse — and the same project again from this repo's own parts |
| `09_guardrails` | Safety and constraint enforcement (not started) |

`projects/` applies chapters together into full systems. `playground/` is
unstructured scratch space, added only if needed.

## Setup

```
conda env create -f environment.yml
conda activate agentic-forge
```
