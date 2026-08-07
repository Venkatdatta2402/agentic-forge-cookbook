# Agent Engineering

A learning repo for building agentic AI systems from scratch — one concept at a
time, with code and a notebook for each.

## How it's organized

Each numbered folder is a self-contained chapter. Inside, the same four
questions apply:

- **README.md** — what is it, why does it exist
- **notebooks/** — how it works
- **\*.py** — the implementation
- **examples/** — how to use it

Chapters build roughly in order, but each one should be usable on its own.

| Folder | Covers |
|---|---|
| `00_concepts` | Ideas and terminology — no code, just notebooks |
| `01_foundations` | Core Python needed later (async, typing, etc.) |
| `02_agent_runtime` | How a single agent executes (Observe/Think/Act) |
| `03_tools` | Giving an agent the ability to act on the world |
| `04_memory` | Short- and long-term state across turns |
| `05_retrieval` | Pulling relevant knowledge into context |
| `06_context` | Context engineering and window management |
| `07_orchestration` | Coordinating multiple agents |
| `08_mcp` | Model Context Protocol |
| `09_sandbox` | Safe code/tool execution |
| `10_guardrails` | Safety and constraint enforcement |
| `11_evaluation` | Measuring agent quality |
| `12_monitoring` | Observability in production |
| `13_frameworks` | Surveying existing agent frameworks |

`projects/` applies chapters together into full systems. `playground/` is
unstructured scratch space, added only if needed.

## Setup

```
conda env create -f environment.yml
conda activate agentic-forge
```
