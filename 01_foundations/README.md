# 01 · Foundations

> The building blocks everything else in this repo is made of: one model call, a conversation,
> tokens, prompts, structured output, tool calling, streaming and async Python.

No agents yet. This chapter makes each primitive work on its own, against a real model, so that
from `02_agent_runtime` onward a chapter can be about agent behaviour and nothing else.

**Contents:** [The idea](#the-idea-in-one-picture) · [What you'll learn](#what-youll-learn) ·
[Notebooks](#notebooks) · [Key ideas](#key-ideas) · [Modules](#modules) ·
[Which model](#which-model-the-outputs-came-from) · [How it connects](#how-it-connects) ·
[Running it](#running-it)

---

## The idea in one picture

A language model call is a function. Messages go in; text, JSON or a request to call a tool comes
out. It remembers nothing between calls.

```
    your code                                   the model (stateless)
 ┌──────────────────────────┐    messages    ┌──────────────────────┐
 │ history, tools, rules    │ ─────────────► │ text, JSON, or a     │
 │ run tools, validate,     │ ◄───────────── │ tool call            │
 │ append the reply         │     reply      └──────────────────────┘
 └──────────────────────────┘
```

Everything an "agent" seems to do, such as remembering, using tools or following rules, is code on
the left side of that picture. This chapter writes that code one piece at a time.

## What you'll learn

- How to make a chat call and what `temperature`, `top_p` and `max_tokens` actually change
- Why the model is **stateless**, and how a `Conversation` object keeps history for it
- How to count tokens, fit history into a budget, and what a call really costs
- How to write prompts that hold up, including against a real prompt-injection attempt
- How to get **structured output** you can trust, and what a schema can't promise
- How **function calling** works, and the two different ways a tool call fails
- How to stream tokens, and how `asyncio` runs calls concurrently
- How the pieces combine into a first small agent

## Notebooks

Each notebook ends with **What you built**, and the next one opens with **Prerequisites**, so the
chapter reads in order.

| # | Notebook | What it shows |
|---|---|---|
| 1 | [`01_your_first_llm`](notebooks/01_your_first_llm.ipynb) | One chat completion call, and what `temperature`, `top_p` and `max_tokens` do |
| 2 | [`02_conversations`](notebooks/02_conversations.ipynb) | System, user and assistant roles; proving the model is stateless; the `Conversation` helper |
| 3 | [`03_tokens_and_context`](notebooks/03_tokens_and_context.ipynb) | Tokenization, real versus estimated counts, the context window, `trim_to_budget`, real cost and latency |
| 4 | [`04_prompting`](notebooks/04_prompting.ipynb) | Instructions, few-shot examples, delimiters (with a real prompt-injection demo), `PromptTemplate` |
| 5 | [`05_structured_outputs`](notebooks/05_structured_outputs.ipynb) | What JSON mode really guarantees, a real parsing failure, Pydantic validation, the `extract()` retry loop |
| 6 | [`06_function_calling`](notebooks/06_function_calling.ipynb) | Tool schemas and arguments, two real failure categories, `execute_tool_call()` and a retrying `resolve_tools()` |
| 7 | [`07_streaming`](notebooks/07_streaming.ipynb) | Streaming tokens, the chunk structure, a measured latency comparison, streaming tool calls |
| 8 | [`08_asyncio`](notebooks/08_asyncio.ipynb) | The event loop, a real speedup from `gather`, why `TaskGroup` handles failure better, `Semaphore`, cancellation (no LLM calls) |
| 9 | [`09_build_an_agent`](notebooks/09_build_an_agent.ipynb) | Everything above combined: a `Conversation` that runs tools, an `Agent` called Loopy, and `extract()` as a `/summary` command |

The repo usually has 2–5 notebooks per chapter. This one has 9 because it's a set of separate
primitives rather than one deep topic.

## Key ideas

### The model is stateless, so history is yours to keep
Each call sees only what you send it. Notebook 2 proves this, then builds `Conversation`, which
keeps the message list and resends it. Every later chapter's "memory" starts from this list.

### Tokens are the budget for everything
Cost, speed and what fits in one call are all counted in tokens. Notebook 3 counts them for real
and builds `trim_to_budget()`, which drops the oldest turns to fit. It also shows the weakness:
trimming by age throws away rules given at the start. Chapter 5 deals with that.

### A schema controls the shape of an answer, not whether it makes sense
`extract()` asks the API to enforce a JSON schema first. If the model doesn't support that mode,
it falls back to plain JSON. Either way it then validates the result with Pydantic. If validation
fails, it tells the model exactly what was wrong and asks again. Validation is still needed when
the schema is enforced. A schema guarantees the right field names and types, but it can't check
the meaning: "one score per candidate" is not something JSON Schema can say.

### A tool call is a proposal; your code does the work
The model never runs anything. It asks for a tool by name with arguments, your code checks the
arguments and runs the function, and the result goes back as a message. Notebook 6 shows two ways
this fails:

- **The API rejects the call** (for example, a tool that doesn't exist). Groq checks this
  server-side.
- **The arguments are wrong for your domain** (for example, a value that must not be zero).
  Pydantic types and `field_validator`s catch these.

### Run independent calls at the same time
Model calls spend most of their time waiting on the network. Notebook 8 measures the speedup from
running them concurrently. It also shows why `TaskGroup` is safer than `gather` when one call
fails, and how a `Semaphore` limits how many run at once.

## Modules

| File | What's in it |
|---|---|
| [`llm.py`](llm.py) | The OpenAI SDK pointed at Groq. `chat()`, streaming helpers, `extract()` (structured output with validation and retries), `execute_tool_call()`, `resolve_tools()`, and `tool_call_failure()`, which turns Groq's rejection of a malformed tool call into a message the model can act on |
| [`chat.py`](chat.py) | `Conversation` (history, tool loop, `notes`, `resources`), `count_tokens()`, `trim_to_budget()`, `ConversationMemory`, and the `Agent` wrapper (Loopy) |
| [`prompts.py`](prompts.py) | `PromptTemplate`, which refuses to format when a variable is missing |

Every notebook in the repo imports from these.

## Which model the outputs came from

This chapter was written and run against **`llama-3.3-70b-versatile`**. Groq has since retired that
model for newer accounts, so `DEFAULT_MODEL` is now **`openai/gpt-oss-120b`**. If you re-run a
notebook, the text will differ from the saved outputs; that's expected. Where a passage quotes a
number that depends on the model, it says which model it means.

One difference changes the lesson of notebook 5. llama-3.3 did not support Groq's strict
`json_schema` mode, and gpt-oss-120b does. The notebook still teaches the fallback, because many
models and providers offer nothing better.

## How it connects

**Used by every later chapter.** `02_agent_runtime` splits `asend_with_tools()` into the
Observe / Think / Act loop, reuses `execute_tool_call()` directly, and builds every structured
decision with `extract()`.

**Changed later, because later chapters needed it:**

| Change | Asked for by | Why |
|---|---|---|
| `max_retries=8` on both clients | `02_agent_runtime` | Concurrent branches burst through the per-minute token limit; the SDK now waits and retries a 429 instead of failing |
| `tool_call_failure()` | `02_agent_runtime` | A generic "try again" gives the model nothing to fix; Groq's error says exactly what was malformed |
| `Conversation.resources`, `close()`, `with Conversation() as c` | `04_memory` and `07_mcp` | Somewhere to keep live things, such as a memory store or an MCP session, so they can be reached and replaced. The runtime does not close them; whoever opened the conversation does |

## Running it

```bash
conda activate agentic-forge
# .env needs GROQ_API_KEY  (https://console.groq.com/keys)
```

Open the notebooks in order with the `agentic-forge` kernel.

- **Free tier:** about 100k tokens a day and a per-minute limit. The clients retry per-minute
  limits automatically; when the daily limit is reached, the only fix is to wait until the next day.
- **Embeddings** are not covered here. They arrive in `04_memory`, where there is something to
  retrieve.

## Going deeper

[`NOTES.md`](NOTES.md) has the full write-up: each notebook in detail, the model change, and why
`Conversation.resources` was added and what it is for.
