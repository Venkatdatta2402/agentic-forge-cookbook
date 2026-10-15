# 07 · MCP

> The Model Context Protocol: using tools that live in someone else's process, exposing this
> repo's tools to others, and what changes about a tool call once it crosses a process boundary.

Up to now, every tool an agent could use had to be written in Python, live in this repo, and be
imported when the process started. MCP removes that ceiling. An agent can use a tool someone else
maintains, in any language, without them sending you code.

**Contents:** [The idea](#the-idea-in-one-picture) · [What you'll learn](#what-youll-learn) ·
[Notebooks](#notebooks) · [Key ideas](#key-ideas) · [Modules](#modules) ·
[How it connects](#how-it-connects) · [Running it](#running-it)

---

## The idea in one picture

**MCP does not change what a tool is. It changes who owns the process it runs in.**

That isn't a slogan; notebook 1 measures it. The schema an MCP server sends for a tool and the
schema chapter 3's `Tool.schema` builds for the same function are **the same JSON**. The model
can't tell them apart. The only difference is that one is `import`ed and the other is
*launched*, and every interesting result in this chapter follows from that.

```
Host                        ← decides which servers exist, and what the model may see
  │
  ├── Client ──────┐        ← one per server, owns one session
  │                │ MCP protocol (JSON-RPC 2.0)
  │                ▼
  │            Server       ← a separate process
  │              ├── Tools      (things the model can call)
  │              ├── Resources  (data the host can read)
  │              └── Prompts    (templates a person can pick)
  │
  └── Client ──→ another Server
```

There are three roles, not two. The usual diagram leaves out the **host**, which is the role your
agent plays and the one with no code in the SDK. The host decides which servers to connect to,
which of their tools the model sees, and what to do when one fails. Here, chapter 2's runtime is
the host, and chapter 3's `Registry.subset()` decides which discovered tools the model sees.

MCP is not a framework. It specifies the wire between client and server and says nothing about how
an agent is built, which is why chapter 2's loop needs no changes to use it.

## What you'll learn

- What MCP actually standardises (and what it leaves to you), measured rather than assumed
- How to use a server someone else wrote with **no changes** to chapters 2 and 3
- Tools vs resources vs prompts: who each one is for
- How to expose this repo's own `Registry` as an MCP server
- Why a separate process is **not** a sandbox, and the one line that leaks your API keys
- How to keep an agent working when a server dies mid-run

## Notebooks

| # | Notebook | What it shows |
|---|---|---|
| 1 | [`01_mcp_concepts`](notebooks/01_mcp_concepts.ipynb) | Chapter 3's schema and the MCP schema side by side, identical. Host / client / server, the handshake, empty capabilities, discovery, the `content` list, and the error channels. **No model calls** |
| 2 | [`02_mcp_client`](notebooks/02_mcp_client.ipynb) | Using a foreign server with nothing in chapters 2–3 changed: a sync/async bridge, `SchemaArgs` (the argument check built from JSON Schema), and MCP's errors translated into the ones `invoke()` already handles |
| 3 | [`03_tools_resources_prompts`](notebooks/03_tools_resources_prompts.ipynb) | The two primitives notebook 1 found empty. A 12,780-token resource with no size limit, wrapped as a tool to get one back. Prompts skipped on purpose |
| 4 | [`04_mcp_server`](notebooks/04_mcp_server.ipynb) | This repo's `Registry` exposed with `expose()`; notebook 2's client recovers every result kind unchanged; verification runs on the server |
| 5 | [`05_mcp_transport`](notebooks/05_mcp_transport.ipynb) | The process boundary vs the trust boundary; two clocks; streamable HTTP; a Windows/Jupyter failure taken apart |
| 6 | [`06_build_an_mcp_agent`](notebooks/06_build_an_mcp_agent.ipynb) | A live agent using tools in another process; a server killed mid-run and reconnected |

## Key ideas

### MCP's errors are one bool; this repo has six result kinds
Chapter 3 sorts every call into `ok`, `bad_arguments`, `refused`, `unverified`, `timeout` or
`broken`. MCP has `is_error: bool`. Only one distinction is really lost against a server that
doesn't explain itself: `refused` vs `broken` inside a live tool. The fix is a convention, not a
protocol feature. Put something the model can act on in `content`, and put the rest in `_meta`,
which only the host reads.

| A tool that… | What the model receives |
|---|---|
| lets a `ValueError` escape | `Error executing tool raises` (the cause went to the server's stderr) |
| catches its own failure | `No weather station for Vestby. Try a larger nearby city.` |
| catches it and splits the audience | `Weather service unavailable.`, while the real cause (`ConnectionResetError at pool.py:412…`) goes in `_meta` for the host |

`notifications/message`, the textbook logging channel, is **deprecated** in protocol version
`2026-07-28` (SEP-2577), so `_meta` is what's left. This repo's own keys are namespaced `forge/*`
and are always safe to ignore.

### A resource is a context-window decision
`weather://history` is 12,780 tokens, and `resources/read` has no size limit. Wrapping the read
as a tool puts chapter 3's limit back (168 tokens in, all 12,780 kept on `Result.value`). Then the
notebook prints what the model actually received, thirty rows cut off mid-date. **Truncation is
a safety limit, not an access strategy.** For anything corpus-sized, use retrieval
(`04_memory`).

### Verification belongs on the server
Notebook 2 called `unverified` impossible over MCP, because a client can't inspect another
process's world. Notebook 4 showed the reasoning was right but the conclusion wrong. Run
`verified()` **on the server**, where the world is, and the result kind travels back in `_meta`.
A tool that crosses a boundary has to carry its own verification.

### Security: a separate process is not a sandbox
**The process boundary and the trust boundary are not the same line.** A stdio server is not
sandboxed by being a separate process: we launch it, so it starts holding whatever we were
holding, and the notebook reads a planted API key back out of it. Whose fault is that? Not the
protocol's: **the SDK ships an allowlist** (`DEFAULT_INHERITED_ENV_VARS`, twelve variables on
Windows, no secrets) and builds the child's environment from it. Writing the obvious
`env={**os.environ}` overrides that careful default with everything. `inherit=False` takes it
back, down to 18 variables.

So a stdio server starts with your authority: it can read your credentials, open your files and
reach the network as you. The separateness buys isolation of *failures*, not of *authority*. And
the leak is the client's doing. In this repo, `stdio_params(inherit=False)` is the switch.

### Nothing cancels a call in flight
A client-side timeout stops the client waiting, but the tool keeps running and answers into a
void. MCP has no "stop that" message. So of the two clocks, the **server's** should be the
tighter one, and a non-repeatable tool must not be retried after a timeout.

### A live connection must be replaceable
A tool that captures a session in a closure can't be repaired when the server dies. So tools hold
a **`Connection`** box instead, kept on `Conversation.resources`. Replacing the session is one
assignment, and every tool built earlier keeps working. `Reconnect` is an ordinary loop component
that decides *when* to repair:

```python
Runtime(loop=[Observe(...), Reconnect(), Think(...), Act(...), Observe()], repeat_from=1)
```

| The situation | What happens | What the model is told |
|---|---|---|
| server answering | one `tools/list` (~2.5 ms) | nothing |
| died, restartable | repaired and logged | **nothing**; it can't act on "we reconnected" |
| never coming back | gives up after `attempts` | "the mcp service is unreachable…" |

`ping` can't be used as a health check: servers built with this SDK answer `Method not found`.

## Modules

| File | What's in it |
|---|---|
| [`client.py`](client.py) | `stdio_params()` (how a server is launched and what it inherits), `connect()` |
| [`session.py`](session.py) | `Session`: one MCP session on a background thread with its own event loop, callable from synchronous code, over stdio or `url=` |
| [`remote.py`](remote.py) | `remote_tool()` / `remote_tools()` (a discovered tool as a chapter 3 `Tool`), `resource_reader()`, `SchemaArgs` (the argument check from JSON Schema, strict) |
| [`expose.py`](expose.py) | `expose(registry)`: a chapter 3 `Registry` as an MCP server, with `Result` carried in `_meta` |
| [`repair.py`](repair.py) | `Reconnect`: attempts, backoff, a repair budget, and which outcomes the model hears |
| [`demo_tools.py`](demo_tools.py) | Plain functions for the demo server (never imports `mcp`) |
| [`server.py`](server.py), [`data_server.py`](data_server.py), [`registry_server.py`](registry_server.py), [`transport_server.py`](transport_server.py) | The demo servers used by notebooks 1, 3, 4 and 5 |

## How it connects

The promise was that chapters 2 and 3 would not change to support MCP. Checked at the end with
`ast`, `act.py`, `runtime.py`, `think.py`, `tools.py`, `registry.py` and `invoke.py` import `mcp`
zero times.

| Chapter | Changes |
|---|---|
| `02_agent_runtime` | **none** |
| `03_tools` | **one**: `invoker()` now enforces `for_the_model`, so a broken result's cause stays out of the conversation |
| `01_foundations` | **one**: `Conversation` gained `resources` and `close()` |

Neither change was an MCP requirement; MCP just made both gaps impossible to ignore.

**What chapter 3 gives MCP for free:** `Registry` refuses a second server's `weather` silently
replacing the first; `prefix=` keeps two servers' `search` tools apart; and `SchemaArgs(strict=True)`
puts back `extra="forbid"`, without which `{"city": "Oslo", "citty": "Bergen"}` validates and the
typo is silently dropped.

## Running it

```bash
conda activate agentic-forge
# notebooks 1-5 need no API key; notebook 6 needs GROQ_API_KEY
```

- **`mcp` 2.2.0.** 2.x is a breaking rewrite: `FastMCP` is now `MCPServer`, and Python objects
  are snake_case (`tool.input_schema`, `result.is_error`). Nearly every tutorial online is 1.x.
- **Windows + Jupyter.** A stdio server can't be launched from a kernel with the defaults
  (`io.UnsupportedOperation: fileno`). `connect()` sends the server's stderr to
  `07_mcp/server.log` instead. That file is also the first place to look when a server misbehaves.
- Notebook 1 calls no model at all, so every pid, schema and error string in its outputs is real
  and reproducible for free.

## Going deeper

[`NOTES.md`](NOTES.md) has the full write-up, including every finding about errors, resources,
transport and reconnection.
