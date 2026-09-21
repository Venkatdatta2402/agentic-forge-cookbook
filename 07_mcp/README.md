# MCP

**What:** The Model Context Protocol — what it actually is, how an agent consumes a server
someone else wrote, how the tools this repo already has get exposed as one, and what changes
about a tool call once it crosses a process boundary.

**Why:** `03_tools` built everything an agent needs to use a tool: `Tool` reads a function's
signature and produces both the JSON the model sees and the check on the arguments coming
back, `Registry` holds a lot of them, `invoke()` runs one with a time limit and sorts every
way it can fail by who is able to act on it. All of it is Python, in one process, reached by
`import`.

That is a hard ceiling, and it is not about scale. It means every tool an agent will ever have
must be written in your language, live in your repo, and be present at the moment the process
starts. An agent cannot use a tool that someone else maintains without that someone else
sending you code.

**The core idea, and the thing this chapter keeps returning to:** MCP does not change what a
tool is. It changes who owns the process it runs in.

That is not a slogan — it is measurable, and notebook 1 measures it. The schema an MCP server
puts on the wire for a tool and the schema `Tool.schema` builds for the same function are the
same JSON. The model cannot tell them apart, because there is nothing to tell apart. What is
different is that one of them is `import`ed and the other is *launched*, and every interesting
consequence in this chapter follows from that rather than from anything about the protocol's
design.

## The shape of it

```
Host                        ← decides which servers exist, and what the model may see
  │
  ├── Client ──────┐        ← one per server, owns one session
  │                │ MCP protocol (JSON-RPC 2.0)
  │                ▼
  │            Server       ← a separate process
  │              ├── Tools      (things the model can call)
  │              ├── Resources  (data the host can read)
  │              └── Prompts    (templates a user can pick)
  │
  └── Client ──→ another Server
```

Three roles, not two. The usual diagram omits the **host**, which is the role with no code in
the SDK and the one your agent is playing. It makes every decision the protocol deliberately
does not: which servers to connect to, which of their tools to put in front of the model, what
to do when one fails. When this reaches `02_agent_runtime`'s loop, the runtime is the host and
`Registry` is what decides which discovered tools the model is allowed to see.

This is also the answer to "is MCP a framework?" It is not. It specifies the wire between
client and server and says nothing about how an agent is built, which is why chapter 2's loop
needs no modification to use it.

## Notebooks

1. `01_mcp_concepts.ipynb` — the premise taken apart: chapter 3's `weather` schema and the same
   tool's MCP schema printed side by side and found identical, so the difference has to be
   somewhere else. It is two pids. Host/client/server; the session handshake and what gets
   negotiated in it; capabilities that are declared and empty (`resources` and `prompts` both
   `[]` on a server that declares both); discovery as a runtime fact rather than an import; the
   `content` list that can hold an image where `invoke()` ends in `str(result)`; and the error
   channels, where `03_tools`' six endings meet one bool — and `_meta` turns out to be the only
   surviving way to tell the host something the model must not see ✅
2. `02_mcp_client.ipynb` — **consuming a server someone else wrote, with nothing in
   `02_agent_runtime` or `03_tools` changed.** Three obstacles, each a different shape. The
   **sync/async split** is plumbing: the runtime is synchronous all the way down and every MCP
   call is a coroutine, `asyncio.run` per call is impossible rather than merely slow (a session
   belongs to the loop that opened it), and one background thread holding one loop is 5× faster
   than a session per call — and, unlike an async context, survives across notebook cells. The
   **argument check** is `args_model()` run backwards: no signature exists, the schema *is* the
   primary artifact, so `SchemaArgs` builds the check from JSON Schema with `jsonschema` while
   still raising Pydantic's `ValidationError` so `invoke()` reads it as `bad_arguments`. The
   **error taxonomy** is the only real loss, and the notebook translates rather than
   reimplements — MCP's answer becomes the exceptions `invoke()` already catches, so the repo
   has one error mechanism and not two. Along the way: `Registry` refusing a second `weather`
   from a second server, 12 discovered tools costing 749 tokens on every turn, `subset()` as
   the host's actual job, and `Act` running a tool in another process without knowing it ✅
3. `03_tools_resources_prompts.ipynb` — **the two primitives notebook 1 found empty, and
   whether they are worth having.** The distinction is one column wide: a tool is invoked by
   the *model*, a resource is pulled by the *host*, a prompt is picked by a *person* — three
   actors, and everything awkward follows from the third. Two traps first: resource
   *templates* do not appear in `resources/list` at all, so a server offering only templates
   answers `[]`; and a parameterised resource is so nearly a tool that the distinction stops
   meaning anything. Then the column that matters — `weather://history` is **12,780 tokens**,
   and `resources/read` has no size negotiation, so the safeguard `03_tools` built for a 7,601
   token tool result simply does not apply on this path. Wrapping the read *as a tool* restores
   it — 168 tokens in, 12,780 kept on `Result.value` — and the notebook immediately prints what
   the model actually received, thirty rows cut off mid-date, to make the point that truncation
   is a safety limit rather than an access strategy. That is `04_memory`'s
   injection-versus-tool fork again, one layer out, with the same answer. A third
   error channel turns up too — a bad URI is a protocol error, not `is_error`, so it lands in
   `broken` and is hidden from the model unless caught. Prompts get a straight recommendation
   against: they are a **UI affordance**, and this repo has no UI ✅
4. `04_mcp_server.ipynb` — **the other direction, and the payoff for the whole chapter.**
   `Registry` exposed over MCP by `expose()`, with `Tool.schema` published as-is so the wire
   description cannot drift from the function — which also fixes both defects notebook 1
   measured, since this time we own the server. Calls route through `invoke()`, so a real
   `Result` is produced on the far side and serialised into `_meta`, and **notebook 2's client
   recovers all five kinds with no modification** — it already read `forge/*`; it had simply
   never met a server that sent them. The sharpest finding corrects notebook 2: `unverified`
   was called structurally impossible, and it is not — the reasoning assumed the *client* had
   to verify. `verified()` runs on the server, which is the **only** place it can, because
   that is where the world is. Also: one validation rather than two (advertised schema strict,
   internal gate deliberately permissive, so the check happens where it can produce a `kind`);
   `repeatable` crossing via MCP's own `idempotentHint` rather than another `forge/` key,
   because annotations arrive at discovery when the client actually needs them; and **a
   `Registry` is not an export list** — which tools may cross a boundary is a different
   question from which tools an agent may use ✅
5. `05_mcp_transport.ipynb` — **the process boundary and the trust boundary are not the same
   line.** A stdio server is not sandboxed by being a separate process: we launch it, so it
   starts holding whatever we were holding, and the notebook reads a planted API key back out
   of it. The sharpest finding is whose fault that is — **the SDK ships an allowlist**
   (`DEFAULT_INHERITED_ENV_VARS`, twelve variables on Windows, no secrets) and builds the
   child's environment as `get_default_environment() | server.env`. Writing the obvious
   `env={**os.environ}` overrides that careful default with everything. The protocol did not
   hand the server our keys; the client did, in a line that looks like housekeeping.
   `inherit=False` takes it back — everything down to 18 variables. Then: who owns the
   process, and closing stdin as the only way to end it; **two clocks** — the server's
   `expose(timeout=)` produces a real `timeout` kind, the client's produces a bare exception,
   and neither cancels anything, so the tool runs on into a void and the server's limit should
   be the tighter one; the same server over streamable HTTP, where the lesson is that *HTTP is
   not safer, being started by someone else is*; the Windows/Jupyter `fileno` failure taken
   apart; and `PYTHONIOENCODING` tested to destruction — unicode round-trips byte-identical
   through a child forced to **ascii** ✅
6. `06_build_an_mcp_agent.ipynb` — **the test of the promise, and the only notebook here that
   calls a real model.** `03_tools` notebook 3's runtime, copied unchanged, with tools that
   live in another process; `subset()` as the host role in two lines. The live run is there
   because one question cannot be settled offline: *does a sentence a server wrote survive all
   the way to a person?* It does — `No weather station for Vestby. Try a larger nearby city.`
   crosses a protocol, two `invoke()` calls and a summariser intact. It also produced a result
   worth recording honestly: the model **relayed** that advice rather than retrying a larger
   city, which is defensible and looks from one turn away exactly like the stalling failure
   this chapter guards against — an argument for `kind` being inspectable rather than inferred
   from behaviour. Then the gap `04_memory` notebook 5 also found: **nothing on `Conversation`
   holds a handle**, so a live session exists only in a closure (`co_freevars` says
   `('mcp_tool', 'session')`). Ends on a server killed mid-run, which lands as `broken` and is
   correctly withheld from the model ✅

## What the whole chapter cost the earlier ones

The promise every notebook here made was that `02_agent_runtime` and `03_tools` would not have
to change to accommodate MCP. Measured at the end, with `ast` rather than by assertion —
`act.py`, `runtime.py`, `think.py`, `tools.py`, `registry.py` and `invoke.py` all import `mcp`
zero times:

| | |
|---|---|
| `02_agent_runtime` | **no changes** |
| `03_tools` | **one**: `invoker()` now honours `for_the_model` |

That single change was not an MCP requirement. `for_the_model` was computed and read by nobody,
which was survivable while `broken` meant a bug in a local function and stopped being
survivable once it meant "a server we do not control declined to explain itself".

## What this chapter does with chapters 1–3

| From earlier chapters | What `07_mcp` does with it |
|---|---|
| `Tool.schema` building the model's JSON from a signature (3) | Held up against the same tool's MCP schema in notebook 1. Same `type`, same `properties`, same `required` — the model cannot distinguish them |
| `args_model()` setting `extra="forbid"` (3) | Absent from MCP's schema, and notebook 2 shows what that costs: `{"city": "Oslo", "citty": "Bergen"}` validates, the typo is dropped, and the model gets a confident answer about one city having asked about two. `SchemaArgs(strict=True)` puts it back |
| `Registry` refusing a duplicate name (3) | Written against a typo; here it is the only thing between the agent and a second server's `weather` silently replacing the first |
| `Registry.subset()` (3) | Becomes the host's real job. Connecting to a server and putting its tools in front of the model are two decisions, and this is where they come apart |
| `Tool.schema` stripping Pydantic's `title` keys to save tokens (3) | The MCP server keeps them — `"title": "City"` on a field called `city`. Notebook 1 called this unfixable on a foreign server and was **wrong**: `Tool.schema` strips whatever it is handed, so wrapping a discovered tool fixes it. What is ours is the description we *send*; what is not is how the server behaves on receipt |
| `Registry` refusing a name already taken; `prefix=`, `merge()` (3) | Stops being bookkeeping. The names now arrive from someone else's process, and `prefix=` is what keeps two servers' `search` tools apart |
| `invoke()`'s six endings — `ok`, `bad_arguments`, `refused`, `unverified`, `timeout`, `broken` (3) | Meet `is_error: bool`. The two questions the loop asks — should the model see this, is it safe to retry — have nothing to read |
| `Result.for_the_model` (3) | Exists on the far side, but as a decision the *server author* makes, by choosing to put something in `_meta` rather than `content`. Not something a client can rely on |
| `verified()` — the call returned, but did it do the job (3) | Cannot be done *by a client* — but notebook 4 runs it on the server, where the world is, and the `unverified` kind crosses in `_meta`. The boundary does not weaken verification; it decides which side has to do it |
| `Tool.repeatable` (3) | Crosses as MCP's own `idempotentHint`, not as a `forge/` key — annotations arrive at discovery, which is when a client builds its `Tool` and has to decide |
| `invoke()` ending every call with `str(result)` (3) | MCP's `content` is a typed list — `TextContent`, `ImageContent`, `AudioContent`, `ResourceLink`, `EmbeddedResource`. A tool can return an image and `str()` has no way to say so |
| `invoke(max_tokens=)` trimming a 7,601-token result (3) | Does not apply to `resources/read`, which has no size negotiation at all. Notebook 3's 12,780-token resource is the case, and wrapping the read as a tool is the fix |
| Recall by injection vs recall as a tool (4, nb 5) | The same fork, one layer out: a resource pasted in by the host every turn, or read by the model when it thinks to. Same answer — injection for what must be dependable, a tool for what must be cheap |
| Retrieval over a bulk-loaded corpus (4) | What a resource actually is. A server exposing a handbook or a schema is a corpus the agent was never present for, so the useful shape is `recall`/`open_memory`, not `read(uri)` |
| `Runtime(max_seconds=)` and `invoke(timeout=)` (2, 3) | A third owner appears. The client has `read_timeout_seconds`, the server has its own ideas, and neither knows about the other |
| Any object with `.run(conversation)` is a component (2) | Still true, and still the hook. Nothing in `02_agent_runtime` needs to change for notebook 6 |

## Modules

Built as the notebooks need them, same as every other chapter. Two so far.

- `client.py` — `stdio_params()`: how the server process is launched (which interpreter, which
  directory, what it inherits). `connect()`: one live session, as an async context manager,
  and the decision about where the server's stderr goes. Two findings are written up in it as
  module constants rather than buried in commit history — `JUPYTER` and `ENCODING`, both below
- `demo_tools.py` — the ordinary functions the demo server exposes, as plain Python that never
  imports mcp. `weather` lives here because notebook 1 describes **the same function object**
  two ways; written out once per side, the comparison would only prove someone had kept two
  copies in step, which is `03_tools` notebook 1's opening failure four chapters later
- `transport_server.py` — an `expose()`d registry whose tools are instruments rather than
  tools: they report the process the server is running in, what it inherited, and how long it
  can be made to take. Runs over stdio or streamable HTTP depending on one launch argument,
  which is notebook 5's point about how little transport touches your code
- `session.py` — `Session`: one MCP session, opened on a background thread that owns one event
  loop, callable from synchronous code. Takes a script to launch or a `url=` to connect to —
  both reach the same `Client`, because a transport in this SDK is just an async context
  manager yielding two streams. `timeout` and `startup` are separate clocks on purpose:
  sharing one number means a one-second call limit also gives the interpreter one second to
  boot, and the resulting error talks about a closed connection rather than the limit that
  caused it. Exists because `02_agent_runtime` is synchronous and
  every MCP call is a coroutine, and because the two obvious fixes are both wrong — making the
  runtime async is a later chapter rewriting an earlier one, and `asyncio.run` per call cannot
  reuse a session at all
- `remote.py` — `remote_tool()` / `remote_tools()`: a discovered MCP tool as a `03_tools`
  `Tool`. `resource_reader()` does the same for resources, handing the model one tool that
  reads them by URI — which is how a resource gets back under `invoke()`'s size limit, and
  where a bad URI gets reclassified from `broken` to `refused` so the model can correct it. `SchemaArgs` is `args_model()` run backwards, building the argument check from JSON
  Schema instead of from a signature, and putting back the `additionalProperties: false` the
  server omitted. The result translation deliberately produces **no** `Result` — it raises what
  `invoke()` already catches, so there is one error mechanism in this repo rather than two.
  (The original sketch called this `schemas.py`; it turned out to be twenty lines and one idea
  with the tool adapter, so it did not earn a file of its own)
- `expose.py` — `expose(registry)`: a `03_tools` `Registry` as an MCP server. Publishes
  `Tool.schema` verbatim so the wire description cannot drift from the function; routes every
  call through `invoke()` and serialises the `Result` into `_meta`; splits the message on
  `for_the_model`, so a bug's cause travels where no model reads it. The one counterintuitive
  choice is documented in `_permissive_signature`: the advertised schema is strict and the
  handler's own signature is wide open, so validation happens once, in the layer that can
  return a `kind` instead of a string
- `registry_server.py` — an `expose()`d registry, for notebook 4. Three ordinary chapter-3
  tools: one that works and can refuse, one with a `KeyError` in it, and one wrapped in
  `verified()` that claims to save a note and truncates it. None of them know they are
  reachable over a protocol
- `data_server.py` — a second server, offering resources and prompts, for notebook 3.
  It exists so `server.py` can stay tools-only: notebook 1's finding is a server that
  *declares* both capabilities and implements neither, and bolting data onto it would have
  quietly deleted that. Two services written by two people is also the ordinary arrangement
- `server.py` — the chapter's demo server, run by the notebooks and imported by none of them.
  Registers `demo_tools` over MCP without restating any descriptions — `MCPServer` reads
  docstrings exactly as `Tool` does. Adds `described` and `detailed`, which are here rather
  than in `demo_tools` because they are not about weather: they have to speak MCP's own result
  type to show what a failure looks like from the far side of a boundary

## Findings worth keeping

**There is no protocol mechanism for an error's cause.** Three tools failing three ways all
return `is_error=True`:

| | what the model receives |
|---|---|
| `raises` — lets a `ValueError` escape | `Error executing tool raises`. The cause went to the server's stderr, in a process the client does not read |
| `described` — catches its own failure | `No weather station for Vestby. Try a larger nearby city.` |
| `detailed` — catches it and splits the audience | `Weather service unavailable.` in `content`, which the model sees; `ConnectionResetError at pool.py:412, retry 3/3 exhausted` in `_meta`, which only the host does |

The convention is real — catch your own errors, put something the model can act on in
`content`, put the rest in `_meta` — but it is a convention, and what you get is a property of
the server author's discipline rather than of MCP.

`_meta` rather than the obvious channel, and this is worth knowing before writing any server:
MCP *does* have a logging channel, `notifications/message`, and it is the textbook answer here.
It is **deprecated** as of protocol version `2026-07-28` (SEP-2577) — the version the SDK in
this repo negotiates — along with the `roots` capability. Every `ctx.error()` raises
`MCPDeprecationWarning`. `_meta` is what remains, it is passed through untouched, and it is
where notebook 2 puts `Result.kind` anyway, so both problems get one answer. Namespace the
keys: the SDK puts its own `io.modelcontextprotocol/serverInfo` in that same dict. Which is exactly why a client cannot rely on it, and why notebook 2's
mapping onto `Result` has to guess. The guess is not symmetric: read `is_error` as `broken` and
the model never sees *"try a larger nearby city"*, so the agent stalls holding advice it was
never shown; read it as `refused` and a genuine bug costs a few wasted retries. The cheap
mistake wins.

**A capability is not an inventory — and neither is one inventory.** The demo server declares
`tools`, `resources` and `prompts`, and has zero resources and zero prompts: a capability says
"I implement `resources/list`", and the honest answer to that call is `[]`. Notebook 3 makes it
worse. `resources/list` and `resources/templates/list` are **different calls**, and the first
omits the second entirely, so a server offering nothing but parameterised resources answers
`[]` to the obvious question. A host that concludes "no data here" is wrong, silently.

**A resource is a context-window decision in the costume of a data-access one.**
`weather://history` is 12,780 tokens, and `resources/read` has no size negotiation, no
`max_tokens`, no warning — the host asked, so the host gets all of it. `03_tools` built
`invoke(max_tokens=)` for exactly this after a 7,601-token tool result, and it does not apply
on this path.

Wrapping the read as a tool restores the limit — 168 tokens in, all 12,780 still on
`Result.value` — and the notebook then prints what the model actually got, because the number
flatters the result: thirty rows of CSV cut off mid-date. **Truncation is a safety limit, not
an access strategy.** It stops one resource eating the window; it does not make a large
resource usable. The wrapper genuinely solves the small-resource case and only converts an
overflow into a quieter failure for the large one. Which leaves the same choice `04_memory`
reached — injection for what must be dependable, a tool for what must be cheap — and, for
anything corpus-sized, retrieval rather than either.

**A stdio server starts with your authority, not merely in your process tree.** It is a
separate process, which sounds like containment and is not — it can read your credentials,
open your files and reach the network as you. Notebook 5 reads a planted key back out of one.
The separateness buys isolation of *failures*, not of *authority*, which is a much smaller
claim than "it runs in its own process" suggests.

And the leak is the client's doing. `stdio_client` inherits an allowlist by default and merges
`StdioServerParameters.env` on top, so `env={**os.environ}` — the obvious line, and a common
one in real configs — overrides a careful default with everything. `stdio_params(inherit=False)`
is the switch, and `08_guardrails` inherits the argument.

**Nothing cancels a call in flight.** A client-side timeout stops the client waiting; the tool
keeps running to completion and answers into a void. There is no "stop that" message in MCP.
So a timeout is a statement about the caller's patience rather than about the work, which is
why `03_tools` made `repeatable` a property of the tool — and why, of the two clocks, the
server's should be the tighter one: it is the only one attached to the thing doing the work.

**An extension must be safe to ignore.** `forge/*` is ours, and a client that has never heard
of it must still get a correct answer — so nothing in `_meta` is load-bearing and nothing a
stranger reads is wrong. A server putting the real message in `_meta` and a placeholder in
`content` would be broken for everyone but itself. The rule that falls out: **say it in the
protocol where the protocol has a word for it** (`idempotentHint`), and in `_meta` only where
it does not (`forge/kind`).

**Prompts want a person.** `prompts/get` returns messages for someone to start a conversation
from, chosen off a menu. An autonomous agent has no one at the keyboard, and handing the menu
to the *model* just makes them tools with no result to act on. Skipped deliberately rather than
by omission.

**Only one of the six endings is genuinely lost, and only against a silent server.** The bool
makes it look like five. Measured in notebook 2: `ok` comes from `is_error`; `bad_arguments`
never crosses, because the schema is checked locally first; `timeout` belongs to the client's
own clock; a dead server raises on our side rather than answering. What cannot be recovered
from a server that declines to explain itself is `refused` versus `broken` *inside a live
tool*.

Notebook 2 also listed `unverified` as structurally impossible, and **notebook 4 shows that
was wrong.** The reasoning — `verified()` cannot inspect another process's world — was right,
but it assumed the client had to be the one verifying. Run `verified()` on the *server*, where
the world actually is, and the kind rides back in `_meta` like any other. Which is the
stronger claim: verification does not merely survive the boundary, the server is the only
place it can happen at all, so a tool that crosses a boundary has to carry its own.

**`for_the_model` was computed and nothing read it — fixed in `03_tools`.** `execute_tool_call`
ends with `str(result)`, and nothing on that path consulted the flag, so a `broken` result's
text entered the conversation regardless. The model was being handed things like
`RuntimeError: ConnectionResetError at pool.py:412` and invited to work around them.

The gap dates from `03_tools` and was survivable while `broken` meant a bug in a local
function. It stops being survivable here, because `broken` is the honest reading of any failure
a foreign server declines to explain. `invoker()` now honours the flag: the conversation gets
`"<tool> could not be completed."` and the cause moves to `Result.value`. Deliberately not done
in `Result.__str__` — printing a broken result and reading what it says is how `03_tools`
notebook 3 makes the distinction visible, and redacting it there would delete the lesson.
`02_agent_runtime` is untouched.

**A session cannot span notebook cells — unless it is on a thread.** It survives across them well enough to answer calls,
but closing it from a different cell raises `RuntimeError: Attempted to exit cancel scope in a
different task than it was entered in` — anyio binds cancel scopes to the task that opened
them, and ipykernel runs every cell in a fresh task. So every cell in notebook 1 that talks to
the server opens its own session, and results that outlive the block are kept as plain data.

Notebook 2's `Session` is not subject to this, and the reason is worth keeping: its loop lives
in a background thread rather than in the cell's task, so the task that opened the session is
still around to close it whenever that happens. The bridge built for the synchronous runtime
turns out to fix the notebook problem too.

## What this chapter needs that earlier ones did not

**`mcp` 2.2.0**, added to `environment.yml` with the eleven transitive dependencies it brings.
Worth knowing before reading any MCP example found elsewhere: 2.x is a breaking rewrite.
`FastMCP` is now `MCPServer`, and the Python objects are snake_case — `tool.input_schema`,
`result.is_error` — while the JSON on the wire stays camelCase. Essentially every tutorial
online is 1.x and will fail on import or on attribute access. `pip install 'mcp<2'` is the
escape hatch if a 1.x example has to run as-is.

The protocol itself has moved too, and in a direction that invalidates advice rather than just
code: at version `2026-07-28`, SEP-2577 deprecates both the `logging` and `roots` capabilities.
Anything written about MCP before that describes logging as the way a server talks to a host,
and it still runs — with a `MCPDeprecationWarning` per call. Nothing in this chapter uses it.

**A stdio server cannot be launched from a Jupyter kernel on Windows with the defaults**, and
the traceback does not explain itself — it ends in `io.UnsupportedOperation: fileno` from
inside `subprocess.Popen`. The chain: ipykernel runs a `SelectorEventLoop` on Windows because
pyzmq needs one; selector loops have no async subprocess support, so mcp falls back to a plain
`Popen`; `Popen` needs a real OS handle for the child's stderr; and in a kernel `sys.stderr` is
an ipykernel `OutStream` that forwards over a socket and has no file descriptor to give.

The fix is not about async at all — give the child a real file. `connect()` checks whether
`sys.stderr` can be inherited and falls back to `07_mcp/server.log`, which is worth having
regardless: a stdio server's stderr is the only place its crashes go, and `server.log` is the
first thing to open when anything in this chapter behaves strangely. It is generated, and
gitignored.

**No API keys, and no model.** Notebook 1 talks to a local subprocess and calls no LLM at all,
which makes its saved outputs worth reading — every pid, schema and error string in it is a
real one, reproducible without spending a token.
