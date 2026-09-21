# Tools

**What:** Everything between "a model asked to call something" and "a result
came back" — how a tool is defined, how it's described to the model, how many
of them an agent can be handed at once, what happens at the moment of the call,
and how you know the call actually did what it said.

**Why:** `01_foundations` made a tool call work and `02_agent_runtime` ran one
inside a loop, both with two tools written by hand. Neither had to deal with
what happens once tools are something you have a lot of.

Written by hand, one tool is the same fact stated three times: the function,
the JSON the model reads, and the check on the arguments that come back.
Nothing keeps those three in step. Rename a parameter and the check still
passes — right up until the call crashes — because the check is one of the
three that went stale, not something watching the other two.

**The core idea:** write it once, work the rest out from it. The function
signature is the one copy that cannot go stale, so the JSON and the check are
both read off it.

## Notebooks

1. `01_what_is_a_tool.ipynb` — a renamed parameter breaking a working tool,
   then `args_model()` and `Tool`/`@tool`: the argument check built from a
   signature, the model's JSON built from that, `Annotated` for parameter
   descriptions, `extra="forbid"`, and a way out for rules a signature can't
   express ✅
2. `02_the_registry.ipynb` — the three dicts notebook 1 builds by hand become
   one object; a plain dict quietly keeping one of two tools called `search`;
   prefixes; what the API really accepts as a tool name; what thirty tools
   cost in tokens on every turn; subsets by name and by tag ✅
3. `03_invocation.ipynb` — what happens at the moment of the call: a tool that
   hangs and why chapter 2's own time budget can't stop it, failures sorted by
   who can act on them, a 7,601-token result cut down to what a conversation
   can carry, and a timeout on a tool that must not be repeated ✅
4. `04_verification.ipynb` — checking the arguments going in is not the same
   as checking the world afterwards. A tool that saves 60 characters of a
   110-character note and reports success; why a check that reads the tool's
   own answer catches nothing; `verified()`, and `unverified` as its own
   ending; what cannot be verified at all ✅
5. `05_built_in_tools.ipynb` — the toolbox against things that don't
   cooperate: a live API (real 404s, a real dead host, a real 7,254-token
   response), file tools confined to one folder, and a calculator that isn't
   a way to run arbitrary Python. Catches `Observe` doing the agent's
   arithmetic inside its own phrasing call ✅

## What this chapter does with chapters 1 and 2

| From earlier chapters | What `03_tools` does with it |
|---|---|
| Hand-typed `tools=[...]` JSON (1, nb 6) | Built by `Tool.schema` from the signature, so it can't disagree with the function |
| Pydantic argument checking (1, nb 6) | Same check, same Pydantic — but the model doing the checking is generated too, and `Tool.__call__` is the only route to the function |
| `field_validator` for rules like "not zero" (1, nb 6) | Kept, as `Tool(..., args=YourModel)`. A signature can't say "not zero", so reading the signature is the default rather than the only option |
| `execute_tool_call()` (1, nb 6) | Still what `Act` runs. `Tool.args` drops into its `schemas` dict unchanged |
| A tool call that returns = success (1, nb 6) | Not the same claim. A tool can return cheerfully and not have done the job, so `verified()` checks the world afterwards |
| `str(result)` ending every call (1, nb 6) | Still the last step, but `invoke` decides how much of it the conversation can carry, and keeps the whole thing on `Result.value` |
| `except Exception: "Error: {e}"` (1, nb 6) | Split by who can act on it. Wrong arguments and a tool that says no go to the model; a bug in the tool does not |
| `Runtime(max_seconds=...)` (2) | Checked between components, so it cannot stop a blocking tool. `invoke(timeout=)` is the per-call limit that can |
| `Observe`'s LLM summarizing (2) | Caught twice: dropping the half of a refusal that said what to do next, and converting a value the agent had a tool for — see below |
| `Act(functions=..., schemas=...)` (2) | Fed from `Tool` objects with a comprehension, or from `Registry.invokers()`. No change to `02_agent_runtime` needed for any of the wiring |
| `Think(allow_fork=False)` (2) | The plain ReAct loop, used to run every demo here — this chapter is about tools, not about patterns |
| `DebugThink` repairing arguments (2) | Where a stale description hurts most: the model is asked to fix arguments that were right for the description it was given, and wastes every retry |

**Two changes were made *to* chapter 2, forced by this chapter, both in the same
prompt** — the one `Observe` uses to turn a raw tool result into an observation.

Notebook 3 gives tools a way to fail usefully: `ToolError("No weather station for Vestby. Try a
larger nearby city.")` is written for the model to act on. But `Observe` turns a
raw tool result into an observation with an LLM, and its prompt asked for "one
plain factual sentence" — so the suggestion was summarized away as commentary
before the model ever saw it. The prompt in `02_agent_runtime/observe.py` now
says to keep whatever the tool itself said about what to do next.

Notebook 5 then found another clause in it causing the opposite problem. Asked to
state a result "using the conversation only to phrase it correctly (e.g. which
units)", `Observe` converted 20 °C to 68 °F itself — so the `calculate` tool was
never called and the arithmetic appears in no tool result anywhere. The answer
was right, which is the problem. That clause is now replaced by an instruction
to copy every value exactly as the tool gave it, and to convert or calculate
nothing, even when the conversation asks for different units.

Nothing else in `02_agent_runtime` changes. All the wiring in this chapter goes
through `Act` and `Think` exactly as they already are.

**One change was later made *to* this chapter, forced by `07_mcp`.** `Result.for_the_model`
was computed here and read by nobody: `invoker()` handed the whole `Result` to
`execute_tool_call`, which finishes with `str(result)`, so a `broken` result — the one kind
that is supposed to stay out of the conversation — went into it anyway. That was survivable
while `broken` meant a bug in a function in this repo, which is rare and gets fixed the same
afternoon. It stopped being survivable in `07_mcp`, where `broken` is the honest reading of
any failure a server someone else wrote declines to explain.

`invoker()` now checks the flag: when `for_the_model` is False the conversation gets
`"<tool> could not be completed."` and the cause moves to `Result.value`, where whoever is
reading the run can still find it. It is **not** done in `Result.__str__`, because notebook 3
prints a broken result next to its flag precisely to show what that message says — redacting
it there would remove the demonstration. Nothing in this chapter's notebooks changes, because
none of them route a `broken` result through `invoker()`.

## Modules

Built incrementally as the notebooks need them, same as every other chapter.

- `tools.py` — `args_model()` (signature → the Pydantic model that checks
  arguments), `Tool` (the function, plus the model's JSON and the argument
  check both read off it), `@tool`
- `registry.py` — `Registry`: the tools an agent can use, and the three views
  of them chapter 2's loop needs. Refuses a name already taken or one the API
  won't accept; `prefix=` and `merge()` for combining sources; `subset()` by
  name or tag; `invokers()` to hand `Act` tools that go through `invoke`
- `invoke.py` — `invoke()`: one tool call, with a time limit, a size limit,
  and every ending returned as a `Result` rather than raised — `ok`,
  `bad_arguments`, `refused`, `unverified`, `timeout`, `broken`. `ToolError`
  for a tool that ran and said no; `truncate()`; `Result.for_the_model` and
  `Result.safe_to_retry`. `invoker()` is the adapter into chapter 2's
  `Act(functions=)`, and the one place `for_the_model` is enforced rather than
  merely computed — see the note above
- `verify.py` — `verified()`: a copy of a tool with a check wrapped around it,
  run after the call and looking at the world rather than at what the tool
  said. `VerificationError`
- `toolbox.py` — the standard tools: `http_get` (real network, real status
  codes), `file_tools(root)` and `within()` for reading and writing confined
  to one folder, `written_correctly(root)` as its check, and `calculate()`,
  which walks parsed nodes instead of calling `eval`. Deliberately not named
  `builtins.py`, which would shadow Python's own
