# Foundations

**What:** The fundamental building blocks every LLM/agent system in this repo
is built on — talking to an LLM, messages, tokens, prompting, structured
output, tool calling, streaming, async Python. Nothing here requires an
agent; this chapter only teaches the primitives later chapters compose.

**Why:** `02_agent_runtime` onward assumes you can already make an LLM call,
shape a prompt, parse a structured response, and call a tool. Keeping that
here means later chapters can focus purely on agent behavior.

**Deliberate exception:** this chapter has 9 notebooks instead of the
repo's usual 2–5, because it's a collection of independent primitives, not
one deep topic. Later chapters should go back to 2–5.

Each notebook ends with a **What you built** section and the next one opens
with **Prerequisites**, so the chapter reads as a curriculum, not a
reference.

## Providers

`llm.py` wraps the OpenAI SDK — this is what every notebook in the repo
imports from. It ships with three providers, all of which speak the OpenAI
wire format, so moving between them changes the base URL, the key, and the
model name, and nothing else:

| provider | free tier (measured, not published) | watch out for |
|---|---|---|
| `groq` (default) | 100k tokens/day, 12k tokens/minute | the daily cap is reachable in one run of `02_agent_runtime`'s forking notebook |
| `gemini` | ~20 requests/day **per model**, 5–15/minute, 1M context | published figures (1500/day) assume a verified or billing-enabled project; an ordinary key gets far less |
| `cerebras` | 1M tokens/day | only an 8K context window — fine here, tight for later chapters |

Those numbers are what the APIs actually returned when the limits were hit, not
what the docs advertise. Gemini's quota is counted per model, so
`llm.use("gemini", "gemini-2.5-flash-lite")` gets a separate daily allowance
from `gemini-2.5-flash`.

Copy `.env.example` to `.env` and fill in a key for whichever you want. Only
the active provider's key is required.

```python
import llm

llm.providers()          # every provider, and whether its key is actually set
llm.use("gemini")        # switch for the rest of the session
llm.current_model()      # what the active provider will be called with
llm.models()             # ask the provider what it will serve today
```

`use()` takes effect immediately, including in code imported long beforehand.
Two details make that work, and both are worth knowing about because they look
like over-engineering until you need them:

- `client` and `async_client` are thin proxies that forward to whichever
  provider is active. `from llm import client` binds a reference once, at
  import time, so without the indirection `use()` could only ever rebind
  `llm.client` — `chat.py` and `think.py` would go on talking to the old one.
- Functions take `model=None` and resolve the active model when the call is
  made, rather than defaulting to `model=DEFAULT_MODEL` and freezing it when
  the function was defined.

`DEFAULT_MODEL` still exists and `use()` keeps it current, but prefer
`llm.current_model()` — a `from llm import DEFAULT_MODEL` done before a switch
holds a stale string, which is exactly the trap the two points above avoid.

Model names in `PROVIDERS` go stale as providers retire them; `llm.models()`
is the one-line way to see what your key actually reaches.

### Rate limits

`llm.complete()` wraps `client.chat.completions.create` and is what everything
here calls. The SDK already retries 429s with backoff and honors a
`Retry-After` header — but Gemini never sends that header. It puts the wait in
the response body as `retryDelay`, where the SDK can't see it, so it backs off
blind, caps out around 8s, and gives up while the server is still asking for
49. `complete()` reads the number the server actually gave and waits it out.

It refuses to wait on a *daily* cap, which matters because those report a short
`retryDelay` too — Gemini will say "retry in 21s" about a quota that resets
tomorrow. Sleeping through that burns attempts and still fails. The answer to a
daily cap is `llm.use(...)` another provider.

### Thinking models

Gemini 3.x returns an opaque `thought_signature` alongside each function call
and rejects any replayed call that arrives without it — that's how it resumes
its own reasoning across a tool round trip. `02_agent_runtime`'s `Think` stores
it on the decision and hands it straight back, so those models work. They're
not the default only because `gemini-3.6-flash` allows 20 requests per day.

## Notebooks

1. `01_your_first_llm.ipynb` — a single chat completion call; temperature,
   top_p, max_tokens ✅
2. `02_conversations.ipynb` — system/user/assistant roles, proving statelessness,
   `Conversation` helper ✅
3. `03_tokens_and_context.ipynb` — tokenization, real vs. estimated token
   counts, context window, `trim_to_budget`, real cost & latency ✅
4. `04_prompting.ipynb` — instructions, few-shot, delimiters (incl. a real
   prompt-injection demo), `PromptTemplate` ✅
5. `05_structured_outputs.ipynb` — JSON mode's real guarantees, a real
   parsing failure, Pydantic validation, `extract()` retry loop ✅
6. `06_function_calling.ipynb` — tool schemas, arguments, two real failure
   categories (API-level rejection, which Groq already validates tool
   existence for; and argument validation, ours to catch via Pydantic
   types and custom `field_validator`s for domain rules), `execute_tool_call()`
   and a retrying `resolve_tools()` (basis for `03_tools`) ✅
7. `07_streaming.ipynb` — streaming tokens, chunk/event structure, a real
   latency comparison, streaming tool calls, `stream_and_print()` ✅
8. `08_asyncio.ipynb` — event loop, async/await, a real speedup from `gather`,
   why `TaskGroup` beats it on failure, `Semaphore`, cancellation (pure
   Python, no LLM calls) ✅
9. `09_build_an_agent.ipynb` — gives `Conversation` an `asend_with_tools()`,
   the same call/execute/feed-back loop as `resolve_tools()` (notebook 6) but
   writing into real conversation history instead of a throwaway list; proves
   a validation failure and its corrected retry both land in `convo.messages`,
   an `Agent` wrapper in `chat.py` (named Loopy), and `extract()` as a
   `/summary` feature ✅

(Embeddings intentionally live in `05_retrieval`, not here — they only
matter once there's something to retrieve.)
