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

`llm.py` wraps the OpenAI SDK pointed at Groq's endpoint (`GROQ_API_KEY` from
`.env`) — this is what every notebook below imports from. Copy `.env.example`
to `.env` and add a key from [console.groq.com](https://console.groq.com/keys).

## Which model these notebooks were written against

**This chapter was written and executed against `llama-3.3-70b-versatile`**, and
the saved outputs below came from it. Groq has since retired that model for newer
accounts — `client.models.list()` no longer offers it, and a call returns
`404 model_not_found` — so `DEFAULT_MODEL` is now `openai/gpt-oss-120b`.

Re-running a notebook will therefore produce different text from the outputs
committed here. That is expected. Where a passage names a model or quotes a
number measured from one — a price, a tokenizer's exact split, a failure the
model has — it says which model it means, because those do not carry across.

One difference is worth knowing before you start, because it changes what
notebook 5 is teaching rather than just its numbers: **llama-3.3-70b did not
support Groq's strict `json_schema` mode, and `gpt-oss-120b` does.** Notebook 5
builds structured output the way you have to when the API will only promise you
*some* valid JSON. That technique is still worth having — plenty of models and
providers offer nothing better, and a schema constrains shape but never sense —
so the notebook keeps it, and `extract()` now asks for schema enforcement first
and falls back to it. See the notebook for what each layer actually catches.

The free tier allows 100k tokens/day and 12k tokens/minute. The per-minute cap
is handled for you: the clients pass `max_retries=8` so the SDK backs off and
retries a 429 rather than raising, which matters once `02_agent_runtime` starts
making concurrent calls. The daily cap is not something retrying can fix.

`tool_call_failure()` turns Groq's rejection of a malformed tool call into text
worth showing a model. Groq validates tool-call generations server-side and
returns HTTP 400 with the exact broken output it produced (`failed_generation`)
and, for type errors, the offending field by name. Handing that back beats a
generic "try again", which asks the model to fix something without saying what
was wrong — see `02_agent_runtime/think.py`.

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
