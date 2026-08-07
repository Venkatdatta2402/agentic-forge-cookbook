import json
import os
import re
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError, OpenAI, RateLimitError
from pydantic import ValidationError

load_dotenv()

# Every provider below speaks the OpenAI wire format, so moving between them changes exactly
# three things -- the base URL, which env var holds the key, and the default model name --
# and nothing downstream of this file needs to know which one is active.
#
# `free_tier` is a note to humans, not something the code enforces. These numbers change
# without notice; models() below asks the provider what it will actually serve today.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "free_tier": "100k tokens/day, 12k tokens/minute -- fast, but the daily cap is easy "
                     "to reach once 02_agent_runtime starts forking",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "model": "gpt-oss-120b",
        "free_tier": "1M tokens/day -- ten times Groq's budget, but only an 8K context window, "
                     "which 02_agent_runtime's longer conversations can exceed",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        # Not a Gemini 3.x model, purely for quota: gemini-3.6-flash allows 20 requests per DAY
        # on the free tier, which one agent loop can spend. (Its thinking-model behavior is
        # handled -- Think now stores and replays the thought_signature those models require --
        # so 3.x works fine on a paid key. See think.py's _build_messages.)
        # Per-minute caps differ too: 2.5-flash is 5 RPM, 2.5-flash-lite roughly 15. For the
        # forking notebook in 02_agent_runtime, prefer lite: LLM_MODEL=gemini-2.5-flash-lite
        "model": "gemini-2.5-flash",
        # Measured on an unverified free key, not taken from the published figures: roughly
        # 20 requests per DAY per model, and 5-15 per minute. Published numbers are far higher
        # (1500/day) and evidently assume a verified or billing-enabled project. Counted per
        # model, so switching model gets you a fresh daily allowance.
        "free_tier": "~20 requests/day per model and 5-15/minute on an unverified key, 1M "
                     "context. Generous per request, very tight per day -- fine for one "
                     "notebook, not for a chapter",
    },
}

# max_retries covers HTTP 429s, which the SDK retries with backoff honoring Retry-After.
# The default of 2 is fine for one call at a time, but 02_agent_runtime runs branches and
# tool calls concurrently, and a free-tier tokens-per-minute cap is easy to burst through.
MAX_RETRIES = 8

DEFAULT_MODEL = None  # set by use(), called at the bottom of this file

_active = {"provider": None, "model": None, "client": None, "async_client": None}
_clients = {}


def use(provider=None, model=None):
    """Point every client in this repo at `provider` for the rest of the session.

    Called with no arguments it reads LLM_PROVIDER and LLM_MODEL from .env, falling back to
    groq and that provider's default. Call it again at any time to switch --
    `llm.use("gemini")`, or `llm.use("gemini", "gemini-2.5-flash-lite")` -- including from
    inside a notebook.
    """
    provider = provider or os.getenv("LLM_PROVIDER") or "groq"
    model = model or os.getenv("LLM_MODEL")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")

    spec = PROVIDERS[provider]
    api_key = os.getenv(spec["key_env"])
    if not api_key:
        raise RuntimeError(
            f"provider {provider!r} needs {spec['key_env']} set in your .env file.\n"
            f"Free tier: {spec['free_tier']}"
        )

    if provider not in _clients:
        _clients[provider] = (
            OpenAI(api_key=api_key, base_url=spec["base_url"], max_retries=MAX_RETRIES),
            AsyncOpenAI(api_key=api_key, base_url=spec["base_url"], max_retries=MAX_RETRIES),
        )

    global DEFAULT_MODEL
    _active["provider"] = provider
    _active["model"] = model or spec["model"]
    _active["client"], _active["async_client"] = _clients[provider]
    DEFAULT_MODEL = _active["model"]
    return provider


def current_provider():
    return _active["provider"]


def current_model():
    return _active["model"]


def providers():
    """Every known provider, and whether its key is actually present in this environment."""
    return {
        name: {
            "model": spec["model"],
            "key_env": spec["key_env"],
            "key_set": bool(os.getenv(spec["key_env"])),
            "free_tier": spec["free_tier"],
            "active": name == _active["provider"],
        }
        for name, spec in PROVIDERS.items()
    }


def models():
    """Ask the active provider what it will serve right now.

    The defaults in PROVIDERS go stale -- providers retire model names on their own schedule.
    This is the one-line way to find out what your key can actually reach today.
    """
    return sorted(m.id for m in _active["client"].models.list())


class _ActiveClient:
    """Forwards every attribute to whichever provider's client is currently active.

    This indirection is what makes use() work after the fact. `from llm import client` binds a
    reference once, at import time -- so rebinding llm.client would do nothing for chat.py and
    think.py, which grabbed it at startup and would go on talking to the old provider. Holding
    a proxy instead means there is only ever one place the answer lives.
    """

    def __init__(self, kind):
        self._kind = kind

    def __getattr__(self, name):
        return getattr(_active[self._kind], name)

    def __repr__(self):
        return f"<{self._kind} -> provider {_active['provider']!r}, model {_active['model']!r}>"


client = _ActiveClient("client")
async_client = _ActiveClient("async_client")

use()  # honors LLM_PROVIDER from .env; defaults to groq


# How long a per-minute cap is ever worth waiting out. Anything longer is a daily quota
# wearing a retry delay as a disguise, and sleeping through it helps nobody.
MAX_RATE_LIMIT_WAIT = 90

_RETRY_DELAY = re.compile(r'retryDelay["\']?[:\s]+["\']?(\d+(?:\.\d+)?)s')

# A daily cap reports a short retryDelay too -- Gemini will happily say "retry in 21s" about a
# quota that does not reset until tomorrow. Waiting that out just burns six attempts and still
# fails, so daily limits are detected by name and re-raised immediately.
_PER_DAY = re.compile(r"PerDay|per day|\bTPD\b|RequestsPerDay", re.IGNORECASE)


def _requested_delay(error):
    """Seconds to wait, if the server asked for a wait that waiting can actually satisfy."""
    text = str(error)
    if _PER_DAY.search(text):
        return None
    match = _RETRY_DELAY.search(text)
    return float(match.group(1)) if match else None


def complete(**kwargs):
    """`client.chat.completions.create`, but honoring a rate-limit delay stated in the body.

    The SDK already retries 429s with exponential backoff, and honors a Retry-After header when
    there is one. Gemini never sends that header -- it puts the wait in the JSON body as
    `retryDelay` -- so the SDK backs off blind, caps out around 8s per attempt, and gives up
    while the server is still asking for 49. Reading the number it actually gave us turns a
    hard failure into a pause, which is the difference between a free tier being usable here
    and not.

    A delay longer than MAX_RATE_LIMIT_WAIT is re-raised rather than slept through: that is a
    daily cap, and the answer to a daily cap is llm.use(...) another provider, not waiting.
    """
    attempts = 6
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as error:
            delay = _requested_delay(error)
            if delay is None or delay > MAX_RATE_LIMIT_WAIT or attempt == attempts - 1:
                raise
            time.sleep(delay + 1)  # +1 so we land just past the window, not on its edge
    raise RuntimeError("unreachable")


# Every function below takes model=None rather than model=DEFAULT_MODEL, so the active model is
# looked up when the call is made instead of frozen when the function was defined. That is what
# lets use() take effect in code that was imported long before it was called.

def chat(messages, model=None, **kwargs):
    response = complete(
        model=model or current_model(),
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content


def stream_chat(messages, model=None, **kwargs):
    stream = complete(
        model=model or current_model(),
        messages=messages,
        stream=True,
        **kwargs,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def async_stream_chat(messages, model=None, **kwargs):
    stream = await async_client.chat.completions.create(
        model=model or current_model(),
        messages=messages,
        stream=True,
        **kwargs,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_and_print(messages, model=None, **kwargs):
    full = []
    for delta in stream_chat(messages, model=model, **kwargs):
        print(delta, end="", flush=True)
        full.append(delta)
    print()
    return "".join(full)


def extract(prompt, schema, retries=3, model=None):
    messages = [{"role": "user", "content": prompt}]
    for _ in range(retries):
        reply = chat(messages, model=model, response_format={"type": "json_object"})
        messages.append({"role": "assistant", "content": reply})
        try:
            return schema.model_validate_json(reply)
        except (json.JSONDecodeError, ValidationError) as e:
            messages.append({
                "role": "user",
                "content": f"That was invalid: {e}\nReturn corrected JSON only, matching the schema.",
            })
    raise ValueError(f"Failed to get valid {schema.__name__} after {retries} attempts")


def execute_tool_call(tool_call, functions, schemas=None):
    name = tool_call.function.name
    if name not in functions:
        raise ValueError(f"no tool named '{name}' is available")
    args = json.loads(tool_call.function.arguments)
    if schemas is not None:
        args = schemas[name].model_validate(args).model_dump()
    result = functions[name](**args)
    return str(result)


def resolve_tools(messages, tools, functions, schemas=None, retries=3, model=None):
    messages = list(messages)
    for _ in range(retries):
        try:
            response = complete(model=model or current_model(), messages=messages, tools=tools)
        except BadRequestError:
            messages.append({
                "role": "user",
                "content": "That tool call could not be processed. Try again, calling one of the available tools correctly.",
            })
            continue

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            try:
                content = execute_tool_call(tool_call, functions, schemas)
            except Exception as e:
                content = f"Error: {e}"
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})

    raise ValueError(f"Failed to get a final answer after {retries} attempts")
