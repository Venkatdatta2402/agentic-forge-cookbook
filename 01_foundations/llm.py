import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError, OpenAI
from pydantic import ValidationError

load_dotenv()

# Groq retired llama-3.3-70b-versatile for newer accounts -- `client.models.list()` no longer
# offers it, and a call returns 404 model_not_found. Much of this repo's tuning was measured
# against it, so where a note says "this model does X", check which one it means.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# max_retries covers HTTP 429s and 5xx, which the SDK retries with backoff, honoring Retry-After.
# The default of 2 is fine for one call at a time, but 02_agent_runtime runs branches and tool
# calls concurrently, and Groq's free-tier tokens-per-minute cap is easy to burst through.
# The daily cap is a different matter -- no amount of retrying clears that one.
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    max_retries=8,
)

async_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    max_retries=8,
)


def chat(messages, model=DEFAULT_MODEL, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content


def stream_chat(messages, model=DEFAULT_MODEL, **kwargs):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **kwargs,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def async_stream_chat(messages, model=DEFAULT_MODEL, **kwargs):
    stream = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **kwargs,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_and_print(messages, model=DEFAULT_MODEL, **kwargs):
    full = []
    for delta in stream_chat(messages, model=model, **kwargs):
        print(delta, end="", flush=True)
        full.append(delta)
    print()
    return "".join(full)


def _response_format(schema):
    """Ask the API to enforce the shape, if this model can.

    Groq's `json_schema` mode constrains the reply to the schema itself, rather than merely to
    "some valid JSON". llama-3.3-70b did not support it, which is why chapter 1's notebook 5
    teaches the validate-and-retry approach; gpt-oss-120b does. Both are kept: see extract().
    """
    return {"type": "json_schema",
            "json_schema": {"name": schema.__name__.lower(), "schema": schema.model_json_schema()}}


def extract(prompt, schema, retries=3, model=DEFAULT_MODEL):
    """Get a validated Pydantic object back from the model.

    Two lines of defence, and the second is not redundant.

    `json_schema` mode makes the API enforce the SHAPE -- right field names, right types, nothing
    missing. Where the model supports it that removes a whole class of failure at the source, and
    it is tried first. A model that rejects the mode falls back to plain JSON, so this still works
    against anything that can return `{...}` at all.

    What no schema can constrain is SENSE. Measured, from `SelectThink` asked to score three
    candidates out of ten: `scores=[869.0]` -- one number for three candidates, and 869 out of a
    stated maximum of 10 -- alongside `keep_indices=[20]`, an index into a list of length three.
    Every one of those is schema-valid. "One score per candidate" and "an index that exists" are
    not things a JSON schema can say.

    So the validation loop stays, and it stays for the same reason `DebugThink` gets the real
    error rather than a nudge: the model is told exactly what was wrong with what it produced.
    """
    messages = [{"role": "user", "content": prompt}]
    response_format = _response_format(schema)
    for _ in range(retries):
        try:
            reply = chat(messages, model=model, response_format=response_format)
        except BadRequestError:
            # this model cannot enforce schemas -- ask for plain JSON and lean on validation
            response_format = {"type": "json_object"}
            reply = chat(messages, model=model, response_format=response_format)
        messages.append({"role": "assistant", "content": reply})
        try:
            return schema.model_validate_json(reply)
        except (json.JSONDecodeError, ValidationError) as e:
            messages.append({
                "role": "user",
                "content": f"That was invalid: {e}\nReturn corrected JSON only, matching the schema.",
            })
    raise ValueError(f"Failed to get valid {schema.__name__} after {retries} attempts")


def tool_call_failure(error):
    """What Groq actually said about a rejected tool call, phrased back at the model.

    Groq validates tool-call generations server-side and returns HTTP 400 (so the SDK raises
    BadRequestError) in two distinct cases, and its body carries far more than the exception's
    str() shows:

      - broken syntax  -> `failed_generation` holds the exact malformed text it produced
      - wrong types    -> `message` names the offending field and the expected type

    Both are worth handing back verbatim. A generic "that didn't work, try again" tells the model
    nothing it can act on; the text below tells it precisely what it emitted and what was wrong
    with it -- the same reason DebugThink gets the real schema and the real error rather than a
    nudge. Returns None if this wasn't a tool-call rejection.
    """
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    err = body.get("error") if isinstance(body.get("error"), dict) else body
    if err.get("code") != "tool_use_failed":
        return None

    parts = [f"Your last tool call was rejected: {err.get('message', 'it could not be processed')}"]
    failed = err.get("failed_generation")
    if failed:
        parts.append(f"This is exactly what you generated, and it is not valid:\n{failed}")
    # Naming the wrapper specifically, because it is the failure mode this model actually has.
    # Watching every rejected attempt in one run, all five looked like:
    #     To answer these questions, I need to make two function calls.
    #     <function=get_current_time{"timezone": "Asia/Tokyo"}</function>
    # -- prose, then the call written as TEXT rather than emitted through the tool-calling
    # mechanism. A generic "issue the call again" left it repeating the same shape five times,
    # because from where it sat the call looked fine.
    parts.append(
        "Issue the call again as one complete, valid tool call. Do not write it out as text: "
        "anything of the form <function=name{...}</function> in your reply is not a tool call "
        "and will be rejected again. Do not explain first -- make the call and nothing else. "
        "Make one call, not several. Close every brace and bracket, "
        "give booleans as true/false rather than the strings \"true\"/\"false\", match every "
        "declared type, and leave out any optional field you do not actually need."
    )
    return "\n\n".join(parts)


def execute_tool_call(tool_call, functions, schemas=None):
    name = tool_call.function.name
    if name not in functions:
        raise ValueError(f"no tool named '{name}' is available")
    args = json.loads(tool_call.function.arguments)
    if schemas is not None:
        args = schemas[name].model_validate(args).model_dump()
    result = functions[name](**args)
    return str(result)


def resolve_tools(messages, tools, functions, schemas=None, retries=3, model=DEFAULT_MODEL):
    messages = list(messages)
    for _ in range(retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        except BadRequestError as e:
            messages.append({
                "role": "user",
                "content": tool_call_failure(e) or "That tool call could not be processed. Try again, calling one of the available tools correctly.",
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
