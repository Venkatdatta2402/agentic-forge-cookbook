"""Stand-ins for the two paid services, so all of 07 runs offline first.

    fake_groq()     05's real `chat_model` -- its metering, the LangSmith wrapper, LlamaIndex -- with only the
                    network replaced: Groq's replies are scripted at the HTTP layer, and every request body is
                    kept, so a test can compare what was sent with tracing on and off
    FakeGemini      the judge's client: returns scripted verdicts, or raises Gemini's real error for a rate limit
    rate_limited()  that error, per minute (with the wait Gemini asks for) or per day
"""

import json

import httpx

import ask_index as A


def _completion(content=None, tool=None, prompt_tokens=900, completion_tokens=60):
    message = {"role": "assistant", "content": content}
    if tool:
        name, args = tool
        message["tool_calls"] = [{"id": f"call_{name}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
    return {"id": "fake", "object": "chat.completion", "created": 0, "model": A.MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}}


def fake_groq(agent_steps=(("search_cookbook", {"query": "shared board owner"}),), answer="An answer [1].",
              agent_answer="The board is not memory [06_orchestration/state.py]."):
    """A `chat_model` whose clients reach a scripted Groq. Returns (chat_model, requests sent).

    Offered no tools (the engine), it answers. Offered tools (the agent), it makes `agent_steps` calls in order,
    one per request, then answers.
    """
    sent = []

    def reply(request):
        body = json.loads(request.content)
        sent.append(body)
        if not body.get("tools"):
            return httpx.Response(200, json=_completion(answer))
        tool_results = sum(m["role"] == "tool" for m in body["messages"])
        if tool_results < len(agent_steps):
            return httpx.Response(200, json=_completion(tool=agent_steps[tool_results]))
        return httpx.Response(200, json=_completion(agent_answer))

    def through_fake(client, wrap=None):
        import openai
        http = (httpx.AsyncClient if isinstance(client, openai.AsyncOpenAI) else httpx.Client)(
            transport=(httpx.MockTransport(reply)))
        fake = client.with_options(http_client=http)
        return wrap(fake) if wrap else fake

    def chat_model(budget=None, wrap=None, **kwargs):
        return A.chat_model(budget=budget, wrap=lambda client: through_fake(client, wrap), **kwargs)

    return chat_model, sent


def rate_limited(per_day=False, delay="7s"):
    """Gemini's 429, as its SDK raises it: what quota was hit, and how long to wait."""
    from google.genai import errors
    quota = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if per_day else \
        "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota",
                      "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                   "violations": [{"quotaId": quota}]},
                                  {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}
    return errors.ClientError(429, body)


class FakeGemini:
    """`client.models.generate_content`, replying from a script: a dict in the shape the call asks for (the
    `response_schema` it is given), or an exception to raise."""

    def __init__(self, script):
        self.script, self.prompts = list(script), []
        self.models = self

    def generate_content(self, model=None, contents=None, config=None):
        from types import SimpleNamespace as NS
        self.prompts.append(contents)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        parsed = config.response_schema.model_validate(step)
        return NS(parsed=parsed, text=parsed.model_dump_json(), usage_metadata=NS(total_token_count=1234))
