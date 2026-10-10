"""What LangSmith does that you have to know, each reproduced offline: no LangSmith call, no model call.

    otel_ids()            why LlamaIndex's OpenTelemetry steps vanish inside an evaluation experiment
    traceable_signature() what `traceable` does to a tool's schema, and what `tracing.traced_tool` keeps
    tracing_is_invisible() the requests 05 sends Groq, traced and untraced, compared byte for byte
"""

import uuid


def otel_ids():
    """The two halves of LangSmith disagree on how a run's id becomes an OpenTelemetry span id.

    Inside `evaluate()` each row's run is created by LangSmith's client, with a 16-byte UUID. OpenTelemetry span
    ids are 8 bytes, so when LlamaIndex's instrumentation opens a step under that run, the client hands it a span
    id made of the run id's FIRST 8 bytes. LangSmith's server turns a span id back into a run id by padding it on
    the LEFT with zeros (`langsmith_run_id_from_otel_span_id`, whose docstring states the rule). The step's parent
    comes back as a run that does not exist, and the step is not shown. Measured live while building this: 14
    LlamaIndex steps on their own, all in LangSmith; the same query inside an experiment, none.
    """
    from langsmith._internal._otel_utils import get_otel_span_id_from_uuid
    from langsmith.integrations.otel import langsmith_run_id_from_otel_span_id
    run_id = uuid.UUID("01a11ad7-b618-71b3-b1ab-d927d8bc4db1")       # a row's run id, from that live test
    span_id = get_otel_span_id_from_uuid(run_id)
    back = langsmith_run_id_from_otel_span_id(span_id)
    return {"the row's run id": str(run_id), "span id its children are given": hex(span_id),
            "the run id the server makes of it": str(back), "the same run": back == run_id}


def traceable_signature():
    """`traceable` adds a `config` keyword; LlamaIndex reads a tool's schema off its signature."""
    import inspect

    from langsmith import traceable
    from llama_index.core.tools import FunctionTool

    import tracing as T

    def search_cookbook(query: str) -> str:
        """Search the cookbook."""
        return query

    schema = lambda f: sorted(FunctionTool.from_defaults(f).metadata.get_parameters_dict()["properties"])
    return {"plain": {"signature": str(inspect.signature(search_cookbook)), "the model is offered": schema(search_cookbook)},
            "traceable": {"signature": str(inspect.signature(traceable(run_type="tool")(search_cookbook))),
                          "the model is offered": schema(traceable(run_type="tool")(search_cookbook))},
            "tracing.traced_tool": {"signature": str(inspect.signature(T.traced_tool(search_cookbook))),
                                    "the model is offered": schema(T.traced_tool(search_cookbook))}}


async def tracing_is_invisible(index, question, agent_question):
    """One question through 05's engine and one through its agent, with LangSmith's wrappers and without, against
    a scripted Groq: if the requests differ, the traced system is not the system being evaluated."""
    from langsmith import tracing_context

    import ask_index as A
    import eval_stubs as S
    import tracing as T
    sent = {}
    with tracing_context(enabled=False):
        for traced in (False, True):
            chat_model, requests = S.fake_groq()
            wrap, tools = (T.wrap_client, T.traced_tool) if traced else (None, None)
            A.ask(A.engine(index, chat_model(budget=A.Budget(10 ** 6), wrap=wrap)), question)
            await A.run_agent(A.agent(index, chat_model(budget=A.Budget(10 ** 6), wrap=wrap), wrap=tools),
                              agent_question)
            sent[traced] = requests
    return {"requests, untraced": len(sent[False]), "requests, traced": len(sent[True]),
            "identical": sent[False] == sent[True],
            "tools offered, traced": [t["function"]["name"] + str(sorted(t["function"]["parameters"]["properties"]))
                                      for t in sent[True][1]["tools"]]}
