"""Tracing 05's cookbook assistant in LangSmith, at the seams 05 leaves open for it.

LangSmith traces LangChain by itself and LlamaIndex not at all. Its route for LlamaIndex is OpenTelemetry
(`openinference-instrumentation-llama-index`), which works for a run on its own and loses every LlamaIndex step
inside an evaluation experiment (`traces_checks.otel_ids`). So this traces with LangSmith's own tools:

    wrap_openai     around 05's model clients (`chat_model(wrap=...)`): every model call, with its whole prompt,
                    its reply, its tokens and how long it took
    traced_tool     around the agent's two tools (`agent(wrap=...)`): each call, its arguments, what it returned
    traced_engine   around the one-shot engine's retrieval: the passages found, before the model sees them

All three nest under whatever run is current, so inside `evaluate()` each question's trace sits under its row.
"""

import functools
import inspect


def wrap_client(client):
    """LangSmith's `wrap_openai`: every call on the client becomes an `llm` run."""
    from langsmith.wrappers import wrap_openai
    return wrap_openai(client)


def traced_tool(function):
    """A tool function as a LangSmith `tool` run, with the signature it had.

    `traceable` alone adds a `config` keyword to the function's signature, and LlamaIndex reads a tool's
    schema off the signature: the model would be offered a `config` argument it was never offered untraced,
    and a traced run would no longer be the system being evaluated. So the traced function is called through
    one whose signature is the original's.
    """
    from langsmith import traceable
    traced = traceable(run_type="tool", name=function.__name__)(function)

    @functools.wraps(function)
    def call(*args, **kwargs):
        return traced(*args, **kwargs)

    call.__signature__ = inspect.signature(function)
    return call


def _passages(nodes):
    # LangSmith shows a retriever run's output as documents when it is a list of {page_content, metadata}
    return {"documents": [{"page_content": n.node.get_content(), "type": "Document",
                           "metadata": {"file": n.node.metadata.get("file"), "score": n.score}} for n in nodes]}


def traced_engine(query_engine):
    """The engine's retrieval as a `retriever` run: what was found, and how near, before any model call."""
    from langsmith import traceable
    retriever = query_engine.retriever
    inner = retriever._retrieve
    retriever._retrieve = traceable(run_type="retriever", name="CookbookRetriever",
                                    process_inputs=lambda inputs: {"query": inputs["query_bundle"].query_str},
                                    process_outputs=_passages)(inner)
    return query_engine
