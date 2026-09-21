"""Exposing a `03_tools` `Registry` over MCP -- the other direction from `remote.py`.

`remote.py` takes somebody else's tools and makes them ours. This takes ours and makes them
somebody else's, and the whole value of doing it in that order is that we already know exactly
what a client wishes a server would do, because notebook 2 was spent working around servers
that did not.

Three things this fixes, all of them things notebook 1 measured and complained about:

  the advertised schema      comes from `Tool.schema`, so it cannot drift from the function,
                             carries no Pydantic `title` noise, and says
                             `additionalProperties: false` like a local tool does
  the argument check         happens exactly once, in `invoke()`, so a bad call produces a
                             real `bad_arguments` Result rather than an SDK string
  the failure kind           rides across in `_meta`, so a client that understands `forge/*`
                             recovers the whole taxonomy instead of inferring from one bool

The last one is the point of the chapter. `07_mcp/remote.py` reads those keys, so our client
talking to our server loses nothing at all -- and the notebook measures that against the same
client talking to a server that says nothing.
"""

import inspect
import warnings
from typing import Any

import mcp.types as t
from mcp.server.mcpserver import MCPServer
from pydantic.json_schema import PydanticJsonSchemaWarning

from invoke import invoke

# Stands in for "the caller did not send this argument". A plain `None` default cannot do the
# job: `None` is a value a caller might legitimately send, and the two have to stay
# distinguishable all the way to the one place that is allowed to have an opinion about it.
MISSING = object()


def _permissive_signature(tool):
    """A signature accepting anything the advertised schema might name, all optional.

    This looks like giving up on validation and is the opposite. `MCPServer` builds its own
    Pydantic model from the handler's signature and checks arguments against it *before* the
    handler runs -- so a strict signature here means every bad call is rejected by the SDK,
    with an SDK-shaped error, before `invoke()` ever sees it. The caller then gets a string
    instead of a `bad_arguments` Result, and none of chapter 3's classification happens.

    So the advertised schema and the internal check are deliberately different things. The
    schema on the wire is strict, because that is what the model reads. The gate in front of
    the handler is wide open, because the real check lives one layer further in, where it can
    produce a `Result` with a `kind`.
    """
    properties = tool.args.model_json_schema().get("properties", {})
    return inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=MISSING, annotation=Any)
        for name in properties
    ])


def _result_to_mcp(result):
    """A `03_tools` `Result`, as far across the wire as MCP will carry it.

    `content` is what the model may read, and `for_the_model` decides whether that is the real
    message or an acknowledgement -- the same rule `invoker()` applies locally, applied here
    because the conversation is on the other side of the wire and cannot apply it itself.

    Everything else goes in `_meta`, which MCP passes through untouched and never shows a
    model. The keys are namespaced because it is shared ground: the SDK puts its own
    `io.modelcontextprotocol/serverInfo` in the same dict.
    """
    meta = {
        "forge/kind": result.kind,
        "forge/repeatable": result.repeatable,
        "forge/truncated": result.truncated,
    }
    if not result.for_the_model:
        # The cause travels, but in the half of the message a model never reads. This is the
        # thing notebook 1 found no server doing, and the reason it could not be done from the
        # client side: only the server knows which of its failures was a bug.
        #
        # `invoke()` framed the content as "<tool> is broken: ..." for a local reader. That
        # framing is stripped before sending, because the client's own `invoke()` will add its
        # own on the way back in -- and a cause that arrives pre-framed comes out doubled,
        # "pressure is broken: RuntimeError: pressure is broken: KeyError".
        prefix = f"{result.tool} is broken: "
        cause = result.content
        meta["forge/cause"] = cause[len(prefix):] if cause.startswith(prefix) else cause
        text = f"{result.tool} could not be completed."
    else:
        text = result.content

    return t.CallToolResult(
        content=[t.TextContent(type="text", text=text)],
        is_error=not result.ok,
        meta=meta,
    )


def _handler(tool, timeout, max_tokens):
    def call(**kwargs) -> t.CallToolResult:
        sent = {name: value for name, value in kwargs.items() if value is not MISSING}
        return _result_to_mcp(invoke(tool, sent, timeout=timeout, max_tokens=max_tokens))

    call.__name__ = tool.name
    call.__doc__ = tool.description
    call.__signature__ = _permissive_signature(tool)
    return call


def expose(registry, name="forge-registry", version="0.1.0", timeout=None, max_tokens=None,
           names=None, tags=None):
    """Build an MCP server offering the tools in `registry`.

    `names` and `tags` narrow what is published, and they are not a convenience. A `Registry`
    is the set of tools *this agent* may use; a server is the set of tools *anyone who
    connects* may use. Those are different questions with different answers -- `03_tools`'
    file tools are confined to a folder on this machine, which is a sentence about this
    machine and not a promise to a stranger -- so publishing defaults to nothing implicit and
    the caller has to mean it.
    """
    published = registry.subset(names=names, tags=tags) if (names or tags) else registry

    server = MCPServer(name, version=version)
    for tool in published:
        # `add_tool` generates a JSON schema from the permissive signature, and Pydantic warns
        # that the MISSING sentinel is not serializable as a default. It is right, and it does
        # not matter: that schema is overwritten two lines below with `Tool.schema`, and the
        # sentinel never reaches the wire. Suppressed rather than worked around, because the
        # alternative is choosing a sentinel for the schema generator's benefit instead of for
        # the one job it has -- being distinguishable from every value a caller might send.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PydanticJsonSchemaWarning)
            server.add_tool(
                _handler(tool, timeout, max_tokens),
                name=tool.name,
                description=tool.description,
                # `Tool.repeatable` said whether calling this again after a timeout is safe,
                # and MCP has a standard place to say so. Using it rather than another
                # `forge/` key matters: annotations arrive at *discovery*, which is when the
                # client builds its `Tool` and has to decide -- and any client reads them,
                # not only ours.
                annotations=t.ToolAnnotations(idempotent_hint=tool.repeatable),
            )
        # The advertised schema, replacing the one MCPServer just generated from the permissive
        # signature. `Tool.schema` read it off the function, so this is the same description
        # the model would see if the tool were local -- titles stripped, extras forbidden.
        server._tool_manager._tools[tool.name].parameters = tool.schema["function"]["parameters"]

    return server
