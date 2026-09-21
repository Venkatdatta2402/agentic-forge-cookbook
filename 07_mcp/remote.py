"""Turning a tool discovered over MCP into one of `03_tools`' `Tool` objects.

The point of this module is what it does *not* do. It does not produce a `Result`, and it does
not reimplement anything in `invoke.py`. It translates what MCP answers into the returns and
exceptions `invoke()` already understands, and lets `invoke()` do the classifying exactly as it
does for a local function:

    MCP says                        this module does        invoke() calls it
    ------------------------------  ----------------------  -----------------
    is_error False                  returns the text        ok
    _meta forge/kind = refused      raises ToolError        refused
    _meta forge/kind = unverified   raises VerificationError unverified
    _meta forge/kind = broken       raises RuntimeError     broken
    _meta forge/kind = timeout      raises TimeoutError     timeout
    is_error True, no _meta         raises ToolError        refused   (the one guess)
    the session is dead             raises                  broken

So there is one error mechanism in this repo, not two, and `02_agent_runtime` needs no more
change to run a remote tool than it needed to run a local one.

`forge/*` keys are our own convention on `_meta`, which is MCP's sanctioned place for
implementation-specific data. A server we wrote sends them; nobody else's does. The table above
is the honest division: with them, every ending survives the wire; without them, one distinction
is lost -- `refused` versus `broken` inside a tool that chose to say nothing about which it was.
"""

import concurrent.futures
import copy

import jsonschema
from mcp import MCPError
from pydantic import ValidationError

from invoke import ToolError
from tools import Tool
from verify import VerificationError

# What a foreign `is_error: True` becomes when the server offered no opinion. `refused` rather
# than `broken`, because the two mistakes are not the same size: calling a genuine bug `refused`
# costs a few retries, while calling a genuine refusal `broken` hides "try a larger nearby city"
# from the model, and the agent stalls holding advice it was never shown.
UNKNOWN_FAILURE = "refused"


class SchemaArgs:
    """The argument check for a tool whose schema arrived from the wire.

    `args_model()` in `03_tools` reads a Python signature and builds a Pydantic model from it.
    There is no signature here -- the schema *is* the primary artifact, and it came from
    somebody else's process -- so the check is built the other way round, straight from JSON
    Schema, with `jsonschema`.

    It is deliberately shaped like the Pydantic model `Tool` expects, because `Tool` should not
    have to know which end its check came from. That includes raising Pydantic's own
    `ValidationError`, which is what `invoke()` catches to say `bad_arguments`.

    Validating here is what keeps a malformed call off the wire entirely -- so `bad_arguments`
    stays a real, distinguishable ending instead of one more thing a foreign server may or may
    not bother to tell us apart from a crash.

    `strict` is the interesting argument, and notebook 1 predicted it. `args_model()` sets
    `extra="forbid"`, so a local tool rejects an argument it does not recognise; MCP does not
    require `additionalProperties: false` and the SDK does not emit it, so a discovered tool
    accepts `{"city": "Oslo", "citty": "Bergen"}` and quietly ignores the typo. The model then
    gets a plausible answer to a question it did not ask, which is the worst available outcome.

    So when the server's schema says nothing about extra properties, we say no on its behalf.
    This is a real judgement and not a free win: a server that *meant* to accept extras is
    broken by it. It is the default because silence here is far more often an omission than a
    decision -- but it is a flag, because that is a guess about somebody else's intent.
    """

    def __init__(self, schema, name="Arguments", strict=True):
        self.schema = copy.deepcopy(schema) if schema else {"type": "object", "properties": {}}
        if strict and "additionalProperties" not in self.schema:
            self.schema["additionalProperties"] = False
        self.name = name
        self.validator = jsonschema.Draft202012Validator(self.schema)

    def model_json_schema(self):
        # A copy, because `Tool.schema` mutates what it is given -- it strips Pydantic's `title`
        # keys. Which means wrapping a discovered tool in `Tool` also strips the titles the MCP
        # server put there, and the request we send the model is the leaner one after all.
        return copy.deepcopy(self.schema)

    def model_validate(self, args):
        errors = sorted(self.validator.iter_errors(args), key=lambda e: list(e.path))
        if errors:
            raise ValidationError.from_exception_data(
                self.name,
                [
                    {
                        "type": "value_error",
                        "loc": tuple(e.path),
                        "input": e.instance,
                        "ctx": {"error": ValueError(e.message)},
                    }
                    for e in errors
                ],
            )
        return _Checked(args)


class _Checked:
    """What `model_validate` hands back, so `Tool.validate` can call `.model_dump()` on it."""

    def __init__(self, args):
        self.args = args

    def model_dump(self):
        return dict(self.args)


def _text(result):
    """Everything the result said, as one string.

    `content` is a list of typed blocks and only some of them are text. Flattening loses the
    others, which is a real loss -- an image comes back as a placeholder -- and it is what
    `03_tools` assumed when it ended every call with `str(result)`. Named here rather than
    hidden, because notebook 3 has to decide what to do about it.
    """
    parts = []
    for block in result.content:
        parts.append(getattr(block, "text", None) or f"<{block.type}>")
    return "\n".join(parts) or "(no content)"


def _repeatable(mcp_tool):
    """Whether calling this again after a timeout is safe.

    MCP lets a server say so with `readOnlyHint` / `idempotentHint`. When it says nothing, the
    answer is no -- not because the tool is probably dangerous, but because we do not know, and
    this flag is only ever read after a timeout, where not knowing is exactly the situation
    that calls for caution.
    """
    hints = getattr(mcp_tool, "annotations", None)
    if hints is None:
        return False
    return bool(getattr(hints, "read_only_hint", False) or getattr(hints, "idempotent_hint", False))


def remote_tool(session, mcp_tool, name=None, strict=True):
    """One discovered MCP tool, as a `Tool` that `Registry` and `invoke()` can hold."""

    def call(**kwargs):
        result = session.call(mcp_tool.name, kwargs)

        meta = result.meta or {}
        kind = meta.get("forge/kind")
        cause = meta.get("forge/cause")

        if not result.is_error:
            return _text(result)

        if kind is None:
            kind = UNKNOWN_FAILURE

        if kind == "refused" or kind == "bad_arguments":
            # bad_arguments should be impossible -- SchemaArgs checked before we sent. If a
            # server says it anyway, its schema and its behaviour disagree, and that is still
            # something the model can act on.
            raise ToolError(_text(result))
        if kind == "unverified":
            raise VerificationError(_text(result))
        if kind == "timeout":
            raise concurrent.futures.TimeoutError(_text(result))
        # "broken", and anything a future server invents that we do not recognise.
        raise RuntimeError(cause or _text(result))

    return Tool(
        call,
        name=name or mcp_tool.name,
        description=mcp_tool.description or "",
        args=SchemaArgs(mcp_tool.input_schema, f"{mcp_tool.name}Arguments", strict=strict),
        repeatable=_repeatable(mcp_tool),
    )


def resource_reader(session, name="read_resource", uris=()):
    """A resource, handed to the model as a tool it may choose to call.

    Resources are host-pulled by design: the host reads one and decides whether it belongs in
    the context window. This inverts that, and it is what most real hosts actually ship,
    because an agent with no human at the wheel has nobody to do the choosing.

    The trade is worth stating rather than assuming. Host-pulled, the data arrives reliably and
    costs its tokens every single turn whether or not it was needed. Model-pulled, it costs
    nothing until asked for -- and the model has to think to ask, which it often does not.

    One thing comes back for free, and it is not a small one: as a tool, a resource read goes
    through `invoke()`, so `max_tokens` applies. A host that pastes `weather://history` into the
    window has no such limit and no warning.
    """
    listed = list(uris) or [str(r.uri) for r in session.list_resources()]
    allowed = ", ".join(listed) if listed else "(none listed)"

    def read(uri: str) -> str:
        try:
            contents = session.read_resource(uri).contents
        except MCPError as e:
            # A URI the server does not have. This arrives as a protocol error rather than as
            # `is_error`, so without this it lands in the `except Exception` arm of `invoke()`
            # and is classified `broken` -- hidden from the model as a fault it cannot fix.
            # It is the opposite: naming a resource that does not exist is exactly the kind of
            # mistake the model can correct, given the list.
            raise ToolError(f"{e}. Available: {allowed}") from e
        return "\n".join(
            getattr(c, "text", f"<{getattr(c, 'mime_type', 'binary')}>") for c in contents
        )

    read.__doc__ = f"Read one of this server's resources by URI. Available: {allowed}"
    return Tool(read, name=name)


def remote_tools(session, prefix=None, strict=True):
    """Every tool on a session, as `Tool` objects.

    `prefix` is not decoration. Two servers each offering `search` is the ordinary case, and
    `Registry.add` refuses the second one -- so the prefix is what makes merging possible at
    all. It is applied here rather than by `Registry.add(prefix=)` so the name the model sees
    and the name sent over the wire stay visibly different things.
    """
    tools = []
    for mcp_tool in session.list_tools():
        name = f"{prefix}_{mcp_tool.name}" if prefix else mcp_tool.name
        tools.append(remote_tool(session, mcp_tool, name=name, strict=strict))
    return tools
