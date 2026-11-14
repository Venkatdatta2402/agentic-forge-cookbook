"""Tools, the registry, and what comes back from calling one.

From 03_tools (tools.py, registry.py, invoke.py), with one change: a tool's function may be a
coroutine. `invoke` awaits it, or runs a plain function in a worker thread, under one timeout.
Everything else is kept as the chapter measured it:

  - a tool's JSON schema and its argument check come from one signature (`args_model`)
  - `extra="forbid"`, so an argument the tool never declared is rejected, not silently dropped
  - every call ends as a `Result` with a kind (ok, bad_arguments, refused, timeout, broken),
    nothing is raised, and a `broken` result's cause stays out of the conversation
"""

import asyncio
import copy
import inspect
import re
from dataclasses import dataclass
from typing import Annotated, get_args, get_origin

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .llm import count_tokens, _encoding


class ToolError(Exception):
    """Raised by a tool to say: I ran, and the answer is no, for a reason the model can act on."""


def args_model(function, name=None):
    fields = {}
    for pname, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError(f"{function.__name__}: *{pname} cannot be described to a model")
        hint = parameter.annotation
        if hint is inspect.Parameter.empty:
            raise TypeError(f"{function.__name__}: parameter '{pname}' has no type annotation")
        description = None
        if get_origin(hint) is Annotated:
            hint, *extras = get_args(hint)
            description = next((e for e in extras if isinstance(e, str)), None)
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[pname] = (hint, Field(default, description=description))
    return create_model(f"{name or function.__name__}_args",
                        __config__=ConfigDict(extra="forbid"), **fields)


def _strip_titles(schema):
    """Trim what Pydantic emits to what the model needs; every token here is paid on every call.

    Titles say nothing the name doesn't. An optional field comes out as anyOf [type, null] with
    "default": null; not being in `required` already says it's optional. `additionalProperties`
    is dropped: unknown arguments are still refused, by the pydantic check on our side. Measured
    on the labour analyst: 837 tokens of tool schemas, sent with each of its 8 calls.
    """
    if isinstance(schema, dict):
        schema.pop("title", None)
        if schema.get("additionalProperties") in (False, True):     # True is JSON Schema's default
            schema.pop("additionalProperties")
        if schema.get("default", ...) is None:
            schema.pop("default")
        options = schema.get("anyOf")
        if isinstance(options, list):
            rest = [o for o in options if o != {"type": "null"}]
            if len(rest) == 1 and len(rest) < len(options):
                schema.pop("anyOf")
                schema.update(rest[0])
        for value in list(schema.values()):
            _strip_titles(value)
    elif isinstance(schema, list):
        for value in schema:
            _strip_titles(value)
    return schema


class SchemaArgs:
    """The argument check for a tool whose schema came from elsewhere (an MCP server), shaped like
    the pydantic model `Tool` expects (from 07_mcp's remote.py).

    Strict: an argument the schema doesn't declare is refused here, because the server won't
    refuse it. Measured on our analysis server: a call with `wher` instead of `where` was
    answered, the typo silently dropped, and the model would have read the wrong figure.
    """

    def __init__(self, schema, name="arguments"):
        self.schema = copy.deepcopy(schema) if schema else {"type": "object", "properties": {}}
        self.schema.setdefault("additionalProperties", False)
        self.name = name
        self.model_fields = list(self.schema.get("properties", {}))
        self._validator = jsonschema.Draft202012Validator(self.schema)

    def model_json_schema(self):
        return copy.deepcopy(self.schema)

    def model_validate(self, args):
        errors = sorted(self._validator.iter_errors(args), key=lambda e: list(e.path))
        if errors:
            raise ValidationError.from_exception_data(self.name, [
                {"type": "value_error", "loc": tuple(e.path), "input": e.instance,
                 "ctx": {"error": ValueError(e.message)}} for e in errors])
        return _Checked(args)


class _Checked:
    def __init__(self, args):
        self.args = args

    def model_dump(self):
        return dict(self.args)


class Tool:
    def __init__(self, function, name=None, description=None, args: type[BaseModel] | None = None,
                 repeatable=True):
        self.function = function
        self.name = name or function.__name__
        self.repeatable = repeatable   # safe to call again after a timeout? (reads: yes)
        self.description = description or inspect.getdoc(function) or ""
        self.args = args or args_model(function, self.name)

    @property
    def schema(self):
        parameters = _strip_titles(self.args.model_json_schema())
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": parameters}}

    def validate(self, args):
        return self.args.model_validate(args).model_dump()

    def __repr__(self):
        return f"<Tool {self.name}({', '.join(self.args.model_fields)})>"


def tool(function=None, *, name=None, description=None, args=None, repeatable=True):
    if function is None:
        return lambda f: Tool(f, name=name, description=description, args=args, repeatable=repeatable)
    return Tool(function, name=name, description=description, args=args, repeatable=repeatable)


NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Registry:
    """The tools one agent may use, by name; a name can be registered only once."""

    def __init__(self, tools=None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t, name=None):
        if not isinstance(t, Tool):
            t = Tool(t)
        name = name or t.name
        if not NAME.match(name):
            raise ValueError(f"'{name}' cannot be used as a tool name")
        if name in self._tools:
            raise ValueError(f"a tool named '{name}' is already registered")
        if name != t.name:
            t = copy.copy(t)
            t.name = name
        self._tools[name] = t
        return t

    @property
    def schemas(self):
        """The JSON the model reads."""
        return [t.schema for t in self._tools.values()]

    @property
    def names(self):
        return list(self._tools)


    def __getitem__(self, name):
        return self._tools[name]

    def __contains__(self, name):
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self):
        return len(self._tools)


OURS_TO_FIX = "broken"


@dataclass
class Result:
    tool: str
    kind: str          # ok | bad_arguments | refused | timeout | broken
    content: str       # what the conversation sees
    value: object = None
    truncated: bool = False
    repeatable: bool = True

    @property
    def ok(self):
        return self.kind == "ok"

    @property
    def for_the_model(self):
        return self.kind != OURS_TO_FIX

    @property
    def as_seen(self):
        """The text the model is shown. A broken tool is ours to fix, so its cause isn't in it."""
        return self.content if self.for_the_model else f"{self.tool} could not be completed."


def truncate(text, max_tokens):
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    kept = _encoding.decode(tokens[:max_tokens])
    return f"{kept}\n[... cut short: {len(tokens):,} tokens in all, {max_tokens:,} kept]", True


async def invoke(t: Tool, args, timeout=None, max_tokens=None) -> Result:
    try:
        checked = t.validate(args)
    except ValidationError as e:
        return Result(t.name, "bad_arguments", _short_validation(e), repeatable=t.repeatable)
    try:
        if inspect.iscoroutinefunction(t.function):
            work = t.function(**checked)
        else:
            work = asyncio.to_thread(t.function, **checked)
        value = await asyncio.wait_for(work, timeout) if timeout else await work
    except TimeoutError:
        advice = (" It only reads, so calling it again is safe." if t.repeatable else
                  " It may or may not have gone through. Do not call it again -- check first.")
        return Result(t.name, "timeout", f"'{t.name}' took longer than {timeout} seconds."
                      + advice, repeatable=t.repeatable)
    except ToolError as e:
        return Result(t.name, "refused", str(e), repeatable=t.repeatable)
    except Exception as e:  # noqa: BLE001 -- a bug or an outage: ours to fix, kept off the conversation
        return Result(t.name, "broken", f"{t.name} is broken: {type(e).__name__}: {e}",
                      repeatable=t.repeatable)

    content = value if isinstance(value, str) else str(value)
    truncated = False
    if max_tokens is not None:
        content, truncated = truncate(content, max_tokens)
    return Result(t.name, "ok", content, value=value, truncated=truncated, repeatable=t.repeatable)


def _short_validation(error: ValidationError):
    """Pydantic's message, without the URLs to its docs: the field and the problem are what help."""
    lines = []
    for e in error.errors():
        where = ".".join(str(p) for p in e["loc"]) or "arguments"
        lines.append(f"{where}: {e['msg']}")
    return "Invalid arguments -- " + "; ".join(lines)


__all__ = ["Registry", "Result", "SchemaArgs", "Tool", "ToolError", "args_model", "count_tokens",
           "invoke", "tool", "truncate"]
