import inspect
from typing import Annotated, get_args, get_origin

from pydantic import ConfigDict, Field, create_model


def args_model(function, name=None):
    """Build the Pydantic model that checks a function's arguments, out of its own signature.

    Written by hand, a tool is the same fact written three times: the function, the JSON the
    model reads, and the check on what comes back. Nothing keeps those three equal, so renaming
    a parameter breaks the tool without breaking anything loudly.

    Read off the signature instead, there is only one copy. Rename a parameter and the JSON and
    the check both follow.

    Types are required here, not optional. A parameter with no type is one the model is told
    nothing about, and guessing a type from the default value would quietly produce a
    description that doesn't match the function.
    """
    fields = {}
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            # There is no way to describe *args/**kwargs, so a model given this tool would be
            # guessing at what to send. Refused now, while whoever wrote the tool can still fix
            # it, rather than later in front of the model.
            raise TypeError(
                f"{function.__name__}: *{parameter_name} cannot be described to a model. "
                "Declare the arguments the tool actually takes."
            )
        hint = parameter.annotation
        if hint is inspect.Parameter.empty:
            raise TypeError(
                f"{function.__name__}: parameter '{parameter_name}' has no type annotation, "
                "so there is nothing to tell the model about it."
            )
        # Annotated[str, "the city to look up"] puts each parameter's description in the
        # signature. The alternative was reading them out of a docstring's "Args:" block, which
        # means picking one docstring style: write it any other way and the descriptions vanish
        # from the schema with no error at all, leaving a tool that just works slightly worse.
        description = None
        if get_origin(hint) is Annotated:
            hint, *extras = get_args(hint)
            description = next((e for e in extras if isinstance(e, str)), None)
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter_name] = (hint, Field(default, description=description))

    return create_model(
        f"{name or function.__name__}_args",
        # extra="forbid" rejects an argument the function never declared. Pydantic normally
        # ignores those instead: model_dump() drops them and the function runs as if they were
        # never sent, so the model gets a normal-looking result back and never learns that part
        # of what it asked for was thrown away.
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class Tool:
    """A function, the JSON the model reads, and the check on what it sends back -- one object,
    with the last two worked out from the first.

    `args` takes a Pydantic model written by hand instead, for rules a signature cannot express:
    `01_foundations/06_function_calling` rejected a divisor of zero with a field_validator, and
    no type says "not zero". Reading the signature is the default, not the only option.
    """

    def __init__(self, function, name=None, description=None, args=None, repeatable=True):
        self.function = function
        self.name = name or function.__name__
        # Whether calling this twice is the same as calling it once. Only looked at when a call
        # times out, which is the one moment nobody knows whether it ran -- see invoke.py. Tools
        # that only read are safe; anything that sends, pays, or deletes is not.
        self.repeatable = repeatable
        # getdoc, not __doc__: it strips the indentation every line of a docstring below the
        # first one carries, which would otherwise be sent to the model as it is.
        self.description = description or inspect.getdoc(function) or ""
        self.args = args or args_model(function, self.name)

    @property
    def schema(self):
        """The entry for the API's `tools=[...]` list."""
        parameters = self.args.model_json_schema()
        # Pydantic gives every model and every field a title ("Args Model", "City") for the
        # benefit of documentation tools. They say nothing the field name doesn't, and they cost
        # tokens in every request, so they come out.
        parameters.pop("title", None)
        for field in parameters.get("properties", {}).values():
            field.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def validate(self, args):
        """Arguments in, checked and converted, back out as a plain dict -- or ValidationError."""
        return self.args.model_validate(args).model_dump()

    def __call__(self, **kwargs):
        # Checking here means there is no way to reach the function without going past its own
        # description first. This is the first half of chapter 1's execute_tool_call(). What
        # goes around it -- timeouts, failures handed back instead of raised, results too big to
        # put in a conversation -- belongs to invoke.py, not to the tool.
        return self.function(**self.validate(kwargs))

    def __repr__(self):
        return f"<Tool {self.name}({', '.join(self.args.model_fields)})>"


def tool(function=None, *, name=None, description=None, args=None, repeatable=True):
    """Decorator form: @tool, or @tool(description=...) when something needs overriding."""
    if function is None:
        return lambda f: Tool(f, name=name, description=description, args=args,
                              repeatable=repeatable)
    return Tool(function, name=name, description=description, args=args, repeatable=repeatable)
