import copy
import re

from tools import Tool

# What the API accepts as a tool name: letters, digits, underscore, hyphen, up to 64 characters.
# Checked when a tool is registered rather than when a request is sent, because the API rejects
# the WHOLE request over one bad name -- so a single mistyped tool takes every other tool with it,
# and the error arrives with no clue about which agent or which turn caused it.
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Registry:
    """The tools an agent can use, and the three views of them the runtime needs.

    Notebook 1 ended by building those three views by hand, in three comprehensions over one
    list. That works until the list gets long enough to fall out of sync with itself -- which is
    the same problem notebook 1 opened with, one level up.

    A note on the names. `tools` here means the JSON the model reads, and `schemas` means the
    Pydantic models that check arguments. That is backwards from how it sounds, and it is
    deliberate: those are the names chapter 2's `Think(tools=...)` and `Act(schemas=...)` already
    use, so the wiring reads straight at the call site. The actual `Tool` objects are reached by
    name -- `registry["get_weather"]` -- or by iterating the registry.
    """

    def __init__(self, tools=None):
        self._tools = {}
        self._tags = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool, name=None, prefix=None, tags=None):
        # A plain function is wrapped on the way in, so registering never needs the @tool
        # decorator to have been used at the definition site -- which matters for functions
        # coming from somewhere you don't control.
        if not isinstance(tool, Tool):
            tool = Tool(tool)

        name = name or tool.name
        if prefix:
            name = f"{prefix}_{name}"
        if not NAME.match(name):
            raise ValueError(
                f"'{name}' cannot be used as a tool name: only letters, digits, underscores and "
                "hyphens are allowed, up to 64 characters."
            )
        if name in self._tools:
            # A dict would take the second one silently. Two tools called `search` from two
            # different places is an ordinary thing to happen, and the quiet version of it means
            # every `search` call goes to whichever was registered last -- a tool that answers
            # confidently with the wrong thing, which is worse than one that fails.
            raise ValueError(
                f"a tool named '{name}' is already registered. Pass prefix= or name= to give "
                "this one a different name."
            )

        if name != tool.name:
            # A copy, because the name is part of what the model is shown: the schema is built
            # from tool.name, and it has to match the key the call comes back under. Renaming the
            # original would change it for everyone else holding a reference to it.
            tool = copy.copy(tool)
            tool.name = name

        self._tools[name] = tool
        self._tags[name] = set(tags or ())
        return tool

    def merge(self, other, prefix=None):
        """Take everything from another registry, optionally under a prefix."""
        for name, tool in other._tools.items():
            self.add(tool, name=name, prefix=prefix, tags=other._tags[name])
        return self

    @property
    def tools(self):
        """The JSON the model reads -- for chapter 2's `Think(tools=...)`."""
        return [tool.schema for tool in self._tools.values()]

    @property
    def functions(self):
        """Name -> the plain function -- for chapter 2's `Act(functions=...)`."""
        return {name: tool.function for name, tool in self._tools.items()}

    @property
    def schemas(self):
        """Name -> the Pydantic model that checks arguments -- for `Act(schemas=...)`."""
        return {name: tool.args for name, tool in self._tools.items()}

    def invokers(self, timeout=None, max_tokens=None):
        """Name -> the tool wrapped by `invoke`, ready for `Act(functions=...)`.

        Pass `schemas=None` to Act alongside this, so the arguments are checked once, here,
        rather than once by Act and again by invoke.
        """
        from invoke import invoker  # imported here so tools.py/registry.py stay standalone

        return {name: invoker(tool, timeout=timeout, max_tokens=max_tokens)
                for name, tool in self._tools.items()}

    @property
    def names(self):
        return list(self._tools)

    def tags(self, name):
        return self._tags[name]

    def subset(self, names=None, tags=None):
        """A smaller registry, holding the same tools.

        Every tool in a registry is described to the model on every single turn, so what an agent
        is given decides what it pays and how much it has to choose between. Handing over the
        whole registry is a decision, not a default.
        """
        wanted = list(names or [])
        if tags:
            tags = set(tags)
            wanted += [n for n in self._tools if self._tags[n] & tags and n not in wanted]
        missing = [n for n in wanted if n not in self._tools]
        if missing:
            # Quietly returning a smaller registry would produce an agent that is simply missing
            # a tool, and the run only goes wrong later, somewhere else, for reasons that look
            # nothing like a typo in a name.
            raise KeyError(f"no tool named {', '.join(repr(n) for n in missing)} is registered")

        smaller = Registry()
        for name in wanted:
            smaller.add(self._tools[name], tags=self._tags[name])
        return smaller

    def __getitem__(self, name):
        return self._tools[name]

    def __contains__(self, name):
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"<Registry {len(self._tools)} tools: {', '.join(self._tools)}>"
