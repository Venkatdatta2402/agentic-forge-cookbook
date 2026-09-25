"""What several agents are working on at once, while they work on it.

This is not memory. `04_memory` keeps what should outlive a run, in a store on disk; a case like
"the customer who wrote in about AR2-2409-01544" is finished when the reply goes out. What this
holds is the working state of that case, alive for minutes, read on nearly every turn by
whoever touches it -- so it lives in an ordinary object in memory and is put into each agent's
context as it is needed.

Three decisions are made by the person wiring the agents together, not by the agents:

    who may read a field    `readers`, by field. Default: everyone on the case.
    who may write it        `owner`, by field. Default: nobody -- the field must be claimed
                            explicitly, because a value two agents can both set is a value
                            neither of them can trust.
    what stays private      anything an agent keeps to itself never goes on the board at all.

Reading is injection: `view(agent)` renders what that agent is allowed to see, and the caller
puts it in the prompt. Writing is a tool, bound to one agent's name, so a write carries an
author and ownership can actually be checked.
"""

from typing import Annotated

from tools import Tool


class Denied(Exception):
    """An agent tried to write a field it does not own."""


class Shared:
    """A small board several agents read, and a declared few write.

    Shared(fields={"batch": "which batch the unit is from", ...},
           owner={"cell_lot": "engineer"},
           readers={"legal_risk": ["lawyer", "supervisor"]})
    """

    def __init__(self, fields=None, owner=None, readers=None):
        self.fields = dict(fields or {})       # name -> what it means, shown to the agents
        self.owner = dict(owner or {})         # name -> the one agent allowed to write it
        self.readers = dict(readers or {})     # name -> who may see it (absent: everyone)
        self.values = {}
        self.history = []                      # every write, in order: who, what, from, to
        self.watchers = {}                     # field -> things to run when it is written

    # ----------------------------------------------------------------- reading
    def visible_to(self, agent):
        return {name: self.values[name] for name in self.values
                if agent in self.readers.get(name, (agent,))}

    def view(self, agent):
        """What this agent may see, as the text that goes into its context."""
        visible = self.visible_to(agent)
        lines = []
        for name, meaning in self.fields.items():
            if name in visible:
                who = self.owner.get(name)
                lines.append(f"- {name}: {visible[name]}" + (f"  (set by {who})" if who else ""))
            elif name in self.readers and agent not in self.readers[name]:
                continue                       # not this agent's business that it exists
            else:
                lines.append(f"- {name}: (not established yet) -- {meaning}")
        return "What the team has established so far:\n" + "\n".join(lines) if lines else ""

    # ----------------------------------------------------------------- writing
    def set(self, author, field, value):
        owner = self.owner.get(field)
        if owner is not None and owner != author:
            # Refused rather than recorded. The point of an owner is that the field means the
            # same thing every time it is read, which survives exactly as long as nobody else
            # can set it -- see the two-writers section in notebook 3.
            raise Denied(f"{field} is {owner}'s to set, not {author}'s")
        before = self.values.get(field)
        self.values[field] = value
        self.history.append({"author": author, "field": field, "from": before, "to": value})
        for watcher in self.watchers.get(field, ()):
            # A board says where information lives, never who acts on it. This is the smallest
            # thing that turns a write into an activation: when this field changes, run this.
            # Notebook 4 uses it as the event-driven alternative to a manager choosing.
            watcher(self, field, value)
        return before

    def watch(self, field, callback):
        """Run `callback(board, field, value)` whenever this field is written."""
        self.watchers.setdefault(field, []).append(callback)
        return callback

    def tools(self, agent):
        """The write tool for one agent, with its name and the fields it owns already in it."""
        owned = [f for f, who in self.owner.items() if who == agent]
        if not owned:
            return []
        described = ", ".join(f"{f} ({self.fields.get(f, '')})" for f in owned)

        def record(
            field: Annotated[str, f"Which field to set. Yours are: {described}"],
            value: Annotated[str, "The value, short and exact -- what the others will read."],
        ) -> str:
            """Record something the rest of the team needs, on the shared board."""
            try:
                before = self.set(agent, field, value)
            except Denied as refused:
                return f"Refused: {refused}"
            return (f"{field} is now: {value}" if before is None
                    else f"{field} changed from '{before}' to '{value}'")

        return [Tool(record, name="record")]


def on_board(agent, board):
    """The same agent, with what it may see rendered into its context and its own write tool.

    Built fresh for each turn, because the board changes while the team works: an agent that
    read it once would be working from what was true when it started.
    """
    from dataclasses import replace

    from registry import Registry

    view = board.view(agent.name)
    return replace(agent,
                   persona=f"{agent.persona}\n\n{view}" if view else agent.persona,
                   tools=Registry([*agent.tools, *board.tools(agent.name)]))
