"""An agent is a definition: a name, a persona, a description, and the tools it may use.

From 06_orchestration (supervisor.py `Agent`, messages.py `form_tool` / `answer`), async. An agent
finishes by filling in a form: calling `json` stops its loop through the runtime's `cancel`
hook, and the program gets the pydantic object, not prose to parse.

The form's tool is called `json` (it was `submit`): handing in, gpt-oss called a tool named
'json' 6 times in the 5 baseline runs, and 'commentary' 3 times, with a complete
hand-in as the arguments each time. The runtime recovers a call made under a wrong name
(Think._recover); naming the tool what the model reaches for avoids most of them.
"""

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, is_dataclass, replace

from pydantic import BaseModel

from ..core import meter, trace
from ..core.runtime import Act, AsyncRuntime, Conversation, Observe, Think, fitting, parse_call
from ..core.tools import Registry, Tool, invoke

FORM = "json"
TRIES = 3          # an answer written as text gets read as the form this many times, at most


@dataclass
class Agent:
    name: str
    persona: str                       # its system prompt: how to BE this agent
    description: str = ""              # what a supervisor is told: when to ASK this agent
    tools: Registry = field(default_factory=Registry)
    model: str = ""
    max_tokens: int = 800              # per model call, reasoning included
    reasoning_effort: str | None = "low"
    max_iterations: int = 27           # components, so about 9 turns of Think/Act/Observe
    max_tool_calls: int = 10
    budget: int | None = None          # tokens this agent may spend on one request
    answer_on_stop: bool = True        # False: a stopped run returns no prose (see AsyncRuntime)
    shorten: tuple[str, ...] = ()      # tools whose old results are cut to two lines (see history)
    keep_recent: int = 2               # results kept whole however old (see history)
    form_build: Callable[[BaseModel], Awaitable[object]] | None = None   # see form_tool
    tool_timeout: float = 60           # seconds per tool call; the lead's ask_analysts runs minutes
    label: str | None = None           # its name in Langfuse, when that should say more ("analyst f1: <task>")
    choose: Callable | None = None     # conversation -> the tool the next call must use, or None (see Think)
    when_stuck: str | None = None      # the next move, said to it when it writes prose instead of a call (see Think)

    def runtime(self, cancel=None):
        return AsyncRuntime(
            loop=[Think(self.tools, self.model, self.max_tokens, self.reasoning_effort,
                        shorten=self.shorten, keep_recent=self.keep_recent, require_tool=True,
                        choose=self.choose, when_stuck=self.when_stuck),
                  Act(self.tools, timeout=self.tool_timeout), Observe()],
            max_iterations=self.max_iterations, max_tool_calls=self.max_tool_calls,
            max_tokens=self.budget, cancel=cancel, answer_on_stop=self.answer_on_stop)


class Stopped(BaseModel):
    """What comes back when an agent didn't hand in its form."""

    agent: str
    reason: str
    said: str = ""


def form_tool(model: type[BaseModel], holder: list, name=FORM,
              description="Hand in your result. Call this exactly once, when you are done.",
              build=None):
    """A pydantic model offered as a form.

    `build(form)` turns the form into the result the program keeps, or raises ToolError with what
    is wrong. A refused hand-in comes back to the agent as `refused` while it can still fix it --
    07_mcp's lesson that verification belongs where the world is.
    """
    async def fill(**fields):
        form = model(**fields)
        result = await build(form) if build is not None else form
        holder.append(result)
        return "Handed in."
    fill.__name__ = name
    return Tool(fill, name=name, description=description, args=model)


async def _read_as_form(text, form, conversation, tools):
    """A reply written as text: if it is the form's fields, hand it in through the same checks.
    Returns True if that worked; otherwise says what to do -- reply with one of the tools, not
    "hand in now": a lead that writes text before its findings are in still has tasks to give
    (span tests: the lead answered a question in text from the profiles alone)."""
    parsed = parse_call(text)
    if parsed is not None and fitting([form], parsed[1]):
        result = await invoke(form, parsed[1])
        if result.ok:
            trace.note("text read as the form")
            return True
        conversation.messages.append({"role": "user", "content": (
            f"Your reply was read as your hand-in, and refused: {result.content}\n"
            f"Fix that and call `{FORM}`.")})
        return False
    names = ", ".join(f"`{n}`" for n in tools.names)
    conversation.messages.append({"role": "user", "content": (
        f"Reply by calling one of your tools ({names}), not in text: `{FORM}` only when you have "
        "finished.")})
    return False


def system(persona):
    """An agent's system prompt: its persona, and how to hand in."""
    return (persona + f"\n\nWhen you have finished, call `{FORM}` with your result. Do not write your "
                      "result as ordinary text.")


async def answer(agent: Agent, request: str, form: type[BaseModel], context: str = "",
                 transcript: list | None = None, history: list | None = None):
    """Run `agent` on `request` until it hands in `form`. Returns the form, or `Stopped`.

    `transcript`, if given, IS the agent's conversation as it happens: the question's record holds
    it, so a question stopped halfway (a closed tab) still has every message so far.
    `history`: messages that already happened after the request (the span tests start the lead
    at a later step this way).
    """
    holder: list = []
    form_call = form_tool(form, holder, build=agent.form_build)
    working = replace(agent, tools=Registry([*agent.tools, form_call]))
    conversation = Conversation(system=system(agent.persona))
    if transcript is not None:
        transcript.clear()
        conversation.messages = transcript
    if context:
        conversation.messages.append({"role": "user", "content": context})
    conversation.messages.append({"role": "user", "content": request})
    conversation.messages.extend(history or [])
    with meter.acting_as(agent.name), trace.span(agent.name, input=request, label=agent.label) as observed:
        for _ in range(TRIES):
            await working.runtime(cancel=lambda: bool(holder)).run(conversation)
            last = conversation.messages[-1] if conversation.messages else {}
            if holder or conversation.stopped or last.get("role") != "assistant":
                break
            if await _read_as_form(last.get("content"), form_call, conversation, working.tools):
                break
        last = conversation.messages[-1] if conversation.messages else {}
        result = holder[0] if holder else Stopped(
            agent=agent.name, reason=conversation.stopped or "answered in text instead of handing in",
            said=(last.get("content") or "")[:1500])
        observed.output = (result.model_dump() if hasattr(result, "model_dump")
                           else asdict(result) if is_dataclass(result) else repr(result))
    return result
