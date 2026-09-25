"""What one agent sends another, and what comes back.

Notebook 2 passed plain strings both ways, and both directions leaked:

    the request   the supervisor asked engineering about "this serial" and left the serial out.
                  A whole agent run was spent answering "I need the specific serial number".
    the reply     support answered "no other batches are affected". Nothing in that string could
                  say how sure it was, where it came from, or that it was a guess about a batch
                  it cannot see -- so the supervisor had nothing to check and passed it on.

A shape fixes different halves of that. On the way in, a typed tool signature means a missing
serial is a rejected call, caught before any agent runs. On the way out, the agent finishes by
calling `submit`, so the fields exist because the model filled them in, not because something
parsed prose afterwards.

`submit` is the same trick `Chat` uses for handoffs: an ordinary tool whose effect is to stop
the loop, through Runtime's `cancel`.
"""

from dataclasses import replace
from typing import Annotated, Literal

from chat import Conversation
from pydantic import BaseModel, Field
from registry import Registry
from tools import Tool


class Response(BaseModel):
    """What an agent hands back. Every field is one the string in notebook 2 had nowhere to put."""

    answer: str = Field(
        description="Your answer, in full. This is what the person asking will act on.")
    confident: bool = Field(
        description="True only if your tools or the brief support every part of the answer.")
    visibility: Literal["internal", "external"] = Field(
        description="external if this may be sent to a customer, journalist or regulator as it "
                    "stands; internal if any of it must stay inside the company.")
    unknowns: str = Field(
        default="", description="Anything you could not establish. Empty if there is nothing.")
    ask_next: str = Field(
        default="", description="Which specialist should look at what, if anyone. Empty if nobody.")

    def render(self):
        """The form as the asking agent sees it in its conversation."""
        lines = [self.answer,
                 f"[confident: {'yes' if self.confident else 'no'}; visibility: {self.visibility}]"]
        if self.unknowns:
            lines.append(f"[not known: {self.unknowns}]")
        if self.ask_next:
            lines.append(f"[should be checked with: {self.ask_next}]")
        return "\n".join(lines)


def form_tool(model, holder, name="submit",
              description="Hand in your answer. Call this exactly once, when you are done."):
    """A pydantic model, offered to an agent as a form to fill in.

    `Tool(args=...)` takes a model written by hand instead of reading a signature, which is
    exactly what is wanted here: the fields are the shape of the answer, and the descriptions on
    them are what the model is told about each one.
    """
    def fill(**fields):
        holder.append(model(**fields))
        return "Submitted."

    fill.__name__ = name
    return Tool(fill, name=name, description=description, args=model)


def answer(agent, request, model=Response):
    """Run an agent that finishes by filling in a form, and return the form it filled in."""
    holder = []
    reporting = replace(agent, tools=Registry([*agent.tools, form_tool(model, holder)]),
                        persona=agent.persona + "\n\nWhen you have finished, call `submit` with your "
                                                "answer. Do not write your answer as ordinary text.")
    runtime = reporting.runtime()
    runtime.cancel = lambda: bool(holder)

    conversation = Conversation(system=reporting.persona)
    conversation.messages.append({"role": "user", "content": request})
    runtime.run(conversation)

    if holder:
        return holder[0]
    # It answered in prose instead of calling the tool. Said plainly rather than parsed into a
    # Response, because a form nobody filled in should not look like one that was.
    text = conversation.messages[-1]["content"]
    if model is not Response:
        raise ValueError(f"{agent.name} did not call submit; it said: {text[:200]}")
    return Response(answer=text, confident=False, visibility="internal",
                    unknowns="This agent answered as text instead of calling submit; nothing in it is checked.")


def as_form_tool(agent, log=None, holder=None):
    """An agent wrapped as a tool that answers with a form.

    `supervisor.as_tool` is the notebook 2 version of this: same wrapping, but the reply is
    whatever text the agent wrote. The name is different so that a call site says which it is.

    The asking agent still reads text, because a conversation is text -- but it is text with
    named fields, and the program around it gets the `Response` itself, which is the part that
    can be acted on without another model call.
    """
    def ask(request: Annotated[str, "Everything this specialist needs to know to do the job. "
                                    "They see nothing else: not the original message, not other replies."]) -> str:
        response = answer(agent, request)
        if log is not None:
            log.append({"agent": agent.name, "request": request, "response": response})
        if holder is not None:
            holder.append((agent.name, response))
        return response.render()

    return Tool(ask, name=f"ask_{agent.name}", description=agent.description)
