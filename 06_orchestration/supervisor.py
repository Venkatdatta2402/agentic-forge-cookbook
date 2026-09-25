"""Agents defined once, a supervisor that calls them, and a conversation that can change hands.

Notebook 1 wrote `Agent` inside the notebook. It lives here now because notebook 2 needs more
than one of them, and something that calls them:

    Agent       a name, a persona, a description, and the tools it may use
    as_tool     an agent wrapped as a chapter 3 Tool, so another agent can call it
    Chat        a conversation with a customer that one agent at a time owns, and that an agent
                can hand to another one

Nothing here is a new kind of runtime. A supervisor is an ordinary Agent whose tools happen to be
other agents, running chapter 2's loop unchanged. A handoff is an ordinary tool whose effect is
to change who answers next.
"""

from dataclasses import dataclass, field, replace
from typing import Annotated

import think
from act import Act
from chat import Conversation
from observe import Observe
from registry import Registry
from runtime import Runtime
from tools import Tool

MODEL = "openai/gpt-oss-20b"


class Think(think.Think):
    """Chapter 2's Think with a ceiling on output that can be set.

    gpt-oss-20b stops at 2,048 output tokens unless told otherwise, and its hidden reasoning
    counts. On a long task it can spend the lot thinking and return an empty answer.
    """

    def __init__(self, *args, max_tokens=16000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens = max_tokens

    def _sampling_kwargs(self):
        return {**super()._sampling_kwargs(), "max_tokens": self.max_tokens}


def verbatim(effect):
    # What a tool returned goes into the conversation as it is. Observe's default is to have a
    # model restate it in one sentence, which is right for a sensor reading and wrong for another
    # agent's reply: the reply IS the work, and a one-sentence paraphrase of it throws the work away.
    return effect["content"]


@dataclass
class Agent:
    name: str
    persona: str                  # its system prompt: how to BE this agent
    description: str = ""         # what other agents are told: when to ASK this agent
    tools: Registry = field(default_factory=Registry)
    model: str = MODEL
    # Runtime counts every COMPONENT it runs, not every turn, and this loop has three of them.
    # So 24 is about eight turns: enough for a couple of tool calls, a look at what came back,
    # and an answer. It was 8 while notebook 2 ran, which is under three turns, and that is why
    # a specialist there ran out mid-job -- see notebook 2's dynamic delegation section.
    max_iterations: int = 24

    def runtime(self):
        return Runtime(
            loop=[Think(model=self.model, tools=self.tools.tools, allow_fork=False, explain=False),
                  Act(functions=self.tools.functions, schemas=self.tools.schemas),
                  Observe(extractors={name: verbatim for name in self.tools.names})],
            repeat_from=0,
            max_iterations=self.max_iterations,
        )

    def respond(self, messages, system=None):
        """Run the loop over a conversation that already exists, and return the reply."""
        conversation = Conversation(system=system or self.persona)
        conversation.messages.extend(messages)
        self.runtime().run(conversation)
        return conversation.messages[-1]["content"]

    def run(self, task):
        return self.respond([{"role": "user", "content": task}])


def as_tool(agent, log=None):
    """An agent, callable by another agent as an ordinary tool.

    The tool's description is the agent's `description`, not its persona. That is the whole of
    what a supervisor knows about who to ask, so it decides routing. The one argument says what
    chapter 2 learned about branch goals: the agent being asked sees nothing but this.

    `log`, if given, gets one entry per call -- who was asked what, and what came back -- which
    is otherwise buried in the supervisor's conversation.
    """
    def ask(request: Annotated[str, "Everything this specialist needs to know to do the job. "
                                    "They see nothing else: not the original message, not other replies."]) -> str:
        reply = agent.run(request)
        if log is not None:
            log.append({"agent": agent.name, "request": request, "reply": reply})
        return reply

    return Tool(ask, name=f"ask_{agent.name}", description=agent.description)


def supervisor(persona, roster, log=None, **kwargs):
    """An Agent whose tools are the agents it may call."""
    return Agent("supervisor", persona, tools=Registry([as_tool(a, log) for a in roster]), **kwargs)


class Chat:
    """A conversation with one person, owned by one agent at a time.

    Delegation keeps one agent in charge: it asks another and relays the answer, and every turn
    comes back through it. A handoff moves the conversation itself. The agent that hands over
    steps out, and the one it hands to talks to the person directly from then on, seeing the
    whole conversation so far.

    `handoffs` says who may hand to whom: {"triage": [support, pr]}. Each target becomes a
    `transfer_to_<name>` tool for that agent. Calling it stops the agent's loop on the spot --
    through Runtime's `cancel`, the same hook chapter 2 uses to stop an abandoned branch -- and
    the new owner answers the same turn, so the person never waits for a second message.
    """

    def __init__(self, agent, handoffs=None):
        self.active = agent
        self.handoffs = handoffs or {}
        self.transcript = []      # what the person saw: their turns, and who answered each one
        self.transfers = []       # every handoff, in order
        self._note = None         # what the last agent told the one taking over
        self._pending = None

    def _transfer_tool(self, target):
        def transfer(reason: Annotated[str, "Why you are handing over, and anything the next agent "
                                            "must know that is not already in the conversation."]) -> str:
            self._pending = (target, reason)
            return f"Handing the conversation to {target.name}."

        return Tool(transfer, name=f"transfer_to_{target.name}",
                    description=f"Hand this conversation over to {target.name} and step out. {target.description} "
                                f"They will see the whole conversation and answer the customer directly.")

    def send(self, text):
        self.transcript.append({"role": "user", "content": text})
        return self._answer(hops=0)

    def _answer(self, hops):
        agent = self.active
        targets = self.handoffs.get(agent.name, [])
        worker = replace(agent, tools=Registry([*agent.tools, *(self._transfer_tool(t) for t in targets)]))
        system = agent.persona
        if self._note:
            system += f"\n\n{self._note}"

        conversation = Conversation(system=system)
        conversation.messages.extend({"role": m["role"], "content": m["content"]} for m in self.transcript)
        runtime = worker.runtime()
        runtime.cancel = lambda: self._pending is not None
        runtime.run(conversation)

        if self._pending is not None and hops < 3:
            target, reason = self._pending
            self._pending = None
            self.transfers.append({"from": agent.name, "to": target.name, "reason": reason})
            self._note = f"{agent.name} has just handed this conversation to you. Their note: {reason}"
            self.active = target
            return self._answer(hops + 1)

        reply = conversation.messages[-1]["content"]
        self.transcript.append({"role": "assistant", "content": reply, "agent": agent.name})
        return reply
