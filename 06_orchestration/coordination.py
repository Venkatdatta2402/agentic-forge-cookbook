"""Ways of putting agents together, on two axes rather than as a list of architectures.

Named `coordination` rather than `patterns` because `02_agent_runtime` already has a
`patterns.py`, and two modules with one name on the path is a bug waiting for whoever
imports them in the wrong order.

Three questions get answered separately, and a real system answers all three:

    organisation   who can talk to whom.  A supervisor over a roster (notebook 2); peers holding
                   each other directly (`peers`); everyone reading one board (`state.Shared`).
    control        who runs next.  A manager that picks (`GroupChat`); an event, when a field is
                   written (`Shared.watch`); or a fixed order (`pipeline`).
    execution      what shape the work moves in.  One after another (`pipeline`); all at once and
                   combined (`aggregate.opinions`); turn and turn about (`debate`).

Shared state answers only the first. It says where information lives and nothing about who acts
on it, which is why a blackboard always comes with a scheduler of some kind -- and why "group
chat with a manager" is not a third kind of architecture but a board plus one of these controls.
Debate is not a fourth: it is turn-taking, with the turns fixed instead of chosen.
"""

from dataclasses import replace

from pydantic import BaseModel, Field

from messages import answer, as_form_tool
from registry import Registry
from state import on_board


# ----------------------------------------------------------------- organisation
def peers(agent, others, log=None):
    """The same agent, holding its colleagues as tools it can call itself.

    This is the whole of peer-to-peer. Nothing above them decides anything: the agent that has
    the question asks the agent that can answer it, and the supervisor is simply absent.
    """
    return replace(agent, tools=Registry([*agent.tools, *(as_form_tool(other, log) for other in others)]))


# ----------------------------------------------------------------- execution shape
def pipeline(stages, task, board=None):
    """Each agent in turn, each one seeing what the one before it produced.

    The control here is the list: who runs next was decided when the list was written. Nothing
    is choosing at run time, which is the point -- a pipeline is the cheapest possible scheduler
    and the least able to react.
    """
    carried, done = task, []
    for agent in stages:
        working = on_board(agent, board) if board is not None else agent
        output = working.run(carried)
        done.append({"agent": agent.name, "output": output})
        carried = f"{task}\n\nWhat {agent.name} produced:\n{output}"
    return done


# ----------------------------------------------------------------- control: a manager
class Pick(BaseModel):
    """A manager's decision about who speaks next."""

    speaker: str = Field(description="The name of the agent who should speak next, exactly as "
                                     "listed. Use 'done' when the group has settled the question.")
    why: str = Field(description="One short sentence: what you need from them.")


class GroupChat:
    """Everyone sees one transcript; a manager decides who speaks next.

    The manager is an agent like any other, and its decision is a form (`Pick`) rather than
    prose, so the program can act on it without parsing anything. Pair it with a `Shared` board
    and this is the blackboard pattern: the board holds what is established, the transcript
    holds what was said, and the manager is the scheduler that a board on its own does not have.
    """

    def __init__(self, manager, agents, board=None, max_turns=6):
        self.manager = manager
        self.agents = {agent.name: agent for agent in agents}
        self.board = board
        self.max_turns = max_turns
        self.transcript = []      # {"speaker": name, "text": ...}
        self.picks = []           # every decision the manager made, with its reason

    def _render(self):
        if not self.transcript:
            return "Nobody has spoken yet."
        return "\n\n".join(f"{turn['speaker']}: {turn['text']}" for turn in self.transcript)

    def run(self, topic):
        for _ in range(self.max_turns):
            roster = "\n".join(f"- {name}: {agent.description}" for name, agent in self.agents.items())
            pick = answer(self.manager,
                          f"The question: {topic}\n\nWho is available:\n{roster}\n\n"
                          f"What has been said so far:\n{self._render()}\n\n"
                          "Pick who should speak next, or 'done' if the question has been settled.",
                          model=Pick)
            self.picks.append(pick)
            if pick.speaker not in self.agents:
                # 'done', or a name nobody has. Either way there is nobody to run, and inventing
                # a speaker would be worse than stopping.
                return self.transcript
            agent = self.agents[pick.speaker]
            working = on_board(agent, self.board) if self.board is not None else agent
            text = working.run(f"The question: {topic}\n\nWhat has been said so far:\n{self._render()}\n\n"
                               f"The manager has asked you to speak next: {pick.why}\n\n"
                               # Without this they hold a meeting: they answer from what they already
                               # know and leave their tools alone. A speaker in a group chat is still an
                               # agent doing its job, not a participant with an opinion.
                               "Use your tools to check anything you are not sure of before you answer, "
                               "and record anything the others will need on the board.")
            self.transcript.append({"speaker": agent.name, "text": text})
        return self.transcript


# ----------------------------------------------------------------- execution shape: debate
def debate(topic, sides, rounds=2, board=None):
    """Two agents argue the same question, each answering what the other just said.

    Turn-taking with the turns fixed in advance, so no manager is needed -- which is exactly what
    makes it a different pattern from `GroupChat` rather than a different architecture. What it
    buys over asking both in parallel is that each side gets to answer the other's reasons.
    """
    said = []
    for round_number in range(rounds):
        for agent in sides:
            others = "\n\n".join(f"{turn['agent']}: {turn['text']}" for turn in said) or "Nobody yet."
            working = on_board(agent, board) if board is not None else agent
            text = working.run(
                f"The question: {topic}\n\nWhat has been argued so far:\n{others}\n\n"
                + ("Make your case, in your own terms." if not said else
                   "Answer the points above and make your case. Say plainly where the other side is wrong, "
                   "and concede anything they have established."))
            said.append({"round": round_number + 1, "agent": agent.name, "text": text})
    return said
