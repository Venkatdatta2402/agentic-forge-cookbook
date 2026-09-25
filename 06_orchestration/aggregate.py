"""Several agents answer the same question. Something has to turn that into one answer.

Chapter 2 already combined answers from *identical* branches, where counting them makes sense:
ask the same agent five times and the majority is the model's own consensus. Here the answers
come from agents that know different things, and counting them is the wrong move -- a majority
is only meaningful when the voters are interchangeable, and the whole point of a roster is that
they are not.

What is left is deciding whose answer counts on which question. That is a decision made when
the team is designed, not by the agents arguing, so it lives here as plain functions over a
handful of opinions.
"""

import concurrent.futures
from typing import Literal

from pydantic import BaseModel, Field

from messages import answer


class Opinion(BaseModel):
    """One agent's position on a yes/no decision, and what it is standing on."""

    position: Literal["yes", "no", "unsure"] = Field(
        description="Your answer to the question as asked: yes, no, or unsure.")
    because: str = Field(description="Your reason, in one or two sentences.")
    evidence: str = Field(
        description="What you actually checked: the tool you called and what it returned, or the "
                    "line of the brief you are relying on. Say 'none' if you are reasoning without either.")
    confident: bool = Field(description="True only if what you checked settles it.")


def opinions(agents, question, max_workers=4):
    """Ask every agent the same question, at the same time, each in its own context."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        got = pool.map(lambda agent: (agent.name, answer(agent, question, model=Opinion)), agents)
    return dict(got)


def tally(opinions):
    counts = {"yes": 0, "no": 0, "unsure": 0}
    for opinion in opinions.values():
        counts[opinion.position] += 1
    return counts


def majority(opinions):
    """The position most agents hold. Included to be measured against, not because it is right."""
    counts = tally(opinions)
    best = max(counts.values())
    winners = [position for position, count in counts.items() if count == best]
    return winners[0] if len(winners) == 1 else "unsure"


def weighted(opinions, weights):
    """The position with the most weight behind it, where you decided the weights.

    `weights` is per agent: who is worth listening to on *this* question. An engineer's reading
    of a cell lot should not be outvoted by two colleagues who did not look it up.
    """
    scores = {"yes": 0.0, "no": 0.0, "unsure": 0.0}
    for name, opinion in opinions.items():
        scores[opinion.position] += weights.get(name, 1)
    best = max(scores.values())
    winners = [position for position, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else "unsure"


def decisive(opinions, who):
    """`who` decides, if it is confident and checked something. Otherwise nobody does.

    The strongest form of "whose answer counts on which question": one agent's evidence settles
    it, and everyone else is advice. Returns the position and why it was taken.
    """
    opinion = opinions.get(who)
    if opinion is None:
        return "unsure", f"{who} was not asked"
    if not opinion.confident or opinion.evidence.strip().lower() in ("", "none"):
        return "unsure", f"{who} is not confident or checked nothing: {opinion.evidence}"
    return opinion.position, f"{who} checked: {opinion.evidence}"


def disagreement(opinions):
    """Who is on each side, for a reader who has to look at it rather than count it."""
    sides = {}
    for name, opinion in opinions.items():
        sides.setdefault(opinion.position, []).append(name)
    return sides


def render(opinions):
    return "\n\n".join(
        f"{name} says {o.position} ({'confident' if o.confident else 'not confident'})\n"
        f"  because: {o.because}\n  checked: {o.evidence}"
        for name, o in opinions.items())
