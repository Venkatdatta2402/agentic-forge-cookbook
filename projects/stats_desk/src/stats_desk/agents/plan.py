"""The lead's plan. How to plan -- explore the data as an analyst would,
root causes for a "why" -- is said in these fields' descriptions, read as the plan is written,
not in the lead's instructions: span tests of the same guidance in the instructions
had the lead skip planning and write its brief from the profiles in 5 of 6 runs; in the fields,
1 of 4. The lead plans the way an analyst would -- what
would answer the question, what it needs to know first, what it would check next depending on
that -- and hands the plan in with ask_analysts:

    approach   a sentence or two: how the tasks answer the question
    tasks      the tasks that need no other task's answer, in words, run at once, one analyst each
    later      the tasks that wait for an answer, in words, each saying which; given (as the
               findings then make them look) in a later ask_analysts call, or dropped

All in words. Tried and dropped the same day: tasks as structured fields (indicator, frequency,
a field per group with its published values listed, checked by code against MoSPI's published
combinations before any analyst started). In span tests gpt-oss-120b invented field names
("id", "action", "measure1") and Groq refused 5 of the calls against the schema (~4k tokens
each); when a call did fit, it wrote the groups in the sentence and left their fields empty, so
the check passed without checking anything. Text tasks were given sensibly 3 times of 3.
What still keeps tasks to what is published: the list of profiles says which fields MoSPI
publishes together, groups are matched leniently ("graduate" -> "7.graduate"), and a refused
analysis tells the analyst, and through a task that came back empty the lead, what is.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Plan(BaseModel):
    """ask_analysts' arguments."""
    model_config = ConfigDict(extra="ignore")      # a field the model adds is ignored, not refused
    approach: Annotated[str, Field(description=(
        "your plan in a sentence or two, as an analyst exploring the data would: where it is high or low and for "
        "whom, how it has moved, what moves with it; for a \"why\" question, where the problem concentrates first, "
        "then what goes along with it there. Prefer tasks that tell competing explanations apart"))]
    tasks: Annotated[list[str], Field(description=(
        "the tasks that need no other task's answer, at most 3, run now at the same time, one analyst each: each "
        "says what to find out, which indicator (unemployment, participation or worker population), and for "
        "whom, naming groups as the profiles write them"))]
    later: Annotated[list[str], Field(default_factory=list, description=(
        "what you would check next depending on what the tasks find, each saying which answer it waits for"))]


def later_text(later: list[str]) -> str:
    """The plan's waiting tasks, put back in front of the lead with the findings."""
    if not later:
        return ""
    return ("\n\nYour plan's later tasks:\n" + "\n".join(f"- {x}" for x in later)
            + "\nGive the ones still needed with ask_analysts, as these findings now make them look; drop those "
              "they make unnecessary. Or hand in the brief if the question is answered.")
