"""Stand-ins for every model call, so both builds run offline.

    GOLD          the answers the policy pack supports, as forms
    ScriptedLLM   a CrewAI LLM that writes CrewAI's text tool format: a specialist reads one document
                  ("Action: read_document"), then gives its section as the Final Answer; the lead
                  gives the questionnaire. `drop_first=True` makes the lead leave out Q9 the first
                  time, so the guardrail has something to send back.
    answer        chapter 6's `answer()` stood in for, for the scratch build
"""

import json

from crewai import BaseLLM

import tallyfold as T

_TEXT = {
    "Q1": "No. We hold a SOC 2 Type I report (as of 2026-03-14), available under NDA. Our Type II observation "
          "period ends 2026-12-31; the report is expected in February 2027.",
    "Q2": "No. Tallyfold is not ISO 27001 certified and does not plan to be before 2028.",
    "Q3": "Partly. All US employees are background-checked at hire. Engineering contractors in Poland, engaged "
          "through Kodera Sp. z o.o., are not checked by Tallyfold; Kodera attests to its own screening.",
    "Q4": "Yes. AES-256 at rest (AWS KMS, keys rotated annually); TLS 1.2 or higher in transit.",
    "Q5": "RPO 24 hours, RTO 8 hours. Failover is manual.",
    "Q6": "No. An on-call engineer covers availability alerts 24/7; security alerts are triaged in US Eastern "
          "business hours, with on-call escalation for critical alerts only.",
    "Q7": "Northgate Security, March 2026: no critical or high findings; 2 medium, remediated by May 2026; "
          "1 low open (a verbose server banner on the marketing site), fix scheduled for Q4 2026.",
    "Q8": "No. Tallyfold is hosted in AWS us-east-1 with backups in us-west-2; there is no EU region. EU personal "
          "data is covered by the Standard Contractual Clauses in our DPA.",
    "Q9": "No, your data is never used to train AI models. Receipt images are sent to OpenAI's API for text "
          "extraction under a zero-data-retention agreement.",
}

GOLD = {qid: T.Answer(id=qid, status=key["status"], answer=_TEXT[qid], evidence=sorted(key["rests_on"]),
                      needs_review=False) for qid, key in T.KEY.items()}


def section(role):
    return T.Section(answers=[GOLD[q] for q, (who, _) in T.QUESTIONS.items() if who == role])


def questionnaire(drop=()):
    return T.Questionnaire(answers=[a for q, a in GOLD.items() if q not in drop],
                           notes_for_sales="Five answers will disappoint: no Type II report yet, no ISO 27001, "
                                           "contractors not checked by us, no 24/7 SOC, no EU hosting.")


FIRST_DOC = {"compliance": "certifications", "security": "infrastructure", "privacy": "privacy"}
ROLE_BY_TITLE = {title: role for role, (title, _, _) in T.PERSONAS.items()}


class ScriptedLLM(BaseLLM):
    """Writes what a sensible agent would, in CrewAI's Thought / Action / Final Answer format."""

    def __init__(self, drop_first=False):
        super().__init__(model="scripted")
        self.drop_first = drop_first
        self.calls = []

    def supports_function_calling(self):
        return False

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None):
        text = messages if isinstance(messages, str) else "\n".join(str(m.get("content", "")) for m in messages)
        title = getattr(from_agent, "role", None) or next((t for t in ROLE_BY_TITLE if f"You are {t}" in text), None)
        role = ROLE_BY_TITLE.get(title, "manager")
        self.calls.append(role)
        if role in T.ROLES:
            # not "Observation:" -- CrewAI's own prompt template contains that word before any tool has run
            if not any(f"[{sid}]" in text for sid in T.PACK[FIRST_DOC[role]]):
                return (f"Thought: I should read the {FIRST_DOC[role]} document first.\nAction: read_document\n"
                        f'Action Input: {{"name": "{FIRST_DOC[role]}"}}')
            return f"Thought: I now know the final answer\nFinal Answer: {section(role).model_dump_json()}"
        drop = ("Q9",) if self.drop_first and self.calls.count(role) == 1 else ()
        return f"Thought: I now know the final answer\nFinal Answer: {questionnaire(drop).model_dump_json()}"


def answer(agent, request, model):
    """The scratch build's stand-in: `agent` is the role's name."""
    return section(agent) if model is T.Section else questionnaire()
