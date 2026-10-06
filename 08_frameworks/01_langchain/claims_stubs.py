"""Stand-ins for every model and embedding call, so both builds run offline.

    embed / WordEmbeddings   a deterministic bag-of-words embedding -- crude, but retrieval works
    Scripted                 a LangChain chat model that plays back each step's reply per question
    rewrite, lookups, respond   the same script, as the scratch build's three model-calling functions

The script is one table, SCRIPT, so both builds are given identical "model" behaviour and any
difference in what they return is a difference in the wiring.
"""

import hashlib
import json
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

import meridian as M

STOP = set("a an and are as at be by do for from i if in is it its my of on or so that the this to was we what with "
           "you your can how will much me".split())


def embed(texts, task=None, dim=256):
    vectors = []
    for text in texts:
        v = [0.0] * dim
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in STOP:
                v[int(hashlib.md5(word[:6].encode()).hexdigest(), 16) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vectors.append([x / norm for x in v])
    return vectors


class WordEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return embed(texts)

    def embed_query(self, text):
        return embed([text])[0]


Q = [message for _, _, message in M.QUESTIONS]
STANDALONE = {Q[2]: "How long do I have to send the police report requested for claim CLM-20931?"}


def _asked(**fields):
    return {"benefit": "none", **fields}


SCRIPT = {   # standalone question -> (what the lookups need, the reply)
    Q[0]: (_asked(benefit="baggage_item", amount=1800),
           {"covered": "yes", "answer": "Yes -- locked in the safe it was not unattended. You'll get $400.",
            "payout": 400.0, "clauses": ["5.1", "1.3"], "next_steps": ["Send the police report."]}),
    Q[1]: (_asked(claim_id="CLM-20931"),
           {"covered": "not_a_coverage_question", "answer": "CLM-20931 is on hold until we get the police report.",
            "payout": None, "clauses": [], "next_steps": ["Send the police report."]}),
    STANDALONE[Q[2]]: (_asked(claim_id="CLM-20931"),
                       {"covered": "not_a_coverage_question", "answer": "30 days from 29 September: by 29 October.",
                        "payout": None, "clauses": ["8.3"], "next_steps": []}),
    Q[3]: (_asked(),
           {"covered": "no", "answer": "Not without the Winter Sports add-on.", "payout": None,
            "clauses": ["4.3", "6.2"], "next_steps": ["Add Winter Sports before you travel."]}),
    Q[4]: (_asked(),
           {"covered": "yes", "answer": "Yes, your mother is a close relative.", "payout": None,
            "clauses": ["2.1", "1.1", "8.2"], "next_steps": ["Medical certificate", "Cancellation confirmation"]}),
    Q[5]: (_asked(benefit="travel_delay", hours=14),
           {"covered": "no", "answer": "This delay was already paid: $100 on 2 September (CLM-20877).",
            "payout": None, "clauses": ["3.2"], "next_steps": []}),
    Q[6]: (_asked(benefit="medical", amount=2300),
           {"covered": "yes", "answer": "Yes -- asthma is an accepted condition. $2,200 after the excess.",
            "payout": 2200.0, "clauses": ["4.1", "4.2"], "next_steps": ["Send the hospital invoice."]}),
    Q[7]: (_asked(),
           {"covered": "no", "answer": "Not covered: it was left unattended.", "payout": None,
            "clauses": ["5.2", "1.3"], "next_steps": []}),
}


class Scripted(BaseChatModel):
    """Plays back SCRIPT, recognising the step by its system prompt."""

    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        human = [m.content for m in messages if isinstance(m, HumanMessage)][-1]
        if system.startswith("Rewrite"):
            message = AIMessage(content=STANDALONE.get(human, human))
        elif system.startswith("From a travel-insurance"):
            message = AIMessage(content="", tool_calls=[{"name": "Lookups", "args": SCRIPT[human][0], "id": "l",
                                                         "type": "tool_call"}])
        else:
            reply = SCRIPT[human][1]
            if kwargs.get("tools"):        # with_structured_output on a non-OpenAI class is a forced tool call
                message = AIMessage(content="", tool_calls=[{"name": "Reply", "args": reply, "id": "r",
                                                             "type": "tool_call"}])
            elif M.FIELDS_MARKER in system:  # streaming: prose, then the fields after the marker
                message = AIMessage(content=prose_then_fields(reply))
            else:                          # JSON mode: the shape asked for in the prompt
                message = AIMessage(content=json.dumps(reply))
        return ChatResult(generations=[ChatGeneration(message=message)])


# the scratch build's three model-calling functions, from the same script

def rewrite(history, message):
    return STANDALONE.get(message, message)


def lookups(question):
    return M.Lookups(**SCRIPT[question][0])


def respond(question, clauses, facts, policy_number):
    return M.Reply(**SCRIPT[question][1])



def prose_then_fields(reply):
    fields = {k: v for k, v in reply.items() if k != "answer"}
    return f"{reply['answer']}\n{M.FIELDS_MARKER} {json.dumps(fields)}"


def chunks(question, clauses, facts, policy_number, size=12):
    """The scripted reply, prose then fields, cut into pieces the way a stream arrives."""
    text = prose_then_fields(SCRIPT[question][1])
    return [text[i:i + size] for i in range(0, len(text), size)]
