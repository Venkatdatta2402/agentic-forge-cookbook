"""Meridian's claims assistant, from this repo's own parts.

The same four steps as `claims_chain.py` -- rewrite, gather, respond, check -- as plain functions
called in order. Nothing composes them but Python:

    rewrite      chapter 1's `chat()`
    retrieve     chapter 4's `embed()` and `cosine()`, over one vector per clause, with chapter 4's floor:
                 a question the policy does not cover gets no clauses rather than the four least bad
    look up      chapter 1's `extract()` for what the lookups need, then chapter 3 `Tool`s, called by code
    respond      chapter 1's `extract()` -- schema mode first, validation and retry behind it
    memory       chapter 1's `Conversation`, one per customer
    in parallel  a thread pool, as chapter 5's `gather()` does it

What LangChain gives the composed chain for free -- batch, streaming, fallbacks, callbacks -- is
written out below where it is needed, so it can be counted.
"""

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import meridian as M

_REPO = Path(__file__).resolve().parents[2]
for _chapter in ("01_foundations", "03_tools", "04_memory"):
    if str(_REPO / _chapter) not in sys.path:
        sys.path.append(str(_REPO / _chapter))

MODEL = "openai/gpt-oss-120b"
FALLBACK = "openai/gpt-oss-20b"


# --------------------------------------------------------------------------- retrieval

# Chapter 4's floor: below this raw similarity a clause is not about the question at all. Measured on
# this wording with gemini-embedding-001, not chosen: every clause in the eight questions' top 4 scores
# 0.616 or more, and two questions the policy does not cover ("Can I bring my dog on the plane?", the
# head office's phone number) top out at 0.576. A thin margin from two examples; and the same figure
# chapter 4 measured on its journal. It cannot separate a needed clause from an unneeded one -- those
# overlap, 0.643-0.718 against up to 0.704 -- only "something relevant" from "nothing".
FLOOR = 0.6


class ClauseIndex:
    """One vector per clause, searched by cosine similarity. What `InMemoryVectorStore` is, written out --
    plus the one thing a vector store does not do: say it has nothing.

    A floor belongs to an embedding model and a corpus, so it applies by default only to the live one;
    a stand-in embedding scores on its own scale and passes its own floor, or none.
    """

    def __init__(self, embed=None, floor=None):
        if embed is None:
            from stores import embed
            floor = FLOOR if floor is None else floor
        self.embed, self.floor = embed, floor
        self.clauses = [(number, f"{number} {section}: {text}") for number, (section, text) in M.WORDING.items()]
        self.vectors = self.embed([text for _, text in self.clauses], task="RETRIEVAL_DOCUMENT")

    def search(self, query, k=4):
        from stores import cosine
        q = self.embed([query], task="RETRIEVAL_QUERY")[0]
        ranked = sorted(((cosine(q, v), clause) for v, clause in zip(self.vectors, self.clauses)), reverse=True)
        # the floor runs on RAW similarity, before anything else, as chapter 4's `Recall` does
        return [clause for score, clause in ranked[:k] if self.floor is None or score >= self.floor]


# --------------------------------------------------------------------------- the steps

def live_rewrite(history, message):
    from llm import chat
    return chat([{"role": "system", "content": M.REWRITE_SYSTEM}, *history,
                 {"role": "user", "content": message}], model=MODEL)


def live_lookups(question):
    """What the account lookups need, extracted from the question by chapter 1's `extract()`."""
    from llm import extract
    try:
        return extract(f"{M.LOOKUPS_SYSTEM}\n\nThe customer asks: {question}", M.Lookups, model=MODEL)
    except ValueError:                  # extraction failed: the schedule and claims still go in
        return M.Lookups(benefit="none")


def run_lookups(policy_number, asked):
    """The account facts, by code, each call through chapter 3's argument check."""
    from registry import Registry
    from tools import Tool
    registry = Registry([Tool(f) for f in M.TOOLS])
    calls = [("lookup_policy", {"policy_number": policy_number}), ("claims_for", {"policy_number": policy_number})]
    if asked.claim_id:
        calls.append(("claim_status", {"claim_id": asked.claim_id}))
    if asked.benefit != "none":
        calls.append(("estimate_payout", {"policy_number": policy_number, "benefit": asked.benefit,
                                          "amount": asked.amount, "hours": asked.hours}))
    return "\n".join(registry.functions[name](**registry.schemas[name].model_validate(args).model_dump())
                     for name, args in calls)


def answer_prompt(question, clauses, facts, policy_number):
    return (M.ANSWER_SYSTEM.format(clauses="\n".join(text for _, text in clauses), policy_number=policy_number,
                                   facts=facts) + f"\n\nThe customer asks: {question}")


def live_respond(question, clauses, facts, policy_number):
    from llm import extract
    prompt = answer_prompt(question, clauses, facts, policy_number)
    try:
        return extract(prompt, M.Reply, model=MODEL)
    except Exception:                  # noqa: BLE001 -- the fallback, written out
        return extract(prompt, M.Reply, model=FALLBACK)


def live_chunks(question, clauses, facts, policy_number):
    """The reply as the text chunks the API streams it in: prose first, then the fields after a marker."""
    import json
    from llm import client
    prompt = answer_prompt(question, clauses, facts, policy_number) + M.PROSE_THEN_FIELDS.format(
        schema=json.dumps(M.fields_schema()))
    stream = client.chat.completions.create(model=MODEL, stream=True,
                                            messages=[{"role": "user", "content": prompt}])
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def prose_then_fields(chunks):
    """Text chunks -> the answer so far, then the whole text. What the chain's `RunnableGenerator` wraps."""
    text, shown = "", 0
    for delta in chunks:
        text += delta
        prose = M.visible_prose(text)
        if len(prose) > shown:
            shown = len(prose)
            yield {"answer": prose}
    yield {"answer": text.partition(M.FIELDS_MARKER)[0].rstrip(), "finished": text}


# --------------------------------------------------------------------------- the assistant

class Assistant:
    def __init__(self, index=None, rewrite=None, lookups=None, respond=None):
        self.index = index or ClauseIndex()
        self.rewrite = rewrite or live_rewrite
        self.lookups = lookups or live_lookups
        self.respond = respond or live_respond
        self.conversations = {}
        self.steps = []                      # what the chain's callbacks report, recorded by hand

    def _timed(self, name, fn, *args):
        began = time.perf_counter()
        result = fn(*args)
        self.steps.append({"step": name, "seconds": round(time.perf_counter() - began, 2)})
        return result

    def _gather(self, session, policy_number, message):
        from chat import Conversation
        conversation = self.conversations.setdefault(session, Conversation())
        history = list(conversation.messages)
        question = self._timed("rewrite", self.rewrite, history, message) if history else message
        with concurrent.futures.ThreadPoolExecutor() as pool:     # retrieval does not wait for the tools
            clauses = pool.submit(self._timed, "retrieve", self.index.search, question)
            asked = pool.submit(self._timed, "look_up", self.lookups, question)
            clauses, asked = clauses.result(), asked.result()
        self._asked = asked                   # for `enforce`, which needs what was extracted
        return conversation, question, clauses, run_lookups(policy_number, asked)

    def ask(self, session, policy_number, message):
        conversation, question, clauses, facts = self._gather(session, policy_number, message)
        reply = self._timed("respond", self.respond, question, clauses, facts, policy_number)
        unsupported = sorted(set(reply.clauses) - {number for number, _ in clauses})
        reply, held = M.enforce(reply, policy_number, self._asked)
        conversation.messages += [{"role": "user", "content": message},
                                  {"role": "assistant", "content": reply.answer}]
        return {"message": message, "question": question, "clauses": clauses, "facts": facts,
                "reply": reply, "unsupported": unsupported, "held": held}

    def stream(self, session, policy_number, message, chunks=None):
        """Yield the answer as it is written, then the finished reply with its fields checked."""
        conversation, question, clauses, facts = self._gather(session, policy_number, message)
        finished = ""
        for partial in prose_then_fields((chunks or live_chunks)(question, clauses, facts, policy_number)):
            if "finished" in partial:
                finished = partial["finished"]
            else:
                yield partial
        try:
            reply = M.reply_from(finished)
        except ValueError:                   # nothing enforced the fields: the enforced call answers
            reply = self.respond(question, clauses, facts, policy_number)
        reply, _ = M.enforce(reply, policy_number, self._asked)
        conversation.messages += [{"role": "user", "content": message},
                                  {"role": "assistant", "content": reply.answer}]
        yield {"reply": reply}

    def batch(self, requests, max_workers=4):
        """Several customers at once -- what `chain.batch(max_concurrency=)` does."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda r: self.ask(*r), requests))
