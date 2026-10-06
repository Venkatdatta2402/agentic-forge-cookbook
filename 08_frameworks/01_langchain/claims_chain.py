"""Meridian's claims assistant, composed from LangChain's building blocks.

Not an agent. Every message goes through the same fixed sequence, and the model never decides
what runs next:

    rewrite      a follow-up ("how long do I have to send it?") -> a standalone question
    gather       two things at once: retrieve the policy clauses it is about, and look up the
                 account facts -- the model extracts what the lookups need, code calls the tools
    respond      clauses + facts -> a `Reply`, through structured output
    check        in code: every clause the reply cites must be one that was retrieved

Each step is a Runnable, and the whole thing is one Runnable built with `|`. That is what
LangChain is for: the composed chain gets `invoke`, `batch`, `stream`, retries, fallbacks and
callbacks without any of the steps implementing them.

Memory is kept outside the chain, by `Assistant`, as a plain list of messages per customer --
LangChain 1.x has deprecated both of its own classes for it (see `Assistant`).
"""

import json
import os
import sys
import time
from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableGenerator, RunnableLambda, RunnablePassthrough
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore

import meridian as M

_REPO = Path(__file__).resolve().parents[2]
load_dotenv(_REPO / ".env")
if str(_REPO / "04_memory") not in sys.path:
    sys.path.append(str(_REPO / "04_memory"))       # for chapter 4's Gemini `embed()`

MODEL = "openai/gpt-oss-120b"
FALLBACK = "openai/gpt-oss-20b"
GROQ = "https://api.groq.com/openai/v1"


def chat_model(name=MODEL, max_retries=8):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=name, base_url=GROQ, api_key=os.environ["GROQ_API_KEY"], temperature=0,
                      max_retries=max_retries)


# --------------------------------------------------------------------------- retrieval

class GeminiEmbeddings(Embeddings):
    """Chapter 4's `embed()` behind LangChain's `Embeddings` interface.

    Two methods are the whole contract, and the split between them is the same one chapter 4
    makes with Gemini's task types: a document going into an index and a query searching it are
    embedded differently. Anything implementing these two plugs into every LangChain vector store.
    """

    def embed_documents(self, texts):
        from stores import embed
        return embed(texts, task="RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        from stores import embed
        return embed([text], task="RETRIEVAL_QUERY")[0]


def clause_documents():
    """One Document per clause: the clause is the unit a reply cites, so it is the unit retrieved."""
    return [Document(page_content=f"{number} {section}: {text}", metadata={"clause": number, "section": section})
            for number, (section, text) in M.WORDING.items()]


def build_retriever(embeddings=None, k=4):
    store = InMemoryVectorStore(embeddings or GeminiEmbeddings())
    store.add_documents(clause_documents())
    return store.as_retriever(search_kwargs={"k": k}).with_config(run_name="retrieve")


def format_clauses(docs):
    return "\n".join(doc.page_content for doc in docs)


# --------------------------------------------------------------------------- tools

TOOLS = [StructuredTool.from_function(f) for f in M.TOOLS]
BY_NAME = {t.name: t for t in TOOLS}


def run_lookups(inputs):
    """The account facts, by code: the tools are Runnables, invoked with the fields the model extracted."""
    policy, asked = inputs["policy_number"], inputs["lookups"]
    calls = [("lookup_policy", {"policy_number": policy}), ("claims_for", {"policy_number": policy})]
    if asked.claim_id:
        calls.append(("claim_status", {"claim_id": asked.claim_id}))
    if asked.benefit != "none":
        calls.append(("estimate_payout", {"policy_number": policy, "benefit": asked.benefit,
                                          "amount": asked.amount, "hours": asked.hours}))
    return "\n".join(BY_NAME[name].invoke(args) for name, args in calls)


NOTHING_TO_CALCULATE = M.Lookups(benefit="none")


# --------------------------------------------------------------------------- the chain

def build_gather(model, retriever):
    """message + history -> question, the clauses it is about, and the account facts it needs."""
    rewrite = (ChatPromptTemplate.from_messages([("system", M.REWRITE_SYSTEM), MessagesPlaceholder("history"),
                                                 ("human", "{message}")])
               | model | StrOutputParser()).with_config(run_name="rewrite")
    # a first message has nothing to rewrite against, so it skips the model call
    standalone = RunnableBranch((lambda x: not x["history"], itemgetter("message")), rewrite)

    lookups = (ChatPromptTemplate.from_messages([("system", M.LOOKUPS_SYSTEM), ("human", "{question}")])
               | model.with_structured_output(M.Lookups, method="json_schema"))
    # if the extraction fails, the answer still gets the schedule and the claims -- just no figure
    lookups = lookups.with_fallbacks([RunnableLambda(lambda _: NOTHING_TO_CALCULATE)])
    # returns the extracted fields too, so `check` can apply the rules that need them
    look_up = (RunnablePassthrough.assign(lookups=lookups)
               | RunnablePassthrough.assign(facts=RunnableLambda(run_lookups))
               | RunnableLambda(lambda x: {"lookups": x["lookups"], "facts": x["facts"]})).with_config(run_name="look_up")

    return (RunnablePassthrough.assign(question=standalone)
            # two keys in one assign() run in parallel -- retrieval does not wait for the tools
            | RunnablePassthrough.assign(docs=itemgetter("question") | retriever, account=look_up)
            | RunnablePassthrough.assign(facts=lambda x: x["account"]["facts"], lookups=lambda x: x["account"]["lookups"],
                                         clauses=lambda x: format_clauses(x["docs"])))


ANSWER_PROMPT = ChatPromptTemplate.from_messages([("system", M.ANSWER_SYSTEM), ("human", "{question}")])


def _json_prompt():
    """The answer prompt with the reply's JSON shape spelled out, for the JSON-mode fallback."""
    import json
    schema = json.dumps(M.Reply.model_json_schema()).replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages([("system", M.ANSWER_SYSTEM + M.JSON_INSTRUCTION.replace("{schema}", schema)),
                                             ("human", "{question}")])


def build_respond(model, fallback=None):
    """clauses + facts -> a `Reply`, in the order chapter 1's `extract()` tries things.

    Schema mode first: Groq enforces the shape. When Groq cannot -- on the first live run it
    refused with `output_parse_failed` and an empty `failed_generation` -- JSON mode, with the
    shape in the prompt and pydantic checking it. Then the second model. Each is a Runnable, so
    the order is one `with_fallbacks` call.
    """
    tries = [_json_prompt() | model.with_structured_output(M.Reply, method="json_mode")]
    if fallback is not None:
        tries.append(ANSWER_PROMPT | fallback.with_structured_output(M.Reply, method="json_schema"))
    respond = (ANSWER_PROMPT | model.with_structured_output(M.Reply, method="json_schema")).with_fallbacks(tries)
    return respond.with_config(run_name="respond")


def _prose_then_fields(chunks):
    """Streamed chunks -> the customer's answer so far, then the whole text once it is finished.

    A plain generator, made a streaming step by `RunnableGenerator`. It yields `{"answer": ...}`
    as the prose grows and stops showing text at the FIELDS marker, which the customer never sees.
    """
    text, shown = "", 0
    for chunk in chunks:
        text += chunk.content
        prose = M.visible_prose(text)
        if len(prose) > shown:
            shown = len(prose)
            yield {"answer": prose}
    yield {"answer": text.partition(M.FIELDS_MARKER)[0].rstrip(), "finished": text}


def build_streaming(model):
    """The respond step, streamed: prose first, then the fields as JSON after a marker.

    Why not stream the structured reply: Groq delivers a JSON reply from gpt-oss in one piece, with
    `response_format` (measured, LangChain and raw client alike: claims_checks.json_mode_streaming)
    and, on this assistant's prompt, without it -- the first live run's "stream" arrived as a single
    368-character partial. Prose streams. `Assistant.stream` checks the fields when it is done.
    """
    schema = json.dumps(M.fields_schema()).replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", M.ANSWER_SYSTEM + M.PROSE_THEN_FIELDS.replace("{schema}", schema)), ("human", "{question}")])
    return prompt | model | RunnableGenerator(_prose_then_fields)


def check(turn):
    """The rules a reply must pass before anything acts on it. Code, not a prompt.

    Every clause it cites must have been in front of the model; and a payout must not repeat one
    already made (`meridian.enforce`).
    """
    retrieved = {doc.metadata["clause"] for doc in turn["docs"]}
    turn["unsupported"] = sorted(set(turn["reply"].clauses) - retrieved)
    turn["reply"], turn["held"] = M.enforce(turn["reply"], turn["policy_number"], turn.get("lookups"))
    return turn


def build_chain(model, retriever, fallback=None):
    """The whole assistant as one Runnable."""
    return (build_gather(model, retriever)
            | RunnablePassthrough.assign(reply=build_respond(model, fallback))
            | RunnableLambda(check).with_config(run_name="check"))


# --------------------------------------------------------------------------- memory, and the assistant

class Assistant:
    """The chain, plus each customer's conversation so far. The only state anywhere is here.

    A conversation is a plain list of messages, handed to the prompt's `MessagesPlaceholder`.
    LangChain 1.x deprecated both of its own ways of keeping one -- `RunnableWithMessageHistory`
    ("use LangGraph's built-in persistence instead") and, in 1.6.4, `InMemoryChatMessageHistory`
    itself. Memory across turns is LangGraph's job now; what LangChain keeps is the message types.
    """

    def __init__(self, model=None, retriever=None, fallback=None):
        self.model = model or chat_model()
        self.retriever = retriever or build_retriever()
        self.gather = build_gather(self.model, self.retriever)
        self.respond = build_respond(self.model, fallback)
        self.chain = build_chain(self.model, self.retriever, fallback)
        self.streaming = build_streaming(self.model)
        self.histories = {}

    def _inputs(self, session, policy_number, message):
        return {"message": message, "policy_number": policy_number,
                "history": list(self.histories.get(session, []))}

    def _remember(self, session, message, answer):
        self.histories.setdefault(session, []).extend([HumanMessage(message), AIMessage(answer)])

    def ask(self, session, policy_number, message, callbacks=None):
        turn = self.chain.invoke(self._inputs(session, policy_number, message),
                                 config={"callbacks": callbacks or [], "run_name": f"turn:{session}"})
        self._remember(session, message, turn["reply"].answer)
        return turn

    def batch(self, requests, max_concurrency=4, callbacks=None):
        """Several customers at once. `batch` comes with every Runnable; nothing here implements it.

        `return_exceptions=True`: one customer's failure comes back as that customer's result
        instead of failing the other two -- the first live run lost a whole batch to one error.
        """
        turns = self.chain.batch([self._inputs(*r) for r in requests], return_exceptions=True,
                                 config={"max_concurrency": max_concurrency, "callbacks": callbacks or []})
        for (session, _, message), turn in zip(requests, turns):
            if not isinstance(turn, Exception):
                self._remember(session, message, turn["reply"].answer)
        return turns

    def stream(self, session, policy_number, message):
        """Yield the answer as it is written, then the finished reply with its fields checked.

        If the fields do not parse or validate, the enforced (non-streaming) call answers instead:
        the customer has already read the prose, so the fallback only has to get the fields right.
        """
        gathered = self.gather.invoke(self._inputs(session, policy_number, message))
        finished = ""
        for partial in self.streaming.stream(gathered):
            if "finished" in partial:
                finished = partial["finished"]
            else:
                yield partial
        try:
            reply = M.reply_from(finished)
        except ValueError:
            reply = self.respond.invoke(gathered)
        reply, _ = M.enforce(reply, policy_number, gathered.get("lookups"))
        self._remember(session, message, reply.answer)
        yield {"reply": reply}


# --------------------------------------------------------------------------- callbacks

class Steps(BaseCallbackHandler):
    """Times every named step and counts the model's tokens, from LangChain's callback events.

    Nothing in the chain knows this exists. Callbacks are handed to `invoke` in its config and
    LangChain passes them down to every Runnable inside -- which is also how LangSmith traces.
    """

    NAMED = {"rewrite", "look_up", "retrieve", "respond", "check"}

    def __init__(self):
        self.rows, self._started = [], {}

    def _begin(self, name, run_id):
        if name in self.NAMED:
            self._started[run_id] = (name, time.perf_counter())

    def _end(self, run_id, detail=""):
        if run_id in self._started:
            name, began = self._started.pop(run_id)
            self.rows.append({"step": name, "seconds": round(time.perf_counter() - began, 2), "detail": detail})

    def on_chain_start(self, serialized, inputs, *, run_id, name=None, **kw):
        self._begin(name or kw.get("run_name"), run_id)

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._end(run_id)

    def on_retriever_start(self, serialized, query, *, run_id, name=None, **kw):
        self._begin(name or "retrieve", run_id)

    def on_retriever_end(self, documents, *, run_id, **kw):
        self._end(run_id, "clauses " + ", ".join(d.metadata["clause"] for d in documents))

    def on_llm_end(self, response, **kw):
        usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
        self.rows.append({"step": "  model call", "seconds": None,
                          "detail": f"{usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out"})
