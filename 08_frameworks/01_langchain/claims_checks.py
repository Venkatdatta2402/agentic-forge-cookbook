"""What building the claims assistant found out about LangChain, each one reproducible.

    one_shot_tools        LIVE. The first design: the model picks its lookups in one tool-calling
                          turn. Counts which tools it actually calls.
    json_mode_streaming   LIVE. How many chunks the reply's content arrives in, with and without
                          `response_format` -- through LangChain and through the raw client.
    deprecations          offline. What LangChain 1.x says when you use its own memory classes.
    structured_output     offline. What `with_structured_output(method="json_schema")` becomes on
                          a model class that is not ChatOpenAI.
    out_of_scope          EMBEDDINGS ONLY. What each build retrieves for questions the policy does not cover.
    grade                 a run's replies against `meridian.EXPECTED`.
"""

import collections
import json
import warnings

import claims_chain as C
import meridian as M


def one_shot_tools(questions=(0, 5, 6), trials=5):
    """Ask the model to "call every tool you need at once" and count what it calls."""
    from langchain_core.prompts import ChatPromptTemplate
    pick = (ChatPromptTemplate.from_messages([("system", M.TOOLS_SYSTEM), ("human", "{question}")])
            | C.chat_model().bind_tools(C.TOOLS))
    found = {}
    for i in questions:
        _, policy, message = M.QUESTIONS[i]
        seen = collections.Counter()
        for _ in range(trials):
            calls = pick.invoke({"question": message, "policy_number": policy}).tool_calls
            seen[", ".join(sorted(c["name"] for c in calls)) or "(none)"] += 1
        found[f"Q{i + 1}: {message[:60]}..."] = dict(seen)
    return found


OUT_OF_SCOPE = ["Can I bring my dog on the plane?", "What's your head office's phone number?"]


def out_of_scope():
    """The clauses each build hands the model, for one covered question and two the policy says nothing about."""
    import claims_scratch as X
    chain, scratch = C.build_retriever(), X.ClauseIndex()
    asked = [M.QUESTIONS[0][2]] + OUT_OF_SCOPE
    return {question[:60]: {"LangChain retriever (k=4)": [d.metadata["clause"] for d in chain.invoke(question)],
                            f"scratch (k=4, floor {X.FLOOR})": [number for number, _ in scratch.search(question)]}
            for question in asked}


def json_mode_streaming():
    """Content chunks per response mode: plain, json_object, json_schema."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2] / "01_foundations"))
    from llm import client

    ask = "Reply with a JSON object with keys answer (two sentences about travel insurance) and covered (yes or no)."
    schema = {"type": "json_schema", "json_schema": {"name": "reply", "schema": M.Reply.model_json_schema()}}
    modes = {"plain": {}, "json_object": {"response_format": {"type": "json_object"}},
             "json_schema": {"response_format": schema}}
    model, found = C.chat_model(), {}
    for mode, extra in modes.items():
        bound = model.bind(**extra) if extra else model
        langchain = sum(1 for chunk in bound.stream([("user", ask)]) if chunk.content)
        raw = sum(1 for chunk in client.chat.completions.create(model=C.MODEL, stream=True,
                                                                messages=[{"role": "user", "content": ask}], **extra)
                  if chunk.choices and chunk.choices[0].delta.content)
        found[mode] = {"LangChain content chunks": langchain, "raw client content chunks": raw}
    return found


def deprecations():
    """Construct LangChain's two memory classes and report what they say."""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables import RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory

    said = {}
    for name, make in [("InMemoryChatMessageHistory", InMemoryChatMessageHistory),
                       ("RunnableWithMessageHistory",
                        lambda: RunnableWithMessageHistory(RunnableLambda(lambda x: x),
                                                           lambda session: InMemoryChatMessageHistory()))]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make()
        said[name] = next((str(w.message).split("See ")[0].strip() for w in caught
                           if "deprecated" in str(w.message).lower() and name in str(w.message)), "no warning")
    return said


def structured_output():
    """The same call on ChatOpenAI and on any other chat model class."""
    import claims_stubs as S
    openai_side = C.chat_model().with_structured_output(M.Reply, method="json_schema")
    other_side = S.Scripted().with_structured_output(M.Reply, method="json_schema")
    bound = lambda chain: chain.first.kwargs if hasattr(chain, "first") else {}
    return {"ChatOpenAI sends": sorted(k for k in bound(openai_side) if k in ("response_format", "tools", "tool_choice")),
            "any other chat model sends": sorted(k for k in bound(other_side) if k in ("response_format", "tools",
                                                                                        "tool_choice")),
            "and is told method=": "json_schema, which BaseChatModel.with_structured_output pops and ignores"}


def grade(turns):
    """Each reply against the answer key: covered right, payout right, cites what it should, cites nothing unseen."""
    rows = []
    for turn, want in zip(turns, M.EXPECTED):
        reply = turn["reply"]
        rows.append({"q": turn["message"][:48],
                     "covered": reply.covered, "covered ok": reply.covered == want["covered"],
                     "payout": reply.payout, "payout ok": reply.payout == want["payout"],
                     "missing clauses": sorted(want["clauses"] - set(reply.clauses)),
                     "cited unseen": turn["unsupported"]})
    return rows


def show_grades(rows):
    good = sum(r["covered ok"] and r["payout ok"] for r in rows)
    for i, r in enumerate(rows, 1):
        mark = "ok " if r["covered ok"] and r["payout ok"] else "BAD"
        print(f"Q{i} {mark} {r['covered']:24} payout={str(r['payout']):8} missing={r['missing clauses']} "
              f"unseen={r['cited unseen']}  {r['q']}")
    print(f"\n{good} of {len(rows)} right on coverage and payout")
    return good


def as_json(result):
    return json.dumps(result, indent=1, default=str)
