"""Offline tests for "ask the cookbook".   python test_ask.py   (costs nothing; no model, no embedding API)"""

import asyncio
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("GROQ_API_KEY", "stub")

import ask_checks as K  # noqa: E402
import ask_index as A  # noqa: E402
import ask_stubs as S  # noqa: E402
import cookbook as CB  # noqa: E402


def built():
    index = A.open_index(path=tempfile.mkdtemp(), embed=S.WordEmbedding())
    A.refresh(index, CB.documents(), path=None)
    return index


def test_the_answer_key_is_true():
    docs = {d.id_: d.text for d in CB.documents()}
    for question, files, word in CB.EVAL:
        assert files <= set(docs), (question, files - set(docs))
        assert any(CB.mentions(docs[f], word) for f in files), question
    assert not any("05_llamaindex" in f for f in docs)            # the assistant never reads its own key
    assert not any("07_langsmith" in f or "08_langfuse" in f for f in docs)      # nor 07's, which grades it


def test_index_build_and_refresh():
    index = built()
    assert len(index.ref_doc_info) == len(CB.documents())
    chunks = list(index.docstore.docs.values())
    assert all(len("".join(ch for ch in c.text if ch.isalnum())) >= 25 for c in chunks)   # no empty chunks
    assert {c.metadata["kind"] for c in chunks} == {"code", "notebook", "readme"}
    again = A.refresh(index, CB.documents(), path=None)
    assert again["changed"] == [] and again["removed"] == []      # nothing changed, nothing re-embedded


def test_engine_and_floor():
    index = built()
    answered = A.ask(A.engine(index, S.answering(), floor=None), CB.EVAL[0][0])
    assert not answered["refused"] and answered["sources"]
    turned_away = A.ask(A.engine(index, S.answering(), floor=1.01), CB.EVAL[0][0])
    assert turned_away["refused"] and turned_away["answer"] == A.NOT_COVERED and not turned_away["sources"]


def test_agent_tools():
    index = built()
    out = asyncio.run(A.run_agent(A.agent(index, S.scripted()), "Why does Recall apply its floor first?"))
    assert [c["tool"] for c in out["calls"]] == ["search_cookbook", "open_file"]
    assert "04_memory/recall.py" in out["answer"]
    open_file = next(t for t in A.tools(index, S.answering()) if t.metadata.name == "open_file")
    assert str(open_file.call(path="04_memory/recall.py", line_start=3, line_end=5)).startswith(
        "[04_memory/recall.py, lines 3-5]\n   3  ")                                # what it read, named
    assert "Not a cookbook file" in str(open_file.call(path="../.env"))           # nothing outside the index
    assert "Not a cookbook file" in str(open_file.call(path=".env"))
    search = next(t for t in A.tools(index, S.answering(), floor=None) if t.metadata.name == "search_cookbook")
    assert "already made exactly this call" not in str(search.call(query="similarity floor"))
    assert "already made exactly this call" in str(search.call(query="  Similarity   FLOOR "))     # the same, again


def test_a_stopped_agent_keeps_its_record():
    # the model is cut off after the agent's first search, as a spent budget cuts it off live
    index = built()
    llm = S.scripted()
    respond = llm._response_generator

    def cut_off(messages, **kwargs):
        if any(m.role.value == "tool" for m in messages):
            raise A.BudgetExceeded("token budget of 1 reached (stand-in)")
        return respond(messages, **kwargs)

    llm._response_generator = cut_off
    out = asyncio.run(A.run_agent(A.agent(index, llm), "Why does Recall apply its floor first?"))
    assert out["answer"] is None and "BudgetExceeded" in out["stopped"]
    assert [c["tool"] for c in out["calls"]] == ["search_cookbook"]


def test_the_scratch_agent_keeps_each_call_in_budget():
    import agent_scratch as X
    import llm
    import tiktoken
    encode = tiktoken.get_encoding("cl100k_base").encode
    index = built()
    steps = [("tool", "search_cookbook", {"query": q}) for q in ("shared board", "memory stores", "Records count",
                                                                 "Journal resemblance", "on_board view", "Denied write",
                                                                 "KeyValue", "Graph walk")] + [("answer", "done [x]")]
    grown = {}
    for tokens in (1_000_000, 5000, 1500):
        seen, real = [], llm.client.chat.completions.create
        llm.client.chat.completions.create = S.scripted_client(steps, seen)
        try:
            out = X.run(index, CB.CROSS_CHAPTER[0], A.Budget(10 ** 9, "test"), context_tokens=tokens,
                        max_tool_calls=None)
        finally:
            llm.client.chat.completions.create = real
        grown[tokens] = (max(sum(len(encode(str(m.get("content") or ""))) for m in sent) for sent in seen), out)
    assert grown[5000][0] < grown[1_000_000][0] and grown[5000][0] < 5600      # the budget holds each call down
    assert grown[5000][1]["answer"] == "done [x]" and len(grown[5000][1]["calls"]) == 8
    assert grown[1500][1]["answer"] is None and grown[1500][1]["stopped"].startswith("selection refused")


def test_both_agents_stop_at_eight_tool_calls_and_answer():
    import agent_scratch as X
    import llm
    from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
    from llama_index.core.llms import MockFunctionCallingLLM
    index = built()
    topics = ["shared board owner", "memory stores sqlite", "doom loop runtime", "tool registry names",
              "context compaction pinned", "MCP server environment", "CrewAI pin chromadb", "AutoGen termination",
              "LangGraph interrupt resume", "reciprocal rank fusion"]
    real = llm.client.chat.completions.create
    llm.client.chat.completions.create = S.scripted_client([("tool", "search_cookbook", {"query": t}) for t in topics])
    try:
        scratch = X.run(index, CB.CROSS_CHAPTER[1], A.Budget(10 ** 9, "test"))
    finally:
        llm.client.chat.completions.create = real
    assert len(scratch["calls"]) == A.MAX_TOOL_CALLS and "maximum of 8 tool calls" in scratch["stopped"]
    assert scratch["answer"].startswith("best answer")                  # chapter 2's graceful finish, through Focus

    offered = []

    def respond(messages, **kwargs):
        offered.append(bool(kwargs.get("tools")))
        done = sum(m.role == MessageRole.TOOL for m in messages)
        if not kwargs.get("tools"):
            return ChatMessage(role=MessageRole.ASSISTANT, content="best answer from what was found")
        return ChatMessage(role=MessageRole.ASSISTANT, blocks=[ToolCallBlock(
            tool_call_id=f"c{done}", tool_name="search_cookbook", tool_kwargs={"query": topics[done]})])

    llm_ = MockFunctionCallingLLM(response_generator=respond, is_chat_model=True)
    llama = asyncio.run(A.run_agent(A.agent(index, llm_), CB.CROSS_CHAPTER[1]))
    assert len(llama["calls"]) == A.MAX_TOOL_CALLS and offered[-1] is False      # the last call: no tools offered
    assert llama["answer"] == "best answer from what was found"


def test_search_says_where_each_passage_is():
    index = built()
    search = A.tool_functions(index, S.answering())[0]
    out = search("doom loop identical tool call")
    assert ", lines " in out and "Nothing in the cookbook" not in out


def test_retrieval_scores_run():
    index = built()
    rows = asyncio.run(A.retrieval_scores(index, A.CookbookRetriever(index, S.answering(), floor=None)))
    assert len(rows) == len(CB.EVAL) and all({"hit_rate", "mrr"} <= set(r) for r in rows)


def test_the_live_model_is_metered():
    # the real client, built but never let through: at its budget, a call is refused before it is sent
    llm = A.chat_model(budget=A.Budget(0, "test"))
    assert llm.budget.limit == 0
    try:
        llm.complete("hello")
        assert False, "a call went out past the budget"
    except A.BudgetExceeded:
        pass
    agent_llm = A.chat_model(budget=A.Budget(0, "test"))
    try:
        asyncio.run(agent_llm.acomplete("hello"))
        assert False, "an async call went out past the budget"
    except A.BudgetExceeded:
        pass


def test_an_unparsed_reply_is_asked_again():
    # Groq's 400 for a reply it cannot parse, then a good reply: sent again, and counted. Any other 400 raises at once
    import httpx
    import openai

    def through(script):
        sent = []

        def reply(request):
            sent.append(1)
            status, code = script[min(len(sent), len(script)) - 1]
            if status == 200:
                return httpx.Response(200, json={
                    "id": "x", "object": "chat.completion", "created": 0, "model": A.MODEL,
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
            return httpx.Response(400, json={"error": {"message": "Parsing failed.", "type": "invalid_request_error",
                                                       "code": code}})

        def wrap(client):
            kind = httpx.AsyncClient if isinstance(client, openai.AsyncOpenAI) else httpx.Client
            return client.with_options(http_client=kind(transport=httpx.MockTransport(reply)))
        return wrap, sent

    wrap, sent = through([(400, "output_parse_failed"), (200, None)])
    llm = A.chat_model(budget=A.Budget(10 ** 6), wrap=wrap)
    assert llm.complete("hi").text == "ok" and len(sent) == 2 and llm.budget.retried == 1 and llm.budget.calls == 1
    wrap, sent = through([(400, "invalid_request"), (200, None)])
    try:
        A.chat_model(budget=A.Budget(10 ** 6), wrap=wrap).complete("hi")
        assert False, "a real 400 was retried"
    except openai.BadRequestError:
        assert len(sent) == 1
    wrap, sent = through([(400, "tool_use_failed")])
    try:
        asyncio.run(A.chat_model(budget=A.Budget(10 ** 6), wrap=wrap).acomplete("hi"))
        assert False, "retried for ever"
    except openai.BadRequestError:
        assert len(sent) == 1 + A.PARSE_RETRIES


def test_no_agent_request_outgrows_its_limit():
    # the real metered model and LlamaIndex's agent, against a scripted Groq that searches ten times: without the cut
    # the requests grow past 8,000 tokens; with it, none carries more than AGENT_CONTEXT_TOKENS of messages, every
    # tool result keeps its call id, and the answer it is made to give at the step limit is cut too
    import json as js

    import httpx
    import openai
    index = built()
    topics = ["shared board owner", "memory stores sqlite", "doom loop runtime", "tool registry names",
              "context compaction pinned", "MCP server environment", "CrewAI pin chromadb", "AutoGen termination",
              "LangGraph interrupt resume", "reciprocal rank fusion"]
    sent = []

    def reply(request):
        body = js.loads(request.content)
        sent.append(body)
        done = sum(m["role"] == "tool" for m in body["messages"])
        message = {"role": "assistant", "content": "the answer"}
        if body.get("tools") and done < len(topics):
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{done}", "type": "function",
                "function": {"name": "search_cookbook", "arguments": js.dumps({"query": topics[done]})}}]}
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": A.MODEL,
                                         "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    def wrap(client):
        kind = httpx.AsyncClient if isinstance(client, openai.AsyncOpenAI) else httpx.Client
        return client.with_options(http_client=kind(transport=httpx.MockTransport(reply)))

    def tokens(body):
        from llama_index.core.base.llms.types import ChatMessage
        return sum(A._tokens(ChatMessage(role="user", content=m.get("content") or "")) for m in body["messages"])

    out = asyncio.run(A.run_agent(A.agent(index, A.chat_model(budget=A.Budget(10 ** 9), wrap=wrap)),
                                  CB.CROSS_CHAPTER[1]))
    assert out["answer"] == "the answer" and len(out["calls"]) == A.MAX_TOOL_CALLS
    assert max(tokens(b) for b in sent) <= A.AGENT_CONTEXT_TOKENS
    # the answer it is made to give at the step limit: no tools offered, and everything the eight calls found in it
    # (LlamaIndex alone sends only the system prompt, the question and the stop notice -- see ask_index._finishing)
    # -- as plain text: no tool call or tool result in it for the model to carry on from
    last = sent[-1]
    assert not last.get("tools") and not any(m["role"] == "tool" or m.get("tool_calls") for m in last["messages"])
    found = [m["content"] for m in last["messages"] if "What your tool calls found" in (m.get("content") or "")]
    assert len(found) == 1 and "cut to their labels" in found[0] and "\n[" in found[0]     # labelled, older ones cut
    assert tokens(last) <= A.AGENT_CONTEXT_TOKENS

    sent.clear()                                  # the same run, uncut: what the limit is for
    asyncio.run(A.run_agent(A.agent(index, A.chat_model(budget=A.Budget(10 ** 9), wrap=wrap), context_tokens=None),
                            CB.CROSS_CHAPTER[1]))
    assert max(tokens(b) for b in sent) > A.AGENT_CONTEXT_TOKENS


def test_a_step_limit_answer_that_calls_a_tool_is_asked_plainly():
    # the answer at the step limit is asked for with no tools, and the model calls one anyway; Groq refuses it
    # (tool_use_failed), the same on every try -- until it is told plainly that no tools are left
    import json as js

    import httpx
    import openai
    index = built()
    sent = []

    def reply(request):
        body = js.loads(request.content)
        sent.append(body)
        done = sum(m["role"] == "tool" for m in body["messages"])
        if body.get("tools"):
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{done}", "type": "function",
                "function": {"name": "search_cookbook", "arguments": js.dumps({"query": f"topic {done}"})}}]}
        elif not any(A.NO_TOOLS_NOW in (m.get("content") or "") for m in body["messages"]):
            return httpx.Response(400, json={"error": {"message": "Tool choice is none, but model called a tool",
                                                       "type": "invalid_request_error", "code": "tool_use_failed"}})
        else:
            message = {"role": "assistant", "content": "the answer, at last"}
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": A.MODEL,
                                         "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    def wrap(client):
        kind = httpx.AsyncClient if isinstance(client, openai.AsyncOpenAI) else httpx.Client
        return client.with_options(http_client=kind(transport=httpx.MockTransport(reply)))

    out = asyncio.run(A.run_agent(A.agent(index, A.chat_model(budget=A.Budget(10 ** 9), wrap=wrap)),
                                  CB.CROSS_CHAPTER[1]))
    assert out["answer"] == "the answer, at last" and out["stopped"] is None
    plain = [b for b in sent if not b.get("tools")]
    assert len(plain) == 1 + A.PARSE_RETRIES + 1                  # refused three times as asked; then answered
    assert any("What your tool calls found" in (m.get("content") or "") for m in plain[-1]["messages"])


def test_checks():
    assert K.default_models()["Settings.llm"].startswith("ValueError")
    splitter = K.code_splitter()
    assert '"""' in splitter["defaults (40 lines, 1,500 chars)"] and '"""' not in splitter["used (80 lines, 4,000 chars)"]
    loaded = K.loaded_splitter()
    assert loaded["loaded without"]["splits with"] == ["SentenceSplitter"]
    assert loaded["loaded without"]["chunks that start mid-definition"] > 0
    assert loaded["loaded with its splitter"]["chunks that start mid-definition"] == 0
    deleted = K.deleted_file()
    assert "06_orchestration/failures.py" in deleted["refresh_ref_docs"]["files still indexed"]
    assert "06_orchestration/failures.py" not in deleted["ask_index.refresh"]["files still indexed"]
    resumed = K.resumable()
    assert resumed["files saved when it stopped"] > 0 and resumed["files in the index"] == len(CB.documents())
    assert resumed["files the second run embedded"] == resumed["files in the index"] - resumed["files saved when it stopped"]
    assert K.fused_scores()["fused, after a 0.6 floor"] == 0
    sizes = list(asyncio.run(K.agent_memory_limit())["tokens sent on each call"].values())
    assert all(s == sizes[0] for s in sizes)                      # the memory limit changes nothing in one turn
    assert max(sizes[0]) < 6000                                   # cut results keep six searches well under 8,000
    cited = K.citation_chunks()
    assert cited["citation_chunk_size=512"]["numbered sources the model sees"] > cited["citation_chunk_size=512"]["passages retrieved"]


if __name__ == "__main__":
    for test in (test_the_answer_key_is_true, test_index_build_and_refresh, test_engine_and_floor, test_agent_tools,
                 test_a_stopped_agent_keeps_its_record,
                 test_the_scratch_agent_keeps_each_call_in_budget,
                 test_both_agents_stop_at_eight_tool_calls_and_answer, test_search_says_where_each_passage_is,
                 test_retrieval_scores_run, test_the_live_model_is_metered, test_an_unparsed_reply_is_asked_again,
                 test_no_agent_request_outgrows_its_limit, test_a_step_limit_answer_that_calls_a_tool_is_asked_plainly,
                 test_checks):
        test()
        print("ok ", test.__name__)
