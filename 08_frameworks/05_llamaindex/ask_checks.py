"""What building "ask the cookbook" found out about LlamaIndex, each one reproduced.

    default_models      what LlamaIndex reaches for when no model is configured
    code_splitter       CodeSplitter's defaults on one of this repo's modules, and the settings used instead
    loaded_splitter     a saved index loaded without its splitter: what a changed file is then split with
    deleted_file        `refresh_ref_docs` after a file is deleted, against `ask_index.refresh`
    resumable           a first build stopped half-way by its embedding provider, and resumed
    fused_scores        what a similarity floor sees after reciprocal-rank fusion
    citation_chunks     how many numbered sources the citation engine makes from the same five passages
    agent_memory_limit  how big the agent's requests grow over six searches, under three memory limits
    token_counts        LIVE. LlamaIndex's own token counter against what the API billed

All but the last run offline, on the real corpus, with `ask_stubs.WordEmbedding`.
"""

import os
import tempfile

import ask_index as A
import ask_stubs as S
import cookbook as CB


def _index(files=None, embed=None, path=None):
    docs = [d for d in CB.documents() if files is None or d.id_ in files]
    index = A.open_index(path=path or tempfile.mkdtemp(), embed=embed or S.WordEmbedding())
    A.refresh(index, docs, path=path)
    return index, docs


def default_models():
    """Settings.llm and Settings.embed_model, never set, with no OpenAI key in the environment."""
    from llama_index.core import Settings
    hidden = os.environ.pop("OPENAI_API_KEY", None)
    found = {}
    try:
        for name in ("llm", "embed_model"):
            try:
                found[f"Settings.{name}"] = type(getattr(Settings, name)).__name__
            except Exception as error:     # noqa: BLE001 -- what it raises is the finding
                said = next((l.strip() for l in str(error).splitlines() if l.strip().strip("*")), "")
                found[f"Settings.{name}"] = f"{type(error).__name__}: {said[:150]}"
    finally:
        if hidden is not None:
            os.environ["OPENAI_API_KEY"] = hidden
    return found


def code_splitter(file="06_orchestration/failures.py"):
    """One module through CodeSplitter: its defaults, and the settings `ask_index.split` uses."""
    from llama_index.core import Document
    from llama_index.core.node_parser import CodeSplitter
    doc = Document(text=(CB.REPO / file).read_text(encoding="utf-8"))
    first = lambda nodes: [n.text.strip().splitlines()[0][:40] if n.text.strip() else "" for n in nodes]
    default = CodeSplitter(language="python").get_nodes_from_documents([doc])
    used = CodeSplitter(language="python", chunk_lines=80, chunk_lines_overlap=10,
                        max_chars=4000).get_nodes_from_documents([doc])
    return {"file": file, "defaults (40 lines, 1,500 chars)": first(default),
            "used (80 lines, 4,000 chars)": first(used)}


def loaded_splitter(file="02_agent_runtime/runtime.py"):
    """Save an index, load it back with and without its splitter, change a module, refresh."""
    from llama_index.core import StorageContext, load_index_from_storage
    path = tempfile.mkdtemp()
    index, docs = _index({file}, path=path)
    changed = [d for d in docs if d.id_ == file][0]
    changed.set_content(changed.text + "\n\n# edited\n")
    seen = {}
    for how, kwargs in [("loaded with its splitter", {"transformations": [A.split()]}),
                        ("loaded without", {})]:
        loaded = load_index_from_storage(StorageContext.from_defaults(persist_dir=path),
                                         embed_model=S.WordEmbedding(), **kwargs)
        loaded.refresh_ref_docs([changed])
        chunks = [loaded.docstore.get_node(i) for i in loaded.ref_doc_info[file].node_ids]
        seen[how] = {"splits with": [type(t).__name__ for t in loaded._transformations],
                     "chunks": len(chunks),
                     "chunks that start mid-definition": sum(
                         1 for c in chunks if not c.text.lstrip().startswith(('"""', "def ", "class ", "from ", "import ", "#", "@")))}
    return seen


def deleted_file(gone="06_orchestration/failures.py", question="Why does attempt not retry a worker by default?"):
    """Three modules indexed; one is deleted; the index refreshed both ways; then asked about the deleted one."""
    files = {gone, "06_orchestration/state.py", "06_orchestration/messages.py"}
    found = {}
    for how in ("refresh_ref_docs", "ask_index.refresh"):
        index, docs = _index(files)
        remaining = [d for d in docs if d.id_ != gone]
        if how == "refresh_ref_docs":
            index.refresh_ref_docs(remaining)
        else:
            A.refresh(index, remaining, path=None)
        hits = index.as_retriever(similarity_top_k=3).retrieve(question)
        found[how] = {"files still indexed": sorted(index.ref_doc_info),
                      "the question retrieves": [n.node.metadata["file"] for n in hits]}
    return found


def resumable(stop_after=300):
    """A first build whose embedding provider fails after `stop_after` texts, then the same build run again."""
    class Fails(S.WordEmbedding):
        def _get_text_embeddings(self, texts):
            if self.texts + len(texts) > stop_after:
                raise RuntimeError("429 RESOURCE_EXHAUSTED (stand-in)")
            return super()._get_text_embeddings(texts)

    path, docs = tempfile.mkdtemp(), CB.documents()
    first = Fails()
    try:
        A.refresh(A.open_index(path=path, embed=first), docs, path=path, log=lambda s: None)
        stopped = None
    except RuntimeError as error:
        stopped = str(error)
    again = S.WordEmbedding()
    index = A.open_index(path=path, embed=again)
    saved = len(index.ref_doc_info)
    out = A.refresh(index, docs, path=path, log=lambda s: None)
    return {"first build stopped with": stopped, "texts embedded before it stopped": first.texts,
            "files saved when it stopped": saved, "files the second run embedded": len(out["changed"]),
            "texts the second run embedded": again.texts, "files in the index": out["files"]}


def fused_scores(question="Why does Recall apply its similarity floor before rescaling the scores?"):
    """The same question's scores from the vector search and after fusion, and a 0.6 floor applied to each."""
    from llama_index.core.postprocessor import SimilarityPostprocessor
    index, _ = _index()
    retriever = A.CookbookRetriever(index, S.answering(), floor=None)
    vector = retriever.vector.retrieve(question)
    fused = retriever.retrieve(question)
    floor = SimilarityPostprocessor(similarity_cutoff=0.6)
    return {"vector search scores": [round(n.score, 3) for n in vector[:5]],
            "fused scores": [round(n.score, 4) for n in fused],
            "fused, after a 0.6 floor": len(floor.postprocess_nodes(fused)),
            "fused files": [n.node.metadata["file"] for n in fused]}


def citation_chunks(question="How does the agent runtime stop an agent that keeps making the same tool call?"):
    """Five retrieved passages through CitationQueryEngine at its default citation size and at the one used."""
    from llama_index.core.query_engine import CitationQueryEngine
    index, _ = _index()
    found = {}
    for size in (512, 1024):
        retriever = A.CookbookRetriever(index, S.answering(), floor=None)
        engine = CitationQueryEngine.from_args(index, retriever=retriever, llm=S.answering(),
                                               citation_chunk_size=size, citation_chunk_overlap=0)
        response = engine.query(question)
        found[f"citation_chunk_size={size}"] = {"passages retrieved": len(retriever.retrieve(question)),
                                               "numbered sources the model sees": len(response.source_nodes)}
    return found


async def agent_memory_limit(limits=(1_000_000, 6_000, 1_500), searches=6):
    """Tokens the agent sends on each call while it searches six times, under LlamaIndex memories of three sizes."""
    import tiktoken
    from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
    from llama_index.core.llms import MockFunctionCallingLLM
    from llama_index.core.memory import Memory
    encode = tiktoken.get_encoding("cl100k_base").encode
    index, _ = _index()
    queries = ["memory stores", "shared board owner", "Records count", "Journal resemblance", "on_board view",
               "Denied write"]
    sent = []

    def respond(messages, **kwargs):
        sent.append(sum(len(encode(str(m.content or ""))) for m in messages))
        done = sum(m.role == MessageRole.TOOL for m in messages)
        if done < searches:
            return ChatMessage(role=MessageRole.ASSISTANT, blocks=[ToolCallBlock(
                tool_call_id=f"c{len(sent)}", tool_name="search_cookbook", tool_kwargs={"query": queries[done]})])
        return ChatMessage(role=MessageRole.ASSISTANT, content="done")

    found = {}
    for limit in limits:
        sent.clear()
        llm = MockFunctionCallingLLM(response_generator=respond, is_chat_model=True)
        memory = Memory.from_defaults(session_id=f"limit{limit}", token_limit=limit)
        await A.run_agent(A.agent(index, llm, floor=None), "How is the board different from memory stores?",
                          max_iterations=20, memory=memory)
        found[f"Memory(token_limit={limit:,})"] = list(sent)
    return {"tokens sent on each call": found}


def token_counts(index, llm, question):
    """LIVE. One question through the engine, counted by LlamaIndex's `TokenCountingHandler` and by the API.

    The counter goes on `Settings.callback_manager`, before the engine is built: that is where the engine's
    events are sent. Attached to the model instead, after the model was made, it counts nothing -- 0 tokens,
    measured offline -- and says nothing about it.
    """
    import tiktoken
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
    counter = TokenCountingHandler(tokenizer=tiktoken.get_encoding("cl100k_base").encode)
    kept = Settings.callback_manager
    Settings.callback_manager = CallbackManager([counter])
    before = llm.budget.summary()
    try:
        A.ask(A.engine(index, llm), question)
    finally:
        Settings.callback_manager = kept
    after = llm.budget.summary()
    return {"LlamaIndex's count": {"prompt": counter.prompt_llm_token_count,
                                   "completion": counter.completion_llm_token_count,
                                   "total": counter.total_llm_token_count},
            "the API's usage": {k: after[k] - before[k] for k in ("calls", "prompt", "completion", "total")}}
