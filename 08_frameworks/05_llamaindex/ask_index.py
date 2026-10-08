"""Ask the cookbook, with LlamaIndex: an index over this repo, a cited question-answering engine, and an agent.

    the models      `chat_model()` -- gpt-oss-120b on Groq through `OpenAILike`, its client metered and capped;
                    `embed_model()` -- Gemini's gemini-embedding-001 at 768 dimensions, as chapter 4 uses it
    splitting       `split()` -- prose by its markdown headings then to size; code by LlamaIndex's
                    `CodeSplitter`, which parses Python with tree-sitter and cuts between definitions
    the index       `open_index()` and `refresh()` -- a `VectorStoreIndex` saved to disk, brought up to date
                    file by file: changed files re-split and re-embedded, deleted ones removed
    retrieval       `CookbookRetriever` -- vector search and BM25 fused by reciprocal rank, behind a floor:
                    a question nothing in the cookbook is near gets no passages at all
    answering       `engine()` -- LlamaIndex's `CitationQueryEngine`, so every claim points at a numbered source
    the agent       `agent()` -- a `FunctionAgent` that can search more than once and open a file to read on
    evaluation      `retrieval_scores()` -- hit rate and MRR with LlamaIndex's `RetrieverEvaluator`

Nothing that calls a model is made at import time; every model is an argument, so the whole system runs
offline against stand-ins (`ask_stubs.py`) before any token is spent.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.retrievers import BaseRetriever

import cookbook as CB

load_dotenv(CB.REPO / ".env")

MODEL = "openai/gpt-oss-120b"
GROQ = "https://api.groq.com/openai/v1"
EMBED_MODEL, EMBED_DIM = "gemini-embedding-001", 768
STORE = Path(__file__).resolve().parent / "store"
INDEX_DIR = STORE / "index"
PROGRESS = STORE / "progress.log"


# --------------------------------------------------------------------------- the models

class BudgetExceeded(RuntimeError):
    """A run reached its token budget. Raised before the next model call."""


class Budget:
    """Every model call's billed tokens, from the API's own `usage`, and no call once the limit is reached."""

    def __init__(self, limit=60_000, label="?"):
        self.limit, self.label = limit, label
        self.calls = self.prompt = self.completion = 0
        self.largest = 0        # the biggest single request sent, in prompt tokens: Groq's free tier refuses >8,000
        self.per_call = []      # each call's tokens, in + out

    @property
    def spent(self):
        return self.prompt + self.completion

    def check(self):
        if self.spent >= self.limit:
            self._log("BUDGET REACHED -- stopping")
            raise BudgetExceeded(f"token budget of {self.limit:,} reached ({self.spent:,} spent)")

    def charge(self, response):
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.calls += 1
            self.prompt += usage.prompt_tokens
            self.largest = max(self.largest, usage.prompt_tokens)
            self.per_call.append(usage.prompt_tokens + usage.completion_tokens)
            self.completion += usage.completion_tokens
            self._log()

    def summary(self):
        return {"calls": self.calls, "prompt": self.prompt, "completion": self.completion, "total": self.spent,
                "largest request": self.largest, "largest call (in + out)": max(self.per_call, default=0),
                "per call": list(self.per_call)}

    def _log(self, note=""):
        STORE.mkdir(exist_ok=True)
        with open(PROGRESS, "a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%H:%M:%S')} {self.label} total={self.spent} {note}\n")


def chat_model(model=MODEL, budget=None, max_tokens=4096):
    """gpt-oss on Groq, through LlamaIndex's `OpenAILike`, with clients that count and cap what is spent.

    `OpenAILike` reuses the OpenAI clients it is handed, so the clients are where the budget lives:
    every call LlamaIndex makes -- an answer, an agent's turn -- goes through them.
    """
    import openai
    from llama_index.llms.openai_like import OpenAILike
    budget = budget or Budget()
    key = os.environ["GROQ_API_KEY"]
    sync = openai.OpenAI(api_key=key, base_url=GROQ, max_retries=8)
    aio = openai.AsyncOpenAI(api_key=key, base_url=GROQ, max_retries=8)
    real_sync, real_aio = sync.chat.completions.create, aio.chat.completions.create

    def create(*args, **kwargs):
        budget.check()
        response = real_sync(*args, **kwargs)
        budget.charge(response)
        return response

    async def acreate(*args, **kwargs):
        budget.check()
        response = await real_aio(*args, **kwargs)
        budget.charge(response)
        return response

    sync.chat.completions.create, aio.chat.completions.create = create, acreate
    llm = _metered()(model=model, api_base=GROQ, api_key=key, openai_client=sync, async_openai_client=aio,
                     is_chat_model=True, is_function_calling_model=True, context_window=131_072,
                     max_tokens=max_tokens, temperature=0, timeout=120)
    llm._budget = budget
    return llm


_METERED = None


def _metered():
    """`OpenAILike` with somewhere to keep its budget. LlamaIndex's models are Pydantic models and refuse an
    attribute they do not declare -- `llm.budget = ...` raises -- so the budget is a declared private one."""
    global _METERED
    if _METERED is None:
        from llama_index.llms.openai_like import OpenAILike
        from pydantic import PrivateAttr

        class MeteredOpenAILike(OpenAILike):
            _budget: object = PrivateAttr(default=None)

            @property
            def budget(self):
                return self._budget

        _METERED = MeteredOpenAILike
    return _METERED


def embed_model():
    """Gemini, as chapter 4 uses it. LlamaIndex's integration sends RETRIEVAL_DOCUMENT for what goes into
    the index and RETRIEVAL_QUERY for a question -- the distinction chapter 4 measured -- but its default
    model is a preview one, so the model and the size are pinned."""
    from google.genai import types
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    return GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=os.environ["GEMINI_API_KEY"],
                                embedding_config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
                                embed_batch_size=50, retries=8, retry_max_seconds=65)


# --------------------------------------------------------------------------- splitting

def split():
    """One transformation that splits prose and code differently -- one, so a saved index has one thing to
    re-run on a changed file.

    prose   `MarkdownNodeParser` cuts at headings and keeps the heading path as metadata; a section longer
            than 800 tokens is then cut by `SentenceSplitter`
    code    `CodeSplitter` parses the module with tree-sitter and cuts between definitions. Its defaults
            (40 lines, 1,500 characters) cut a module's docstring into fragments -- one of them just `\"\"\"`
            -- so a chunk may run to 80 lines and 4,000 characters
    """
    from llama_index.core.node_parser import CodeSplitter, MarkdownNodeParser, SentenceSplitter
    from llama_index.core.schema import TransformComponent

    prose_parser, cap = MarkdownNodeParser(), SentenceSplitter(chunk_size=800, chunk_overlap=80)
    code_parser = CodeSplitter(language="python", chunk_lines=80, chunk_lines_overlap=10, max_chars=4000)

    class Split(TransformComponent):
        def __call__(self, nodes, **kwargs):
            prose = [n for n in nodes if n.metadata.get("kind") != "code"]
            code = [n for n in nodes if n.metadata.get("kind") == "code"]
            out = joined(cap(prose_parser(prose))) + joined(code_parser(code))
            for node in out:
                # the file and the heading go into the vector: "chapter 4" in a question should find chapter 4
                node.excluded_embed_metadata_keys = ["kind"]
            return out

    return Split()


def joined(nodes, least=25):
    """A chunk with almost nothing in it, joined onto the next chunk of the same file.

    Both parsers make them on this repo: `CodeSplitter` leaves a class's header on its own ("class Runtime:"),
    so its methods lose the class's name; and a notebook heading whose section is code ("## Setup") is a
    section with no prose once the code cells are left out. Alone, each is a chunk that matches nothing
    and still takes a place in the index.
    """
    import re
    out, carry = [], None
    for node in nodes:
        if carry is not None and carry.ref_doc_id == node.ref_doc_id:
            node.set_content(carry.get_content() + "\n" + node.get_content())
            carry = None
        elif carry is not None:
            out.append(carry)
            carry = None
        if len(re.sub(r"\W", "", node.get_content())) < least:
            carry = node
        else:
            out.append(node)
    if carry is not None:
        out.append(carry)
    return out


# --------------------------------------------------------------------------- the index

def open_index(path=INDEX_DIR, embed=None):
    """The saved index, or an empty one. The splitter is passed back in on load: a loaded index otherwise
    re-splits a changed file with LlamaIndex's default splitter, not the one it was built with."""
    from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
    embed = embed or embed_model()
    if (Path(path) / "docstore.json").exists():
        storage = StorageContext.from_defaults(persist_dir=str(path))
        return load_index_from_storage(storage, embed_model=embed, transformations=[split()])
    return VectorStoreIndex([], embed_model=embed, transformations=[split()])


def refresh(index, documents, path=INDEX_DIR, every=10, log=print):
    """Bring the index up to date with the files, and save it as it goes.

    LlamaIndex's `refresh_ref_docs` re-splits and re-embeds a document whose hash has changed and inserts a
    new one. Two things it does not do are added here: it never removes a document that is no longer in
    the corpus, so a deleted file would go on being retrieved; and it saves nothing, so a first build
    stopped half-way -- by a rate limit, say -- would start again from nothing. Saving every `every` files
    makes the build resumable, and a resumed build skips every file already in.
    """
    changed = []
    for document in documents:
        if index.docstore.get_document_hash(document.id_) == document.hash:
            continue
        index.refresh_ref_docs([document])
        changed.append(document.id_)
        if path and len(changed) % every == 0:
            index.storage_context.persist(persist_dir=str(path))
            log(f"  {len(changed)} files in, latest {document.id_}")
    present = {d.id_ for d in documents}
    removed = sorted(ref for ref in index.ref_doc_info if ref not in present)
    for ref in removed:
        index.delete_ref_doc(ref, delete_from_docstore=True)
    if path:
        index.storage_context.persist(persist_dir=str(path))
    return {"changed": changed, "removed": removed, "files": len(index.ref_doc_info), "chunks": len(index.docstore.docs)}


# --------------------------------------------------------------------------- retrieval

# Measured with measure_floor() on the built index, not chosen: the 15 questions the cookbook answers score
# 0.702-0.787 on their best passage, the three it does not 0.539, 0.547 and 0.624, and the floor sits midway
# between 0.624 and 0.702. The 0.624 is "How do I fine-tune Llama 3 on my own data?": the cookbook talks about
# Llama models throughout, so the question is near in words, not in subject. From 18 questions; it moves with
# the corpus and the embedding model.
FLOOR = 0.66


class CookbookRetriever(BaseRetriever):
    """Vector search and BM25, fused by reciprocal rank -- behind a floor on the vector search.

    Fusion turns scores into ranks, so after it there is no similarity left to set a floor on: a question
    about the capital of Australia still gets five "best" passages. The floor is applied first, on the
    vector search's raw cosine, as chapter 4's `Recall` does: if nothing in the cookbook is near the
    question, no passages are returned and no answer is written. The question is embedded once; the vector
    retriever stores the embedding on the query, and the fused search reuses it.

    BM25 is built from the index's chunks when this is made, so make a new one after a refresh.
    """

    def __init__(self, index, llm, k=5, floor="default"):
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever
        self.vector = index.as_retriever(similarity_top_k=2 * k)
        self.bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=2 * k)
        # num_queries=1: fuse the two searches of the question as asked, without an LLM writing variants.
        # The LLM is still passed: QueryFusionRetriever otherwise reaches for LlamaIndex's global default
        self.fused = QueryFusionRetriever([self.vector, self.bm25], llm=llm, num_queries=1,
                                          mode="reciprocal_rerank", similarity_top_k=k, use_async=False)
        self.floor = FLOOR if floor == "default" else floor
        self.best = None
        super().__init__()

    def _retrieve(self, query_bundle):
        near = self.vector.retrieve(query_bundle)
        self.best = near[0].score if near else None
        if self.floor is not None and (self.best is None or self.best < self.floor):
            return []
        return self.fused.retrieve(query_bundle)


def measure_floor(index, in_scope=None, out_of_scope=None):
    """The best vector score for questions the cookbook answers, and for ones it does not."""
    vector = index.as_retriever(similarity_top_k=1)
    best = lambda q: vector.retrieve(q)[0].score
    return {"in scope": {q: round(best(q), 3) for q, _, _ in (in_scope or CB.EVAL)},
            "out of scope": {q: round(best(q), 3) for q in (out_of_scope or CB.OUT_OF_SCOPE)}}


def label(node):
    """Where a passage comes from, as a reader would look for it."""
    meta = node.metadata
    where = meta.get("header_path", "").strip("/").replace("/", " > ")
    return f"{meta['file']}" + (f" § {where}" if where else "")


# --------------------------------------------------------------------------- answering

NOT_COVERED = "The cookbook does not cover that."


def engine(index, llm, k=5, floor="default"):
    """LlamaIndex's `CitationQueryEngine` over the cookbook retriever: sources are numbered, and the answer
    cites them as [1], [2]. It re-cuts each retrieved passage into citation-sized pieces first, so code
    chunks get a citation size large enough to keep a function whole."""
    from llama_index.core.query_engine import CitationQueryEngine
    return CitationQueryEngine.from_args(index, retriever=CookbookRetriever(index, llm, k=k, floor=floor),
                                         llm=llm, citation_chunk_size=1024, citation_chunk_overlap=0)


def ask(query_engine, question):
    """One question: the answer, its numbered sources, and whether the floor turned it away."""
    response = query_engine.query(question)
    sources = [{"n": i + 1, "where": label(s.node)} for i, s in enumerate(response.source_nodes)]
    refused = not response.source_nodes
    return {"question": question, "answer": NOT_COVERED if refused else str(response).strip(),
            "sources": sources, "refused": refused, "best score": query_engine.retriever.best,
            "response": response}


# --------------------------------------------------------------------------- the agent

AGENT_PROMPT = """You answer questions about agentic-forge-cookbook, a repo that teaches agent engineering in chapters
(01_foundations to 08_frameworks). Use search_cookbook to find passages -- more than once if the question has several
parts -- and open_file to read on when a passage stops short. Answer only from what you found, and cite the files you
used in square brackets, like [04_memory/recall.py]. If the cookbook does not cover the question, say so plainly."""


# What one tool result may put into the agent's conversation. Live, with 5 whole passages a search (11-14k
# characters), both questions died on Groq's 8,000-token limit on a single request after 3-5 tool calls: every
# result stays in the conversation, and every call re-sends all of it. Chapter 3's lesson, in a new place: a
# tool result is cut to what a conversation can carry, and the whole of it stays one call away (open_file).
SEARCH_K, PASSAGE_CHARS = 3, 1200
FILE_LINES, MAX_FILE_LINES = 60, 100
# LlamaIndex's `Memory(token_limit=...)` is not a net under this: it drops old messages between turns, and one
# question is one turn, kept whole however long it grows -- requests measured identical at a limit of 1,500
# tokens and of 1,000,000 (`ask_checks.agent_memory_limit`). So the cut has to be made in the tools.


_TEXTS = {}


def file_text(rel):
    """A cookbook file as the index read it: a notebook as its prose, anything else as it is."""
    if rel not in _TEXTS:
        path = CB.REPO / rel
        _TEXTS[rel] = CB._notebook_prose(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8")
    return _TEXTS[rel]


def first_line(node):
    """The line of its file a passage starts on, or None if it cannot be found there (the file has changed)."""
    head = next((l.strip() for l in node.get_content().splitlines() if len(l.strip()) >= 12), "")
    text = file_text(node.metadata["file"])
    at = text.find(head) if head else -1
    return None if at < 0 else text.count("\n", 0, at) + 1


def excerpt(text, query, size=PASSAGE_CHARS):
    """The run of whole lines, at most `size` characters, that shares the most words with the query.

    Cutting a passage to its first 1,200 characters cut the answer out of it: in case.py's main chunk, the
    sentence about a resumed node re-running from its first line starts at character 1,559. So the window
    that bears on the question is shown instead. Returns (first, last, text), lines counted within the passage.
    """
    import re
    words = set(re.findall(r"[a-z_]{3,}", query.lower()))
    lines = text.splitlines() or [""]
    best = (-1, 0, 1)
    for i in range(len(lines)):
        j, used = i, 0
        while j < len(lines) and used + len(lines[j]) + 1 <= size:
            used += len(lines[j]) + 1
            j += 1
        j = max(j, i + 1)
        score = sum(word in line.lower() for line in lines[i:j] for word in words)
        if score > best[0]:
            best = (score, i, j)
    _, i, j = best
    return i, j - 1, "\n".join(lines[i:j])[:size]


def tool_functions(index, llm, k=SEARCH_K, floor=None):
    """The agent's two tools as plain functions: LlamaIndex's agent and the scratch agent both get these.

    No floor by default. The floor is for the one-shot engine, which must turn away a question nothing is near;
    an agent searching step by step judges relevance itself, and a floor on meaning alone turns away an exact
    name: "double_send" scored 0.644 and came back empty, where BM25 would have matched it.
    """
    retriever = CookbookRetriever(index, llm, k=k, floor=floor)
    readable = {p.relative_to(CB.REPO).as_posix(): p for p in CB.files()}

    # Chapter 2's doom-loop rule, which LlamaIndex's FunctionAgent has no equivalent of: the same call with
    # the same arguments again is refused. Live, on a two-sided question, the agent searched "hazard" twice
    # running and drifted until its budget ran out. One set of calls per `tools()`, so one per question.
    made = set()

    def again(*call):
        key = tuple(" ".join(str(part).lower().split()) for part in call)
        if key in made:
            return ("You already made exactly this call, and what it returned is above. Answer from what you have "
                    "found, or try something different.")
        made.add(key)
        return None

    def search_cookbook(query: str) -> str:
        """Search the cookbook. Returns the best passages, each labelled with its file, section and the lines it
        came from; a long passage shows the part nearest the query. Use open_file with those lines to read more."""
        repeated = again("search", query)
        if repeated:
            return repeated
        nodes = retriever.retrieve(query)
        if not nodes:
            return "Nothing in the cookbook is close to that."
        shown = []
        for hit in nodes:
            text, start = hit.node.get_content(), first_line(hit.node)
            i, j, part = excerpt(text, query)
            where = f", lines {start + i}-{start + j}" if start else ""
            longer = "" if part == text else (
                f"  (part of a longer passage" + (f", lines {start}-{start + text.count(chr(10))}" if start else "") + ")")
            shown.append(f"[{label(hit.node)}{where}]{longer}\n{part}")
        return "\n\n".join(shown)

    # `line_start`/`line_end`: the names gpt-oss reaches for first. Named the other way round, its first call
    # failed on every question and it tried again -- a whole call, and a whole conversation re-sent, wasted
    def open_file(path: str, line_start: int = 1, line_end: int = FILE_LINES) -> str:
        """Read lines of one cookbook file, by the path a search result gave, at most 100 lines at a time.
        Notebooks are read as their prose."""
        if path not in readable:
            # only what is indexed: not .env, not anything outside the repo, whatever the path says
            return f"Not a cookbook file: {path!r}. Use a path exactly as search_cookbook labels it."
        repeated = again("open", path, line_start, line_end)
        if repeated:
            return repeated
        file = readable[path]
        text = CB._notebook_prose(file) if file.suffix == ".ipynb" else file.read_text(encoding="utf-8")
        lines = text.splitlines()
        start = max(1, line_start)
        end = min(len(lines), line_end, start + MAX_FILE_LINES - 1)
        return "\n".join(f"{i:>4}  {lines[i - 1]}" for i in range(start, end + 1)) + \
            (f"\n... ({len(lines)} lines in all)" if end < len(lines) else "")

    return [search_cookbook, open_file]


def tools(index, llm, k=SEARCH_K, floor=None):
    """The two tools as LlamaIndex `FunctionTool`s, which read the schemas off the functions' signatures."""
    from llama_index.core.tools import FunctionTool
    return [FunctionTool.from_defaults(f) for f in tool_functions(index, llm, k=k, floor=floor)]


# Both agents get eight tool calls and then must answer from what they found: chapter 2's `max_tool_calls` with
# its graceful finish, and LlamaIndex's `max_iterations` with `early_stopping_method="generate"` -- the same
# thing in each, a last call with no tools offered. Without it, both kept exploring a two-sided question
# until something else stopped them.
MAX_TOOL_CALLS = 8


def agent(index, llm, k=SEARCH_K, floor=None):
    from llama_index.core.agent.workflow import FunctionAgent
    # streaming off: a streamed reply carries no `usage`, and the budget is counted from `usage`
    return FunctionAgent(tools=tools(index, llm, k=k, floor=floor), llm=llm, system_prompt=AGENT_PROMPT,
                         streaming=False, timeout=300, early_stopping_method="generate")


async def run_agent(the_agent, question, max_iterations=MAX_TOOL_CALLS + 1, memory=None):
    """One question to the agent: its answer, and every tool call it made on the way.

    A run stopped part-way -- its budget spent, or LlamaIndex's step limit reached -- still comes back with
    the calls it made and why it stopped, rather than as an exception that takes the record with it.
    """
    from llama_index.core.agent.workflow import ToolCallResult
    handler = the_agent.run(user_msg=question, max_iterations=max_iterations,
                            **({"memory": memory} if memory is not None else {}))
    calls, answer, stopped = [], None, None
    try:
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                calls.append({"tool": event.tool_name, "args": event.tool_kwargs,
                              "returned chars": len(str(event.tool_output))})
        answer = str(await handler).strip()
    except Exception as error:      # noqa: BLE001 -- a stopped run is reported, not raised
        stopped = f"{type(error).__name__}: {error}"
    return {"question": question, "answer": answer, "stopped": stopped, "calls": calls}


def refused_size(stopped):
    """The size of a request Groq refused as too large (a 413), from why a run stopped; else None.

    Only "Request too large": a daily-limit 429 also says "Requested N", and is not about size.
    """
    import re
    found = re.search(r"Requested (\d+)", stopped) if stopped and "Request too large" in stopped else None
    return int(found.group(1)) if found else None


# --------------------------------------------------------------------------- evaluation

def expected_ids(index, files):
    """Every chunk of the files an answer is in: a retrieval hits if it returns any of them."""
    return [node_id for ref, info in index.ref_doc_info.items() if ref in files for node_id in info.node_ids]


async def retrieval_scores(index, retriever, questions=None):
    """Hit rate and MRR per question, by LlamaIndex's `RetrieverEvaluator`."""
    from llama_index.core.evaluation import RetrieverEvaluator
    evaluator = RetrieverEvaluator.from_metric_names(["hit_rate", "mrr"], retriever=retriever)
    rows = []
    for question, files, _ in questions or CB.EVAL:
        result = await evaluator.aevaluate(query=question, expected_ids=expected_ids(index, files))
        rows.append({"question": question, **{k: round(v, 3) for k, v in result.metric_vals_dict.items()}})
    return rows


def save(name, record):
    """A live result, kept on disk so the notebook's summary does not depend on one kernel."""
    STORE.mkdir(exist_ok=True)
    (STORE / f"{name}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record


def load(name):
    path = STORE / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
