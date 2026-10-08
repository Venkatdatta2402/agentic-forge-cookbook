"""The corpus: this repo, as the documents an "ask the cookbook" assistant answers from, and the questions it is tested on.

    documents()    every README, the prose of every notebook, and every module, as LlamaIndex `Document`s
    EVAL           questions the cookbook answers, each with the file(s) the answer is in, and a word the
                   answer must contain to be about the right thing
    OUT_OF_SCOPE   questions it does not answer, and should say so

What goes in, and what stays out:
    in     READMEs; notebook MARKDOWN cells, which is where the notebooks explain; .py modules, which is
           where the explained things are implemented
    out    notebook code cells and outputs -- the code is mostly calls into the modules, which are indexed
           themselves, and the outputs are logs and tables; tests and stand-ins; stored runs; and this
           folder, so the assistant cannot find its own answer key
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve().parent

_SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "store", ".git"}


def _skipped(path):
    parts = set(path.relative_to(REPO).parts)
    return (bool(parts & _SKIP_DIRS) or SELF in path.parents
            or path.name.startswith("test_") or path.name.endswith("_stubs.py"))


def files():
    """Every file the assistant reads, in a stable order."""
    found = [REPO / "README.md"]
    for chapter in sorted(p for p in REPO.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        for pattern in ("**/README.md", "**/*.ipynb", "**/*.py"):
            found += sorted(p for p in chapter.glob(pattern) if not _skipped(p))
    return found


def _notebook_prose(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")


def documents():
    """One `Document` per file. Its id is the file's path, so a changed file replaces its old version."""
    from llama_index.core import Document
    docs = []
    for path in files():
        rel = path.relative_to(REPO).as_posix()
        kind = "notebook" if path.suffix == ".ipynb" else "code" if path.suffix == ".py" else "readme"
        text = _notebook_prose(path) if kind == "notebook" else path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        chapter = rel.split("/")[0] if "/" in rel else "(repo)"
        docs.append(Document(id_=rel, text=text, metadata={"file": rel, "chapter": chapter, "kind": kind}))
    return docs


# --------------------------------------------------------------------------- the questions

# (question, files the answer is in, a word a right answer contains). Each question is one a reader would ask after
# reading a chapter's README -- the answer may be in the README, a notebook or the code. Every expected file was
# checked to hold the answer before it was written down here.
EVAL = [
    ("The README says extract() asks the API to enforce the schema first. If the API enforces it, why validate as well?",
     {"01_foundations/llm.py", "01_foundations/notebooks/05_structured_outputs.ipynb", "01_foundations/README.md"}, "sense"),
    ("What counts as a doom loop, and what does the runtime do when it detects one?",
     {"02_agent_runtime/runtime.py", "02_agent_runtime/notebooks/02_the_agent_loop.ipynb"}, "identical"),
    ("Why do Graph and forking use threads instead of asyncio.gather?",
     {"02_agent_runtime/README.md", "02_agent_runtime/notebooks/04_planning.ipynb"}, "event loop"),
    ("Why can't the runtime's time budget stop a tool that hangs, and what can?",
     {"03_tools/README.md", "03_tools/invoke.py", "03_tools/notebooks/03_invocation.ipynb"}, "timeout"),
    ("What is the difference between a tool that refuses and one that is broken, and which does the model get to see?",
     {"03_tools/invoke.py", "03_tools/README.md", "03_tools/notebooks/03_invocation.ipynb"}, "could not be completed"),
    ("Why shouldn't agent memory start with a vector database?",
     {"04_memory/README.md", "04_memory/notebooks/01_what_memory_is.ipynb", "04_memory/notebooks/02_stores.ipynb",
      "04_memory/stores.py"}, "approximat"),
    ("Why is recall injected into the loop while remembering is a tool the agent calls?",
     {"04_memory/README.md", "04_memory/notebooks/05_memory_in_the_loop.ipynb", "04_memory/memory_tools.py"}, "decid"),
    ("What does compaction leave untouched, and why?",
     {"05_context/compression.py", "05_context/notebooks/03_context_compression.ipynb"}, "pinned"),
    ("What is the difference between handing off a conversation and delegating to another agent?",
     {"06_orchestration/supervisor.py", "06_orchestration/notebooks/02_delegation_and_supervision.ipynb",
      "06_orchestration/README.md"}, "handoff"),
    ("Why did the multi-agent desk in chapter 6 skip two of its own patterns, and what fixed it?",
     {"06_orchestration/notebooks/05_build_a_multi_agent_system.ipynb", "06_orchestration/README.md"}, "form"),
    ("Does running an MCP server as a separate process sandbox it? Could it read my API keys?",
     {"07_mcp/README.md", "07_mcp/notebooks/05_mcp_transport.ipynb"}, "environment"),
    ("Why can verified() only run on the MCP server's side?",
     {"07_mcp/README.md", "07_mcp/notebooks/04_mcp_server.ipynb"}, "world"),
    ("Why is CrewAI pinned to an old version in this repo?",
     {"08_frameworks/README.md"}, "chromadb"),
    ("Why couldn't AutoGen's TokenUsageTermination enforce the token budget?",
     {"08_frameworks/04_autogen/analysis_team.py", "08_frameworks/README.md",
      "08_frameworks/04_autogen/analysis_team.ipynb"}, "turn"),
    ("How does the scratch accounts-payable desk wait for a person without re-running a step?",
     {"08_frameworks/02_langgraph/case.py", "08_frameworks/02_langgraph/ap_desk.ipynb"}, "Wait"),
]


def _plain(text):
    # case, thousands separators and hyphens of every kind: "2,048" is "2048", and "hand‑off", written with
    # the non-breaking hyphen gpt-oss favours, is "handoff"
    for mark in (",", "-", "‐", "‑", "‒", "–"):
        text = text.replace(mark, "")
    return " ".join(text.lower().split())


def mentions(answer, word):
    """Whether an answer contains the word, ignoring case, thousands separators and hyphens."""
    return _plain(word) in _plain(answer)


OUT_OF_SCOPE = [
    "How do I fine-tune Llama 3 on my own data?",
    "What is the capital of Australia?",
    "Can I bring my dog on the plane?",
]

# Questions that need more than one place in the repo -- for the agent, which can search more than once
CROSS_CHAPTER = [
    "How is chapter 6's shared board different from chapter 4's memory stores? When would you use each?",
    "In chapter 8's LangGraph project, compare how the LangGraph desk and the from-scratch case file wait for a "
    "person, and what can go wrong with each.",
]
