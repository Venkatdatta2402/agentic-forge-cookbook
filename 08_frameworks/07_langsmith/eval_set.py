"""What the cookbook assistant is evaluated on: ten questions, what a right answer says, and where that is written.

    QUESTIONS   seven one-shot questions, one per chapter 1-7, for 05's cited engine, and three cross-chapter
                questions for 05's agent. Each has
                    points     what a right answer says, one statement each, and for each the exact words in the
                               repo it rests on
                    evidence   the lines those words are in, as (file, first line, last line)
                    files      files a right answer may cite
                    word       (one-shot only) a word a right answer contains, as in 05's grading
    evidence()  the evidence lines, numbered as the agent's `open_file` numbers them -- a notebook as its prose,
                anything else as it is -- so the judge reads what the assistant could read
    unquoted()  every quote that is not in its question's evidence: empty, or a point rests on nothing

A point is written in plain words; its quotes are copied from the repo, and `unquoted()` checks each is really
there, in the lines the judge is shown. A reworded source breaks the check rather than leaving a point that is no
longer true. The seven one-shot questions are 05's (`cookbook.EVAL`), one per chapter; the first two cross-chapter
questions are 05's `CROSS_CHAPTER`, and the third is new and, like them, needs two places in the repo.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA = HERE.parent / "05_llamaindex"
if str(LLAMA) not in sys.path:
    sys.path.append(str(LLAMA))

import cookbook as CB      # noqa: E402 -- 05's corpus: the files, and notebooks read as their prose

AP = "08_frameworks/02_langgraph/ap_desk.ipynb"


def _one_shot(question, points, evidence):
    files, word = next((files, word) for q, files, word in CB.EVAL if q == question)
    return {"kind": "one-shot", "question": question, "points": points, "evidence": evidence,
            "files": sorted(files), "word": word}


def _agent(question, points, evidence, files):
    return {"kind": "agent", "question": question, "points": points, "evidence": evidence, "files": sorted(files)}


QUESTIONS = [
    _one_shot(
        "The README says extract() asks the API to enforce the schema first. If the API enforces it, why validate as well?",
        [("Schema enforcement (json_schema mode) only guarantees the shape: the right field names and types, nothing "
          "missing.",
          ["`json_schema` mode makes the API enforce the SHAPE -- right field names, right types, nothing missing."]),
         ("No schema can constrain sense: rules like one score per candidate, or an index that exists, cannot be "
          "written in a JSON schema.",
          ["What no schema can constrain is SENSE.",
           "\"One score per candidate\" and \"an index that exists\" are not things a JSON schema can say."]),
         ("Measured: asked to score three candidates out of ten, SelectThink returned scores=[869.0] -- one number for "
          "three candidates, and over the maximum of 10 -- and keep_indices=[20] into a list of three; all of it "
          "schema-valid.",
          ["`scores=[869.0]` -- one number for three candidates, and 869 out of a stated maximum of 10 -- alongside "
           "`keep_indices=[20]`, an index into a list of length three. Every one of those is schema-valid."]),
         ("A model that rejects json_schema mode falls back to plain JSON, where validation is the only check.",
          ["A model that rejects the mode falls back to plain JSON",
           "this model cannot enforce schemas -- ask for plain JSON and lean on validation"]),
         ("When validation fails, the model is told exactly what was wrong and asked again.",
          ["the model is told exactly what was wrong with what it produced",
           "Return corrected JSON only, matching the schema."])],
        [("01_foundations/llm.py", 86, 122)]),
    _one_shot(
        "What counts as a doom loop, and what does the runtime do when it detects one?",
        [("The same tool called with identical arguments twice in a row.",
          ["return \"the same tool call with identical arguments repeated\""]),
         ("The same error twice: the last two tool errors are identical.",
          ["if len(errors) >= 2 and errors[-1] == errors[-2]:", "return \"the same error repeated\""]),
         ("Oscillating between the same two calls (A, B, A, B).",
          ["return \"oscillating between the same two calls\""]),
         ("The same tool asked nearly the same thing (similar arguments) and returning the same result.",
          ["was asked the same thing again and returned the same result"]),
         ("The run stops, and the Think gets one more turn without tools to answer from what was gathered.",
          ["So the Think gets one more turn, with no tools, and answers from what is there, saying what it could not "
           "find."])],
        [("02_agent_runtime/runtime.py", 181, 249)]),
    _one_shot(
        "What is the difference between a tool that refuses and one that is broken, and which does the model get to see?",
        [("A refusal is the tool running and saying no (it raises ToolError); its message goes to the model, which "
          "can act on it.",
          ["a failure the model can do something about -- wrong arguments, a tool that ran and said no",
           "return Result(tool.name, \"refused\", str(e), repeatable=tool.repeatable)"]),
         ("A broken tool is a failure of the tool itself -- a missing key, a bad import, a bug -- which is ours to fix.",
          ["The tool itself is broken -- a missing key, a bad import, a bug.",
           "False only when the tool itself is broken, which is ours to fix"]),
         ("For a broken tool the model sees only \"<tool> could not be completed.\"; the real cause is kept in "
          "Result.value for whoever reads the run.",
          ["`\"<tool> could not be completed.\"` and the cause moves to `Result.value`, where whoever is reading the "
           "run can still find it."]),
         ("The cause is kept from the model because telling it invites it to work around something it cannot see.",
          ["Saying so in the conversation invites the model to work around something it cannot see"])],
        [("03_tools/invoke.py", 50, 72), ("03_tools/invoke.py", 128, 181), ("03_tools/README.md", 86, 99)]),
    _one_shot(
        "Why is recall injected into the loop while remembering is a tool the agent calls?",
        [("Recall is a component (Retrieve) that runs before Think on every turn, whether or not anyone asked; "
          "remember is a tool the model calls when it decides something is worth keeping.",
          ["Retrieve a COMPONENT. Runs before Think, every turn, whether or not anyone asked.",
           "remember a TOOL. The model calls it when it decides something is worth keeping."]),
         ("Offered recall as a tool, the model called it on every question about the past (15 of 15), but consulted "
          "memory only 7 times in 15 when asked to decide something; the misses answered from general knowledge while "
          "the team's own rule sat unread.",
          ["over fifteen trials on questions that *ask about the past* it called it fifteen times.",
           "it consulted memory seven times in fifteen, and the misses answer out of general knowledge while the "
           "team's own rule (*page Priya above two percent*) sits unread."]),
         ("That failure is invisible: no error and no empty result, just a fluent answer that consulted nothing.",
          ["And the failure is invisible: no empty result, no error, just a fluent answer that consulted nothing."]),
         ("Not writing something down costs nothing, and judging what is worth keeping is a judgement, which a model "
          "is good at -- so the model decides what to remember.",
          ["There is no cost to *not* writing something down, judging what is worth keeping is a judgement, and a "
           "model is good at judgements. So the model decides."])],
        [("04_memory/memory_tools.py", 1, 20), ("04_memory/README.md", 186, 199)]),
    _one_shot(
        "What does compaction leave untouched, and why?",
        [("Pinned items: a summarizer that paraphrases a rule has changed it.",
          ["pinned items: a summarizer that paraphrases a rule has changed it"]),
         ("Task state (the current plan, the scratchpad): it is already the run's own summary, and the next step "
          "starts from it.",
          ["task state (the current plan, the scratchpad): it is already the run's own summary of where things "
           "stand, and the next step starts from it"]),
         ("The most recent turns (keep_recent): what the next step is most likely to build on, kept verbatim.",
          ["the last `keep_recent` turns: what the next step is most likely to build on, verbatim"]),
         ("Anything that did not come from the run -- a pasted profile, a catalogue, a retrieved memory: compaction "
          "compresses the run, not what was pasted into it.",
          ["anything that did not come from the run: a pasted profile, an always-present catalogue, a retrieved "
           "memory.", "**Compaction compresses the run, not what was pasted into it.**"])],
        [("05_context/compression.py", 258, 282)]),
    _one_shot(
        "What is the difference between handing off a conversation and delegating to another agent?",
        [("Delegation keeps one agent in charge: it asks another agent (called as a tool) and relays the answer, and "
          "every turn comes back through it.",
          ["Delegation keeps one agent in charge: it asks another and relays the answer, and every turn comes back "
           "through it.",
           "as_tool an agent wrapped as a chapter 3 Tool, so another agent can call it"]),
         ("A handoff moves the conversation itself: the new agent talks to the person directly from then on, seeing the "
          "whole conversation so far.",
          ["A handoff moves the conversation itself.",
           "the one it hands to talks to the person directly from then on, seeing the whole conversation so far."]),
         ("A handoff is an ordinary transfer_to_<name> tool that stops the current agent through the runtime's cancel "
          "hook, and the new owner answers the same turn.",
          ["Each target becomes a `transfer_to_<name>` tool for that agent. Calling it stops the agent's loop on the "
           "spot -- through Runtime's `cancel`",
           "the new owner answers the same turn"]),
         ("Measured on one customer chat: the handoff was 2.5 times cheaper and kept the specialist's own voice, but "
          "the supervisor is out of the conversation.",
          ["Then **handoffs** against delegation on one customer chat: 2.5× cheaper, the specialist's own voice, but "
           "the supervisor is out of it."])],
        [("06_orchestration/supervisor.py", 1, 13), ("06_orchestration/supervisor.py", 110, 122),
         ("06_orchestration/README.md", 57, 68)]),
    _one_shot(
        "Does running an MCP server as a separate process sandbox it? Could it read my API keys?",
        [("No: a separate process isolates failures, not authority -- a stdio server starts with your authority and "
          "can read your credentials, open your files and reach the network as you.",
          ["A stdio server is not sandboxed by being a separate process",
           "it can read your credentials, open your files and reach the network as you.",
           "The separateness buys isolation of *failures*, not of *authority*"]),
         ("Yes: the notebook reads a planted API key back out of one.",
          ["the notebook reads a planted API key back out of it."]),
         ("The MCP SDK passes the server only an allowlist of environment variables by default (no secrets); "
          "env={**os.environ} overrides that with everything, so the leak is the client's doing.",
          ["**the SDK ships an allowlist** (`DEFAULT_INHERITED_ENV_VARS`, twelve variables on Windows, no secrets)",
           "`env={**os.environ}` overrides that careful default with everything.",
           "And the leak is the client's doing."]),
         ("The repo's stdio_params(inherit=False) takes it back.",
          ["`stdio_params(inherit=False)` is the switch", "`inherit=False` takes it back"])],
        [("07_mcp/README.md", 109, 117), ("07_mcp/README.md", 355, 364)]),

    _agent(
        "How is chapter 6's shared board different from chapter 4's memory stores? When would you use each?",
        [("The board holds the working state of one case while agents work on it, alive for minutes, in an ordinary "
          "object in memory; it is not memory.",
          ["This is not memory.",
           "What this holds is the working state of that case, alive for minutes, read on nearly every turn by "
           "whoever touches it -- so it lives in an ordinary object in memory"]),
         ("A board field can have one owner, the only agent allowed to write it: an agent gets a write tool only for "
          "the fields it owns, and a write to someone else's field is refused. A value two agents can both set is one "
          "neither can trust.",
          ["owned = [f for f, who in self.owner.items() if who == agent]",
           "raise Denied(f\"{field} is {owner}'s to set, not {author}'s\")",
           "a value two agents can both set is a value neither of them can trust."]),
         ("Reading the board is injection into each agent's context; writing is a tool bound to one agent, so every "
          "write has an author and ownership can be checked.",
          ["Reading is injection",
           "Writing is a tool, bound to one agent's name, so a write carries an author and ownership can actually be "
           "checked."]),
         ("Chapter 4's stores keep what should outlive a run, on disk (SQLite and Chroma to files, Neo4j to a server).",
          ["`04_memory` keeps what should outlive a run, in a store on disk",
           "**Everything here persists** -- SQLite and Chroma to files, Neo4j to a server."]),
         ("Which store a memory goes in is decided by what will be asked of it later: KeyValue looked up by name, "
          "Records filtered and counted, Journal searched by words and by resemblance, Graph walked.",
          ["**what will be asked of it later**", "KeyValue it will be looked up by name",
           "Records it will be filtered, ordered, counted", "Journal it will be searched by literal and by resemblance",
           "Graph it will be walked"]),
         ("So the board is for what agents need while working one case together, finished when the case is; a store "
          "is for what must still be there in a later run.",
          ["is finished when the reply goes out.", "`04_memory` keeps what should outlive a run"])],
        [("06_orchestration/state.py", 1, 20), ("06_orchestration/state.py", 67, 73),
         ("06_orchestration/state.py", 89, 108), ("04_memory/stores.py", 1, 12), ("04_memory/stores.py", 49, 52)],
        {"06_orchestration/state.py", "04_memory/stores.py", "04_memory/README.md", "06_orchestration/README.md",
         "06_orchestration/notebooks/03_communication_state_and_collaboration.ipynb",
         "04_memory/notebooks/02_stores.ipynb"}),
    _agent(
        "In chapter 8's LangGraph project, compare how the LangGraph desk and the from-scratch case file wait for a "
        "person, and what can go wrong with each.",
        [("LangGraph waits with interrupt(), and resuming re-runs the node from its first line.",
          ["`interrupt()`; resuming **re-runs the node** from its first line",
           "**A resumed node runs again from its first line.**"]),
         ("So a side effect before the interrupt in the same node, such as an email, happens twice; the case file "
          "sends it once.",
          ["A node that sends an email and then waits for the reply sends it twice. The same step in the case file "
           "sends once"]),
         ("LangGraph matches answers to interrupt() calls by position, so a deploy that reorders them swaps the "
          "answers; the case file stores each answer in its approver's own field.",
          ["matched by **position**: the answers swap", "each answer is its approver's field: nothing to swap"]),
         ("Command(resume=...) accepts whoever answers: the graph records the wrong person's approval as the right "
          "person's, while the case file refuses it and keeps waiting.",
          ["`Command(resume=...)` resumes the run with whatever it is given",
           "| an approval answered by the wrong person | recorded as the right person's | refused; the case still "
           "waits |"]),
         ("In the case file, asking ends the step: a Wait names the field, its owner and the step that continues, and "
          "the owner writing that field starts the next step.",
          ["A step that needs a person returns a `Wait`: the field that will hold the answer, the person who owns it, "
           "and the step that continues. Answering is that person writing that field."]),
         ("In both, durability stops at the workflow: the payment made in another process existed only in that "
          "process's memory.",
          ["**Durability stops at the workflow, in both.**", "its payment exists only in that process's memory."]),
         ("The case file gives up what LangGraph has: rewinding and correcting a past step, parallel steps, streaming "
          "and a server.",
          ["| rewinding and correcting a past step | `get_state_history`, `update_state` forks | not built |",
           "| parallel steps, streaming, caching, timeouts, a server | yes | not built |"])],
        [(AP, 366, 390), (AP, 493, 497), (AP, 519, 541)],
        {AP, "08_frameworks/02_langgraph/case.py", "08_frameworks/02_langgraph/ap_graph.py",
         "08_frameworks/02_langgraph/ap_scratch.py", "08_frameworks/README.md"}),
    _agent(
        "How does chapter 2's runtime stop an agent that keeps calling the same tool, and how does LangGraph's limit "
        "compare in chapter 8's LangGraph project?",
        [("Chapter 2's runtime stops on the second identical call (doom-loop detection).",
          ["Chapter 2's `Runtime` stops on the *second identical call* — doom-loop detection"]),
         ("It then gives the agent one turn without tools to answer from what it found.",
          ["gives the agent one turn without tools to say what it found"]),
         ("LangGraph's default step limit is 10,007.",
          ["**The default step limit is 10,007.**"]),
         ("Measured: with the default limit the same agent ran 5,004 model calls before LangGraph stopped it.",
          ["with the default limit, the same agent ran **5,004 model calls**"]),
         ("A count cannot tell an agent working hard from one going round in circles.",
          ["A count cannot tell an agent working hard from one going round in circles."]),
         ("A loop through interrupt() is never stopped by the step limit, because each resume is a new invoke.",
          ["**A loop through `interrupt()` is never stopped by `recursion_limit`**, because the limit is per "
           "`invoke` and every resume is a new `invoke`."]),
         ("In the project's side-by-side, the runaway was held after 9 calls at the limit the project set, and after "
          "3 by chapter 2's doom-loop check.",
          ["| a runaway agent | a step count, default **10,007**; held after 9 calls at our limit | chapter 2's "
           "doom-loop check; held after **3** |"])],
        [("02_agent_runtime/runtime.py", 181, 186), ("02_agent_runtime/runtime.py", 220, 232), (AP, 499, 503),
         (AP, 530, 530)],
        {"02_agent_runtime/runtime.py", AP, "02_agent_runtime/notebooks/02_the_agent_loop.ipynb",
         "08_frameworks/README.md"}),
]


def _text(rel):
    path = CB.REPO / rel
    return CB._notebook_prose(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8")


def _lines(rel, first, last):
    lines = _text(rel).splitlines()
    return [(n, lines[n - 1]) for n in range(first, min(last, len(lines)) + 1)]


def evidence(question):
    """The evidence lines of one question, numbered, under the file they come from."""
    return "\n\n".join(f"[{rel}, lines {first}-{last}]\n" + "\n".join(f"{n:>4}  {line}" for n, line in
                                                                      _lines(rel, first, last))
                       for rel, first, last in question["evidence"])


def _flat(text):
    # a sentence can run over lines, and a code comment's lines each start with "#": compare without either
    return " ".join(" ".join(re.sub(r"^\s*#\s?", "", line) for line in text.splitlines()).split())


def unquoted(question):
    """Every quote of a question's points that its evidence lines do not contain, as (point number, quote)."""
    shown = " ".join(_flat("\n".join(line for _, line in _lines(*span))) for span in question["evidence"])
    return [(i, quote) for i, (_, quotes) in enumerate(question["points"], 1) for quote in quotes
            if _flat(quote) not in shown]
