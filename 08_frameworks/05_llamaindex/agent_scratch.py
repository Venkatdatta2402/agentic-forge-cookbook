"""The cookbook agent again, from this repo's parts -- with chapter 5 deciding what each model call is sent.

LlamaIndex's agent keeps everything it has found and sends all of it on every call. On Groq's free tier,
which refuses a single request over 8,000 tokens, that gave it room for about six tool calls per question,
and a two-sided question needed more. Chapter 5's whole subject is the alternative: context chosen per call.

    the loop       chapter 2's `Runtime`: Think -> Act -> Observe, repeated until Think answers
    the tools      the same two functions LlamaIndex's agent gets (`ask_index.tool_functions`), over the same
                   index, as chapter 3 `Tool`s -- so what differs is the loop and its context, not the search
    observing      chapter 6's `verbatim`: a tool's result goes into the conversation as it is, as it does
                   for LlamaIndex's agent
    the context    chapter 5's `ContextSystem` around Think, with a `Policy`: a token budget per call, the
                   question pinned, the last results always kept, and older ones kept or dropped by how
                   much they bear on the question. The conversation keeps everything; each CALL gets a
                   chosen part of it, so no call can outgrow the budget however much has been found
    metering       `ask_index.Budget` around chapter 1's client: every call counted, none past the budget
"""

import sys
from pathlib import Path

import ask_index as A
import cookbook as CB

# 04_memory before 06_orchestration, and the order matters: both have a `recall.py` -- chapter 4's ranking, which
# chapter 5's selection imports, and chapter 6's battery-recall scenario -- and whichever is first on the path
# is the one `import recall` finds. Chapter 6 warns about exactly this in `coordination.py`.
for _chapter in ("01_foundations", "02_agent_runtime", "03_tools", "04_memory", "05_context", "06_orchestration"):
    if str(CB.REPO / _chapter) not in sys.path:
        sys.path.append(str(CB.REPO / _chapter))

CONTEXT_TOKENS = 5000      # what one Think call may be sent: 8,000 is the request limit, and tool schemas,
KEEP_LAST = 2              # the system prompt and the model's own output fit in what is left
# Every kind but `tool_result`: Observe passes a result through verbatim, so the observation IS the tool's
# result, and chapter 5 lists the raw result again as its own item. Measured offline: 6,538 tokens of
# observations and 6,413 of the same text again as raw results.
THINK_SEES = ("instruction", "request", "constraint", "message", "observation", "decision", "state",
              "fact", "memory", "summary")


class metered:
    """`ask_index.Budget` around chapter 1's client, for the length of a `with` block."""

    def __init__(self, budget):
        self.budget = budget

    def __enter__(self):
        import llm
        real = self._real = llm.client.chat.completions.create

        def create(*args, **kwargs):
            self.budget.check()
            response = real(*args, **kwargs)
            self.budget.charge(response)
            return response

        llm.client.chat.completions.create = create
        return self.budget

    def __exit__(self, *exc):
        import llm
        llm.client.chat.completions.create = self._real


def build(index, model=A.MODEL, context_tokens=CONTEXT_TOKENS, keep_last=KEEP_LAST, floor=None,
          max_tool_calls=A.MAX_TOOL_CALLS):
    """The loop, and the context system that chooses what its Think is sent."""
    from llama_index.core.llms import MockLLM

    from act import Act
    from observe import Observe
    from registry import Registry
    from runtime import Runtime
    from supervisor import Think, verbatim
    from system import ContextSystem, Policy
    from tools import Tool

    from focus import Focus, View, to_messages

    class FocusedThink(Focus):
        """Chapter 5's `Focus`, passing chapter 2's graceful finish through.

        Runtime ends a stopped run by giving the last component with a `finish` one more turn, without
        tools, to answer from what it gathered. `Focus` has no `finish`, so a focused Think is invisible to
        it and a stopped run ends on "Stopping: ... partial progress" instead. Forwarded here with the same
        chosen context as every other call -- the whole conversation would be the request over 8,000 tokens.
        """

        def finish(self, conversation, reason):
            return self.component.finish(View(conversation, to_messages(self.items(conversation))), reason)

    # the retriever's fusion step takes an LLM it never calls (one query, no variants): LlamaIndex's MockLLM
    # says so plainly, and keeps the scratch agent's model calls all on chapter 1's client
    registry = Registry([Tool(f) for f in A.tool_functions(index, MockLLM(), floor=floor)])
    context = ContextSystem()
    think = Think(model=model, tools=registry.tools, allow_fork=False, explain=False, max_tokens=4096)
    policy = Policy(budget=context_tokens, keep_last=keep_last, kinds=THINK_SEES)
    focused = FocusedThink(think, prepare=lambda items, conversation: context.apply(items, conversation, policy))
    runtime = Runtime(loop=[focused,
                            Act(functions=registry.functions, schemas=registry.schemas),
                            Observe(extractors={name: verbatim for name in registry.names})],
                      # chapter 2's tool-call budget: after this many, Think answers from what it has
                      repeat_from=0, max_iterations=36, max_tool_calls=max_tool_calls)
    return runtime, context


def run(index, question, budget, **settings):
    """One question: the answer, every tool call, why it stopped if it did, and what each call was sent."""
    from chat import Conversation
    runtime, context = build(index, **settings)
    conversation = Conversation(system=A.AGENT_PROMPT)
    conversation.messages.append({"role": "user", "content": question})
    answer, stopped = None, None
    with metered(budget):
        try:
            runtime.run(conversation)
            answer = conversation.messages[-1]["content"]
            stopped = conversation.stopped or None
        except A.BudgetExceeded as error:
            stopped = f"BudgetExceeded: {error}"
        except ValueError as error:
            # chapter 5's selection refusing: what must go in -- the question, the last results -- does not
            # fit the budget. Reported like a spent budget, because it is one
            if "Selection can leave things out" not in str(error):
                raise
            stopped = f"selection refused: {error}"
        except Exception as error:      # noqa: BLE001 -- Groq refusing a request too large, reported as a stop
            import openai
            if not isinstance(error, openai.APIStatusError):
                raise
            stopped = f"{type(error).__name__}: {error}"
    calls = [{"tool": m["tool"], "args": m["content"]} for m in conversation.messages
             if m.get("role") == "decision" and m.get("type") == "tool_call"]
    return {"question": question, "answer": answer, "stopped": stopped, "calls": calls,
            "context per call": [str(t) for t in context.traces],
            "usage": {**budget.summary(), "refused request": A.refused_size(stopped)}}
