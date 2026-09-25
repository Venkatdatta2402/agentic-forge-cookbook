import concurrent.futures
import json
import re

from openai import BadRequestError
from pydantic import BaseModel, Field

from llm import DEFAULT_MODEL, client, extract, tool_call_failure


def render(conversation):
    return "\n".join(f"{m['role']}: {_rendered_content(m)}" for m in conversation.messages)


def _rendered_content(m):
    # A tool_call decision keeps its tool's name in `tool` and only the arguments in `content`,
    # so rendering `content` alone turned get_metrics and get_deploys into the same line --
    # `decision: {"service": "checkout-api"}` -- in every prompt built from this function.
    # Found by 05_context's first notebook, looking at what Observe was actually sent. Written
    # the way transcript() already writes a call, so both views of a run read alike.
    if m.get("role") == "decision" and m.get("type") == "tool_call":
        return f"{m['tool']}({m['content']})"
    return m["content"]


def branch_label(branch):
    # whatever a branch should be called in a conversation -- its id when it is addressable, so a
    # Think can name it back when deciding who continues
    return branch.get("label") or branch.get("goal") or branch.get("id")


def transcript(branch):
    # a branch's whole working, compacted: goal, the calls it made and why, what came back, and
    # its answer. Lives here beside render() rather than on Observe because two very different
    # callers need it -- Observe, to show a caller the working instead of just the answer, and
    # ExpandThink, to hand a child the state it inherits from its parent.
    lines = [f"[{branch_label(branch)}]"]
    if branch.get("goal"):
        lines.append(f"  goal: {branch['goal']}")
    for m in getattr(branch.get("conversation"), "messages", []):
        if m.get("role") == "decision" and m.get("type") == "tool_call":
            why = f"  -- {m['reasoning']}" if m.get("reasoning") else ""
            lines.append(f"  called {m['tool']}({m['content']}){why}")
        elif m.get("role") == "tool":
            lines.append(f"  got {m['content']}")
    lines.append(f"  answer: {branch['content']}")
    return "\n".join(lines)


_BUDGET_LABELS = {
    "seconds": "seconds",
    "iterations": "iterations",
    "tool_calls": "tool calls",
    "tokens": "tokens",
}

# Budgets come in two kinds, and telling a fork to divide the wrong one is a real mistake -- the
# first version of this told the model to split all four "between branches as whole numbers".
#
# POOLED: consumed for real, so three branches cost three times as much. These must be divided.
# PER-BRANCH: a ceiling each branch faces on its own. Branches run CONCURRENTLY, so three of them
#   finish when the slowest finishes, not when their times add up -- and each Runtime counts its
#   own iterations, so a branch's loop never decrements the caller's. Dividing these protects
#   nothing and just cripples every branch: given 45 seconds, three branches should each be able
#   to use most of the 45, not 15 apiece.
_POOLED_BUDGETS = ("tokens", "tool_calls")
_PER_BRANCH_BUDGETS = ("seconds", "iterations")


def budget_guidance(conversation):
    """What remains, and which kinds may be divided -- as prose for the system message.

    This lived in `_delegate_tool`'s own description until it was measured. `delegate` is already
    the hardest schema here for llama-3.3 to emit, and two more sentences of budget prose tipped
    it over: run live against llama-3.3-70b, the outer Think produced

        <function=delegate{"branches":[{"goal":"Get the current time in Tokyo",...

    -- near-correct JSON in llama's text-wrapper form -- five attempts running, and the run died.
    The same request with no budget note attached succeeded. Every character of tool schema is
    re-sent on every call and costs accuracy on every OTHER tool too; the system message is plain
    text and costs the generator nothing. Same fix as `replan` and `advance`.
    """
    budget = getattr(conversation, "remaining_budget", None) or {}

    def listing(keys):
        return ", ".join(f"{budget[k]:.0f} {_BUDGET_LABELS[k]}"
                         for k in keys if budget.get(k) is not None)

    pooled, per_branch = listing(_POOLED_BUDGETS), listing(_PER_BRANCH_BUDGETS)
    notes = []
    if pooled:
        # divided by share of the work, not by branch count -- and not divided to the last unit,
        # because the caller still has to read the results and answer once they come back
        notes.append(
            f"\nOne shared pot remains, and every branch spends out of it: {pooled}. Divide it "
            "between branches in proportion to how much of the work each is doing, and keep some "
            "back for answering once they report."
        )
    if per_branch:
        notes.append(
            f"\nThese are a limit per branch, not a pot to divide, because branches run at the "
            f"same time: {per_branch}. Give each branch what its own sub-goal needs, up to that "
            "limit, leaving room for the work that follows."
        )
    return "".join(notes)


def _delegate_tool(conversation):
    """The `delegate` schema, kept as small as it can be while still describing every field.

    Every field still carries a description -- a field described only by its type is one the
    model fills in anyway, badly, and this schema's own history proves it: left undescribed,
    these drew `"broadcast": "false"` as a string into a boolean and invented `max_iterations: 1`
    for a two-step lookup, crippling a branch before it started.

    But the descriptions are now a handful of words each, and the prose that used to live here --
    how to write a goal, how to divide a budget -- moved to the system message. Measured: at 2898
    characters llama-3.3-70b could not emit the call at all, producing its
    `<function=delegate{...}` text form five attempts running until the run died. The schema is
    re-sent on every call and costs accuracy on every OTHER tool in the list too. Short and
    specific beat both bare types and full sentences.
    """
    return {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": "Split the work across independent sub-agents, each running its own agent loop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "persistent": {"type": "boolean",
                                   "description": "Keep branches alive for a later turn."},
                    "broadcast": {"type": "boolean",
                                  "description": "Let branches see each other's replies."},
                    "start": {"type": "string",
                              "description": "Id of the one branch to run first."},
                    "wait_for": {"type": "array", "items": {"type": "string"},
                                 "description": "Ids you need now; others keep running."},
                    "branches": {
                        "type": "array",
                        "description": "One entry per sub-agent.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Short name, e.g. 'pro'."},
                                "goal": {"type": "string",
                                         "description": "Self-contained instruction; it sees nothing else."},
                                "allow_fork": {"type": "boolean",
                                               "description": "True if this sub-goal needs splitting again."},
                                "explain": {"type": "boolean",
                                            "description": "True to see its working, not just its answer."},
                                "temperature": {"type": "number",
                                                "description": "Only when comparing identical goals."},
                                "top_p": {"type": "number",
                                          "description": "Only when comparing identical goals."},
                                "max_seconds": {"type": "number",
                                                "description": "Only if dividing a stated budget."},
                                "max_iterations": {"type": "integer",
                                                   "description": "Only if dividing a stated budget."},
                                "max_tool_calls": {"type": "integer",
                                                   "description": "Only if dividing a stated budget."},
                                "max_tokens": {"type": "integer",
                                               "description": "Only if dividing a stated budget."},
                            },
                            "required": ["goal"],
                        },
                    },
                },
                "required": ["branches"],
            },
        },
    }


def _advance_tool(conversation):
    # Only offered once persistent branches exist. Act has always known how to resume one
    # (_advance_branch); nothing could ever *decide* to, which left two things impossible:
    # taking turns among more than two debaters, and going back to a branch whose answer was
    # set aside. Both are the same missing decision -- pruning only ever filtered `pending`,
    # so a discarded candidate's branch was still sitting in conversation.branches, alive.
    #
    # Kept small for the same reason `delegate` and `replan` are: every character of schema
    # costs accuracy on every OTHER tool the model is holding at the same time. When to prefer
    # advancing, and which branch to pick, is said in the system message instead.
    listing = ", ".join(getattr(conversation, "branches", None) or {})
    return {
        "type": "function",
        "function": {
            "name": "advance",
            "description": "Give one existing sub-agent another turn, continuing from what it has already said.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "description": f"Which branch. One of: {listing}."},
                    "message": {"type": "string",
                                "description": "Optional. What to put to it before it continues."},
                },
                "required": ["branch"],
            },
        },
    }


def _running_goal(conversation, branch_id, limit=90):
    # what a still-running branch was actually asked to do. Its goal is the first user turn in
    # its own conversation, put there when the branch was seeded. Deciding whether to wait means
    # deciding whether the thing still running is the thing you now need, and a bare id says
    # nothing about that.
    branch = (getattr(conversation, "branches", None) or {}).get(branch_id)
    goal = next((m["content"] for m in getattr(branch, "messages", []) if m.get("role") == "user"),
                None)
    if not goal:
        return branch_id
    goal = " ".join(goal.split())
    return f"{branch_id} ({goal[:limit]}{'...' if len(goal) > limit else ''})"


def _replan_tool(conversation):
    # Offered only once a plan of goals exists, because there is nothing to revise before that.
    # Revising is a judgment made while carrying the plan out -- the executing loop is the only
    # thing that learns the plan was wrong -- so it arrives the same way delegating does, as one
    # more tool rather than as a component sitting in the loop replanning every turn whether or
    # not anything changed.
    # Kept deliberately tiny. The first version spelled out when to revise and how, and embedded
    # the current plan -- 1202 characters, twice the size of the two real tools put together.
    # With it offered, llama-3.3-70b rejected every single attempt to call ANY tool, emitting the
    # <function=...> text form five times running; with it removed the same request succeeded on
    # the first try. Prose about how to use a tool belongs in the system message, which costs the
    # generator nothing (see Think._build_messages).
    return {
        "type": "function",
        "function": {
            "name": "replan",
            "description": "Replace the plan of goals with a revised one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goals": {"type": "array", "items": {"type": "string"},
                              "description": "The complete revised plan, in order."},
                    "because": {"type": "string", "description": "What changed."},
                },
                "required": ["goals", "because"],
            },
        },
    }


def _collect_tool(conversation):
    # Offered only while something is actually still running. Deliberately tiny -- one array of
    # ids -- because it is offered alongside `delegate`, and every field added to a schema the
    # model must emit costs accuracy on all of them.
    #
    # The GOALS are listed, not just the ids. Deciding whether to wait means deciding whether the
    # thing still running is the thing you now need, and a bare id says nothing about that -- the
    # model would be choosing between "b3" and "b4" with no way to tell them apart.
    running = list(getattr(conversation, "running", None) or {})
    waiting = ", ".join(_running_goal(conversation, i) for i in running)
    ids = ", ".join(running)
    return {
        "type": "function",
        "function": {
            "name": "collect",
            "description": (
                "Wait for sub-agents that were left running in the background and read what they "
                f"produced. Still running: {waiting}. Use this the moment one of them holds "
                "something you need -- carrying on without it, or delegating the same work again, "
                "wastes the budget it is already spending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "branches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Which to wait for, by id. Any of: {ids}.",
                    },
                },
                "required": ["branches"],
            },
        },
    }


class Fork:
    # A fork the CALLER declares, rather than one a model decides. It emits exactly the decision
    # Think produces for `delegate` -- same "fork" type, same fields -- so Act needs no new branch
    # to handle it. The precedent is Observe(event=...): a component handed fixed content that
    # emits it instead of computing one.
    #
    # It exists because some topologies are the experiment DESIGN, not a judgment call. Debate and
    # self-consistency used to be set up by writing a paragraph asking the model to "delegate to
    # exactly 2 persistent branches that broadcast to each other" -- configuration written in
    # English and parsed by a model that emits `delegate` malformed about a quarter of the time.
    # Self-consistency's prompt went as far as pleading "Do not change the wording between
    # branches", which a list comprehension simply guarantees.
    #
    # Think keeps `delegate`. Deciding how to split an open-ended request is real judgment; being
    # handed the shape of an experiment is not.
    def __init__(self, branches, persistent=False, broadcast=False, start=None, wait_for=None):
        # start: run only this branch now, leaving the others seeded but unrun, so an exchange
        # opens with one speaker. Who goes next is a judgment and belongs to a Think with
        # allow_advance=True -- which is why this is `start` and not an `order` list. An order
        # would decide every turn up front, taking back the very decision `advance` exists to make.
        if start is not None:
            ids = [b.get("id") for b in branches]
            if start not in ids:
                raise ValueError(f"start={start!r} is not one of the branch ids {ids}")
            if not persistent:
                raise ValueError("start= requires persistent=True: the branches that did not "
                                 "start have to survive for a later turn to advance them")
        if wait_for is not None:
            unknown = [i for i in wait_for if i not in [b.get("id") for b in branches]]
            if unknown:
                raise ValueError(f"wait_for names branches that do not exist: {unknown}")
        self.branches = branches
        self.persistent = persistent
        self.broadcast = broadcast
        self.start = start
        # which branches the caller needs before the loop carries on. The rest keep running and
        # are collected later. Declared here when the topology is fixed; decided by the Think,
        # via the same field on `delegate`, when it is not.
        self.wait_for = wait_for

    def run(self, conversation):
        branches = self.branches
        return {
            "role": "decision",
            "type": "fork",
            "content": "Forked into: " + "; ".join(b["goal"] for b in branches),
            "persistent": self.persistent,
            "broadcast": self.broadcast,
            "start": self.start,
            "wait_for": self.wait_for,
            "branches": branches,
        }


class Think:
    # the default, general-purpose Think: decides a tool call, a delegation (fork), or a final
    # answer, all through the one native tool-calling mechanism -- "delegate" is just one more
    # tool in the list, so no separate ToolCallThink/ForkThink is needed for this. Sampling
    # (model/temperature/top_p) is per-instance, so a forked branch can run with its own settings.
    # retries defaults to 5 rather than 3 from a measured failure rate, not superstition:
    # emitting a `delegate` call (an array of objects, several fields each) makes Groq's
    # llama-3.3-70b produce malformed tool-call syntax roughly a quarter of the time -- truncated
    # JSON, a broken <function=...> wrapper, or "false" as a string where a boolean belongs. Each
    # comes back as a BadRequestError and is retried with the real reason (see run()). Ordinary
    # single-tool calls almost never need more than one attempt.
    def __init__(self, tools=None, retries=5, model=DEFAULT_MODEL, temperature=None, top_p=None,
                 allow_fork=True, allow_advance=None, allow_replan=False, explain=True):
        self.tools = tools or []
        self.retries = retries
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        # leaf-level branches doing one concrete task shouldn't be tempted to delegate further --
        # only Thinks that should be able to fork (usually the outer/orchestrating one) need this.
        self.allow_fork = allow_fork
        # Starting branches and continuing them are two different powers, and conflating them made
        # one useful component impossible to build: a Think whose ONLY job is picking who speaks
        # next. Where the branches were declared rather than decided (see Fork), delegating again
        # is exactly the wrong move -- but turning allow_fork off used to take `advance` with it.
        # Defaults to allow_fork, so every existing call site behaves exactly as before.
        self.allow_advance = allow_fork if allow_advance is None else allow_advance
        # only the loop actually executing a plan of goals should be able to revise it, and only
        # that loop ever learns the plan was wrong. Off by default: a Think with no plan in front
        # of it has nothing to replan, and offering the tool anyway is one more thing to get wrong.
        self.allow_replan = allow_replan
        # ask the model to say why it is calling a tool, kept as decision["reasoning"]. Worth it
        # for a Think making real choices; actively harmful for a leaf branch, which exists to
        # return a value -- prompted to narrate, branches answer their goal with an explanation
        # of what they did instead of the result, and the caller gets nothing usable back.
        self.explain = explain

    def _can_fork(self, conversation):
        # `delegate` is withheld for exactly one turn after a fork's results come back, and this
        # replaces what used to be a whole second Think declared with allow_fork=False.
        #
        # The failure it prevents is specific and was measured: handed a pile of branch results,
        # the model reaches for `delegate` again -- and that schema is the hardest thing here for
        # it to emit, so it burns every retry on malformed syntax and the run dies. The old fix
        # was to declare a separate finishing Think, which meant every forking pattern carried a
        # near-duplicate component and a repeat_from pointing past it.
        #
        # Withheld for ONE turn, not for the whole run: forking twice for genuinely different
        # sub-tasks later on is legitimate, and only the immediate return pass is the problem.
        if not self.allow_fork:
            return False
        last = conversation.messages[-1] if conversation.messages else None
        return not (last and last.get("source") == "branches")

    def _sampling_kwargs(self):
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs

    def _build_messages(self, conversation, explain=None):
        # reconstructs the REAL native tool-calling shape (an assistant message carrying a
        # structured tool_calls entry, followed by a tool-role reply with a matching id) --
        # rather than flattening everything to prose. Models are trained to recognize "I already
        # did this" from exactly this structure; narrated text of the same information turned out
        # unreliable (the model would sometimes just call the same tool again). A "decision" of
        # type tool_call is always immediately followed by exactly one "tool" observation in our
        # own conversation, so pairing them up by position is enough -- no id needs to be stored.
        #
        # What DOES get carried across is the model's own rationale for the call, kept on the
        # decision by run() and replayed here as the assistant message's content -- so a later
        # turn sees why a call was made, not merely that it was.
        # Whether delegating is worth it is a judgment, so it is stated as guidance here rather
        # than forced structurally (withholding tools from the caller would leave it no choice,
        # which is not the same thing as deciding). Only added when this Think can actually fork.
        when_to_delegate = (
            "\nDelegating costs a whole extra agent loop per branch. Prefer it for independent "
            "sub-goals that each need several steps of their own, for answering one question "
            "several times to see if the answers agree, or for arguing opposing positions. For "
            "work you could finish in a couple of tool calls, just make the calls -- unless you "
            "were explicitly asked to split the work."
            # The goal-writing contract. This used to be the `goal` field's own description --
            # the longest string in the schema -- but a branch is seeded with its goal and
            # nothing else, so writing it well IS the delegation, and that is a lot to say.
            # Said here it costs the generator nothing.
            "\nEach branch's goal must be a complete, self-contained instruction for an agent "
            "with NO other context: it will not see this conversation or any other branch's "
            "goal. Include everything it needs, such as the specific question or values "
            "involved. Never write it as if the reader already knows the broader request, and "
            "never refer to 'the other task' or another branch."
            + budget_guidance(conversation)
        ) if self._can_fork(conversation) else ""

        # Offered both tools, the model reaches for `delegate` even when a branch that has
        # already done the work is sitting right there -- so say which to prefer. A new branch
        # starts from nothing; advancing keeps everything the existing one worked out. When this
        # Think cannot fork at all that comparison is meaningless, so the clause drops out and
        # what is left is the only decision it still has: which branch goes next.
        when_to_advance = (
            "\nBranches already exist: " + ", ".join(getattr(conversation, "branches", None) or {}) +
            ". To continue an exchange, press a point, or revisit an answer you set aside, use "
            "`advance` on one of them" + (
                " rather than delegating again -- a new branch would start with none of what that "
                "one has already established. Delegate only for genuinely new sub-goals."
                if self.allow_fork else
                ", choosing whichever one is most worth hearing from next."
            )
            # An exchange has no natural end, and without this the Think advances a branch every
            # turn until the budget stops it -- a debate that ends on "Stopping: reached the
            # maximum of 10 iterations" rather than on a conclusion. Advancing needs a way out
            # for the same reason a search does.
            + " Once the exchange has covered what matters and further turns would only repeat "
              "it, stop advancing and answer instead, in your own words, drawing on what the "
              "branches established."
        ) if self.allow_advance and getattr(conversation, "branches", None) else ""

        # Groq returns a bare tool call with no content at all unless asked, so the rationale
        # has to be requested -- together with a guard, because asked for one the model starts
        # explaining itself in its FINAL answers too, returning "the function call was necessary
        # to get the conversion" where the number belongs.
        # Asking for a rationale is also what invites the model to write prose and then render
        # the tool call as text, which Groq rejects. So a retry after a rejection drops the
        # request entirely (explain=False) rather than asking again for the thing that broke it.
        explain = (
            "\nBefore each tool call, briefly say why."
            "\nWhen you are answering instead of calling a tool, give the answer itself and "
            "nothing about your process."
        ) if (self.explain if explain is None else explain) else ""

        # everything the `replan` schema used to say, moved here where it costs nothing to emit
        plan = latest_plan(conversation, kind="goals") if self.allow_replan else None
        when_to_replan = (
            f"\nThe plan you are working to:\n{plan['content']}\n"
            "If what you have learned changes what is left to do -- a goal turned out to be "
            "unnecessary, or the work needs a step nobody knew about -- call `replan` with the "
            "WHOLE plan as it should now stand; it replaces the old one rather than adding to "
            "it. Copy every goal you are not changing word for word, including ones already "
            "achieved, and put new or changed goals at the point in the sequence where they "
            "belong. Otherwise just carry on."
        ) if plan is not None else ""

        messages = [{
            "role": "system",
            "content": (
                "If the user's request has already been fully answered by a tool result shown "
                "below, respond in plain text with that answer -- do not call the tool again."
                # Results that answer the request get reacted to rather than reported. Tree of
                # Thoughts, handed the surviving candidate, replied "Great choice! SwiftStride
                # is a catchy, beginner-friendly name" -- congratulating somebody on a decision
                # the loop had made itself. The answer is in there, wrapped in a reply to nobody.
                "\nWhen results already contain the answer, give the answer itself. Do not "
                "congratulate anyone on it, comment on how good it is, or offer further help."
                + explain
                + when_to_delegate
                + when_to_advance
                + when_to_replan
            ),
        }]
        pending_call_id = None
        for m in conversation.messages:
            role = m["role"]
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": m["content"]})
            elif role == "decision" and m.get("type") == "tool_call":
                pending_call_id = f"call_{len(messages)}"
                messages.append({
                    "role": "assistant",
                    # the model's own one-line rationale for this call, replayed so later turns
                    # see why it did what it did and not just that it did it
                    "content": m.get("reasoning"),
                    "tool_calls": [{
                        "id": pending_call_id,
                        "type": "function",
                        "function": {"name": m["tool"], "arguments": m["content"]},
                    }],
                })
            elif role == "tool" and pending_call_id is not None:
                messages.append({"role": "tool", "tool_call_id": pending_call_id, "content": m["content"]})
                pending_call_id = None
            elif role == "tool":
                # A tool observation with no tool_call before it: a fork's merged branch results,
                # or a search round's survivors. These used to be narrated as the agent's own past
                # turn, which put an ASSISTANT message last in the request -- and a model handed a
                # trailing assistant message continues it rather than replying. Tree of Thoughts
                # answered " This approach seems preferable as it fosters a supportive
                # environment", a sentence fragment picking up mid-thought from the pruning
                # rationale above it.
                #
                # They are things that happened TO the agent, not things it said, so they go in as
                # user turns -- which is also what ends the turn and asks for a fresh reply.
                # Labelled as plainly as possible, because `user` is the only role that both
                # ends the turn AND is not the agent's own voice -- and the model will otherwise
                # read the contents as something the USER said. Sent as bare "Results:", Tree of
                # Thoughts replied "Glad you like StrideRise!", thanking the user for praise
                # nobody had given. It is neither the agent's turn nor the user's, and the text
                # has to say so, since the role cannot.
                # The wording is load-bearing twice over. Bare "Results:" and Tree of Thoughts
                # replied "Glad you like StrideRise!", reading the block as praise from the user.
                # Calling them "results for you to judge" then produced a critique of a candidate
                # instead of an answer -- it did what the label asked. This says whose words they
                # are and nothing about what to do with them.
                messages.append({"role": "user", "content": (
                    "[Results of work you delegated. Not a message from the user.]\n"
                    + m["content"])})
            else:
                # plan and critique messages, and fork decisions -- genuinely the agent's own
                # output, so they stay as its past turns.
                messages.append({"role": "assistant", "content": m["content"]})
        return messages

    def run(self, conversation):
        base = self._build_messages(conversation)
        messages = list(base)
        all_tools = list(self.tools)
        if self._can_fork(conversation):
            all_tools.append(_delegate_tool(conversation))
        # only once there is something to advance -- offering it against an empty branch
        # list would invite a call naming a branch that does not exist
        if self.allow_advance and getattr(conversation, "branches", None):
            all_tools.append(_advance_tool(conversation))
        # nothing left running means nothing to collect, so the tool is not offered at all
        if getattr(conversation, "running", None):
            all_tools.append(_collect_tool(conversation))
        # nothing to revise until a plan of goals exists
        if self.allow_replan and latest_plan(conversation, kind="goals") is not None:
            all_tools.append(_replan_tool(conversation))
        for attempt in range(self.retries):
            try:
                # `tools` is omitted entirely when there are none, rather than sent as an empty
                # list -- some components (ReWOO's solver) deliberately hold no tools so that
                # answering is the only move left, and an empty array is not a valid request.
                response = client.chat.completions.create(
                    model=self.model, messages=messages,
                    **({"tools": all_tools} if all_tools else {}),
                    **self._sampling_kwargs()
                )
            except BadRequestError as e:
                # Groq rejects a malformed tool call and tells you exactly why -- which field had
                # the wrong type, or the literal broken text it produced. Handing that back beats
                # a generic "try again", which asks the model to fix something without saying
                # what was wrong; it re-reads its own context, sees nothing new, and reproduces
                # the same mistake. Same principle as DebugThink, which gets the real schema and
                # the real validation error rather than a nudge.
                #
                # Rebuilt each time rather than appended to, so the model sees one clear
                # correction instead of a growing stack of past failures -- and rebuilt WITHOUT
                # the rationale request, since asking for prose alongside a tool call is what
                # produces the text-wrapper form Groq rejects.
                base = self._build_messages(conversation, explain=False)
                messages = [*base, {
                    "role": "user",
                    "content": tool_call_failure(e) or (
                        "That call could not be processed. Try again, calling one of the "
                        "available tools correctly."
                    ),
                }]
                continue

            message = response.choices[0].message
            if not message.tool_calls:
                if not message.content:
                    messages.append({"role": "assistant", "content": message.content or ""})
                    messages.append({"role": "user", "content": "Write out your final answer as text."})
                    continue
                return {"role": "decision", "type": "respond", "content": message.content}

            call = message.tool_calls[0]
            if call.function.name == "delegate":
                args = json.loads(call.function.arguments)
                branches = args["branches"]
                persistent = args.get("persistent", False)

                # `start` is checked here rather than in Act, for the same reason `advance`'s
                # branch id is: Act indexes with it, and a model naming a branch it did not
                # create would surface as a ValueError from inside execution. An unusable value
                # is dropped, which just falls back to running everything at once.
                start = args.get("start")
                if start is not None and start not in [b.get("id") for b in branches]:
                    start = None

                # same guarantee as `start`: Act matches these against branch ids, so unknown
                # names are dropped here. An empty list after filtering means wait for all --
                # never wait for none, which would leave the caller reasoning about no results.
                known_ids = [b.get("id") for b in branches]
                wait_for = args.get("wait_for")
                if wait_for is not None:
                    wait_for = [i for i in dict.fromkeys(wait_for) if i in known_ids] or None
                # holding branches back only means something if they survive to be advanced, so
                # asking for an opener implies persistence. Coerced rather than rejected: the
                # intent is unambiguous, and failing the call would cost a whole retry to
                # re-derive a decision the model already got right.
                if start is not None:
                    persistent = True

                return {
                    "role": "decision",
                    "type": "fork",
                    "content": "Forked into: " + "; ".join(b["goal"] for b in branches),
                    "persistent": persistent,
                    "broadcast": args.get("broadcast", False),
                    "start": start,
                    "wait_for": wait_for,
                    "branches": branches,
                }

            if call.function.name == "replan":
                args = json.loads(call.function.arguments)
                goals = [g for g in (args.get("goals") or []) if isinstance(g, str) and g.strip()]
                if not goals:
                    # an empty revision would wipe the plan and leave the loop steering by
                    # nothing, which is worse than the plan it was unhappy with
                    messages = [*base, {"role": "user", "content": (
                        "A replan must contain the whole revised plan, with at least one goal. "
                        "Call replan again with every goal it should now have, or carry on."
                    )}]
                    continue
                # a plan, not a decision -- exactly what PlannerThink returns, and for the same
                # reason: there is nothing for Act to execute. A goal is not executable; working
                # out how to reach one is reasoning, which is the next Think's job.
                #
                # This is only safe because Act now crosses a decision off when it carries it
                # out. Before that, a Think returning a non-decision left its previous decision
                # looking unfinished, and the next Act would find and re-run it.
                because = args.get("because") or "the plan needed revising"
                return {
                    "role": "plan",
                    "kind": "goals",
                    "content": f"Replanning: {because}\nGoals:\n"
                               + "\n".join(f"{i}. {g}" for i, g in enumerate(goals)),
                    "steps": [{"goal": g} for g in goals],
                }

            if call.function.name == "collect":
                args = json.loads(call.function.arguments)
                running = list(getattr(conversation, "running", None) or {})
                # Act waits on these futures by id, so an invented one has to be dropped here.
                # An empty result means the model named nothing real -- collecting everything is
                # the safe reading, since leaving branches running is what costs.
                wanted = [i for i in dict.fromkeys(args.get("branches") or []) if i in running]
                return {
                    "role": "decision",
                    "type": "collect",
                    "content": "Waiting for: " + ", ".join(wanted or running),
                    "branches": wanted or running,
                    "reasoning": message.content or None,
                }

            if call.function.name == "advance":
                args = json.loads(call.function.arguments)
                branch = args.get("branch")
                known = list(getattr(conversation, "branches", None) or {})
                if branch not in known:
                    # tolerate a near miss on casing or spacing before making it retry
                    normalise = lambda s: "".join(str(s).lower().split())
                    branch = next((k for k in known if normalise(k) == normalise(branch)), branch)
                if branch not in known:
                    # a name that doesn't exist would raise a KeyError deep inside Act; say so
                    # here instead and let the retry loop take another turn at it
                    messages = [*base, {"role": "user", "content": (
                        f"There is no branch called {branch!r}. The existing branches are: "
                        f"{', '.join(known)}. Choose one of those, or answer instead."
                    )}]
                    continue
                return {
                    "role": "decision",
                    "type": "advance_branch",
                    "branch": branch,
                    "message": args.get("message"),
                    "content": f"Continuing [{branch}]"
                               + (f": {args['message']}" if args.get("message") else ""),
                }

            decision = {
                "role": "decision",
                "type": "tool_call",
                "tool": call.function.name,
                "content": call.function.arguments,
            }
            # The model's own account of why it made this call, which the system message above
            # asks for. Without that instruction Groq returns a bare tool call and this is empty.
            # It is real content -- replayed on later turns, and readable by any Think that comes
            # after -- not just a debugging aid.
            if message.content:
                decision["reasoning"] = message.content.strip()
            return decision
        raise ValueError(f"Failed to get a valid decision after {self.retries} attempts")


class PlanStep(BaseModel):
    tool: str
    args: dict[str, str]


class Plan(BaseModel):
    steps: list[PlanStep]


class ContinueOrFinish(BaseModel):
    done: bool
    final_answer: str = ""
    additional_steps: list[PlanStep] = []


def latest_plan(conversation, kind="tools"):
    # The *latest* plan only. A replan appends a new plan message, and it supersedes the old
    # one completely -- the previous round's steps have already run.
    #
    # kind="tools" -> a plan of tool calls, which Graph can execute on its own.
    # kind="goals" -> a plan of things to achieve; reaching one is reasoning, so only a Think
    # can act on those and the mechanical machinery must leave them alone.
    # role == "plan" specifically, not merely "has steps". Being a plan is a fact about the role,
    # not about which keys happen to be set -- and other message kinds do carry a "steps" field.
    plan = None
    for m in conversation.messages:
        if m.get("role") == "plan" and (kind is None or m.get("kind", "tools") == kind):
            plan = m
    return plan


_STEP_REF = re.compile(r"\$(?:step)?(\d+)(?:\.(\w+))?")


def resolve_text(text, outputs):
    # Substitutes "$stepN"/"$stepN.field" inside ordinary prose. The planner is taught that
    # syntax so it can chain steps, and it then reaches for it in the FINAL ANSWER too: run
    # live, LLM Compiler replied "The current time in Tokyo is $step1 and 100 degrees Fahrenheit
    # is equal to $step0 degrees Celsius." Every component was satisfied -- it was an assistant
    # message, the run terminated -- and the answer was useless.
    #
    # Asking the model not to do it is necessary but not sufficient, so this makes it true.
    def replace(match):
        index, field = int(match.group(1)), match.group(2)
        if index >= len(outputs):
            return match.group(0)
        value = outputs[index]
        if field:
            try:
                value = json.loads(value)[field] if isinstance(value, str) else value[field]
            except (ValueError, TypeError, KeyError, IndexError):
                return match.group(0)
        return str(value)

    return _STEP_REF.sub(replace, text or "")


def step_dependencies(step):
    # which earlier step indices this step's args reference -- read straight off the plan's own
    # "$stepN"/"$stepN.field" syntax. A step with an empty set here has nothing to wait for, and
    # that is exactly what lets Graph start it immediately.
    deps = set()
    for value in step["args"].values():
        match = _STEP_REF.fullmatch(value) if isinstance(value, str) else None
        if match is not None:
            deps.add(int(match.group(1)))
    return deps


def resolve_args(step, results):
    # results: {step_index: raw_content}. A step is only ever resolved once every index in
    # step_dependencies(step) is already a key in results -- the caller's job, not this one's.
    args = {}
    for key, value in step["args"].items():
        match = _STEP_REF.fullmatch(value) if isinstance(value, str) else None
        if match is None:
            args[key] = value
            continue
        index, field = match.groups()
        raw = results[int(index)]
        args[key] = json.loads(raw)[field] if field is not None else raw
    return args


class Goals(BaseModel):
    goals: list[str]


class PlannerThink(Think):
    # Two kinds of plan, and the difference is what a step actually says.
    #
    # goals=False (default): a plan of TOOL CALLS, every argument decided up front -- either a
    # literal already known from the request, or "$stepN" referencing an earlier step's
    # not-yet-known result. Later arguments are then pure substitution (Observe's job), never a
    # fresh decision. Called again after a plan finishes, it switches to deciding whether the
    # request is answered yet or another round of steps is needed.
    #
    # goals=True: a plan of things to ACHIEVE, in plain language, naming no tools at all. It
    # produces no decision and nothing mechanical can advance it -- working out how to reach a
    # goal is reasoning, so the next Think in the loop reads the goals and decides for itself
    # whether to pursue them directly or delegate them. Planning and delegating stay separate
    # decisions; this component only ever makes the first one.
    def __init__(self, tools=None, outputs=None, goals=False):
        self.tools = tools or []
        self.outputs = outputs or {}
        self.goals = goals

    def _tool_descriptions(self):
        return "\n".join(
            f"- {t['function']['name']}({', '.join(t['function']['parameters']['properties'])}) "
            f"returns: {self.outputs.get(t['function']['name'], 'a single value')}"
            for t in self.tools
        )

    def run(self, conversation):
        if self.goals:
            return self._make_goals(conversation)
        if latest_plan(conversation) is None:
            return self._make_plan(conversation)
        return self._continue_or_finish(conversation)

    def _make_goals(self, conversation):
        request = conversation.messages[-1]["content"]
        available = (f"\nTools that will be available to whoever carries these out:\n"
                     f"{self._tool_descriptions()}\n") if self.tools else ""
        result = extract(
            f"A user asked: {request!r}\n{available}\n"
            "Break this into the smallest number of goals that between them fully answer it. "
            "Write each goal as a complete, self-contained instruction that states everything "
            "needed to carry it out -- whoever works on it may see nothing else. Describe WHAT "
            "to achieve, never which tool to call or how. If the request needs no breaking "
            "down, return a single goal.\n"
            f"Return JSON matching this schema:\n{Goals.model_json_schema()}",
            Goals,
        )
        # returned, not appended -- and it is not a "decision", so Act has nothing to execute
        # and the next component is free to make up its own mind about these.
        return {
            "role": "plan",
            "kind": "goals",
            "content": "Goals:\n" + "\n".join(f"{i}. {g}" for i, g in enumerate(result.goals)),
            "steps": [{"goal": g} for g in result.goals],
        }

    def _make_plan(self, conversation):
        request = conversation.messages[-1]["content"]
        plan = extract(
            f"A user asked: {request!r}\n"
            f"Available tools, with their exact argument names and what they return:\n{self._tool_descriptions()}\n\n"
            "Produce, in order, every tool call needed to fully answer this request. For each "
            "step, give the tool name and a value for every one of that tool's argument names "
            "listed above, as strings -- the args object's keys must be exactly those argument "
            "names, nothing else. If an "
            "argument's value is already known from the request, use that literal value. If an "
            "argument's value can only be known after an earlier step's tool actually runs, "
            'reference it with exactly the string "$stepN" (0-indexed, e.g. "$step0" for the '
            'first step\'s result) instead of guessing -- never abbreviate this to "$0" or '
            'anything else. If that earlier step returns a single value, "$stepN" alone is the '
            'whole value. If it returns a JSON object with named fields (see "returns" above), '
            'reference just the one field you need with "$stepN.fieldname" -- never forward the '
            "whole object when only one field is actually needed. "
            "Return JSON matching the given schema.",
            Plan,
        )
        # Only the plan. No first-step decision: Graph reads the plan itself and starts every
        # step whose references are already satisfied, so a decision here would be executed by
        # nobody and would sit in the transcript as noise.
        return {
            "role": "plan",
            "kind": "tools",
            "content": plan.model_dump_json(),
            "steps": [s.model_dump() for s in plan.steps],
        }

    def _continue_or_finish(self, conversation):
        start_index = len(getattr(conversation, "step_outputs", []))
        result = extract(
            f"Conversation so far:\n{render(conversation)}\n\n"
            f"Available tools, with their exact argument names and what they return:\n{self._tool_descriptions()}\n\n"
            "All previously planned tool calls have finished running. Decide whether the "
            'conversation so far fully answers the user\'s original request. If it does, set '
            '"done" to true and write the final answer in "final_answer" -- in plain words with '
            'the actual values written out, never leaving a "$stepN" reference in it. If it does not, set '
            '"done" to false and list the additional tool call steps still needed, in '
            f'"additional_steps", the same format as before. So far {start_index} tool call(s) '
            f'have completed, referenceable as "$step0" through "$step{start_index - 1}". Any '
            "new step you add now runs in order starting immediately, so if a later new step "
            f'needs an earlier new step\'s result, reference it starting from "$step{start_index}" '
            "onward, continuing that same count. Return JSON matching the given schema.",
            ContinueOrFinish,
        )
        if result.done:
            # said in the prompt above AND enforced here, because the model reaches for the
            # reference syntax in prose too and a "$step0" in the final answer helps nobody
            outputs = getattr(conversation, "step_outputs", None) or []
            return {"role": "decision", "type": "respond",
                    "content": resolve_text(result.final_answer, outputs)}
        # not done: a fresh plan, which supersedes the last one. Graph runs whichever plan is
        # newest, so nothing else has to know a round just ended.
        return {
            "role": "plan",
            "kind": "tools",
            "content": json.dumps([s.model_dump() for s in result.additional_steps]),
            "steps": [s.model_dump() for s in result.additional_steps],
        }


class Score(BaseModel):
    index: int = Field(description="Which candidate, by its number in the list above.")
    score: float = Field(description="How good it is, from 0 to 10.")


class Selection(BaseModel):
    keep_indices: list[int]
    # one score per candidate, in the order they were listed. Without these, a search can reopen
    # an abandoned branch but has nothing to tell it WHICH -- the whole point of keeping a tree
    # is knowing where the promising unexplored lines are. Asked for as a plain positional list
    # because that is the shape hardest to get wrong.
    #
    # LABELLED, not positional, and that took two live runs to get right. As a bare parallel
    # array -- "one score per candidate, in the order listed" -- the model returned `[869.0]`:
    # one number for three candidates, and not out of 10 either. The length check then discarded
    # it, so nothing was recorded and backtracking had nothing to steer by.
    #
    # Saying which candidate a score belongs to makes a short or reordered list still usable,
    # and a wrong index detectable. Same lesson as labelling branches by id rather than by goal:
    # a positional convention is one the model has to hold in its head, and it doesn't.
    #
    # Still optional. Choosing what survives is the critical path; scores only enrich a later
    # backtrack, and a model that omits them should not fail the whole selection.
    scores: list[Score] = Field(
        default=[],
        description="A score for every candidate, including the ones being discarded.")
    reasoning: str


class SelectThink(Think):
    # pruning: reads the RAW, not-yet-merged results of a fork (still sitting in
    # conversation.pending as a list) and decides which survive. Produces a "prune" decision;
    # Act applies it mechanically (drop everything not in keep_indices) -- SelectThink judges,
    # it doesn't touch the list itself.
    def __init__(self, keep=1):
        self.keep = keep

    def run(self, conversation):
        # only a fork leaves a LIST in pending. A Think that declined to fork and made an
        # ordinary tool call leaves a single effect dict there instead, and iterating that
        # yields its keys -- strings, which have no .get(). Nothing to select between, so say so.
        pending = getattr(conversation, "pending", None)
        candidates = pending if isinstance(pending, list) else []
        listing = "\n".join(f"{i}. [{c.get('goal', '(no goal)')}] {c['content']}" for i, c in enumerate(candidates))
        result = extract(
            f"Conversation so far:\n{render(conversation)}\n\n"
            f"Candidate results from {len(candidates)} independent branches:\n{listing}\n\n"
            f"Score each against the original goal out of 10, giving one score per candidate in "
            f"the order listed above, and pick the best {self.keep} to continue exploring, by "
            "their index in the list above. Discard the rest.\n"
            # the schema has to be shown, not just referred to -- without it the model invents its
            # own field names (e.g. a list of scored items) and only recovers via extract()'s
            # retry loop, if at all. Same reason DebugThink.fix() embeds its schema.
            f"Return JSON matching this schema:\n{Selection.model_json_schema()}",
            Selection,
        )
        # the model sometimes returns an index that doesn't exist, or more than it was asked for.
        # Act indexes straight into pending, so the guarantee that these are real indices has to
        # be made here, where the candidate list is actually in scope.
        valid = [i for i in dict.fromkeys(result.keep_indices) if 0 <= i < len(candidates)]
        # scores are recorded against branch ids, not indices: indices are per-round and
        # meaningless once the round is over, whereas an id stays valid for as long as the branch
        # does. A discarded branch keeps its score, which is exactly what backtracking needs --
        # the loser of round one may still outscore everything in round three.
        # whatever came back that points at a real candidate, clamped to the stated range -- the
        # model has returned 869.0 for a score out of 10. Partial is fine: a score for two of
        # three candidates is still two more than none.
        scores = {}
        for s in result.scores:
            if 0 <= s.index < len(candidates) and candidates[s.index].get("id"):
                scores[candidates[s.index]["id"]] = max(0.0, min(10.0, s.score))
        # Falling back to "the first N" was the only option when nothing else was known. It is a
        # poor one: a live run returned keep_indices=[20] against three candidates -- garbage --
        # while scoring them 8, 6 and 9. The first two would have been kept and the best thrown
        # away. Where scores survived, rank by them; only with neither is order all there is.
        usable = [(s.index, s.score) for s in result.scores if 0 <= s.index < len(candidates)]
        ranked = [i for i, _ in sorted(usable, key=lambda pair: -pair[1])]
        keep = (valid[: self.keep] or ranked[: self.keep]
                or list(range(min(self.keep, len(candidates)))))
        return {
            "role": "decision",
            "type": "prune",
            "content": result.reasoning,
            "keep_indices": keep,
            "scores": scores,
        }


class Thoughts(BaseModel):
    thoughts: list[str]


class Merge(BaseModel):
    worth_merging: bool
    parents: list[str] = []
    thought: str = ""


class ExpandThink:
    # Grows a search one round. The frontier -- whatever SelectThink last kept -- is expanded by
    # asking the model, once per surviving branch, for `thoughts` ways to continue THAT branch.
    #
    # Which parent each new thought belongs to is not a decision. It is settled by which branch's
    # working was in front of the model when the thought was written, so it is recorded rather
    # than chosen. An earlier version let the model name the parent alongside the thought, which
    # is a choice masquerading as a fact: it could write a thought inspired by one line and label
    # it another, and nothing could detect that, because only the model knew where the thought
    # came from. Asking per parent makes a wrong parent impossible instead of merely unlikely.
    # It is also what the Tree of Thoughts paper does -- its generator is handed one state.
    #
    # The one thing that cannot be structural is a MERGE. "Combine these two" is a real decision;
    # nothing about the loop can work out which thoughts are worth joining. So merging is a
    # separate, explicit question, asked only when max_parents allows it -- which mirrors Graph
    # of Thoughts, where Generate takes one thought and only Aggregate takes several.
    def __init__(self, thoughts=2, max_parents=1, model=DEFAULT_MODEL, temperature=None,
                 max_workers=8):
        # thoughts: the branching factor -- how many ways each surviving line is continued. A
        # resource decision belonging to whoever declared the loop, exactly like SelectThink's
        # `keep`. Together they are the beam: keep b, expand each into k.
        self.thoughts = thoughts
        # 1 means no merge is ever offered, so every thought has exactly one parent and the shape
        # can only split -- a tree, i.e. Tree of Thoughts. Higher allows a merge of up to that
        # many frontier branches, which is the one move a tree cannot make -- Graph of Thoughts.
        self.max_parents = max_parents
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers

    def _ask(self, prompt):
        return extract(f"{prompt}\n\nReturn JSON matching this schema:\n"
                       f"{Thoughts.model_json_schema()}", Thoughts).thoughts[: self.thoughts]

    def _from_parent(self, conversation, parent_id):
        parent = conversation.branches[parent_id]
        answer = next((m["content"] for m in reversed(parent.messages)
                       if m.get("role") == "assistant"), "")
        working = transcript({"label": parent_id, "content": answer, "conversation": parent})
        return [{"parents": [parent_id], "thought": t} for t in self._ask(
            f"The overall task:\n{render(conversation)}\n\n"
            f"One line of work so far:\n{working}\n\n"
            f"Give {self.thoughts} distinct ways to continue THIS line -- each a complete, "
            "self-contained instruction for an agent that will see the working above."
        )]

    def _merge(self, conversation, frontier):
        blocks = []
        for i in frontier:
            branch = conversation.branches[i]
            answer = next((m["content"] for m in reversed(branch.messages)
                           if m.get("role") == "assistant"), "")
            blocks.append(f"[{i}] {answer}")
        result = extract(
            f"The overall task:\n{render(conversation)}\n\n"
            "Separate lines of work so far:\n" + "\n".join(blocks) + "\n\n"
            f"Is any group of {self.max_parents} or fewer of these worth combining into a single "
            "stronger idea? Only say yes if combining genuinely beats continuing them separately."
            f"\n\nReturn JSON matching this schema:\n{Merge.model_json_schema()}", Merge)
        parents = [p for p in dict.fromkeys(result.parents) if p in frontier][: self.max_parents]
        if not result.worth_merging or len(parents) < 2 or not result.thought:
            return []
        return [{"parents": parents, "thought": result.thought}]

    def run(self, conversation):
        # the frontier is what the last prune kept -- set by Act, not read out of pending, which
        # Observe has already consumed by the time this component runs again on the next round.
        frontier = list(getattr(conversation, "frontier", None) or [])

        if not frontier:
            # round one: no parents yet, so the roots are generated straight from the task
            assignments = [{"parents": [], "thought": t} for t in self._ask(
                f"The task:\n{render(conversation)}\n\n"
                f"Give {self.thoughts} genuinely different approaches to it -- each a complete, "
                "self-contained instruction for an agent that will see nothing but that sentence."
            )]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                per_parent = pool.map(lambda i: self._from_parent(conversation, i), frontier)
            assignments = [a for group in per_parent for a in group]
            if self.max_parents and self.max_parents > 1 and len(frontier) > 1:
                assignments += self._merge(conversation, frontier)

        return {
            "role": "decision",
            "type": "expand",
            "content": "; ".join(
                (f"from {'+'.join(a['parents'])}: " if a["parents"] else "") + a["thought"]
                for a in assignments),
            "assignments": assignments,
        }


class DebugThink:
    # not a Think: no run(conversation) -- Act owns this privately, it's never declared into a Runtime's loop
    def __init__(self, schemas):
        self.schemas = schemas

    def fix(self, tool_name, args, error, attempts):
        schema = self.schemas[tool_name]
        history = "\n".join(f"- args {a['args']} failed with: {a['error']}" for a in attempts)
        corrected = extract(
            f"Calling the tool `{tool_name}` with arguments {args} failed with this error:\n{error}\n\n"
            f"Earlier failed attempts on this same call:\n{history}\n\n"
            f"Fix the arguments to match this JSON schema:\n{schema.model_json_schema()}\n"
            "Return corrected JSON matching the schema.",
            schema,
        )
        return corrected.model_dump_json()


class Reflection(BaseModel):
    critique: str
    improvements: str
    good_enough: bool


class ReflectionThink(Think):
    # deliberately doesn't write a final answer -- it only judges and suggests. Its output isn't
    # even a "decision" (nothing for Act to execute); it's a "critique" message that goes straight
    # back into the conversation, and the loop returns to a real Think (e.g. plain Think) next,
    # which reads the critique/good_enough as context and makes its own ordinary decision --
    # try again differently, or respond -- with no separate retry mechanism needed anywhere.
    def run(self, conversation):
        result = extract(
            f"Conversation so far:\n{render(conversation)}\n\n"
            "Critique whether the most recent result actually answers the original request and "
            "was carried out well. Say what was good and what wasn't. If it's not good enough, "
            "describe concrete improvements to try next. Do not write the final answer yourself.\n"
            f"Return JSON matching this schema:\n{Reflection.model_json_schema()}",
            Reflection,
        )
        return {
            "role": "critique",
            "content": result.critique,
            "improvements": result.improvements,
            "good_enough": result.good_enough,
        }
