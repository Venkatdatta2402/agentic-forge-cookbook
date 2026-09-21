"""The agent patterns, as importable factories.

Every one of these is built from the same components: a `Runtime` walking a declared list of
Observe / Think / Act. None of them adds machinery -- the pattern IS the list and the flags.

Why factories rather than the bare declarations they wrap:

Some settings are the caller's business (how many refinement rounds, how wide the beam) and are
parameters here. Others are not choices at all -- they encode failures already measured, and a
pattern is only correct with them set a particular way. `allow_fork=False` on a leaf Think exists
because a branch told "get the time in Tokyo" would otherwise delegate that onward. `explain=True`
on a search branch exists because Tree of Thoughts feeds each parent's working to its children,
and stripping the reasons out of that block loses the part worth inheriting.

Written out by hand every time, those get forgotten. A factory makes the settings that must be
right unavailable to get wrong, and exposes only the ones that are genuinely a decision.
"""

from act import Act
from observe import Observe
from runtime import Graph, Loop, Runtime
from think import (ExpandThink, Fork, PlannerThink, ReflectionThink, SelectThink, Think)


def branch_runtime_factory(tools=None, functions=None, schemas=None, explain=None,
                           allow_fork=None, max_iterations=14):
    """Builds the Runtime a sub-agent runs. Passed to Act, which calls it once per branch.

    explain / allow_fork:
      None  -- the delegating Think decides, per branch, via fields on the `delegate` call. It
               wrote the goal, so it is the only thing that knows whether this sub-goal needs
               splitting again, or whether the caller will want to see the working.
      True/False -- the pattern overrides, for patterns whose own mechanics depend on it.
    """
    def make(branch):
        return Runtime(
            loop=[
                Think(tools=tools,
                      allow_fork=branch.get("allow_fork", False) if allow_fork is None else allow_fork,
                      explain=branch.get("explain", False) if explain is None else explain,
                      temperature=branch.get("temperature"), top_p=branch.get("top_p")),
                # `make` names itself: a branch permitted to fork needs a way to build runtimes
                # for ITS children, and without this Act would reach for a branch_runtime of None
                # the moment it forked. One line, and delegation goes as deep as the budget allows.
                Act(functions=functions, schemas=schemas, branch_runtime=make),
                Observe(),
            ],
            repeat_from=0,
            max_iterations=branch.get("max_iterations") or max_iterations,
            max_seconds=branch.get("max_seconds"),
            max_tool_calls=branch.get("max_tool_calls"),
            max_tokens=branch.get("max_tokens"),
        )
    return make


# ------------------------------------------------------------------ acting patterns

def react(request, tools=None, functions=None, schemas=None, fork=True, **budgets):
    """Interleave reasoning and acting: decide one step, do it, look at the result, decide again.

    fork=True lets the Think split the request into sub-agents if it judges that worth the cost.
    That is not a pattern of its own -- it is this loop with delegation switched on, and the
    branch settings left as None so the delegating Think fills in `allow_fork` and `explain` per
    branch. It wrote each goal, so it is the only thing that knows how big they are.

    The shape does not change when it does. `Think` withholds `delegate` for one turn after
    branch results come back, so a single Think covers both the forking pass and the pass that
    works on what came back -- no shadow component, no second repeat_from.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            Think(tools=tools, allow_fork=fork),
            Act(functions=functions, schemas=schemas,
                branch_runtime=branch_runtime_factory(tools, functions, schemas)),
            Observe(),
        ],
        repeat_from=1,
        **{"max_iterations": 20, **budgets},
    )


def plan_and_execute(request, tools=None, functions=None, schemas=None, outputs=None, fork=True,
                     **budgets):
    """Write the goals up front, then let an ordinary loop work out how to reach each one.

    The planner names no tools and emits no decision -- a goal is not executable. Which loop
    executes it is the caller's choice; this uses ReAct, the simplest one that can.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            PlannerThink(tools=tools, outputs=outputs, goals=True),
            # allow_replan: the planner runs once and is never re-entered (repeat_from skips it),
            # so without this the goals could never be revised no matter what execution turned up.
            # It belongs on the executing Think because that is the only thing that finds out.
            Think(tools=tools, allow_fork=fork, allow_replan=True),
            Act(functions=functions, schemas=schemas,
                branch_runtime=branch_runtime_factory(tools, functions, schemas)),
            Observe(),
        ],
        repeat_from=2,
        **{"max_iterations": 20, **budgets},
    )


def rewoo(request, tools=None, functions=None, schemas=None, outputs=None, **budgets):
    """Planner, workers, solver -- planned once, executed once, answered once.

    Worth recognising rather than choosing: its argument was that decoupling reasoning from
    observation is cheaper than ReAct, and LLM Compiler keeps that while dropping the one-shot
    limitation. The only difference from llm_compiler below is that this cannot replan.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            PlannerThink(tools=tools, outputs=outputs),                 # planner
            Graph(act=Act(functions=functions, schemas=schemas)),       # workers
            # The solver gets NO tools, and that is the whole of what makes it a solver: its job
            # is to answer from the evidence the graph already gathered. Handed the tools again
            # it simply calls them again -- against the real model it re-fetched a time the plan
            # had already fetched, three turns running, until the budget stopped it. Withholding
            # tools here is not cornering it; answering is the only thing left to decide.
            Think(tools=None, allow_fork=False),                        # solver
            # the solver's Act needs the real tools. It usually just converts a `respond`
            # decision into the answer, but a solver handed thin evidence sometimes decides to
            # make one more call -- and a bare Act() has no functions to call it with, which
            # surfaced as a TypeError from inside execution rather than as anything readable.
            Act(functions=functions, schemas=schemas),
            Observe(),
        ],
        # ReWOO is one-shot in the paper, and this was declared as a flat sequence to match.
        # Against the real model that ended a run with no answer at all: the solver decided on
        # one more tool call, Act ran it, and the runtime walked off the end of the list with
        # the last message still a decision. Cycling back to the solver costs nothing when it
        # answers first time and saves the run when it doesn't.
        repeat_from=3,
        **{"max_iterations": 12, **budgets},
    )


def llm_compiler(request, tools=None, functions=None, schemas=None, outputs=None, **budgets):
    """ReWOO's skeleton, allowed to repeat: plan, run the plan as a DAG, replan or finish.

    `Graph` runs each step the moment its own dependencies are done, and absorbs the terminal
    Act -- so a round that decides the request is answered ends the run rather than looping.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            PlannerThink(tools=tools, outputs=outputs),
            Graph(act=Act(functions=functions, schemas=schemas)),
        ],
        repeat_from=1,
        **{"max_iterations": 20, **budgets},
    )


# ------------------------------------------------------------------ critique patterns

def self_refine(request, tools=None, functions=None, schemas=None, refine_rounds=2, fork=True,
                **budgets):
    """Draft, critique your own draft, redraft -- with no tool calls in the cycle.

    The critique loop holds Think and ReflectionThink only. An Act inside it would convert the
    first `respond` decision into an assistant message and end the run before anything was
    refined; the Act comes after, once the refining is done.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            Loop([Think(tools=tools, allow_fork=False), ReflectionThink()], count=refine_rounds),
            # allow_fork on the Think that acts on the refined draft: by then the critique may
            # have turned up work worth splitting. Its Act therefore needs a branch_runtime --
            # a Think that can fork paired with an Act that cannot build a branch is a crash.
            Think(tools=tools, allow_fork=fork),
            Act(functions=functions, schemas=schemas,
                branch_runtime=branch_runtime_factory(tools, functions, schemas)),
            Observe(),
        ],
        # same reason as ReWOO: a Think holding tools may answer OR call one more tool, and a
        # flat sequence ending on Act leaves the second case with no answer at all
        repeat_from=2,
        **{"max_iterations": 12, **budgets},
    )


def reflexion(request, tools=None, functions=None, schemas=None, attempt_turns=3, fork=True,
              **budgets):
    """Try the task for a few turns, reflect on how it went, try again with the critique in view.

    attempt_turns is how many Think/Act/Observe turns make up one attempt before reflecting --
    the caller's decision, since it depends entirely on how long the task takes.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            Loop([Think(tools=tools, allow_fork=fork),
                  Act(functions=functions, schemas=schemas,
                      branch_runtime=branch_runtime_factory(tools, functions, schemas)),
                  Observe()],
                 count=attempt_turns),
            ReflectionThink(),
        ],
        repeat_from=1,
        **{"max_iterations": 24, **budgets},
    )


# ------------------------------------------------------------------ multi-agent patterns

def self_consistency(request, tools=None, functions=None, schemas=None,
                     temperatures=(0.2, 0.8, 1.2), **budgets):
    """Answer the same question several times at different temperatures, then reconcile.

    Declared, not prompted. The earlier version asked the model to "delegate this exact question
    to 3 branches... do not change the wording between branches" -- a constraint pleaded for in
    prose. One comprehension guarantees it instead.

    explain=False: you are comparing answers, and several near-identical accounts of the same
    working is noise. There is no vote component either -- a Think reading labelled answers and
    picking the recurring one is a Think reading its context, and a VoteThink would hardcode
    majority when "two agree but the third reasons better" is worth keeping available.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            Fork(branches=[{"goal": request, "temperature": t} for t in temperatures]),
            Act(functions=functions, schemas=schemas,
                branch_runtime=branch_runtime_factory(tools, functions, schemas, explain=False)),
            Observe(),
            Think(tools=tools, allow_fork=False),
            Act(functions=functions, schemas=schemas),
            Observe(),
        ],
        repeat_from=4,   # the reconciling Think may need a tool call before it can answer
        **{"max_iterations": 12, **budgets},
    )


def debate(request, sides, tools=None, functions=None, schemas=None, start=None, turns=2,
           **budgets):
    """Several positions argued out, each branch keeping its history and seeing the others' turns.

    sides: [{"id": "python", "goal": "Argue that ..."}, ...] -- self-contained goals, since a
    branch sees nothing but its own.

    `start` names who opens. Everything else running in parallel would mean everyone speaks
    first and nobody responds; here the second speaker's history already holds the first's turn.
    After that, who goes next is a judgment, and it belongs to the Think below -- which is why
    this is `start` and not a full running order.

    That Think can advance but not fork: with three or more sides, picking who is worth hearing
    from next is the whole problem, while starting a fresh branch would throw away everything the
    existing one established.
    """
    return Runtime(
        loop=[
            Observe(event=request),
            Fork(branches=list(sides), persistent=True, broadcast=True,
                 start=start or sides[0].get("id")),
            Act(functions=functions, schemas=schemas,
                branch_runtime=branch_runtime_factory(tools, functions, schemas, explain=False)),
            Observe(),
            # No tools, for the same reason ReWOO's solver has none: this Think moderates. Its
            # job is to pick who speaks next and to say when the exchange has settled, and it
            # never needs to look anything up -- the branches do that. Handing it the tools as
            # well pushed the schema past what llama-3.3-70b reliably emitted, and it answered with
            # llama's <function=...> text form until the retries ran out.
            # An exchange has no natural end, so the caller says how long it runs. Cycled with
            # repeat_from instead, this Think advanced a branch every single turn until the budget
            # stopped it -- asked to wrap up, it delegated even THAT, telling a branch to
            # "summarize the main points" rather than concluding itself. Telling a model when to
            # stop is a suggestion; a Loop with a count is a fact.
            Loop([Think(tools=None, allow_fork=False, allow_advance=True),
                  Act(functions=functions, schemas=schemas,
                      branch_runtime=branch_runtime_factory(tools, functions, schemas, explain=False)),
                  Observe()],
                 count=turns),
            # and then a moderator that CANNOT advance, so summing up is the only move left
            Think(tools=None, allow_fork=False, allow_advance=False),
            Act(functions=functions, schemas=schemas),
        ],
        **{"max_iterations": 6 + 3 * turns, **budgets},
    )


# ------------------------------------------------------------------ search patterns

def _search(request, tools, functions, schemas, thoughts, keep, max_parents, budgets):
    return Runtime(
        loop=[
            Observe(event=request),
            ExpandThink(thoughts=thoughts, max_parents=max_parents),
            # explain=True is not optional here: every child inherits its parent's compacted
            # working as the state it grows from, and with explain off that block carries calls
            # and results but none of the reasons -- the part most worth passing on.
            Act(functions=functions, schemas=schemas,
                branch_runtime=branch_runtime_factory(tools, functions, schemas, explain=True)),
            SelectThink(keep=keep),
            Act(functions=functions, schemas=schemas),
            Observe(),
            # A search needs a way out. ExpandThink always expands and SelectThink always prunes
            # -- neither can ever decide the answer is good enough, so without this the loop
            # cycles until the budget stops it and the run ends on "Stopping:" every single time.
            # This Think reads the surviving candidates and either answers or lets the loop go
            # round again, which is the judgment the search itself cannot make.
            Think(tools=tools, allow_fork=False),
            Act(functions=functions, schemas=schemas),
        ],
        repeat_from=1,
        **{"max_iterations": 24, **budgets},
    )


def tree_of_thoughts(request, tools=None, functions=None, schemas=None, thoughts=2, keep=2,
                     **budgets):
    """Explore several lines of thinking, score them, keep the best few, expand those.

    Which parent a new thought belongs to is not decided: ExpandThink asks once per surviving
    branch, showing only that branch's working, so the thought grows from it by construction.

    max_parents=1 is the whole of what makes this a tree -- no merge is ever offered, so the
    shape can only split. Pruned candidates keep their scores, so a line abandoned in round one
    is still reachable in round three: that is the backtracking.
    """
    return _search(request, tools, functions, schemas, thoughts, keep, 1, budgets)


def graph_of_thoughts(request, tools=None, functions=None, schemas=None, thoughts=2, keep=3,
                      max_parents=2, **budgets):
    """Tree of Thoughts plus the one move a tree cannot make: merging several thoughts into one.

    Identical to tree_of_thoughts except max_parents. Above 1, each round also asks whether any
    group of frontier branches is worth combining, and a yes produces a node with several parents.
    Merging is the one thing that cannot be structural -- nothing about the loop can work out
    which two ideas are worth joining.
    """
    return _search(request, tools, functions, schemas, thoughts, keep, max_parents, budgets)


PATTERNS = {
    "ReAct": react,
    "Plan & Execute": plan_and_execute,
    "ReWOO": rewoo,
    "LLM Compiler": llm_compiler,
    "Self-Refine": self_refine,
    "Reflexion": reflexion,
    "Self-Consistency": self_consistency,
    "Debate": debate,
    "Tree of Thoughts": tree_of_thoughts,
    "Graph of Thoughts": graph_of_thoughts,
}
