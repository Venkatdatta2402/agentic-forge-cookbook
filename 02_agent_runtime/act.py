import concurrent.futures
import threading
from types import SimpleNamespace

from chat import Conversation
from llm import execute_tool_call
from think import DebugThink, transcript


def pending_decision(conversation):
    """The most recent decision nobody has carried out yet, or None."""
    return next((m for m in reversed(conversation.messages)
                 if m.get("role") == "decision" and not m.get("applied")), None)


class Act:
    def __init__(self, functions=None, schemas=None, retries=3, branch_runtime=None, max_workers=8):
        self.functions = functions
        self.schemas = schemas
        self.retries = retries
        # branch_runtime(branch) -> a fresh Runtime for that branch's own sub-agent loop.
        # Only needed if this Act will ever see a "fork" or "advance_branch" decision.
        self.branch_runtime = branch_runtime
        self.max_workers = max_workers

    def run(self, conversation):
        decision = pending_decision(conversation)
        if decision is None:
            # Nothing waiting to be done. Reached whenever the component before this one produced
            # something other than a decision -- a plan, a critique -- which is a perfectly normal
            # turn, not an error. Act does nothing and the loop moves on.
            return None
        # crossed off before it is carried out, so no later Act can find it and do it a second
        # time. Act used to ask "what is the latest decision?" when what it meant was "what is
        # the latest decision that has not happened yet", and those two questions only gave the
        # same answer because every loop happened to be arranged so that they did.
        decision["applied"] = True

        if decision["type"] == "respond":
            return {"role": "assistant", "content": decision["content"]}

        if decision["type"] == "fork":
            conversation.pending = self._fork(conversation, decision)
            return None

        if decision["type"] == "advance_branch":
            conversation.pending = self._advance_branch(conversation, decision)
            return None

        if decision["type"] == "expand":
            conversation.pending = self._expand(conversation, decision)
            return None

        if decision["type"] == "collect":
            conversation.pending = self._collect(conversation, decision)
            return None

        if decision["type"] == "prune":
            # scores are kept even for the candidates being dropped, and kept on the conversation
            # rather than on the decision, because that is what makes returning to an abandoned
            # line possible later: `expand` lists every explored branch with its score, so a
            # promising one set aside in round one is still visible in round three.
            if decision.get("scores"):
                if getattr(conversation, "branch_scores", None) is None:
                    conversation.branch_scores = {}
                conversation.branch_scores.update(decision["scores"])
            # same guard as SelectThink: pruning only means anything against a list of branch
            # results. Anything else is left exactly as it was rather than indexed into.
            pending = conversation.pending
            if isinstance(pending, list):
                conversation.pending = [pending[i] for i in decision["keep_indices"]]
                # the frontier: which branches the next round grows from. Recorded on the
                # conversation because ExpandThink runs after Observe has consumed pending, and
                # because expanding is supposed to follow from what selection kept -- the search
                # decides what gets children, not the branches and not the expander.
                conversation.frontier = [b["id"] for b in conversation.pending if b.get("id")]
            return None

        conversation.pending = self.execute(decision["tool"], decision["content"])
        # what the tool really returned, kept on the call that produced it. Observe rewrites the
        # result before the model sees it, and a rewording differs from run to run -- so the
        # only reliable way to tell that two calls got the same answer is to compare these.
        # Runtime's doom-loop check does exactly that (see Runtime._doom_loop).
        decision["result"] = conversation.pending["content"]
        return None

    @staticmethod
    def _conversation_pool(conversation):
        # A pool that outlives the call, needed only when branches are left running. The default
        # path still uses a `with` block, and leaving that block is what waits for everyone --
        # so nothing changes for a fork that waits for all of its branches.
        pool = getattr(conversation, "pool", None)
        if pool is None:
            pool = conversation.pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        return pool

    @staticmethod
    def _reserve(conversation, branch_id, branch):
        # Only the POOLED budgets are reserved. Seconds and iterations are ceilings each branch
        # faces on its own -- branches run concurrently, so a branch still running has not taken
        # wall clock away from anyone. Tokens and tool calls are really spent, and a branch that
        # has been promised 3000 tokens but not yet spent them would otherwise let the next fork
        # hand the same 3000 out again.
        #
        # Nothing is reserved for a branch given no explicit budget: there is no honest number to
        # reserve, and guessing one would understate the remainder rather than overstate it.
        if getattr(conversation, "reserved_budget", None) is None:
            conversation.reserved_budget = {"tokens": 0, "tool_calls": 0}
            conversation.reserved_by_branch = {}
        held = {key: branch.get(field) or 0
                for key, field in (("tokens", "max_tokens"), ("tool_calls", "max_tool_calls"))}
        # remembered per branch so releasing gives back exactly what was taken, without the
        # collecting decision having to carry numbers it never saw
        conversation.reserved_by_branch[branch_id] = held
        for key, amount in held.items():
            conversation.reserved_budget[key] += amount

    @staticmethod
    def _release(conversation, branch_id):
        held = getattr(conversation, "reserved_by_branch", None) or {}
        for key, amount in held.pop(branch_id, {}).items():
            conversation.reserved_budget[key] = max(0, conversation.reserved_budget[key] - amount)

    def _run_branch(self, branch, branch_conversation, abandoned=None):
        runtime = self.branch_runtime(branch)
        if abandoned is not None:
            # A background branch has to be stoppable, or a run that ends while it is still
            # working leaves it burning quota on an answer nobody will read. Threads cannot be
            # killed, but a Runtime checks `cancel` at every iteration, so it stops at the next
            # clean boundary. Composed with whatever cancel the caller's factory already set,
            # rather than replacing it -- theirs is no less valid for us also having one.
            # read defensively: branch_runtime is a caller-supplied factory and may return
            # anything with .run(), not necessarily a Runtime. Reading .cancel directly raised
            # AttributeError *inside the worker thread*, where it stayed invisible until someone
            # collected the future -- the worst place for an error to hide.
            own = getattr(runtime, "cancel", None)
            runtime.cancel = lambda: abandoned.is_set() or bool(own and own())
        runtime.run(branch_conversation)
        return branch_conversation

    def _seed_branch(self, conversation, branch):
        # deliberately NOT seeded with conversation.messages -- a branch only ever gets a system
        # message (if any) plus its own goal. The goal is supposed to be self-contained (Think's
        # job to write it that way); inheriting the full history too would let a branch see the
        # original multi-part request and other branches' tasks, which it should never need to.
        branch_conversation = Conversation()
        branch_conversation.messages = [m for m in conversation.messages if m["role"] == "system"]
        # no goal means an empty branch: an id holding a place in a search, waiting for `expand`
        # to give it a thought. Everything else about it is an ordinary branch.
        if branch.get("goal"):
            branch_conversation.messages.append({"role": "user", "content": branch["goal"]})
        # the stop flag is the one thing a branch DOES inherit, and it has to be the same object,
        # not a copy. A branch that forks children of its own would otherwise mint a fresh flag
        # for them, and cancelling the outer run would never reach the grandchildren -- they would
        # run to completion, and the branch waiting on them could not stop until they did.
        abandoned = getattr(conversation, "abandoned", None)
        if abandoned is not None:
            branch_conversation.abandoned = abandoned
        return branch_conversation

    def _start_branch(self, conversation, branch):
        # seeding and running are separate because a fork with `start` set creates every branch
        # but runs only one of them -- the rest wait, fully formed, for an `advance`.
        branch_conversation = self._seed_branch(conversation, branch)
        return self._run_branch(branch, branch_conversation,
                                getattr(conversation, "abandoned", None))

    def _fork(self, conversation, decision):
        # delegating to sub-agents (notebook 3's "delegate" example) -- doesn't return until
        # every branch is done, same "doesn't return until complete" rule as a normal tool call.
        branches = decision["branches"]
        persistent = decision.get("persistent", False)
        broadcast = decision.get("broadcast", False)
        start = decision.get("start")
        ids = [b.get("id", f"branch{i}") for i, b in enumerate(branches)]

        wait_for = decision.get("wait_for") if start is None else None

        if wait_for is not None and all(b.get("goal") for b in branches):
            # Partial waiting: every branch starts, but the caller carries on as soon as the ones
            # it actually needs are done. This is the ONE place the rule "Act does not return
            # until everything it started has finished" is deliberately broken, and it is why the
            # pool has to outlive the call.
            #
            # A branch left running is always kept addressable, persistent or not -- something
            # has to be able to collect it later, and its budget stays reserved until it is.
            pool = self._conversation_pool(conversation)
            if getattr(conversation, "abandoned", None) is None:
                conversation.abandoned = threading.Event()
            branch_conversations = [self._seed_branch(conversation, b) for b in branches]
            futures = {i: pool.submit(self._run_branch, b, bc, conversation.abandoned)
                       for i, b, bc in zip(ids, branches, branch_conversations)}
            waited = [i for i in ids if i in wait_for] or ids
            concurrent.futures.wait([futures[i] for i in waited])

            if getattr(conversation, "running", None) is None:
                conversation.running = {}
            if getattr(conversation, "branches", None) is None:
                conversation.branches = {}
            conversation.branches.update(zip(ids, branch_conversations))
            for i, b in zip(ids, branches):
                if i not in waited:
                    conversation.running[i] = futures[i]
                    self._reserve(conversation, i, b)

            ran = [(i, b, bc) for i, b, bc in zip(ids, branches, branch_conversations) if i in waited]
        elif start is None and all(b.get("goal") for b in branches):
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                branch_conversations = list(pool.map(lambda b: self._start_branch(conversation, b), branches))
            ran = list(zip(ids, branches, branch_conversations))
        elif start is None:
            # allocation only: branches with no goal yet (Fork(count=N)). They get their ids and
            # nothing else, and wait for an `expand` to write a thought into them. Nothing runs,
            # so there is nothing to report -- Observe says what is now waiting instead.
            branch_conversations = [self._seed_branch(conversation, b) for b in branches]
            ran = []
        else:
            # an exchange with a declared opener: every branch is created so it can be advanced
            # later, but only the opener runs now. Running them all in parallel and broadcasting
            # afterwards makes everyone speak first, which is not a debate -- nobody is responding
            # to anything. Here the second speaker's history already contains the first's turn.
            branch_conversations = [self._seed_branch(conversation, b) for b in branches]
            first = ids.index(start)
            self.branch_runtime(branches[first]).run(branch_conversations[first])
            ran = [(ids[first], branches[first], branch_conversations[first])]

        # capture each branch's OWN result before broadcasting mutates messages[-1].
        # `conversation` carries the branch's whole working -- what it called, why, and what came
        # back. It rides in conversation.pending, never in messages, so it costs nothing unless an
        # Observe is configured to show it (branch_detail="transcript"). Same rule as a raw tool
        # result: Act produces everything, Observe decides what the conversation actually sees.
        # `label` is what the conversation will call this branch. For a persistent fork that has
        # to be the id, because the id is the handle -- a Think asked to advance one can only
        # name what it has seen, and labelling by goal taught it to ask for "argue for Python"
        # when the branch is called "python". Throwaway branches keep the goal, which reads
        # better and is never used to address anything.
        results = [
            {"goal": b["goal"], "id": i, "label": i if persistent else (b["goal"] or i),
             # carried so Observe can follow the decision the delegating Think already made,
             # instead of being asked to make the same one again from the loop declaration
             "explain": bool(b.get("explain")),
             "content": bc.messages[-1]["content"], "conversation": bc}
            for i, b, bc in ran
        ]

        # broadcast: each branch's turn becomes part of every sibling's own history too, clearly
        # attributed to the other branch (a "user" turn reporting what they said) -- not appended
        # under the branch's own role, which would make it look self-authored on its next turn.
        # Driven off `results` rather than zipped against `ids`, because with a declared opener
        # only one branch has spoken and the ones that have not still need to hear it.
        if broadcast:
            for i, bc in zip(ids, branch_conversations):
                for result in results:
                    if result["id"] != i:
                        bc.messages.append({"role": "user", "content": f'[{result["id"]}] said: {result["content"]}'})

        if persistent:
            # merged, not replaced. A Fork declared inside a loop runs again every round, and
            # assigning here would delete every branch of every earlier round -- taking the whole
            # searched tree with it, which is precisely what has to survive for backtracking.
            if getattr(conversation, "branches", None) is None:
                conversation.branches = {}
            conversation.branches.update(zip(ids, branch_conversations))
            conversation.branches_broadcast = broadcast

        return results

    def _expand(self, conversation, decision):
        # one round of a search: every thought in the decision becomes a new branch. Ids are
        # allocated here rather than named by the decision -- numbered on from however many
        # branches already exist, so rounds never collide and every earlier branch stays
        # addressable, which is what makes going back to an abandoned line possible at all.
        #
        # A child opens with its parents' working written out: not their raw message lists, which
        # would drag every intermediate turn along, but the compacted block transcript() renders.
        # That block IS the state it inherits. One parent is a tree; several is the merge a tree
        # cannot make, and ExpandThink decides whether that is even offered.
        assignments = decision["assignments"]
        if getattr(conversation, "branches", None) is None:
            conversation.branches = {}
        if getattr(conversation, "branch_parents", None) is None:
            conversation.branch_parents = {}

        existing = len(conversation.branches)
        ids = [f"b{existing + i}" for i in range(len(assignments))]

        for branch_id, assignment in zip(ids, assignments):
            branch_conversation = Conversation()
            branch_conversation.messages = [m for m in conversation.messages if m["role"] == "system"]
            for parent_id in assignment["parents"]:
                parent = conversation.branches[parent_id]
                answer = next((m["content"] for m in reversed(parent.messages)
                               if m.get("role") == "assistant"), "")
                branch_conversation.messages.append({"role": "user", "content": (
                    "An earlier line of work you are continuing from:\n"
                    + transcript({"label": parent_id, "content": answer, "conversation": parent})
                )})
            branch_conversation.messages.append({"role": "user", "content": assignment["thought"]})
            conversation.branches[branch_id] = branch_conversation
            conversation.branch_parents[branch_id] = assignment["parents"]

        abandoned = getattr(conversation, "abandoned", None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            pool.map(lambda pair: self._run_branch({"goal": pair[1]["thought"]},
                                                   conversation.branches[pair[0]], abandoned),
                     list(zip(ids, assignments)))

        return [{"goal": a["thought"], "id": i, "label": i,
                 "content": conversation.branches[i].messages[-1]["content"],
                 "conversation": conversation.branches[i]}
                for i, a in zip(ids, assignments)]

    def _collect(self, conversation, decision):
        # waits for branches left running by an earlier partial fork, and hands back what they
        # produced. Their reservations are released here and not before: a branch's promised
        # tokens are unavailable to anyone else for exactly as long as it might still spend them.
        running = getattr(conversation, "running", None) or {}
        wanted = [i for i in decision["branches"] if i in running]
        concurrent.futures.wait([running[i] for i in wanted])

        results = []
        for branch_id in wanted:
            branch_conversation = running.pop(branch_id).result()
            self._release(conversation, branch_id)
            results.append({
                "goal": None, "id": branch_id, "label": branch_id,
                "content": branch_conversation.messages[-1]["content"],
                "conversation": branch_conversation,
            })
        return results

    def _advance_branch(self, conversation, decision):
        # a persistent branch gets one more turn -- reuses its own accumulated conversation,
        # exactly like the very first fork call did, just via a fresh Runtime each time.
        branch_id = decision["branch"]
        branch_conversation = conversation.branches[branch_id]
        if decision.get("message"):
            branch_conversation.messages.append({"role": "user", "content": decision["message"]})
        self._run_branch({"goal": None}, branch_conversation,
                         getattr(conversation, "abandoned", None))
        content = branch_conversation.messages[-1]["content"]

        if getattr(conversation, "branches_broadcast", False):
            for other_id, other_conversation in conversation.branches.items():
                if other_id != branch_id:
                    other_conversation.messages.append({"role": "user", "content": f"[{branch_id}] said: {content}"})

        return [{"goal": None, "id": branch_id, "label": branch_id,
                 "content": content, "conversation": branch_conversation}]

    def execute(self, tool_name, args):
        # pure: reads/writes nothing shared, only its own locals -- safe to call from several
        # threads at once (see runtime.Graph, which runs independent steps concurrently).
        debugger = DebugThink(self.schemas)
        attempts = []

        for attempt in range(self.retries):
            tool_call = SimpleNamespace(id="call", function=SimpleNamespace(name=tool_name, arguments=args))
            try:
                result = execute_tool_call(tool_call, self.functions, self.schemas)
                return {"tool": tool_name, "content": result}
            except Exception as e:
                attempts.append({"args": args, "error": str(e)})
                if attempt == self.retries - 1:
                    return {
                        "tool": tool_name,
                        "content": f"Error: permanently failed after {self.retries} attempts: {e}",
                    }
                args = debugger.fix(tool_name, args, str(e), attempts)
