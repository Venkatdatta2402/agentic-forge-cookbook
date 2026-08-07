import concurrent.futures
from types import SimpleNamespace

from chat import Conversation
from llm import execute_tool_call
from think import DebugThink


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
        decision = next(m for m in reversed(conversation.messages) if m.get("role") == "decision")

        if decision["type"] == "respond":
            return {"role": "assistant", "content": decision["content"]}

        if decision["type"] == "fork":
            conversation.pending = self._fork(conversation, decision)
            return None

        if decision["type"] == "advance_branch":
            conversation.pending = self._advance_branch(conversation, decision)
            return None

        if decision["type"] == "prune":
            pending = conversation.pending or []
            conversation.pending = [pending[i] for i in decision["keep_indices"]]
            return None

        conversation.pending = self.execute(decision["tool"], decision["content"])
        return None

    def _start_branch(self, conversation, branch):
        # deliberately NOT seeded with conversation.messages -- a branch only ever gets a system
        # message (if any) plus its own goal. The goal is supposed to be self-contained (Think's
        # job to write it that way); inheriting the full history too would let a branch see the
        # original multi-part request and other branches' tasks, which it should never need to.
        branch_conversation = Conversation()
        branch_conversation.messages = [m for m in conversation.messages if m["role"] == "system"]
        branch_conversation.messages.append({"role": "user", "content": branch["goal"]})
        self.branch_runtime(branch).run(branch_conversation)
        return branch_conversation

    def _fork(self, conversation, decision):
        # delegating to sub-agents (notebook 3's "delegate" example) -- doesn't return until
        # every branch is done, same "doesn't return until complete" rule as a normal tool call.
        branches = decision["branches"]
        persistent = decision.get("persistent", False)
        broadcast = decision.get("broadcast", False)
        ids = [b.get("id", f"branch{i}") for i, b in enumerate(branches)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            branch_conversations = list(pool.map(lambda b: self._start_branch(conversation, b), branches))

        # capture each branch's OWN result before broadcasting mutates messages[-1]
        results = [
            {"goal": b["goal"], "id": i, "content": bc.messages[-1]["content"]}
            for b, i, bc in zip(branches, ids, branch_conversations)
        ]

        # broadcast: each branch's turn becomes part of every sibling's own history too, clearly
        # attributed to the other branch (a "user" turn reporting what they said) -- not appended
        # under the branch's own role, which would make it look self-authored on its next turn.
        if broadcast:
            for i, bc in zip(ids, branch_conversations):
                for other_id, result in zip(ids, results):
                    if other_id != i:
                        bc.messages.append({"role": "user", "content": f'[{other_id}] said: {result["content"]}'})

        if persistent:
            conversation.branches = dict(zip(ids, branch_conversations))
            conversation.branches_broadcast = broadcast

        return results

    def _advance_branch(self, conversation, decision):
        # a persistent branch gets one more turn -- reuses its own accumulated conversation,
        # exactly like the very first fork call did, just via a fresh Runtime each time.
        branch_id = decision["branch"]
        branch_conversation = conversation.branches[branch_id]
        if decision.get("message"):
            branch_conversation.messages.append({"role": "user", "content": decision["message"]})
        self.branch_runtime({"goal": None}).run(branch_conversation)
        content = branch_conversation.messages[-1]["content"]

        if getattr(conversation, "branches_broadcast", False):
            for other_id, other_conversation in conversation.branches.items():
                if other_id != branch_id:
                    other_conversation.messages.append({"role": "user", "content": f"[{branch_id}] said: {content}"})

        return [{"goal": None, "id": branch_id, "content": content}]

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
