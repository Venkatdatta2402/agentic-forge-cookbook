"""The trace: one record per question, and the same tree in Langfuse (a fake one here)."""

import asyncio

from pydantic import BaseModel
from stubs import call, calls

from stats_desk.agents.agent import Agent, answer
from stats_desk.core import trace
from stats_desk.core.tools import Registry, Tool


class Fake:
    """Stands in for a Langfuse observation (and the client): records what is nested under what."""

    def __init__(self, name="root", as_type=None, **fields):
        self.name, self.as_type, self.fields, self.children, self.ended = name, as_type, fields, [], False
        self.trace_id = "t1"

    def start_observation(self, name, as_type, **fields):
        child = Fake(name, as_type, **fields)
        self.children.append(child)
        return child

    def update(self, **fields):
        self.name = fields.pop("name", self.name)        # Langfuse renames an observation the same way
        self.fields.update(fields)

    def end(self):
        self.ended = True

    def tree(self, depth=0):
        return [("  " * depth) + f"{self.as_type}:{self.name}"] + [l for c in self.children for l in c.tree(depth + 1)]


class Total(BaseModel):
    total: int


async def slow_add(a: int, b: int) -> int:
    """Add two numbers."""
    await asyncio.sleep(0.01)
    return a + b


async def test_a_question_is_one_record_and_one_tree_with_each_agent_nested(model, monkeypatch):
    client = Fake("client")
    monkeypatch.setattr(trace.LANGFUSE, "client", client)
    worker = Agent("worker", "You add.", tools=Registry([Tool(slow_add)]), model="m")

    async def ask_workers(tasks: list[str]) -> str:
        """Run workers."""
        done = await asyncio.gather(*(answer(worker, t, Total) for t in tasks))
        return str([d.total for d in done])

    boss = Agent("boss", "You direct.", tools=Registry([Tool(ask_workers)]), model="m")
    model.steps += [calls(("ask_workers", {"tasks": ["1+2", "3+4"]})),
                    call("slow_add", a=1, b=2), call("slow_add", a=3, b=4),
                    call("json", total=3), call("json", total=7),
                    call("json", total=10)]
    with trace.Trace("what is 1+2 and 3+4?") as recorded:
        result = await answer(boss, "what is 1+2 and 3+4?", Total)
        recorded.finish("brief", brief=str(result.total))

    r = recorded.record
    assert r["outcome"] == "brief" and r["tokens"]["total"] == 360 + 5 * 340     # stubs: calls() 360, call() 340
    assert {c["path"] for c in r["model calls"]} == {"boss", "boss > ask_workers > worker"}
    adds = [t for t in r["tools"] if t["tool"] == "slow_add"]
    assert len(adds) == 2 and all(t["path"] == "boss > ask_workers > worker" and t["kind"] == "ok" for t in adds)
    # Langfuse: the same tree, every observation ended
    root = client.children[0]
    assert root.tree()[:3] == ["chain:what is 1+2 and 3+4?", "  agent:boss", "    generation:boss -> ask_workers"]
    asking = next(c for c in root.children[0].children if c.name == "ask_workers")
    assert [c.name for c in asking.children] == ["worker", "worker"]          # both workers under the call
    assert all(len([g for g in w.children if g.as_type == "generation"]) == 2 for w in asking.children)
    assert root.ended and all(c.ended for c in asking.children)


async def test_with_no_trace_open_nothing_is_recorded_and_nothing_breaks(model):
    agent = Agent("calc", "You add.", tools=Registry([Tool(slow_add)]), model="m")
    model.steps += [call("slow_add", a=1, b=1), call("json", total=2)]
    assert await answer(agent, "1+1", Total) == Total(total=2)


async def test_a_broken_langfuse_never_reaches_the_run(model, monkeypatch, capsys):
    class Broken(Fake):
        def start_observation(self, **_):
            raise ConnectionError("langfuse is down")
    monkeypatch.setattr(trace.LANGFUSE, "client", Broken())
    agent = Agent("calc", "You add.", model="m")
    model.steps += [call("json", total=2)]
    with trace.Trace("1+1") as recorded:
        assert await answer(agent, "1+1", Total) == Total(total=2)
        recorded.finish("brief")
    assert recorded.record["model calls"][0]["in"] == 300
    assert "langfuse: ConnectionError" in capsys.readouterr().out
