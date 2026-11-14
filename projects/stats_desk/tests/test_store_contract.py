"""One set of checks for every store: the in-memory one the tests use, and Cosmos DB.

The Cosmos run needs a real account and is skipped unless STATS_DESK_COSMOS_TEST_URL is set:

    $env:STATS_DESK_COSMOS_TEST_URL = "https://<account>.documents.azure.com:443/"
    uv run pytest tests/test_store_contract.py

Every name it writes starts with a fresh prefix, and is deleted afterwards.
"""

import asyncio
import os
import secrets

import pytest

from stats_desk.api.store import BLOCKS, BRIEFS, COUNTERS, LEASES, USAGE, CosmosStore, MemoryStore

COSMOS = os.environ.get("STATS_DESK_COSMOS_TEST_URL")


@pytest.fixture(params=["memory", "cosmos"])
async def store(request):
    if request.param == "memory":
        yield MemoryStore(), "t-"
        return
    if not COSMOS:
        pytest.skip("set STATS_DESK_COSMOS_TEST_URL to run against Cosmos DB")
    s = CosmosStore(COSMOS)
    prefix = f"test-{secrets.token_hex(4)}-"
    yield s, prefix
    for container in (BRIEFS, LEASES, COUNTERS, BLOCKS, USAGE):          # leave nothing behind
        c = await s._c(container)
        async for item in c.query_items("SELECT c.id FROM c WHERE STARTSWITH(c.id, @p)",
                                         parameters=[{"name": "@p", "value": prefix}]):
            await c.delete_item(item["id"], partition_key=item["id"])
    await s.close()


async def test_a_record_is_saved_and_read_back_whole(store):
    s, p = store
    doc = {"id": p + "b1", "question": "q", "status": "done", "public": False,
           "record": '{"findings": {"f1": {"finding": {"claim": "18.9%"}}}}', "nested": {"a": [1, 2]}}
    await s.save(p + "b1", doc)
    assert await s.get(p + "b1") == doc
    assert await s.get(p + "missing") is None


async def test_only_chosen_briefs_are_public_newest_first(store):
    s, p = store
    for i, public in enumerate((True, False, True)):
        await s.save(f"{p}b{i}", {"id": f"{p}b{i}", "created": f"2026-11-1{i}", "public": public})
    mine = lambda docs: [d["id"] for d in docs if d["id"].startswith(p)]
    assert mine(await s.public()) == [p + "b2", p + "b0"]
    assert await s.set_public(p + "b1", True) and await s.set_public(p + "b2", False)
    assert mine(await s.public()) == [p + "b1", p + "b0"]
    assert await s.set_public(p + "missing", True) is False


async def test_a_lease_is_held_by_one_question_until_released(store):
    s, p = store
    key = p + "key"
    assert await s.claim(key, "q1") is None
    assert await s.claim(key, "q2") == "q1"           # busy: says who holds it
    assert await s.claim(key, "q1") is None           # the holder may claim again
    await s.release(key, "q2")                        # not q2's to release
    assert await s.claim(key, "q2") == "q1"
    await s.release(key, "q1")
    assert await s.claim(key, "q2") is None


async def test_two_claims_at_once_one_wins(store):
    s, p = store
    results = await asyncio.gather(*(s.claim(p + "race", f"q{i}") for i in range(4)))
    assert results.count(None) == 1


async def test_counts_add_up_even_when_written_at_once(store):
    s, p = store
    assert await s.count(p + "visitor") == 0
    await asyncio.gather(*(s.add(p + "visitor") for _ in range(5)))
    await s.add(p + "visitor", 2)
    assert await s.count(p + "visitor") == 7


async def test_a_spent_key_is_blocked_for_a_while(store):
    s, p = store
    assert not await s.blocked(p + "spent")
    await s.block(p + "spent", 600)
    assert await s.blocked(p + "spent")
    await s.block(p + "spent", -1)                    # an expired block
    assert not await s.blocked(p + "spent")


async def test_a_keys_tokens_add_up_even_when_written_at_once(store):
    s, p = store
    await asyncio.gather(*(s.add_usage(p + "tokens", 100) for _ in range(4)))
    await s.add_usage(p + "tokens", 50)
    assert await s.usage_24h(p + "tokens") == 450
    assert await s.usage_24h(p + "other") == 0
