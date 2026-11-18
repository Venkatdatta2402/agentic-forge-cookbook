"""What outlives a question, and what several machines running the app must agree on.

    briefs     one document per question: its whole record (agents/record.py), saved when it
               starts, after each finding and at the end. Follow-ups read their earlier record
               from here, not from memory: the host shuts an idle machine down.
    leases     "this key has a question running", so a second question on the same key waits,
               whichever machine it reaches. A lease ends when its question does, or expires
               (LEASE seconds) if the machine died first.
    counters   free questions per visitor, per day (UTC)
    usage      tokens each free key used, by the hour: its last 24 hours, as Groq counts them
               (the keys are used by the desk alone, so this count is the whole of their use)
    blocks     a free key Groq says is spent, until when (from Groq's Retry-After): the
               backstop for when the count is off

Two stores with the same methods (tests/test_store_contract.py runs the same checks on both):
    MemoryStore      tests and local runs (gone on restart; one machine)
    CosmosStore      Azure; picked with STATS_DESK_STORE=cosmos and STATS_DESK_COSMOS_URL

Who sees what: a few chosen briefs are public examples (`public: True`, set
by scripts/publish_brief.py); every other brief opens only by its link. Ids are random and
unguessable, so a link is the permission.
"""

import os
import secrets
import time
from typing import Protocol

BRIEFS, LEASES, COUNTERS, BLOCKS, USAGE = "briefs", "leases", "counters", "blocks", "usage"
LEASE = 1_800          # seconds: a question takes 5-10 minutes; this outlasts any
# Groq's day is a rolling 24 hours; counted in hour slots, the current hour and the 24 before it
# are summed: up to an hour too much, never too little
SLOTS = 25


def new_id() -> str:
    return secrets.token_urlsafe(12)          # 16 characters, 96 random bits


def hour(at: float) -> str:
    return time.strftime("%Y-%m-%dT%H", time.gmtime(at))


def recent_hours(now: float) -> set[str]:
    return {hour(now - 3600 * i) for i in range(SLOTS)}


class Store(Protocol):
    async def save(self, brief_id: str, doc: dict) -> None: ...
    async def get(self, brief_id: str) -> dict | None: ...
    async def public(self, limit: int = 20) -> list[dict]: ...
    async def set_public(self, brief_id: str, public: bool) -> bool: ...
    async def claim(self, name: str, holder: str) -> str | None: ...
    async def release(self, name: str, holder: str) -> None: ...
    async def holder(self, name: str) -> str | None: ...
    async def count(self, name: str) -> int: ...
    async def add(self, name: str, n: int = 1) -> None: ...
    async def block(self, name: str, seconds: float) -> None: ...
    async def blocked(self, name: str) -> bool: ...
    async def add_usage(self, name: str, tokens: int) -> None: ...
    async def usage_24h(self, name: str) -> int: ...


class MemoryStore:
    def __init__(self, clock=time.time):
        self.docs: dict[str, dict] = {}
        self.leases: dict[str, tuple[str, float]] = {}
        self.counters: dict[str, int] = {}
        self.blocks: dict[str, float] = {}
        self.usage: dict[str, dict[str, int]] = {}
        self.clock = clock

    async def add_usage(self, name, tokens):
        slots = self.usage.setdefault(name, {})
        slots[hour(self.clock())] = slots.get(hour(self.clock()), 0) + tokens

    async def usage_24h(self, name):
        recent = recent_hours(self.clock())
        return sum(n for h, n in self.usage.get(name, {}).items() if h in recent)

    async def block(self, name, seconds):
        self.blocks[name] = self.clock() + seconds

    async def blocked(self, name):
        return self.blocks.get(name, 0) > self.clock()

    async def save(self, brief_id, doc):
        self.docs[brief_id] = dict(doc)

    async def get(self, brief_id):
        doc = self.docs.get(brief_id)
        return dict(doc) if doc else None

    async def public(self, limit=20):
        shown = [d for d in self.docs.values() if d.get("public")]
        return sorted(shown, key=lambda d: d.get("created", ""), reverse=True)[:limit]

    async def set_public(self, brief_id, public):
        if brief_id not in self.docs:
            return False
        self.docs[brief_id]["public"] = public
        return True

    async def claim(self, name, holder):
        """None if `holder` now holds `name`; otherwise who does."""
        current = self.leases.get(name)
        if current and current[1] > self.clock() and current[0] != holder:
            return current[0]
        self.leases[name] = (holder, self.clock() + LEASE)
        return None

    async def release(self, name, holder):
        if self.leases.get(name, ("",))[0] == holder:
            del self.leases[name]

    async def holder(self, name):
        """Who holds `name` now, or None."""
        current = self.leases.get(name)
        return current[0] if current and current[1] > self.clock() else None

    async def count(self, name):
        return self.counters.get(name, 0)

    async def add(self, name, n=1):
        self.counters[name] = self.counters.get(name, 0) + n


class CosmosStore:
    """Azure Cosmos DB (NoSQL), free tier: one database at a fixed 1,000 RU/s shared by its five
    containers, each partitioned by `id` (after Google Cloud refused billing
    sign-up with OR_BACR2_59). Signs in by identity, never a password (DefaultAzureCredential):
    on Azure the app's own managed identity, locally the developer's `az login`. Each needs the
    "Cosmos DB Built-in Data Contributor" role on the account.

    "Only one question per key" (claim) is a conditional write: replace only if
    the item is unchanged since it was read (its etag), create only if it doesn't exist. Counts
    are atomic increments (patch)."""

    def __init__(self, url, database="stats_desk", credential=None):
        self.url, self.database = url, database
        self._credential, self._client, self._containers = credential, None, {}

    async def _c(self, name):
        """A container client, made on first use, inside the running event loop."""
        if self._client is None:
            from azure.cosmos.aio import CosmosClient
            if self._credential is None:
                from azure.identity.aio import DefaultAzureCredential
                self._credential = DefaultAzureCredential()
            self._client = CosmosClient(self.url, credential=self._credential)
        if name not in self._containers:
            self._containers[name] = self._client.get_database_client(self.database).get_container_client(name)
        return self._containers[name]

    async def close(self):
        if self._client is not None:
            await self._client.close()
            if hasattr(self._credential, "close"):
                await self._credential.close()
            self._client, self._containers = None, {}

    @staticmethod
    def _plain(item):
        return {k: v for k, v in item.items() if not k.startswith("_")}

    async def _read(self, container, name):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        try:
            return await (await self._c(container)).read_item(name, partition_key=name)
        except CosmosResourceNotFoundError:
            return None

    async def _increment(self, container, name, path, n):
        """Add n at `path`, creating the item (or the field) when it isn't there yet."""
        from azure.cosmos.exceptions import CosmosResourceExistsError, CosmosResourceNotFoundError
        c = await self._c(container)
        try:
            await c.patch_item(name, partition_key=name, patch_operations=[{"op": "incr", "path": path, "value": n}])
        except CosmosResourceNotFoundError:
            keys = path.strip("/").split("/")
            body = {"id": name}
            node = body
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = n
            try:
                await c.create_item(body)
            except CosmosResourceExistsError:           # created by another call just now
                await c.patch_item(name, partition_key=name,
                                   patch_operations=[{"op": "incr", "path": path, "value": n}])

    # ---- briefs

    async def save(self, brief_id, doc):
        await (await self._c(BRIEFS)).upsert_item({**doc, "id": brief_id})

    async def get(self, brief_id):
        item = await self._read(BRIEFS, brief_id)
        return self._plain(item) if item else None

    async def public(self, limit=20):
        c = await self._c(BRIEFS)
        docs = [self._plain(d) async for d in c.query_items("SELECT * FROM c WHERE c.public = true")]
        return sorted(docs, key=lambda d: d.get("created", ""), reverse=True)[:limit]

    async def set_public(self, brief_id, public):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        try:
            await (await self._c(BRIEFS)).patch_item(
                brief_id, partition_key=brief_id, patch_operations=[{"op": "set", "path": "/public", "value": public}])
            return True
        except CosmosResourceNotFoundError:
            return False

    # ---- leases

    async def claim(self, name, holder):
        """None if `holder` now holds `name`; otherwise who does. Two machines claiming at once:
        one write succeeds, the other's condition fails and it reads back the winner."""
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosAccessConditionFailedError, CosmosResourceExistsError
        c = await self._c(LEASES)
        item = await self._read(LEASES, name)
        if item and item["until"] > time.time() and item["holder"] != holder:
            return item["holder"]
        body = {"id": name, "holder": holder, "until": time.time() + LEASE}
        try:
            if item is None:
                await c.create_item(body)
            else:
                await c.replace_item(name, body, etag=item["_etag"], match_condition=MatchConditions.IfNotModified)
            return None
        except (CosmosResourceExistsError, CosmosAccessConditionFailedError):
            winner = await self._read(LEASES, name)
            return winner["holder"] if winner and winner["holder"] != holder else None

    async def release(self, name, holder):
        from azure.core import MatchConditions
        from azure.cosmos.exceptions import CosmosHttpResponseError
        item = await self._read(LEASES, name)
        if item and item["holder"] == holder:
            try:
                await (await self._c(LEASES)).delete_item(name, partition_key=name, etag=item["_etag"],
                                                         match_condition=MatchConditions.IfNotModified)
            except CosmosHttpResponseError:            # taken over or gone meanwhile: not ours to delete
                pass

    async def holder(self, name):
        """Who holds `name` now, or None: one point read (1 RU)."""
        item = await self._read(LEASES, name)
        return item["holder"] if item and item["until"] > time.time() else None

    # ---- counts

    async def count(self, name):
        item = await self._read(COUNTERS, name)
        return item.get("n", 0) if item else 0

    async def add(self, name, n=1):
        await self._increment(COUNTERS, name, "/n", n)

    async def block(self, name, seconds):
        await (await self._c(BLOCKS)).upsert_item({"id": name, "until": time.time() + seconds})

    async def blocked(self, name):
        item = await self._read(BLOCKS, name)
        return bool(item) and item.get("until", 0) > time.time()

    async def add_usage(self, name, tokens):
        """One item per key, a field per hour: one atomic write per model call."""
        await self._increment(USAGE, name, f"/hours/{hour(time.time())}", tokens)

    async def usage_24h(self, name):
        item = await self._read(USAGE, name)
        hours = (item or {}).get("hours", {})
        recent = recent_hours(time.time())
        if len([h for h in hours if h not in recent]) > 48:     # keep the item small: drop hours long past
            await (await self._c(USAGE)).upsert_item({"id": name, "hours": {h: n for h, n in hours.items() if h in recent}})
        return sum(n for h, n in hours.items() if h in recent)


def from_env() -> Store:
    if os.environ.get("STATS_DESK_STORE", "memory") == "cosmos":
        return CosmosStore(os.environ["STATS_DESK_COSMOS_URL"],
                           os.environ.get("STATS_DESK_COSMOS_DATABASE", "stats_desk"))
    return MemoryStore()
