"""The questions on this machine: starting them, running them, saving them, carrying them on.

How a question runs: in a task of its own, its events going into a queue the stream reads. Its
record (agents/record.py) is saved when it starts, after each finding and at the end. Closing
the page stops it (decided 2026-11-13): a serverless host (Azure Container Apps; Cloud Run when
this was first written) keeps a machine only while requests are open and scales it away after,
so a question left running after its stream closed would be cut off. What it found by then is
saved, status "stopped", and it can be carried on.

Keys (DESIGN.md section 7):
  - the user's own: checked first with Groq's model list (no tokens), used for this question
    only (llm.using_key), never stored or logged.
  - none: a free question on one of the desk's keys (config.FREE_KEYS, one per Groq
    organisation, used by the desk alone). Decided 2026-11-13 (user):
      - the only count is per visitor: FREE_PER_VISITOR_PER_DAY (3), new questions and
        follow-ups together, by a hash of their address;
      - no fixed count per key: the desk counts each key's tokens over the last 24 hours (it is
        the keys' only user, so its count is their whole use) and starts a question on the key
        with the most left, not on one with under FREE_MIN_LEFT;
      - a key that runs dry mid-question anyway (Groq's "daily limit": the backstop for when the
        count is off) is marked spent until Groq's Retry-After, and the question carries on on
        the other key; with none left it pauses, saved, and the visitor decides: carry it on
        with their own key, or leave it.
One question at a time per key, whichever machine it reaches (a lease in the store: a free
key's 8,000 tokens a minute can't run two). At most MAX_RUNNING questions on one machine.
"""

import asyncio
import logging
import os
from datetime import UTC, datetime

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..agents.lead import investigate
from ..agents.record import Record, now
from ..core import events, llm
from ..core.governor import DailyLimitReached, account_of
from .progress import step
from .store import Store, new_id

log = logging.getLogger("stats_desk.api")

MAX_RUNNING = int(os.environ.get("STATS_DESK_MAX_RUNNING", 10))   # per machine; to measure once deployed
BUSY_RETRY = 120     # seconds, in Retry-After
OWN_KEY = "Add your own Groq key (free at console.groq.com/keys) to ask more."


class Ask(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    follow_up_of: str | None = Field(default=None, description="the id of the brief this follows up")


class KeyRefused(Exception):
    """The key can't be used, said so the user can act on it."""


async def check_groq_key(key: str) -> None:
    """Groq's model list: no tokens, and a mistyped key fails in a second, not mid-question."""
    client = llm.GroqBackend(api_key=key)
    try:
        models = await client.models()
    except llm.AuthenticationError:
        raise KeyRefused("Groq didn't accept this key. Check it at console.groq.com/keys.") from None
    finally:
        await client.close()
    missing = {config.LEAD_MODEL, config.ANALYST_MODEL} - models
    if missing:
        raise KeyRefused(f"This key can't use {', '.join(sorted(missing))}, which the desk runs on.")


def _leaf(error):
    """The error itself, out of the exception groups the MCP client's task group wraps it in."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


def plain_error(error) -> str:
    if isinstance(_leaf(error), llm.AuthenticationError):
        return "Groq stopped accepting the key during the question."
    return "Something went wrong on our side, and the question stopped. Try again in a minute."


def today() -> str:
    return datetime.now(UTC).date().isoformat()


class Desk:
    def __init__(self, store: Store, connect, check_key, release: str, max_running: int = MAX_RUNNING,
                 free_keys=(), free_per_visitor=config.FREE_PER_VISITOR_PER_DAY,
                 free_per_day=config.FREE_TOKENS_PER_DAY, free_min_left=config.FREE_MIN_LEFT):
        self.store, self.connect, self.check_key = store, connect, check_key
        self.release, self.max_running = release, max_running
        self.free_keys, self.free_per_visitor = list(free_keys), free_per_visitor
        self.free_per_day, self.free_min_left = free_per_day, free_min_left
        self.active = 0                            # questions running on this machine
        self.tasks: set[asyncio.Task] = set()      # held, so a question isn't collected mid-run

    async def save(self, record: Record):
        await self.store.save(record.id, record.to_doc())

    # ------------------------------------------------------------------ starting

    async def ask(self, ask: Ask, key: str | None, visitor: str):
        """Start a question. Returns its record, the queue its events go to, and its task."""
        async def make():
            parent = await self._parent(ask.follow_up_of)
            record = (Record.follow_up(new_id(), ask.question, parent, self.release) if parent
                      else Record(new_id(), ask.question, self.release))
            return record, True                    # a new question or a follow-up: counted
        return await self._start(make, key, visitor)

    async def resume(self, brief_id: str, key: str | None, visitor: str):
        """Carry on a question that stopped partway, from its record."""
        async def make():
            doc = await self.store.get(brief_id)
            if doc is None:
                raise HTTPException(404, f"No brief {brief_id}.")
            record = Record.from_doc(doc)
            if not record.resumable:
                raise HTTPException(409, f"Brief {brief_id} can't be carried on ({record.status}"
                                         f"{', carried on too often' if record.resumes >= Record.RESUMES else ''}).")
            if not key and not record.free:
                raise HTTPException(401, "This question ran on your own key: send it again in X-Groq-Key.")
            # paused because the desk's keys ran dry isn't the visitor's doing; a question they
            # stopped by closing the page counts as one of their free questions again
            counted = record.status != "paused"
            record.resumes += 1
            record.status, record.stopped, record.error, record.finished = "running", None, None, None
            return record, counted
        return await self._start(make, key, visitor)

    async def _start(self, make, key, visitor):
        if self.active >= self.max_running:
            raise HTTPException(503, "The desk is answering as many questions as it can. Try again in a "
                                     "few minutes.", headers={"Retry-After": str(BUSY_RETRY)})
        self.active += 1                           # held before any await, so two asks can't both pass
        lease, record = None, None
        try:
            record, counted = await make()
            if key:
                await self.check_key(key)
                holder = await self.store.claim(account_of(key), record.id)
                if holder:
                    raise HTTPException(409, f"A question on this key is still running: /briefs/{holder}")
                lease, record.free = account_of(key), False
            else:
                key = await self._free_key(visitor, record.id, counted)
                lease, record.free = account_of(key), True
            await self.save(record)
        except KeyRefused as e:
            self.active -= 1
            raise HTTPException(401, str(e)) from None
        except BaseException:
            self.active -= 1
            if lease:
                await self.store.release(lease, record.id)
            raise
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self._run(record, key, lease, queue))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return record, queue, task

    async def _parent(self, parent_id) -> Record | None:
        if parent_id is None:
            return None
        doc = await self.store.get(parent_id)
        if doc is None:
            raise HTTPException(404, f"No brief {parent_id}.")
        parent = Record.from_doc(doc)
        if parent.status != "done":
            raise HTTPException(409, f"Brief {parent_id} has no answer to follow up ({parent.status}).")
        return parent

    # ------------------------------------------------------------------ the free keys

    async def _free_key(self, visitor, holder, counted) -> str:
        if not self.free_keys:
            raise HTTPException(401, "Send your Groq key in the X-Groq-Key header (free at console.groq.com/keys).")
        counter = f"visitor-{visitor}-{today()}"
        if counted and await self.store.count(counter) >= self.free_per_visitor:
            raise HTTPException(429, f"You've used today's {self.free_per_visitor} free question"
                                     f"{'s' if self.free_per_visitor > 1 else ''} (follow-ups included). {OWN_KEY}")
        key, why = await self._take_free_key(holder)
        if key is None and why == "spent":
            raise HTTPException(429, f"Today's free questions are used up. {OWN_KEY}")
        if key is None:
            raise HTTPException(503, f"The free questions are all running right now; try again in a few "
                                     f"minutes. {OWN_KEY}", headers={"Retry-After": str(BUSY_RETRY)})
        if counted:
            await self.store.add(counter)
        return key

    async def tokens_left(self, key) -> int:
        """What the key has left of its day, by the desk's own count; 0 if Groq said it's spent."""
        if await self.store.blocked(f"spent-{account_of(key)}"):
            return 0
        return max(0, self.free_per_day - await self.store.usage_24h(f"tokens-{account_of(key)}"))

    async def _take_free_key(self, holder, skip=None) -> tuple[str | None, str | None]:
        """The free key with the most left (at least free_min_left) that no other question holds:
        (key, None); or (None, "spent") if none has enough left, (None, "busy") if those that
        have are in use."""
        open_ = [(await self.tokens_left(k), k) for k in self.free_keys if k != skip]
        open_ = [(left, k) for left, k in open_ if left >= self.free_min_left]
        for _, key in sorted(open_, key=lambda p: -p[0]):     # most left first; ties in list order
            if await self.store.claim(account_of(key), holder) is None:
                return key, None
        return None, ("busy" if open_ else "spent")

    # ------------------------------------------------------------------ running

    async def _run(self, record: Record, key, lease, queue):
        held = {"lease": lease}

        def listen(event):
            said = step(event)
            if said:
                queue.put_nowait(("step", {"step": said}))

        async def spare(spent_key, retry_after):
            """A free key's day ran out mid-question: mark it spent (every machine skips it until
            Groq's Retry-After) and carry on on another free key, if one has enough left."""
            await self.store.block(f"spent-{account_of(spent_key)}", retry_after)
            await self.store.release(account_of(spent_key), record.id)
            new, _ = await self._take_free_key(record.id, skip=spent_key)
            held["lease"] = account_of(new) if new else None
            if new:
                record.key_switches += 1
                queue.put_nowait(("step", {"step": "One free key's allowance for today ran out; "
                                                   "carrying on with the other free key"}))
            return new

        async def used(on_key, tokens):
            await self.store.add_usage(f"tokens-{account_of(on_key)}", tokens)

        try:
            async with llm.using_key(key, *((spare, used) if record.free else ())):
                with events.listening(listen):
                    async with self.connect() as analyses:
                        await investigate(analyses, via="api", record=record, save=self.save,
                                          tags=("free",) if record.free else ())
        except asyncio.CancelledError:
            record.status, record.stopped = "stopped", "The page was closed before the brief was ready."
            raise
        except Exception as error:                     # noqa: BLE001 -- said to the user, logged by type
            leaf = _leaf(error)
            log.warning("question %s stopped: %s", record.id, type(leaf).__name__)
            if isinstance(leaf, DailyLimitReached):    # every key it could use is spent: saved, to carry on
                record.status, record.stopped = "paused", (
                    "Today's free allowance ran out partway through this question. What it found so far "
                    f"is saved. To finish it, carry it on with your own Groq key (free at "
                    "console.groq.com/keys), or leave it as it is." if record.free else
                    "Your Groq key's daily allowance ran out partway through. What it found so far is "
                    "saved: carry it on when the allowance frees up, or with another key.")
            else:
                record.status, record.error = "failed", plain_error(error)
        finally:
            self.active -= 1
            record.finished = record.finished or now()
            await asyncio.shield(self._close(record, held["lease"]))
        if record.status == "failed":
            queue.put_nowait(("failed", {"status": "failed", "error": record.error}))
        elif record.status == "paused":
            queue.put_nowait(("paused", {"id": record.id, "message": record.stopped,
                                         "carry_on": f"/briefs/{record.id}/resume"}))
        else:
            queue.put_nowait(("brief", record.view()))

    async def _close(self, record, lease):
        try:
            await self.save(record)
        except Exception as error:                     # noqa: BLE001
            log.error("brief %s not saved: %s", record.id, type(error).__name__)
        if lease:
            try:
                await self.store.release(lease, record.id)
            except Exception as error:                 # noqa: BLE001 -- it expires on its own
                log.error("lease for %s not released: %s", record.id, type(error).__name__)
