"""Keeps every request in the process inside Groq's limits, instead of finding them by error.

The repo's clients relied on the SDK retrying 429s. That works for one notebook; for a server it
means concurrent questions collide at the limit and all back off together, and a request over the
per-request cap comes back as a 413 that no retry fixes (05_context notebook 1 hit exactly that).

**Groq's per-minute limit is a refilling bucket, not a 60-second window.** Measured: after a
94-token request, `x-ratelimit-reset-tokens` read `705ms` -- 94 tokens at 8,000/60 per second is
0.705 s. The first version of this module modelled a sliding minute, which held every token for a
full 60 s when Groq gives them back at 133 a second: one analyst run spent 415 of its 425 seconds
waiting here, and 10 in the model.

So: a bucket of `tpm` tokens, refilled at tpm/60 per second. A call takes `prompt + max_tokens`
from it up front (waiting until that much is there), and is refunded what it didn't use once the
API says what it used. Groq's own remaining-tokens header is believed when it's lower than ours:
other processes on the same account spend from the same bucket, and only the header sees them.

Refused before anything is sent, because waiting can't fix them:
    TooLarge             the request alone is over the per-request cap or the bucket's size
    DailyLimitReached    it would go past the rolling 24-hour budget

**One bucket per Groq account and model**: on the web each user brings their own key,
and each key has its own limits at Groq. The caller names the account (`llm.complete` passes the
key its question is on at that moment, which can change mid-question when a free key runs out),
so a question's agents all draw on its key's bucket and nobody else's. An account is a hash of
the key, never the key. For a user's key the app counts
no day (Groq's 429 says when it's spent, and their key may be on a paid plan), and the minute's
size is taken from Groq's `x-ratelimit-limit-tokens` header once it is seen. The server's own key
(scripts, evals: no account set) keeps the free tier's daily count.
"""

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass, replace

from ..config import Limits, limits_for

DAY = 86_400.0
IDLE = 3_600.0          # a user's bucket untouched this long is forgotten (it would be full again)


def account_of(key: str) -> str:
    """A key's account name in the governor: a hash, so no key is kept beyond its question."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class TooLarge(Exception):
    """This request can never be sent as it is: make it smaller."""


class DailyLimitReached(Exception):
    """The model's daily budget is spent. Waiting minutes won't help; `retry_after` says how long."""

    def __init__(self, message, retry_after):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class Ticket:
    model: str
    reserved: int
    day_entry: list      # [wall time, tokens] inside the day window, settled in place
    account: str | None = None


class _Model:
    def __init__(self, limits: Limits, now, count_day=True):
        self.limits = limits
        self.rate = limits.tpm / 60.0
        self.level = float(limits.tpm)
        self.updated = now
        self.day = deque()        # [wall, tokens]
        self.count_day = count_day
        self._lock, self._loop = None, None

    @property
    def lock(self) -> asyncio.Lock:
        """The bucket's line, for the event loop running now. The governor outlives a loop: the
        eval runs each question in its own asyncio.run, and a lock from the last one refused
        ("bound to a different event loop"). The bucket's level carries over."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._lock is None or (loop is not None and loop is not self._loop):
            self._lock, self._loop = asyncio.Lock(), loop
        return self._lock

    def resize(self, tpm):
        """Groq's header says this key's minute holds `tpm` (a paid plan holds more)."""
        if tpm and tpm != self.limits.tpm:
            self.level = self.level * tpm / self.limits.tpm
            self.limits = replace(self.limits, tpm=tpm, per_request=max(self.limits.per_request, tpm))
            self.rate = tpm / 60.0

    def refill(self, now):
        self.level = min(self.limits.tpm, self.level + (now - self.updated) * self.rate)
        self.updated = now

    def prune_day(self, wall):
        self.day = deque(e for e in self.day if e[0] > wall - DAY)

    def used_day(self):
        return sum(e[1] for e in self.day)


class Governor:
    def __init__(self, limits=None, clock=time.monotonic, wall=time.time, sleep=asyncio.sleep):
        self._limits = dict(limits or {})
        self._models: dict[tuple[str | None, str], _Model] = {}
        self.clock, self.wall, self.sleep = clock, wall, sleep
        self.waiting = 0          # requests currently queued, for /usage
        self.waited = 0.0         # seconds spent waiting, in total

    def _model(self, name, who=None):
        key = (who, name)
        if key not in self._models:
            now = self.clock()
            # forget users' buckets left idle (refilled by now, so the same as new), keep the server's
            for k in [k for k, m in self._models.items()
                      if k[0] is not None and now - m.updated > IDLE and not m.lock.locked()]:
                del self._models[k]
            self._models[key] = _Model(self._limits.get(name) or limits_for(name), now,
                                       count_day=who is None)
        return self._models[key]

    async def reserve(self, model, tokens, request=None, account=None, on_wait=None) -> Ticket:
        """Take `tokens` from the bucket. `request` is the whole request (prompt + max_tokens),
        checked against the per-request cap; it defaults to `tokens`. `account`: whose bucket
        (account_of(key)); None is the server's own key. `on_wait`: called once, with the
        seconds it expects, if the call has to wait (in line behind another, or for room)."""
        who = account
        m = self._model(model, who)
        lim = m.limits
        request = tokens if request is None else request
        if request > lim.per_request:
            raise TooLarge(f"{request:,} tokens is over {model}'s {lim.per_request:,}-token "
                           "per-request limit")
        if tokens > lim.tpm:
            raise TooLarge(f"{tokens:,} tokens is over {model}'s {lim.tpm:,} tokens-per-minute limit")

        told = False

        def tell(seconds):
            nonlocal told
            if on_wait is not None and not told:
                told = True
                on_wait(seconds)

        self.waiting += 1
        try:
            if m.lock.locked():
                tell(None)                          # in line behind another call
            # one reserver at a time per model, so requests are admitted in the order they asked
            async with m.lock:
                while True:
                    now, wall = self.clock(), self.wall()
                    m.refill(now)
                    m.prune_day(wall)
                    if m.count_day and m.used_day() + tokens > lim.tpd:
                        oldest = min((e[0] for e in m.day), default=wall)
                        raise DailyLimitReached(
                            f"{model} has used {m.used_day():,} of its {lim.tpd:,} tokens in the "
                            "last 24 hours", retry_after=max(1.0, oldest + DAY - wall))
                    if m.level >= tokens:
                        m.level -= tokens
                        day_entry = [wall, tokens]
                        m.day.append(day_entry)
                        return Ticket(model, tokens, day_entry, who)
                    pause = max(0.05, (tokens - m.level) / m.rate)
                    if pause >= 1:
                        tell(round(pause))
                    self.waited += pause
                    await self.sleep(pause)
        finally:
            self.waiting -= 1

    def room(self, model, account=None) -> float:
        """Tokens the bucket could give now; a bucket with calls waiting in line counts as a
        whole minute short (llm.QuestionKeys sends each call to the key with the most)."""
        m = self._model(model, account)
        m.refill(self.clock())
        return m.level - (m.limits.tpm if m.lock.locked() else 0)

    def settle(self, ticket: Ticket, used=None, remaining=None, limit=None):
        """Refund what wasn't used, and believe Groq's header when it says there's less left.
        `limit`: Groq's minute size for this key (header), applied to a user's bucket.

        `used` is None when the request failed before the API counted anything we can see; the
        reservation then stands, which errs on the side of waiting a little too long.
        """
        m = self._model(ticket.model, ticket.account)
        m.refill(self.clock())
        if ticket.account is not None:
            m.resize(limit)
        if used is not None:
            # a refund if it used less than reserved; a debt (the level can go below zero) if
            # it used more, which the next reservation then waits out
            m.level = min(m.limits.tpm, m.level + ticket.reserved - used)
            ticket.day_entry[1] = used
        if remaining is not None and remaining < m.level:
            m.level = float(remaining)

    def snapshot(self):
        now, wall = self.clock(), self.wall()
        out = {"waiting": self.waiting, "waited_seconds": round(self.waited, 1), "models": {},
               "user_buckets": sum(1 for who, _ in self._models if who is not None)}
        for (who, name), m in self._models.items():
            if who is not None:
                continue
            m.refill(now)
            m.prune_day(wall)
            out["models"][name] = {"minute_available": int(m.level), "minute_limit": m.limits.tpm,
                                   "day_used": m.used_day(), "day_limit": m.limits.tpd}
        return out


governor = Governor()
