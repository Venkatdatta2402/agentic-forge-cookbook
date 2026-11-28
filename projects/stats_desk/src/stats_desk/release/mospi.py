"""MoSPI's official MCP server, as the release job uses it: fetch every published row of a table.

Only the release job calls MoSPI (agents read the release file). Each behaviour here comes from
measuring the live server (2026-11-07 to 11-09):

  - **Paging.** get_data returns 20 rows a page by default; 100 a page works, 1,000 is refused.
  - **Errors arrive as success**: `is_error=False` with an "error" field inside, often listing
    the valid parameters. Raised here as MospiError with MoSPI's own message.
  - **An unpublished rate limit.** After ~150 quick page requests MoSPI's backend answered 429
    ("too many requests"). Large fetches go one page at a time with a pause (`pace`), and a 429
    is waited out (5, 15, 45 s) instead of failing the fetch.
  - **Required selectors it quietly defaults.** PLFS without frequency_code returns the annual
    table even when the monthly one was meant; such a call is refused before it is sent.
  - **`user_query` is never sent**: it's optional, and MoSPI stores it for an engagement report.
  - **A beta service**: a timeout per call, retries with backoff, a circuit breaker, a cache.
"""

import asyncio
import json
import time
from contextlib import AsyncExitStack
from typing import Protocol

from ..datasets import DATASETS

PAGE_SIZE = 100
CACHE_SECONDS = 24 * 3600
BUSY_WAITS = (5.0, 15.0, 45.0)

# selectors get_data needs inside `filters`, which MoSPI otherwise defaults: each description's
# [source] required, read from MoSPI's own schema. Seen there for datasets not yet described
# (for their descriptions): ASUSE frequency_code; NAS base_year; IIP base_year, frequency;
# WPI base_year.
REQUIRED_IN_DATA = {d.source["mcp"]: tuple(d.source.get("required", ())) for d in DATASETS.values()}


class MospiError(Exception):
    """MoSPI refused the request, couldn't be reached, or the request can't be sent as it is."""


class MospiBusy(MospiError):
    """MoSPI's backend said "too many requests" (429). Waited out, not reported."""


class Transport(Protocol):
    async def call(self, tool: str, args: dict) -> dict: ...


class LiveTransport:
    """One MCP session to MoSPI, reopened if it breaks. Call `close()` when done."""

    def __init__(self, url, concurrency=2, timeout=60.0, retries=2):
        self.url, self.timeout, self.retries = url, timeout, retries
        self._sem = asyncio.Semaphore(concurrency)
        self._stack: AsyncExitStack | None = None
        self._client = None
        self._opening = asyncio.Lock()

    async def _open(self):
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        async with self._opening:
            if self._client is None:
                stack = AsyncExitStack()
                self._client = await stack.enter_async_context(
                    Client(streamable_http_client(self.url), read_timeout_seconds=self.timeout))
                self._stack = stack
        return self._client

    async def close(self):
        stack, self._stack, self._client = self._stack, None, None
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:  # noqa: BLE001 -- closing a broken session can itself fail
                pass

    async def call(self, tool, args):
        last = None
        async with self._sem:
            for attempt in range(self.retries + 1):
                try:
                    client = await self._open()
                    result = await asyncio.wait_for(client.call_tool(tool, args), self.timeout)
                    return _payload(result)
                except Exception as e:  # noqa: BLE001 -- the session may be dead; reopen it
                    last = e
                await self.close()
                await asyncio.sleep(1.5 * (attempt + 1))
        raise ConnectionError(f"MoSPI did not answer {tool}: {type(last).__name__}: {last}")


def _payload(result):
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    text = "\n".join(getattr(c, "text", "") for c in result.content)
    try:
        return json.loads(text)
    except ValueError:
        return {"_text": text}


class Breaker:
    """After `threshold` failures in a row, refuse at once for `cooldown` seconds."""

    def __init__(self, threshold=3, cooldown=60.0, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures = 0
        self.opened_at = None

    @property
    def open(self):
        if self.opened_at is None:
            return False
        if self.clock() - self.opened_at >= self.cooldown:
            self.opened_at, self.failures = None, 0       # half-open: let the next call try
            return False
        return True

    def success(self):
        self.failures, self.opened_at = 0, None

    def failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock()


def problem(body) -> str | None:
    """The error MoSPI put inside a 'successful' response, with its hints, or None."""
    if not isinstance(body, dict):
        return None
    if "error" not in body and body.get("valid") is not False and body.get("statusCode") is not False:
        return None
    parts = [str(body.get("error") or body.get("msg") or body.get("message") or "MoSPI refused the request")]
    for key in ("invalid_params", "valid_params", "hint", "suggestion", "troubleshooting"):
        if body.get(key):
            value = body[key]
            parts.append(f"{key}: {', '.join(map(str, value)) if isinstance(value, list) else value}")
    return " | ".join(parts)[:900]


class Mospi:
    def __init__(self, transport: Transport, breaker: Breaker | None = None, clock=time.monotonic,
                 sleep=asyncio.sleep):
        self.transport = transport
        self.breaker = breaker or Breaker()
        self.clock = clock
        self.sleep = sleep
        self._cache: dict[str, tuple[float, dict]] = {}

    async def _call(self, tool, args):
        args = {k: v for k, v in args.items() if v is not None and k != "user_query"}
        key = f"{tool}:{json.dumps(args, sort_keys=True)}"
        hit = self._cache.get(key)
        if hit and self.clock() - hit[0] < CACHE_SECONDS:
            return hit[1]
        if self.breaker.open:
            raise MospiError("MoSPI is not responding right now (it is a beta service)")
        try:
            body = await self.transport.call(tool, args)
        except Exception as e:  # noqa: BLE001
            self.breaker.failure()
            raise MospiError(f"MoSPI could not be reached ({type(e).__name__}: {e})") from e
        self.breaker.success()
        trouble = problem(body)
        if trouble:
            raise (MospiBusy if "429" in trouble else MospiError)(f"MoSPI refused the request: {trouble}")
        self._cache[key] = (self.clock(), body)
        return body

    async def _page(self, dataset, filters, page):
        """One page of get_data, waiting out MoSPI's 'too many requests' a few times."""
        args = {"dataset": dataset, "filters": {**filters, "limit": str(PAGE_SIZE),
                                                **({"page": str(page)} if page > 1 else {})}}
        for wait in (*BUSY_WAITS, None):
            try:
                return await self._call("get_data", args)
            except MospiBusy:
                if wait is None:
                    raise
                await self.sleep(wait)

    async def years(self, dataset, **filters) -> list[str]:
        """The years MoSPI lists for a table, newest first ("PLFS 2025", "PLFS 2024", ... or "2026",
        "2025"): read from its metadata, so a new year is seen whatever it is called."""
        body = await self._call("get_metadata", {"dataset": dataset, **filters})
        return [y["year"] for y in ((body.get("filter_values") or {}).get("data") or {}).get("year") or []]

    async def count(self, dataset, filters: dict) -> int:
        """How many rows MoSPI has for these filters, from one page's meta data; 0 when it says
        "No Data Found" (not published yet). An outage raises: it must never read as "nothing new"."""
        body = await self._page(dataset, filters, 1)
        if not body.get("data") and "no data" in str(body.get("msg", "")).lower():
            return 0
        return int((body.get("meta_data") or {}).get("totalRecords") or len(body.get("data") or []))

    async def data(self, dataset, filters: dict, max_rows=20_000, pace=0.4) -> list[dict]:
        """Every row matching `filters`, one page at a time with `pace` seconds between pages."""
        filters = {k: v for k, v in filters.items() if k not in ("limit", "page", "user_query")}
        missing = [k for k in REQUIRED_IN_DATA.get(dataset, ()) if filters.get(k) in (None, "")]
        if missing:
            raise MospiError(f"{dataset} needs {', '.join(missing)} in filters; left out, MoSPI "
                             "quietly picks a default that may not be the table meant")
        first = await self._page(dataset, filters, 1)
        rows = list(first.get("data") or [])
        meta = first.get("meta_data") or {}
        total = int(meta.get("totalRecords") or len(rows))
        pages = int(meta.get("totalPages") or 1)
        if total > max_rows:
            raise MospiError(f"{total:,} rows match these filters (the limit is {max_rows:,})")
        if not rows:
            raise MospiError("no rows match these filters")
        for p in range(2, pages + 1):
            await self.sleep(pace)
            rows.extend((await self._page(dataset, filters, p)).get("data") or [])
        return rows
