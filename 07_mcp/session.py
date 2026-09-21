"""A live MCP session that synchronous code can call into.

The problem this exists for: `02_agent_runtime` is synchronous all the way down --
`Runtime.run(conversation)` calls `Act.run(conversation)` calls the tool -- and every MCP call
is a coroutine. Something has to bridge that.

The tempting fix is to make the runtime async, and it is the wrong one. It would mean rewriting
two earlier chapters so that a later chapter can plug in, and it would put `async` in the
signature of every component anyone ever writes, whether or not they ever touch a network. A
chapter should pay its own integration costs.

The second tempting fix is `asyncio.run(client.call_tool(...))` per call. That works exactly
once per session, because it creates and destroys an event loop each time, and a session opened
in one loop cannot be used from another -- the same task-affinity that stops a session spanning
notebook cells.

So: one background thread, owning one event loop, holding the session open for its whole life.
Calls are handed to it with `run_coroutine_threadsafe`. The session is opened, used and closed
by a single task, which is the thing anyio insists on, and the caller sees ordinary blocking
functions.
"""

import asyncio
import threading
from contextlib import asynccontextmanager

from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from client import SERVER, connect


class Session:
    """One MCP session, opened on a background thread, callable from synchronous code.

        with Session() as session:
            session.list_tools()
            session.call("weather", {"city": "Oslo"})

    Not an async context manager on purpose -- the whole point is that the caller does not have
    to be async, or know that anything here is.
    """

    def __init__(self, script=SERVER, args=(), env=None, timeout=30, startup=30, url=None,
                 inherit=True):
        self.script = script
        self.args = args
        self.env = env
        # An http(s) URL instead of a script to launch. Everything else about this class is
        # unchanged by it, which is the honest summary of what choosing a transport does to
        # your code: almost nothing. What it changes is who owns the process, what it
        # inherited, and who may reach it -- none of which appear in this signature.
        self.url = url
        self.inherit = inherit
        # Two different clocks, and conflating them is a real bug rather than a tidiness point.
        # `timeout` is how long a *call* may take; `startup` is how long launching a Python
        # process may take. Sharing one number means asking for a 1-second call limit also
        # gives the interpreter 1 second to boot, which it does not manage -- so the session
        # fails to open and the error talks about a closed connection rather than about the
        # limit that actually caused it.
        self.timeout = timeout
        self.startup = startup
        self._loop = None
        self._client = None
        self._thread = None
        self._ready = threading.Event()
        self._stop = None
        self._error = None

    # --- the thread's own world ---------------------------------------------------------

    @asynccontextmanager
    async def _open(self):
        """One session, over whichever transport was asked for.

        Both branches yield the same `Client`, because a transport in this SDK is just an
        async context manager producing a read stream and a write stream. Nothing above this
        method can tell which one it got.
        """
        if self.url:
            async with Client(streamable_http_client(self.url),
                              read_timeout_seconds=self.timeout) as client:
                yield client
        else:
            async with connect(self.script, self.args, self.env, timeout=self.timeout,
                               inherit=self.inherit) as client:
                yield client

    async def _serve(self):
        """Open the session, announce it, and hold it open until asked to stop.

        Everything that touches the session happens inside this one coroutine, so the open and
        the close are the same task. Calls arriving from other threads are scheduled onto this
        loop rather than run on theirs.
        """
        self._stop = asyncio.Event()
        try:
            async with self._open() as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 -- re-raised on the calling thread below
            self._error = e
        finally:
            self._client = None
            self._ready.set()  # unblock a caller waiting on a session that failed to open

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    # --- the caller's world -------------------------------------------------------------

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-session")
        self._thread.start()
        self._ready.wait(self.startup)
        if self._client is None:
            raise RuntimeError(f"session failed to open: {self._error!r}")
        return self

    def __exit__(self, *exc):
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            # `startup`, not `timeout`: shutting the child down is process work, not call work,
            # and a session configured for one-second calls still needs longer than that to
            # close cleanly.
            self._thread.join(self.startup)
        return False

    def _await(self, coro, timeout=None):
        """Run a coroutine on the session's loop and block until it answers."""
        if self._client is None:
            raise RuntimeError("session is not open")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout or self.timeout)

    def list_tools(self):
        return self._await(self._client.list_tools()).tools

    def call(self, name, args, timeout=None):
        return self._await(self._client.call_tool(name, args), timeout)

    def list_resources(self):
        """The server's concrete resources. Does NOT include templates -- see below."""
        return self._await(self._client.list_resources()).resources

    def list_resource_templates(self):
        """Parameterised resources, which `list_resources` leaves out entirely.

        A separate call, and easy to miss: a server offering nothing but templates answers
        `resources/list` with `[]`, and a host that asks only that concludes it has no data.
        """
        return self._await(self._client.list_resource_templates()).resource_templates

    def read_resource(self, uri, timeout=None):
        return self._await(self._client.read_resource(uri), timeout)

    def list_prompts(self):
        return self._await(self._client.list_prompts()).prompts

    def get_prompt(self, name, args=None, timeout=None):
        return self._await(self._client.get_prompt(name, args or {}), timeout)

    @property
    def server_name(self):
        return self._client.server_info.name if self._client else None
