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

    def ping(self, timeout=None):
        """The protocol's own liveness check -- which this SDK's servers do not answer.

        `ping` is in the MCP spec and `Client.send_ping()` sends it, but a server built with
        `MCPServer` replies `MCPError: Method not found`. Kept because it is the right call to
        make against a server that implements it, and because the alternative -- see
        `Connection.healthy()` -- is only necessary once you know this one does not work.
        """
        return self._await(self._client.send_ping(), timeout)

    @property
    def server_name(self):
        return self._client.server_info.name if self._client else None


class Connection:
    """A box holding a `Session`, so the session can be replaced without rebuilding the tools.

    `remote_tool()` builds a function that closes over whatever it is given. Given a `Session`,
    the session is sealed inside it: it works, and when the server dies there is no way to hand
    that function a new one -- every tool has to be built again, and the `Registry` they were
    added to rebuilt with them.

    Given a `Connection`, the box is what gets sealed in. Replacing the session is then one
    assignment, and every tool built earlier keeps working because none of them ever held the
    session in the first place. Notebook 6 kills a server and reconnects to show it.

    It deliberately offers the same methods as `Session` -- `call`, `list_tools`, `read_resource`
    and the rest -- so nothing downstream can tell the difference, and `remote_tools()` did not
    have to change to accept one.

    Put it on a `Conversation`:

        conversation.resources["mcp"] = Connection(lambda: Session(SERVER)).open()

    and `Conversation.close()` closes it, while any component that gets the conversation can
    reach in and call `reconnect()`.
    """

    def __init__(self, factory):
        # A factory rather than a session, because reconnecting means building a *new* one and
        # a box holding a dead session has no way to make another.
        self.factory = factory
        self.session = None
        self.reconnects = 0

    # --- lifetime ------------------------------------------------------------------------

    def open(self):
        if self.session is None:
            self.session = self.factory().__enter__()
        return self

    def close(self):
        if self.session is not None:
            session, self.session = self.session, None
            session.__exit__(None, None, None)

    def reconnect(self):
        """Throw the dead session away and start a fresh one. Same box, same tools."""
        self.close()
        self.reconnects += 1
        return self.open()

    @property
    def alive(self):
        """Whether a session object exists. Says nothing about whether it works."""
        return self.session is not None

    def healthy(self, timeout=2):
        """Whether the server is actually answering -- asked, not assumed.

        `alive` is bookkeeping: it says whether we are holding a session object, which stays
        true for a session whose process died thirty seconds ago. This asks the server, which is
        the difference between believing our own records and checking the world -- the same
        distinction `verified()` makes in `03_tools`.

        It asks with `tools/list` rather than the obvious `ping`. `ping` is in the spec and the
        client can send it, but a server built with this SDK's `MCPServer` answers
        `MCPError: Method not found`, so a liveness check built on it reports every healthy
        server as dead. `tools/list` is the cheapest call every MCP server must implement.

        Cheap over stdio -- about 1.5 ms, a round trip down a local pipe -- and not free over
        HTTP, which is why the component that calls it takes a policy instead of probing on
        every turn by reflex.
        """
        if self.session is None:
            return False
        try:
            self.session.list_tools()
            return True
        except Exception:  # noqa: BLE001 -- any failure to answer means not healthy
            return False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # --- everything a Session can do, forwarded to whichever one is current --------------

    def _live(self):
        if self.session is None:
            raise RuntimeError("connection is closed; call open() or reconnect()")
        return self.session

    def call(self, name, args, timeout=None):
        return self._live().call(name, args, timeout)

    def list_tools(self):
        return self._live().list_tools()

    def list_resources(self):
        return self._live().list_resources()

    def list_resource_templates(self):
        return self._live().list_resource_templates()

    def read_resource(self, uri, timeout=None):
        return self._live().read_resource(uri, timeout)

    def list_prompts(self):
        return self._live().list_prompts()

    def get_prompt(self, name, args=None, timeout=None):
        return self._live().get_prompt(name, args, timeout)

    @property
    def server_name(self):
        return self.session.server_name if self.session else None
