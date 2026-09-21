"""Connecting to an MCP server over stdio, and what that actually costs on Windows.

Two things live here, and both are things every notebook in this chapter would otherwise have
to repeat.

`stdio_params()` decides how the server process gets launched -- which interpreter, which
working directory, what it inherits.

`connect()` decides where that process's stderr goes, which sounds like a detail and is
actually the difference between this chapter running in a notebook and not running in one at
all. See `JUPYTER`.

`05_mcp_transport.ipynb` takes both apart.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
SERVER = HERE / "server.py"
# A second server, offering resources and prompts. Kept separate so `server.py` can stay
# tools-only -- notebook 1's point is a server that declares both capabilities and has neither.
DATA_SERVER = HERE / "data_server.py"
SERVER_LOG = HERE / "server.log"

JUPYTER = """\
A stdio MCP server cannot be launched from a Jupyter kernel on Windows using the default
settings, and the traceback does not say why. It ends in `io.UnsupportedOperation: fileno`,
raised from inside `subprocess.Popen`.

The chain: ipykernel on Windows runs a SelectorEventLoop, because that is what pyzmq needs.
Selector loops have no async subprocess support, so mcp catches the NotImplementedError and
falls back to a plain `subprocess.Popen`. Popen has to hand the child a real OS handle for its
stderr -- and in a kernel, `sys.stderr` is an ipykernel OutStream that forwards to the notebook
over a socket. It has no file descriptor to give.

So the fix is not about async at all: give the child a real file to write its stderr to. That
is what `connect()` does, and the file is worth having anyway -- a stdio server's stderr is the
only place its crashes go.
"""

ENCODING = """\
Setting PYTHONIOENCODING=utf-8 on the child is the standard advice for Windows stdio servers,
and under mcp 2.x it does nothing. Measured: `café — em dash ✓ 日本` round-trips byte-identical
with the child forced to cp1252, and to ascii. The 2.x stdio transport reads and writes the
pipe as bytes and frames UTF-8 itself, so the console code page is never consulted.

The advice is real but it is about mcp 1.x, where the server wrapped sys.stdin and sys.stdout
as text streams and inherited the code page from the console. Worth keeping straight, because
the failure it used to cause -- one em dash in a tool result killing the connection rather than
the call -- is the kind that gets blamed on the protocol.
"""


INHERITANCE = """\
The SDK is careful here and it is easy to undo by accident.

`stdio_client` builds the child's environment as `get_default_environment() | server.env` --
an allowlist (`DEFAULT_INHERITED_ENV_VARS`: PATH, TEMP, USERPROFILE and similar, none of them
secrets), with whatever `StdioServerParameters.env` names merged on top. Left alone, a stdio
server starts with no credentials at all.

Passing `env={**os.environ}` overrides that allowlist with everything, API keys included. It
is the obvious thing to write, it is what a great many real MCP configurations do, and it is
the reason a stdio server ends up holding every secret on the machine. The protocol did not
do that; the client did.

So `inherit=` here is a real switch and not a wrapper's convenience. True reproduces the
common mistake, which is the default because notebook 5 has to measure the thing everyone
actually runs. False passes only what the caller names and lets the SDK's allowlist stand.
"""


def stdio_params(script=SERVER, args=(), env=None, inherit=True):
    """The parameters for launching `script` as a stdio MCP server.

    `sys.executable` rather than "python". There is no `python` on PATH in a conda env on this
    machine unless the env is activated, and a notebook kernel is not an activated shell.
    `sys.executable` is the interpreter already running, which is by definition the one that
    has `mcp` installed.

    `cwd` pinned to this folder, so a server launched from wherever the notebook happens to be
    running can still import its own neighbours.

    `inherit` decides what the child receives -- see `INHERITANCE` above, which is the part of
    this module worth reading. True hands it this whole process's environment and is the
    default because it is what the ecosystem does. False names nothing extra and leaves the
    SDK's own allowlist in charge.
    """
    return StdioServerParameters(
        command=sys.executable,
        args=[str(script), *args],
        cwd=str(HERE),
        env={**os.environ, **(env or {})} if inherit else dict(env or {}),
    )


def can_be_inherited(stream):
    """Whether a child process can be handed this stream as its stderr.

    True for a real file or console. False inside a Jupyter kernel, where stderr is a socket
    wearing a file's interface -- it answers `write` and raises on `fileno`.
    """
    try:
        stream.fileno()
        return True
    except Exception:
        return False


@asynccontextmanager
async def connect(script=SERVER, args=(), env=None, timeout=None, errlog=None, inherit=True,
                  **kwargs):
    """One live session with a stdio server.

        async with connect() as client:
            tools = await client.list_tools()

    Leaving the block closes the child's stdin, which is the only way a stdio server is told to
    exit -- there is no port to close and no shutdown message. A session that is never closed
    leaves a Python process behind.

    `errlog` is where the server's stderr goes. Left alone it goes to this process's stderr
    when that can be inherited, and to `server.log` when it cannot, which is the Jupyter case
    described in `JUPYTER`. Anything the server prints outside the protocol ends up in one of
    those two places and nowhere else.
    """
    opened = None
    if errlog is None:
        if can_be_inherited(sys.stderr):
            errlog = sys.stderr
        else:
            opened = errlog = open(SERVER_LOG, "a", encoding="utf-8")

    try:
        transport = stdio_client(stdio_params(script, args, env, inherit), errlog=errlog)
        async with Client(transport, read_timeout_seconds=timeout, **kwargs) as client:
            yield client
    finally:
        if opened is not None:
            opened.close()
