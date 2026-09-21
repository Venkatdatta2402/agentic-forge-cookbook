"""A server for looking at the transport itself rather than at what a tool does. Notebook 5.

Its tools report on the process the server is running in -- what it inherited, how long it can
be made to take, what it can reach. None of them are useful tools. They are instruments.
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "03_tools"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "01_foundations"))

from registry import Registry
from tools import tool

from expose import expose


@tool
def whoami() -> str:
    """Report the process this server is running in."""
    return f"pid={os.getpid()} cwd={os.getcwd()}"


@tool
def env_count() -> str:
    """How many environment variables this server can see."""
    return f"{len(os.environ)} variables inherited"


@tool
def can_you_see(name: str) -> str:
    """Report whether a named environment variable reached this process, and its length.

    Deliberately returns the length and not the value. The point being made in notebook 5 is
    that the variable arrived at all -- printing a real secret to make that point would be a
    worse mistake than the one being illustrated.
    """
    value = os.environ.get(name)
    if value is None:
        return f"{name}: not set in this process"
    return f"{name}: present, {len(value)} characters"


@tool
def env_names(prefix: str) -> str:
    """Names of inherited variables starting with `prefix`. Never values.

    A prefix is required, and that is not an interface detail. Notebook 5's outputs are
    committed, so a tool that listed everything would publish which services this machine is
    configured for -- a smaller version of exactly the mistake the notebook is about. The
    notebook only ever asks for `FORGE_`, which it planted itself.
    """
    found = sorted(n for n in os.environ if n.startswith(prefix))
    return f"{len(found)} variable(s) starting with {prefix!r}: {', '.join(found) or '(none)'}"


@tool(repeatable=True)
def slow(seconds: float) -> str:
    """Take `seconds` to answer. For looking at who gives up first."""
    time.sleep(seconds)
    return f"finished after {seconds}s"


registry = Registry([whoami, env_count, can_you_see, env_names, slow])

# The server's own time limit. Notebook 5 sets it against the client's.
server = expose(registry, name="forge-transport", timeout=3)

if __name__ == "__main__":
    # The same server, either way down. Which transport it speaks is a launch argument and
    # changes nothing above this line -- which is the point notebook 5 makes with it.
    #
    #   python transport_server.py            -> stdio, launched by a client that owns it
    #   python transport_server.py http 8765  -> listening, owned by whoever started it
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
        server.run(transport="streamable-http", host="127.0.0.1", port=port)
    else:
        server.run()
