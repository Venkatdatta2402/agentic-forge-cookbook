"""The plain Python functions the demo server exposes, kept apart from the server exposing them.

`server.py` registers these over MCP. Notebook 1 imports `weather` and wraps it in `03_tools`'
`Tool`, then holds the two descriptions of it side by side.

That comparison is the notebook's opening argument, and it only means anything if both sides
are describing **the same function object**. Writing `weather` out twice -- once for the server,
once for the notebook -- would make the schemas match because someone kept them matching, which
is `03_tools` notebook 1's opening failure reintroduced four chapters later. So it is written
once, here.

Nothing in this file imports mcp. These are ordinary functions; being reachable over a protocol
is something done *to* them elsewhere.
"""

import os
import sys

READINGS = {"tokyo": 18, "oslo": 4, "cairo": 33}


def weather(city: str, units: str = "celsius") -> str:
    """Look up the current weather in a city."""
    reading = READINGS.get(city.lower())
    return f"{reading} degrees {units}, clear skies in {city}."


def echo(text: str) -> str:
    """Return the text unchanged. Used to test what survives the wire."""
    return text


def whoami() -> str:
    """Report the process this tool actually runs in."""
    return f"pid={os.getpid()} executable={sys.executable}"


def raises() -> str:
    """Fail by raising, with no handling of its own."""
    raise ValueError("connection refused by upstream weather API")
