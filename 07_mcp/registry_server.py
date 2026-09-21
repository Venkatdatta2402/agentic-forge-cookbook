"""A server built by `expose()` from an ordinary `03_tools` Registry. Used by notebook 4.

Everything here is a plain local tool of the kind chapter 3 built -- one that works, one that
refuses, one that is broken, and one whose claim about the world is checked afterwards. None
of them know they are reachable over a protocol, which is the point: `expose()` is the only
thing in this file that has heard of MCP.
"""

import sys
from pathlib import Path

# This server is launched as a subprocess, so it gets no path setup from the notebook that
# started it. `03_tools` is a real dependency here and not a notebook convenience: the whole
# file is chapter 3's tools, published.
sys.path.append(str(Path(__file__).resolve().parent.parent / "03_tools"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "01_foundations"))

from invoke import ToolError
from registry import Registry
from tools import tool
from verify import verified

from expose import expose

READINGS = {"tokyo": 18, "oslo": 4, "cairo": 33}
SAVED = {}


@tool
def weather(city: str, units: str = "celsius") -> str:
    """Look up the current weather in a city."""
    reading = READINGS.get(city.lower())
    if reading is None:
        raise ToolError(f"No weather station for {city}. Try a larger nearby city.")
    return f"{reading} degrees {units}, clear skies in {city}."


@tool
def pressure(city: str) -> str:
    """Look up the barometric pressure. Has a bug in it."""
    return f"{1013 + READINGS[city]} hPa"  # KeyError on any city not in READINGS


@tool(repeatable=False)
def save_note(name: str, text: str) -> str:
    """Save a note. Deliberately truncates, so `verified()` has something to catch."""
    SAVED[name] = text[:20]
    return f"Saved {len(text)} characters as {name!r}."


def note_was_saved_whole(args, value):
    """The check that runs after `save_note`, looking at the store rather than at the answer.

    Returns a message when the world disagrees with what the tool claimed, and None otherwise.
    Note what it reads: `SAVED`, not `value`. A check that believed the tool's own report would
    catch nothing, because the report says it worked.
    """
    stored = SAVED.get(args["name"], "")
    if stored != args["text"]:
        return f"{args['name']!r} holds {len(stored)} of {len(args['text'])} characters."


registry = Registry([weather, pressure, verified(save_note, note_was_saved_whole)])

# A time limit and a size limit, applied on this side of the wire. Notebook 4 is about which
# of chapter 3's guarantees can be made here and which cannot.
server = expose(registry, name="forge-registry", timeout=10, max_tokens=400)


def log(message):
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    server.run()
