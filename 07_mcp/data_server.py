"""A second demo server, this one offering resources and prompts. Used by notebook 3.

`server.py` stays tools-only on purpose. Notebook 1's finding depends on it: a server that
*declares* the resources and prompts capabilities and has none of either, which is what most
of the ecosystem looks like. Adding data to it would have quietly deleted that lesson.

So the data lives here instead, which is also the more honest arrangement -- a weather service
and an archive of its readings are two different things, and in practice they are two servers
written by two different people. This file imports nothing from `demo_tools.py` for that
reason: a separate service does not get to reach into another one's variables, and having it
do so would have made the separation a story rather than a fact.

Three resources, chosen to make the distinction arguable rather than abstract: one small, one
far too large for a context window, and one parameterised. Plus the one prompt, so notebook 3
can look at a real one before recommending against bothering with them.
"""

import json
import sys

from mcp.server.mcpserver import MCPServer

# This archive's own record of which stations it holds. Deliberately not imported from
# `demo_tools`: the two servers agree about the world because they describe the same world,
# not because they share a variable.
STATIONS = {"tokyo": 18, "oslo": 4, "cairo": 33}

server = MCPServer("forge-data", version="0.1.0")


def log(message):
    """stdout is the protocol; stderr is not. Same rule as `server.py`."""
    print(message, file=sys.stderr, flush=True)


@server.resource("weather://stations", mime_type="application/json")
def stations() -> str:
    """The stations this service has readings for."""
    return json.dumps(sorted(STATIONS))


@server.resource("weather://history", mime_type="text/csv")
def history() -> str:
    """Every reading ever taken. Deliberately far larger than a context window wants."""
    rows = ["date,city,celsius"]
    for day in range(1, 366):
        for city, base in STATIONS.items():
            rows.append(f"2026-{(day % 12) + 1:02d}-{(day % 28) + 1:02d},{city},{base + day % 7}")
    return "\n".join(rows)


@server.resource("weather://station/{city}", mime_type="application/json")
def station(city: str) -> str:
    """One station's details. A resource template -- the URI carries the argument."""
    return json.dumps({"city": city, "celsius": STATIONS.get(city.lower()), "source": "forge-data"})


@server.prompt(description="Ask for a plain-language weather briefing for a city.")
def briefing(city: str) -> str:
    return f"Give me a short weather briefing for {city}. Mention whether I need a coat."


if __name__ == "__main__":
    server.run()
