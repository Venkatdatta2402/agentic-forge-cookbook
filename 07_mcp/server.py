"""The chapter's demo MCP server. Run by the notebooks, imported by none of them.

It speaks MCP over stdio, which means stdin and stdout are the wire. Nothing in this file may
print to stdout -- see `log()`.

The tools come from two places, and the split is deliberate. The ordinary ones live in
`demo_tools.py` as plain functions that know nothing about MCP, because notebook 1 needs to
describe the *same function* two ways. The ones below are here instead because they are not
really about weather -- they are about what a failure looks like from the far side of a process
boundary, which means they have to speak MCP's own result type.
"""

import sys

import mcp.types as t
from mcp.server.mcpserver import MCPServer

from demo_tools import READINGS, echo, raises, weather, whoami

server = MCPServer("forge-demo", version="0.1.0")


def log(message):
    """The only safe way to say anything from inside a stdio server.

    stdout is the protocol. A bare print() puts plain text where the client expects a JSON-RPC
    frame, and the client fails parsing something it never asked for -- so the error surfaces
    far from the print that caused it. stderr is not part of the wire, so it is free. In a
    notebook it lands in `server.log`; see `client.py`.
    """
    print(message, file=sys.stderr, flush=True)


for fn in (weather, echo, whoami, raises):
    # description= is left off on purpose: MCPServer reads the docstring, exactly as Tool does
    # in 03_tools. One more thing neither side has to be kept in sync by hand.
    server.add_tool(fn)


# --- three ways a tool can fail, and what reaches the client for each -------------------
#
# `raises` above is the first: it lets a ValueError escape and the SDK replaces it with a
# generic message. The two below are what a server author does instead.


@server.tool(description="Catches its own failure and says what happened.")
def described(city: str) -> t.CallToolResult:
    if city.lower() not in READINGS:
        return t.CallToolResult(
            content=[
                t.TextContent(
                    type="text",
                    text=f"No weather station for {city}. Try a larger nearby city.",
                )
            ],
            is_error=True,
        )
    return t.CallToolResult(content=[t.TextContent(type="text", text=weather(city))])


@server.tool(description="Splits the audience: a short message to the model, detail to the host.")
def detailed() -> t.CallToolResult:
    """Two audiences, one result.

    `content` is what enters the conversation. `meta` rides alongside it and never does --
    the host can read it, the model cannot. A stack trace is for whoever operates the agent,
    not for the thing being prompted.

    The keys are namespaced because `_meta` is shared ground: the SDK puts its own
    `io.modelcontextprotocol/serverInfo` in the same dict.
    """
    return t.CallToolResult(
        content=[t.TextContent(type="text", text="Weather service unavailable.")],
        is_error=True,
        meta={
            "forge/cause": "ConnectionResetError at pool.py:412, retry 3/3 exhausted",
            "forge/kind": "broken",
            "forge/repeatable": False,
        },
    )


if __name__ == "__main__":
    # run() defaults to stdio and owns the event loop. The process stays alive until the client
    # closes stdin -- there is no port, no address, and nothing else to shut down.
    server.run()
