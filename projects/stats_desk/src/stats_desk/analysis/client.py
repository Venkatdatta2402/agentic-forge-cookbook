"""The agents' side of the analysis server: start it, turn its tools into ours, keep every result
by its id, and check a hand-in's numbers against the results it cites.

One connection per question. All the question's agents -- several analyst instances at once, and
code's checks on their findings -- share it, and so share one record of results: an id one
analyst was given can be cited in the brief, and nothing else can. The server behind it is
either the API's one shared server (a URL) or, for a script, a child process of its own.
"""

import re
import sys
from contextlib import asynccontextmanager

from mcp import Client, StdioServerParameters

from ..config import RELEASE_PATH
from ..core import trace
from ..core.tools import Result, SchemaArgs, Tool, ToolError
from .server import KIND

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
RESULT_ID = re.compile(r"\b[a-z]+-[0-9a-f]{6}\b")
# an age group names people, it isn't a figure: "ages 15-29", "15‑29 year olds", "aged 15 and
# above", "15+" (a live run: the lead's brief was refused three times for "29")
DASH = r"\s*[-‐‑–—]\s*"
AGE = re.compile(
    rf"\b(?:ages?|aged)\s+\d{{1,2}}(?:{DASH}\d{{1,2}}|\s*\+|\s+(?:years?\s+)?(?:and|&)\s+(?:above|over|older))?"
    rf"|\b\d{{1,2}}(?:{DASH}\d{{1,2}}|\s*\+)?\s*(?:years?|yrs|year[-‑ ]olds?)(?:\s+(?:and|&)\s+(?:above|over|older))?"
    rf"|\b\d{{1,2}}{DASH}\d{{1,2}}(?=\s*(?:age|year|yr))"
    rf"|\b\d{{1,2}}\+", re.I)


@asynccontextmanager
async def connect(server=None, release=RELEASE_PATH):
    """`server`: a URL (the API's shared server over HTTP), or an MCPServer to talk to in this
    process (tests). Otherwise the server is started as a child process over stdio, reading
    `release`."""
    target = server or StdioServerParameters(
        command=sys.executable, args=["-m", "stats_desk.analysis.server", str(release)])
    async with Client(target) as client:
        listed = await client.list_tools()
        yield Analyses(client, listed.tools)


class Analyses:
    def __init__(self, client, listed):
        self.client = client
        self.results: dict[str, str] = {}       # id -> the result as the agent saw it
        self.calls: dict[str, tuple] = {}       # id -> (analysis, arguments) that produced it
        self.tools = [self._tool(t) for t in listed]

    def _tool(self, mcp_tool):
        async def call(**kwargs):
            reply = await self.client.call_tool(mcp_tool.name, kwargs)
            text = "\n".join(getattr(block, "text", "") for block in reply.content)
            if not reply.is_error:
                rid = text.partition("\n")[0].removeprefix("result ").strip()
                self.results[rid], self.calls[rid] = text, (mcp_tool.name, kwargs)
                return text
            if (reply.meta or {}).get(KIND) == "broken":
                raise RuntimeError(text)
            raise ToolError(text)       # refused, or arguments the server itself turned down

        return Tool(call, name=mcp_tool.name, description=mcp_tool.description or "",
                    args=SchemaArgs(mcp_tool.input_schema, mcp_tool.name))

    def tool(self, name) -> Tool:
        return next(t for t in self.tools if t.name == name)

    async def call(self, name, **args) -> str:
        """A tool called by code, not by a model (e.g. the tables list given to every analyst)."""
        with trace.tool(name, args, by="code") as observed:
            try:
                observed.output = await self.tool(name).function(**args)
            except ToolError as e:
                observed.output = Result(name, "refused", str(e))
                raise
            return observed.output

    def unbacked(self, summary: str, cited: list[str]) -> list[str]:
        """The numbers in `summary` that no cited result shows (signs aside: "fell 0.1" is
        backed by "-0.1"). A number the agent worked out itself is unbacked on purpose: the
        gap, change and relate analyses compute differences, so a finding never needs mental
        arithmetic (on day 1 a model subtracted in its head and got it wrong)."""
        unknown = [i for i in cited if i not in self.results]
        if unknown:
            raise ToolError(f"no result {', '.join(unknown)}; cite the ids your results start with")
        shown = {abs(float(n)) for i in cited
                 for n in NUMBER.findall(self.results[i].partition("\n")[2])}
        return [n for n in NUMBER.findall(AGE.sub("", RESULT_ID.sub("", summary))) if abs(float(n)) not in shown]
