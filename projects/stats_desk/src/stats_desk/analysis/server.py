"""The analysis MCP server: the analyses over one release file, read-only, for the analysts and
code's checks on their findings.

    python -m stats_desk.analysis.server [path/to/release.duckdb]               stdio (scripts, evals)
    python -m stats_desk.analysis.server [path] --http 8765                     HTTP (the API)

The API runs ONE of these for all its questions, over HTTP on the container's loopback:
the usual shape for a server many requests share, where the first version started a
stdio copy per question (the desktop-app pattern), which cost a Python and DuckDB process each.
Sharing is safe: the server only reads, every query gets its own cursor, and each question keeps
its own results on its side (client.py). Stateless HTTP: no session to keep between calls.

How each call ends, kept apart on the wire with `_meta` (MCP's place for a server's own data),
so the agent side can tell the model's mistakes from ours:

    the analysis answered          is_error False (published only for another age group or
                                   measure: answered on the nearest one, its last line saying so)
    Refusal (wrong field, value,   is_error True, kind "refused": the message says what to do
      unpublished combination)
    anything else                  is_error True, kind "broken": a bug, logged to stderr and
                                   kept out of the conversation
"""

import inspect
import sys
import traceback

import mcp.types as types
from mcp.server.mcpserver import MCPServer

from ..config import RELEASE_PATH
from ..datasets import DATASET, words_for
from .analyses import ANALYSES, across_join, nearest
from .data import Refusal, Release

KIND = "stats_desk/kind"
READ_ONLY = types.ToolAnnotations(read_only_hint=True, idempotent_hint=True)


def _reply(text, kind):
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)],
                                is_error=kind != "ok", meta={KIND: kind})


def description(release, analysis) -> str:
    """What the model reads to choose a tool: the docstring as one paragraph, its $names filled in
    from the dataset's description (its examples, what it calls its units) and the release."""
    annual = [t for t in release.tables.values() if not t.freq.within_year]
    return words_for(release.tables.values()).fill(" ".join(analysis.__doc__.split()),
                        annual_years=len(annual[0].periods) if annual else 0)


def _bind(release, analysis):
    """An analysis as a tool: the release filled in, its other parameters advertised."""
    signature = inspect.signature(analysis)

    def call(**kwargs) -> types.CallToolResult:
        try:
            # published only for another age group or measure: the nearest one, said so (analyses.nearest);
            # figures on both sides of a change of base, said so (analyses.across_join)
            return _reply(across_join(release, kwargs, nearest(release, analysis, kwargs)), "ok")
        except Refusal as e:
            return _reply(str(e), "refused")
        except Exception:  # noqa: BLE001 -- ours to fix: logged, not shown to the model
            traceback.print_exc(file=sys.stderr)
            return _reply(f"{analysis.__name__} could not be completed", "broken")

    call.__name__, call.__doc__ = analysis.__name__, description(release, analysis)
    call.__signature__ = signature.replace(
        parameters=[p for name, p in signature.parameters.items() if name != "release"],
        return_annotation=types.CallToolResult)
    call.__annotations__ = {**{k: v for k, v in analysis.__annotations__.items() if k != "release"},
                            "return": types.CallToolResult}
    return call


def create(path=RELEASE_PATH) -> MCPServer:
    release = Release(path)
    server = MCPServer("stats-desk-analysis", version="0.1.0", instructions=(
        f"Analyses over {DATASET.publisher}'s published figures ({release.released}). Every figure is a "
        "published row; nothing averages rates. Results start with an id to cite."))
    for analysis in ANALYSES:
        server.tool(annotations=READ_ONLY)(_bind(release, analysis))
    return server


if __name__ == "__main__":
    args = sys.argv[1:]
    port = None
    if "--http" in args:
        at = args.index("--http")
        port = int(args[at + 1])
        del args[at:at + 2]
    server = create(args[0] if args else RELEASE_PATH)
    if port is None:
        server.run()
    else:
        server.run("streamable-http", host="127.0.0.1", port=port, stateless_http=True, json_response=True)
