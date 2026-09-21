"""The tools almost every agent ends up with, each written around its own hazard.

Not called `builtins.py`. A module by that name sitting on `sys.path` shadows Python's own
`builtins`, and the failure that follows looks like the interpreter itself has broken.
"""

import ast
import operator
from pathlib import Path
from typing import Annotated

import httpx

from invoke import ToolError
from tools import tool


# ---------------------------------------------------------------- the network

@tool
def http_get(url: Annotated[str, "Full URL, including https://"]) -> str:
    """Fetch a web page or API response and return its body as text."""
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True,
                             headers={"User-Agent": "agentic-forge-cookbook"})
    except httpx.HTTPError as e:
        # A grey area, and worth being honest about rather than pretending the rule is clean.
        # "Who can fix it" says a dead host is ours; in practice the commonest cause by far is a
        # URL the model invented, which is the model's to fix. So it goes back as a ToolError
        # with the reason attached. A genuinely offline machine will produce a run where every
        # fetch refuses, which reads clearly enough to whoever is watching.
        raise ToolError(f"Could not reach {url}: {type(e).__name__}. Check the address is right.")

    if response.status_code >= 400:
        # The status code is the part worth passing on. 404 means try a different address, 429
        # means stop for a while, 500 means the other end is broken and waiting might help --
        # three different next moves, and only the number distinguishes them.
        raise ToolError(f"{url} returned HTTP {response.status_code} "
                        f"({response.reason_phrase or 'no reason given'}).")
    return response.text


# ------------------------------------------------------------------- the disk

def within(root, path):
    """Resolve `path` under `root`, refusing anything that ends up outside it.

    `resolve()` is what does the work: it flattens `..` and follows symlinks first, so the check
    is against where the path really lands rather than how it was spelled. Checking the spelling
    instead -- rejecting strings containing ".." -- is the version that gets bypassed, because
    there are always more spellings than you thought.
    """
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"'{path}' is outside the folder this agent may use.")
    return target


def file_tools(root):
    """Reading and writing, confined to one folder.

    A factory rather than two module-level tools, because the folder is not a property of the
    tool -- it is a decision about a particular agent. Two agents in one process can each get
    their own, and neither can reach the other's.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    @tool
    def read_file(path: Annotated[str, "Path to the file, relative to the working folder"]) -> str:
        """Read a text file and return its contents."""
        target = within(root, path)
        if not target.is_file():
            raise ToolError(f"There is no file at '{path}'.")
        return target.read_text(encoding="utf-8")

    # repeatable=False: writing twice is not the same as writing once. The second write is
    # harmless when the text is identical and destroys work when it isn't, and after a timeout
    # there is no way to tell which case you are in -- see invoke.py.
    @tool(repeatable=False)
    def write_file(path: Annotated[str, "Path to the file, relative to the working folder"],
                   text: Annotated[str, "What to write. Replaces the whole file."]) -> str:
        """Write text to a file, replacing anything already there."""
        target = within(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return f"Wrote {len(text)} characters to {path}."

    return read_file, write_file


def written_correctly(root):
    """A check for `write_file`, in the shape notebook 4's `verified()` wants.

    Bound to the same root the tool was, because a check that reads a different folder than the
    one written to is worse than no check: it reports success about the wrong place.
    """
    def check(args, value):
        target = within(root, args["path"])
        if not target.is_file():
            return f"Nothing was written to '{args['path']}'. There is no file there."
        on_disk = target.read_text(encoding="utf-8")
        if on_disk != args["text"]:
            return (f"'{args['path']}' holds {len(on_disk)} characters, not the "
                    f"{len(args['text'])} that were sent.")
        return None

    return check


# ------------------------------------------------------------------- the maths

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _value_of(node):
    # Walking the parsed tree and refusing to handle anything not listed. The opposite approach
    # -- eval() with a list of banned words -- is the one that fails, because it has to think of
    # every dangerous spelling in advance and only needs to miss one.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _value_of(node.left), _value_of(node.right)
        # Refusing anything that has to be stopped is not enough on its own: `9**9**9` is
        # arithmetic, every node here is allowed, and working it out will sit at 100% CPU for
        # longer than anyone will wait. A tool can be dangerous without doing anything it was
        # not asked to do.
        if type(node.op) is ast.Pow and abs(right) > 100:
            raise ToolError(f"An exponent of {right} is too large to work out. Keep it under 100.")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_value_of(node.operand))
    raise ToolError(f"'{ast.dump(node)[:40]}...' is not arithmetic. "
                    "Only numbers and + - * / // % ** are allowed.")


@tool
def calculate(expression: Annotated[str, "An arithmetic expression, e.g. '(3 + 4) * 2'"]) -> str:
    """Work out the value of an arithmetic expression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"'{expression}' is not a complete expression: {e.msg}.")
    return str(_value_of(tree.body))
