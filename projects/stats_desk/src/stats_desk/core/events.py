"""Progress events ("analyst: change on ur_annual…"), for whoever is listening in the current task.

Same mechanism as the meter: a ContextVar of listeners, copied into tasks started inside it, so
one question's stream hears its own agents and nobody else's. The API's SSE stream will be the
listener (no caller yet: the API step adds it).
"""

from contextlib import contextmanager
from contextvars import ContextVar

from . import meter

_listeners: ContextVar[tuple] = ContextVar("listeners", default=())


@contextmanager
def listening(callback):
    token = _listeners.set(_listeners.get() + (callback,))
    try:
        yield
    finally:
        _listeners.reset(token)


def label(analysis: str, args) -> str:
    """An analysis call as a reader knows it: "change on ur_monthly by gender (2026-09)"."""
    args = args if isinstance(args, dict) else {}
    table = args.get("table") or args.get("x_table")
    out = analysis + (f" on {table}" if isinstance(table, str) and table else "")
    if isinstance(args.get("y_table"), str) and args["y_table"] != table:
        out += f" against {args['y_table']}"
    if args.get("by"):
        out += f" by {args['by']}"
    if args.get("field") and args.get("a") and args.get("b"):
        out += f", {args['a']} against {args['b']}"
    where = args.get("where") or args.get("x_where")
    if isinstance(where, dict) and where:
        out += f" ({', '.join(' / '.join(map(str, v)) if isinstance(v, list) else str(v) for v in where.values())})"
    elif isinstance(where, list) and all(isinstance(w, dict) for w in where):
        out += f" ({' / '.join(', '.join(map(str, w.values())) or 'everyone' for w in where)})"
    return out


def emit(kind: str, **data):
    event = {"kind": kind, "agent": meter._agent.get(), **data}
    for callback in _listeners.get():
        try:
            callback(event)
        except Exception:  # noqa: BLE001 -- a broken listener must not break the run
            pass
