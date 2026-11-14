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


def emit(kind: str, **data):
    event = {"kind": kind, "agent": meter._agent.get(), **data}
    for callback in _listeners.get():
        try:
            callback(event)
        except Exception:  # noqa: BLE001 -- a broken listener must not break the run
            pass
