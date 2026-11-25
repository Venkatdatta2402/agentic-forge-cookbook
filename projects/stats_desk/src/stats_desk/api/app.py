"""The web API: a question in, its progress streamed, a brief out.

    uv run uvicorn stats_desk.api.app:app --port 8080

Endpoints
    GET  /                 the page (web/index.html): ask, watch the agents work, read the brief;
                           /?brief=<id> opens a saved brief
    GET  /free             the visitor's free questions left today
    POST /questions       {question, follow_up_of?}; the user's Groq key in the X-Groq-Key
                           header, or none for a free question on the desk's keys -> a stream of
                           server-sent events:
                               started   {id, link, free}  the brief's link, at once
                               step      {step}            "f1 started: ...", "The lead is writing the brief"
                               brief     {the brief}       the end, when there is one
                               paused    {id, message, carry_on}   the keys' day ran out partway:
                                                           saved, to carry on (POST carry_on)
                               failed    {status, error}   the end, when there isn't
    POST /briefs/{id}/resume   carry on a question stopped partway (paused, or its page closed),
                           from its record; key as above; the same stream
    GET  /briefs/{id}      a brief by its link (while it runs too: status "running", findings so far)
    GET  /examples         the few briefs chosen as public examples
    GET  /profiles         the profiles' names and the release;  GET /profiles/{name} one profile
    GET  /health           the analysis server, questions running, the free keys' tokens left

Only the two POSTs spend tokens. How questions run, and the rules for keys: desk.py. The
analyses: analysis_server.py. This file is the web layer: requests, streams, pages.

A comment line every 15 s keeps a stream open through proxies while the agents wait on Groq's
minute limit, and notices a closed page, which stops the question (desk.py says why).
"""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .. import config
from ..agents.record import Record
from ..analysis.data import Release
from ..core import trace
from ..datasets import DATASET
from .analysis_server import SharedAnalysisServer
from .desk import MAX_RUNNING, Ask, Desk, check_groq_key
from .store import Store, from_env

PING = 15.0          # seconds between keep-alive comments on a quiet stream
PAGE = Path(__file__).parent / "web" / "index.html"     # the one page: ask, watch, read the brief


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def visitor_of(request: Request) -> str:
    """Who is asking, for the free questions' per-visitor count: a hash of their address, never
    the address. The host's front end (Azure Container Apps' Envoy) adds the caller's address
    last in X-Forwarded-For (earlier entries are whatever the caller sent, so not trusted): to
    confirm on the first deploy."""
    forwarded = request.headers.get("x-forwarded-for", "")
    address = forwarded.split(",")[-1].strip() if forwarded else (request.client.host if request.client else "")
    return hashlib.sha256(f"visitor:{address}".encode()).hexdigest()[:16]


async def _stream(request: Request, record: Record, queue, task):
    try:
        yield sse("started", {"id": record.id, "link": f"/briefs/{record.id}", "free": record.free})
        while True:
            try:
                event, data = await asyncio.wait_for(queue.get(), PING)
            except TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": still working\n\n"
                continue
            yield sse(event, data)
            if event in ("brief", "failed", "paused"):
                return
    finally:
        if not task.done():                        # the page was closed: stop the question
            task.cancel()


def _streaming(request, record, queue, task):
    return StreamingResponse(_stream(request, record, queue, task), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _key(header) -> str | None:
    return (header or "").strip() or None


def create_app(store: Store | None = None, connect=None, check_key=check_groq_key,
               release_path=config.RELEASE_PATH, max_running=MAX_RUNNING, free_keys=config.FREE_KEYS,
               free_per_visitor=config.FREE_PER_VISITOR_PER_DAY) -> FastAPI:
    """`connect`: how a question reaches the analyses (tests pass an in-process server); by
    default the machine's shared analysis server."""
    release = Release(release_path)                 # read once: the profiles and what's released
    profiles, released = dict(release.profiles), release.released
    release.close()
    shared = None
    if connect is None:
        shared = SharedAnalysisServer(release_path)
        connect = shared.connect
    desk = Desk(store or from_env(), connect, check_key, released, max_running, free_keys, free_per_visitor)

    @asynccontextmanager
    async def lifespan(_app):
        if shared is not None:
            await shared.ensure()
        yield
        if shared is not None:
            shared.stop()
        trace.flush()

    app = FastAPI(title="Stats Desk", lifespan=lifespan,
                  description=f"Questions about {DATASET.subject} ({DATASET.publisher}'s {DATASET.name}), "
                              "answered by agents from published figures, every number checked.")
    app.state.desk = desk

    @app.get("/", include_in_schema=False)
    async def page():
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    @app.get("/free")
    async def free(request: Request):
        """The visitor's free questions left today, and the release, for the page's header."""
        return {"left": await desk.free_left(visitor_of(request)), "per_day": desk.free_per_visitor,
                "release": released}

    @app.get("/health")
    async def health():
        return {"ok": shared is None or shared.alive(), "release": released, "running": desk.active,
                "free_keys_tokens_left": [await desk.tokens_left(k) for k in desk.free_keys]}

    @app.get("/profiles")
    async def profile_names():
        return {"release": released, "profiles": list(profiles)}

    @app.get("/profiles/{name}")
    async def profile(name: str):
        if name not in profiles:
            raise HTTPException(404, f"No profile {name}. Profiles: {', '.join(profiles)}")
        return {"name": name, "release": released, "text": profiles[name]}

    @app.get("/examples")
    async def examples():
        return [{k: d.get(k) for k in ("id", "question", "answer", "created")} for d in await desk.store.public()]

    @app.get("/briefs/{brief_id}")
    async def brief(brief_id: str):
        doc = await desk.store.get(brief_id)
        if doc is None:
            raise HTTPException(404, f"No brief {brief_id}.")
        return Record.from_doc(doc).view()

    @app.post("/questions")
    async def question(ask: Ask, request: Request, x_groq_key: str | None = Header(default=None)):
        return _streaming(request, *await desk.ask(ask, _key(x_groq_key), visitor_of(request)))

    @app.post("/briefs/{brief_id}/resume")
    async def resume(brief_id: str, request: Request, x_groq_key: str | None = Header(default=None)):
        return _streaming(request, *await desk.resume(brief_id, _key(x_groq_key), visitor_of(request)))

    return app


def __getattr__(name):
    """`app` for uvicorn, built on first use (so importing this module opens no release file)."""
    if name == "app":
        globals()["app"] = built = create_app()
        return built
    raise AttributeError(name)
