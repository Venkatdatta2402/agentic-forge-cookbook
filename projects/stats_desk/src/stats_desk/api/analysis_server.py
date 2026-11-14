"""The machine's one analysis server: analysis/server.py as a child process serving MCP over HTTP
on the loopback, started with the app and restarted if it dies. Each question opens its own
connection to it and keeps its own results (decided 2026-11-13: the usual shape for a server
many requests share; one stdio child per question, the desktop-app pattern, cost a Python and
DuckDB process each)."""

import asyncio
import logging
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager

from ..analysis.client import connect as connect_analyses

log = logging.getLogger("stats_desk.api")

ANALYSIS_PORT = int(os.environ.get("STATS_DESK_ANALYSIS_PORT", 8765))


class SharedAnalysisServer:
    def __init__(self, release_path, port=ANALYSIS_PORT):
        self.release_path, self.port = release_path, port
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.process: subprocess.Popen | None = None
        self._lock = asyncio.Lock()

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def ensure(self, wait=30.0):
        """Start it if it isn't running (first use, or it died), and wait until it answers."""
        async with self._lock:
            if self.alive():
                return
            if self.process is not None:
                log.error("analysis server exited (%s); restarting", self.process.returncode)
            self.process = subprocess.Popen([sys.executable, "-m", "stats_desk.analysis.server",
                                             str(self.release_path), "--http", str(self.port)])
            for _ in range(int(wait / 0.2)):
                if self.process.poll() is not None:
                    raise RuntimeError(f"analysis server exited ({self.process.returncode}) while starting")
                try:
                    socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                    return
                except OSError:
                    await asyncio.sleep(0.2)
            raise RuntimeError("analysis server didn't start")

    def stop(self):
        if self.alive():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    @asynccontextmanager
    async def connect(self):
        await self.ensure()
        async with connect_analyses(self.url) as analyses:
            yield analyses
