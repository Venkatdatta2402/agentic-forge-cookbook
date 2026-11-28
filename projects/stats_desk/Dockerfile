# Stats Desk: the API, the analysis server it starts, and the release file, in one image.
#
#   docker build -t stats-desk .
#   docker run --rm -p 8080:8080 stats-desk            # then http://localhost:8080/health
#
# Two stages: dependencies installed with uv from the lock file, then a slim runtime holding only
# the virtual environment and the release file. Runs as a non-root user; the release file is
# read-only. The release job rebuilds this image when MoSPI publishes new figures.

FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependencies first, in their own layer: rebuilt only when the lock file changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.13-slim
# links the image on GitHub Container Registry to its repository
LABEL org.opencontainers.image.source=https://github.com/Venkatdatta2402/agentic-forge-cookbook
LABEL org.opencontainers.image.description="Stats Desk: agents answering questions on India's labour statistics (MoSPI PLFS)"
# the folders first, readable and enterable (755): a COPY --chmod that creates them gives them the
# file's 444 too, and the app then can't reach the release file (found on the first build)
RUN useradd --create-home --uid 10001 desk && mkdir -p /app/data/release
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
# one release file per dataset; the server reads the folder
COPY --chmod=444 data/release/plfs.duckdb data/release/cpi.duckdb /app/data/release/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STATS_DESK_RELEASE=/app/data/release \
    PORT=8080
USER desk
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
# Cloud Run sets PORT; one worker, because running questions and the governor live in the process
CMD ["sh", "-c", "exec uvicorn stats_desk.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
