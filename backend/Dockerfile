# syntax=docker/dockerfile:1.7

# ---------- builder ----------
# Builds wheels into a venv and discards apt/pip cache.  The runtime stage
# copies only the venv, so chromadb's build deps and pip's wheel cache
# never ship in the final image.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

# Install only what wheel-building needs at this layer; the runtime image
# below doesn't need build-essential.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Keep at 1 until _request_windows rate-limit state is moved to a shared
    # store (Redis / MongoDB). With >1 workers each process holds its own
    # dict, making the effective limit RATE_LIMIT_PER_MINUTE × workers.
    UVICORN_WORKERS=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=app:app . .
USER app

EXPOSE 8010

HEALTHCHECK --interval=20s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/api/health', timeout=3)"

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port 8010 --workers ${UVICORN_WORKERS}"]
