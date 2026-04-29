import os
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger import jsonlogger
from app.routes import router
from app.config import CHROMADB_PATH
from app.database import MongoDBHandler, VectorDatabaseHandler
from app.log_parser import LogParser
from app.rag import RAG_Engine

# ContextVar lets the per-request id flow into log records without explicit
# threading.  The middleware sets it; the filter reads it.
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


LOG_JSON = os.environ.get("LOG_JSON", "1").lower() in {"1", "true", "yes"}
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.handlers:
    root_logger.handlers.clear()
handler = logging.StreamHandler()
handler.addFilter(_RequestIdFilter())
if LOG_JSON:
    handler.setFormatter(jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
    ))
else:
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(request_id)s] - %(name)s - %(levelname)s - %(message)s"
    ))
root_logger.addHandler(handler)

logger = logging.getLogger(__name__)

RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_WINDOW_SECONDS = 60
_request_windows: dict[str, list[float]] = {}
API_KEY = os.environ.get("API_KEY", "").strip()

# Paths that bypass API_KEY auth (read-only, low-sensitivity).
AUTH_EXEMPT_PATHS = {"/", "/api/health", "/docs", "/redoc", "/openapi.json"}


async def _init_with_retries(factory, name: str, retries: int = 3, delay_seconds: float = 1.5):
    for attempt in range(1, retries + 1):
        try:
            return factory()
        except Exception:
            logger.warning("Startup init failed for %s (%d/%d)", name, attempt, retries, exc_info=True)
            if attempt < retries:
                await asyncio.sleep(delay_seconds)
    logger.error("Continuing without %s after startup retries", name)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(CHROMADB_PATH).mkdir(parents=True, exist_ok=True)

    app.state.mongo_db = await _init_with_retries(MongoDBHandler, "mongo_db")
    app.state.vector_db = await _init_with_retries(VectorDatabaseHandler, "vector_db")
    app.state.rag_engine = await _init_with_retries(RAG_Engine, "rag_engine")
    app.state.log_parser = await _init_with_retries(LogParser, "log_parser")
    app.state.startup_degraded = any(
        service is None
        for service in (
            app.state.mongo_db,
            app.state.vector_db,
            app.state.rag_engine,
            app.state.log_parser,
        )
    )
    if app.state.startup_degraded:
        logger.error(
            "Application started in degraded mode: mongo=%s vector=%s rag=%s parser=%s",
            app.state.mongo_db is not None,
            app.state.vector_db is not None,
            app.state.rag_engine is not None,
            app.state.log_parser is not None,
        )

    yield

app = FastAPI(
    title="Graph-RCA Backend API",
    description="Log analysis and incident resolution backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: read from env var so any deployment works without code changes.
# e.g. ALLOWED_ORIGINS=https://myrca.example.com,http://localhost:5173
_default_origins = "http://localhost:5173,http://localhost:3000"
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID", "X-Analysis-ID", "X-API-Key"],
)

# Include API routes
app.include_router(router)


def _requires_auth(path: str) -> bool:
    """Return True when API_KEY is set and the path is not auth-exempt.

    All authenticated paths use the same gate regardless of HTTP method —
    GET endpoints can leak progress/context (analysis IDs are upsert keys),
    so they need the same protection as POSTs.
    """
    if not API_KEY:
        return False
    if path in AUTH_EXEMPT_PATHS:
        return False
    return path.startswith("/api/")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    token = _request_id_var.set(request_id)
    try:
        if _requires_auth(request.url.path):
            supplied_key = request.headers.get("X-API-Key", "")
            if supplied_key != API_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized"},
                    headers={"X-Request-ID": request_id},
                )

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS
        # Lazy per-IP eviction: only trim/drop the entry for the IP we're actually
        # serving. A full-dict sweep on every request is O(N×M) and unnecessary.
        ip_events = [ts for ts in _request_windows.get(client_ip, []) if ts >= window_start]
        if not ip_events:
            _request_windows.pop(client_ip, None)
        if len(ip_events) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"X-Request-ID": request_id},
            )
        ip_events.append(now)
        _request_windows[client_ip] = ip_events

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        _request_id_var.reset(token)


@app.get("/")
async def root():
    return {
        "message": "Graph-RCA Backend API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
