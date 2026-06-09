import asyncio
import hashlib
import logging
import re
from uuid import uuid4
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from pydantic import BaseModel

from app.log_parser import LogParser
from app.graph_generator import GraphGenerator
from app.context_builder import ContextBuilder
from app.healthcheck import ServerHealthCheck
from app.database import MongoDBHandler
from app.rag import RAG_Engine
from app.models import Context

logger = logging.getLogger(__name__)


class ResolveRequest(BaseModel):
    context: Context
    root_cause_expln: str

router = APIRouter(prefix="/api")

# Upload limits
MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_LOG_LINES    = 500
MAX_DOC_FILES    = 50
MAX_TOTAL_DOC_BYTES = 25 * 1024 * 1024   # 25 MB total per upload (across all files)

# UUID v4 regex — matches the canonical 8-4-4-4-12 hex form with the
# version/variant nibble constraints.  Client-supplied IDs that don't
# match this are rejected; we generate a fresh UUID instead.  Looser
# patterns invite ID squatting (overwriting another caller's progress
# entry by guessing or harvesting an ID).
_UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _analysis_id_from_request(request: Request) -> str:
    supplied = request.headers.get("X-Analysis-ID", "").strip().lower()
    if supplied and _UUID_V4_PATTERN.match(supplied):
        return supplied
    return str(uuid4())


def get_mongo(request: Request) -> MongoDBHandler:
    mongo = getattr(request.app.state, "mongo_db", None)
    if mongo is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")
    return mongo


def get_rag(request: Request) -> RAG_Engine:
    rag = getattr(request.app.state, "rag_engine", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine unavailable")
    return rag


def get_parser(request: Request) -> LogParser:
    parser = getattr(request.app.state, "log_parser", None)
    if parser is None:
        raise HTTPException(status_code=503, detail="Log parser unavailable")
    return parser


def _safe_upsert_progress(mongo: MongoDBHandler, analysis_id: str, payload: dict) -> None:
    """Upsert progress, swallowing failures with a warning.

    Used in the failure path of analyse_log so we don't mask the original
    user-visible error with a secondary mongo exception, but we DO record
    the secondary failure for operators.
    """
    try:
        mongo.upsert_progress(analysis_id, payload)
    except Exception:
        logger.warning("Failed to upsert progress for %s", analysis_id, exc_info=True)


@router.post("/log/analyse")
async def analyse_log(
    request: Request,
    file: UploadFile = File(...),
    mongo: MongoDBHandler = Depends(get_mongo),
    rag: RAG_Engine = Depends(get_rag),
    parser: LogParser = Depends(get_parser),
):
    """
    Analyse a log file and return severity, root cause, summary, and the
    serialised graph/context so that the client can supply them back on
    the /api/incident/resolve call (keeping the API stateless).
    """
    if not file.filename.endswith((".log", ".txt")):
        raise HTTPException(
            status_code=400, detail="Only .log and .txt files are supported"
        )

    # analysis_id stays None when _analysis_id_from_request itself raises
    # (defensive — it always returns a string).  The None sentinel keeps
    # the except-block's `if analysis_id` guard correct.
    analysis_id: str | None = None
    try:
        analysis_id = _analysis_id_from_request(request)
        mongo.upsert_progress(
            analysis_id,
            {"status": "processing", "processed_batches": 0, "total_batches": 0, "progress": 0},
        )
        content = await file.read()

        # --- size guard ---
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

        logger.info("[▶] Starting log analysis for file: %s", file.filename)
        logger.info("[►] File size: %d bytes", len(content))

        # --- line count guard ---
        text = content.decode("utf-8", errors="ignore")
        raw_line_count = len(text.splitlines())
        lines = [l for l in text.splitlines() if l.strip()]
        truncated = False
        if len(lines) > MAX_LOG_LINES:
            logger.warning("[⚠] Log has %d lines; truncating to %d", len(lines), MAX_LOG_LINES)
            lines = lines[:MAX_LOG_LINES]
            truncated = True
        parse_input = "\n".join(lines)

        # Parse log
        logger.info("[◆] Parsing log entries...")

        async def _on_progress(done: int, total: int):
            await asyncio.to_thread(
                mongo.upsert_progress,
                analysis_id,
                {
                    "status": "processing",
                    "processed_batches": done,
                    "total_batches": total,
                    "progress": round((done / total) * 100) if total else 0,
                },
            )

        log_chain = await parser.parse_log_async(
            parse_input,
            progress_callback=_on_progress,
            source_total_lines=raw_line_count,
        )
        logger.info("[✓] Parsed %d log entries", len(log_chain.log_chain))

        # Generate DAG
        logger.info("[◆] Generating correlation graph...")
        graph_gen = GraphGenerator(log_chain)
        dag = graph_gen.generate_dag()
        logger.info("[✓] Generated DAG with %d nodes", len(dag.nodes))
        logger.info("[●] Root cause identified: %s", dag.root_cause)

        # Build context
        logger.info("[◆] Building analysis context...")
        context_builder = ContextBuilder(dag)
        context = context_builder.build_context()
        logger.info("[✓] Context built with %d causal events", len(context.causal_chain))

        # Generate summary (async to avoid blocking the event loop)
        logger.info("[◆] Generating AI summary and severity assessment...")
        summary = await rag.generate_summary_async(context.causal_chain)
        logger.info("[✓] Analysis complete - Severity: %s", summary.severity)

        # Persist to MongoDB
        dag_payload = dag.model_dump(mode="json")
        dag_payload["_id"] = dag_payload.pop("id")
        mongo.save_dag(dag_payload)
        context_payload = context.model_dump(mode="json")
        context_payload["analysis_id"] = analysis_id
        mongo.save_context(context_payload)
        mongo.upsert_progress(
            analysis_id,
            {
                "status": "completed",
                "progress": 100,
                "parse_errors": log_chain.parse_errors,
                "parsed_lines": log_chain.parsed_lines,
                "total_lines": log_chain.total_lines,
                "truncated": truncated,
            },
        )

        # Return graph data so the client can supply it back on /resolve
        return {
            "analysis_id": analysis_id,
            "severity": summary.severity,
            "root_cause": context.root_cause,
            "summary": summary.summary,
            "summary_parse_failed": summary.parse_failed,
            "context": context.model_dump(mode="json"),
            "root_cause_expln": summary.root_cause_expln,
            "parse_errors": log_chain.parse_errors,
            "parsed_lines": log_chain.parsed_lines,
            "total_lines": log_chain.total_lines,
            "truncated": truncated,
            "max_log_lines": MAX_LOG_LINES,
        }

    except HTTPException:
        raise
    except ConnectionError:
        logger.exception("Service unavailable during log analysis")
        if analysis_id:
            _safe_upsert_progress(mongo, analysis_id, {"status": "failed", "progress": 100})
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception:
        logger.exception("Unexpected error in analyse_log")
        if analysis_id:
            _safe_upsert_progress(mongo, analysis_id, {"status": "failed", "progress": 100})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/docs/upload")
async def upload_docs(
    files: List[UploadFile] = File(...),
    mongo: MongoDBHandler = Depends(get_mongo),
    rag: RAG_Engine = Depends(get_rag),
):
    """
    Upload documentation files and store them in the vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_DOC_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files (max {MAX_DOC_FILES} per request)",
        )

    try:
        docs: list[str] = []
        total_bytes = 0
        for f in files:
            if not f.filename.endswith((".txt", ".md")):
                continue
            content = await f.read()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{f.filename}' too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
                )
            total_bytes += len(content)
            if total_bytes > MAX_TOTAL_DOC_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Total upload exceeds {MAX_TOTAL_DOC_BYTES // (1024 * 1024)} MB; "
                        f"split the request into smaller batches."
                    ),
                )
            docs.append(content.decode("utf-8", errors="ignore"))

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No valid documentation found. Supported formats: .txt, .md",
            )

        docs_payload = "\n\n".join(docs)
        current_hash = hashlib.sha256(docs_payload.encode("utf-8")).hexdigest()
        if mongo.db["doc_hashes"].find_one({"hash": current_hash}):
            return {"count": len(docs), "message": "Using cached documentation"}

        logger.info("[◆] Storing %d documentation files in vector database...", len(docs))
        await rag.store_documentation_async(docs)
        logger.info("[✓] Documentation indexed successfully")

        mongo.db["doc_hashes"].insert_one({"hash": current_hash})
        return {"count": len(docs), "message": f"Stored {len(docs)} documentation chunks"}

    except HTTPException:
        raise
    except ConnectionError:
        logger.exception("Service unavailable during docs upload")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception:
        logger.exception("Unexpected error in upload_docs")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/incident/resolve")
async def resolve_incident(
    request: Request,
    body: ResolveRequest,
    rag: RAG_Engine = Depends(get_rag),
):
    """
    Generate incident resolution.  The client must supply the context and
    root_cause returned by /log/analyse — the API is stateless.

    Cooperative cancellation: if the client disconnects while the LLM is
    running we don't bother returning a payload, but we cannot interrupt
    the in-flight Ollama call.  See README "Cancellation semantics".
    """
    context_data = body.context
    root_cause = body.root_cause_expln

    try:
        logger.info("[▶] Starting incident resolution generation...")
        context_str = "\n".join(context_data.causal_chain)

        logger.info("[◆] Searching documentation for relevant solutions...")
        solution_task = asyncio.create_task(
            rag.generate_solution_async(context=context_str, root_cause=root_cause)
        )
        # Race the LLM call against client disconnects to avoid wasted work
        # when the user clicks [CANCEL] in the UI.
        while not solution_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(solution_task), timeout=1.0)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    solution_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
        solution = solution_task.result()
        logger.info("[✓] Solution generated with %d references", len(solution.sources))

        sources: list[str] = []
        seen_sources: set[str] = set()
        for source in solution.sources:
            if source == "unknown" or source in seen_sources:
                continue
            seen_sources.add(source)
            sources.append(source)

        return {
            "root_cause": root_cause,
            "solution": solution.response,
            "sources": sources,
        }

    except HTTPException:
        raise
    except ConnectionError:
        logger.exception("Service unavailable during incident resolution")
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception:
        logger.exception("Unexpected error in resolve_incident")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint — verifies actual database connectivity."""
    try:
        checker = ServerHealthCheck(
            vector_db=getattr(request.app.state, "vector_db", None),
            mongo_db=getattr(request.app.state, "mongo_db", None),
        )
        status = checker.get_status()
        all_healthy = all(status.values())
        return {
            "status": "ok" if all_healthy else "degraded",
            "services": status,
            "startup_degraded": bool(getattr(request.app.state, "startup_degraded", False)),
        }
    except Exception:
        logger.exception("Health check failed")
        return {"status": "degraded", "services": {"chromadb": False, "mongodb": False}}


@router.get("/analysis/{analysis_id}/progress")
async def analysis_progress(analysis_id: str, mongo: MongoDBHandler = Depends(get_mongo)):
    progress = mongo.get_progress(analysis_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Unknown analysis ID")
    return progress


@router.get("/analysis/{analysis_id}/context")
async def analysis_context(analysis_id: str, mongo: MongoDBHandler = Depends(get_mongo)):
    context = mongo.get_context_by_analysis_id(analysis_id)
    if not context:
        raise HTTPException(status_code=404, detail="Analysis context not found")
    context.pop("_id", None)
    context.pop("analysis_id", None)
    return context
