from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import tempfile
import os
import logging

from app.utils.log_parser import LogParser
from app.utils.graph_generator import GraphGenerator
from app.utils.context_builder import ContextBuilder
from app.core.database_handlers import MongoDBHandler
from app.core.rag import RAG_Engine

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Upload limits
MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_LOG_LINES    = 500

# Initialize shared components (stateless — no session_data dict)
mongo = MongoDBHandler()
rag = RAG_Engine()

# ---------------------------------------------------------------------------
# Shared doc store — this is safe because it is only written by /docs/upload
# and is read-only during /incident/resolve.  It does not carry per-request
# state across the /analyse → /resolve boundary.
# ---------------------------------------------------------------------------
_docs_hash: int | None = None


@router.post("/log/analyse")
async def analyse_log(file: UploadFile = File(...)):
    """
    Analyse a log file and return severity, root cause, summary, and the
    serialised graph/context so that the client can supply them back on
    the /incident/resolve call (keeping the API stateless).
    """
    if not file.filename.endswith((".log", ".txt")):
        raise HTTPException(
            status_code=400, detail="Only .log and .txt files are supported"
        )

    try:
        content = await file.read()

        # --- size guard ---
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
            )

        file_extension = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            logger.info(f"[▶] Starting log analysis for file: {file.filename}")
            logger.info(f"[►] File size: {len(content)} bytes")

            # --- line count guard ---
            text = content.decode("utf-8", errors="ignore")
            lines = [l for l in text.splitlines() if l.strip()]
            if len(lines) > MAX_LOG_LINES:
                logger.warning(
                    f"[⚠] Log has {len(lines)} lines; truncating to {MAX_LOG_LINES}"
                )
                lines = lines[:MAX_LOG_LINES]
                # write truncated version back to tmp file so LogParser reads it
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

            # Parse log
            logger.info("[◆] Parsing log entries...")
            parser = LogParser()
            log_chain = parser.parse_log_from_file(tmp_path)
            logger.info(f"[✓] Parsed {len(log_chain.log_chain)} log entries")

            # Generate DAG
            logger.info("[◆] Generating causal graph (DAG)...")
            graph_gen = GraphGenerator(log_chain)
            dag = graph_gen.generate_dag()
            logger.info(f"[✓] Generated DAG with {len(dag.nodes)} nodes")
            logger.info(f"[●] Root cause identified: {dag.root_cause}")

            # Build context
            logger.info("[◆] Building analysis context...")
            context_builder = ContextBuilder(dag)
            context = context_builder.build_context()
            logger.info(f"[✓] Context built with {len(context.causal_chain)} causal events")

            # Generate summary
            logger.info("[◆] Generating AI summary and severity assessment...")
            summary = rag.generate_summary(context.causal_chain)
            logger.info(f"[✓] Analysis complete - Severity: {summary.severity}")

            # Persist to MongoDB
            mongo.save_dag(dag.model_dump())
            mongo.save_context(context.model_dump())

            # Return graph data so the client can supply it back on /resolve
            # (API is stateless — no server-side session)
            return {
                "severity": summary.severity,
                "root_cause": summary.root_cause_expln,
                "summary": summary.summary,
                # Serialised for the client to echo back:
                "_context": context.model_dump(),
                "_root_cause_expln": summary.root_cause_expln,
            }

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysing log: {str(e)}")


@router.post("/docs/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload documentation files and store them in the vector database.
    """
    global _docs_hash

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        docs = []
        for f in files:
            if not f.filename.endswith((".txt", ".md")):
                continue
            content = await f.read()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{f.filename}' too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)."
                )
            docs.append(content.decode("utf-8", errors="ignore"))

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No valid documentation found. Supported formats: .txt, .md",
            )

        current_hash = hash(tuple(docs))
        if _docs_hash == current_hash:
            return {"count": len(docs), "message": "Using cached documentation"}

        logger.info(f"[◆] Storing {len(docs)} documentation files in vector database...")
        rag.store_documentation(docs)
        logger.info("[✓] Documentation indexed successfully")

        _docs_hash = current_hash
        return {"count": len(docs), "message": f"Stored {len(docs)} documentation chunks"}

    except HTTPException:
        raise
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documentation: {str(e)}")


@router.post("/incident/resolve")
async def resolve_incident(body: dict):
    """
    Generate incident resolution.  The client must supply the context and
    root_cause returned by /log/analyse — the API is stateless.

    Expected body:
        { "_context": {...}, "_root_cause_expln": "..." }
    """
    context_data = body.get("_context")
    root_cause = body.get("_root_cause_expln")

    if not context_data or not root_cause:
        raise HTTPException(
            status_code=400,
            detail=(
                "Request body must include '_context' and '_root_cause_expln' "
                "returned by the /log/analyse endpoint."
            ),
        )

    try:
        logger.info("[▶] Starting incident resolution generation...")
        causal_chain = context_data.get("causal_chain", [])
        context_str = "\n".join(causal_chain) if isinstance(causal_chain, list) else str(causal_chain)

        logger.info("[◆] Searching documentation for relevant solutions...")
        solution = rag.generate_solution(context=context_str, root_cause=root_cause)
        logger.info(f"[✓] Solution generated with {len(solution.sources)} references")

        return {
            "root_cause": root_cause,
            "solution": solution.response,
            "sources": [s for s in solution.sources if s != "unknown"],
        }

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating resolution: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Backend is running"}
