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

# Initialize components (shared across requests)
mongo = MongoDBHandler()
rag = RAG_Engine()

# Session storage (in production, use Redis or similar)
session_data = {
    "processed_log": {
        "file_hash": None,
        "dag": None,
        "context": None,
        "summary": None,
        "severity": None,
        "root_cause": None,
    },
    "stored_docs": {
        "file_hash": None,
        "docs": None,
    },
}


@router.post("/log/analyse")
async def analyse_log(file: UploadFile = File(...)):
    """
    Analyse a log file and return severity, root cause, and summary.
    """
    if not file.filename.endswith((".log", ".txt")):
        raise HTTPException(
            status_code=400, detail="Only .log and .txt files are supported"
        )

    try:
        # Save uploaded file temporarily
        content = await file.read()
        current_file_hash = hash(content)

        # Check cache
        if session_data["processed_log"]["file_hash"] == current_file_hash:
            return {
                "severity": session_data["processed_log"]["severity"],
                "root_cause": session_data["processed_log"]["root_cause"],
                "summary": session_data["processed_log"]["summary"].summary,
            }

        file_extension = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(
            suffix=file_extension, delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Parse log
            logger.info(f"[▶] Starting log analysis for file: {file.filename}")
            logger.info(f"[►] File size: {len(content)} bytes")
            
            parser = LogParser()
            logger.info("[◆] Parsing log entries...")
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
            context_builder = ContextBuilder()
            context = context_builder.build_context(dag)
            logger.info(f"[✓] Context built with {len(context.causal_chain)} causal events")

            # Generate summary
            logger.info("[◆] Generating AI summary and severity assessment...")
            summary = rag.generate_summary(context.causal_chain)
            logger.info(f"[✓] Analysis complete - Severity: {summary.severity}")

            # Store in session
            session_data["processed_log"] = {
                "file_hash": current_file_hash,
                "dag": dag,
                "context": context,
                "summary": summary,
                "severity": summary.severity,
                "root_cause": summary.root_cause_expln,
            }

            # Store in MongoDB
            mongo.save_dag(dag.model_dump())
            mongo.save_context(context.model_dump())

            return {
                "severity": summary.severity,
                "root_cause": summary.root_cause_expln,
                "summary": summary.summary,
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analysing log: {str(e)}")


@router.post("/docs/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload documentation files and store them in the vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        # Read all files
        docs = []
        for f in files:
            if not f.filename.endswith((".txt", ".md")):
                continue
            content = await f.read()
            docs.append(content.decode("utf-8", errors="ignore"))

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No valid documentation found. Supported formats: .txt, .md",
            )

        current_docs_hash = hash(tuple(docs))

        # Check cache
        if session_data["stored_docs"]["file_hash"] == current_docs_hash:
            return {"count": len(docs), "message": "Using cached documentation"}

        # Store in vector DB
        logger.info(f"[◆] Storing {len(docs)} documentation files in vector database...")
        rag.store_documentation(docs)
        logger.info("[✓] Documentation indexed successfully")

        # Update session
        session_data["stored_docs"] = {
            "file_hash": current_docs_hash,
            "docs": docs,
        }

        return {"count": len(docs), "message": f"Stored {len(docs)} documentation chunks"}

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading documentation: {str(e)}"
        )


@router.post("/incident/resolve")
async def resolve_incident():
    """
    Generate incident resolution based on analysed log and documentation.
    """
    if not session_data["processed_log"].get("summary") or not session_data[
        "processed_log"
    ].get("context"):
        raise HTTPException(
            status_code=400,
            detail="Please analyse a log file first before running incident resolution",
        )

    try:
        logger.info("[▶] Starting incident resolution generation...")
        context_data = session_data["processed_log"]["context"].causal_chain
        if isinstance(context_data, (list, tuple)):
            context_str = "\n".join(context_data)
        else:
            context_str = str(context_data)

        logger.info("[◆] Searching documentation for relevant solutions...")
        solution = rag.generate_solution(
            context=context_str,
            root_cause=session_data["processed_log"]["summary"].root_cause_expln,
        )
        logger.info(f"[✓] Solution generated with {len(solution.sources)} references")

        return {
            "root_cause": session_data["processed_log"]["summary"].root_cause_expln,
            "solution": solution.response,
            "sources": [s for s in solution.sources if s != "unknown"],
        }

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating resolution: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Backend is running"}
