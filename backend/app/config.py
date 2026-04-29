"""
Centralized configuration for the GraphRCA backend.

All environment-dependent values live here so they can be changed in one place.
Modules should import from this module instead of reading os.environ directly.
"""

import os
from pathlib import Path

def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid integer for {name}: {raw}") from e

def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid float for {name}: {raw}") from e

# --- Ollama ---
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11435")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT: float = _get_float("OLLAMA_TIMEOUT", 30.0)
OLLAMA_TEMPERATURE: float = _get_float("OLLAMA_TEMPERATURE", 0.2)

# --- MongoDB ---
MONGO_URI: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_TIMEOUT_MS: int = _get_int("MONGO_TIMEOUT_MS", 5000)

# --- ChromaDB ---
# Default to a stable project-root path, independent of the shell cwd.
# Only PersistentClient is used; no host/port — if a remote Chroma server
# is wanted later, add CHROMADB_HOST/CHROMADB_PORT here at the same time
# you wire HttpClient in database.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMADB_PATH: str = os.environ.get("CHROMADB_PATH", str((_PROJECT_ROOT / "data/chroma").resolve()))

# --- RAG ---
RAG_CHUNK_SIZE: int = _get_int("RAG_CHUNK_SIZE", 1000)
RAG_CHUNK_OVERLAP: int = _get_int("RAG_CHUNK_OVERLAP", 200)
