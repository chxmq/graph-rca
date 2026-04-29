import chromadb
from pymongo import MongoClient, ASCENDING
from chromadb.utils.embedding_functions import EmbeddingFunction
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import requests
import logging
from app.config import OLLAMA_HOST, EMBEDDING_MODEL, MONGO_URI, MONGO_TIMEOUT_MS, CHROMADB_PATH, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

@dataclass
class Document:
    text: str
    metadata: dict

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Custom Ollama embedding function for ChromaDB."""
    def __init__(self, model_name: str, url: str):
        self.model_name = model_name
        self.url = url

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using parallel requests.

        Validates dimension as results stream in via ``as_completed`` so we
        bail on mismatch as soon as we have evidence — instead of paying
        for every embedding before raising.
        """
        def _embed_single(text: str) -> list[float]:
            response = requests.post(
                self.url,
                json={"model": self.model_name, "prompt": text},
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            body = response.json()
            if "embedding" not in body:
                raise ValueError("Ollama response did not include 'embedding'")
            embedding = body["embedding"]
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Invalid embedding payload from Ollama")
            return embedding

        embeddings: list[list[float] | None] = [None] * len(input)
        expected_dim: int | None = None
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_embed_single, text): i for i, text in enumerate(input)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    emb = future.result()
                except Exception as e:
                    raise RuntimeError(f"Embedding generation failed for input index {idx}: {e}") from e
                if expected_dim is None:
                    expected_dim = len(emb)
                elif len(emb) != expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch at index {idx}: {len(emb)} != {expected_dim}"
                    )
                embeddings[idx] = emb
        # mypy: embeddings is fully populated here because every index was
        # written before the loop terminated.
        return [e for e in embeddings if e is not None]

class VectorDatabaseHandler:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(
                path=CHROMADB_PATH,
                settings=chromadb.Settings(
                    anonymized_telemetry=False
                )
            )
                
            self.ef = OllamaEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                url=f"{OLLAMA_HOST}/api/embeddings",
            )
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize ChromaDB at persist directory. "
                f"Error: {str(e)}"
            ) from e
    
    def _get_query_collection(self, name: str = "docs"):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.ef,
        )

    def get_collection(self, name: str = "docs"):
        """Public query collection accessor used by health checks and callers."""
        return self._get_query_collection(name=name)

    def _get_write_collection(self, name: str = "docs"):
        return self.client.get_or_create_collection(name=name)

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ):
        collection = self._get_write_collection()
        if ids is None:
            ids = [hashlib.sha256(doc.encode()).hexdigest()[:32] for doc in documents]
        collection.upsert(
            documents=documents,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    
    def query_collection(self, query_texts: list[str], n_results: int = 3):
        collection = self._get_query_collection()
        return collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

    def search(self, query: str, context: str, top_k: int = 5) -> list[Document]:
        """Search implementation using text query"""
        collection = self._get_query_collection()
        query_text = f"{query}\nContext: {context}"
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        if not results or not isinstance(results, dict):
            return []
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        if not documents or not documents[0]:
            return []
        # ChromaDB returns parallel lists with the same length: documents[0]
        # is the first query's matches, metadatas[0] is the corresponding
        # metadata.  ``strict=True`` would catch any future shape regression.
        first_metas = metadatas[0] if metadatas and metadatas[0] else [None] * len(documents[0])
        return [
            Document(text=doc, metadata=meta if meta else {"source": "unknown"})
            for doc, meta in zip(documents[0], first_metas)
        ]

class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIMEOUT_MS)
        # Eagerly validate the connection (MongoClient is lazy by default)
        try:
            self.client.server_info()
        except Exception as e:
            logger.error("Failed to connect to MongoDB at %s: %s", MONGO_URI, e)
            raise ConnectionError(f"MongoDB is unreachable at {MONGO_URI}") from e
        self.db = self.client["log_analysis"]
        self.db["contexts"].create_index("dag_id")
        self.db["contexts"].create_index("analysis_id")
        self.db["contexts"].create_index("created_at")
        # TTL index: progress entries expire automatically after 2 hours
        self.db["analysis_progress"].create_index(
            [("updated_at", ASCENDING)],
            expireAfterSeconds=7200,
            name="progress_ttl",
        )
        self.db["analysis_progress"].create_index("analysis_id", unique=True)
    
    def save_dag(self, dag_data: dict):
        return self.db["dags"].insert_one(dag_data)
    
    def get_context(self, dag_id: str = None):
        """Retrieve context by DAG ID or latest if not specified"""
        if dag_id:
            return self.db["contexts"].find_one({"dag_id": dag_id})
        return self.db["contexts"].find_one(sort=[("created_at", -1)])

    def get_context_by_analysis_id(self, analysis_id: str):
        """Retrieve context by frontend/backend analysis ID."""
        return self.db["contexts"].find_one({"analysis_id": analysis_id})

    def save_context(self, context_data: dict):
        return self.db["contexts"].insert_one(context_data)

    def upsert_progress(self, analysis_id: str, payload: dict) -> None:
        """Upsert analysis progress so any worker can read it."""
        self.db["analysis_progress"].update_one(
            {"analysis_id": analysis_id},
            {"$set": {**payload, "analysis_id": analysis_id,
                      "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )

    def get_progress(self, analysis_id: str) -> dict | None:
        """Retrieve analysis progress, stripping internal fields."""
        doc = self.db["analysis_progress"].find_one({"analysis_id": analysis_id})
        if doc is None:
            return None
        doc.pop("_id", None)
        doc.pop("analysis_id", None)
        doc.pop("updated_at", None)
        return doc