import asyncio
import json
from typing import List
from app.models import SummaryResponse, SolutionQuery
from app.config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TEMPERATURE, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from app.embedding import EmbeddingCreator
from app.database import VectorDatabaseHandler
from app._ollama import response_text
from app.prompts import summary_prompt, solution_prompt
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAG_Engine:
    def __init__(self, timeout: float = 60.0):
        try:
            self.timeout = timeout
            self.embedder = EmbeddingCreator()
            self.vector_db = VectorDatabaseHandler()
            self.ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=timeout)
            self.ollama_async_client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=timeout)
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize RAG engine. "
                f"Make sure required services (ChromaDB, Ollama) are running. "
                f"Error: {str(e)}"
            ) from e

    # ---- Sync entry points (CLI / experiments) -------------------------------

    def generate_summary(self, context: List[str]) -> SummaryResponse:
        """Sync entry point — used by CLI/experiment scripts only."""
        return asyncio.run(self.generate_summary_async(context))

    def generate_solution(self, context: str, root_cause: str) -> SolutionQuery:
        """Sync entry point — used by CLI/experiment scripts only."""
        return asyncio.run(self.generate_solution_async(context, root_cause))

    # ---- Async entry points (FastAPI routes) ---------------------------------

    async def generate_summary_async(self, context: List[str]) -> SummaryResponse:
        """Generate summary using LLM without blocking the event loop."""
        context_text = "\n".join(context)
        prompt = summary_prompt(context_text)

        response = await self.ollama_async_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            format="json",
            options={"temperature": OLLAMA_TEMPERATURE},
        )
        parsed_text = response_text(response)

        try:
            parsed = json.loads(parsed_text)
            return SummaryResponse(
                summary=parsed.get("summary", ["Unable to generate summary"]),
                root_cause_expln=parsed.get("root_cause", "Unable to identify root cause"),
                severity=parsed.get("severity", "Unknown"),
                parse_failed=False,
            )
        except (json.JSONDecodeError, KeyError):
            return SummaryResponse(
                summary=parsed_text.split("\n"),
                root_cause_expln="Unable to parse root cause from LLM response",
                severity="Unknown",
                parse_failed=True,
            )

    async def generate_solution_async(self, context: str, root_cause: str) -> SolutionQuery:
        """Generate solution using RAG without blocking the event loop."""
        automated_query = f"Provide resolution steps for: {root_cause}"
        # vector_db.search hits a sync chromadb client which calls a sync
        # `requests.post` inside the embedding function.  Run it in a thread
        # so the event loop stays responsive while embeddings + retrieval run.
        results = await asyncio.to_thread(
            self.vector_db.search,
            automated_query,
            context,
            5,
        )
        doc_context = "\n".join([doc.text for doc in results])
        prompt = solution_prompt(root_cause=root_cause, context=context, doc_context=doc_context)
        llm_response = await self.ollama_async_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.1},
        )
        parsed_text = response_text(llm_response)
        if not parsed_text:
            raise RuntimeError("No response received from LLM")
        return SolutionQuery(
            context=context,
            query=automated_query,
            response=parsed_text,
            sources=[doc.metadata.get("source", "unknown") for doc in results],
        )

    # ---- Indexing ------------------------------------------------------------

    def store_documentation(self, documents: List[str]) -> None:
        """Store documentation in ChromaDB.

        Sync wrapper kept for CLI callers; FastAPI routes wrap this in
        `asyncio.to_thread` so the event loop is never blocked on embedding.
        """
        if not documents:
            raise ValueError("Received empty documents list")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        chunks: list[str] = []
        metadatas: list[dict] = []
        for doc_idx, doc in enumerate(documents):
            doc_chunks = text_splitter.split_text(doc)
            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                metadatas.append({"source": f"upload_{doc_idx}", "chunk_idx": chunk_idx})

        if not chunks:
            raise ValueError("No text chunks created after splitting")

        embeddings = self.embedder.create_batch_embeddings(chunks)

        self.vector_db.add_documents(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def store_documentation_async(self, documents: List[str]) -> None:
        """Async wrapper around the sync indexing path."""
        await asyncio.to_thread(self.store_documentation, documents)
