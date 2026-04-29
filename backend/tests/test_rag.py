import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from app.rag import RAG_Engine
from app.models import SummaryResponse, SolutionQuery
from app.database import Document


@pytest.fixture
def mock_ollama():
    with patch("ollama.Client") as mock_sync, patch("ollama.AsyncClient") as mock_async:
        yield mock_sync, mock_async


@pytest.fixture
def rag_engine(mock_ollama):
    with patch("app.rag.VectorDatabaseHandler") as mock_vector_db:
        engine = RAG_Engine()
        engine.vector_db = mock_vector_db.return_value
        yield engine


@pytest.fixture
def sample_docs():
    return ["Doc 1 content", "Doc 2 content", "Doc 3 content"]


class TestRAGEngine:
    @pytest.mark.asyncio
    async def test_generate_summary_success(self, rag_engine):
        mock_response = {
            "response": json.dumps({
                "summary": ["Summary line 1", "Summary line 2"],
                "root_cause": "Database connection pool exhausted",
                "severity": "High",
            }),
            "status_code": 200,
        }
        rag_engine.ollama_async_client.generate = AsyncMock(return_value=mock_response)

        context = ["Error log 1", "Warning log 2"]
        result = await rag_engine.generate_summary_async(context)

        assert isinstance(result, SummaryResponse)
        assert len(result.summary) == 2
        assert result.root_cause_expln == "Database connection pool exhausted"
        assert result.severity == "High"
        assert result.parse_failed is False
        called_kwargs = rag_engine.ollama_async_client.generate.call_args.kwargs
        assert called_kwargs["model"] == "llama3.2:3b"
        assert called_kwargs["format"] == "json"
        assert called_kwargs["options"] == {"temperature": 0.2}
        assert "<log_context>" in called_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_generate_summary_fallback_on_malformed_json(self, rag_engine):
        rag_engine.ollama_async_client.generate = AsyncMock(return_value={"response": "line1\nline2"})
        result = await rag_engine.generate_summary_async(["ctx"])
        assert isinstance(result, SummaryResponse)
        assert result.summary == ["line1", "line2"]
        assert result.severity == "Unknown"
        assert result.parse_failed is True

    @pytest.mark.asyncio
    async def test_generate_solution_success(self, rag_engine):
        mock_doc = Document(text="Sample doc text", metadata={"source": "kb123"})
        rag_engine.vector_db.search = Mock(return_value=[mock_doc])
        rag_engine.ollama_async_client.generate = AsyncMock(return_value={
            "response": "Detailed solution steps...",
            "status_code": 200,
        })

        result = await rag_engine.generate_solution_async("error context", "root cause")

        assert isinstance(result, SolutionQuery)
        assert "Detailed solution steps" in result.response
        assert "kb123" in result.sources
        assert result.query.startswith("Provide resolution steps for:")

    @pytest.mark.asyncio
    async def test_generate_solution_vector_search_error(self, rag_engine):
        rag_engine.vector_db.search = Mock(side_effect=Exception("DB error"))
        with pytest.raises(Exception, match="DB error"):
            await rag_engine.generate_solution_async("context", "root cause")

    @pytest.mark.asyncio
    async def test_generate_solution_llm_failure(self, rag_engine):
        valid_doc = Document(text="valid doc", metadata={"source": "unknown"})
        rag_engine.vector_db.search = Mock(return_value=[valid_doc])
        rag_engine.ollama_async_client.generate = AsyncMock(side_effect=Exception("LLM down"))
        with pytest.raises(Exception, match="LLM down"):
            await rag_engine.generate_solution_async("context", "root cause")

    @pytest.mark.asyncio
    async def test_generate_solution_malformed_llm_response(self, rag_engine):
        valid_doc = Document(text="valid doc", metadata={"source": "unknown"})
        rag_engine.vector_db.search = Mock(return_value=[valid_doc])
        rag_engine.ollama_async_client.generate = AsyncMock(return_value={})
        with pytest.raises(RuntimeError, match="No response received from LLM"):
            await rag_engine.generate_solution_async("context", "root cause")

    def test_store_documentation_success(self, rag_engine, sample_docs):
        rag_engine.embedder.create_batch_embeddings = Mock(return_value=[[0.1] * 768])
        rag_engine.vector_db.add_documents = Mock()

        rag_engine.store_documentation(sample_docs)

        rag_engine.vector_db.add_documents.assert_called_once()
        assert rag_engine.embedder.create_batch_embeddings.call_count == 1

    def test_store_documentation_empty_input(self, rag_engine):
        with pytest.raises(ValueError):
            rag_engine.store_documentation([])

    def test_store_documentation_chunk_validation(self, rag_engine):
        rag_engine.embedder.create_batch_embeddings = Mock(return_value=[])
        rag_engine.vector_db.add_documents = Mock(side_effect=ValueError("mismatch"))
        with pytest.raises(ValueError, match="mismatch"):
            rag_engine.store_documentation(["valid docs"])
