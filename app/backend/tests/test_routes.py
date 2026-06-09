import io
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# Disable API_KEY in case the test environment has it set, so the tests
# below exercise the open path. There are dedicated middleware tests for
# the authenticated path in test_middleware.py.
os.environ.pop("API_KEY", None)

from main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures – replace module-level singletons with mocks for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_app_state():
    app.state.rag_engine = MagicMock()
    app.state.rag_engine.generate_summary_async = AsyncMock()
    app.state.rag_engine.generate_solution_async = AsyncMock()
    app.state.rag_engine.store_documentation_async = AsyncMock()
    app.state.mongo_db = MagicMock()
    app.state.mongo_db.upsert_progress = MagicMock()
    app.state.mongo_db.get_progress = MagicMock(return_value=None)
    app.state.vector_db = MagicMock()
    app.state.log_parser = MagicMock()


# ---------------------------------------------------------------------------
# POST /api/log/analyse
# ---------------------------------------------------------------------------

class TestAnalyseRoute:

    @patch("app.routes.GraphGenerator")
    @patch("app.routes.ContextBuilder")
    def test_analyse_success(self, MockCtx, MockGen, make_entry, make_chain, make_dag, make_context, make_summary):
        """Happy path: valid .log file → 200 with context + severity."""
        app.state.log_parser.parse_log_async = AsyncMock(return_value=make_chain(
            make_entry(msg="DB pool exhausted", level="ERROR"),
            make_entry(ts="2023-01-01 00:00:01", msg="Query timeout", level="ERROR"),
        ))
        MockGen.return_value.generate_dag.return_value = make_dag()
        MockCtx.return_value.build_context.return_value = make_context()
        app.state.rag_engine.generate_summary_async = AsyncMock(return_value=make_summary())
        app.state.mongo_db.save_context = MagicMock()

        data = b"2023-01-01 00:00:00 ERROR DB pool exhausted\n"
        response = client.post(
            "/api/log/analyse",
            files={"file": ("incident.log", io.BytesIO(data), "text/plain")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "severity" in body and isinstance(body["severity"], str)
        assert "root_cause" in body and isinstance(body["root_cause"], str)
        assert "summary" in body and isinstance(body["summary"], list)
        assert "context" in body and isinstance(body["context"], dict)
        assert "root_cause_expln" in body and isinstance(body["root_cause_expln"], str)
        assert "truncated" in body and body["truncated"] is False

        app.state.mongo_db.save_dag.assert_called_once()
        app.state.mongo_db.save_context.assert_called_once()

    def test_analyse_rejects_wrong_extension(self):
        """Non .log/.txt files → 400 or 422."""
        response = client.post(
            "/api/log/analyse",
            files={"file": ("data.csv", io.BytesIO(b"a,b"), "text/csv")},
        )
        assert response.status_code in (400, 422)

    def test_analyse_rejects_oversized_file(self):
        """Files over 5 MB → 400 or 413."""
        big = b"x" * (5 * 1024 * 1024 + 1)
        response = client.post(
            "/api/log/analyse",
            files={"file": ("big.log", io.BytesIO(big), "text/plain")},
        )
        assert response.status_code in (400, 413)

    def test_analyse_missing_file_returns_422(self):
        """No file body → FastAPI validation 422."""
        response = client.post("/api/log/analyse")
        assert response.status_code == 422

    def test_analyse_rejects_bad_analysis_id(self, make_entry, make_chain, make_dag, make_context, make_summary):
        """A non-UUIDv4 X-Analysis-ID header is ignored — server generates its own."""
        with patch("app.routes.GraphGenerator") as MockGen, patch("app.routes.ContextBuilder") as MockCtx:
            app.state.log_parser.parse_log_async = AsyncMock(return_value=make_chain(
                make_entry(msg="m", level="ERROR"),
            ))
            MockGen.return_value.generate_dag.return_value = make_dag()
            MockCtx.return_value.build_context.return_value = make_context()
            app.state.rag_engine.generate_summary_async = AsyncMock(return_value=make_summary())

            response = client.post(
                "/api/log/analyse",
                files={"file": ("a.log", io.BytesIO(b"x"), "text/plain")},
                headers={"X-Analysis-ID": "12345678"},  # 8-char alnum, NOT a uuid
            )
            assert response.status_code == 200
            body = response.json()
            # Server-generated UUIDv4
            import re
            assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", body["analysis_id"])


# ---------------------------------------------------------------------------
# POST /api/incident/resolve
# ---------------------------------------------------------------------------

class TestResolveRoute:

    def test_resolve_success(self):
        """Client echoes context back → 200 with solution."""
        mock_sol = MagicMock()
        mock_sol.response = "Increase max_connections to 200."
        mock_sol.sources = ["runbook-db.md", "unknown", "runbook-db.md", "service-guide.md"]
        app.state.rag_engine.generate_solution_async = AsyncMock(return_value=mock_sol)

        payload = {
            "context": {
                "dag_id": "dag-1",
                "created_at": "2023-01-01T00:00:00",
                "root_cause": "DB pool exhausted",
                "causal_chain": ["DB pool exhausted", "Query timeout"],
            },
            "root_cause_expln": "The DB connection pool was exhausted.",
        }
        response = client.post("/api/incident/resolve", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "solution" in body and isinstance(body["solution"], str)
        assert "root_cause" in body and isinstance(body["root_cause"], str)
        assert body["sources"] == ["runbook-db.md", "service-guide.md"]

    def test_resolve_missing_context_returns_error(self):
        """Empty JSON body → 400 or 422 (missing required keys)."""
        response = client.post("/api/incident/resolve", json={})
        assert response.status_code in (400, 422)

    def test_resolve_no_body_returns_422(self):
        """No body at all → 422."""
        response = client.post("/api/incident/resolve")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/docs/upload
# ---------------------------------------------------------------------------

class TestDocsUploadRoute:

    def test_upload_succeeds(self):
        """New doc file → stored and 200 returned."""
        app.state.mongo_db.db = MagicMock()
        app.state.mongo_db.db.__getitem__.return_value.find_one.return_value = None
        app.state.rag_engine.store_documentation_async = AsyncMock()

        response = client.post(
            "/api/docs/upload",
            files=[("files", ("runbook.md", io.BytesIO(b"# Runbook\nrestart the db"), "text/markdown"))],
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        app.state.rag_engine.store_documentation_async.assert_awaited_once()

    def test_upload_cached_doc_skips_indexing(self):
        """Same doc uploaded twice → second call returns cached message."""
        app.state.mongo_db.db = MagicMock()
        app.state.mongo_db.db.__getitem__.return_value.find_one.return_value = {"hash": "abc"}
        app.state.rag_engine.store_documentation_async = AsyncMock()

        response = client.post(
            "/api/docs/upload",
            files=[("files", ("guide.txt", io.BytesIO(b"some guide"), "text/plain"))],
        )
        assert response.status_code == 200
        assert "cached" in response.json()["message"].lower()
        app.state.rag_engine.store_documentation_async.assert_not_awaited()

    def test_upload_rejects_wrong_extension(self):
        """Non .txt/.md files → 400 (no valid docs)."""
        response = client.post(
            "/api/docs/upload",
            files=[("files", ("data.csv", io.BytesIO(b"a,b"), "text/csv"))],
        )
        assert response.status_code == 400

    def test_upload_no_files_returns_422(self):
        """No files body → 422."""
        response = client.post("/api/docs/upload")
        assert response.status_code == 422

    def test_upload_too_many_files_rejected(self):
        """More than MAX_DOC_FILES → 413."""
        from app.routes import MAX_DOC_FILES
        files = [("files", (f"f{i}.md", io.BytesIO(b"x"), "text/markdown")) for i in range(MAX_DOC_FILES + 1)]
        response = client.post("/api/docs/upload", files=files)
        assert response.status_code == 413


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

@patch("app.routes.ServerHealthCheck")
def test_health_check_healthy(mock_checker_cls):
    """Health endpoint returns ok when all services are up."""
    mock_checker_cls.return_value.get_status.return_value = {
        "chromadb": True, "mongodb": True
    }
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["services"]["chromadb"] is True
    assert body["services"]["mongodb"] is True


@patch("app.routes.ServerHealthCheck")
def test_health_check_degraded(mock_checker_cls):
    """Health endpoint returns degraded when a service is down."""
    mock_checker_cls.return_value.get_status.return_value = {
        "chromadb": True, "mongodb": False
    }
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# GET /api/analysis/{id}/progress
# ---------------------------------------------------------------------------

class TestProgressRoute:

    def test_progress_404_when_unknown(self):
        app.state.mongo_db.get_progress = MagicMock(return_value=None)
        response = client.get("/api/analysis/00000000-0000-4000-8000-000000000000/progress")
        assert response.status_code == 404

    def test_progress_returns_state(self):
        app.state.mongo_db.get_progress = MagicMock(
            return_value={"status": "completed", "progress": 100, "parsed_lines": 5}
        )
        response = client.get("/api/analysis/00000000-0000-4000-8000-000000000000/progress")
        assert response.status_code == 200
        assert response.json() == {"status": "completed", "progress": 100, "parsed_lines": 5}


# ---------------------------------------------------------------------------
# GET /api/analysis/{id}/context
# ---------------------------------------------------------------------------

class TestContextRoute:

    def test_context_404_when_unknown(self):
        app.state.mongo_db.get_context_by_analysis_id = MagicMock(return_value=None)
        response = client.get("/api/analysis/abc/context")
        assert response.status_code == 404

    def test_context_returns_payload_without_internals(self):
        app.state.mongo_db.get_context_by_analysis_id = MagicMock(return_value={
            "_id": "mongo-id",
            "analysis_id": "abc",
            "dag_id": "dag-1",
            "root_cause": "x",
            "causal_chain": ["a", "b"],
        })
        response = client.get("/api/analysis/abc/context")
        assert response.status_code == 200
        body = response.json()
        assert "_id" not in body
        assert "analysis_id" not in body
        assert body["dag_id"] == "dag-1"
