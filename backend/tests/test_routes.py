"""
Tests for the three API routes:
  POST /api/log/analyse
  POST /api/docs/upload
  POST /api/incident/resolve

Strategy: routes.py instantiates RAG_Engine() and MongoDBHandler() at module
level (lines 27-28). We patch those globals on the already-imported module
object directly via monkeypatch rather than trying to intercept constructors.
"""

import io
import sys
import pytest
from unittest.mock import MagicMock, patch

# ── Stub heavy third-party libs before any app import ─────────────────────
for _mod in ("chromadb", "pymongo", "ollama",
             "langchain", "langchain.text_splitter", "langchain.schema"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Patch ChromaDB PersistentClient so database_handlers import doesn't crash
with patch("chromadb.PersistentClient", MagicMock()), \
     patch("chromadb.HttpClient", MagicMock()):
    from main import app  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
import app.api.routes as routes_module      # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures – replace module-level singletons with mocks for each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_module_globals(monkeypatch):
    """Replace the shared rag and mongo singletons in routes with mocks."""
    monkeypatch.setattr(routes_module, "rag",   MagicMock())
    monkeypatch.setattr(routes_module, "mongo", MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(ts="2023-01-01 00:00:00", msg="DB pool exhausted", level="ERROR"):
    from app.models.parsing_data_models import LogEntry
    return LogEntry(timestamp=ts, message=msg, level=level)


def _make_chain():
    from app.models.parsing_data_models import LogChain
    return LogChain(log_chain=[
        _make_entry(msg="DB pool exhausted", level="ERROR"),
        _make_entry(ts="2023-01-01 00:00:01", msg="Query timeout", level="ERROR"),
    ])


def _make_dag():
    from app.models.graph_data_models import DAG, DAGNode
    n1 = DAGNode(id="1", parent_id=None, children=["2"], log_entry=_make_entry())
    n2 = DAGNode(id="2", parent_id="1",  children=[],    log_entry=_make_entry(msg="Q"))
    return DAG(nodes=[n1, n2], root_id="1",
               root_cause="DB pool exhausted", leaf_ids=["2"])


def _make_context():
    from app.models.context_data_models import Context
    return Context(root_cause="DB pool exhausted",
                   causal_chain=["DB pool exhausted", "Query timeout"])


def _make_summary():
    from app.models.rag_response_data_models import SummaryResponse
    return SummaryResponse(
        summary=["Restart connection pool"],
        root_cause_expln="The DB connection pool was exhausted.",
        severity="High"
    )


# ---------------------------------------------------------------------------
# POST /api/log/analyse
# ---------------------------------------------------------------------------

class TestAnalyseRoute:

    @patch("app.api.routes.LogParser")
    @patch("app.api.routes.GraphGenerator")
    @patch("app.api.routes.ContextBuilder")
    def test_analyse_success(self, MockCtx, MockGen, MockParser):
        """Happy path: valid .log file → 200 with context + severity."""
        MockParser.return_value.parse_log_from_file.return_value = _make_chain()
        MockGen.return_value.generate_dag.return_value = _make_dag()
        MockCtx.return_value.build_context.return_value = _make_context()
        routes_module.rag.generate_summary.return_value = _make_summary()
        routes_module.mongo.save_context = MagicMock()

        data = b"2023-01-01 00:00:00 ERROR DB pool exhausted\n"
        response = client.post(
            "/api/log/analyse",
            files={"file": ("incident.log", io.BytesIO(data), "text/plain")},
        )
        assert response.status_code == 200
        body = response.json()
        assert any(k in body for k in ("_context", "root_cause", "severity", "summary"))

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


# ---------------------------------------------------------------------------
# POST /api/incident/resolve
# ---------------------------------------------------------------------------

class TestResolveRoute:

    def test_resolve_success(self):
        """Client echoes context back → 200 with solution."""
        mock_sol = MagicMock()
        mock_sol.response = "Increase max_connections to 200."
        routes_module.rag.generate_solution.return_value = mock_sol

        payload = {
            "_context": {
                "root_cause": "DB pool exhausted",
                "causal_chain": ["DB pool exhausted", "Query timeout"],
            },
            "_root_cause_expln": "The DB connection pool was exhausted.",
        }
        response = client.post("/api/incident/resolve", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert any(k in body for k in ("solution", "response", "result", "steps"))

    def test_resolve_missing_context_returns_error(self):
        """Empty JSON body → 400 or 422 (missing required keys)."""
        response = client.post("/api/incident/resolve", json={})
        assert response.status_code in (400, 422)

    def test_resolve_no_body_returns_422(self):
        """No body at all → 422."""
        response = client.post("/api/incident/resolve")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

def test_health_check():
    """Health endpoint always responds 200."""
    response = client.get("/api/health")
    assert response.status_code == 200
