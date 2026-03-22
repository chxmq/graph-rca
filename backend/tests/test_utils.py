import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from unittest.mock import Mock, patch

from app.utils.log_parser import LogParser
from app.utils.graph_generator import GraphGenerator
from app.utils.context_builder import ContextBuilder
from app.utils.database_healthcheck import ServerHealthCheck
from app.models.parsing_data_models import LogChain, LogEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(ts="2023-01-01", msg="msg", level="INFO") -> LogEntry:
    return LogEntry(timestamp=ts, message=msg, level=level)


def _make_chain(*entries) -> LogChain:
    return LogChain(log_chain=list(entries))


@pytest.fixture
def two_entry_chain() -> LogChain:
    return _make_chain(
        _make_entry(ts="2023-01-01 00:00:00", msg="Error occurred", level="ERROR"),
        _make_entry(ts="2023-01-01 00:00:01", msg="System recovery", level="INFO"),
    )


# ---------------------------------------------------------------------------
# TestLogParser
# ---------------------------------------------------------------------------

class TestLogParser:
    @pytest.fixture
    def mock_ollama(self):
        with patch("ollama.Client") as mock:
            yield mock

    def test_parse_valid_log(self, mock_ollama):
        """parse_log returns a LogChain with one correctly parsed entry."""
        mock_response = Mock()
        mock_response.response = json.dumps({
            "timestamp": "2023-01-01",
            "message": "Test message",
            "level": "INFO",
            "pid": "1234",
            "component": "",
            "error_code": "",
            "username": "",
            "ip_address": "",
            "group": "",
            "trace_id": "",
            "request_id": ""
        })
        mock_ollama.return_value.generate.return_value = mock_response

        parser = LogParser()
        result = parser.parse_log("2023-01-01 INFO Test message")
        assert len(result.log_chain) == 1
        assert result.log_chain[0].message == "Test message"

    def test_parse_invalid_file_type(self, tmp_path):
        """parse_log_from_file raises ValueError for unsupported extensions."""
        parser = LogParser()
        bad_file = tmp_path / "data.csv"
        bad_file.touch()
        with pytest.raises(ValueError, match="Unsupported file type"):
            parser.parse_log_from_file(str(bad_file))

    def test_empty_log_raises(self):
        """parse_log raises RuntimeError on empty input."""
        parser = LogParser()
        with pytest.raises((RuntimeError, ValueError)):
            parser.parse_log("")


# ---------------------------------------------------------------------------
# TestGraphGenerator
# ---------------------------------------------------------------------------

class TestGraphGenerator:
    def test_dag_generation(self, two_entry_chain):
        gen = GraphGenerator(two_entry_chain)
        dag = gen.generate_dag()

        assert len(dag.nodes) == 2
        assert dag.root_id is not None
        assert len(dag.leaf_ids) == 1

    def test_parent_child_relationships(self, two_entry_chain):
        gen = GraphGenerator(two_entry_chain)
        dag = gen.generate_dag()

        root = next(n for n in dag.nodes if n.id == dag.root_id)
        assert len(root.children) == 1

    def test_single_node_dag(self):
        chain = _make_chain(_make_entry(msg="Only entry"))
        dag = GraphGenerator(chain).generate_dag()
        assert dag.root_id == dag.leaf_ids[0]

    def test_dag_is_acyclic(self, two_entry_chain):
        """Every edge must go from parent → child with no back-edges."""
        dag = GraphGenerator(two_entry_chain).generate_dag()
        node_map = {n.id: n for n in dag.nodes}
        visited: set = set()

        def dfs(nid):
            assert nid not in visited, f"Cycle detected at node {nid}"
            visited.add(nid)
            for child in node_map[nid].children:
                dfs(child)

        dfs(dag.root_id)

    def test_temporal_ordering(self, two_entry_chain):
        """Earlier timestamp must be the root, later must be the leaf."""
        dag = GraphGenerator(two_entry_chain).generate_dag()
        root = next(n for n in dag.nodes if n.id == dag.root_id)
        leaf_id = dag.leaf_ids[0]
        leaf = next(n for n in dag.nodes if n.id == leaf_id)
        assert root.log_entry.timestamp <= leaf.log_entry.timestamp


# ---------------------------------------------------------------------------
# TestContextBuilder
# ---------------------------------------------------------------------------

class TestContextBuilder:
    def test_valid_context_creation(self, two_entry_chain):
        dag = GraphGenerator(two_entry_chain).generate_dag()
        builder = ContextBuilder(dag)
        context = builder.build_context()

        assert len(context.causal_chain) == 2
        assert any("Error occurred" in msg for msg in context.causal_chain)

    def test_empty_dag_raises(self):
        """ContextBuilder raises ValueError when given None."""
        with pytest.raises(ValueError):
            ContextBuilder(None)

    def test_single_node_context(self):
        chain = _make_chain(_make_entry(msg="Solo event", level="ERROR"))
        dag = GraphGenerator(chain).generate_dag()
        ctx = ContextBuilder(dag).build_context()
        assert len(ctx.causal_chain) == 1
        assert ctx.causal_chain[0] == "Solo event"

    def test_root_cause_propagated(self, two_entry_chain):
        dag = GraphGenerator(two_entry_chain).generate_dag()
        dag.root_cause = "Connection pool exhaustion"
        ctx = ContextBuilder(dag).build_context()
        assert ctx.root_cause == "Connection pool exhaustion"


# ---------------------------------------------------------------------------
# TestDatabaseHealthCheck
# Note: ServerHealthCheck.__init__ calls BOTH chromadb.HttpClient AND
# pymongo.MongoClient, so every test needs both patches active.
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_services():
    """Patch both DB clients so __init__ never touches a real service."""
    with patch("chromadb.HttpClient") as mock_chroma, \
         patch("pymongo.MongoClient") as mock_mongo:
        mock_chroma.return_value.heartbeat.return_value = 1
        mock_mongo.return_value.server_info.return_value = {"ok": 1.0}
        yield mock_chroma, mock_mongo


class TestDatabaseHealthCheck:
    def test_chroma_connection_success(self, mock_db_services):
        mock_chroma, _ = mock_db_services
        assert ServerHealthCheck().check_chroma() is True

    def test_mongo_connection_success(self, mock_db_services):
        _, mock_mongo = mock_db_services
        assert ServerHealthCheck().check_mongo() is True

    def test_chroma_connection_failure(self, mock_db_services):
        """Returns False (not raises) when ChromaDB is unreachable."""
        mock_chroma, _ = mock_db_services
        mock_chroma.return_value.heartbeat.side_effect = Exception("Connection failed")
        assert ServerHealthCheck().check_chroma() is False

    def test_mongo_connection_failure(self, mock_db_services):
        """Returns False (not raises) when MongoDB is unreachable."""
        _, mock_mongo = mock_db_services
        mock_mongo.return_value.server_info.side_effect = Exception("Auth failed")
        assert ServerHealthCheck().check_mongo() is False

    def test_chroma_returns_zero(self, mock_db_services):
        """heartbeat() == 0 → False."""
        mock_chroma, _ = mock_db_services
        mock_chroma.return_value.heartbeat.return_value = 0
        assert ServerHealthCheck().check_chroma() is False

    def test_mongo_returns_bad_ok(self, mock_db_services):
        """ok != 1.0 → False."""
        _, mock_mongo = mock_db_services
        mock_mongo.return_value.server_info.return_value = {"ok": 0.0}
        assert ServerHealthCheck().check_mongo() is False

    def test_get_status_returns_dict(self, mock_db_services):
        """get_status() returns dict with chromadb and mongodb keys."""
        status = ServerHealthCheck().get_status()
        assert "chromadb" in status
        assert "mongodb" in status
        assert status["chromadb"] is True
        assert status["mongodb"] is True

    def test_get_status_reflects_failures(self, mock_db_services):
        mock_chroma, _ = mock_db_services
        mock_chroma.return_value.heartbeat.side_effect = Exception("down")
        status = ServerHealthCheck().get_status()
        assert status["chromadb"] is False
