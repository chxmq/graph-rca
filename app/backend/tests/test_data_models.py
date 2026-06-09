import pytest
from datetime import datetime
from app.models import Context
from app.models import DAG, DAGNode
from app.models import LogEntry, LogChain
from app.models import SummaryResponse, SolutionQuery

# ---------------------------------------------------------------------------
# LogEntry Tests
# ---------------------------------------------------------------------------

def test_log_entry_with_all_fields():
    log_entry = LogEntry(
        timestamp="2023-01-01 10:00:00",
        message="Test message",
        level="INFO",
        pid="1234",
        component="test-component",
        error_code="500",
        username="test-user",
        ip_address="127.0.0.1",
        group="test-group",
        trace_id="abc-123",
        request_id="req-123"
    )
    assert log_entry.pid == "1234"
    assert log_entry.username == "test-user"
    assert log_entry.trace_id == "abc-123"


def test_log_entry_minimal_fields():
    """Optional fields default to empty string."""
    log_entry = LogEntry(timestamp="2023-01-01", message="test", level="INFO")
    assert isinstance(log_entry.timestamp, datetime)
    assert log_entry.pid == ""
    assert log_entry.component == ""


def test_log_entry_defaults():
    """All optional fields default to empty string, not None."""
    entry = LogEntry(timestamp="2023-01-01 00:00:00", message="m", level="INFO")
    for field in ("pid", "component", "error_code", "username", "ip_address",
                  "group", "trace_id", "request_id"):
        assert getattr(entry, field) == "", f"{field} should default to ''"


# ---------------------------------------------------------------------------
# LogChain Tests
# ---------------------------------------------------------------------------

def _make_entry(ts="2023-01-01", msg="msg", level="INFO") -> LogEntry:
    return LogEntry(timestamp=ts, message=msg, level=level)


def test_log_chain_single():
    chain = LogChain(log_chain=[_make_entry()])
    assert len(chain.log_chain) == 1


def test_log_chain_multiple():
    chain = LogChain(log_chain=[_make_entry(), _make_entry(msg="second")])
    assert len(chain.log_chain) == 2


# ---------------------------------------------------------------------------
# DAGNode / DAG Tests
# ---------------------------------------------------------------------------

def test_dag_with_multiple_nodes():
    node1 = DAGNode(id="1", parent_ids=[], children=["2"],
                    log_entry=_make_entry(msg="first"))
    node2 = DAGNode(id="2", parent_ids=["1"], children=[],
                    log_entry=_make_entry(level="ERROR", msg="second"))

    dag = DAG(nodes=[node1, node2], root_id="1",
              root_cause="Test cause", leaf_ids=["2"])
    assert len(dag.nodes) == 2
    assert dag.leaf_ids == ["2"]


def test_dag_without_root_cause():
    node = DAGNode(id="1", parent_ids=[], children=[],
                   log_entry=_make_entry())
    dag = DAG(nodes=[node], root_id="1", leaf_ids=["1"])
    assert dag.root_cause is None


def test_dag_single_node_is_root_and_leaf():
    node = DAGNode(id="x", parent_ids=[], children=[],
                   log_entry=_make_entry())
    dag = DAG(nodes=[node], root_id="x", root_cause=None, leaf_ids=["x"])
    assert dag.root_id == dag.leaf_ids[0]


# ---------------------------------------------------------------------------
# Context Tests
# ---------------------------------------------------------------------------

def test_context_with_empty_causal_chain():
    context = Context(dag_id="dag-1", root_cause="Test cause", causal_chain=[])
    assert len(context.causal_chain) == 0


def test_context_with_chain():
    context = Context(dag_id="dag-1", root_cause="DB timeout", causal_chain=["step1", "step2"])
    assert context.causal_chain[0] == "step1"


# ---------------------------------------------------------------------------
# SummaryResponse Tests
# ---------------------------------------------------------------------------

def test_summary_response_with_empty_summary():
    summary = SummaryResponse(summary=[], root_cause_expln="No issues found",
                              severity="Low")
    assert len(summary.summary) == 0


# ---------------------------------------------------------------------------
# SolutionQuery Tests
# ---------------------------------------------------------------------------

def test_solution_query_with_sources():
    query = SolutionQuery(
        context="Error occurred in system",
        query="How to fix?",
        response="Try restarting the service",
        sources=["runbook.md", "sla.txt"]
    )
    assert query.context == "Error occurred in system"
    assert query.sources == ["runbook.md", "sla.txt"]


def test_solution_query_sources_default():
    query = SolutionQuery(context="Error", query="Fix?", response="Check logs")
    assert query.context == "Error"
    assert query.sources == []