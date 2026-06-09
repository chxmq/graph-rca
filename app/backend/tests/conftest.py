import pytest
from app.models import LogChain, LogEntry, DAG, DAGNode, Context, SummaryResponse


@pytest.fixture
def make_entry():
    def _make(ts="2023-01-01 00:00:00", msg="msg", level="INFO", **kwargs) -> LogEntry:
        return LogEntry(timestamp=ts, message=msg, level=level, **kwargs)
    return _make


@pytest.fixture
def make_chain():
    def _make(*entries) -> LogChain:
        return LogChain(log_chain=list(entries), total_lines=len(entries), parsed_lines=len(entries))
    return _make


@pytest.fixture
def make_dag(make_entry):
    def _make() -> DAG:
        n1 = DAGNode(id="1", parent_ids=[], children=["2"], log_entry=make_entry(msg="DB pool exhausted", level="ERROR"))
        n2 = DAGNode(id="2", parent_ids=["1"], children=[], log_entry=make_entry(ts="2023-01-01 00:00:01", msg="Query timeout", level="ERROR"))
        return DAG(nodes=[n1, n2], root_id="1", root_cause="DB pool exhausted", leaf_ids=["2"])
    return _make


@pytest.fixture
def make_context():
    def _make() -> Context:
        return Context(
            dag_id="dag-1",
            root_cause="DB pool exhausted",
            causal_chain=["DB pool exhausted", "Query timeout"],
        )
    return _make


@pytest.fixture
def make_summary():
    def _make() -> SummaryResponse:
        return SummaryResponse(
            summary=["Restart connection pool"],
            root_cause_expln="The DB connection pool was exhausted.",
            severity="High",
        )
    return _make
