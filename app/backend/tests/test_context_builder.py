import pytest
from app.context_builder import ContextBuilder
from app.graph_generator import GraphGenerator


def test_valid_context_creation(make_entry, make_chain):
    dag = GraphGenerator(make_chain(
        make_entry(ts="2023-01-01 00:00:00", msg="Error occurred", level="ERROR"),
        make_entry(ts="2023-01-01 00:00:01", msg="System recovery", level="INFO"),
    )).generate_dag()
    context = ContextBuilder(dag).build_context()
    assert len(context.causal_chain) == 2
    assert any("Error occurred" in msg for msg in context.causal_chain)


def test_empty_dag_raises():
    with pytest.raises(ValueError):
        ContextBuilder(None)
