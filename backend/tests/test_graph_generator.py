from app.graph_generator import GraphGenerator


def test_dag_generation(make_entry, make_chain):
    chain = make_chain(
        make_entry(ts="2023-01-01 00:00:00", msg="Error occurred", level="ERROR"),
        make_entry(ts="2023-01-01 00:00:01", msg="System recovery", level="INFO"),
    )
    dag = GraphGenerator(chain).generate_dag()
    assert len(dag.nodes) == 2
    assert dag.root_id is not None
    assert len(dag.leaf_ids) >= 1


def test_temporal_ordering(make_entry, make_chain):
    chain = make_chain(
        make_entry(ts="2023-01-01 00:00:00", msg="first", level="INFO"),
        make_entry(ts="2023-01-01 00:00:01", msg="second", level="ERROR"),
    )
    dag = GraphGenerator(chain).generate_dag()
    root = next(n for n in dag.nodes if n.id == dag.root_id)
    leaf = next(n for n in dag.nodes if n.id == dag.leaf_ids[0])
    assert root.log_entry.timestamp <= leaf.log_entry.timestamp


def test_unrelated_logs_do_not_get_temporal_edge(make_entry, make_chain):
    chain = make_chain(
        make_entry(ts="2023-01-01 00:00:00", msg="first", level="INFO"),
        make_entry(ts="2023-01-01 00:00:01", msg="second", level="INFO"),
    )
    dag = GraphGenerator(chain).generate_dag()
    assert all(not node.parent_ids for node in dag.nodes)


def test_trace_id_links_beyond_previous_lookback_window(make_entry, make_chain):
    entries = [
        make_entry(ts="2023-01-01 00:00:00", msg="request start", level="INFO", trace_id="trace-1")
    ]
    entries.extend(
        make_entry(ts=f"2023-01-01 00:00:{i:02d}", msg=f"noise {i}", level="INFO")
        for i in range(1, 25)
    )
    entries.append(
        make_entry(ts="2023-01-01 00:00:25", msg="request failed", level="ERROR", trace_id="trace-1")
    )

    dag = GraphGenerator(make_chain(*entries)).generate_dag()
    failed = next(node for node in dag.nodes if node.log_entry.message == "request failed")
    start = next(node for node in dag.nodes if node.log_entry.message == "request start")
    assert failed.parent_ids == [start.id]


def test_root_cause_candidates_not_first_five_only(make_entry, make_chain):
    entries = [make_entry(ts=f"2023-01-01 00:00:{i:02d}", msg=f"info {i}", level="INFO") for i in range(6)]
    entries.append(make_entry(ts="2023-01-01 00:00:06", msg="late error", level="ERROR"))
    dag = GraphGenerator(make_chain(*entries)).generate_dag()
    assert dag.root_cause == "late error"
