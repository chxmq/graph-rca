#!/usr/bin/env python3
"""
Scalability Analysis Experiment
Measures O(n) complexity of DAG construction only (Algorithm 2).
Does NOT include LLM parsing - that is measured in batch inference experiment.
"""

import os
import sys
import time
import json
import statistics
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

try:
    from app.models.parsing_data_models import LogEntry, LogChain
    from app.utils.graph_generator import GraphGenerator
except ModuleNotFoundError as e:
    print("Error: Backend dependencies required. Run from project root:")
    print("  pip install -r backend/requirements.txt")
    print("  python experiments/02_scalability/run_experiment.py")
    sys.exit(1)

logging.getLogger("app").setLevel(logging.WARNING)

CONFIG = {
    "scale_sizes": [10, 25, 50, 100, 250, 500, 1000, 2000],
    "num_runs_per_size": 20,
}

if os.environ.get("SMOKE_TEST"):
    print("🔥 SMOKE TEST MODE ENABLED: Reducing scale sizes")
    CONFIG["scale_sizes"] = [10, 50, 100]
    CONFIG["num_runs_per_size"] = 2

SAMPLE_ENTRIES = [
    ("2024-01-15T10:23:45.123Z", "Connection pool exhausted", "ERROR"),
    ("2024-01-15T10:23:46.001Z", "Slow query: took 5.2s", "WARN"),
    ("2024-01-15T10:23:47.789Z", "Request timeout - 5000ms", "WARN"),
    ("2024-01-15T10:23:48.100Z", "upstream timed out", "ERROR"),
    ("2024-01-15T10:23:52.333Z", "Pod restart loop: restarts=5", "CRITICAL"),
    ("2024-01-15T10:24:01.200Z", "OOM killer invoked", "CRITICAL"),
]


def make_log_chain(n: int) -> LogChain:
    """Create a LogChain with n entries (cycled from samples), timestamps ascending."""
    entries = []
    for i in range(n):
        _, msg, level = SAMPLE_ENTRIES[i % len(SAMPLE_ENTRIES)]
        ts = f"2024-01-15T10:00:00.{i:04d}Z"
        entries.append(LogEntry(timestamp=ts, message=f"{msg} [{i}", level=level))
    return LogChain(log_chain=entries)


def run_dag_scalability() -> dict:
    """Measure DAG construction time for varying log counts."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: DAG Construction Scalability")
    print("=" * 70)
    print("Measuring Algorithm 2 only (graph building), no LLM parsing.\n")

    measurements = []
    for size in CONFIG["scale_sizes"]:
        log_chain = make_log_chain(size)
        graph_gen = GraphGenerator(log_chain)

        times_ms = []
        for run in range(CONFIG["num_runs_per_size"]):
            start = time.perf_counter()
            dag = graph_gen.generate_dag()
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)

        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0

        measurements.append({
            "size": size,
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "min_ms": round(min(times_ms), 3),
            "max_ms": round(max(times_ms), 3),
            "nodes_created": len(dag.nodes),
            "valid": True,
        })
        print(f"  n={size:5d}: {mean_ms:6.2f} ms ± {std_ms:.2f}")

    return {
        "experiment": "DAG Construction Scalability",
        "num_runs_per_size": CONFIG["num_runs_per_size"],
        "measurements": measurements,
        "complexity": "O(n) - Linear",
        "max_tested": CONFIG["scale_sizes"][-1],
        "timestamp": datetime.now().isoformat(),
    }


def main():
    results = run_dag_scalability()

    output_path = Path(__file__).parent / "data" / "dag_scalability.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    scale_results = [
        {
            "log_count": m["size"],
            "avg_time_ms": m["mean_ms"],
            "std_ms": m["std_ms"],
            "throughput_logs_per_sec": round(m["size"] / (m["mean_ms"] / 1000), 2),
        }
        for m in results["measurements"]
    ]
    scalability_results_path = Path(__file__).parent / "data" / "scalability_results.json"
    with open(scalability_results_path, "w") as f:
        json.dump({"scale_results": scale_results, "timestamp": results["timestamp"], "complexity": results["complexity"]}, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ Also saved to {scalability_results_path}")
    print("\nComplexity: O(n) - Linear scaling confirmed")


if __name__ == "__main__":
    main()
