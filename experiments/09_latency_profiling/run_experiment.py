#!/usr/bin/env python3
"""
Experiment 09: End-to-End Latency Profiling
Measures time spent in each component: LLM Parsing, RAG Retrieval, DAG Construction
CORRECTED: Uses actual backend components for measurement.
"""

import time
import json
import statistics
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Backend Integration ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
try:
    from app.utils.log_parser import LogParser
    from app.core.database_handlers import VectorDatabaseHandler
    from app.models.parsing_data_models import LogEntry, LogChain
    from app.utils.graph_generator import GraphGenerator
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)
# ---------------------------

# Setup Env
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']
    
EXP_CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db_latency"
os.environ["CHROMADB_PATH"] = str(EXP_CHROMA_DIR.absolute())

SAMPLE_LOGS = [
    "2024-01-15T10:23:45.123Z ERROR [payment-service] Transaction failed: timeout connecting to payment gateway",
    "2024-01-15T10:23:46.456Z WARN [database-primary] High query latency detected: 2500ms for SELECT on orders table",
    "2024-01-15T10:23:47.789Z ERROR [api-gateway] Request timeout for /api/checkout - latency exceeded 5000ms threshold",
]

def measure_llm_parsing(log_entry: str) -> float:
    """Measure LogParser latency"""
    parser = LogParser(model="llama3.2:3b")
    start = time.perf_counter()
    try:
        parser.extract_log_info_by_llm(log_entry)
        elapsed = time.perf_counter() - start
        return elapsed
    except Exception as e:
        print(f"LLM Error: {e}")
        return time.perf_counter() - start

def measure_dag_construction(log_entries: list) -> float:
    """Measure GraphGenerator latency"""
    # Create chain manually to isolate graph gen time
    entries = []
    for i, txt in enumerate(log_entries):
        entries.append(LogEntry(
            timestamp=f"2024-01-01T10:00:0{i}Z", message=txt, level="INFO",
            pid="", component="", error_code="", username="", ip_address="", group="", trace_id="", request_id=""
        ))
    chain = LogChain(log_chain=entries)
    
    start = time.perf_counter()
    try:
        gen = GraphGenerator(chain)
        gen.generate_dag()
        elapsed = time.perf_counter() - start
        return elapsed
    except Exception as e:
        print(f"DAG Error: {e}")
        return time.perf_counter() - start

def measure_rag_retrieval(query: str) -> float:
    """Measure VectorDB search latency"""
    vdb = VectorDatabaseHandler()
    
    # Ensure there's at least one doc so search works
    try:
        if not EXP_CHROMA_DIR.exists():
            vdb.ef(["warmup"]) # warmup
            vdb.add_documents(["test doc"], [[0.1]*768])
    except: pass
        
    start = time.perf_counter()
    try:
        vdb.search(query=query, context="", top_k=1)
        elapsed = time.perf_counter() - start
        return elapsed
    except Exception as e:
        print(f"RAG Error: {e}")
        return time.perf_counter() - start

def run_profiling(num_runs: int = 5):
    if os.environ.get("SMOKE_TEST"):
        print("🔥 SMOKE TEST MODE ENABLED: 1 run only")
        num_runs = 1
    print("=" * 60)
    print("End-to-End Latency Profiling Experiment (CORRECTED)")
    print("=" * 60)
    
    llm_latencies = []
    dag_latencies = []
    rag_latencies = []
    
    print("Warming up...")
    measure_llm_parsing(SAMPLE_LOGS[0])
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        
        # 1. Parsing
        t = measure_llm_parsing(SAMPLE_LOGS[0])
        llm_latencies.append(t * 1000)
        print(f"  LLM: {t*1000:.0f}ms")
        
        # 2. DAG
        t = measure_dag_construction(SAMPLE_LOGS)
        dag_latencies.append(t * 1000)
        print(f"  DAG: {t*1000:.1f}ms")
        
        # 3. RAG
        t = measure_rag_retrieval("payment failure")
        rag_latencies.append(t * 1000)
        print(f"  RAG: {t*1000:.0f}ms")

    # Stats
    llm_mean = statistics.mean(llm_latencies)
    dag_mean = statistics.mean(dag_latencies)
    rag_mean = statistics.mean(rag_latencies)
    total_mean = llm_mean + dag_mean + rag_mean
    
    results = {
        "experiment": "latency_profiling",
        "timestamp": datetime.now().isoformat(),
        "latency_ms": {
            "llm_parsing": {"mean": round(llm_mean, 1)},
            "dag_construction": {"mean": round(dag_mean, 1)},
            "rag_retrieval": {"mean": round(rag_mean, 1)},
            "total": {"mean": round(total_mean, 1)}
        },
        "breakdown_percentage": {
            "llm_parsing": round(llm_mean / total_mean * 100, 1),
            "rag_retrieval": round(rag_mean / total_mean * 100, 1),
            "dag_construction": round(dag_mean / total_mean * 100, 1),
        }
    }
    
    return results

def main():
    results = run_profiling(num_runs=3)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nLatency Breakdown (ms):")
    for component in ["llm_parsing", "dag_construction", "rag_retrieval"]:
        stats = results["latency_ms"][component]
        print(f"  {component}: {stats['mean']:.1f}ms")
    print(f"  TOTAL: {results['latency_ms']['total']['mean']:.1f}ms")
    
    output_path = Path(__file__).parent / "data" / "latency_breakdown.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # cleanup
    import shutil
    if EXP_CHROMA_DIR.exists():
        shutil.rmtree(EXP_CHROMA_DIR)

if __name__ == "__main__":
    main()
