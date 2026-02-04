#!/usr/bin/env python3
"""
Experiment 09: End-to-End Latency Profiling
Measures time spent in each component: LLM Parsing, RAG Retrieval, DAG Construction

Run this to get accurate latency breakdown for Fig 7.
"""

import sys
import os
import time
import json
import statistics
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

# Sample log entries for testing
SAMPLE_LOGS = [
    "2024-01-15T10:23:45.123Z ERROR [payment-service] Transaction failed: timeout connecting to payment gateway - order_id: ORD-789456",
    "2024-01-15T10:23:46.456Z WARN [database-primary] High query latency detected: 2500ms for SELECT on orders table",
    "2024-01-15T10:23:47.789Z ERROR [api-gateway] Request timeout for /api/checkout - latency exceeded 5000ms threshold",
    "2024-01-15T10:23:48.234Z INFO [cache-server] Cache miss rate increased to 45% - possible memory pressure",
    "2024-01-15T10:23:49.567Z ERROR [auth-service] JWT validation failed: token expired for user_id: USR-123",
]

def measure_llm_parsing(log_entry: str) -> tuple[dict, float]:
    """Measure LLM parsing latency"""
    try:
        from app.services.ollama_service import parse_log
        
        start = time.perf_counter()
        result = parse_log(log_entry)
        elapsed = time.perf_counter() - start
        
        return result, elapsed
    except Exception as e:
        print(f"LLM parsing error: {e}")
        return {}, 0.0

def measure_dag_construction(parsed_log: dict) -> tuple[str, float]:
    """Measure DAG construction latency"""
    try:
        from app.services.dag_service import add_incident
        
        start = time.perf_counter()
        incident_id = add_incident(parsed_log)
        elapsed = time.perf_counter() - start
        
        return incident_id, elapsed
    except Exception as e:
        print(f"DAG construction error: {e}")
        return "", 0.0

def measure_rag_retrieval(query: str) -> tuple[list, float]:
    """Measure RAG retrieval latency"""
    try:
        from app.services.rag_service import query_documentation
        
        start = time.perf_counter()
        results = query_documentation(query, n_results=5)
        elapsed = time.perf_counter() - start
        
        return results, elapsed
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return [], 0.0

def run_profiling(num_runs: int = 10):
    """Run latency profiling experiment"""
    print("=" * 60)
    print("End-to-End Latency Profiling Experiment")
    print("=" * 60)
    print(f"Runs per log: {num_runs}")
    print(f"Sample logs: {len(SAMPLE_LOGS)}")
    print()
    
    llm_latencies = []
    dag_latencies = []
    rag_latencies = []
    total_latencies = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        
        for log in SAMPLE_LOGS:
            total_start = time.perf_counter()
            
            # 1. LLM Parsing
            parsed, llm_time = measure_llm_parsing(log)
            llm_latencies.append(llm_time * 1000)  # Convert to ms
            
            if parsed:
                # 2. DAG Construction
                incident_id, dag_time = measure_dag_construction(parsed)
                dag_latencies.append(dag_time * 1000)
                
                # 3. RAG Retrieval (using root cause as query)
                root_cause = parsed.get('root_cause', log[:100])
                docs, rag_time = measure_rag_retrieval(root_cause)
                rag_latencies.append(rag_time * 1000)
            else:
                dag_latencies.append(0)
                rag_latencies.append(0)
            
            total_elapsed = (time.perf_counter() - total_start) * 1000
            total_latencies.append(total_elapsed)
    
    # Calculate statistics
    results = {
        "experiment": "latency_profiling",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_runs": num_runs,
            "num_logs": len(SAMPLE_LOGS),
            "total_samples": num_runs * len(SAMPLE_LOGS)
        },
        "latency_ms": {
            "llm_parsing": {
                "mean": round(statistics.mean(llm_latencies), 1),
                "std": round(statistics.stdev(llm_latencies) if len(llm_latencies) > 1 else 0, 1),
                "median": round(statistics.median(llm_latencies), 1),
                "min": round(min(llm_latencies), 1),
                "max": round(max(llm_latencies), 1),
            },
            "dag_construction": {
                "mean": round(statistics.mean(dag_latencies), 1),
                "std": round(statistics.stdev(dag_latencies) if len(dag_latencies) > 1 else 0, 1),
                "median": round(statistics.median(dag_latencies), 1),
                "min": round(min(dag_latencies), 1),
                "max": round(max(dag_latencies), 1),
            },
            "rag_retrieval": {
                "mean": round(statistics.mean(rag_latencies), 1),
                "std": round(statistics.stdev(rag_latencies) if len(rag_latencies) > 1 else 0, 1),
                "median": round(statistics.median(rag_latencies), 1),
                "min": round(min(rag_latencies), 1),
                "max": round(max(rag_latencies), 1),
            },
            "total": {
                "mean": round(statistics.mean(total_latencies), 1),
                "std": round(statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0, 1),
                "median": round(statistics.median(total_latencies), 1),
            }
        }
    }
    
    # Calculate percentages
    total_mean = results["latency_ms"]["total"]["mean"]
    if total_mean > 0:
        results["breakdown_percentage"] = {
            "llm_parsing": round(results["latency_ms"]["llm_parsing"]["mean"] / total_mean * 100, 1),
            "dag_construction": round(results["latency_ms"]["dag_construction"]["mean"] / total_mean * 100, 1),
            "rag_retrieval": round(results["latency_ms"]["rag_retrieval"]["mean"] / total_mean * 100, 1),
        }
        # Calculate "other" as remainder
        accounted = sum(results["breakdown_percentage"].values())
        results["breakdown_percentage"]["other"] = round(100 - accounted, 1)
    
    return results

def main():
    print("Starting latency profiling experiment...")
    print("Make sure Ollama is running and MongoDB is accessible.\n")
    
    # Run experiment
    results = run_profiling(num_runs=5)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nLatency Breakdown (ms):")
    for component, stats in results["latency_ms"].items():
        if component != "total":
            print(f"  {component}: {stats['mean']:.1f}ms (±{stats['std']:.1f})")
    print(f"  TOTAL: {results['latency_ms']['total']['mean']:.1f}ms")
    
    print("\nPercentage Breakdown:")
    for component, pct in results["breakdown_percentage"].items():
        print(f"  {component}: {pct:.1f}%")
    
    # Save results
    os.makedirs("data", exist_ok=True)
    output_file = "data/latency_breakdown.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print values for fig7 script
    print("\n" + "=" * 60)
    print("VALUES FOR FIG 7 SCRIPT:")
    print("=" * 60)
    llm = results["latency_ms"]["llm_parsing"]["mean"]
    rag = results["latency_ms"]["rag_retrieval"]["mean"]
    dag = results["latency_ms"]["dag_construction"]["mean"]
    total = results["latency_ms"]["total"]["mean"]
    other = total - llm - rag - dag
    
    print(f"times_ms = [{int(llm)}, {int(rag)}, {int(dag)}, {int(max(0, other))}]")
    print(f"# LLM: {results['breakdown_percentage']['llm_parsing']}%")
    print(f"# RAG: {results['breakdown_percentage']['rag_retrieval']}%")
    print(f"# DAG: {results['breakdown_percentage']['dag_construction']}%")
    print(f"# Other: {results['breakdown_percentage']['other']}%")

if __name__ == "__main__":
    main()
