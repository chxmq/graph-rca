#!/usr/bin/env python3
"""
Experiment 09: End-to-End Latency Profiling
Measures time spent in each component: LLM Parsing, RAG Retrieval, DAG Construction

Run: python run_experiment.py
Requires: pip install ollama chromadb
"""

import time
import json
import statistics
from datetime import datetime

try:
    import ollama
except ImportError:
    print("Install ollama: pip install ollama")
    exit(1)

try:
    import chromadb
except ImportError:
    print("Install chromadb: pip install chromadb")
    exit(1)

# Sample log entries
SAMPLE_LOGS = [
    "2024-01-15T10:23:45.123Z ERROR [payment-service] Transaction failed: timeout connecting to payment gateway - order_id: ORD-789456",
    "2024-01-15T10:23:46.456Z WARN [database-primary] High query latency detected: 2500ms for SELECT on orders table",
    "2024-01-15T10:23:47.789Z ERROR [api-gateway] Request timeout for /api/checkout - latency exceeded 5000ms threshold",
    "2024-01-15T10:23:48.234Z INFO [cache-server] Cache miss rate increased to 45% - possible memory pressure",
    "2024-01-15T10:23:49.567Z ERROR [auth-service] JWT validation failed: token expired for user_id: USR-123",
]

def measure_llm_parsing(log_entry: str, client: ollama.Client) -> tuple[dict, float]:
    """Measure LLM parsing latency"""
    prompt = f"""Parse this log entry into JSON format:
    {log_entry}
    
    Required fields: timestamp, message, level, component
    Return JSON only."""
    
    start = time.perf_counter()
    try:
        response = client.generate(
            model="llama3.2:3b",
            prompt=prompt,
            format="json"
        )
        elapsed = time.perf_counter() - start
        return {"response": response.response[:100]}, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  LLM error: {e}")
        return {}, elapsed

def measure_dag_construction() -> tuple[str, float]:
    """Measure DAG construction (simulated - actual graph ops)"""
    start = time.perf_counter()
    
    # Simulate DAG operations (these are typically fast in-memory ops)
    # Actual implementation would add nodes/edges to NetworkX graph
    import hashlib
    node_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    
    # Simulate some graph operations
    graph_data = {"nodes": [], "edges": []}
    for i in range(10):
        graph_data["nodes"].append({"id": f"node_{i}", "type": "incident"})
    for i in range(9):
        graph_data["edges"].append({"from": f"node_{i}", "to": f"node_{i+1}"})
    
    elapsed = time.perf_counter() - start
    return node_id, elapsed

def measure_rag_retrieval(query: str) -> tuple[list, float]:
    """Measure RAG retrieval latency"""
    start = time.perf_counter()
    
    try:
        # Create ephemeral client (no persistence needed for timing)
        client = chromadb.Client()
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name="test_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add some test documents if empty
        if collection.count() == 0:
            collection.add(
                documents=[
                    "Payment gateway timeout occurs when the payment service cannot reach the gateway within 30 seconds.",
                    "Database latency issues are often caused by missing indexes or connection pool exhaustion.",
                    "API gateway timeouts happen when backend services don't respond within configured thresholds.",
                    "Cache miss rates increase when memory pressure causes evictions of frequently accessed data.",
                    "JWT validation failures occur when tokens expire or signature verification fails.",
                ],
                ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
            )
        
        # Query
        results = collection.query(query_texts=[query], n_results=3)
        elapsed = time.perf_counter() - start
        
        return results.get("documents", []), elapsed
        
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  RAG error: {e}")
        return [], elapsed

def run_profiling(num_runs: int = 5):
    """Run latency profiling experiment"""
    print("=" * 60)
    print("End-to-End Latency Profiling Experiment")
    print("=" * 60)
    
    # Initialize Ollama client
    print("Connecting to Ollama...")
    try:
        # Try common ports
        for port in [11434, 11435]:
            try:
                client = ollama.Client(host=f'http://localhost:{port}', timeout=60.0)
                client.list()  # Test connection
                print(f"Connected to Ollama on port {port}")
                break
            except:
                continue
        else:
            print("ERROR: Cannot connect to Ollama. Is it running?")
            print("Try: ollama serve")
            return None
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return None
    
    print(f"Runs per log: {num_runs}")
    print(f"Sample logs: {len(SAMPLE_LOGS)}")
    print()
    
    llm_latencies = []
    dag_latencies = []
    rag_latencies = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        
        for i, log in enumerate(SAMPLE_LOGS):
            print(f"  Log {i+1}/{len(SAMPLE_LOGS)}...", end=" ", flush=True)
            
            # 1. LLM Parsing
            parsed, llm_time = measure_llm_parsing(log, client)
            llm_latencies.append(llm_time * 1000)  # Convert to ms
            print(f"LLM: {llm_time*1000:.0f}ms", end=" ", flush=True)
            
            # 2. DAG Construction
            node_id, dag_time = measure_dag_construction()
            dag_latencies.append(dag_time * 1000)
            print(f"DAG: {dag_time*1000:.1f}ms", end=" ", flush=True)
            
            # 3. RAG Retrieval
            docs, rag_time = measure_rag_retrieval(log[:100])
            rag_latencies.append(rag_time * 1000)
            print(f"RAG: {rag_time*1000:.0f}ms")
    
    # Calculate statistics
    llm_mean = statistics.mean(llm_latencies)
    dag_mean = statistics.mean(dag_latencies)
    rag_mean = statistics.mean(rag_latencies)
    total_mean = llm_mean + dag_mean + rag_mean
    
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
                "mean": round(llm_mean, 1),
                "std": round(statistics.stdev(llm_latencies) if len(llm_latencies) > 1 else 0, 1),
            },
            "dag_construction": {
                "mean": round(dag_mean, 1),
                "std": round(statistics.stdev(dag_latencies) if len(dag_latencies) > 1 else 0, 1),
            },
            "rag_retrieval": {
                "mean": round(rag_mean, 1),
                "std": round(statistics.stdev(rag_latencies) if len(rag_latencies) > 1 else 0, 1),
            },
            "total": {
                "mean": round(total_mean, 1),
            }
        },
        "breakdown_percentage": {
            "llm_parsing": round(llm_mean / total_mean * 100, 1) if total_mean > 0 else 0,
            "rag_retrieval": round(rag_mean / total_mean * 100, 1) if total_mean > 0 else 0,
            "dag_construction": round(dag_mean / total_mean * 100, 1) if total_mean > 0 else 0,
        }
    }
    
    return results

def main():
    print("Starting latency profiling experiment...")
    print("Requires: Ollama running with llama3.2:3b model\n")
    
    results = run_profiling(num_runs=3)
    
    if results is None:
        return
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nLatency Breakdown (ms):")
    for component in ["llm_parsing", "dag_construction", "rag_retrieval"]:
        stats = results["latency_ms"][component]
        print(f"  {component}: {stats['mean']:.1f}ms (±{stats['std']:.1f})")
    print(f"  TOTAL: {results['latency_ms']['total']['mean']:.1f}ms")
    
    print("\nPercentage Breakdown:")
    for component, pct in results["breakdown_percentage"].items():
        print(f"  {component}: {pct:.1f}%")
    
    # Save results
    import os
    os.makedirs("data", exist_ok=True)
    with open("data/latency_breakdown.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: data/latency_breakdown.json")
    
    # Print values for fig7
    print("\n" + "=" * 60)
    print("VALUES FOR FIG 7 SCRIPT:")
    print("=" * 60)
    llm = int(results["latency_ms"]["llm_parsing"]["mean"])
    rag = int(results["latency_ms"]["rag_retrieval"]["mean"])
    dag = int(results["latency_ms"]["dag_construction"]["mean"])
    total = int(results["latency_ms"]["total"]["mean"])
    other = max(0, total - llm - rag - dag)
    
    print(f"times_ms = [{llm}, {rag}, {dag}, {other}]")
    print(f"# Total: {total}ms")
    print(f"# LLM: {results['breakdown_percentage']['llm_parsing']}%")
    print(f"# RAG: {results['breakdown_percentage']['rag_retrieval']}%")
    print(f"# DAG: {results['breakdown_percentage']['dag_construction']}%")

if __name__ == "__main__":
    main()
