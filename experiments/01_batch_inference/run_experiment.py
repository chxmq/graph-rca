#!/usr/bin/env python3
"""
Batch Inference Experiment
Tests batch sizes 1, 8, 16, 32 for LLM parsing performance.
Matches paper methodology: 128 logs, 5 runs per batch size.
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path
from datetime import datetime

if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "temperature": 0.2,
    "batch_sizes": [1, 8, 16, 32],
    "num_runs": 5,
    "num_logs": 128,
}

SAMPLE_LOGS = [
    "2024-01-15T10:23:45.123Z ERROR [db-pool] Connection pool exhausted - no available connections after 30s timeout",
    "2024-01-15T10:23:46.001Z WARN [mysql] Slow query detected: SELECT * FROM users took 5.2s",
    "2024-01-15T10:23:46.234Z ERROR [postgres] FATAL: too many connections for role 'app_user'",
    "2024-01-15T10:23:46.456Z INFO [db-replica] Replication lag: primary=db-01, lag_ms=1500",
    "2024-01-15T10:23:47.789Z WARN [api-gateway] Request timeout for /api/users - latency 5000ms",
    "2024-01-15T10:23:48.100Z ERROR [nginx] upstream timed out (110: Connection timed out)",
    "2024-01-15T10:23:49.000Z INFO [auth-service] User login successful: user_id=12345",
    "Jan 15 10:23:48 server01 sshd[1234]: Failed password for invalid user admin from 10.0.0.1",
    "2024-01-15T10:23:49.500Z WARN [auth] Multiple failed login attempts: user=admin, attempts=5",
    "2024-01-15T10:23:52.333Z CRITICAL [k8s-controller] Pod restart loop: pod=api-server, restarts=5",
    "2024-01-15T10:23:52.444Z INFO [kubelet] Container started: container=app, image=myapp:v2.1.0",
    "2024-01-15T10:24:01.200Z CRITICAL [memory-monitor] OOM killer invoked: process=java, memory=8GB",
    "2024-01-15T10:24:01.300Z WARN [gc] GC pause exceeded threshold: pause=2.5s, heap=7.2GB",
    "2024-01-15T10:23:53.444Z INFO [load-balancer] Backend health check passed: latency=15ms",
    "2024-01-15T10:23:53.555Z WARN [haproxy] Server backend/server2 is DOWN: Layer4 timeout",
    "2024-01-15T10:23:54.555Z WARN [storage-svc] Disk usage above threshold: usage=92%",
    "2024-01-15T10:23:54.666Z ERROR [nfs] NFS: server not responding",
    "2024-01-15T10:23:55.666Z ERROR [msg-queue] Message processing failed: retry=3/5",
    "2024-01-15T10:23:55.777Z WARN [kafka] Consumer lag: topic=events, lag=50000",
    "2024-01-15T10:23:56.666Z INFO [scheduler] Cron job completed: duration=45s",
    "2024-01-15T10:23:57.777Z ERROR [dns-resolver] DNS resolution timeout: 10s",
]


def run_batch_experiment(client: ollama.Client) -> dict:
    """Run batch inference benchmark."""
    print("=" * 70)
    print("EXPERIMENT: Batch Inference Benchmark")
    print(f"Config: {CONFIG['num_logs']} logs, {CONFIG['num_runs']} runs per batch size")
    print(f"Device: {CONFIG.get('device', 'Unknown')}")
    print("=" * 70)
    
    logs = (SAMPLE_LOGS * 10)[:CONFIG["num_logs"]]
    options = ollama.Options(temperature=CONFIG["temperature"])
    results = {"batch_results": [], "latex": ""}
    baseline_throughput = None
    
    for batch_size in CONFIG["batch_sizes"]:
        print(f"\nBatch size: {batch_size}")
        run_times = []
        
        for run in range(1, CONFIG["num_runs"] + 1):
            start = time.time()
            
            if batch_size == 1:
                for log in logs:
                    try:
                        client.generate(
                            model=CONFIG["model"],
                            prompt=f"Parse log to JSON: {log}",
                            options=options, format="json"
                        )
                    except: pass
            else:
                for i in range(0, len(logs), batch_size):
                    batch = logs[i:i+batch_size]
                    try:
                        client.generate(
                            model=CONFIG["model"],
                            prompt=f"Parse {len(batch)} logs to JSON array:\n" + 
                                   "\n".join([f"[{j+1}]: {l}" for j,l in enumerate(batch)]),
                            options=options, format="json"
                        )
                    except: pass
            
            elapsed = time.time() - start
            run_times.append(elapsed)
            print(f"  Run {run}: {elapsed:.1f}s ({len(logs)/elapsed:.2f} logs/s)")
        
        avg_time = statistics.mean(run_times)
        throughput = len(logs) / avg_time
        latency = (avg_time * 1000) / len(logs)
        std = statistics.stdev(run_times) if len(run_times) > 1 else 0
        
        if baseline_throughput is None:
            baseline_throughput = throughput
        
        results["batch_results"].append({
            "batch_size": batch_size,
            "throughput": round(throughput, 3),
            "latency_ms": round(latency, 1),
            "std_dev": round(std, 3),
            "speedup": round(throughput / baseline_throughput, 2),
            "runs": run_times
        })
        
        print(f"  → Avg: {throughput:.2f} logs/s, {latency:.0f}ms/log, {throughput/baseline_throughput:.1f}× speedup")
    
    results["timestamp"] = datetime.now().isoformat()
    results["hardware"] = CONFIG.get("device", "Unknown")
    return results


import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Batch Inference Experiment")
    parser.add_argument("--device", type=str, required=True, help="Device name (e.g., 'Quadro GV100', 'NVIDIA A100-40GB')")
    args = parser.parse_args()
    
    CONFIG["device"] = args.device

    client = ollama.Client(host=CONFIG["ollama_host"])
    results = run_batch_experiment(client)
    
    # Create filename safe string
    device_slug = args.device.lower().replace(" ", "_").replace("-", "_")
    output_filename = f"batch_results_{device_slug}.json"
    
    output_path = Path(__file__).parent / "data" / output_filename
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\nSummary:")
    for r in results["batch_results"]:
        print(f"  Batch {r['batch_size']:2d}: {r['throughput']:.2f} logs/s ({r['speedup']:.1f}× speedup)")


if __name__ == "__main__":
    main()
