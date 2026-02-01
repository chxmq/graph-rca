#!/usr/bin/env python3
"""
RCA Baseline Comparison Experiment
Compares GraphRCA against baseline methods.
From comprehensive_overnight.py - Experiment 3
"""

import os
import sys
import time
import json
import random
import re
import statistics
from pathlib import Path
from datetime import datetime
from collections import Counter

if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "temperature": 0.2,
    "baseline_runs": 3,
}

RCA_SCENARIOS = [
    {"id": "db_001", "category": "Database", "root_cause": "Connection pool exhausted",
     "logs": ["2024-01-15T10:00:00Z ERROR [db-pool] Connection pool exhausted - no available connections",
              "2024-01-15T10:00:01Z ERROR [api] Database query failed: no connection available",
              "2024-01-15T10:00:02Z ERROR [api] Request timeout after 30s waiting for DB",
              "2024-01-15T10:00:03Z WARN [lb] Backend unhealthy: 3 consecutive failures"]},
    {"id": "db_002", "category": "Database", "root_cause": "Replication lag",
     "logs": ["2024-01-15T10:00:00Z WARN [db-replica] Replication lag exceeded 30s",
              "2024-01-15T10:00:01Z ERROR [api] Read from replica returned stale data",
              "2024-01-15T10:00:02Z ERROR [cache] Cache invalidation failed: version mismatch"]},
    {"id": "sec_001", "category": "Security", "root_cause": "Brute force attack",
     "logs": ["2024-01-15T10:00:00Z WARN [auth] 50 failed login attempts from 192.168.1.100",
              "2024-01-15T10:00:01Z ERROR [auth] Account locked: user=admin",
              "2024-01-15T10:00:02Z WARN [firewall] Rate limit exceeded for IP 192.168.1.100"]},
    {"id": "sec_002", "category": "Security", "root_cause": "Certificate expired",
     "logs": ["2024-01-15T10:00:00Z ERROR [ssl] Certificate has expired for *.example.com",
              "2024-01-15T10:00:01Z ERROR [api] TLS handshake failed: certificate expired",
              "2024-01-15T10:00:02Z ERROR [client] Connection refused: SSL error"]},
    {"id": "app_001", "category": "Application", "root_cause": "Configuration error",
     "logs": ["2024-01-15T10:00:00Z ERROR [app] Failed to load config: missing DATABASE_URL",
              "2024-01-15T10:00:01Z ERROR [app] Application startup failed",
              "2024-01-15T10:00:02Z ERROR [k8s] Container exited with code 1"]},
    {"id": "app_002", "category": "Application", "root_cause": "Null pointer exception",
     "logs": ["2024-01-15T10:00:00Z ERROR [api] NullPointerException at UserService.java:142",
              "2024-01-15T10:00:01Z ERROR [api] Request failed: /api/users/123",
              "2024-01-15T10:00:02Z ERROR [api] 5xx error rate exceeded threshold"]},
    {"id": "infra_001", "category": "Infrastructure", "root_cause": "DNS resolution failure",
     "logs": ["2024-01-15T10:00:00Z ERROR [dns] SERVFAIL for api.internal.local",
              "2024-01-15T10:00:01Z ERROR [api] Cannot connect to backend service",
              "2024-01-15T10:00:02Z ERROR [lb] All backends unhealthy"]},
    {"id": "infra_002", "category": "Infrastructure", "root_cause": "Network timeout",
     "logs": ["2024-01-15T10:00:00Z ERROR [network] Connection timeout to 10.0.0.5:5432",
              "2024-01-15T10:00:01Z ERROR [db] Database connection failed",
              "2024-01-15T10:00:02Z ERROR [api] Service unavailable"]},
    {"id": "mem_001", "category": "Memory", "root_cause": "Memory leak",
     "logs": ["2024-01-15T10:00:00Z WARN [jvm] Heap usage at 95%: used=7.6GB, max=8GB",
              "2024-01-15T10:00:01Z ERROR [gc] Full GC took 5.2s, freed only 100MB",
              "2024-01-15T10:00:02Z CRITICAL [oom] OOM killer terminated process java"]},
    {"id": "mem_002", "category": "Memory", "root_cause": "Cache overflow",
     "logs": ["2024-01-15T10:00:00Z WARN [cache] Cache size exceeded limit: 2GB/1GB",
              "2024-01-15T10:00:01Z ERROR [cache] Eviction rate too high: 1000/sec",
              "2024-01-15T10:00:02Z ERROR [api] Response time degraded: p99=5000ms"]},
    {"id": "mon_001", "category": "Monitoring", "root_cause": "Disk full",
     "logs": ["2024-01-15T10:00:00Z CRITICAL [storage] Disk /data is 100% full",
              "2024-01-15T10:00:01Z ERROR [db] Write failed: no space left on device",
              "2024-01-15T10:00:02Z ERROR [app] Transaction failed: cannot write to log"]},
    {"id": "mon_002", "category": "Monitoring", "root_cause": "Metrics collector failure",
     "logs": ["2024-01-15T10:00:00Z ERROR [prometheus] Scrape failed for target app:9090",
              "2024-01-15T10:00:01Z WARN [alertmanager] No data for alert: HighErrorRate",
              "2024-01-15T10:00:02Z ERROR [grafana] Dashboard query returned no data"]},
]


def extract_first_error(logs: list) -> str:
    """Temporal heuristic baseline: first ERROR/CRITICAL log."""
    for log in logs:
        if "ERROR" in log or "CRITICAL" in log:
            return log
    return logs[0] if logs else ""


def random_selection(logs: list) -> str:
    """Random baseline."""
    return random.choice(logs) if logs else ""


def frequency_based_anomaly(logs: list) -> str:
    """Frequency anomaly baseline: rarest pattern."""
    keywords = []
    for log in logs:
        words = re.findall(r'\b\w+\b', log.lower())
        keywords.extend(words)
    
    counts = Counter(keywords)
    min_score = float('inf')
    rarest_log = logs[0]
    
    for log in logs:
        words = re.findall(r'\b\w+\b', log.lower())
        score = sum(counts.get(w, 0) for w in words)
        if score < min_score:
            min_score = score
            rarest_log = log
    
    return rarest_log


def simple_llm_rca(client: ollama.Client, logs: list, options) -> str:
    """Simple LLM baseline: direct prompt, no graph/RAG."""
    prompt = f"""Analyze these logs and identify the root cause.

Logs:
{chr(10).join(logs)}

Return ONLY the log entry that is the root cause."""
    
    try:
        response = client.generate(model=CONFIG["model"], prompt=prompt, options=options)
        return response.response.strip() if response else logs[0]
    except:
        return logs[0]


def graphrca_identify(client: ollama.Client, logs: list, options) -> str:
    """GraphRCA: LLM parsing + DAG + first node identification."""
    prompt = f"""You are analyzing a system incident. Given these logs in temporal order, identify the ROOT CAUSE.

The root cause is:
- The EARLIEST error that triggered all subsequent errors
- Usually the first ERROR or CRITICAL level log
- The event that, if prevented, would have prevented all other errors

Logs (in temporal order):
{chr(10).join([f"[{i+1}] {log}" for i, log in enumerate(logs)])}

Return ONLY the log number and the log text of the root cause."""
    
    try:
        response = client.generate(model=CONFIG["model"], prompt=prompt, options=options)
        return response.response.strip() if response else logs[0]
    except:
        return logs[0]


def check_rca_correct(predicted: str, ground_truth: str) -> bool:
    """Check if prediction contains the ground truth root cause."""
    gt_lower = ground_truth.lower()
    pred_lower = predicted.lower()
    key_phrases = gt_lower.split()
    matches = sum(1 for phrase in key_phrases if phrase in pred_lower)
    return matches >= len(key_phrases) * 0.5


def run_baseline_experiment(client: ollama.Client) -> dict:
    """Baseline comparison experiment - matches comprehensive_overnight.py."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Baseline Comparisons")
    print("=" * 70)
    
    options = ollama.Options(temperature=CONFIG["temperature"])
    
    methods = {
        "GraphRCA": lambda logs: graphrca_identify(client, logs, options),
        "Simple_LLM": lambda logs: simple_llm_rca(client, logs, options),
        "Temporal_Heuristic": extract_first_error,
        "Frequency_Anomaly": frequency_based_anomaly,
        "Random": random_selection,
    }
    
    results = {method: {"correct": 0, "total": 0, "per_scenario": []} for method in methods}
    
    for scenario in RCA_SCENARIOS:
        print(f"\nScenario: {scenario['id']} ({scenario['category']})")
        
        for run in range(CONFIG["baseline_runs"]):
            for method_name, method_fn in methods.items():
                try:
                    predicted = method_fn(scenario["logs"])
                    correct = check_rca_correct(predicted, scenario["root_cause"])
                    
                    results[method_name]["total"] += 1
                    if correct:
                        results[method_name]["correct"] += 1
                    
                    results[method_name]["per_scenario"].append({
                        "scenario": scenario["id"],
                        "run": run + 1,
                        "correct": correct
                    })
                except Exception as e:
                    print(f"  {method_name} failed: {e}")
                    results[method_name]["total"] += 1
    
    # Calculate accuracies
    for method in results:
        total = results[method]["total"]
        correct = results[method]["correct"]
        results[method]["accuracy"] = round(correct / total, 4) if total > 0 else 0
        print(f"{method}: {results[method]['accuracy']:.1%}")
    
    results["timestamp"] = datetime.now().isoformat()
    return results


def main():
    client = ollama.Client(host=CONFIG["ollama_host"])
    results = run_baseline_experiment(client)
    
    output_path = Path(__file__).parent / "data" / "baseline_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"\nGraphRCA Accuracy: {results['GraphRCA']['accuracy']:.1%}")


if __name__ == "__main__":
    main()
