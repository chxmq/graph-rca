#!/usr/bin/env python3
"""
Documentation Ablation Experiment
Tests impact of documentation on RCA accuracy.
From comprehensive_overnight.py - Experiment 5
"""

import os
import sys
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
    "ablation_runs": 3,
}

RCA_SCENARIOS = [
    {"id": "db_001", "category": "Database", "root_cause": "Connection pool exhausted",
     "logs": ["ERROR [db-pool] Connection pool exhausted", "ERROR [api] Database query failed"]},
    {"id": "sec_001", "category": "Security", "root_cause": "Brute force attack",
     "logs": ["WARN [auth] 50 failed logins", "ERROR [auth] Account locked"]},
    {"id": "mem_001", "category": "Memory", "root_cause": "Memory leak",
     "logs": ["WARN [jvm] Heap usage 95%", "ERROR [gc] Full GC 5.2s", "CRITICAL [oom] OOM killer"]},
]

DOCUMENTATION = [
    """# Database Connection Pool Troubleshooting
    
    ## Symptoms
    - ERROR: Connection pool exhausted
    - Timeout waiting for connection
    
    ## Root Cause
    - Pool size too small for load
    - Connection leaks (connections not returned)
    
    ## Solution
    1. Check current pool size: SHOW max_connections;
    2. Increase pool size in config
    3. Add connection lifecycle monitoring
    4. Implement connection validation
    """,
    """# SSL/TLS Certificate Management
    
    ## Symptoms
    - Certificate expired errors
    - TLS handshake failures
    
    ## Root Cause
    - Certificate not renewed before expiry
    - Certificate chain incomplete
    
    ## Solution
    1. Check certificate expiry: openssl x509 -noout -dates
    2. Renew certificate with CA
    3. Update certificate in secrets
    4. Set up auto-renewal with certbot
    """,
    """# Memory Management and OOM Troubleshooting
    
    ## Symptoms
    - OOM killer invoked
    - High GC pause times
    - Heap usage above 90%
    
    ## Root Cause
    - Memory leak in application
    - Insufficient heap size
    - Too many cached objects
    
    ## Solution
    1. Analyze heap dump: jmap -dump:live,format=b,file=heap.hprof
    2. Identify leak with MAT or VisualVM
    3. Fix memory leak in code
    4. Increase heap size if appropriate
    """,
]


def check_rca_correct(predicted: str, ground_truth: str) -> bool:
    gt_lower = ground_truth.lower()
    pred_lower = predicted.lower()
    key_phrases = gt_lower.split()
    matches = sum(1 for phrase in key_phrases if phrase in pred_lower)
    return matches >= len(key_phrases) * 0.5


def run_doc_ablation(client: ollama.Client) -> dict:
    """Documentation ablation study - matches comprehensive_overnight.py."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Documentation Ablation")
    print("=" * 70)
    
    options = ollama.Options(temperature=CONFIG["temperature"])
    
    configs = {
        "full_docs": DOCUMENTATION,
        "half_docs": DOCUMENTATION[:len(DOCUMENTATION)//2 + 1],
        "no_docs": [],
    }
    
    results = {"configs": {}}
    
    for config_name, docs in configs.items():
        print(f"\nConfig: {config_name}")
        correct = 0
        total = 0
        
        for scenario in RCA_SCENARIOS:
            for run in range(CONFIG["ablation_runs"]):
                # Build prompt
                doc_context = "\n\n".join(docs) if docs else ""
                log_text = "\n".join(scenario["logs"])
                
                if doc_context:
                    prompt = f"""Documentation:
{doc_context}

Logs:
{log_text}

Based on the documentation, identify the root cause:"""
                else:
                    prompt = f"""Logs:
{log_text}

Identify the root cause:"""
                
                try:
                    response = client.generate(model=CONFIG["model"], prompt=prompt, options=options)
                    predicted = response.response.strip()
                    if check_rca_correct(predicted, scenario["root_cause"]):
                        correct += 1
                except Exception as e:
                    print(f"  Error: {e}")
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        results["configs"][config_name] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total
        }
        print(f"  {config_name}: {accuracy:.1%}")
    
    results["timestamp"] = datetime.now().isoformat()
    return results


def main():
    client = ollama.Client(host=CONFIG["ollama_host"])
    results = run_doc_ablation(client)
    
    output_path = Path(__file__).parent / "data" / "doc_ablation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    full = results["configs"]["full_docs"]["accuracy"]
    none = results["configs"]["no_docs"]["accuracy"]
    print(f"\nDifference: {(full - none) * 100:.1f} percentage points")


if __name__ == "__main__":
    main()
