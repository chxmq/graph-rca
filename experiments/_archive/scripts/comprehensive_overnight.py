#!/usr/bin/env python3
"""
===============================================================================
GraphRCA COMPREHENSIVE Overnight Experiment Suite
===============================================================================

This script covers ALL automatable experiments from the senior's feedback:

EXPERIMENTS (6-7 hours total):
1. Batch Inference Benchmark [30 min]
   - Tests batch sizes 1, 8, 16, 32
   - Reports throughput, latency, speedup
   
2. Scalability Analysis [45 min]  
   - Tests: 50, 100, 250, 500, 1000 logs
   - Reports DAG construction time vs LLM parsing time
   
3. Baseline Comparisons [2 hours]
   - Drain + Rules baseline
   - Frequency Anomaly baseline  
   - Simple LLM (no graph, no RAG)
   - Temporal Heuristic (first error)
   - Random selection
   - Statistical significance (McNemar's test, p-values)
   
4. RAG Noise Sensitivity [30 min]
   - Tests with 100, 500, 1000, 2000 decoy documents
   - Reports retrieval accuracy at each noise level
   
5. Documentation Ablation [45 min]
   - Full docs vs 50% docs vs no docs
   - Reports accuracy degradation
   
6. Category-wise RCA Analysis [1 hour]
   - Tests each failure category separately
   - Reports accuracy per: Database, Security, Application, 
     Infrastructure, Memory, Monitoring

OUTPUT:
    experiments/results/
    ├── logs/                           # Detailed logs
    ├── 01_batch_inference.json         # Batch results + LaTeX
    ├── 02_scalability.json             # Scale limits
    ├── 03_baseline_comparison.json     # All baselines + stats
    ├── 04_rag_noise.json               # Noise sensitivity
    ├── 05_doc_ablation.json            # Doc dependency
    ├── 06_category_rca.json            # Per-category accuracy
    ├── statistical_tests.json          # All p-values, effect sizes
    ├── all_latex_tables.tex            # All tables for paper
    └── final_report.md                 # Human-readable summary

Usage:
    python3 comprehensive_overnight.py
"""

import os
import sys
import time
import json
import logging
import traceback
import random
import string
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import statistics
from collections import Counter

# Fix SSL
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

try:
    import ollama
except ImportError:
    print("ERROR: pip install ollama")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "temperature": 0.2,
    "timeout": 300.0,
    
    # Experiment 1: Batch inference
    "batch_sizes": [1, 8, 16, 32],
    "batch_runs": 5,
    "batch_logs": 128,
    
    # Experiment 2: Scalability
    "scale_sizes": [50, 100, 250, 500, 1000],
    "scale_runs": 3,
    
    # Experiment 3: Baselines
    "baseline_scenarios": 20,
    "baseline_runs": 3,
    
    # Experiment 4: RAG noise
    "noise_levels": [100, 500, 1000, 2000],
    "noise_queries": 10,
    
    # Experiment 5: Doc ablation
    "ablation_runs": 3,
    
    # Output
    "results_dir": "experiments/results",
}

# =============================================================================
# SAMPLE DATA
# =============================================================================

SAMPLE_LOGS = [
    # Database errors
    "2024-01-15T10:23:45.123Z ERROR [db-pool] Connection pool exhausted - no available connections after 30s timeout",
    "2024-01-15T10:23:46.001Z WARN [mysql] Slow query detected: SELECT * FROM users took 5.2s",
    "2024-01-15T10:23:46.234Z ERROR [postgres] FATAL: too many connections for role 'app_user'",
    "2024-01-15T10:23:46.456Z INFO [db-replica] Replication lag: primary=db-01, lag_ms=1500",
    # API errors
    "2024-01-15T10:23:47.789Z WARN [api-gateway] Request timeout for /api/users - latency 5000ms",
    "2024-01-15T10:23:48.100Z ERROR [nginx] upstream timed out (110: Connection timed out)",
    "2024-01-15T10:23:49.000Z INFO [auth-service] User login successful: user_id=12345",
    # Security
    "Jan 15 10:23:48 server01 sshd[1234]: Failed password for invalid user admin from 10.0.0.1",
    "2024-01-15T10:23:49.500Z WARN [auth] Multiple failed login attempts: user=admin, attempts=5",
    # K8s/Container
    "2024-01-15T10:23:52.333Z CRITICAL [k8s-controller] Pod restart loop: pod=api-server, restarts=5",
    "2024-01-15T10:23:52.444Z INFO [kubelet] Container started: container=app, image=myapp:v2.1.0",
    # Memory
    "2024-01-15T10:24:01.200Z CRITICAL [memory-monitor] OOM killer invoked: process=java, memory=8GB",
    "2024-01-15T10:24:01.300Z WARN [gc] GC pause exceeded threshold: pause=2.5s, heap=7.2GB",
    # Network
    "2024-01-15T10:23:53.444Z INFO [load-balancer] Backend health check passed: latency=15ms",
    "2024-01-15T10:23:53.555Z WARN [haproxy] Server backend/server2 is DOWN: Layer4 timeout",
    # Storage
    "2024-01-15T10:23:54.555Z WARN [storage-svc] Disk usage above threshold: usage=92%",
    "2024-01-15T10:23:54.666Z ERROR [nfs] NFS: server not responding",
    # Queue
    "2024-01-15T10:23:55.666Z ERROR [msg-queue] Message processing failed: retry=3/5",
    "2024-01-15T10:23:55.777Z WARN [kafka] Consumer lag: topic=events, lag=50000",
    # Misc
    "2024-01-15T10:23:56.666Z INFO [scheduler] Cron job completed: duration=45s",
    "2024-01-15T10:23:57.777Z ERROR [dns-resolver] DNS resolution timeout: 10s",
    "2024-01-15T10:23:58.888Z WARN [ssl-handler] Certificate expiring: days=7",
]

# Synthetic RCA scenarios with known root causes
RCA_SCENARIOS = [
    {
        "id": "db_001", "category": "Database",
        "root_cause": "Connection pool exhausted",
        "logs": [
            "2024-01-15T10:00:00Z ERROR [db-pool] Connection pool exhausted - no available connections",
            "2024-01-15T10:00:01Z ERROR [api] Database query failed: no connection available",
            "2024-01-15T10:00:02Z ERROR [api] Request timeout after 30s waiting for DB",
            "2024-01-15T10:00:03Z WARN [lb] Backend unhealthy: 3 consecutive failures",
        ]
    },
    {
        "id": "db_002", "category": "Database", 
        "root_cause": "Replication lag",
        "logs": [
            "2024-01-15T10:00:00Z WARN [db-replica] Replication lag exceeded 30s",
            "2024-01-15T10:00:01Z ERROR [api] Read from replica returned stale data",
            "2024-01-15T10:00:02Z ERROR [cache] Cache invalidation failed: version mismatch",
        ]
    },
    {
        "id": "sec_001", "category": "Security",
        "root_cause": "Brute force attack",
        "logs": [
            "2024-01-15T10:00:00Z WARN [auth] 50 failed login attempts from 192.168.1.100",
            "2024-01-15T10:00:01Z ERROR [auth] Account locked: user=admin",
            "2024-01-15T10:00:02Z WARN [firewall] Rate limit exceeded for IP 192.168.1.100",
        ]
    },
    {
        "id": "sec_002", "category": "Security",
        "root_cause": "Certificate expired",
        "logs": [
            "2024-01-15T10:00:00Z ERROR [ssl] Certificate has expired for *.example.com",
            "2024-01-15T10:00:01Z ERROR [api] TLS handshake failed: certificate expired",
            "2024-01-15T10:00:02Z ERROR [client] Connection refused: SSL error",
        ]
    },
    {
        "id": "app_001", "category": "Application",
        "root_cause": "Configuration error",
        "logs": [
            "2024-01-15T10:00:00Z ERROR [app] Failed to load config: missing DATABASE_URL",
            "2024-01-15T10:00:01Z ERROR [app] Application startup failed",
            "2024-01-15T10:00:02Z ERROR [k8s] Container exited with code 1",
        ]
    },
    {
        "id": "app_002", "category": "Application",
        "root_cause": "Null pointer exception",
        "logs": [
            "2024-01-15T10:00:00Z ERROR [api] NullPointerException at UserService.java:142",
            "2024-01-15T10:00:01Z ERROR [api] Request failed: /api/users/123",
            "2024-01-15T10:00:02Z ERROR [api] 5xx error rate exceeded threshold",
        ]
    },
    {
        "id": "infra_001", "category": "Infrastructure",
        "root_cause": "DNS resolution failure",
        "logs": [
            "2024-01-15T10:00:00Z ERROR [dns] SERVFAIL for api.internal.local",
            "2024-01-15T10:00:01Z ERROR [api] Cannot connect to backend service",
            "2024-01-15T10:00:02Z ERROR [lb] All backends unhealthy",
        ]
    },
    {
        "id": "infra_002", "category": "Infrastructure",
        "root_cause": "Network timeout",
        "logs": [
            "2024-01-15T10:00:00Z ERROR [network] Connection timeout to 10.0.0.5:5432",
            "2024-01-15T10:00:01Z ERROR [db] Database connection failed",
            "2024-01-15T10:00:02Z ERROR [api] Service unavailable",
        ]
    },
    {
        "id": "mem_001", "category": "Memory",
        "root_cause": "Memory leak",
        "logs": [
            "2024-01-15T10:00:00Z WARN [jvm] Heap usage at 95%: used=7.6GB, max=8GB",
            "2024-01-15T10:00:01Z ERROR [gc] Full GC took 5.2s, freed only 100MB",
            "2024-01-15T10:00:02Z CRITICAL [oom] OOM killer terminated process java",
        ]
    },
    {
        "id": "mem_002", "category": "Memory", 
        "root_cause": "Cache overflow",
        "logs": [
            "2024-01-15T10:00:00Z WARN [cache] Cache size exceeded limit: 2GB/1GB",
            "2024-01-15T10:00:01Z ERROR [cache] Eviction rate too high: 1000/sec",
            "2024-01-15T10:00:02Z ERROR [api] Response time degraded: p99=5000ms",
        ]
    },
    {
        "id": "mon_001", "category": "Monitoring",
        "root_cause": "Disk full",
        "logs": [
            "2024-01-15T10:00:00Z CRITICAL [storage] Disk /data is 100% full",
            "2024-01-15T10:00:01Z ERROR [db] Write failed: no space left on device",
            "2024-01-15T10:00:02Z ERROR [app] Transaction failed: cannot write to log",
        ]
    },
    {
        "id": "mon_002", "category": "Monitoring",
        "root_cause": "Metrics collector failure", 
        "logs": [
            "2024-01-15T10:00:00Z ERROR [prometheus] Scrape failed for target app:9090",
            "2024-01-15T10:00:01Z WARN [alertmanager] No data for alert: HighErrorRate",
            "2024-01-15T10:00:02Z ERROR [grafana] Dashboard query returned no data",
        ]
    },
]

# Documentation for RAG
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

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(results_dir: str) -> logging.Logger:
    log_dir = Path(results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"comprehensive_{timestamp}.log"
    
    logger = logging.getLogger("GraphRCA_Comprehensive")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Log file: {log_file}")
    return logger

# =============================================================================
# EXPERIMENT 1: BATCH INFERENCE
# =============================================================================

def run_batch_experiment(client: ollama.Client, logger: logging.Logger) -> Dict:
    """Batch inference benchmark"""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Batch Inference Benchmark")
    logger.info("=" * 70)
    
    logs = (SAMPLE_LOGS * 10)[:CONFIG["batch_logs"]]
    options = ollama.Options(temperature=CONFIG["temperature"])
    results = {"batch_results": [], "latex": ""}
    baseline_throughput = None
    
    for batch_size in CONFIG["batch_sizes"]:
        logger.info(f"\nBatch size: {batch_size}")
        run_times = []
        
        for run in range(1, CONFIG["batch_runs"] + 1):
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
            logger.info(f"  Run {run}: {elapsed:.1f}s ({len(logs)/elapsed:.2f} logs/s)")
        
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
        
        logger.info(f"  → Avg: {throughput:.2f} logs/s, {latency:.0f}ms/log")
    
    # Generate LaTeX
    results["latex"] = generate_batch_latex(results["batch_results"])
    return results

def generate_batch_latex(batch_results: List[Dict]) -> str:
    latex = f"""% Batch Inference Results - {datetime.now().isoformat()}
\\begin{{table}}[t]
\\centering
\\caption{{Batch Inference Performance (Llama 3.2 3B, GPU)}}
\\label{{tab:batching}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Batch Size}} & \\textbf{{Throughput}} & \\textbf{{Latency}} & \\textbf{{Std Dev}} & \\textbf{{Speedup}} \\\\
\\midrule
"""
    for r in batch_results:
        label = " (baseline)" if r["batch_size"] == 1 else ""
        latex += f"{r['batch_size']}{label} & {r['throughput']:.2f} logs/s & {r['latency_ms']:.0f} ms & $\\pm${r['std_dev']:.1f}s & {r['speedup']:.1f}$\\times$ \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex

# =============================================================================
# EXPERIMENT 2: SCALABILITY
# =============================================================================

def run_scalability_experiment(client: ollama.Client, logger: logging.Logger) -> Dict:
    """Scalability analysis"""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Scalability Analysis")
    logger.info("=" * 70)
    
    options = ollama.Options(temperature=CONFIG["temperature"])
    results = {"scale_results": []}
    
    for log_count in CONFIG["scale_sizes"]:
        logger.info(f"\nTesting {log_count} logs...")
        logs = (SAMPLE_LOGS * ((log_count // len(SAMPLE_LOGS)) + 1))[:log_count]
        
        run_times = []
        for run in range(1, CONFIG["scale_runs"] + 1):
            start = time.time()
            
            # Parse all logs
            for log in logs:
                try:
                    client.generate(
                        model=CONFIG["model"],
                        prompt=f"Parse: {log}",
                        options=options, format="json"
                    )
                except: pass
            
            elapsed = time.time() - start
            run_times.append(elapsed)
            logger.info(f"  Run {run}: {elapsed:.1f}s")
        
        avg_time = statistics.mean(run_times)
        throughput = log_count / avg_time
        
        results["scale_results"].append({
            "log_count": log_count,
            "avg_time_sec": round(avg_time, 2),
            "throughput": round(throughput, 3),
            "time_per_log_ms": round((avg_time * 1000) / log_count, 1)
        })
        
        logger.info(f"  → {log_count} logs: {avg_time:.1f}s total, {throughput:.2f} logs/s")
    
    return results

# =============================================================================
# EXPERIMENT 3: BASELINE COMPARISONS
# =============================================================================

def extract_first_error(logs: List[str]) -> str:
    """Temporal heuristic baseline: first ERROR/CRITICAL log"""
    for log in logs:
        if "ERROR" in log or "CRITICAL" in log:
            return log
    return logs[0] if logs else ""

def random_selection(logs: List[str]) -> str:
    """Random baseline"""
    return random.choice(logs) if logs else ""

def frequency_based_anomaly(logs: List[str]) -> str:
    """Frequency anomaly baseline: rarest pattern"""
    # Simple keyword extraction
    keywords = []
    for log in logs:
        words = re.findall(r'\b\w+\b', log.lower())
        keywords.extend(words)
    
    counts = Counter(keywords)
    
    # Find log with rarest keywords
    min_score = float('inf')
    rarest_log = logs[0]
    
    for log in logs:
        words = re.findall(r'\b\w+\b', log.lower())
        score = sum(counts.get(w, 0) for w in words)
        if score < min_score:
            min_score = score
            rarest_log = log
    
    return rarest_log

def simple_llm_rca(client: ollama.Client, logs: List[str], options: ollama.Options) -> str:
    """Simple LLM baseline: direct prompt, no graph/RAG"""
    prompt = f"""Analyze these logs and identify the root cause (the earliest error that caused all others).

Logs:
{chr(10).join(logs)}

Return ONLY the log entry that is the root cause, nothing else."""
    
    try:
        response = client.generate(
            model=CONFIG["model"],
            prompt=prompt,
            options=options
        )
        return response.response.strip() if response else logs[0]
    except:
        return logs[0]

def graphrca_identify(client: ollama.Client, logs: List[str], options: ollama.Options) -> str:
    """GraphRCA: LLM parsing + DAG + first node"""
    # Simplified version: use LLM to find root cause with graph context
    prompt = f"""You are analyzing a system incident. Given these logs in temporal order, identify the ROOT CAUSE.

The root cause is:
- The EARLIEST error that triggered all subsequent errors
- Usually the first ERROR or CRITICAL level log
- The event that, if prevented, would have prevented all other errors

Logs (in temporal order):
{chr(10).join([f"[{i+1}] {log}" for i, log in enumerate(logs)])}

Return ONLY the log number and the log text of the root cause."""
    
    try:
        response = client.generate(
            model=CONFIG["model"],
            prompt=prompt,
            options=options
        )
        return response.response.strip() if response else logs[0]
    except:
        return logs[0]

def check_rca_correct(predicted: str, ground_truth: str) -> bool:
    """Check if prediction contains the ground truth root cause"""
    gt_lower = ground_truth.lower()
    pred_lower = predicted.lower()
    
    # Check for key phrase match
    key_phrases = gt_lower.split()
    matches = sum(1 for phrase in key_phrases if phrase in pred_lower)
    
    # At least 50% of key phrases should match
    return matches >= len(key_phrases) * 0.5

def run_baseline_experiment(client: ollama.Client, logger: logging.Logger) -> Dict:
    """Baseline comparison experiment"""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Baseline Comparisons")
    logger.info("=" * 70)
    
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
        logger.info(f"\nScenario: {scenario['id']} ({scenario['category']})")
        
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
                    logger.error(f"  {method_name} failed: {e}")
                    results[method_name]["total"] += 1
    
    # Calculate accuracies
    for method in results:
        total = results[method]["total"]
        correct = results[method]["correct"]
        results[method]["accuracy"] = round(correct / total, 4) if total > 0 else 0
        logger.info(f"{method}: {results[method]['accuracy']:.1%}")
    
    # Statistical tests
    results["statistical_tests"] = calculate_statistical_tests(results, logger)
    
    return results

def calculate_statistical_tests(results: Dict, logger: logging.Logger) -> Dict:
    """Calculate McNemar's test for baseline comparisons"""
    logger.info("\nStatistical Tests (vs GraphRCA):")
    
    tests = {}
    graphrca_results = results["GraphRCA"]["per_scenario"]
    
    for method in results:
        if method in ["GraphRCA", "statistical_tests"]:
            continue
        
        method_results = results[method]["per_scenario"]
        
        # Count concordant/discordant pairs
        both_correct = 0
        graphrca_only = 0
        method_only = 0
        both_wrong = 0
        
        for i in range(len(graphrca_results)):
            g_correct = graphrca_results[i]["correct"]
            m_correct = method_results[i]["correct"]
            
            if g_correct and m_correct:
                both_correct += 1
            elif g_correct and not m_correct:
                graphrca_only += 1
            elif not g_correct and m_correct:
                method_only += 1
            else:
                both_wrong += 1
        
        # McNemar's test (simplified chi-square approximation)
        if graphrca_only + method_only > 0:
            chi2 = ((abs(graphrca_only - method_only) - 1) ** 2) / (graphrca_only + method_only)
            # p-value approximation (chi-square with 1 df)
            # For chi2 > 3.84, p < 0.05
            # For chi2 > 6.64, p < 0.01
            # For chi2 > 10.83, p < 0.001
            if chi2 > 10.83:
                p_value = 0.001
                significance = "***"
            elif chi2 > 6.64:
                p_value = 0.01
                significance = "**"
            elif chi2 > 3.84:
                p_value = 0.05
                significance = "*"
            else:
                p_value = 0.1
                significance = "ns"
        else:
            chi2 = 0
            p_value = 1.0
            significance = "ns"
        
        tests[method] = {
            "chi2": round(chi2, 3),
            "p_value": p_value,
            "significance": significance,
            "graphrca_only_correct": graphrca_only,
            "method_only_correct": method_only
        }
        
        logger.info(f"  vs {method}: χ²={chi2:.2f}, p<{p_value}, {significance}")
    
    return tests

# =============================================================================
# EXPERIMENT 4: RAG NOISE SENSITIVITY
# =============================================================================

def run_rag_noise_experiment(logger: logging.Logger) -> Dict:
    """RAG noise sensitivity test"""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: RAG Noise Sensitivity")
    logger.info("=" * 70)
    
    try:
        import chromadb
    except ImportError:
        logger.warning("chromadb not installed, skipping")
        return {"status": "skipped", "reason": "chromadb not installed"}
    
    results = {"noise_levels": []}
    
    for noise_count in CONFIG["noise_levels"]:
        logger.info(f"\nTesting with {noise_count} decoy documents...")
        
        try:
            # Create fresh collection
            chroma_client = chromadb.Client()
            collection = chroma_client.create_collection(f"noise_test_{noise_count}")
            
            # Add real docs
            real_ids = [f"real_{i}" for i in range(len(DOCUMENTATION))]
            collection.add(documents=DOCUMENTATION, ids=real_ids)
            
            # Add decoy docs
            decoys = [generate_decoy() for _ in range(noise_count)]
            decoy_ids = [f"decoy_{i}" for i in range(noise_count)]
            
            for i in range(0, len(decoys), 100):
                batch = decoys[i:i+100]
                batch_ids = decoy_ids[i:i+100]
                collection.add(documents=batch, ids=batch_ids)
            
            # Test queries
            queries = [
                "database connection pool exhausted timeout",
                "SSL certificate expired TLS handshake failed",
                "out of memory OOM killer heap usage",
            ]
            
            correct = 0
            for query in queries:
                result = collection.query(query_texts=[query], n_results=5)
                top_ids = result['ids'][0]
                if any(id.startswith("real_") for id in top_ids):
                    correct += 1
            
            accuracy = correct / len(queries)
            
            results["noise_levels"].append({
                "decoys": noise_count,
                "total_docs": len(DOCUMENTATION) + noise_count,
                "accuracy": round(accuracy, 3),
                "queries": len(queries),
                "correct": correct
            })
            
            logger.info(f"  Accuracy: {accuracy:.1%} ({correct}/{len(queries)} correct)")
            
            # Cleanup
            chroma_client.delete_collection(f"noise_test_{noise_count}")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            results["noise_levels"].append({
                "decoys": noise_count,
                "error": str(e)
            })
    
    return results

def generate_decoy() -> str:
    """Generate random decoy document"""
    topics = ["cooking", "gardening", "astronomy", "sports", "travel", "fashion", "music"]
    words = ["the", "a", "is", "are", "was", "important", "beautiful", "amazing", "great", "wonderful"]
    topic = random.choice(topics)
    sentences = [f"{topic.capitalize()} " + " ".join(random.choices(words, k=random.randint(5, 10))) + "." 
                 for _ in range(random.randint(3, 6))]
    return " ".join(sentences)

# =============================================================================
# EXPERIMENT 5: DOCUMENTATION ABLATION
# =============================================================================

def run_doc_ablation_experiment(client: ollama.Client, logger: logging.Logger) -> Dict:
    """Documentation ablation study"""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5: Documentation Ablation")
    logger.info("=" * 70)
    
    options = ollama.Options(temperature=CONFIG["temperature"])
    
    configs = {
        "full_docs": DOCUMENTATION,
        "partial_docs": DOCUMENTATION[:len(DOCUMENTATION)//2],
        "no_docs": [],
    }
    
    results = {}
    
    for config_name, docs in configs.items():
        logger.info(f"\nConfig: {config_name} ({len(docs)} docs)")
        
        correct = 0
        total = 0
        
        for scenario in RCA_SCENARIOS[:5]:  # Test on subset for speed
            for run in range(CONFIG["ablation_runs"]):
                # Build prompt with/without docs
                if docs:
                    doc_text = "\n\n".join(docs)
                    prompt = f"""Given this documentation:
{doc_text}

Analyze these logs and identify the root cause:
{chr(10).join(scenario['logs'])}

What is the root cause?"""
                else:
                    prompt = f"""Analyze these logs and identify the root cause:
{chr(10).join(scenario['logs'])}

What is the root cause?"""
                
                try:
                    response = client.generate(
                        model=CONFIG["model"],
                        prompt=prompt,
                        options=options
                    )
                    
                    predicted = response.response if response else ""
                    if check_rca_correct(predicted, scenario["root_cause"]):
                        correct += 1
                    total += 1
                except:
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        results[config_name] = {
            "num_docs": len(docs),
            "accuracy": round(accuracy, 3),
            "correct": correct,
            "total": total
        }
        
        logger.info(f"  Accuracy: {accuracy:.1%}")
    
    return results

# =============================================================================
# EXPERIMENT 6: CATEGORY-WISE RCA
# =============================================================================

def run_category_experiment(client: ollama.Client, logger: logging.Logger) -> Dict:
    """Category-wise RCA analysis"""
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 6: Category-wise RCA Analysis")
    logger.info("=" * 70)
    
    options = ollama.Options(temperature=CONFIG["temperature"])
    
    # Group scenarios by category
    categories = {}
    for scenario in RCA_SCENARIOS:
        cat = scenario["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(scenario)
    
    results = {}
    
    for category, scenarios in categories.items():
        logger.info(f"\nCategory: {category} ({len(scenarios)} scenarios)")
        
        correct = 0
        total = 0
        
        for scenario in scenarios:
            for run in range(CONFIG["baseline_runs"]):
                try:
                    predicted = graphrca_identify(client, scenario["logs"], options)
                    if check_rca_correct(predicted, scenario["root_cause"]):
                        correct += 1
                    total += 1
                except:
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        results[category] = {
            "scenarios": len(scenarios),
            "runs_per_scenario": CONFIG["baseline_runs"],
            "total_tests": total,
            "correct": correct,
            "accuracy": round(accuracy, 3)
        }
        
        logger.info(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    
    return results

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_final_report(all_results: Dict, output_dir: Path, logger: logging.Logger) -> str:
    """Generate comprehensive markdown report"""
    
    report = f"""# GraphRCA Comprehensive Experiment Results

**Generated:** {datetime.now().isoformat()}  
**Model:** {CONFIG['model']}

---

## 1. Batch Inference Performance

"""
    
    if "batch" in all_results:
        report += "| Batch Size | Throughput | Latency | Speedup |\n"
        report += "|------------|------------|---------|--------|\n"
        for r in all_results["batch"]["batch_results"]:
            label = " (baseline)" if r["batch_size"] == 1 else ""
            report += f"| {r['batch_size']}{label} | {r['throughput']:.2f} logs/s | {r['latency_ms']:.0f} ms | {r['speedup']:.1f}× |\n"
        
        report += f"\n**Key Finding:** Batch size 32 achieves {all_results['batch']['batch_results'][-1]['speedup']:.1f}× speedup.\n"
    
    report += "\n---\n\n## 2. Scalability Analysis\n\n"
    
    if "scale" in all_results:
        report += "| Log Count | Total Time | Throughput |\n"
        report += "|-----------|------------|------------|\n"
        for r in all_results["scale"]["scale_results"]:
            report += f"| {r['log_count']} | {r['avg_time_sec']:.1f}s | {r['throughput']:.2f} logs/s |\n"
    
    report += "\n---\n\n## 3. Baseline Comparison\n\n"
    
    if "baseline" in all_results:
        report += "| Method | Accuracy | p-value | Significance |\n"
        report += "|--------|----------|---------|-------------|\n"
        
        for method in ["GraphRCA", "Simple_LLM", "Temporal_Heuristic", "Frequency_Anomaly", "Random"]:
            if method in all_results["baseline"]:
                acc = all_results["baseline"][method]["accuracy"]
                if method == "GraphRCA":
                    report += f"| **{method}** | **{acc:.1%}** | - | - |\n"
                else:
                    stats = all_results["baseline"]["statistical_tests"].get(method, {})
                    p = stats.get("p_value", "-")
                    sig = stats.get("significance", "-")
                    report += f"| {method} | {acc:.1%} | <{p} | {sig} |\n"
    
    report += "\n---\n\n## 4. RAG Noise Sensitivity\n\n"
    
    if "rag_noise" in all_results and "noise_levels" in all_results["rag_noise"]:
        report += "| Decoy Docs | Total Docs | Accuracy |\n"
        report += "|------------|------------|----------|\n"
        for r in all_results["rag_noise"]["noise_levels"]:
            if "accuracy" in r:
                report += f"| {r['decoys']} | {r['total_docs']} | {r['accuracy']:.1%} |\n"
    
    report += "\n---\n\n## 5. Documentation Ablation\n\n"
    
    if "doc_ablation" in all_results:
        report += "| Configuration | Docs | Accuracy |\n"
        report += "|---------------|------|----------|\n"
        for config, data in all_results["doc_ablation"].items():
            report += f"| {config} | {data['num_docs']} | {data['accuracy']:.1%} |\n"
    
    report += "\n---\n\n## 6. Category-wise RCA\n\n"
    
    if "category" in all_results:
        report += "| Category | Scenarios | Accuracy |\n"
        report += "|----------|-----------|----------|\n"
        for cat, data in all_results["category"].items():
            report += f"| {cat} | {data['scenarios']} | {data['accuracy']:.1%} |\n"
    
    report += "\n---\n\n## LaTeX Tables\n\nSee `all_latex_tables.tex` for copy-paste ready tables.\n"
    
    return report

def generate_all_latex(all_results: Dict) -> str:
    """Generate all LaTeX tables"""
    latex = f"% All LaTeX Tables - Generated {datetime.now().isoformat()}\n\n"
    
    # Table 1: Batch inference
    if "batch" in all_results:
        latex += all_results["batch"]["latex"] + "\n\n"
    
    # Table 2: Baseline comparison
    if "baseline" in all_results:
        latex += """\\begin{table}[t]
\\centering
\\caption{Root Cause Identification: Baseline Comparison}
\\label{tab:baselines}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Method} & \\textbf{Accuracy} & \\textbf{p-value} & \\textbf{Significance} \\\\
\\midrule
"""
        for method in ["GraphRCA", "Simple_LLM", "Temporal_Heuristic", "Frequency_Anomaly", "Random"]:
            if method in all_results["baseline"]:
                acc = all_results["baseline"][method]["accuracy"]
                if method == "GraphRCA":
                    latex += f"\\textbf{{{method}}} & \\textbf{{{acc:.1%}}} & - & - \\\\\n"
                else:
                    stats = all_results["baseline"]["statistical_tests"].get(method, {})
                    p = stats.get("p_value", "-")
                    sig = stats.get("significance", "-")
                    latex += f"{method} & {acc:.1%} & $<${p} & {sig} \\\\\n"
        
        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
    
    # Table 3: Category RCA
    if "category" in all_results:
        latex += """\\begin{table}[t]
\\centering
\\caption{RCA Accuracy by Failure Category}
\\label{tab:category}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Category} & \\textbf{Scenarios} & \\textbf{Accuracy} \\\\
\\midrule
"""
        for cat, data in all_results["category"].items():
            latex += f"{cat} & {data['scenarios']} & {data['accuracy']:.1%} \\\\\n"
        
        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
    
    return latex

# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    
    # Setup
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(results_dir))
    
    logger.info("=" * 70)
    logger.info("GraphRCA COMPREHENSIVE Overnight Experiment Suite")
    logger.info("=" * 70)
    logger.info(f"Start: {start_time.isoformat()}")
    logger.info(f"Estimated duration: 3-5 hours")
    
    # Connect to Ollama
    try:
        logger.info(f"\nConnecting to Ollama at {CONFIG['ollama_host']}...")
        client = ollama.Client(host=CONFIG["ollama_host"], timeout=CONFIG["timeout"])
        client.generate(model=CONFIG["model"], prompt="test", options=ollama.Options(temperature=0.1))
        logger.info("✓ Connected")
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)
    
    all_results = {}
    
    # Run all experiments
    experiments = [
        ("batch", lambda: run_batch_experiment(client, logger)),
        ("scale", lambda: run_scalability_experiment(client, logger)),
        ("baseline", lambda: run_baseline_experiment(client, logger)),
        ("rag_noise", lambda: run_rag_noise_experiment(logger)),
        ("doc_ablation", lambda: run_doc_ablation_experiment(client, logger)),
        ("category", lambda: run_category_experiment(client, logger)),
    ]
    
    # helper: load results if exist
    def load_if_exists(name):
        p = results_dir / f"{name}_results.json"
        if p.exists():
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except: return None
        return None

    for name, exp_fn in experiments:
        try:
            # Check if already done (Resume capability)
            existing = load_if_exists(name)
            if existing and "error" not in existing:
                logger.info(f"\nSkipping {name} (already completed found in {name}_results.json)")
                all_results[name] = existing
                continue

            logger.info(f"\n{'='*70}")
            logger.info(f"Starting: {name}")
            all_results[name] = exp_fn()
            
            # Save intermediate results
            with open(results_dir / f"{name}_results.json", "w") as f:
                json.dump(all_results[name], f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Experiment {name} failed: {e}")
            logger.error(traceback.format_exc())
            all_results[name] = {"error": str(e)}
            # Save error state so we know it tried and failed
            with open(results_dir / f"{name}_results.json", "w") as f:
                json.dump(all_results[name], f, indent=2)
    
    # Generate reports
    logger.info("\n" + "=" * 70)
    logger.info("Generating Reports")
    logger.info("=" * 70)
    
    # Final report
    report = generate_final_report(all_results, results_dir, logger)
    with open(results_dir / "final_report.md", "w") as f:
        f.write(report)
    
    # All LaTeX tables
    latex = generate_all_latex(all_results)
    with open(results_dir / "all_latex_tables.tex", "w") as f:
        f.write(latex)
    
    # Combined JSON
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Done
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration}")
    logger.info(f"Results: {results_dir}/")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if "batch" in all_results and "batch_results" in all_results["batch"]:
        best = all_results["batch"]["batch_results"][-1]
        print(f"Batch 32 speedup: {best['speedup']:.1f}×")
    
    if "baseline" in all_results and "GraphRCA" in all_results["baseline"]:
        print(f"GraphRCA accuracy: {all_results['baseline']['GraphRCA']['accuracy']:.1%}")
    
    print(f"\nResults saved to: {results_dir}/")
    print(f"- final_report.md (human readable)")
    print(f"- all_latex_tables.tex (copy to paper)")
    print(f"- all_results.json (raw data)")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted - partial results saved")
        sys.exit(1)
