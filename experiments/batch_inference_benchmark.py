#!/usr/bin/env python3
"""
Batch Inference Benchmark for GraphRCA LLM Parser
=================================================

This script benchmarks LLM log parsing throughput at different batch sizes.
Results will be used to update Section IV-A of the IEEE paper.

Usage:
    python3 batch_inference_benchmark.py

Requirements:
    - Ollama service running with llama3.2:3b model
    - Python packages: ollama, pandas
"""

import os
import time
import json
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

# Fix for SSL_CERT_FILE environment variable issue
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

try:
    import ollama
except ImportError:
    print("Error: ollama package not installed. Run: pip install ollama")
    exit(1)

# Configuration
MODEL = "llama3.2:3b"
OLLAMA_HOST = "http://localhost:11434"  # Standard Ollama port (change to 11435 for Docker)
TEMPERATURE = 0.2
TIMEOUT = 300.0  # Longer timeout for batch processing

# Batch sizes to test
BATCH_SIZES = [1, 8, 16, 32]
RUNS_PER_SIZE = 5  # 5 runs for statistical significance (paper-quality)

# Total logs to process per test (larger = more robust results)
TOTAL_LOGS = 128  # Use 128 logs for solid paper results

# Sample log entries for benchmarking (diverse formats from real systems)
SAMPLE_LOGS = [
    # Database logs
    "2024-01-15T10:23:45.123Z ERROR [db-pool] Connection pool exhausted - no available connections after 30s timeout",
    "2024-01-15T10:23:46.001Z WARN [mysql] Slow query detected: SELECT * FROM users WHERE status='active' took 5.2s",
    "2024-01-15T10:23:46.234Z ERROR [postgres] FATAL: too many connections for role 'app_user'",
    "2024-01-15T10:23:46.456Z INFO [db-replica] Replication lag: primary=db-01, replica=db-02, lag_ms=1500",
    
    # API Gateway logs
    "2024-01-15T10:23:47.789Z WARN [api-gateway] Request timeout for /api/users endpoint - upstream latency exceeded 5000ms",
    "2024-01-15T10:23:48.100Z ERROR [nginx] upstream timed out (110: Connection timed out) while reading response header from upstream",
    "2024-01-15T10:23:48.200Z INFO [kong] rate limit exceeded for client 192.168.1.100, limit=1000/hour",
    
    # Authentication logs  
    "2024-01-15T10:23:49.000Z INFO [auth-service] User login successful: user_id=12345, ip=192.168.1.100, method=oauth2",
    "Jan 15 10:23:48 server01 sshd[1234]: Failed password for invalid user admin from 10.0.0.1 port 22 ssh2",
    "2024-01-15T10:23:49.500Z WARN [auth] Multiple failed login attempts: user=admin, attempts=5, ip=203.0.113.50",
    "2024-01-15T10:23:49.600Z ERROR [jwt] Token validation failed: expired at 2024-01-15T09:00:00Z",
    
    # Payment/Transaction logs
    "2024-01-15T10:23:50.111Z ERROR [payment-svc] Transaction failed: tx_id=TXN-789, error_code=INSUFFICIENT_FUNDS, amount=1500.00",
    "2024-01-15T10:23:50.222Z INFO [stripe] Payment processed: charge_id=ch_abc123, amount=$99.99, status=succeeded",
    "2024-01-15T10:23:50.333Z WARN [payment] Retry attempt 3/5 for transaction TXN-790, reason=gateway_timeout",
    
    # Cache/Memory logs
    "2024-01-15T10:23:51.222Z DEBUG [cache-layer] Cache miss for key: user_profile_12345, fetching from database",
    "2024-01-15T10:23:51.333Z INFO [redis] BGSAVE completed successfully, 1.2GB saved to disk in 3.5s",
    "2024-01-15T10:23:51.444Z WARN [memcached] Eviction rate high: 1500 items/sec, memory_usage=95%",
    
    # Kubernetes/Container logs
    "2024-01-15T10:23:52.333Z CRITICAL [k8s-controller] Pod restart loop detected: pod=api-server-abc123, restarts=5",
    "2024-01-15T10:23:52.444Z INFO [kubelet] Container started: container=app, image=myapp:v2.1.0, pod=web-server-xyz",
    "2024-01-15T10:23:52.555Z WARN [k8s] Node memory pressure: node=worker-03, available=512MB, threshold=1GB",
    "2024-01-15T10:23:52.666Z ERROR [docker] OCI runtime exec failed: exec failed: unable to start container process",
    
    # Load Balancer/Network logs
    "2024-01-15T10:23:53.444Z INFO [load-balancer] Backend server health check passed: server=backend-01, latency=15ms",
    "2024-01-15T10:23:53.555Z WARN [haproxy] Server backend/server2 is DOWN, reason: Layer4 timeout",
    "2024-01-15T10:23:53.666Z ERROR [envoy] upstream connect error or disconnect/reset before headers, reset reason: connection timeout",
    
    # Storage logs
    "2024-01-15T10:23:54.555Z WARN [storage-svc] Disk usage above threshold: mount=/data, usage=92%, total=500GB",
    "2024-01-15T10:23:54.666Z ERROR [nfs] NFS: server not responding, still trying",
    "2024-01-15T10:23:54.777Z INFO [s3] Object uploaded: bucket=logs-archive, key=2024/01/15/app.log.gz, size=45MB",
    
    # Message Queue logs
    "2024-01-15T10:23:55.666Z ERROR [msg-queue] Message processing failed: queue=orders, msg_id=MSG-456, retry=3/5",
    "2024-01-15T10:23:55.777Z WARN [kafka] Consumer lag detected: topic=events, partition=3, lag=50000 messages",
    "2024-01-15T10:23:55.888Z INFO [rabbitmq] Queue declared: queue=notifications, durable=true, messages=0",
    
    # Scheduler/Cron logs
    "2024-01-15T10:23:56.666Z INFO [scheduler] Cron job completed: job=daily-backup, duration=45s, status=success",
    "2024-01-15T10:23:56.777Z ERROR [celery] Task failed: task=send_email, exception=SMTPServerDisconnected",
    
    # DNS/Network Resolution logs
    "2024-01-15T10:23:57.777Z ERROR [dns-resolver] DNS resolution timeout for api.external-service.com after 10s",
    "2024-01-15T10:23:57.888Z WARN [dns] SERVFAIL looking up A for db.internal.local",
    
    # SSL/TLS logs
    "2024-01-15T10:23:58.888Z WARN [ssl-handler] Certificate expiring soon: domain=*.example.com, days_remaining=7",
    "2024-01-15T10:23:58.999Z ERROR [tls] handshake failure: no common cipher suites",
    
    # Metrics/Monitoring logs
    "2024-01-15T10:23:59.100Z INFO [metrics-collector] Metrics flush completed: metrics_count=1500, duration=200ms",
    "2024-01-15T10:23:59.200Z WARN [prometheus] Scrape failed for target http://app:9090/metrics: context deadline exceeded",
    
    # Application-specific logs
    "2024-01-15T10:24:00.100Z ERROR [api] Unhandled exception in /api/v2/users: NullPointerException at UserService.java:142",
    "2024-01-15T10:24:00.200Z INFO [flask] 192.168.1.50 - - [15/Jan/2024 10:24:00] \"GET /health HTTP/1.1\" 200 -",
    "2024-01-15T10:24:00.300Z DEBUG [spring] Initializing bean 'dataSource' with autowired dependencies",
    
    # Memory/Resource logs
    "2024-01-15T10:24:01.200Z CRITICAL [memory-monitor] OOM killer invoked: process=java, pid=5678, memory=8GB",
    "2024-01-15T10:24:01.300Z WARN [gc] GC pause time exceeded threshold: pause=2.5s, threshold=1s, heap_used=7.2GB",
    "2024-01-15T10:24:01.400Z INFO [jvm] Heap memory: used=4.2GB, committed=6GB, max=8GB",
    
    # Security logs
    "2024-01-15T10:24:02.100Z ERROR [firewall] Blocked suspicious request: ip=45.33.32.156, path=/admin, reason=rate_limit",
    "2024-01-15T10:24:02.200Z WARN [waf] SQL injection attempt detected: param=id, value='1 OR 1=1--'",
    "2024-01-15T10:24:02.300Z INFO [audit] User action logged: user=admin, action=delete_user, target_id=789",
]


@dataclass
class BenchmarkResult:
    """Stores results for a single benchmark run"""
    batch_size: int
    total_logs: int
    total_time_sec: float
    throughput_logs_per_sec: float
    avg_latency_per_log_ms: float
    successful_parses: int
    failed_parses: int


def create_batch_prompt(log_entries: List[str]) -> str:
    """Create a single prompt containing multiple log entries for batch processing"""
    numbered_logs = "\n".join([f"[LOG {i+1}]: {log}" for i, log in enumerate(log_entries)])
    
    prompt = f"""Parse the following {len(log_entries)} log entries into JSON format.
Return a JSON array with one object per log entry.

{numbered_logs}

For each log entry, extract these fields:
- log_number (which LOG number from the input)
- timestamp (ISO 8601 format if possible)
- message (original log message)
- level (log severity level: DEBUG, INFO, WARN, ERROR, CRITICAL)
- component (source component/module if present)
- error_code (if present)
- ip_address (if present)

Return ONLY a valid JSON array. Example format:
[
  {{"log_number": 1, "timestamp": "2024-01-15T10:23:45Z", "message": "...", "level": "ERROR", "component": "db-pool", "error_code": "", "ip_address": ""}},
  {{"log_number": 2, ...}}
]
"""
    return prompt


def benchmark_single_log_processing(client: ollama.Client, logs: List[str], options: ollama.Options) -> Tuple[float, int, int]:
    """Benchmark sequential processing (batch_size=1)"""
    system_prompt = "You are an expert in log parsing. Extract and return structured JSON for log entries."
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for log in logs:
        try:
            prompt = f"""Parse this log entry into JSON format:
{log}

Required fields: timestamp, message, level
Optional fields: component, error_code, ip_address, pid

Return ONLY valid JSON."""

            response = client.generate(
                model=MODEL,
                prompt=prompt,
                system=system_prompt,
                options=options,
                format="json"
            )
            
            if response and response.response.strip():
                # Validate it's parseable JSON
                json.loads(response.response)
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            
    total_time = time.time() - start_time
    return total_time, successful, failed


def benchmark_batch_processing(client: ollama.Client, logs: List[str], batch_size: int, options: ollama.Options) -> Tuple[float, int, int]:
    """Benchmark batch processing with specified batch size"""
    system_prompt = "You are an expert in log parsing. Parse multiple log entries and return a JSON array."
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    # Process logs in batches
    for i in range(0, len(logs), batch_size):
        batch = logs[i:i+batch_size]
        
        try:
            prompt = create_batch_prompt(batch)
            
            response = client.generate(
                model=MODEL,
                prompt=prompt,
                system=system_prompt,
                options=options,
                format="json"
            )
            
            if response and response.response.strip():
                # Try to parse as JSON array
                parsed = json.loads(response.response)
                if isinstance(parsed, list):
                    successful += len([p for p in parsed if isinstance(p, dict)])
                    failed += len(batch) - successful
                else:
                    # Single object returned - count as partial success
                    successful += 1
                    failed += len(batch) - 1
            else:
                failed += len(batch)
                
        except json.JSONDecodeError:
            # Attempt to count partial success
            failed += len(batch)
        except Exception as e:
            failed += len(batch)
            
    total_time = time.time() - start_time
    return total_time, successful, failed


def run_benchmark(total_logs: int = 32) -> List[BenchmarkResult]:
    """Run complete benchmark across all batch sizes"""
    
    print("=" * 70)
    print("GraphRCA Batch Inference Benchmark")
    print("=" * 70)
    
    # Initialize Ollama client
    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=TIMEOUT)
        # Warm-up call
        print(f"\n🔄 Warming up model ({MODEL})...")
        client.generate(model=MODEL, prompt="Hello", options=ollama.Options(temperature=0.1))
        print("✅ Model ready")
    except Exception as e:
        print(f"\n❌ Failed to connect to Ollama at {OLLAMA_HOST}")
        print(f"   Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  - Docker: docker exec -it graph-rca-ollama ollama list")
        print("  - Local: ollama serve & ollama pull llama3.2:3b")
        return []
    
    options = ollama.Options(temperature=TEMPERATURE)
    
    # Expand sample logs to desired count
    logs = (SAMPLE_LOGS * ((total_logs // len(SAMPLE_LOGS)) + 1))[:total_logs]
    
    results = []
    
    for batch_size in BATCH_SIZES:
        print(f"\n{'=' * 60}")
        print(f"Testing batch_size = {batch_size} ({RUNS_PER_SIZE} runs)")
        print(f"{'=' * 60}")
        
        run_times = []
        run_successful = []
        run_failed = []
        
        for run in range(1, RUNS_PER_SIZE + 1):
            print(f"  Run {run}/{RUNS_PER_SIZE}...", end=" ", flush=True)
            
            if batch_size == 1:
                total_time, successful, failed = benchmark_single_log_processing(
                    client, logs, options
                )
            else:
                total_time, successful, failed = benchmark_batch_processing(
                    client, logs, batch_size, options
                )
            
            run_times.append(total_time)
            run_successful.append(successful)
            run_failed.append(failed)
            
            throughput = len(logs) / total_time if total_time > 0 else 0
            print(f"✅ {total_time:.2f}s ({throughput:.2f} logs/sec)")
        
        # Calculate averages
        avg_time = statistics.mean(run_times)
        avg_successful = statistics.mean(run_successful)
        avg_failed = statistics.mean(run_failed)
        
        throughput = len(logs) / avg_time if avg_time > 0 else 0
        latency_per_log = (avg_time * 1000) / len(logs)  # ms per log
        
        result = BenchmarkResult(
            batch_size=batch_size,
            total_logs=len(logs),
            total_time_sec=round(avg_time, 3),
            throughput_logs_per_sec=round(throughput, 3),
            avg_latency_per_log_ms=round(latency_per_log, 1),
            successful_parses=int(avg_successful),
            failed_parses=int(avg_failed)
        )
        results.append(result)
        
        print(f"\n  📊 Average: {avg_time:.2f}s, {throughput:.2f} logs/sec, {latency_per_log:.0f}ms/log")
    
    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print LaTeX-ready table of results"""
    if not results:
        return
    
    baseline = results[0].throughput_logs_per_sec if results[0].batch_size == 1 else results[0].throughput_logs_per_sec
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n📋 Console Table:")
    print("-" * 70)
    print(f"{'Batch Size':<12} {'Throughput':<15} {'Latency/Log':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for r in results:
        speedup = r.throughput_logs_per_sec / baseline if baseline > 0 else 1.0
        print(f"{r.batch_size:<12} {r.throughput_logs_per_sec:.2f} logs/s{'':<6} {r.avg_latency_per_log_ms:.0f} ms{'':<8} {speedup:.1f}×")
    
    print("-" * 70)
    
    print("\n📄 LaTeX Table (copy to paper):")
    print("-" * 70)
    print(r"""
\begin{table}[t]
\centering
\caption{Batch Inference Performance (GPU, Llama 3.2 3B)}
\label{tab:batching}
\begin{tabular}{lccc}
\toprule
\textbf{Batch Size} & \textbf{Throughput} & \textbf{Latency/Log} & \textbf{Speedup} \\
\midrule""")
    
    for r in results:
        speedup = r.throughput_logs_per_sec / baseline if baseline > 0 else 1.0
        label = "(baseline)" if r.batch_size == 1 else ""
        print(f"{r.batch_size} {label} & {r.throughput_logs_per_sec:.2f} logs/s & {r.avg_latency_per_log_ms:.0f} ms & {speedup:.1f}$\\times$ \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print("-" * 70)


def save_results(results: List[BenchmarkResult], output_file: str = "batch_benchmark_results.json"):
    """Save results to JSON file"""
    if not results:
        return
    
    output = {
        "model": MODEL,
        "ollama_host": OLLAMA_HOST,
        "batch_sizes_tested": BATCH_SIZES,
        "runs_per_size": RUNS_PER_SIZE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": [asdict(r) for r in results]
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    print("\n🚀 Starting Batch Inference Benchmark (Paper-Quality)")
    print("   This will test batch sizes: " + ", ".join(map(str, BATCH_SIZES)))
    print(f"   Each size will be tested {RUNS_PER_SIZE} times")
    print(f"   Logs per test: {TOTAL_LOGS}")
    print(f"   Unique log formats: {len(SAMPLE_LOGS)}")
    print()
    
    results = run_benchmark(total_logs=TOTAL_LOGS)
    
    if results:
        print_results_table(results)
        save_results(results)
        
        print("\n✅ Benchmark complete!")
        print("   Copy the LaTeX table above into Section IV-A of access.tex")
    else:
        print("\n❌ Benchmark failed - check Ollama connection")

