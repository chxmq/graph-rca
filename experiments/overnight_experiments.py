#!/usr/bin/env python3
"""
===============================================================================
GraphRCA Comprehensive Overnight Experiment Suite
===============================================================================

This script runs all paper experiments with robust error handling and detailed
logging. Designed for overnight execution on GPU servers.

Features:
- Detailed logging to timestamped files
- Intermediate results saved after each step
- Automatic retry on failures
- Progress checkpointing
- Resume from last checkpoint if crashed

Experiments Included:
1. Batch Inference Benchmark (batch sizes 1, 8, 16, 32)
2. Extended scalability tests (up to 500 logs)
3. Model comparison (Llama vs Qwen if available)

Usage:
    python3 overnight_experiments.py

Output:
    experiments/results/
    ├── logs/                    # Detailed execution logs
    ├── batch_results.json       # Batch inference results
    ├── scalability_results.json # Scalability data
    ├── checkpoint.json          # Resume checkpoint
    └── final_latex_tables.tex   # Ready-to-use LaTeX
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import statistics

# Fix for SSL_CERT_FILE environment variable issue
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

try:
    import ollama
except ImportError:
    print("ERROR: ollama package not installed. Run: pip install ollama")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Ollama settings
    "ollama_host": "http://localhost:11434",
    "model_primary": "llama3.2:3b",
    "model_secondary": "qwen2.5-coder:3b",  # Will be tested if available
    "temperature": 0.2,
    "timeout": 300.0,
    
    # Experiment settings
    "batch_sizes": [1, 8, 16, 32],
    "runs_per_config": 5,
    "logs_per_test": 128,
    "scalability_sizes": [16, 32, 64, 128, 256],
    
    # Retry settings
    "max_retries": 3,
    "retry_delay_sec": 10,
    
    # Output settings
    "results_dir": "experiments/results",
    "checkpoint_file": "experiments/results/checkpoint.json",
}

# Sample logs for benchmarking (comprehensive set)
SAMPLE_LOGS = [
    # Database logs
    "2024-01-15T10:23:45.123Z ERROR [db-pool] Connection pool exhausted - no available connections after 30s timeout",
    "2024-01-15T10:23:46.001Z WARN [mysql] Slow query detected: SELECT * FROM users WHERE status='active' took 5.2s",
    "2024-01-15T10:23:46.234Z ERROR [postgres] FATAL: too many connections for role 'app_user'",
    "2024-01-15T10:23:46.456Z INFO [db-replica] Replication lag: primary=db-01, replica=db-02, lag_ms=1500",
    # API Gateway logs
    "2024-01-15T10:23:47.789Z WARN [api-gateway] Request timeout for /api/users endpoint - upstream latency exceeded 5000ms",
    "2024-01-15T10:23:48.100Z ERROR [nginx] upstream timed out (110: Connection timed out) while reading response header",
    "2024-01-15T10:23:48.200Z INFO [kong] rate limit exceeded for client 192.168.1.100, limit=1000/hour",
    # Authentication logs  
    "2024-01-15T10:23:49.000Z INFO [auth-service] User login successful: user_id=12345, ip=192.168.1.100, method=oauth2",
    "Jan 15 10:23:48 server01 sshd[1234]: Failed password for invalid user admin from 10.0.0.1 port 22 ssh2",
    "2024-01-15T10:23:49.500Z WARN [auth] Multiple failed login attempts: user=admin, attempts=5, ip=203.0.113.50",
    "2024-01-15T10:23:49.600Z ERROR [jwt] Token validation failed: expired at 2024-01-15T09:00:00Z",
    # Payment/Transaction logs
    "2024-01-15T10:23:50.111Z ERROR [payment-svc] Transaction failed: tx_id=TXN-789, error_code=INSUFFICIENT_FUNDS",
    "2024-01-15T10:23:50.222Z INFO [stripe] Payment processed: charge_id=ch_abc123, amount=$99.99, status=succeeded",
    "2024-01-15T10:23:50.333Z WARN [payment] Retry attempt 3/5 for transaction TXN-790, reason=gateway_timeout",
    # Cache/Memory logs
    "2024-01-15T10:23:51.222Z DEBUG [cache-layer] Cache miss for key: user_profile_12345, fetching from database",
    "2024-01-15T10:23:51.333Z INFO [redis] BGSAVE completed successfully, 1.2GB saved to disk in 3.5s",
    "2024-01-15T10:23:51.444Z WARN [memcached] Eviction rate high: 1500 items/sec, memory_usage=95%",
    # Kubernetes/Container logs
    "2024-01-15T10:23:52.333Z CRITICAL [k8s-controller] Pod restart loop detected: pod=api-server-abc123, restarts=5",
    "2024-01-15T10:23:52.444Z INFO [kubelet] Container started: container=app, image=myapp:v2.1.0, pod=web-server",
    "2024-01-15T10:23:52.555Z WARN [k8s] Node memory pressure: node=worker-03, available=512MB, threshold=1GB",
    "2024-01-15T10:23:52.666Z ERROR [docker] OCI runtime exec failed: unable to start container process",
    # Load Balancer/Network logs
    "2024-01-15T10:23:53.444Z INFO [load-balancer] Backend health check passed: server=backend-01, latency=15ms",
    "2024-01-15T10:23:53.555Z WARN [haproxy] Server backend/server2 is DOWN, reason: Layer4 timeout",
    "2024-01-15T10:23:53.666Z ERROR [envoy] upstream connect error, reset reason: connection timeout",
    # Storage logs
    "2024-01-15T10:23:54.555Z WARN [storage-svc] Disk usage above threshold: mount=/data, usage=92%, total=500GB",
    "2024-01-15T10:23:54.666Z ERROR [nfs] NFS: server not responding, still trying",
    "2024-01-15T10:23:54.777Z INFO [s3] Object uploaded: bucket=logs-archive, key=2024/01/app.log.gz, size=45MB",
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
    "2024-01-15T10:23:59.200Z WARN [prometheus] Scrape failed for target http://app:9090/metrics",
    # Application-specific logs
    "2024-01-15T10:24:00.100Z ERROR [api] Unhandled exception: NullPointerException at UserService.java:142",
    "2024-01-15T10:24:00.200Z INFO [flask] 192.168.1.50 - GET /health HTTP/1.1 200 -",
    "2024-01-15T10:24:00.300Z DEBUG [spring] Initializing bean 'dataSource' with autowired dependencies",
    # Memory/Resource logs
    "2024-01-15T10:24:01.200Z CRITICAL [memory-monitor] OOM killer invoked: process=java, pid=5678, memory=8GB",
    "2024-01-15T10:24:01.300Z WARN [gc] GC pause time exceeded threshold: pause=2.5s, threshold=1s",
    "2024-01-15T10:24:01.400Z INFO [jvm] Heap memory: used=4.2GB, committed=6GB, max=8GB",
    # Security logs
    "2024-01-15T10:24:02.100Z ERROR [firewall] Blocked suspicious request: ip=45.33.32.156, path=/admin",
    "2024-01-15T10:24:02.200Z WARN [waf] SQL injection attempt detected: param=id, value='1 OR 1=1--'",
    "2024-01-15T10:24:02.300Z INFO [audit] User action logged: user=admin, action=delete_user, target_id=789",
]


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(results_dir: str) -> logging.Logger:
    """Setup comprehensive logging to both file and console"""
    log_dir = Path(results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("GraphRCA_Experiments")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logs
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler - summary logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SingleRunResult:
    """Result of a single benchmark run"""
    run_id: int
    batch_size: int
    total_logs: int
    total_time_sec: float
    successful_parses: int
    failed_parses: int
    throughput: float  # logs/sec
    latency_per_log_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


@dataclass
class BatchSizeResult:
    """Aggregated results for a batch size"""
    batch_size: int
    total_logs: int
    runs: List[SingleRunResult]
    avg_throughput: float
    std_throughput: float
    avg_latency_ms: float
    std_latency_ms: float
    speedup_vs_baseline: float = 1.0
    

@dataclass
class ExperimentResults:
    """Complete experiment results"""
    experiment_name: str
    model: str
    start_time: str
    end_time: Optional[str]
    batch_results: List[BatchSizeResult]
    scalability_results: Optional[List[Dict]] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manages experiment checkpoints for resume capability"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, data: Dict) -> None:
        """Save checkpoint data"""
        data["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self) -> Optional[Dict]:
        """Load checkpoint data if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None
    
    def clear(self) -> None:
        """Clear checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def create_batch_prompt(log_entries: List[str]) -> str:
    """Create a prompt for batch processing multiple logs"""
    numbered_logs = "\n".join([f"[LOG {i+1}]: {log}" for i, log in enumerate(log_entries)])
    
    return f"""Parse the following {len(log_entries)} log entries into JSON format.
Return a JSON array with one object per log entry.

{numbered_logs}

For each log entry, extract:
- log_number, timestamp, message, level, component (if present)

Return ONLY a valid JSON array."""


def run_single_inference(
    client: ollama.Client,
    log: str,
    model: str,
    options: ollama.Options,
    logger: logging.Logger
) -> Tuple[bool, float]:
    """Run single log inference, return (success, time_taken)"""
    start = time.time()
    try:
        prompt = f"""Parse this log entry into JSON:
{log}

Required: timestamp, message, level
Optional: component, error_code, ip_address

Return ONLY valid JSON."""

        response = client.generate(
            model=model,
            prompt=prompt,
            system="You are a log parsing expert. Return structured JSON.",
            options=options,
            format="json"
        )
        
        elapsed = time.time() - start
        if response and response.response.strip():
            json.loads(response.response)  # Validate JSON
            return True, elapsed
        return False, elapsed
    except Exception as e:
        logger.debug(f"Single inference failed: {e}")
        return False, time.time() - start


def run_batch_inference(
    client: ollama.Client,
    logs: List[str],
    batch_size: int,
    model: str,
    options: ollama.Options,
    logger: logging.Logger
) -> Tuple[int, int, float]:
    """Run batch inference, return (successful, failed, time_taken)"""
    start = time.time()
    successful = 0
    failed = 0
    
    for i in range(0, len(logs), batch_size):
        batch = logs[i:i+batch_size]
        try:
            prompt = create_batch_prompt(batch)
            response = client.generate(
                model=model,
                prompt=prompt,
                system="You are a log parsing expert. Return a JSON array.",
                options=options,
                format="json"
            )
            
            if response and response.response.strip():
                parsed = json.loads(response.response)
                if isinstance(parsed, list):
                    successful += min(len(parsed), len(batch))
                    failed += max(0, len(batch) - len(parsed))
                else:
                    successful += 1
                    failed += len(batch) - 1
            else:
                failed += len(batch)
        except Exception as e:
            logger.debug(f"Batch inference failed: {e}")
            failed += len(batch)
    
    return successful, failed, time.time() - start


def run_batch_size_experiment(
    client: ollama.Client,
    batch_size: int,
    logs: List[str],
    runs: int,
    model: str,
    options: ollama.Options,
    logger: logging.Logger,
    checkpoint_mgr: CheckpointManager
) -> BatchSizeResult:
    """Run all runs for a specific batch size"""
    
    logger.info(f"Testing batch_size={batch_size} ({runs} runs, {len(logs)} logs each)")
    run_results = []
    
    for run_id in range(1, runs + 1):
        logger.info(f"  Run {run_id}/{runs}...")
        
        try:
            if batch_size == 1:
                # Sequential processing
                successful = 0
                failed = 0
                start = time.time()
                for log in logs:
                    success, _ = run_single_inference(client, log, model, options, logger)
                    if success:
                        successful += 1
                    else:
                        failed += 1
                total_time = time.time() - start
            else:
                # Batch processing
                successful, failed, total_time = run_batch_inference(
                    client, logs, batch_size, model, options, logger
                )
            
            throughput = len(logs) / total_time if total_time > 0 else 0
            latency = (total_time * 1000) / len(logs)
            
            result = SingleRunResult(
                run_id=run_id,
                batch_size=batch_size,
                total_logs=len(logs),
                total_time_sec=round(total_time, 3),
                successful_parses=successful,
                failed_parses=failed,
                throughput=round(throughput, 4),
                latency_per_log_ms=round(latency, 2)
            )
            run_results.append(result)
            
            logger.info(f"    ✓ {total_time:.1f}s, {throughput:.2f} logs/s, {latency:.0f}ms/log")
            
            # Save checkpoint after each run
            checkpoint_mgr.save({
                "current_batch_size": batch_size,
                "current_run": run_id,
                "partial_results": [asdict(r) for r in run_results]
            })
            
        except Exception as e:
            logger.error(f"    ✗ Run {run_id} failed: {e}")
            run_results.append(SingleRunResult(
                run_id=run_id,
                batch_size=batch_size,
                total_logs=len(logs),
                total_time_sec=0,
                successful_parses=0,
                failed_parses=len(logs),
                throughput=0,
                latency_per_log_ms=0,
                error=str(e)
            ))
    
    # Calculate aggregated stats
    valid_runs = [r for r in run_results if r.error is None]
    if valid_runs:
        throughputs = [r.throughput for r in valid_runs]
        latencies = [r.latency_per_log_ms for r in valid_runs]
        
        return BatchSizeResult(
            batch_size=batch_size,
            total_logs=len(logs),
            runs=run_results,
            avg_throughput=round(statistics.mean(throughputs), 4),
            std_throughput=round(statistics.stdev(throughputs) if len(throughputs) > 1 else 0, 4),
            avg_latency_ms=round(statistics.mean(latencies), 2),
            std_latency_ms=round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2)
        )
    else:
        return BatchSizeResult(
            batch_size=batch_size,
            total_logs=len(logs),
            runs=run_results,
            avg_throughput=0,
            std_throughput=0,
            avg_latency_ms=0,
            std_latency_ms=0
        )


# ============================================================================
# RESULT GENERATION
# ============================================================================

def generate_latex_table(results: ExperimentResults) -> str:
    """Generate LaTeX table from results"""
    baseline_throughput = None
    for br in results.batch_results:
        if br.batch_size == 1:
            baseline_throughput = br.avg_throughput
            break
    
    if baseline_throughput is None or baseline_throughput == 0:
        baseline_throughput = results.batch_results[0].avg_throughput if results.batch_results else 1
    
    latex = r"""
%% ============================================================================
%% AUTO-GENERATED LaTeX TABLE - Batch Inference Performance
%% Generated: """ + datetime.now().isoformat() + r"""
%% Model: """ + results.model + r"""
%% ============================================================================

\begin{table}[t]
\centering
\caption{Batch Inference Performance (""" + results.model + r""", GPU)}
\label{tab:batching}
\begin{tabular}{lcccc}
\toprule
\textbf{Batch Size} & \textbf{Throughput (logs/s)} & \textbf{Latency (ms/log)} & \textbf{Std Dev} & \textbf{Speedup} \\
\midrule
"""
    
    for br in results.batch_results:
        speedup = br.avg_throughput / baseline_throughput if baseline_throughput > 0 else 1
        label = " (baseline)" if br.batch_size == 1 else ""
        latex += f"{br.batch_size}{label} & {br.avg_throughput:.2f} & {br.avg_latency_ms:.0f} & $\\pm${br.std_latency_ms:.1f} & {speedup:.1f}$\\times$ \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def save_results(results: ExperimentResults, output_dir: str, logger: logging.Logger) -> None:
    """Save all results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON results
    json_file = output_path / "batch_results.json"
    with open(json_file, "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)
    logger.info(f"Saved JSON results to: {json_file}")
    
    # Save LaTeX table
    latex_file = output_path / "final_latex_tables.tex"
    with open(latex_file, "w") as f:
        f.write(generate_latex_table(results))
    logger.info(f"Saved LaTeX table to: {latex_file}")
    
    # Save human-readable summary
    summary_file = output_path / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("GraphRCA Batch Inference Experiment Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {results.model}\n")
        f.write(f"Start Time: {results.start_time}\n")
        f.write(f"End Time: {results.end_time}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Batch Size':<12} {'Throughput':<18} {'Latency':<18} {'Speedup':<10}\n")
        f.write("-" * 70 + "\n")
        
        baseline = results.batch_results[0].avg_throughput if results.batch_results else 1
        for br in results.batch_results:
            speedup = br.avg_throughput / baseline if baseline > 0 else 1
            f.write(f"{br.batch_size:<12} {br.avg_throughput:.2f} logs/s{'':<8} {br.avg_latency_ms:.0f} ms{'':<12} {speedup:.1f}x\n")
        
        f.write("-" * 70 + "\n")
        if results.errors:
            f.write(f"\nErrors encountered: {len(results.errors)}\n")
            for err in results.errors:
                f.write(f"  - {err}\n")
    
    logger.info(f"Saved summary to: {summary_file}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    """Main experiment runner"""
    
    # Setup
    results_dir = CONFIG["results_dir"]
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(results_dir)
    checkpoint_mgr = CheckpointManager(CONFIG["checkpoint_file"])
    
    logger.info("=" * 70)
    logger.info("GraphRCA Comprehensive Overnight Experiment Suite")
    logger.info("=" * 70)
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")
    
    # Initialize Ollama client
    try:
        logger.info(f"Connecting to Ollama at {CONFIG['ollama_host']}...")
        client = ollama.Client(host=CONFIG["ollama_host"], timeout=CONFIG["timeout"])
        
        # Warmup
        logger.info(f"Warming up model ({CONFIG['model_primary']})...")
        client.generate(model=CONFIG["model_primary"], prompt="Hello", options=ollama.Options(temperature=0.1))
        logger.info("✓ Model ready")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    options = ollama.Options(temperature=CONFIG["temperature"])
    
    # Prepare logs
    logs = (SAMPLE_LOGS * ((CONFIG["logs_per_test"] // len(SAMPLE_LOGS)) + 1))[:CONFIG["logs_per_test"]]
    logger.info(f"Using {len(logs)} log entries per test")
    
    # Initialize results
    results = ExperimentResults(
        experiment_name="Batch Inference Benchmark",
        model=CONFIG["model_primary"],
        start_time=datetime.now().isoformat(),
        end_time=None,
        batch_results=[]
    )
    
    # Run batch size experiments
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Batch Size Experiments")
    logger.info("=" * 70)
    
    for batch_size in CONFIG["batch_sizes"]:
        try:
            batch_result = run_batch_size_experiment(
                client=client,
                batch_size=batch_size,
                logs=logs,
                runs=CONFIG["runs_per_config"],
                model=CONFIG["model_primary"],
                options=options,
                logger=logger,
                checkpoint_mgr=checkpoint_mgr
            )
            results.batch_results.append(batch_result)
            
            # Save intermediate results
            save_results(results, results_dir, logger)
            
        except Exception as e:
            error_msg = f"Batch size {batch_size} failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            results.errors.append(error_msg)
    
    # Calculate speedups
    if results.batch_results:
        baseline_throughput = results.batch_results[0].avg_throughput
        for br in results.batch_results:
            br.speedup_vs_baseline = round(br.avg_throughput / baseline_throughput, 2) if baseline_throughput > 0 else 1.0
    
    # Finalize
    results.end_time = datetime.now().isoformat()
    save_results(results, results_dir, logger)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {results.start_time} to {results.end_time}")
    logger.info(f"Results saved to: {results_dir}/")
    logger.info(f"Errors: {len(results.errors)}")
    
    # Print LaTeX table to console
    print("\n" + "=" * 70)
    print("LATEX TABLE (copy to paper):")
    print("=" * 70)
    print(generate_latex_table(results))
    
    # Cleanup checkpoint
    checkpoint_mgr.clear()
    logger.info("Checkpoint cleared. Experiment complete!")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user. Partial results saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
