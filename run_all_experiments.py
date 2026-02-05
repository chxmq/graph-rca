#!/usr/bin/env python3
"""
Master Script: Run All GraphRCA Experiments

This script runs all experiments sequentially with robust error handling.
If an experiment fails, it logs the error and continues to the next one.
Results are saved in timestamped folders for each experiment run.

Usage:
    python run_all_experiments.py [--resume-from EXPERIMENT_NUM] [--skip SKIP_LIST]
    
Examples:
    python run_all_experiments.py                    # Run all experiments
    python run_all_experiments.py --resume-from 5    # Resume from experiment 5
    python run_all_experiments.py --skip 4,7          # Skip experiments 4 and 7
"""

import os
import sys
import json
import time
import subprocess
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_BASE = PROJECT_ROOT / "experiments" / "results"

# Experiment configuration
EXPERIMENTS = [
    {
        "num": 1,
        "name": "Batch Inference",
        "dir": "01_batch_inference",
        "script": "run_experiment.py",
        "description": "Tests batch sizes 1, 8, 16, 32 for LLM parsing performance",
        "estimated_time_min": 30,
        "requires": ["ollama", "llama3.2:3b"],
    },
    {
        "num": 2,
        "name": "DAG Scalability",
        "dir": "02_scalability",
        "script": "run_experiment.py",
        "description": "Measures O(n) complexity of DAG construction",
        "estimated_time_min": 5,
        "requires": ["backend dependencies"],
    },
    {
        "num": 3,
        "name": "Baseline Comparison",
        "dir": "03_baseline_comparison",
        "script": "run_experiment.py",
        "description": "Full pipeline RCA accuracy across 20 scenarios",
        "estimated_time_min": 120,
        "requires": ["ollama", "llama3.2:3b"],
    },
    {
        "num": 4,
        "name": "Documentation Ablation",
        "dir": "04_doc_ablation",
        "script": "run_experiment.py",
        "description": "Tests impact of documentation on RCA accuracy",
        "estimated_time_min": 60,
        "requires": ["ollama", "llama3.2:3b", "qwen3:32b", "real_incidents data"],
    },
    {
        "num": 5,
        "name": "Noise Sensitivity",
        "dir": "05_noise_sensitivity",
        "script": "run_experiment.py",
        "description": "Tests RAG retrieval accuracy at multiple noise levels",
        "estimated_time_min": 45,
        "requires": ["ollama", "nomic-embed-text", "chromadb", "real_incidents data"],
    },
    {
        "num": 6,
        "name": "Parser Accuracy",
        "dir": "06_parser_accuracy",
        "script": "run_experiment.py",
        "description": "Tests LLM parsing accuracy on LogHub datasets (BGL, HDFS)",
        "estimated_time_min": 240,  # ~4 hours
        "requires": ["ollama", "llama3.2:3b", "loghub data"],
    },
    {
        "num": 7,
        "name": "Multi-Judge Validation",
        "dir": "07_multi_judge_validation",
        "script": "run_experiment.py",
        "description": "Cross-validates RCA accuracy using multiple LLM judges",
        "estimated_time_min": 180,  # ~3 hours
        "requires": ["ollama", "qwen3:32b", "real_incidents data"],
        "optional": ["OPENAI_API_KEY", "GROQ_API_KEY"],
    },
    {
        "num": 8,
        "name": "RAG Real-World",
        "dir": "08_rag_real_world",
        "script": "run_experiment.py",
        "description": "Compares baseline vs RAG accuracy on real-world incidents",
        "estimated_time_min": 90,
        "requires": ["ollama", "llama3.2:3b", "chromadb", "real_incidents data"],
    },
    {
        "num": 9,
        "name": "Latency Profiling",
        "dir": "09_latency_profiling",
        "script": "run_experiment.py",
        "description": "Measures end-to-end latency breakdown",
        "estimated_time_min": 10,
        "requires": ["ollama", "llama3.2:3b", "chromadb"],
    },
]

# Global state
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG = []
SUMMARY = {
    "run_id": RUN_ID,
    "start_time": datetime.now().isoformat(),
    "experiments": [],
    "total_time_seconds": 0,
    "successful": 0,
    "failed": 0,
    "skipped": 0,
}


def log(message: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    RUN_LOG.append(log_entry)
    return log_entry


def check_prerequisites(exp: Dict) -> tuple[bool, List[str]]:
    """Check if prerequisites for experiment are met."""
    missing = []
    
    # Check Ollama
    if "ollama" in exp.get("requires", []):
        try:
            import ollama
            client = ollama.Client(host="http://localhost:11434", timeout=5.0)
            client.list()
        except Exception as e:
            missing.append(f"Ollama not running or not accessible: {e}")
    
    # Check models
    if "llama3.2:3b" in exp.get("requires", []):
        try:
            # Robust model extraction (exact match to check_prerequisites.py)
            try:
                # Handle different versions of ollama python client
                resp = client.list()
                models_list = resp.get("models", [])
            except:
                models_list = client.list()
                
            models = []
            for m in models_list:
                # Try dict access (newer/older mismatch)
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                # Try object attribute checks
                else:
                    name = getattr(m, "name", None) or getattr(m, "model", None)
                if name: models.append(name)

            if "llama3.2:3b" not in models:
                missing.append("Model 'llama3.2:3b' not found in Ollama")
                log(f"DEBUG: Found models: {models}", "ERROR")
        except:
            missing.append("Cannot check Ollama models")
    
    if "qwen3:32b" in exp.get("requires", []):
        try:
            import ollama
            client = ollama.Client(host="http://localhost:11434", timeout=5.0)
            
            # Robust model extraction (exact match to check_prerequisites.py)
            try:
                # Handle different versions of ollama python client
                resp = client.list()
                models_list = resp.get("models", [])
            except:
                models_list = client.list()
                
            models = []
            for m in models_list:
                # Try dict access (newer/older mismatch)
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                # Try object attribute checks
                else:
                    name = getattr(m, "name", None) or getattr(m, "model", None)
                if name: models.append(name)

            if "qwen3:32b" not in models:
                missing.append("Model 'qwen3:32b' not found in Ollama")
                log(f"DEBUG: Found models: {models}", "ERROR")
        except:
            missing.append("Cannot check Ollama models")
    
    # Check data directories
    if "real_incidents data" in exp.get("requires", []):
        incident_dir = PROJECT_ROOT / "data" / "real_incidents"
        if not incident_dir.exists() or len(list(incident_dir.glob("incident_*"))) == 0:
            missing.append(f"Real incidents data not found at {incident_dir}")
    
    if "loghub data" in exp.get("requires", []):
        loghub_dir = PROJECT_ROOT / "data" / "loghub"
        if not loghub_dir.exists():
            missing.append(f"LogHub data directory not found at {loghub_dir}")
    
    # Check backend dependencies
    if "backend dependencies" in exp.get("requires", []):
        backend_path = PROJECT_ROOT / "backend"
        if not backend_path.exists():
            missing.append("Backend directory not found")
        else:
            try:
                sys.path.insert(0, str(backend_path))
                from app.utils.graph_generator import GraphGenerator
            except ImportError as e:
                missing.append(f"Backend dependencies not available: {e}")
    
    # Check optional requirements (warn but don't fail)
    for opt_req in exp.get("optional", []):
        if opt_req not in os.environ:
            log(f"⚠ Optional requirement '{opt_req}' not set (experiment may have limited functionality)", "WARN")
    
    return len(missing) == 0, missing


def run_experiment(exp: Dict, results_dir: Path) -> Dict:
    """Run a single experiment."""
    exp_num = exp["num"]
    exp_name = exp["name"]
    exp_dir = EXPERIMENTS_DIR / exp["dir"]
    script_path = exp_dir / exp["script"]
    
    log(f"\n{'='*80}")
    log(f"EXPERIMENT {exp_num}: {exp_name}")
    log(f"{'='*80}")
    log(f"Description: {exp['description']}")
    log(f"Estimated time: {exp['estimated_time_min']} minutes")
    log(f"Directory: {exp_dir}")
    
    # Check prerequisites
    log("Checking prerequisites...")
    prerequisites_ok, missing = check_prerequisites(exp)
    
    if not prerequisites_ok:
        error_msg = f"Prerequisites not met: {', '.join(missing)}"
        log(f"❌ SKIPPING: {error_msg}", "ERROR")
        return {
            "num": exp_num,
            "name": exp_name,
            "status": "skipped",
            "error": error_msg,
            "duration_seconds": 0,
        }
    
    log("✓ Prerequisites met")
    
    # Create experiment-specific results directory
    exp_results_dir = results_dir / f"{exp_num:02d}_{exp['dir']}"
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to experiment directory
    original_cwd = os.getcwd()
    os.chdir(exp_dir)
    
    try:
        # Run experiment script
        log(f"Running: python {script_path}")
        start_time = time.time()
        
        # Run with timeout (estimated_time * 2 + 30 minutes buffer)
        timeout_seconds = (exp["estimated_time_min"] * 2 + 30) * 60
        
        cmd = [sys.executable, script_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "SMOKE_TEST": "1" if os.environ.get("SMOKE_TEST") else ""},
        )
        
        duration = time.time() - start_time
        
        # Save stdout/stderr
        stdout_file = exp_results_dir / "stdout.txt"
        stderr_file = exp_results_dir / "stderr.txt"
        stdout_file.write_text(result.stdout)
        stderr_file.write_text(result.stderr)
        
        if result.returncode == 0:
            log(f"✓ Experiment {exp_num} completed successfully in {duration/60:.1f} minutes")
            
            # Copy experiment results if they exist
            exp_data_dir = exp_dir / "data"
            if exp_data_dir.exists():
                import shutil
                for result_file in exp_data_dir.glob("*"):
                    if result_file.is_file() and not result_file.name.startswith("."):
                        shutil.copy2(result_file, exp_results_dir / result_file.name)
                        log(f"  Copied result file: {result_file.name}")
            
            return {
                "num": exp_num,
                "name": exp_name,
                "status": "success",
                "duration_seconds": duration,
                "output_files": [f.name for f in exp_results_dir.glob("*") if f.is_file()],
            }
        else:
            error_msg = f"Script exited with code {result.returncode}"
            log(f"❌ FAILED: {error_msg}", "ERROR")
            log(f"Stderr: {result.stderr[:500]}", "ERROR")
            
            # Save error details
            error_file = exp_results_dir / "error.json"
            error_file.write_text(json.dumps({
                "returncode": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout[:1000],  # First 1000 chars
            }, indent=2))
            
            return {
                "num": exp_num,
                "name": exp_name,
                "status": "failed",
                "error": error_msg,
                "duration_seconds": duration,
                "returncode": result.returncode,
            }
    
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        error_msg = f"Experiment timed out after {timeout_seconds/60:.1f} minutes"
        log(f"❌ TIMEOUT: {error_msg}", "ERROR")
        return {
            "num": exp_num,
            "name": exp_name,
            "status": "timeout",
            "error": error_msg,
            "duration_seconds": duration,
        }
    
    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = f"Unexpected error: {str(e)}"
        error_trace = traceback.format_exc()
        log(f"❌ ERROR: {error_msg}", "ERROR")
        log(f"Traceback: {error_trace}", "ERROR")
        
        # Save error details
        error_file = exp_results_dir / "error.json"
        error_file.write_text(json.dumps({
            "error": str(e),
            "traceback": error_trace,
        }, indent=2))
        
        return {
            "num": exp_num,
            "name": exp_name,
            "status": "error",
            "error": error_msg,
            "duration_seconds": duration,
        }
    
    finally:
        os.chdir(original_cwd)


def save_summary(results_dir: Path):
    """Save run summary."""
    SUMMARY["end_time"] = datetime.now().isoformat()
    SUMMARY["total_time_seconds"] = sum(exp.get("duration_seconds", 0) for exp in SUMMARY["experiments"])
    SUMMARY["successful"] = sum(1 for exp in SUMMARY["experiments"] if exp["status"] == "success")
    SUMMARY["failed"] = sum(1 for exp in SUMMARY["experiments"] if exp["status"] in ["failed", "error", "timeout"])
    SUMMARY["skipped"] = sum(1 for exp in SUMMARY["experiments"] if exp["status"] == "skipped")
    
    # Save summary JSON
    summary_file = results_dir / "summary.json"
    summary_file.write_text(json.dumps(SUMMARY, indent=2))
    
    # Save log
    log_file = results_dir / "run.log"
    log_file.write_text("\n".join(RUN_LOG))
    
    # Print summary
    log("\n" + "="*80)
    log("RUN SUMMARY")
    log("="*80)
    log(f"Run ID: {RUN_ID}")
    log(f"Total time: {SUMMARY['total_time_seconds']/60:.1f} minutes")
    log(f"Successful: {SUMMARY['successful']}")
    log(f"Failed: {SUMMARY['failed']}")
    log(f"Skipped: {SUMMARY['skipped']}")
    log(f"Results directory: {results_dir}")
    log("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run all GraphRCA experiments")
    parser.add_argument(
        "--resume-from",
        type=int,
        help="Resume from experiment number (skip previous experiments)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of experiment numbers to skip (e.g., '4,7')",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List all experiments and exit",
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke test mode (minimal data, fast verify)",
    )
    
    args = parser.parse_args()
    
    if args.smoke:
        os.environ["SMOKE_TEST"] = "1"
        log("🔥 RUNNING IN SMOKE TEST MODE (Minimal data for fast verification) 🔥")
    
    # List experiments if requested
    if args.list_only:
        print("\nAvailable Experiments:")
        print("="*80)
        for exp in EXPERIMENTS:
            print(f"{exp['num']:2d}. {exp['name']:30s} ({exp['estimated_time_min']:3d} min) - {exp['description']}")
        print("="*80)
        return
    
    # Parse skip list
    skip_list = []
    if args.skip:
        skip_list = [int(x.strip()) for x in args.skip.split(",")]
    
    # Create results directory
    results_dir = RESULTS_BASE / RUN_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"Starting experiment run: {RUN_ID}")
    log(f"Results will be saved to: {results_dir}")
    
    # Determine which experiments to run
    experiments_to_run = []
    start_idx = 0
    
    if args.resume_from:
        start_idx = next((i for i, exp in enumerate(EXPERIMENTS) if exp["num"] == args.resume_from), 0)
        log(f"Resuming from experiment {args.resume_from} (index {start_idx})")
    
    for i, exp in enumerate(EXPERIMENTS):
        if i < start_idx:
            continue
        if exp["num"] in skip_list:
            log(f"Skipping experiment {exp['num']} (--skip flag)")
            SUMMARY["experiments"].append({
                "num": exp["num"],
                "name": exp["name"],
                "status": "skipped",
                "reason": "user requested skip",
            })
            continue
        experiments_to_run.append(exp)
    
    log(f"Will run {len(experiments_to_run)} experiments")
    total_estimated_time = sum(exp["estimated_time_min"] for exp in experiments_to_run)
    log(f"Estimated total time: {total_estimated_time} minutes ({total_estimated_time/60:.1f} hours)")
    
    # Run experiments
    for exp in experiments_to_run:
        result = run_experiment(exp, results_dir)
        SUMMARY["experiments"].append(result)
        
        # Brief pause between experiments
        time.sleep(2)
    
    # Save summary
    save_summary(results_dir)
    
    # Exit with error code if any failed
    if SUMMARY["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
