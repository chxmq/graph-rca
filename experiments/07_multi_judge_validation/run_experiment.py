#!/usr/bin/env python3
"""
Multi-Judge RCA Validation Experiment
Tests RCA accuracy using multiple LLM judges.
CORRECTED: Uses actual GraphRCA (GraphGenerator) for prediction, validating the ALGORITHM not just the LLM.
"""

import os
import sys
import json
import time
import random
import re
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# --- Backend Integration ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
try:
    from app.models.parsing_data_models import LogEntry, LogChain
    from app.utils.graph_generator import GraphGenerator
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)
# ---------------------------

if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "temperature": 0.2,
    "runs_per_incident": 3,
    "train_ratio": 0.75,
    "random_seed": 42,
    "judges": {
        "qwen": {"model": "qwen3:32b", "type": "ollama"},
        "gpt": {"model": "gpt-4o-mini", "type": "openai"},
        "groq": {"model": "llama-3.3-70b-versatile", "type": "groq"},
    }
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"


class JudgeClient:
    """Unified interface for different LLM judges."""
    
    def __init__(self, judge_name: str):
        self.judge_name = judge_name
        self.judge_config = CONFIG["judges"][judge_name]
        self.client = None
        self.api_keys = []
        self.current_key_idx = 0
        self._init_client()
    
    def _init_client(self):
        judge_type = self.judge_config["type"]
        
        if judge_type == "ollama":
            self.client = ollama.Client(host=CONFIG["ollama_host"])
            
        elif judge_type == "openai":
            keys = os.environ.get("OPENAI_API_KEY", "").split(",")
            self.api_keys = [k.strip() for k in keys if k.strip()]
            if not self.api_keys:
                 self.client = None
            else:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_keys[0])
            
        elif judge_type == "groq":
            keys = os.environ.get("GROQ_API_KEY", "").split(",")
            self.api_keys = [k.strip() for k in keys if k.strip()]
            if not self.api_keys:
                self.client = None
            else:
                from groq import Groq
                self.client = Groq(api_key=self.api_keys[0])
    
    def _rotate_key(self):
        """Rotate to next API key if available."""
        if not self.api_keys or len(self.api_keys) <= 1:
            return False
        
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_idx]
        print(f"  ↻ Rotating API key to index {self.current_key_idx}...")
        
        judge_type = self.judge_config["type"]
        if judge_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=new_key)
        elif judge_type == "groq":
            from groq import Groq
            self.client = Groq(api_key=new_key)
        return True
    
    def score(self, prediction: str, ground_truth: str) -> float:
        """Score prediction against ground truth."""
        if not prediction or len(prediction.strip()) < 5:
            return 0.0
        
        if not self.client: # Mock
            return 0.5
        
        prompt = f"""Compare these two root cause descriptions and rate their similarity from 0.0 to 1.0.

Ground Truth: {ground_truth}
Prediction: {prediction}

Respond with ONLY a number between 0.0 and 1.0:"""

        for attempt in range(3):
            try:
                score = self._call_judge(prompt)
                if score is not None:
                    return score
            except Exception as e:
                print(f"  Judge attempt {attempt+1}/3 failed: {e}")
                time.sleep(2)
        
        return 0.5
    
    def _call_judge(self, prompt: str) -> Optional[float]:
        judge_type = self.judge_config["type"]
        model = self.judge_config["model"]
        
        # Retry logic with key rotation
        max_retries = len(self.api_keys) if self.api_keys else 1
        # Cap retries to avoid infinite loops but allow at least one full rotation + extra
        retries = max(3, max_retries + 1)
        
        for attempt in range(retries):
            try:
                if judge_type == "ollama":
                    response = self.client.generate(
                        model=model,
                        prompt=prompt,
                        options={"temperature": 0.0}
                    )
                    text = response.response.strip()
                    # Strip <think>...</think> blocks from qwen3 responses
                    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                    
                elif judge_type in ["openai", "groq"]:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=10
                    )
                    text = response.choices[0].message.content.strip()
                
                match = re.search(r'(0\.\d+|1\.0|0|1)', text)
                if match:
                    return min(1.0, max(0.0, float(match.group(1))))
                return None
                
            except Exception as e:
                # If we have multiple keys, rotate and retry
                if (judge_type in ["openai", "groq"]) and self._rotate_key():
                    time.sleep(1) # Brief pause
                    continue
                else:
                    raise e # Re-raise if rotation didn't happen (or not supported)
        return None


def load_incidents() -> List[Dict]:
    """Load all real-world incidents."""
    incidents = []
    
    for folder in sorted(INCIDENT_DIR.glob("incident_*")):
        try:
            with open(folder / "ground_truth.json") as f:
                gt = json.load(f)
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()
            logs_file = folder / "logs.txt"
            logs = logs_file.read_text() if logs_file.exists() else ""
            
            incidents.append({
                "id": folder.name,
                "root_cause": gt.get("root_cause", ""),
                "category": gt.get("category", "Unknown"),
                "postmortem": postmortem,
                "logs": logs
            })
        except:
            pass
    return incidents


def parse_logs_to_chain(logs_str: str) -> LogChain:
    """Helper to convert string logs to LogChain using robust parsing."""
    logs = logs_str.split("\n")
    entries = []
    
    # Robust patterns for various log formats
    patterns = [
        # ISO timestamp with brackets: 2023-03-08 14:30:00 [INFO] component: msg
        re.compile(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(.+)$'),
        # ISO timestamp plain: 2024-01-15T10:23:45.123Z ERROR msg
        re.compile(r'^(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s+(\w+)\s+(.+)$'),
        # Syslog: Jan 15 10:23:48 server01 sshd[1234]: msg
        re.compile(r'^(\w{3}\s+\d+\s+[\d:]+)\s+\S+\s+\S+:\s+(.+)$'),
    ]
    
    for log in logs:
        log = log.strip()
        if not log or log.startswith('#'):
            continue
        
        ts, level, msg = "unknown", "INFO", log
        parsed = False
        
        for pattern in patterns:
            match = pattern.match(log)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    ts, level, msg = groups
                elif len(groups) == 2:
                    ts, msg = groups
                    level = "INFO"
                parsed = True
                break
        
        if not parsed:
            # Fallback: try to extract level from anywhere in log
            level_match = re.search(r'\b(CRITICAL|ERROR|WARN(?:ING)?|INFO|DEBUG)\b', log, re.I)
            if level_match:
                level = level_match.group(1).upper()
                if level == "WARNING":
                    level = "WARN"
        
        entries.append(LogEntry(
            timestamp=ts,
            level=level,
            message=msg,
            pid="", component="", error_code="", username="", ip_address="", group="", trace_id="", request_id=""
        ))
    return LogChain(log_chain=entries)


def run_graphrca(logs: str) -> str:
    """Run actual GraphRCA algorithm."""
    try:
        if not logs or len(logs) < 10:
             return "Insufficient logs"

        # 1. Parse
        log_chain = parse_logs_to_chain(logs)
        
        # 2. Build DAG
        generator = GraphGenerator(log_chain)
        dag = generator.generate_dag()
        
        # 3. Find Root Cause using graph traversal
        return dag.root_cause
        
    except Exception as e:
        return f"GraphRCA failed: {str(e)}"


def run_validation(judge_name: str) -> Dict:
    """Run multi-judge validation."""
    print("=" * 70)
    print(f"MULTI-JUDGE VALIDATION: {judge_name.upper()} (CORRECTED)")
    print(f"Validating: GraphRCA Algorithm")
    print("=" * 70)
    
    judge = JudgeClient(judge_name)
    incidents = load_incidents()
    
    print(f"Loaded {len(incidents)} incidents\n")
    
    if os.environ.get("SMOKE_TEST"):
        print("🔥 SMOKE TEST MODE ENABLED: Reducing to 5 incidents")
        incidents = incidents[:5]
    
    results = {"incidents": [], "by_category": {}, "total_incidents": len(incidents)}
    all_scores = []
    
    for idx, incident in enumerate(incidents):
        print(f"[{idx+1}/{len(incidents)}] {incident['id']} ({incident['category']})")
        
        logs = incident["logs"]
        
        # For GraphRCA, we only need 1 run because it's deterministic given the same logs
        # (Assuming the log parsing part is handled consistently or simple heuristic here)
        # Note: The 'parse_logs_to_chain' is a simple splitter. 
        # If we wanted full fidelity we'd use LogParser (LLM), but for this val script
        # keeping it deterministic is often preferred for algorithmic validation.
        
        prediction = run_graphrca(logs)
        
        score = judge.score(prediction, incident["root_cause"])
        scores = [score] # Treating as 1 run for deterministic algo
        
        avg_score = score
        correct = avg_score >= 0.7
        
        results["incidents"].append({
            "id": incident["id"],
            "prediction": prediction[:100] + "...",
            "avg_score": round(avg_score, 3),
            "correct": correct
        })
        all_scores.extend(scores)
        
        cat = incident["category"]
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"correct": 0, "total": 0}
        results["by_category"][cat]["total"] += 1
        if correct:
            results["by_category"][cat]["correct"] += 1
        
        print(f"  → Score: {avg_score:.2f} ({'✓' if correct else '✗'})")
    
    # Summary
    total_correct = sum(1 for i in results["incidents"] if i["correct"])
    
    results["overall_accuracy"] = round(total_correct / len(incidents), 4) if incidents else 0
    results["judge"] = judge_name
    
    print(f"\n{'='*70}")
    print(f"RESULTS ({judge_name.upper()}): {results['overall_accuracy']*100:.1f}% accuracy")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["qwen", "gpt", "groq", "all"], default="qwen")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    judges = ["qwen", "gpt", "groq"] if args.judge == "all" else [args.judge]
    
    for j in judges:
        try:
            results = run_validation(j)
            with open(output_dir / f"results_{j}.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Judge {j} failed: {e}")

if __name__ == "__main__":
    main()
