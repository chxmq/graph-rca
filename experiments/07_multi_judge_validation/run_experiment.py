#!/usr/bin/env python3
"""
Multi-Judge RCA Validation Experiment

Tests RCA accuracy using multiple LLM judges for cross-validation:
  - Qwen 32B (local via Ollama)
  - GPT-4o-mini (OpenAI API)
  - Llama-70B (Groq API)

Usage:
    python run_experiment.py --judge qwen      # Local Qwen (default)
    python run_experiment.py --judge gpt       # OpenAI GPT-4o-mini
    python run_experiment.py --judge groq      # Groq Llama-70B
    python run_experiment.py --judge all       # Run all judges
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
        self._init_client()
    
    def _init_client(self):
        judge_type = self.judge_config["type"]
        
        if judge_type == "ollama":
            self.client = ollama.Client(host=CONFIG["ollama_host"])
            
        elif judge_type == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key'")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            
        elif judge_type == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set. Get free key at: https://console.groq.com/keys")
            from groq import Groq
            self.client = Groq(api_key=api_key)
    
    def score(self, prediction: str, ground_truth: str) -> float:
        """Score prediction against ground truth."""
        if not prediction or len(prediction.strip()) < 5:
            return 0.0
        
        prompt = f"""Compare these two root cause descriptions and rate their similarity from 0.0 to 1.0.

Ground Truth: {ground_truth}
Prediction: {prediction}

Scoring guide:
- 1.0: Same root cause identified
- 0.7-0.9: Right direction, missing details
- 0.4-0.6: Related but not the core issue
- 0.1-0.3: Tangentially related
- 0.0: Completely wrong

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
        
        if judge_type == "ollama":
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.0}
            )
            text = response["response"].strip()
            
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


def load_incidents() -> List[Dict]:
    """Load all real-world incidents."""
    incidents = []
    for folder in sorted(INCIDENT_DIR.glob("incident_*")):
        try:
            with open(folder / "ground_truth.json") as f:
                gt = json.load(f)
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()[:2000]
            logs_file = folder / "logs.txt"
            logs = logs_file.read_text()[:2000] if logs_file.exists() else ""
            
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


def get_prediction(ollama_client, logs: str) -> str:
    """Get RCA prediction from inference model."""
    prompt = f"Identify the root cause from these logs:\n{logs}\n\nRoot Cause:"
    response = ollama_client.generate(
        model=CONFIG["model"],
        prompt=prompt,
        options={"temperature": CONFIG["temperature"]}
    )
    return response["response"].strip()


def run_validation(judge_name: str) -> Dict:
    """Run multi-judge validation for a specific judge."""
    print("=" * 70)
    print(f"MULTI-JUDGE VALIDATION: {judge_name.upper()}")
    print(f"Judge model: {CONFIG['judges'][judge_name]['model']}")
    print("=" * 70)
    
    # Initialize
    ollama_client = ollama.Client(host=CONFIG["ollama_host"])
    judge = JudgeClient(judge_name)
    incidents = load_incidents()
    
    print(f"Loaded {len(incidents)} incidents\n")
    
    results = {"incidents": [], "by_category": {}}
    all_scores = []
    
    for idx, incident in enumerate(incidents):
        print(f"[{idx+1}/{len(incidents)}] {incident['id']} ({incident['category']})")
        
        logs = incident["logs"] if incident["logs"] else incident["postmortem"]
        
        scores = []
        for run in range(CONFIG["runs_per_incident"]):
            try:
                prediction = get_prediction(ollama_client, logs)
                score = judge.score(prediction, incident["root_cause"])
                scores.append(score)
            except Exception as e:
                print(f"  Run {run+1} failed: {e}")
        
        if scores:
            avg_score = statistics.mean(scores)
            correct = avg_score >= 0.7
            
            results["incidents"].append({
                "id": incident["id"],
                "category": incident["category"],
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
    results["total_incidents"] = len(incidents)
    results["overall_accuracy"] = round(total_correct / len(incidents), 4) if incidents else 0
    results["avg_score"] = round(statistics.mean(all_scores), 4) if all_scores else 0
    results["judge"] = judge_name
    results["judge_model"] = CONFIG["judges"][judge_name]["model"]
    results["timestamp"] = datetime.now().isoformat()
    
    print(f"\n{'='*70}")
    print(f"RESULTS ({judge_name.upper()}): {results['overall_accuracy']*100:.1f}% accuracy")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Judge RCA Validation")
    parser.add_argument("--judge", choices=["qwen", "gpt", "groq", "all"], default="qwen",
                       help="Which judge to use (default: qwen)")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    judges_to_run = ["qwen", "gpt", "groq"] if args.judge == "all" else [args.judge]
    
    all_results = {}
    
    for judge_name in judges_to_run:
        try:
            results = run_validation(judge_name)
            all_results[judge_name] = results
            
            # Save individual results
            with open(output_dir / f"results_{judge_name}.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"✓ Saved to data/results_{judge_name}.json\n")
            
        except Exception as e:
            print(f"❌ {judge_name} failed: {e}\n")
    
    # Save combined summary
    if len(all_results) > 1:
        summary = {
            "judges": list(all_results.keys()),
            "accuracy": {j: r["overall_accuracy"] for j, r in all_results.items()},
            "timestamp": datetime.now().isoformat()
        }
        with open(output_dir / "multi_judge_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("=" * 70)
        print("MULTI-JUDGE SUMMARY")
        print("=" * 70)
        for j, acc in summary["accuracy"].items():
            print(f"  {j:8s}: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
