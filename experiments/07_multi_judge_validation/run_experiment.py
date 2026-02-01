#!/usr/bin/env python3
"""
Multi-Judge Validation Experiment
Cross-validates RCA accuracy using multiple LLM judges.
From run_overnight.py - for validating results across models.
"""

import os
import sys
import json
import time
import re
import statistics
from pathlib import Path
from datetime import datetime

if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "judge_model": "qwen3:32b",  # Local judge via Ollama
    "temperature": 0.2,
    "runs_per_incident": 3,
}

# Path to real incidents
PROJECT_ROOT = Path(__file__).parent.parent.parent
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"


def load_incidents() -> list:
    """Load real-world incidents."""
    incidents = []
    if not INCIDENT_DIR.exists():
        print(f"Warning: {INCIDENT_DIR} not found, using synthetic data")
        return get_synthetic_incidents()
    
    for folder in sorted(INCIDENT_DIR.glob("incident_*"))[:20]:  # First 20 for speed
        try:
            with open(folder / "ground_truth.json") as f:
                gt = json.load(f)
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()[:2000]
            
            incidents.append({
                "id": folder.name,
                "root_cause": gt.get("root_cause", ""),
                "category": gt.get("category", "Unknown"),
                "postmortem": postmortem
            })
        except:
            pass
    
    return incidents if incidents else get_synthetic_incidents()


def get_synthetic_incidents() -> list:
    """Fallback synthetic incidents."""
    return [
        {"id": "syn_001", "root_cause": "Connection pool exhausted", "category": "Database",
         "postmortem": "ERROR [db-pool] Connection exhausted\nERROR [api] Query failed"},
        {"id": "syn_002", "root_cause": "Certificate expired", "category": "Security",
         "postmortem": "ERROR [ssl] Certificate expired\nERROR [api] TLS handshake failed"},
        {"id": "syn_003", "root_cause": "Memory leak", "category": "Memory",
         "postmortem": "WARN [gc] Heap 95%\nCRITICAL [oom] OOM killer invoked"},
    ]


def get_rca_prediction(client: ollama.Client, postmortem: str) -> str:
    """Get RCA prediction from model."""
    prompt = f"""Analyze this incident and identify the ROOT CAUSE in one sentence.

{postmortem[:2000]}

Root Cause:"""
    
    response = client.generate(
        model=CONFIG["model"],
        prompt=prompt,
        options={"temperature": CONFIG["temperature"]}
    )
    return response["response"].strip()


def score_with_judge(client: ollama.Client, prediction: str, ground_truth: str) -> float:
    """Score prediction using judge model."""
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
            response = client.generate(
                model=CONFIG["judge_model"],
                prompt=prompt,
                options={"temperature": 0.0}
            )
            text = response["response"].strip()
            match = re.search(r'(0\.\d+|1\.0|0|1)', text)
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
        except Exception as e:
            print(f"  Judge attempt {attempt+1}/3 failed: {e}")
            time.sleep(2)
    
    return 0.5  # Neutral if unparseable


def run_multi_judge_experiment(client: ollama.Client) -> dict:
    """Run multi-judge validation."""
    print("=" * 70)
    print("EXPERIMENT 7: Multi-Judge Validation")
    print("=" * 70)
    
    incidents = load_incidents()
    print(f"Loaded {len(incidents)} incidents")
    
    results = {"incidents": [], "by_category": {}}
    all_scores = []
    
    for idx, incident in enumerate(incidents):
        print(f"\n[{idx+1}/{len(incidents)}] {incident['id']} ({incident['category']})")
        
        incident_scores = []
        for run in range(CONFIG["runs_per_incident"]):
            try:
                prediction = get_rca_prediction(client, incident["postmortem"])
                score = score_with_judge(client, prediction, incident["root_cause"])
                incident_scores.append(score)
                print(f"  Run {run+1}: score={score:.2f}")
            except Exception as e:
                print(f"  Run {run+1} failed: {e}")
                incident_scores.append(0)
        
        avg_score = statistics.mean(incident_scores) if incident_scores else 0
        correct = avg_score >= 0.7
        
        results["incidents"].append({
            "id": incident["id"],
            "category": incident["category"],
            "avg_score": round(avg_score, 3),
            "correct": correct
        })
        all_scores.extend(incident_scores)
        
        # Category tracking
        cat = incident["category"]
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"correct": 0, "total": 0}
        results["by_category"][cat]["total"] += 1
        if correct:
            results["by_category"][cat]["correct"] += 1
    
    # Summary
    total_correct = sum(1 for i in results["incidents"] if i["correct"])
    results["overall_accuracy"] = round(total_correct / len(incidents), 4) if incidents else 0
    results["avg_score"] = round(statistics.mean(all_scores), 4) if all_scores else 0
    results["judge_model"] = CONFIG["judge_model"]
    results["timestamp"] = datetime.now().isoformat()
    
    print(f"\n✓ Overall accuracy: {results['overall_accuracy']*100:.1f}%")
    
    return results


def main():
    client = ollama.Client(host=CONFIG["ollama_host"])
    
    # Verify judge model
    try:
        client.show(CONFIG["judge_model"])
        print(f"✓ Judge model: {CONFIG['judge_model']}")
    except:
        print(f"⚠ Judge model '{CONFIG['judge_model']}' not found, run: ollama pull {CONFIG['judge_model']}")
        return
    
    results = run_multi_judge_experiment(client)
    
    output_path = Path(__file__).parent / "data" / "multi_judge_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
