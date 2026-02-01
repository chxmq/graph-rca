#!/usr/bin/env python3
"""
Documentation Ablation Experiment
Tests impact of documentation on RCA accuracy using real-world incidents.

Compares:
- full_docs: All postmortems in RAG corpus
- half_docs: 50% of postmortems
- no_docs: No documentation (LLM knowledge only)
"""

import os
import sys
import json
import random
import statistics
from pathlib import Path
from datetime import datetime

if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "judge_model": "qwen3:32b",
    "temperature": 0.2,
    "runs_per_incident": 3,
    "random_seed": 42,
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"


def load_incidents() -> list:
    """Load real-world incidents."""
    incidents = []
    if not INCIDENT_DIR.exists():
        print(f"⚠ {INCIDENT_DIR} not found")
        return []
    
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


def score_prediction(client: ollama.Client, prediction: str, ground_truth: str) -> float:
    """Score prediction using judge model."""
    if not prediction or len(prediction.strip()) < 5:
        return 0.0
    
    prompt = f"""Rate similarity 0.0-1.0:
Ground Truth: {ground_truth}
Prediction: {prediction}
Score (just the number):"""

    try:
        response = client.generate(
            model=CONFIG["judge_model"],
            prompt=prompt,
            options={"temperature": 0.0}
        )
        import re
        match = re.search(r'(0\.\d+|1\.0|0|1)', response["response"])
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.5


def run_doc_ablation(client: ollama.Client) -> dict:
    """Documentation ablation using real incidents."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Documentation Ablation (Real Incidents)")
    print("=" * 70)
    
    incidents = load_incidents()
    if not incidents:
        print("No incidents found!")
        return {}
    
    print(f"Loaded {len(incidents)} real incidents")
    
    # Split: use some for docs, test on rest
    random.seed(CONFIG["random_seed"])
    shuffled = incidents.copy()
    random.shuffle(shuffled)
    
    split = len(shuffled) // 2
    doc_pool = shuffled[:split]  # Incidents to use as documentation
    test_set = shuffled[split:]  # Incidents to test on
    
    print(f"Doc pool: {len(doc_pool)}, Test set: {len(test_set)}")
    
    configs = {
        "full_docs": doc_pool,
        "half_docs": doc_pool[:len(doc_pool)//2],
        "no_docs": [],
    }
    
    results = {"configs": {}, "test_size": len(test_set)}
    
    for config_name, docs in configs.items():
        print(f"\nConfig: {config_name} ({len(docs)} docs)")
        scores = []
        
        # Build documentation context
        doc_context = ""
        if docs:
            doc_texts = [f"Incident: {d['category']}\nRoot Cause: {d['root_cause']}\n{d['postmortem'][:500]}" 
                        for d in docs[:10]]  # Limit context size
            doc_context = "\n---\n".join(doc_texts)
        
        for idx, test_case in enumerate(test_set[:20]):  # Test on subset for speed
            for run in range(CONFIG["runs_per_incident"]):
                input_text = test_case["logs"][:1500] if test_case["logs"] else test_case["postmortem"][:1500]
                
                if doc_context:
                    prompt = f"""Historical Incidents:
{doc_context}

Current Logs:
{input_text}

Based on the historical incidents, identify the root cause:"""
                else:
                    prompt = f"""Logs:
{input_text}

Identify the root cause:"""
                
                try:
                    response = client.generate(
                        model=CONFIG["model"], 
                        prompt=prompt, 
                        options={"temperature": CONFIG["temperature"]}
                    )
                    prediction = response["response"].strip()
                    score = score_prediction(client, prediction, test_case["root_cause"])
                    scores.append(score)
                except Exception as e:
                    print(f"  Error: {e}")
            
            if (idx + 1) % 5 == 0:
                print(f"  Progress: {idx+1}/{min(len(test_set), 20)}")
        
        accuracy = statistics.mean(scores) if scores else 0
        results["configs"][config_name] = {
            "accuracy": round(accuracy, 4),
            "num_docs": len(docs),
            "tests": len(scores)
        }
        print(f"  {config_name}: {accuracy:.1%}")
    
    results["timestamp"] = datetime.now().isoformat()
    return results


def main():
    client = ollama.Client(host=CONFIG["ollama_host"])
    results = run_doc_ablation(client)
    
    if not results:
        return
    
    output_path = Path(__file__).parent / "data" / "doc_ablation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    full = results["configs"]["full_docs"]["accuracy"]
    none = results["configs"]["no_docs"]["accuracy"]
    print(f"\nDifference (full - none): {(full - none) * 100:.1f} percentage points")


if __name__ == "__main__":
    main()
