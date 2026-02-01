#!/usr/bin/env python3
"""
RAG Real-World Evaluation Experiment
Compares baseline (no RAG) vs RAG accuracy on real-world incidents.
From run_overnight.py - Experiment 2
"""

import os
import sys
import json
import time
import re
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
    "embed_model": "nomic-embed-text",
    "temperature": 0.2,
    "train_ratio": 0.75,
    "random_seed": 42,
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"


def load_incidents() -> list:
    """Load real-world incidents."""
    incidents = []
    if not INCIDENT_DIR.exists():
        print(f"Warning: {INCIDENT_DIR} not found, using synthetic")
        return get_synthetic_incidents()
    
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
    
    return incidents if incidents else get_synthetic_incidents()


def get_synthetic_incidents() -> list:
    """Fallback synthetic incidents."""
    return [
        {"id": "syn_001", "root_cause": "Connection pool exhausted", "category": "Database",
         "logs": "ERROR [db-pool] Connection exhausted", "postmortem": "DB connection pool failed"},
        {"id": "syn_002", "root_cause": "Certificate expired", "category": "Security",
         "logs": "ERROR [ssl] Certificate expired", "postmortem": "TLS handshake failed"},
    ]


def get_prediction(client: ollama.Client, logs: str, context: str = "") -> str:
    """Get RCA prediction."""
    if context:
        prompt = f"Using this context: {context}\n\nIdentify root cause:\n{logs}\n\nRoot Cause:"
    else:
        prompt = f"Identify root cause:\n{logs}\n\nRoot Cause:"
    
    response = client.generate(
        model=CONFIG["model"],
        prompt=prompt,
        options={"temperature": CONFIG["temperature"]}
    )
    return response["response"].strip()


def find_similar(client: ollama.Client, test_case: dict, train_set: list) -> dict:
    """Find most similar incident using embeddings."""
    try:
        test_emb = client.embeddings(
            model=CONFIG["embed_model"],
            prompt=f"{test_case['category']} {test_case['root_cause']}"
        )["embedding"]
        
        best = train_set[0]
        best_sim = -1
        
        for train in train_set[:20]:  # Limit for speed
            train_emb = client.embeddings(
                model=CONFIG["embed_model"],
                prompt=f"{train['category']} {train['root_cause']}"
            )["embedding"]
            
            sim = sum(a*b for a,b in zip(test_emb, train_emb))
            if sim > best_sim:
                best_sim = sim
                best = train
        
        return best
    except:
        return train_set[0]


def score_with_judge(client: ollama.Client, prediction: str, ground_truth: str) -> float:
    """Score using judge model."""
    if not prediction:
        return 0.0
    
    prompt = f"""Rate similarity 0.0-1.0:
Ground Truth: {ground_truth}
Prediction: {prediction}
Score:"""

    try:
        response = client.generate(
            model=CONFIG["judge_model"],
            prompt=prompt,
            options={"temperature": 0.0}
        )
        match = re.search(r'(0\.\d+|1\.0|0|1)', response["response"])
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.5


def run_rag_comparison(client: ollama.Client) -> dict:
    """Compare baseline vs RAG."""
    print("=" * 70)
    print("EXPERIMENT 8: RAG vs Baseline Comparison")
    print("=" * 70)
    
    incidents = load_incidents()
    
    # Split train/test
    random.seed(CONFIG["random_seed"])
    shuffled = incidents.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * CONFIG["train_ratio"])
    train_set, test_set = shuffled[:split], shuffled[split:]
    
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    results = {"train_size": len(train_set), "test_size": len(test_set), "tests": []}
    
    for idx, test_case in enumerate(test_set):
        print(f"\n[{idx+1}/{len(test_set)}] {test_case['id']}")
        
        input_text = test_case["logs"][:2000] if test_case["logs"] else test_case["postmortem"][:2000]
        
        try:
            # Baseline (no RAG)
            baseline_pred = get_prediction(client, input_text, context="")
            baseline_score = score_with_judge(client, baseline_pred, test_case["root_cause"])
            
            # RAG - find similar incident
            similar = find_similar(client, test_case, train_set)
            rag_context = f"Historical: {similar['category']} - {similar['root_cause']}"
            rag_pred = get_prediction(client, input_text, context=rag_context)
            rag_score = score_with_judge(client, rag_pred, test_case["root_cause"])
            
            results["tests"].append({
                "id": test_case["id"],
                "baseline_score": round(baseline_score, 3),
                "rag_score": round(rag_score, 3),
                "improvement": round(rag_score - baseline_score, 3)
            })
            
            print(f"  Baseline: {baseline_score:.2f}, RAG: {rag_score:.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results["tests"].append({"id": test_case["id"], "error": str(e)})
    
    # Aggregate
    valid = [t for t in results["tests"] if "baseline_score" in t]
    results["baseline_avg"] = round(statistics.mean([t["baseline_score"] for t in valid]), 4) if valid else 0
    results["rag_avg"] = round(statistics.mean([t["rag_score"] for t in valid]), 4) if valid else 0
    results["improvement"] = round(results["rag_avg"] - results["baseline_avg"], 4)
    results["timestamp"] = datetime.now().isoformat()
    
    print(f"\n✓ Baseline: {results['baseline_avg']*100:.1f}%, RAG: {results['rag_avg']*100:.1f}%")
    
    return results


def main():
    client = ollama.Client(host=CONFIG["ollama_host"])
    results = run_rag_comparison(client)
    
    output_path = Path(__file__).parent / "data" / "rag_comparison_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
