#!/usr/bin/env python3
"""
RAG Real-World Evaluation with Multi-Judge Validation

Compares baseline (no RAG) vs RAG accuracy on real-world incidents.
Uses multiple LLM judges for cross-validation: Qwen, Llama-70B, GPT-4o-mini.

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
import re
import random
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
    "embed_model": "nomic-embed-text",
    "temperature": 0.2,
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
    if not INCIDENT_DIR.exists():
        print(f"⚠ {INCIDENT_DIR} not found, using synthetic data")
        return get_synthetic_incidents()
    
    print(f"Loading incidents from {INCIDENT_DIR}...")
    for folder in sorted(INCIDENT_DIR.glob("incident_*")):
        try:
            with open(folder / "ground_truth.json") as f:
                gt = json.load(f)
            with open(folder / "metadata.json") as f:
                meta = json.load(f)
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()[:2000]
            logs_file = folder / "logs.txt"
            logs = logs_file.read_text()[:2000] if logs_file.exists() else ""
            
            incidents.append({
                "id": folder.name,
                "company": meta.get("company", "Unknown"),
                "root_cause": gt.get("root_cause", ""),
                "category": gt.get("category", "Unknown"),
                "postmortem": postmortem,
                "logs": logs
            })
        except:
            pass
    
    print(f"Loaded {len(incidents)} incidents")
    return incidents if incidents else get_synthetic_incidents()


def get_synthetic_incidents() -> List[Dict]:
    """Fallback synthetic incidents."""
    return [
        {"id": "syn_001", "company": "ACME", "root_cause": "Connection pool exhausted", "category": "Database",
         "logs": "ERROR [db-pool] Connection exhausted", "postmortem": "DB connection pool failed"},
        {"id": "syn_002", "company": "ACME", "root_cause": "Certificate expired", "category": "Security",
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
        
        for train in train_set[:20]:
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


def run_experiment(ollama_client: ollama.Client, judge: JudgeClient) -> dict:
    """Compare baseline vs RAG using a specific judge."""
    print("=" * 70)
    print(f"EXPERIMENT: RAG vs Baseline Comparison")
    print(f"Judge: {judge.judge_name.upper()} ({judge.judge_config['model']})")
    print(f"Config: {CONFIG['train_ratio']*100:.0f}% train / {(1-CONFIG['train_ratio'])*100:.0f}% test split")
    print("=" * 70)
    
    incidents = load_incidents()
    
    # Split train/test
    random.seed(CONFIG["random_seed"])
    shuffled = incidents.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * CONFIG["train_ratio"])
    train_set, test_set = shuffled[:split], shuffled[split:]
    
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    results = {
        "judge": judge.judge_name,
        "judge_model": judge.judge_config["model"],
        "train_size": len(train_set), 
        "test_size": len(test_set), 
        "tests": []
    }
    
    for idx, test_case in enumerate(test_set):
        print(f"\n[{idx+1}/{len(test_set)}] {test_case['id']} ({test_case.get('company', 'Unknown')})")
        
        input_text = test_case["logs"][:2000] if test_case["logs"] else test_case["postmortem"][:2000]
        
        try:
            # Baseline (no RAG)
            baseline_pred = get_prediction(ollama_client, input_text, context="")
            baseline_score = judge.score(baseline_pred, test_case["root_cause"])
            
            # RAG - find similar incident
            similar = find_similar(ollama_client, test_case, train_set)
            rag_context = f"Historical: {similar['category']} - {similar['root_cause']}"
            rag_pred = get_prediction(ollama_client, input_text, context=rag_context)
            rag_score = judge.score(rag_pred, test_case["root_cause"])
            
            results["tests"].append({
                "id": test_case["id"],
                "company": test_case.get("company", "Unknown"),
                "baseline_score": round(baseline_score, 3),
                "rag_score": round(rag_score, 3),
                "improvement": round(rag_score - baseline_score, 3)
            })
            
            print(f"  Baseline: {baseline_score:.2f}, RAG: {rag_score:.2f}, Δ: {rag_score-baseline_score:+.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results["tests"].append({"id": test_case["id"], "error": str(e)})
    
    # Aggregate
    valid = [t for t in results["tests"] if "baseline_score" in t]
    results["baseline_avg"] = round(statistics.mean([t["baseline_score"] for t in valid]), 4) if valid else 0
    results["rag_avg"] = round(statistics.mean([t["rag_score"] for t in valid]), 4) if valid else 0
    results["improvement"] = round(results["rag_avg"] - results["baseline_avg"], 4)
    results["timestamp"] = datetime.now().isoformat()
    
    print(f"\n{'='*70}")
    print(f"RESULTS ({judge.judge_name.upper()}):")
    print(f"  Baseline: {results['baseline_avg']*100:.1f}%")
    print(f"  RAG:      {results['rag_avg']*100:.1f}%")
    print(f"  Change:   {results['improvement']*100:+.1f}%")
    print(f"{'='*70}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG vs Baseline with Multi-Judge Validation")
    parser.add_argument("--judge", choices=["qwen", "gpt", "groq", "all"], 
                        default="qwen", help="LLM judge to use")
    args = parser.parse_args()
    
    ollama_client = ollama.Client(host=CONFIG["ollama_host"])
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    judges_to_run = list(CONFIG["judges"].keys()) if args.judge == "all" else [args.judge]
    all_results = {}
    
    for judge_name in judges_to_run:
        print(f"\n{'#'*70}")
        print(f"# Running with {judge_name.upper()} judge")
        print(f"{'#'*70}")
        
        try:
            judge = JudgeClient(judge_name)
            results = run_experiment(ollama_client, judge)
            all_results[judge_name] = results
            
            # Save individual judge results
            out_file = output_dir / f"rag_comparison_{judge_name}.json"
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved to {out_file}")
            
        except Exception as e:
            print(f"✗ Failed with {judge_name}: {e}")
    
    # Save combined results if running all judges
    if len(all_results) > 1:
        summary = {
            "judges": list(all_results.keys()),
            "baseline_by_judge": {k: v["baseline_avg"] for k, v in all_results.items()},
            "rag_by_judge": {k: v["rag_avg"] for k, v in all_results.items()},
            "change_by_judge": {k: v["improvement"] for k, v in all_results.items()},
            "avg_baseline": statistics.mean([v["baseline_avg"] for v in all_results.values()]),
            "avg_rag": statistics.mean([v["rag_avg"] for v in all_results.values()]),
            "avg_change": statistics.mean([v["improvement"] for v in all_results.values()]),
            "timestamp": datetime.now().isoformat()
        }
        
        combined_file = output_dir / "rag_comparison_all_judges.json"
        with open(combined_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print("MULTI-JUDGE SUMMARY:")
        print(f"{'='*70}")
        for judge_name, data in all_results.items():
            print(f"  {judge_name.upper():6s}: Baseline {data['baseline_avg']*100:.1f}% → RAG {data['rag_avg']*100:.1f}% ({data['improvement']*100:+.1f}%)")
        print(f"{'='*70}")
        print(f"  AVERAGE: Baseline {summary['avg_baseline']*100:.1f}% → RAG {summary['avg_rag']*100:.1f}% ({summary['avg_change']*100:+.1f}%)")
        print(f"{'='*70}")
        print(f"✓ Combined results saved to {combined_file}")


if __name__ == "__main__":
    main()
