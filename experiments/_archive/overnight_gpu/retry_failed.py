#!/usr/bin/env python3
"""
Retry failed incidents from the main experiment run.
Merges results back into the original JSON files.
"""

import os
import sys
import json
import time
import random
import logging
import statistics
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "judge_model": "qwen3:32b",
    "embed_model": "nomic-embed-text",
    "temperature": 0.2,
    "timeout": 300.0,
    "incident_dir": PROJECT_ROOT / "data" / "real_incidents",
    "results_dir": SCRIPT_DIR / "results",
    "runs_per_test": 3,
    "random_seed": 42,
}

def setup_logging():
    logger = logging.getLogger("RetryFailed")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    return logger

def load_incident(incident_id: str):
    """Load a single incident by ID."""
    folder = CONFIG["incident_dir"] / incident_id
    if not folder.exists():
        return None
    
    try:
        with open(folder / "metadata.json") as f:
            metadata = json.load(f)
        with open(folder / "ground_truth.json") as f:
            ground_truth = json.load(f)
        with open(folder / "postmortem.md") as f:
            postmortem = f.read()
        
        logs_file = folder / "logs.txt"
        logs = logs_file.read_text() if logs_file.exists() else ""
        
        return {
            "id": folder.name,
            "metadata": metadata,
            "ground_truth": ground_truth,
            "postmortem": postmortem,
            "logs": logs,
            "root_cause": ground_truth.get("root_cause", ""),
            "category": ground_truth.get("category", metadata.get("category", "Unknown"))
        }
    except Exception as e:
        return None

def score_with_ollama(ollama_client, prediction: str, ground_truth: str, logger) -> float:
    """Score prediction using Qwen judge."""
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
            response = ollama_client.generate(
                model=CONFIG["judge_model"],
                prompt=prompt,
                options={"temperature": 0.0}
            )
            
            text = response["response"].strip()
            import re
            match = re.search(r'(0\.\d+|1\.0|0|1)', text)
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2)
    
    return 0.5

def evaluate_single_rca(incident, ollama_client, logger):
    """Evaluate RCA for a single incident."""
    logs = incident["logs"][:3000] if incident["logs"] else incident["postmortem"][:3000]
    ground_truth = incident["root_cause"]
    
    prompt = f"""Analyze these logs and identify the root cause of the incident.

LOGS:
{logs}

Provide a concise root cause analysis (1-2 sentences):"""

    scores = []
    for run in range(CONFIG["runs_per_test"]):
        try:
            response = ollama_client.generate(
                model=CONFIG["model"],
                prompt=prompt,
                options={"temperature": CONFIG["temperature"]}
            )
            prediction = response["response"].strip()
            score = score_with_ollama(ollama_client, prediction, ground_truth, logger)
            scores.append(score)
        except Exception as e:
            logger.warning(f"  Run {run+1} failed: {e}")
    
    if scores:
        avg_score = statistics.mean(scores)
        return {
            "correct": 1 if avg_score >= 0.5 else 0,
            "avg_score": round(avg_score, 4),
            "runs": len(scores)
        }
    return None

def main():
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("RETRY FAILED INCIDENTS")
    logger.info("=" * 70)
    
    # Load existing results
    rca_file = CONFIG["results_dir"] / "01_rca_accuracy.json"
    if not rca_file.exists():
        logger.error("No results file found!")
        return
    
    with open(rca_file) as f:
        results = json.load(f)
    
    # Find failed incidents
    failed = [r for r in results.get("incidents", []) if r.get("error") or r.get("runs", 3) < 3]
    
    if not failed:
        logger.info("No failed incidents found!")
        return
    
    logger.info(f"Found {len(failed)} failed incidents to retry")
    for f_inc in failed:
        logger.info(f"  - {f_inc['id']}")
    
    # Connect to Ollama
    import ollama
    ollama_client = ollama.Client(host=CONFIG["ollama_host"], timeout=CONFIG["timeout"])
    
    try:
        ollama_client.list()
        logger.info("✓ Ollama connected")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        return
    
    # Retry each failed incident
    retried = []
    for f_inc in failed:
        incident_id = f_inc["id"]
        logger.info(f"Retrying {incident_id}...")
        
        incident = load_incident(incident_id)
        if not incident:
            logger.warning(f"  Could not load {incident_id}")
            continue
        
        result = evaluate_single_rca(incident, ollama_client, logger)
        if result:
            retried.append({
                "id": incident_id,
                "category": incident["category"],
                **result
            })
            logger.info(f"  ✓ Score: {result['avg_score']:.2f}")
        else:
            logger.warning(f"  Still failed")
    
    # Merge results
    if retried:
        # Remove old failed entries
        failed_ids = {f["id"] for f in failed}
        results["incidents"] = [r for r in results["incidents"] if r["id"] not in failed_ids]
        
        # Add retried results
        results["incidents"].extend(retried)
        
        # Recalculate accuracy
        valid = [r for r in results["incidents"] if "correct" in r]
        results["accuracy"] = round(sum(r["correct"] for r in valid) / len(valid), 4) if valid else 0
        results["retried_at"] = datetime.now().isoformat()
        results["retried_count"] = len(retried)
        
        # Save
        with open(rca_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Retried {len(retried)} incidents")
        logger.info(f"New accuracy: {results['accuracy']*100:.1f}%")
    else:
        logger.info("No incidents successfully retried")

if __name__ == "__main__":
    main()
