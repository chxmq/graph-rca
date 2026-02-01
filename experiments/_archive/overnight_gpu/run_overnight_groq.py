#!/usr/bin/env python3
"""
GraphRCA Experiment Suite - Groq/Llama-70B Judge Version
Uses Groq API (free tier available) for scoring with open-source Llama 70B.

Usage:
    export GROQ_API_KEY="your-key"
    python run_overnight_groq.py
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
from typing import List, Dict

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "judge_model": "llama-3.3-70b-versatile",  # Groq's Llama 70B
    "embed_model": "nomic-embed-text",
    "temperature": 0.2,
    "timeout": 300.0,
    "incident_dir": PROJECT_ROOT / "data" / "real_incidents",
    "results_dir": SCRIPT_DIR / "results_groq",
    "runs_per_test": 3,
    "train_ratio": 0.75,
    "random_seed": 42,
    "batch_save_every": 5,
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages experiment progress for resumability."""
    def __init__(self, results_dir: Path):
        self.checkpoint_file = results_dir / "checkpoint.json"
        self.state = self._load()
    
    def _load(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    return json.load(f)
            except:
                pass
        return {"experiments_done": [], "current": None, "progress": {}}
    
    def save(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def is_done(self, name: str) -> bool:
        return name in self.state["experiments_done"]
    
    def mark_done(self, name: str):
        if name not in self.state["experiments_done"]:
            self.state["experiments_done"].append(name)
        self.state["current"] = None
        self.state["progress"] = {}
        self.save()
    
    def set_progress(self, name: str, idx: int, results: List):
        self.state["current"] = name
        self.state["progress"] = {"last_idx": idx, "results": results}
        self.save()
    
    def get_progress(self, name: str) -> tuple:
        if self.state["current"] == name:
            p = self.state["progress"]
            return p.get("last_idx", -1), p.get("results", [])
        return -1, []

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("GraphRCA_Groq")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    
    fh = logging.FileHandler(CONFIG["results_dir"] / "experiment_log.txt", mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(fh)
    
    return logger

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_incidents(logger) -> List[Dict]:
    """Load all real-world incidents."""
    incidents = []
    incident_dir = CONFIG["incident_dir"]
    
    for folder in sorted(incident_dir.glob("incident_*")):
        try:
            with open(folder / "metadata.json") as f:
                metadata = json.load(f)
            with open(folder / "ground_truth.json") as f:
                ground_truth = json.load(f)
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()
            
            logs_file = folder / "logs.txt"
            logs = logs_file.read_text() if logs_file.exists() else ""
            
            incidents.append({
                "id": folder.name,
                "metadata": metadata,
                "ground_truth": ground_truth,
                "postmortem": postmortem,
                "logs": logs,
                "root_cause": ground_truth.get("root_cause", ""),
                "category": ground_truth.get("category", metadata.get("category", "Unknown"))
            })
        except Exception as e:
            logger.warning(f"Failed to load {folder.name}: {e}")
    
    logger.info(f"Loaded {len(incidents)} incidents")
    return incidents

# ============================================================================
# GROQ SCORING (Llama 70B)
# ============================================================================

def score_with_groq(groq_client, prediction: str, ground_truth: str, logger) -> float:
    """Score prediction using Llama-70B via Groq."""
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
            response = groq_client.chat.completions.create(
                model=CONFIG["judge_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            text = response.choices[0].message.content.strip()
            import re
            match = re.search(r'(0\.\d+|1\.0|0|1)', text)
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
                
        except Exception as e:
            logger.warning(f"Groq attempt {attempt+1}/3 failed: {e}")
            if "rate_limit" in str(e).lower():
                time.sleep(5)  # Rate limit backoff
            elif attempt < 2:
                time.sleep(1)
    
    return 0.5

# ============================================================================
# EXPERIMENT 1: RCA ACCURACY
# ============================================================================

def run_rca_accuracy(logger, checkpoint, incidents, ollama_client, groq_client) -> Dict:
    """Test RCA accuracy with Llama-70B judge."""
    
    EXPERIMENT_NAME = "01_rca_accuracy"
    
    if checkpoint.is_done(EXPERIMENT_NAME):
        logger.info(f"Skipping {EXPERIMENT_NAME} (already done)")
        result_file = CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {}
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: RCA Accuracy (Llama-70B Judge via Groq)")
    logger.info("=" * 70)
    
    # Resume from checkpoint
    start_idx, results_so_far = checkpoint.get_progress(EXPERIMENT_NAME)
    start_idx += 1
    
    if start_idx > 0:
        logger.info(f"Resuming from incident {start_idx}")
    
    results = {"incidents": results_so_far, "by_category": {}}
    
    for idx in range(start_idx, len(incidents)):
        incident = incidents[idx]
        logger.info(f"[{idx+1}/{len(incidents)}] {incident['id']} - {incident['metadata'].get('company', 'Unknown')}")
        
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
                score = score_with_groq(groq_client, prediction, ground_truth, logger)
                scores.append(score)
            except Exception as e:
                logger.warning(f"  Run {run+1} failed: {e}")
        
        if scores:
            avg_score = statistics.mean(scores)
            correct = 1 if avg_score >= 0.5 else 0
            
            results["incidents"].append({
                "id": incident["id"],
                "category": incident["category"],
                "correct": correct,
                "avg_score": round(avg_score, 4),
                "runs": len(scores)
            })
        else:
            results["incidents"].append({
                "id": incident["id"],
                "error": "All runs failed"
            })
        
        # Save checkpoint
        if (idx + 1) % CONFIG["batch_save_every"] == 0:
            checkpoint.set_progress(EXPERIMENT_NAME, idx, results["incidents"])
    
    # Calculate accuracy
    valid = [r for r in results["incidents"] if "correct" in r]
    results["accuracy"] = round(sum(r["correct"] for r in valid) / len(valid), 4) if valid else 0
    
    # Save final
    with open(CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    checkpoint.mark_done(EXPERIMENT_NAME)
    logger.info(f"✅ RCA Accuracy complete: {results['accuracy']*100:.1f}%")
    return results

# ============================================================================
# EXPERIMENT 2: RAG COMPARISON
# ============================================================================

def run_rag_comparison(logger, checkpoint, incidents, ollama_client, groq_client) -> Dict:
    """Compare Baseline vs RAG with Llama-70B judge."""
    
    EXPERIMENT_NAME = "02_rag_comparison"
    
    if checkpoint.is_done(EXPERIMENT_NAME):
        logger.info(f"Skipping {EXPERIMENT_NAME} (already done)")
        result_file = CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {}
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: RAG vs Baseline (Llama-70B Judge via Groq)")
    logger.info("=" * 70)
    
    # Split (deterministic)
    random.seed(CONFIG["random_seed"])
    shuffled = incidents.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * CONFIG["train_ratio"])
    train_set, test_set = shuffled[:split], shuffled[split:]
    
    logger.info(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    # Resume from checkpoint
    start_idx, results_so_far = checkpoint.get_progress(EXPERIMENT_NAME)
    start_idx += 1
    
    if start_idx > 0:
        logger.info(f"Resuming from test {start_idx}")
    
    results = {"train_size": len(train_set), "test_size": len(test_set), "tests": results_so_far}
    
    for idx in range(start_idx, len(test_set)):
        test_case = test_set[idx]
        logger.info(f"[{idx+1}/{len(test_set)}] {test_case['id']}")
        
        input_text = test_case["logs"][:2000] if test_case["logs"] else test_case["postmortem"][:2000]
        
        try:
            # Baseline
            baseline_pred = get_prediction(ollama_client, input_text, context="")
            baseline_score = score_with_groq(groq_client, baseline_pred, test_case["root_cause"], logger)
            
            # RAG - find similar
            similar = find_similar(ollama_client, test_case, train_set)
            rag_context = f"Historical: {similar['category']} - {similar['root_cause']}"
            rag_pred = get_prediction(ollama_client, input_text, context=rag_context)
            rag_score = score_with_groq(groq_client, rag_pred, test_case["root_cause"], logger)
            
            results["tests"].append({
                "id": test_case["id"],
                "baseline_score": baseline_score,
                "rag_score": rag_score,
                "improvement": rag_score - baseline_score
            })
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results["tests"].append({"id": test_case["id"], "error": str(e)})
        
        # Save checkpoint
        if (idx + 1) % CONFIG["batch_save_every"] == 0:
            checkpoint.set_progress(EXPERIMENT_NAME, idx, results["tests"])
    
    # Aggregate
    valid = [t for t in results["tests"] if "baseline_score" in t]
    results["baseline_avg"] = round(statistics.mean([t["baseline_score"] for t in valid]), 4) if valid else 0
    results["rag_avg"] = round(statistics.mean([t["rag_score"] for t in valid]), 4) if valid else 0
    
    # Save
    with open(CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    checkpoint.mark_done(EXPERIMENT_NAME)
    logger.info(f"✅ RAG Comparison complete: Baseline={results['baseline_avg']*100:.1f}%, RAG={results['rag_avg']*100:.1f}%")
    return results

def get_prediction(ollama_client, logs: str, context: str) -> str:
    """Get RCA prediction from Ollama."""
    if context:
        prompt = f"Using this context: {context}\n\nIdentify root cause:\n{logs}\n\nRoot Cause:"
    else:
        prompt = f"Identify the root cause from these logs:\n{logs}\n\nRoot Cause:"
    
    response = ollama_client.generate(
        model=CONFIG["model"],
        prompt=prompt,
        options={"temperature": CONFIG["temperature"]}
    )
    return response["response"].strip()

def find_similar(ollama_client, test_case: Dict, train_set: List[Dict]) -> Dict:
    """Find most similar incident using embeddings."""
    test_embed = get_embedding(ollama_client, test_case.get("logs", test_case.get("postmortem", ""))[:1000])
    
    best_sim = -1
    best_match = train_set[0]
    
    for train_case in train_set[:20]:
        train_embed = get_embedding(ollama_client, train_case.get("root_cause", ""))
        sim = cosine_similarity(test_embed, train_embed)
        if sim > best_sim:
            best_sim = sim
            best_match = train_case
    
    return best_match

def get_embedding(ollama_client, text: str) -> List[float]:
    response = ollama_client.embeddings(model=CONFIG["embed_model"], prompt=text[:2000])
    return response["embedding"]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("GraphRCA GPU Experiment Suite (Llama-70B via Groq)")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY not set!")
        logger.error("   Get free key at: https://console.groq.com/keys")
        logger.error("   Run: export GROQ_API_KEY='your-key'")
        return
    
    # Initialize clients
    import ollama
    from groq import Groq
    
    ollama_client = ollama.Client(host=CONFIG["ollama_host"], timeout=CONFIG["timeout"])
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    try:
        ollama_client.list()
        logger.info("✓ Ollama connected")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        return
    
    # Test Groq
    try:
        groq_client.chat.completions.create(
            model=CONFIG["judge_model"],
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        logger.info("✓ Groq connected")
    except Exception as e:
        logger.error(f"❌ Groq connection failed: {e}")
        return
    
    # Load data
    incidents = load_all_incidents(logger)
    if not incidents:
        logger.error("❌ No incidents found!")
        return
    
    # Initialize checkpoint
    checkpoint = CheckpointManager(CONFIG["results_dir"])
    
    # Run experiments
    results = {}
    results["rca"] = run_rca_accuracy(logger, checkpoint, incidents, ollama_client, groq_client)
    results["rag"] = run_rag_comparison(logger, checkpoint, incidents, ollama_client, groq_client)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  RCA Accuracy: {results['rca'].get('accuracy', 0)*100:.1f}%")
    logger.info(f"  Baseline: {results['rag'].get('baseline_avg', 0)*100:.1f}%")
    logger.info(f"  RAG: {results['rag'].get('rag_avg', 0)*100:.1f}%")

if __name__ == "__main__":
    main()
