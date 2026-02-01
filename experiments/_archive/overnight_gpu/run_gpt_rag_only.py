#!/usr/bin/env python3
"""
GPT RAG Comparison Only - With Checkpointing
Runs just Experiment 2 (Baseline vs RAG) using GPT-4o-mini judge.

Usage:
    export OPENAI_API_KEY="your-key"
    python run_gpt_rag_only.py
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

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "judge_model": "gpt-4o-mini",
    "embed_model": "nomic-embed-text",
    "temperature": 0.2,
    "timeout": 300.0,
    "incident_dir": PROJECT_ROOT / "data" / "real_incidents",
    "results_dir": SCRIPT_DIR / "results_gpt",
    "train_ratio": 0.75,
    "random_seed": 42,
    "batch_save_every": 5,
}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ============================================================================
# CHECKPOINT
# ============================================================================

class CheckpointManager:
    def __init__(self, results_dir: Path):
        self.checkpoint_file = results_dir / "checkpoint_rag.json"
        self.state = self._load()
    
    def _load(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    return json.load(f)
            except:
                pass
        return {"last_idx": -1, "results": []}
    
    def save(self, idx: int, results: List):
        self.state = {"last_idx": idx, "results": results}
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_progress(self) -> tuple:
        return self.state.get("last_idx", -1), self.state.get("results", [])
    
    def clear(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("GPT_RAG")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    
    fh = logging.FileHandler(CONFIG["results_dir"] / "rag_experiment_log.txt", mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(fh)
    
    return logger

# ============================================================================
# DATA
# ============================================================================

def load_all_incidents(logger) -> List[Dict]:
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
# GPT SCORING
# ============================================================================

def score_with_gpt(openai_client, prediction: str, ground_truth: str, logger) -> float:
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
            response = openai_client.chat.completions.create(
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
            logger.warning(f"GPT attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2)
    
    return 0.5

# ============================================================================
# HELPERS
# ============================================================================

def get_prediction(ollama_client, logs: str, context: str) -> str:
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
# MAIN EXPERIMENT
# ============================================================================

def run_rag_comparison(logger, checkpoint, incidents, ollama_client, openai_client):
    logger.info("=" * 70)
    logger.info("RAG vs Baseline Comparison (GPT-4o-mini Judge)")
    logger.info("=" * 70)
    
    # Deterministic split
    random.seed(CONFIG["random_seed"])
    shuffled = incidents.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * CONFIG["train_ratio"])
    train_set, test_set = shuffled[:split], shuffled[split:]
    
    logger.info(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    # Resume from checkpoint
    start_idx, results_so_far = checkpoint.get_progress()
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
            baseline_score = score_with_gpt(openai_client, baseline_pred, test_case["root_cause"], logger)
            
            # RAG
            similar = find_similar(ollama_client, test_case, train_set)
            rag_context = f"Historical: {similar['category']} - {similar['root_cause']}"
            rag_pred = get_prediction(ollama_client, input_text, context=rag_context)
            rag_score = score_with_gpt(openai_client, rag_pred, test_case["root_cause"], logger)
            
            results["tests"].append({
                "id": test_case["id"],
                "baseline_score": baseline_score,
                "rag_score": rag_score,
                "improvement": rag_score - baseline_score
            })
            
            logger.debug(f"  Baseline: {baseline_score:.2f}, RAG: {rag_score:.2f}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results["tests"].append({"id": test_case["id"], "error": str(e)})
        
        # Checkpoint
        if (idx + 1) % CONFIG["batch_save_every"] == 0:
            checkpoint.save(idx, results["tests"])
    
    # Final aggregation
    valid = [t for t in results["tests"] if "baseline_score" in t]
    results["baseline_avg"] = round(statistics.mean([t["baseline_score"] for t in valid]), 4) if valid else 0
    results["rag_avg"] = round(statistics.mean([t["rag_score"] for t in valid]), 4) if valid else 0
    
    # Save final
    with open(CONFIG["results_dir"] / "02_rag_comparison_gpt.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    checkpoint.clear()
    
    logger.info(f"✅ Complete: Baseline={results['baseline_avg']*100:.1f}%, RAG={results['rag_avg']*100:.1f}%")
    return results

def main():
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("GPT-4o-mini RAG Comparison (with checkpointing)")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY not set!")
        return
    
    import ollama
    from openai import OpenAI
    
    ollama_client = ollama.Client(host=CONFIG["ollama_host"], timeout=CONFIG["timeout"])
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        ollama_client.list()
        logger.info("✓ Ollama connected")
    except Exception as e:
        logger.error(f"❌ Ollama failed: {e}")
        return
    
    incidents = load_all_incidents(logger)
    checkpoint = CheckpointManager(CONFIG["results_dir"])
    
    results = run_rag_comparison(logger, checkpoint, incidents, ollama_client, openai_client)
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Baseline: {results['baseline_avg']*100:.1f}%")
    logger.info(f"  RAG: {results['rag_avg']*100:.1f}%")

if __name__ == "__main__":
    main()
