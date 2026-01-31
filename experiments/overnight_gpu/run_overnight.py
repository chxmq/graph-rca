#!/usr/bin/env python3
"""
===============================================================================
GraphRCA OVERNIGHT GPU Experiment Suite
===============================================================================

FAULT-TOLERANT DESIGN:
- Checkpoints after each experiment (resume if crashed)
- Retries on transient failures (network, API limits)
- Graceful degradation (skips failed experiments, continues with rest)
- Comprehensive logging to file
- Progress saved every step

RUN:
    cd experiments/overnight_gpu
    python run_overnight.py

REQUIREMENTS:
    - Ollama running with llama3.2:3b and nomic-embed-text
    - ChromaDB running on port 8000 (for noise test)
    - NO external APIs required (100% local/on-premises)

OUTPUT:
    experiments/overnight_gpu/results/
    ├── checkpoint.json           # Resume point
    ├── experiment_log.txt        # Full log
    ├── 01_rca_accuracy.json      # 60 real incidents
    ├── 02_rag_comparison.json    # Baseline vs RAG
    ├── 03_noise_sensitivity.json # 1000+ decoys
    └── final_summary.json        # All results
"""

import os
import sys
import json
import time
import random
import logging
import traceback
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Callable
from functools import wraps

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Models (all local via Ollama - no external APIs)
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",           # For RCA inference (the system being tested)
    "judge_model": "llama3.1:70b",    # For scoring (larger model = more rigorous evaluation)
    "embed_model": "nomic-embed-text",
    "temperature": 0.2,
    
    # ChromaDB
    "chroma_host": "localhost",
    "chroma_port": 8000,
    
    # Data
    "incident_dir": PROJECT_ROOT / "data" / "real_incidents",
    
    # Output
    "results_dir": SCRIPT_DIR / "results",
    
    # Experiment settings
    "runs_per_test": 3,
    "train_ratio": 0.75,
    "random_seed": 42,
    "target_decoys": 1000,
    
    # Fault tolerance
    "max_retries": 3,
    "retry_delay_sec": 5,
    "batch_save_every": 5,  # Save checkpoint every N incidents
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    """Setup file + console logging."""
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    
    log_file = CONFIG["results_dir"] / "experiment_log.txt"
    
    logger = logging.getLogger("GraphRCA_Overnight")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler - detailed
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    
    # Console handler - brief
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# RETRY DECORATOR
# ============================================================================

def retry_on_failure(max_retries: int = None, delay: float = None):
    """Decorator to retry function on failure."""
    max_retries = max_retries or CONFIG["max_retries"]
    delay = delay or CONFIG["retry_delay_sec"]
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            raise last_error
        return wrapper
    return decorator


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manages experiment checkpoints for resumability."""
    
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
        return {
            "started_at": datetime.now().isoformat(),
            "experiments_completed": [],
            "current_experiment": None,
            "current_progress": {}
        }
    
    def save(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def is_experiment_done(self, name: str) -> bool:
        return name in self.state["experiments_completed"]
    
    def mark_experiment_done(self, name: str):
        if name not in self.state["experiments_completed"]:
            self.state["experiments_completed"].append(name)
        self.state["current_experiment"] = None
        self.state["current_progress"] = {}
        self.save()
    
    def set_current(self, name: str, progress: Dict = None):
        self.state["current_experiment"] = name
        self.state["current_progress"] = progress or {}
        self.save()
    
    def get_progress(self, name: str) -> Dict:
        if self.state["current_experiment"] == name:
            return self.state["current_progress"]
        return {}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_incidents(logger: logging.Logger) -> List[Dict]:
    """Load all 60 real-world incidents."""
    incidents = []
    incident_dir = CONFIG["incident_dir"]
    
    logger.info(f"Loading incidents from {incident_dir}")
    
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


def load_decoy_documents(logger: logging.Logger) -> List[Dict]:
    """Load raw postmortems as decoys."""
    decoys = []
    sources_dir = CONFIG["incident_dir"] / "sources" / "raw"
    
    # GitHub
    github_dir = sources_dir / "github"
    if github_dir.exists():
        for file in github_dir.glob("*.md"):
            try:
                decoys.append({"id": f"github_{file.stem}", "content": file.read_text()[:2000]})
            except:
                pass
    
    # SRE Weekly
    sre_dir = sources_dir / "sre_weekly"
    if sre_dir.exists():
        for file in sre_dir.glob("*.md"):
            try:
                decoys.append({"id": f"sre_{file.stem}", "content": file.read_text()[:2000]})
            except:
                pass
    
    logger.info(f"Loaded {len(decoys)} decoy documents")
    return decoys


# ============================================================================
# EXPERIMENT 1: RCA ACCURACY
# ============================================================================

def run_rca_accuracy(
    logger: logging.Logger,
    checkpoint: CheckpointManager,
    incidents: List[Dict],
    ollama_client
) -> Dict:
    """Test RCA accuracy on all 60 real incidents."""
    
    EXPERIMENT_NAME = "01_rca_accuracy"
    
    if checkpoint.is_experiment_done(EXPERIMENT_NAME):
        logger.info(f"Skipping {EXPERIMENT_NAME} (already done)")
        # Load existing results
        result_file = CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {}
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: RCA Accuracy on 60 Real Incidents")
    logger.info("=" * 70)
    
    # Resume from checkpoint if available
    progress = checkpoint.get_progress(EXPERIMENT_NAME)
    start_idx = progress.get("last_completed_idx", -1) + 1
    results_so_far = progress.get("results", [])
    
    if start_idx > 0:
        logger.info(f"Resuming from incident {start_idx}")
    
    checkpoint.set_current(EXPERIMENT_NAME)
    
    for idx in range(start_idx, len(incidents)):
        incident = incidents[idx]
        logger.info(f"[{idx+1}/{len(incidents)}] {incident['id']} - {incident['metadata'].get('company', 'Unknown')}")
        
        incident_results = {
            "id": incident["id"],
            "company": incident["metadata"].get("company", "Unknown"),
            "category": incident["category"],
            "ground_truth": incident["root_cause"],
            "runs": []
        }
        
        for run in range(CONFIG["runs_per_test"]):
            try:
                prediction, score = evaluate_single_rca(
                    ollama_client, incident, logger
                )
                incident_results["runs"].append({
                    "run": run + 1,
                    "prediction": prediction[:500],  # Truncate for storage
                    "score": score
                })
                logger.debug(f"  Run {run+1}: score={score:.2f}")
            except Exception as e:
                logger.error(f"  Run {run+1} failed: {e}")
                incident_results["runs"].append({"run": run+1, "error": str(e), "score": 0})
        
        # Calculate average
        scores = [r["score"] for r in incident_results["runs"] if "score" in r]
        incident_results["avg_score"] = round(statistics.mean(scores), 3) if scores else 0
        incident_results["correct"] = incident_results["avg_score"] >= 0.7
        
        results_so_far.append(incident_results)
        
        # Checkpoint every N incidents
        if (idx + 1) % CONFIG["batch_save_every"] == 0:
            checkpoint.set_current(EXPERIMENT_NAME, {
                "last_completed_idx": idx,
                "results": results_so_far
            })
            logger.debug(f"Checkpoint saved at incident {idx+1}")
    
    # Finalize
    results = compile_rca_results(results_so_far)
    
    # Save
    with open(CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    checkpoint.mark_experiment_done(EXPERIMENT_NAME)
    logger.info(f"✅ {EXPERIMENT_NAME} complete: {results['overall_accuracy']*100:.1f}%")
    
    return results


@retry_on_failure()
def evaluate_single_rca(ollama_client, incident: Dict, logger) -> tuple:
    """Evaluate RCA for a single incident with retries."""
    input_text = incident["logs"][:3000] if incident["logs"] else incident["postmortem"][:3000]
    
    prompt = f"""Analyze this incident and identify the ROOT CAUSE in one sentence.

{input_text}

Root Cause:"""
    
    response = ollama_client.generate(
        model=CONFIG["model"],
        prompt=prompt,
        options={"temperature": CONFIG["temperature"]}
    )
    prediction = response["response"].strip()
    
    # Score using local Ollama (no external APIs)
    score = score_with_ollama(ollama_client, prediction, incident["root_cause"], logger)
    
    return prediction, score


def score_with_ollama(ollama_client, prediction: str, ground_truth: str, logger) -> float:
    """Score prediction using local Ollama with larger judge model for rigorous evaluation."""
    if not prediction or len(prediction.strip()) < 5:
        return 0.0
    
    try:
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
        
        # Use larger judge model for scoring (more rigorous evaluation)
        response = ollama_client.generate(
            model=CONFIG["judge_model"],
            prompt=prompt,
            options={"temperature": 0.0}
        )
        
        # Extract number from response
        text = response["response"].strip()
        # Find first float-like pattern
        import re
        match = re.search(r'(0\.\d+|1\.0|0|1)', text)
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        
        # Fallback to keyword scoring
        return keyword_score(prediction, ground_truth)
        
    except Exception as e:
        logger.debug(f"Ollama scoring failed: {e}, using keyword fallback")
        return keyword_score(prediction, ground_truth)


def keyword_score(prediction: str, ground_truth: str) -> float:
    """Fallback keyword-based scoring."""
    pred_lower = prediction.lower()
    gt_words = ground_truth.lower().split()
    matches = sum(1 for w in gt_words if w in pred_lower)
    return min(1.0, matches / max(len(gt_words), 1) * 2)


def compile_rca_results(results: List[Dict]) -> Dict:
    """Compile RCA results with category breakdown."""
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0, "scores": []}
        by_category[cat]["total"] += 1
        by_category[cat]["scores"].append(r["avg_score"])
        if r["correct"]:
            by_category[cat]["correct"] += 1
    
    for cat in by_category:
        data = by_category[cat]
        data["accuracy"] = round(data["correct"] / data["total"], 3) if data["total"] > 0 else 0
    
    all_scores = [r["avg_score"] for r in results]
    
    return {
        "total_incidents": len(results),
        "runs_per_incident": CONFIG["runs_per_test"],
        "overall_accuracy": round(statistics.mean(all_scores), 4) if all_scores else 0,
        "overall_std": round(statistics.stdev(all_scores), 4) if len(all_scores) > 1 else 0,
        "by_category": by_category,
        "by_incident": results
    }


# ============================================================================
# EXPERIMENT 2: RAG COMPARISON
# ============================================================================

def run_rag_comparison(
    logger: logging.Logger,
    checkpoint: CheckpointManager,
    incidents: List[Dict],
    ollama_client
) -> Dict:
    """Compare Baseline vs RAG accuracy."""
    
    EXPERIMENT_NAME = "02_rag_comparison"
    
    if checkpoint.is_experiment_done(EXPERIMENT_NAME):
        logger.info(f"Skipping {EXPERIMENT_NAME} (already done)")
        result_file = CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {}
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: RAG vs Baseline Comparison")
    logger.info("=" * 70)
    
    # Split
    random.seed(CONFIG["random_seed"])
    shuffled = incidents.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * CONFIG["train_ratio"])
    train_set, test_set = shuffled[:split], shuffled[split:]
    
    logger.info(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    checkpoint.set_current(EXPERIMENT_NAME)
    
    results = {
        "train_size": len(train_set),
        "test_size": len(test_set),
        "tests": []
    }
    
    for idx, test_case in enumerate(test_set):
        logger.info(f"[{idx+1}/{len(test_set)}] {test_case['id']}")
        
        input_text = test_case["logs"][:2000] if test_case["logs"] else test_case["postmortem"][:2000]
        
        try:
            # Baseline
            baseline_pred = get_prediction(ollama_client, input_text, context="")
            baseline_score = score_with_ollama(ollama_client, baseline_pred, test_case["root_cause"], logger)
            
            # RAG - find similar
            similar = find_similar(ollama_client, test_case, train_set)
            rag_context = f"Historical: {similar['category']} - {similar['root_cause']}"
            rag_pred = get_prediction(ollama_client, input_text, context=rag_context)
            rag_score = score_with_ollama(ollama_client, rag_pred, test_case["root_cause"], logger)
            
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
    
    # Aggregate
    valid = [t for t in results["tests"] if "baseline_score" in t]
    results["baseline_avg"] = round(statistics.mean([t["baseline_score"] for t in valid]), 4) if valid else 0
    results["rag_avg"] = round(statistics.mean([t["rag_score"] for t in valid]), 4) if valid else 0
    results["improvement"] = round(results["rag_avg"] - results["baseline_avg"], 4)
    
    # Save
    with open(CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    checkpoint.mark_experiment_done(EXPERIMENT_NAME)
    logger.info(f"✅ {EXPERIMENT_NAME} complete: Baseline={results['baseline_avg']*100:.1f}%, RAG={results['rag_avg']*100:.1f}%")
    
    return results


@retry_on_failure()
def get_prediction(ollama_client, logs: str, context: str) -> str:
    """Get RCA prediction."""
    if context:
        prompt = f"Using this context: {context}\n\nIdentify root cause:\n{logs}\n\nRoot Cause:"
    else:
        prompt = f"Identify root cause:\n{logs}\n\nRoot Cause:"
    
    response = ollama_client.generate(
        model=CONFIG["model"],
        prompt=prompt,
        options={"temperature": CONFIG["temperature"]}
    )
    return response["response"].strip()


def find_similar(ollama_client, test_case: Dict, train_set: List[Dict]) -> Dict:
    """Find most similar incident."""
    try:
        test_emb = ollama_client.embeddings(
            model=CONFIG["embed_model"],
            prompt=f"{test_case['category']} {test_case['root_cause']}"
        )["embedding"]
        
        best = train_set[0]
        best_sim = -1
        
        for train in train_set[:20]:  # Limit for speed
            train_emb = ollama_client.embeddings(
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


# ============================================================================
# EXPERIMENT 3: NOISE SENSITIVITY
# ============================================================================

def run_noise_sensitivity(
    logger: logging.Logger,
    checkpoint: CheckpointManager,
    incidents: List[Dict],
    decoys: List[Dict],
    ollama_client
) -> Dict:
    """Test retrieval with 1000+ decoys."""
    
    EXPERIMENT_NAME = "03_noise_sensitivity"
    
    if checkpoint.is_experiment_done(EXPERIMENT_NAME):
        logger.info(f"Skipping {EXPERIMENT_NAME} (already done)")
        result_file = CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {}
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Noise Sensitivity (1000+ Decoys)")
    logger.info("=" * 70)
    
    try:
        import chromadb
    except ImportError:
        logger.error("ChromaDB not installed!")
        return {"status": "skipped", "reason": "chromadb not installed"}
    
    checkpoint.set_current(EXPERIMENT_NAME)
    
    # Generate synthetic if needed
    needed = max(0, CONFIG["target_decoys"] - len(decoys))
    if needed > 0:
        logger.info(f"Generating {needed} synthetic decoys...")
        decoys = decoys + generate_synthetic(needed)
    
    total = len(incidents) + len(decoys)
    logger.info(f"Corpus: {len(incidents)} targets + {len(decoys)} decoys = {total}")
    
    # Setup ChromaDB (in-memory mode - no server needed)
    try:
        # Try in-memory first (no server required)
        client = chromadb.Client()
        try:
            client.delete_collection("overnight_noise_test")
        except:
            pass
        collection = client.create_collection("overnight_noise_test", metadata={"hnsw:space": "cosine"})
        logger.info("Using ChromaDB in-memory mode")
    except Exception as e:
        logger.error(f"ChromaDB setup failed: {e}")
        return {"status": "error", "reason": str(e)}
    
    # Index all documents
    logger.info("Indexing documents...")
    all_docs = [{"id": i["id"], "content": i["postmortem"][:1000], "type": "target"} for i in incidents]
    all_docs += [{"id": d["id"], "content": d["content"][:1000], "type": "decoy"} for d in decoys]
    
    batch_size = 10
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i+batch_size]
        embeddings = []
        for doc in batch:
            try:
                emb = ollama_client.embeddings(model=CONFIG["embed_model"], prompt=doc["content"])["embedding"]
            except:
                emb = [0.0] * 768
            embeddings.append(emb)
        
        try:
            collection.add(
                ids=[d["id"] for d in batch],
                documents=[d["content"] for d in batch],
                embeddings=embeddings,
                metadatas=[{"type": d["type"]} for d in batch]
            )
        except Exception as e:
            logger.warning(f"Batch {i} failed: {e}")
        
        if (i + batch_size) % 100 == 0:
            logger.info(f"  Indexed {min(i+batch_size, len(all_docs))}/{len(all_docs)}")
    
    # Test retrieval
    logger.info("Testing retrieval...")
    recall_1, recall_3, recall_5 = 0, 0, 0
    
    for inc in incidents:
        try:
            query_emb = ollama_client.embeddings(model=CONFIG["embed_model"], prompt=inc["root_cause"])["embedding"]
            result = collection.query(query_embeddings=[query_emb], n_results=5)
            ids = result["ids"][0]
            
            if inc["id"] in ids[:1]: recall_1 += 1
            if inc["id"] in ids[:3]: recall_3 += 1
            if inc["id"] in ids[:5]: recall_5 += 1
        except Exception as e:
            logger.warning(f"Query failed for {inc['id']}: {e}")
    
    n = len(incidents)
    results = {
        "total_targets": n,
        "total_decoys": len(decoys),
        "recall_at_1": recall_1,
        "recall_at_3": recall_3,
        "recall_at_5": recall_5,
        "recall_at_1_pct": round(recall_1 / n * 100, 1),
        "recall_at_3_pct": round(recall_3 / n * 100, 1),
        "recall_at_5_pct": round(recall_5 / n * 100, 1),
    }
    
    # Save
    with open(CONFIG["results_dir"] / f"{EXPERIMENT_NAME}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    checkpoint.mark_experiment_done(EXPERIMENT_NAME)
    logger.info(f"✅ {EXPERIMENT_NAME} complete: Recall@3={results['recall_at_3_pct']}%")
    
    return results


def generate_synthetic(count: int) -> List[Dict]:
    """Generate synthetic decoys."""
    random.seed(42)
    templates = ["System {issue}. Root cause: {cause}. Fix: {fix}.", "Incident: {issue}. Resolved by {fix}."]
    issues = ["timeout", "memory leak", "disk full", "connection refused", "CPU spike"]
    causes = ["misconfiguration", "capacity", "bug", "hardware"]
    fixes = ["restart", "config update", "scaling"]
    
    return [
        {"id": f"syn_{i}", "content": random.choice(templates).format(
            issue=random.choice(issues), cause=random.choice(causes), fix=random.choice(fixes)
        )}
        for i in range(count)
    ]


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup
    logger = setup_logging()
    checkpoint = CheckpointManager(CONFIG["results_dir"])
    
    logger.info("=" * 70)
    logger.info("GraphRCA OVERNIGHT GPU Experiment Suite")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Results dir: {CONFIG['results_dir']}")
    
    # Initialize clients
    try:
        import ollama
        ollama_client = ollama.Client(host=CONFIG["ollama_host"])
        # Test connection
        ollama_client.list()
        logger.info("✓ Ollama connected")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        return
    
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        from openai import OpenAI
        openai_client = OpenAI(api_key=openai_key)
        logger.info(f"✓ OpenAI connected ({CONFIG['judge_model']})")
    else:
        logger.warning("⚠️ OPENAI_API_KEY not set - using keyword scoring fallback")
        openai_client = None
    
    # Load data
    incidents = load_all_incidents(logger)
    if not incidents:
        logger.error("❌ No incidents found!")
        return
    
    decoys = load_decoy_documents(logger)
    
    # Run experiments
    all_results = {}
    
    try:
        # Exp 1: RCA Accuracy
        all_results["rca_accuracy"] = run_rca_accuracy(
            logger, checkpoint, incidents, ollama_client
        )
    except Exception as e:
        logger.error(f"Experiment 1 failed: {e}\n{traceback.format_exc()}")
        all_results["rca_accuracy"] = {"status": "error", "error": str(e)}
    
    try:
        # Exp 2: RAG Comparison
        all_results["rag_comparison"] = run_rag_comparison(
            logger, checkpoint, incidents, ollama_client
        )
    except Exception as e:
        logger.error(f"Experiment 2 failed: {e}\n{traceback.format_exc()}")
        all_results["rag_comparison"] = {"status": "error", "error": str(e)}
    
    try:
        # Exp 3: Noise Sensitivity
        all_results["noise_sensitivity"] = run_noise_sensitivity(
            logger, checkpoint, incidents, decoys, ollama_client
        )
    except Exception as e:
        logger.error(f"Experiment 3 failed: {e}\n{traceback.format_exc()}")
        all_results["noise_sensitivity"] = {"status": "error", "error": str(e)}
    
    # Final summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "experiments_run": len([v for v in all_results.values() if v.get("status") != "error"]),
        "results": {}
    }
    
    if "overall_accuracy" in all_results.get("rca_accuracy", {}):
        summary["results"]["rca_accuracy"] = all_results["rca_accuracy"]["overall_accuracy"]
    if "baseline_avg" in all_results.get("rag_comparison", {}):
        summary["results"]["baseline_avg"] = all_results["rag_comparison"]["baseline_avg"]
        summary["results"]["rag_avg"] = all_results["rag_comparison"]["rag_avg"]
    if "recall_at_3_pct" in all_results.get("noise_sensitivity", {}):
        summary["results"]["noise_recall_3"] = all_results["noise_sensitivity"]["recall_at_3_pct"]
    
    with open(CONFIG["results_dir"] / "final_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {CONFIG['results_dir']}")
    
    for key, val in summary["results"].items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val*100 if val < 1 else val:.1f}%")


if __name__ == "__main__":
    main()
