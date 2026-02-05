#!/usr/bin/env python3
"""
Documentation Ablation Experiment
Tests impact of documentation on RCA accuracy using real-world incidents.
CORRECTED: Uses actual VectorDatabaseHandler for RAG retrieval instead of string concatenation.
"""

import os
import sys
import json
import random
import statistics
import shutil
from pathlib import Path
from datetime import datetime

# --- Backend Integration ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
try:
    from app.core.database_handlers import VectorDatabaseHandler
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
    "judge_model": "qwen3:32b",
    "temperature": 0.2,
    "runs_per_incident": 3,
    "random_seed": 42,
}

if os.environ.get("SMOKE_TEST"):
    print("🔥 SMOKE TEST MODE ENABLED: 1 run only")
    CONFIG["runs_per_incident"] = 1

PROJECT_ROOT = Path(__file__).parent.parent.parent
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"

# Setup local ChromaDB for experiment
EXP_CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db_ablation"
os.environ["CHROMADB_PATH"] = str(EXP_CHROMA_DIR.absolute())


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
        match = re.search(r'(0\.\d+|1\.0|0|1)', response.response)
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.5


def setup_ablation_db(docs: list):
    """Setup ChromaDB with a specific set of documentation."""
    # Clean previous
    if EXP_CHROMA_DIR.exists():
        shutil.rmtree(EXP_CHROMA_DIR)
    EXP_CHROMA_DIR.mkdir(parents=True)
    
    if not docs:
        return None
        
    vdb = VectorDatabaseHandler()
    documents = []
    
    print(f"  Indexing {len(docs)} documents...")
    for d in docs:
        # Index as documentation
        doc_text = f"Incident: {d['category']}\nRoot Cause: {d['root_cause']}\n{d['postmortem']}"
        documents.append(doc_text)
        
    # Generate embeddings (VectorDatabaseHandler's EF handles this if configured correctly,
    # but based on our analysis of database_handlers.py, add_documents expects embeddings)
    # So we used the EF attached to the vdb instance
    embeddings = vdb.ef(documents)
    vdb.add_documents(documents=documents, embeddings=embeddings)
    
    return vdb


def run_doc_ablation(client: ollama.Client) -> dict:
    """Documentation ablation using real incidents."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Documentation Ablation (CORRECTED)")
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
    
    if os.environ.get("SMOKE_TEST"):
        print("  Using reduced dataset (10 incidents) for smoke test")
        shuffled = shuffled[:10]
    
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
        
        # Setup Vector DB for this ablation level
        vdb = setup_ablation_db(docs)
        scores = []
        
        for idx, test_case in enumerate(test_set):
            # Only log every 5th item to reduce clutter
            if (idx + 1) % 5 == 0:
                print(f"  Progress: {idx+1}/{len(test_set)}")
            
            for run in range(CONFIG["runs_per_incident"]):
                input_text = test_case["logs"] if test_case["logs"] else test_case["postmortem"]
                
                context_str = ""
                if vdb:
                     # Retrieve relevant docs
                    query = f"Incident logs: {input_text[:500]}"
                    retrieved = vdb.search(query=query, context="", top_k=3) # Retrieve top 3
                    if retrieved:
                         context_str = "\n---\n".join([d.text for d in retrieved])

                if context_str:
                    prompt = f"""Using this documentation of past incidents:
{context_str}

Identify the root cause for these logs:
{input_text}

Root Cause:"""
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
                    prediction = response.response.strip()
                    score = score_prediction(client, prediction, test_case["root_cause"])
                    scores.append(score)
                except Exception as e:
                    print(f"  Error: {e}")
        
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
    
    # Clean up
    if EXP_CHROMA_DIR.exists():
        shutil.rmtree(EXP_CHROMA_DIR)


if __name__ == "__main__":
    main()
