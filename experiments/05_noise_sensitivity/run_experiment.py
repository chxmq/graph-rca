#!/usr/bin/env python3
"""
Progressive Noise Sensitivity Test for RAG Retrieval
Tests retrieval accuracy at multiple noise levels using Project Infrastructure.
CORRECTED: Uses app.core.embedding.EmbeddingCreator and app.core.database_handlers.VectorDatabaseHandler
"""

import os
import sys
import json
import time
import random
import shutil
from pathlib import Path
from typing import List, Dict

# --- Backend Integration ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
try:
    from app.core.database_handlers import VectorDatabaseHandler
    from app.core.embedding import EmbeddingCreator
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)
# ---------------------------

# Configuration
COLLECTION_NAME = "noise_sensitivity_test"
EXP_CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db_noise"
os.environ["CHROMADB_PATH"] = str(EXP_CHROMA_DIR.absolute())

NOISE_LEVELS = [0, 100, 250, 500, 750, 1000]

if os.environ.get("SMOKE_TEST"):
    print("🔥 SMOKE TEST MODE ENABLED: Reduced noise levels")
    NOISE_LEVELS = [0, 100]

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "real_incidents"

# Templates for synthetic decoys
COMPONENTS = ["database", "cache", "API gateway", "load balancer", "message queue", "storage", 
              "authentication", "CDN", "DNS", "container orchestrator", "monitoring", "logging"]
ISSUES = ["high latency", "timeout errors", "connection refused", "memory exhaustion", "CPU spike",
          "disk full", "network partition", "certificate expiry", "rate limiting", "deadlock"]
CAUSES = ["misconfiguration", "capacity exhaustion", "software bug", "hardware failure", 
          "dependency failure", "traffic spike", "deployment error", "DNS propagation delay"]
FIXES = ["rolling restart", "configuration update", "capacity scaling", "hotfix deployment",
         "failover activation", "cache flush", "dependency upgrade", "traffic rerouting"]


def load_incidents() -> List[Dict]:
    """Load all real-world incidents."""
    incidents = []
    for folder in sorted(DATA_DIR.glob("incident_*")):
        try:
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()
            with open(folder / "ground_truth.json") as f:
                gt = json.load(f)
            incidents.append({
                "id": folder.name,
                "content": postmortem,
                "root_cause": gt.get("root_cause", ""),
                "category": gt.get("category", "Unknown"),
            })
        except:
            pass
    return incidents


def generate_synthetic_decoys(count: int, seed: int = 42) -> List[Dict]:
    """Generate synthetic decoy documents."""
    random.seed(seed)
    decoys = []
    for i in range(count):
        content = f"Incident report: {random.choice(COMPONENTS)} {random.choice(ISSUES)}. " + \
                  f"Root cause: {random.choice(CAUSES)}. Resolution: {random.choice(FIXES)}."
        decoys.append({
            "id": f"decoy_{i:04d}",
            "content": content,
            "category": "Decoy"
        })
    return decoys


def run_single_noise_level(incidents: List[Dict], num_decoys: int, test_indices: List[int], embedder: EmbeddingCreator) -> Dict:
    """Run retrieval test at a single noise level."""
    # Reset DB
    if EXP_CHROMA_DIR.exists():
        shutil.rmtree(EXP_CHROMA_DIR)
    EXP_CHROMA_DIR.mkdir(parents=True)
    
    vdb = VectorDatabaseHandler()
    
    # Prepare all documents (Targets + Decoys)
    all_docs = []
    
    # Add targets
    for inc in incidents:
        all_docs.append(inc["content"])
    
    # Add decoys
    if num_decoys > 0:
        decoys = generate_synthetic_decoys(num_decoys)
        for d in decoys:
            all_docs.append(d["content"])
            
    print(f"  Indexing {len(all_docs)} documents (Targets: {len(incidents)}, Decoys: {num_decoys})...")
    
    # Batch embeddings via EmbeddingCreator (as per plan)
    # We do it in chunks to be safe with memory
    chunk_size = 100
    for i in range(0, len(all_docs), chunk_size):
        chunk = all_docs[i:i+chunk_size]
        embeddings = embedder.create_batch_embeddings(chunk)
        try:
            print(f"DEBUG: Adding {len(chunk)} docs to VDB. Embeddings len: {len(embeddings) if embeddings else 'None'}")
            vdb.add_documents(documents=chunk, embeddings=embeddings)
        except Exception as e:
            print(f"CRITICAL ERROR adding documents: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
    # Run retrieval test
    hits = 0
    decoys_in_top5 = 0
    
    # To count decoys accurately, we need to know what we retrieved.
    # The VectorDatabaseHandler.search returns Document objects. 
    # But it doesn't return IDs directly in the Document object as defined in database_handlers.py
    # However, the textual content of our decoys is unique enough.
    # Decoys start with "Incident report: [Component]..."
    
    for idx in test_indices:
        target = incidents[idx]
        query = target["root_cause"] + " " + target["category"]
        
        # Use vdb.search
        results = vdb.search(query=query, context="", top_k=5)
        
        # Check hits
        # A hit is if the target content is in the results
        # We compare content because IDs are hashed in vdb
        target_found = False
        decoy_count = 0
        
        for res in results:
            if res.text == target["content"]:
                target_found = True
            if res.text.startswith("Incident report:"):
                decoy_count += 1
                
        if target_found:
            hits += 1
        decoys_in_top5 += decoy_count
    
    accuracy = (hits / len(test_indices)) * 100
    avg_decoys = decoys_in_top5 / len(test_indices)
    
    return {
        "decoys": num_decoys,
        "collection_size": len(incidents) + num_decoys,
        "accuracy": accuracy,
        "avg_decoys_in_top5": round(avg_decoys, 2),
        "hits": hits,
        "total": len(test_indices)
    }


def main():
    print("=" * 60)
    print("PROGRESSIVE NOISE SENSITIVITY TEST (CORRECTED)")
    print(f"Testing at noise levels: {NOISE_LEVELS}")
    print("=" * 60)
    
    embedder = EmbeddingCreator()
    incidents = load_incidents()
    print(f"\nLoaded {len(incidents)} incidents")
    
    test_indices = list(range(len(incidents)))
    
    results = []
    
    for noise_level in NOISE_LEVELS:
        print(f"\n{'='*60}")
        print(f"Testing with {noise_level} decoys...")
        print(f"{'='*60}")
        
        result = run_single_noise_level(incidents, noise_level, test_indices, embedder)
        results.append(result)
        
        print(f"  Accuracy: {result['accuracy']:.1f}%")
        print(f"  Avg decoys in top-5: {result['avg_decoys_in_top5']}")
    
    # Save results
    output_file = Path(__file__).parent / "data" / "03_noise_sensitivity.json"
    output_file.parent.mkdir(exist_ok=True)
    
    noise_results = {
        "noise_levels": {str(r["decoys"]): {
            "accuracy": r["accuracy"] / 100,
            "avg_decoys_in_top5": r["avg_decoys_in_top5"],
            "collection_size": r["collection_size"]
        } for r in results},
        "test_count": len(test_indices)
    }
    
    with open(output_file, "w") as f:
        json.dump(noise_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Clean up
    if EXP_CHROMA_DIR.exists():
        shutil.rmtree(EXP_CHROMA_DIR)

if __name__ == "__main__":
    main()
