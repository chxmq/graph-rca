#!/usr/bin/env python3
"""
Progressive Noise Sensitivity Test for RAG Retrieval

Tests retrieval accuracy at multiple noise levels (0, 100, 250, 500, 750, 1000 decoys)
to generate data matching paper's Table tab:noise_sensitivity.

Usage:
    python run_experiment.py
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

# Configuration
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "noise_sensitivity_test"
CHROMA_PERSIST_DIR = Path(__file__).parent / "data" / "chroma_db"
NOISE_LEVELS = [0, 100, 250, 500, 750, 1000]  # Progressive noise levels
TEST_SAMPLE_SIZE = 20  # Number of test incidents per noise level

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "real_incidents"

# Synthetic decoy templates
COMPONENTS = ["database", "cache", "API gateway", "load balancer", "message queue", "storage", 
              "authentication", "CDN", "DNS", "container orchestrator", "monitoring", "logging"]
ISSUES = ["high latency", "timeout errors", "connection refused", "memory exhaustion", "CPU spike",
          "disk full", "network partition", "certificate expiry", "rate limiting", "deadlock"]
CAUSES = ["misconfiguration", "capacity exhaustion", "software bug", "hardware failure", 
          "dependency failure", "traffic spike", "deployment error", "DNS propagation delay"]
FIXES = ["rolling restart", "configuration update", "capacity scaling", "hotfix deployment",
         "failover activation", "cache flush", "dependency upgrade", "traffic rerouting"]


def get_ollama_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    import requests
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


def load_incidents() -> List[Dict]:
    """Load all real-world incidents."""
    incidents = []
    for folder in sorted(DATA_DIR.glob("incident_*")):
        try:
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()[:2000]
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
            "type": "decoy"
        })
    return decoys


def run_single_noise_level(incidents: List[Dict], num_decoys: int, test_indices: List[int]) -> Dict:
    """Run retrieval test at a single noise level."""
    # Create fresh ChromaDB client
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    # Delete existing collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    # Create collection
    collection = client.create_collection(name=COLLECTION_NAME)
    
    # Index all incidents (as targets)
    print(f"  Indexing {len(incidents)} incidents...")
    for inc in incidents:
        embedding = get_ollama_embedding(inc["content"][:1000])
        collection.add(
            ids=[inc["id"]],
            embeddings=[embedding],
            metadatas=[{"type": "target", "category": inc["category"]}],
            documents=[inc["content"][:500]]
        )
    
    # Generate and index decoys
    if num_decoys > 0:
        print(f"  Indexing {num_decoys} decoys...")
        decoys = generate_synthetic_decoys(num_decoys)
        for decoy in decoys:
            embedding = get_ollama_embedding(decoy["content"])
            collection.add(
                ids=[decoy["id"]],
                embeddings=[embedding],
                metadatas=[{"type": "decoy"}],
                documents=[decoy["content"]]
            )
    
    # Run retrieval test
    hits = 0
    decoys_in_top5 = 0
    
    for idx in test_indices:
        target = incidents[idx]
        query_embedding = get_ollama_embedding(target["root_cause"] + " " + target["category"])
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        
        retrieved_ids = results["ids"][0]
        if target["id"] in retrieved_ids:
            hits += 1
        
        # Count decoys in top-5
        decoys_in_top5 += sum(1 for r in retrieved_ids if r.startswith("decoy_"))
    
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
    print("PROGRESSIVE NOISE SENSITIVITY TEST")
    print(f"Testing at noise levels: {NOISE_LEVELS}")
    print("=" * 60)
    
    # Load incidents
    incidents = load_incidents()
    print(f"\nLoaded {len(incidents)} incidents")
    
    # Fixed test set for consistency across noise levels
    random.seed(42)
    test_indices = random.sample(range(len(incidents)), min(TEST_SAMPLE_SIZE, len(incidents)))
    print(f"Using {len(test_indices)} test incidents")
    
    results = []
    
    for noise_level in NOISE_LEVELS:
        print(f"\n{'='*60}")
        print(f"Testing with {noise_level} decoys...")
        print(f"{'='*60}")
        
        result = run_single_noise_level(incidents, noise_level, test_indices)
        results.append(result)
        
        print(f"  Accuracy: {result['accuracy']:.1f}%")
        print(f"  Avg decoys in top-5: {result['avg_decoys_in_top5']}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("📊 PROGRESSIVE NOISE SENSITIVITY RESULTS")
    print("=" * 60)
    print(f"{'Decoys':<10} {'Collection':<15} {'Accuracy':<12} {'Avg Decoys Top-5'}")
    print("-" * 60)
    for r in results:
        print(f"{r['decoys']:<10} {r['collection_size']:<15} {r['accuracy']:.1f}%{'':<7} {r['avg_decoys_in_top5']}")
    
    # Save results
    output = {
        "test_sample_size": len(test_indices),
        "noise_levels": NOISE_LEVELS,
        "results": results
    }
    
    output_file = Path(__file__).parent / "data" / "progressive_noise_results.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print LaTeX table for paper
    print("\n" + "=" * 60)
    print("📝 LATEX TABLE FOR PAPER:")
    print("=" * 60)
    print("""
\\begin{table}[t]
\\centering
\\caption{Noise Sensitivity Analysis (20 test incidents)}
\\label{tab:noise_sensitivity}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Decoys} & \\textbf{Collection Size} & \\textbf{Accuracy} & \\textbf{Avg Decoys in Top-5} \\\\
\\midrule""")
    for r in results:
        print(f"{r['decoys']} & {r['collection_size']} & {r['accuracy']:.1f}\\% & {r['avg_decoys_in_top5']:.2f} \\\\")
    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")


if __name__ == "__main__":
    main()
