#!/usr/bin/env python3
"""
===============================================================================
GraphRCA Noise Sensitivity Test (Standalone)
===============================================================================

Run this on a machine with SQLite >= 3.35.0 for ChromaDB.

Usage:
    python run_noise_test.py

Requirements:
    - Ollama with llama3.2:3b and nomic-embed-text
    - SQLite >= 3.35.0 (for ChromaDB)
"""

import os
import sys
import json
import random
import logging
import statistics
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "judge_model": "qwen3:32b",
    "embed_model": "nomic-embed-text",
    "temperature": 0.2,
    "incident_dir": PROJECT_ROOT / "data" / "real_incidents",
    "results_dir": SCRIPT_DIR / "results",
    "target_decoys": 1000,
    "test_incidents": 20,  # Number of incidents to test
    "random_seed": 42,
}

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("NoiseTest")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    
    fh = logging.FileHandler(CONFIG["results_dir"] / "noise_test_log.txt", mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(fh)
    
    return logger

# ============================================================================
# DATA LOADING
# ============================================================================

def load_incidents(logger) -> list:
    """Load all incidents from the data directory."""
    incidents = []
    incident_dir = CONFIG["incident_dir"]
    
    # Look for incident_*/ground_truth.json pattern
    for folder in sorted(incident_dir.glob("incident_*")):
        if not folder.is_dir():
            continue
        gt_file = folder / "ground_truth.json"
        meta_file = folder / "metadata.json"
        logs_file = folder / "logs.txt"
        postmortem_file = folder / "postmortem.md"
        
        if gt_file.exists():
            try:
                with open(gt_file) as fp:
                    gt = json.load(fp)
                
                meta = {}
                if meta_file.exists():
                    with open(meta_file) as fp:
                        meta = json.load(fp)
                
                logs = ""
                if logs_file.exists():
                    logs = logs_file.read_text()
                
                postmortem = ""
                if postmortem_file.exists():
                    postmortem = postmortem_file.read_text()
                
                data = {
                    "id": folder.name,
                    "root_cause": gt.get("root_cause", ""),
                    "category": gt.get("category", meta.get("category", "Unknown")),
                    "company": meta.get("company", "Unknown"),
                    "logs": logs,
                    "postmortem": postmortem
                }
                incidents.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {folder}: {e}")
    
    logger.info(f"Loaded {len(incidents)} incidents")
    return incidents

def generate_decoys(incidents: list, target: int) -> list:
    """Generate decoy documents by mixing incident data."""
    decoys = []
    random.seed(CONFIG["random_seed"])
    
    while len(decoys) < target:
        i1, i2 = random.sample(incidents, 2)
        decoy = {
            "title": f"Mixed: {i1.get('company', 'Unknown')} + {i2.get('company', 'Unknown')}",
            "content": f"{i1.get('postmortem', '')[:500]} ... {i2.get('root_cause', '')}",
            "is_decoy": True
        }
        decoys.append(decoy)
    
    return decoys

# ============================================================================
# NOISE SENSITIVITY TEST
# ============================================================================

def run_noise_test(logger, ollama_client, incidents: list) -> dict:
    """Test RAG accuracy with increasing noise levels."""
    import chromadb
    
    logger.info("=" * 70)
    logger.info("NOISE SENSITIVITY TEST")
    logger.info("=" * 70)
    
    # Generate decoys
    decoys = generate_decoys(incidents, CONFIG["target_decoys"])
    logger.info(f"Generated {len(decoys)} decoy documents")
    
    # Select test incidents
    random.seed(CONFIG["random_seed"])
    test_incidents = random.sample(incidents, min(CONFIG["test_incidents"], len(incidents)))
    logger.info(f"Testing with {len(test_incidents)} incidents")
    
    noise_levels = [0, 100, 250, 500, 750, 1000]
    results = {"noise_levels": {}, "test_count": len(test_incidents)}
    
    for noise_count in noise_levels:
        logger.info(f"\n--- Noise Level: {noise_count} decoys ---")
        
        # Create fresh ChromaDB for each noise level
        client = chromadb.Client()
        collection = client.create_collection(
            name=f"noise_test_{noise_count}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add real incidents
        for inc in incidents:
            doc_text = f"{inc.get('company', '')} {inc.get('category', '')} {inc.get('root_cause', '')}"
            embedding = get_embedding(ollama_client, doc_text)
            collection.add(
                ids=[inc["id"]],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[{"root_cause": inc.get("root_cause", ""), "is_decoy": False}]
            )
        
        # Add decoys
        for i, decoy in enumerate(decoys[:noise_count]):
            embedding = get_embedding(ollama_client, decoy["content"][:1000])
            collection.add(
                ids=[f"decoy_{i}"],
                embeddings=[embedding],
                documents=[decoy["content"][:1000]],
                metadatas=[{"root_cause": "", "is_decoy": True}]
            )
        
        logger.info(f"Collection size: {collection.count()}")
        
        # Test retrieval accuracy
        correct = 0
        decoy_retrieved = 0
        
        for test in test_incidents:
            query_text = test.get("logs", test.get("postmortem", ""))[:1000]
            query_embedding = get_embedding(ollama_client, query_text)
            
            results_query = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            top_ids = results_query["ids"][0] if results_query["ids"] else []
            top_metadatas = results_query["metadatas"][0] if results_query["metadatas"] else []
            
            # Check if correct incident is in top-5
            if test["id"] in top_ids:
                correct += 1
            
            # Count decoys in top-5
            decoy_retrieved += sum(1 for m in top_metadatas if m.get("is_decoy", False))
        
        accuracy = correct / len(test_incidents)
        avg_decoys_in_top5 = decoy_retrieved / len(test_incidents)
        
        results["noise_levels"][str(noise_count)] = {
            "accuracy": round(accuracy, 4),
            "avg_decoys_in_top5": round(avg_decoys_in_top5, 2),
            "collection_size": collection.count()
        }
        
        logger.info(f"  Accuracy: {accuracy*100:.1f}%, Avg decoys in top-5: {avg_decoys_in_top5:.2f}")
    
    # Save results
    with open(CONFIG["results_dir"] / "03_noise_sensitivity.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n✅ Noise test complete!")
    return results

def get_embedding(ollama_client, text: str) -> list:
    """Get embedding from Ollama."""
    response = ollama_client.embeddings(
        model=CONFIG["embed_model"],
        prompt=text[:2000]
    )
    return response["embedding"]

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("GraphRCA Noise Sensitivity Test")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    
    # Connect to Ollama
    import ollama
    ollama_client = ollama.Client(host=CONFIG["ollama_host"])
    
    try:
        ollama_client.list()
        logger.info("✓ Ollama connected")
    except Exception as e:
        logger.error(f"❌ Ollama connection failed: {e}")
        return
    
    # Load data
    incidents = load_incidents(logger)
    if not incidents:
        logger.error("❌ No incidents found!")
        return
    
    # Run test
    try:
        results = run_noise_test(logger, ollama_client, incidents)
        logger.info(f"\nResults saved to: {CONFIG['results_dir']}/03_noise_sensitivity.json")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
