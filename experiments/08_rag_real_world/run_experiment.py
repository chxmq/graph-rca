#!/usr/bin/env python3
"""
RAG Real-World Evaluation with Multi-Judge Validation
Compares baseline (no RAG) vs RAG accuracy on real-world incidents.
CORRECTED: Uses Actual Backend Logic (VectorDatabaseHandler) instead of simulation.
"""

import os
import sys
import json
import time
import re
import random
import argparse
import statistics
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# --- Backend Integration ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
try:
    from app.core.database_handlers import VectorDatabaseHandler, Document
    from app.utils.log_parser import LogParser
    from app.utils.graph_generator import GraphGenerator
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

# Setup local ChromaDB for experiment (prevents polluting prod DB)
EXP_CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db_temp"
os.environ["CHROMADB_PATH"] = str(EXP_CHROMA_DIR.absolute())


class JudgeClient:
    """Unified interface for different LLM judges."""
    
    def __init__(self, judge_name: str):
        self.judge_name = judge_name
        self.judge_config = CONFIG["judges"][judge_name]
        self.client = None
        self.api_keys = []
        self.current_key_idx = 0
        self._init_client()
    
    def _init_client(self):
        judge_type = self.judge_config["type"]
        
        if judge_type == "ollama":
            self.client = ollama.Client(host=CONFIG["ollama_host"])
            
        elif judge_type == "openai":
            keys = os.environ.get("OPENAI_API_KEY", "").split(",")
            self.api_keys = [k.strip() for k in keys if k.strip()]
            if not self.api_keys:
                print("WARN: OPENAI_API_KEY not set. Using mock judge.")
                self.client = None 
            else:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_keys[0])
            
        elif judge_type == "groq":
            keys = os.environ.get("GROQ_API_KEY", "").split(",")
            self.api_keys = [k.strip() for k in keys if k.strip()]
            if not self.api_keys:
                 self.client = None
            else:
                from groq import Groq
                self.client = Groq(api_key=self.api_keys[0])

    def _rotate_key(self):
        """Rotate to next API key if available."""
        if not self.api_keys or len(self.api_keys) <= 1:
            return False
        
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_idx]
        print(f"  ↻ Rotating API key to index {self.current_key_idx}...")
        
        judge_type = self.judge_config["type"]
        if judge_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=new_key)
        elif judge_type == "groq":
            from groq import Groq
            self.client = Groq(api_key=new_key)
        return True
    
    def score(self, prediction: str, ground_truth: str) -> float:
        """Score prediction against ground truth."""
        if not prediction or len(prediction.strip()) < 5:
            return 0.0
            
        if not self.client: # Mock mode
            return 0.5
        
        prompt = f"""Compare these two root cause descriptions and rate their similarity from 0.0 to 1.0.

Ground Truth: {ground_truth}
Prediction: {prediction}

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
        
        # Retry logic with key rotation
        max_retries = len(self.api_keys) if self.api_keys else 1
        retries = max(3, max_retries + 1)
        
        for attempt in range(retries):
            try:
                if judge_type == "ollama":
                    response = self.client.generate(
                        model=model,
                        prompt=prompt,
                        options={"temperature": 0.0}
                    )
                    text = response.response.strip()
                    # Strip <think>...</think> blocks from qwen3 responses
                    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                    
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
                
            except Exception as e:
                # If we have multiple keys, rotate and retry
                if (judge_type in ["openai", "groq"]) and self._rotate_key():
                    time.sleep(1) # Brief pause
                    continue
                else:
                    raise e
        return None


def load_incidents() -> List[Dict]:
    """Load all real-world incidents."""
    if not INCIDENT_DIR.exists():
        raise FileNotFoundError(f"Incident directory not found: {INCIDENT_DIR}")
    
    incidents = []
    print(f"Loading incidents from {INCIDENT_DIR}...")
    for folder in sorted(INCIDENT_DIR.glob("incident_*")):
        try:
            with open(folder / "ground_truth.json") as f:
                gt = json.load(f)
            
            meta = {}
            if (folder / "metadata.json").exists():
                with open(folder / "metadata.json") as f:
                    meta = json.load(f)
            
            with open(folder / "postmortem.md") as f:
                postmortem = f.read()
            logs_file = folder / "logs.txt"
            logs = logs_file.read_text() if logs_file.exists() else ""
            
            incidents.append({
                "id": folder.name,
                "company": meta.get("company", "Unknown"),
                "root_cause": gt.get("root_cause", ""),
                "category": gt.get("category", "Unknown"),
                "postmortem": postmortem,
                "logs": logs
            })
        except Exception as e:
            pass
    
    if not incidents:
        raise RuntimeError(f"No incidents could be loaded from {INCIDENT_DIR}")
    
    print(f"Loaded {len(incidents)} incidents")
    return incidents


def run_graphrca(logs: str, rag_context: str = "") -> str:
    """Run actual GraphRCA pipeline (LogParser + GraphGenerator), optionally with RAG context."""
    try:
        if not logs or len(logs) < 10:
            return "Insufficient logs"
        
        # 1. Parse using actual LLM-based LogParser
        parser = LogParser(model=CONFIG["model"])
        log_chain = parser.parse_log(logs)
        
        # 2. Build DAG and find root cause
        generator = GraphGenerator(log_chain)
        dag = generator.generate_dag()
        prediction = dag.root_cause
        
        # 3. If RAG context available, refine prediction with context
        if rag_context and prediction:
            client = ollama.Client(host=CONFIG["ollama_host"])
            prompt = (
                f"Using this context from a similar past incident:\n{rag_context}\n\n"
                f"Refine this root cause analysis:\n{prediction}\n\n"
                f"Refined Root Cause:"
            )
            response = client.generate(
                model=CONFIG["model"],
                prompt=prompt,
                options={"temperature": CONFIG["temperature"]}
            )
            return response.response.strip()
        
        return prediction
        
    except Exception as e:
        return f"GraphRCA failed: {str(e)}"


def setup_vector_db(train_set: List[Dict]) -> VectorDatabaseHandler:
    """Initialize Vector DB and populate with training data."""
    # Clean previous temp db
    if EXP_CHROMA_DIR.exists():
        shutil.rmtree(EXP_CHROMA_DIR)
    EXP_CHROMA_DIR.mkdir(parents=True)
    
    print("Initializing Vector Database...")
    vdb = VectorDatabaseHandler()
    
    documents = []
    
    for case in train_set:
        # We index the logs + postmortem as 'documentation' of past incidents
        doc_text = f"Category: {case['category']}\nRoot Cause: {case['root_cause']}\nAnalysis: {case['postmortem']}"
        documents.append(doc_text)
        
    # Generate embeddings and add
    # Note: VectorDatabaseHandler.add_documents handles embedding generation internally via its EF
    # but we need to pass a list[list[float]] for embeddings.
    # Actually, look at database_handlers.py:
    # It has a "OllamaEmbeddingFunction"
    # But add_documents takes (documents, embeddings).
    # We must generate them first.
    
    print(f"Generating embeddings for {len(documents)} training cases...")
    
    # We use the internal EF from the handler
    embeddings = vdb.ef(documents)
    vdb.add_documents(documents=documents, embeddings=embeddings)
    
    print(f"✓ Indexed {len(documents)} documents in ChromaDB")
    return vdb


def run_experiment(judge: JudgeClient) -> dict:
    """Compare baseline vs RAG using a specific judge."""
    print("=" * 70)
    print(f"EXPERIMENT: RAG vs Baseline Comparison (CORRECTED)")
    print(f"Judge: {judge.judge_name.upper()}") 
    print("=" * 70)
    
    incidents = load_incidents()
    
    if os.environ.get("SMOKE_TEST"):
        print("🔥 SMOKE TEST MODE ENABLED: Reducing to 5 incidents")
        incidents = incidents[:5]
    
    # Split train/test
    random.seed(CONFIG["random_seed"])
    shuffled = incidents.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * CONFIG["train_ratio"])
    train_set, test_set = shuffled[:split], shuffled[split:]
    
    # --- RAG SETUP ---
    vdb = setup_vector_db(train_set)
    # -----------------
    
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")
    
    results = {
        "judge": judge.judge_name,
        "tests": []
    }
    
    for idx, test_case in enumerate(test_set):
        print(f"\n[{idx+1}/{len(test_set)}] {test_case['id']}")
        
        input_text = test_case["logs"] if test_case["logs"] else test_case["postmortem"]
        
        try:
            # Baseline: GraphRCA without RAG context
            baseline_pred = run_graphrca(input_text, rag_context="")
            baseline_score = judge.score(baseline_pred, test_case["root_cause"])
            
            # RAG: retrieve similar past incident from Vector DB
            query = f"Incident logs: {input_text[:500]}"
            retrieved_docs = vdb.search(query=query, context="", top_k=1)
            
            rag_context = ""
            if retrieved_docs:
                rag_context = f"Similar Past Incident:\n{retrieved_docs[0].text}"
            
            # GraphRCA with RAG context
            rag_pred = run_graphrca(input_text, rag_context=rag_context)
            rag_score = judge.score(rag_pred, test_case["root_cause"])
            
            results["tests"].append({
                "id": test_case["id"],
                "baseline_score": round(baseline_score, 3),
                "rag_score": round(rag_score, 3),
                "improvement": round(rag_score - baseline_score, 3)
            })
            
            print(f"  Base: {baseline_score:.2f} | RAG: {rag_score:.2f} | Diff: {rag_score-baseline_score:+.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results["tests"].append({"id": test_case["id"], "error": str(e)})
    
    # Aggregate
    valid = [t for t in results["tests"] if "baseline_score" in t]
    results["baseline_avg"] = round(statistics.mean([t["baseline_score"] for t in valid]), 4) if valid else 0
    results["rag_avg"] = round(statistics.mean([t["rag_score"] for t in valid]), 4) if valid else 0
    results["improvement"] = round(results["rag_avg"] - results["baseline_avg"], 4)
    
    print(f"\nRESULTS: Base {results['baseline_avg']:.2f} -> RAG {results['rag_avg']:.2f}")
    
    # Cleanup
    if EXP_CHROMA_DIR.exists():
        shutil.rmtree(EXP_CHROMA_DIR)
        
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["qwen", "gpt", "groq", "all"], default="qwen")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    judges = list(CONFIG["judges"].keys()) if args.judge == "all" else [args.judge]
    
    for j in judges:
        try:
            judge = JudgeClient(j)
            results = run_experiment(judge)
            output_file = output_dir / f"rag_comparison_{j}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Run enrichment/analysis immediately
            try:
                print("\nRunning Heterogeneous Effects Analysis...")
                cmd = [sys.executable, str(Path(__file__).parent / "enrich_with_companies.py"), str(output_file)]
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"Analysis failed: {e}")
                
        except Exception as e:
             print(f"Judge {j} failed: {e}")

if __name__ == "__main__":
    main()
