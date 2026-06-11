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
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "app" / "backend"))
try:
    from app.database import VectorDatabaseHandler, Document
    from app.log_parser import LogParser
    from app.graph_generator import GraphGenerator
    from app.context_builder import ContextBuilder
    from app.prompts import summary_prompt
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)
# ---------------------------

if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama

CONFIG = {
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11435"),
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
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set or is empty. "
                    "Run: export OPENAI_API_KEY=your_key_here\n"
                    "Without a valid key, judge scores would be fabricated. Aborting."
                )
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_keys[0])
            
        elif judge_type == "groq":
            keys = os.environ.get("GROQ_API_KEY", "").split(",")
            self.api_keys = [k.strip() for k in keys if k.strip()]
            if not self.api_keys:
                raise ValueError(
                    "GROQ_API_KEY environment variable is not set or is empty. "
                    "Run: export GROQ_API_KEY=your_key_here\n"
                    "Without a valid key, judge scores would be fabricated. Aborting."
                )
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
        
        # All retries exhausted — return None so the caller can EXCLUDE this
        # condition from aggregates instead of silently counting it as wrong.
        print(f"  WARNING: all scoring attempts failed for judge '{self.judge_name}'. "
              f"Marking unscored. Check API key and connectivity.")
        return None
    
    def predict_root_cause(self, logs: str) -> str:
        """Zero-shot condition: the judge's own model does RCA directly
        from raw logs, with no parsing, temporal chain, or RAG.  This is
        the 'inherent LLM capability' baseline reported in the paper."""
        prompt = (
            "Analyze these incident logs and identify the most likely root cause.\n\n"
            f"Logs:\n{logs}\n\n"
            "Respond with 1-3 sentences describing the root cause only:"
        )
        judge_type = self.judge_config["type"]
        model = self.judge_config["model"]
        try:
            if judge_type == "ollama":
                response = self.client.generate(
                    model=model, prompt=prompt, options={"temperature": 0.0}
                )
                text = response.response.strip()
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Zero-shot prediction failed: {e}")
            return ""

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


def run_pipeline(logs: str) -> dict:
    """Run the GraphRCA pipeline ONCE per incident: parse, build the graph,
    and articulate the causal narrative (production root_cause_expln).
    Both the baseline and RAG conditions reuse this single pass, so the two
    conditions differ only in retrieval — not in parsing noise."""
    if not logs or len(logs) < 10:
        return {"narrative": "Insufficient logs", "chain": "", "parse_stats": {}}

    parser = LogParser(model=CONFIG["model"], timeout=180)
    log_chain = parser.parse_log(logs)
    dag = GraphGenerator(log_chain).generate_dag()
    context = ContextBuilder(dag).build_context()
    chain_text = "\n".join(context.causal_chain)

    client = ollama.Client(host=CONFIG["ollama_host"], timeout=180)
    narrative = ""
    try:
        resp = client.generate(
            model=CONFIG["model"],
            prompt=summary_prompt(chain_text),
            format="json",
            options={"temperature": CONFIG["temperature"], "num_ctx": 8192},
        )
        narrative = json.loads(resp.response).get("root_cause", "") or ""
    except Exception as e:
        print(f"  narrative generation failed: {e}")

    return {
        "narrative": narrative,
        "chain": chain_text,
        "parse_stats": {
            "parsed_lines": log_chain.parsed_lines,
            "total_lines": log_chain.total_lines,
            "parse_errors": len(log_chain.parse_errors),
        },
    }


def refine_with_rag(narrative: str, chain_text: str, rag_context: str) -> str:
    """RAG condition: re-articulate the root cause with a similar past
    incident as additional evidence, grounded in the same causal chain."""
    client = ollama.Client(host=CONFIG["ollama_host"], timeout=180)
    prompt = (
        "You are analyzing a production incident.\n\n"
        f"Observed causal chain from the logs:\n{chain_text}\n\n"
        f"A similar past incident from the knowledge base:\n{rag_context}\n\n"
        f"Current root cause hypothesis:\n{narrative}\n\n"
        "Using the past incident as evidence (only where it genuinely fits the "
        "observed symptoms), state the most likely root cause in 1-3 sentences:"
    )
    response = client.generate(
        model=CONFIG["model"],
        prompt=prompt,
        options={"temperature": CONFIG["temperature"], "num_ctx": 8192},
    )
    return response.response.strip()


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
    
    # Checkpoint/resume: scored test cases are appended to JSONL so a crash
    # or restart never redoes completed (expensive) work.
    checkpoint = Path(__file__).parent / "data" / f"checkpoint_{judge.judge_name}.jsonl"
    checkpoint.parent.mkdir(exist_ok=True)
    done = {}
    if checkpoint.exists():
        for raw in checkpoint.read_text().splitlines():
            try:
                rec = json.loads(raw)
                done[rec["id"]] = rec
            except json.JSONDecodeError:
                continue
        if done:
            print(f"Resuming: {len(done)} test cases already scored")

    results = {
        "judge": judge.judge_name,
        "tests": []
    }

    for idx, test_case in enumerate(test_set):
        print(f"\n[{idx+1}/{len(test_set)}] {test_case['id']}")

        if test_case["id"] in done:
            results["tests"].append(done[test_case["id"]])
            print("  resumed from checkpoint")
            continue

        # The postmortem contains the ground-truth root cause; feeding it to
        # the pipeline as input would leak the answer.  Skip instead.
        if not test_case["logs"].strip():
            print("  No logs — skipped to avoid ground-truth leakage")
            results["tests"].append({"id": test_case["id"], "error": "no logs"})
            continue
        input_text = test_case["logs"]

        try:
            # Zero-shot: judge's own model, raw logs, no pipeline.  An empty
            # prediction means the GENERATION call failed (infrastructure),
            # not that the model answered badly — mark unscored, don't zero.
            zero_shot_pred = judge.predict_root_cause(input_text)
            zero_shot_score = judge.score(zero_shot_pred, test_case["root_cause"]) if zero_shot_pred else None

            # One pipeline pass per incident; baseline and RAG share it so
            # the conditions differ only in retrieval, not parsing noise.
            pipe = run_pipeline(input_text)
            baseline_pred = pipe["narrative"]
            baseline_score = judge.score(baseline_pred, test_case["root_cause"])

            # RAG: retrieve the most similar past incident, then re-articulate.
            # The retrieval DISTANCE is recorded so adaptive-retrieval gates
            # (retrieve only when a sufficiently similar incident exists) can
            # be designed and evaluated on legitimate inputs post-hoc.
            query = f"Incident logs: {input_text[:500]}"
            res = vdb.get_collection().query(
                query_texts=[query], n_results=1,
                include=["documents", "distances"],
            )
            docs = (res.get("documents") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            rag_context = f"Similar Past Incident:\n{docs[0]}" if docs else ""
            retrieval_distance = round(float(dists[0]), 4) if dists else None
            rag_pred = refine_with_rag(baseline_pred, pipe["chain"], rag_context) if rag_context else baseline_pred
            rag_score = judge.score(rag_pred, test_case["root_cause"])

            record = {
                "id": test_case["id"],
                "zero_shot_score": round(zero_shot_score, 3) if zero_shot_score is not None else None,
                "baseline_score": round(baseline_score, 3) if baseline_score is not None else None,
                "rag_score": round(rag_score, 3) if rag_score is not None else None,
                "improvement": round(rag_score - baseline_score, 3)
                               if rag_score is not None and baseline_score is not None else None,
                "zero_shot_pred": zero_shot_pred or "",
                "baseline_pred": baseline_pred or "",
                "rag_pred": rag_pred or "",
                "retrieval_distance": retrieval_distance,
                "parse_stats": pipe["parse_stats"],
            }
            results["tests"].append(record)
            with open(checkpoint, "a") as f:
                f.write(json.dumps(record) + "\n")

            fmt = lambda v: "fail" if v is None else f"{v:.2f}"
            print(f"  Zero: {fmt(zero_shot_score)} | Base: {fmt(baseline_score)} | RAG: {fmt(rag_score)}")

        except Exception as e:
            print(f"  Failed: {e}")
            results["tests"].append({"id": test_case["id"], "error": str(e)})

    # Aggregate — judge-failed conditions (None) are excluded per-condition.
    def _metrics(key: str) -> dict:
        vals = [t[key] for t in results["tests"] if t.get(key) is not None]
        if not vals:
            return {"mean_score": 0, "accuracy_at_0.5": 0, "accuracy_at_0.7": 0, "n": 0}
        return {
            "mean_score": round(statistics.mean(vals), 4),
            "accuracy_at_0.5": round(sum(1 for v in vals if v >= 0.5) / len(vals), 4),
            "accuracy_at_0.7": round(sum(1 for v in vals if v >= 0.7) / len(vals), 4),
            "n": len(vals),
        }

    results["zero_shot"] = _metrics("zero_shot_score")
    results["baseline"] = _metrics("baseline_score")
    results["rag"] = _metrics("rag_score")
    # Backward-compatible keys for enrich_with_companies.py
    results["zero_shot_avg"] = results["zero_shot"]["mean_score"]
    results["baseline_avg"] = results["baseline"]["mean_score"]
    results["rag_avg"] = results["rag"]["mean_score"]
    results["improvement"] = round(results["rag_avg"] - results["baseline_avg"], 4)
    results["config"] = {
        "pipeline_model": CONFIG["model"],
        "judge_model": CONFIG["judges"][judge.judge_name]["model"],
        "train_ratio": CONFIG["train_ratio"],
        "random_seed": CONFIG["random_seed"],
        "dataset": "symptom-only logs v2",
        "run_date": datetime.now().isoformat(),
    }

    print(f"\nRESULTS mean: Zero {results['zero_shot_avg']:.2f} -> Base {results['baseline_avg']:.2f} -> RAG {results['rag_avg']:.2f}")
    print(f"acc@0.7:      Zero {results['zero_shot']['accuracy_at_0.7']:.0%} -> Base {results['baseline']['accuracy_at_0.7']:.0%} -> RAG {results['rag']['accuracy_at_0.7']:.0%}")
    
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
