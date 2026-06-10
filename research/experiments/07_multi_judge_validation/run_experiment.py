#!/usr/bin/env python3
"""
Multi-Judge RCA Validation Experiment
Tests RCA accuracy using multiple LLM judges.
CORRECTED: Uses actual GraphRCA (GraphGenerator) for prediction, validating the ALGORITHM not just the LLM.
"""

import os
import sys
import json
import time
import random
import re
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# --- Backend Integration ---
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "app" / "backend"))
try:
    from app.models import LogEntry, LogChain
    from app.graph_generator import GraphGenerator
    from app.log_parser import LogParser
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
    "runs_per_incident": 3,
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
        # incident from accuracy instead of silently counting it as wrong.
        print(f"  ⚠ WARNING: all scoring attempts failed for judge '{self.judge_name}'. "
              f"Marking unscored. Check API key and connectivity.")
        return None
    
    def _call_judge(self, prompt: str) -> Optional[float]:
        judge_type = self.judge_config["type"]
        model = self.judge_config["model"]
        
        # Retry logic with key rotation
        max_retries = len(self.api_keys) if self.api_keys else 1
        # Cap retries to avoid infinite loops but allow at least one full rotation + extra
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
                    raise e # Re-raise if rotation didn't happen (or not supported)
        return None


def load_incidents() -> List[Dict]:
    """Load all real-world incidents."""
    incidents = []
    
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


def run_graphrca(logs: str) -> dict:
    """Run the full GraphRCA pipeline and return BOTH of its outputs:

    - "line": the deterministic heuristic root cause (a verbatim log line) —
      what the UI shows as the stable identifier;
    - "narrative": the LLM-articulated root-cause explanation generated from
      the causal chain (the system's root_cause_expln) — the system's actual
      causal claim, and the fair object of comparison against a narrative
      ground truth.
    """
    try:
        if not logs or len(logs) < 10:
            return {"line": "Insufficient logs", "narrative": "Insufficient logs"}

        # 1. Parse using actual LLM-based LogParser
        parser = LogParser(model=CONFIG["model"], timeout=180)
        log_chain = parser.parse_log(logs)
        parse_stats = {
            "parsed_lines": log_chain.parsed_lines,
            "total_lines": log_chain.total_lines,
            "parse_errors": len(log_chain.parse_errors),
        }

        # 2. Build DAG and find heuristic root cause
        generator = GraphGenerator(log_chain)
        dag = generator.generate_dag()

        # 3. Articulate the causal narrative from the chain (same prompt and
        #    model as the production summary stage).
        context = ContextBuilder(dag).build_context()
        client = ollama.Client(host=CONFIG["ollama_host"], timeout=120)
        narrative = ""
        try:
            resp = client.generate(
                model=CONFIG["model"],
                prompt=summary_prompt("\n".join(context.causal_chain)),
                format="json",
                options={"temperature": CONFIG["temperature"], "num_ctx": 8192},
            )
            narrative = json.loads(resp.response).get("root_cause", "") or ""
        except Exception as e:
            print(f"  narrative generation failed: {e}")

        return {"line": dag.root_cause or "", "narrative": narrative, "parse_stats": parse_stats}

    except Exception as e:
        return {"line": f"GraphRCA failed: {str(e)}", "narrative": "", "parse_stats": {}}


def run_validation(judge_name: str) -> Dict:
    """Run multi-judge validation."""
    print("=" * 70)
    print(f"MULTI-JUDGE VALIDATION: {judge_name.upper()} (CORRECTED)")
    print(f"Validating: GraphRCA Algorithm")
    print("=" * 70)
    
    judge = JudgeClient(judge_name)
    incidents = load_incidents()
    
    print(f"Loaded {len(incidents)} incidents\n")
    
    if os.environ.get("SMOKE_TEST"):
        print("🔥 SMOKE TEST MODE ENABLED: Reducing to 5 incidents")
        incidents = incidents[:5]
    
    # Checkpoint/resume: each scored incident is appended to a JSONL file so
    # a crash or restart skips completed work instead of redoing hours of it.
    checkpoint = Path(__file__).parent / "data" / f"checkpoint_{judge_name}.jsonl"
    checkpoint.parent.mkdir(exist_ok=True)
    done: Dict[str, dict] = {}
    if checkpoint.exists():
        for raw in checkpoint.read_text().splitlines():
            try:
                rec = json.loads(raw)
                done[rec["id"]] = rec
            except json.JSONDecodeError:
                continue
        if done:
            print(f"Resuming: {len(done)} incidents already scored in {checkpoint.name}")

    results = {"incidents": [], "by_category": {}, "total_incidents": len(incidents)}

    for idx, incident in enumerate(incidents):
        print(f"[{idx+1}/{len(incidents)}] {incident['id']} ({incident['category']})")

        if incident["id"] in done:
            results["incidents"].append(done[incident["id"]])
            print("  → resumed from checkpoint")
            continue

        preds = run_graphrca(incident["logs"])
        line_score = judge.score(preds["line"], incident["root_cause"])
        narrative_score = judge.score(preds["narrative"], incident["root_cause"])

        record = {
            "id": incident["id"],
            "category": incident["category"],
            "prediction": preds["narrative"][:160],
            "prediction_line": preds["line"][:160],
            # Primary metric: the narrative (the system's actual causal
            # claim); the heuristic line rides along as an ablation.
            "avg_score": round(narrative_score, 3) if narrative_score is not None else None,
            "line_score": round(line_score, 3) if line_score is not None else None,
            "correct": (narrative_score or 0) >= 0.7,
            "correct_line": (line_score or 0) >= 0.7,
            "parse_stats": preds.get("parse_stats", {}),
        }
        results["incidents"].append(record)
        with open(checkpoint, "a") as f:
            f.write(json.dumps(record) + "\n")

        ns = "fail" if narrative_score is None else f"{narrative_score:.2f}"
        ls = "fail" if line_score is None else f"{line_score:.2f}"
        print(f"  → narrative: {ns} ({'✓' if record['correct'] else '✗'}) | line: {ls}")

    # Summary — judge-failed incidents (score None) are excluded from
    # accuracy denominators and reported separately.
    scored = [i for i in results["incidents"] if i["avg_score"] is not None]
    scored_line = [i for i in results["incidents"] if i["line_score"] is not None]
    results["unscored_incidents"] = len(results["incidents"]) - len(scored)

    for inc in scored:
        cat = inc["category"]
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"correct": 0, "total": 0}
        results["by_category"][cat]["total"] += 1
        if inc["correct"]:
            results["by_category"][cat]["correct"] += 1

    def _metrics(rows: list, key: str) -> dict:
        vals = [r[key] for r in rows]
        return {
            "mean_score": round(statistics.mean(vals), 4) if vals else 0,
            "accuracy_at_0.5": round(sum(1 for v in vals if v >= 0.5) / len(vals), 4) if vals else 0,
            "accuracy_at_0.7": round(sum(1 for v in vals if v >= 0.7) / len(vals), 4) if vals else 0,
        }

    results["narrative"] = _metrics(scored, "avg_score")
    results["line"] = _metrics(scored_line, "line_score")
    # Kept for backward compatibility with older analysis scripts.
    results["overall_accuracy"] = results["narrative"]["accuracy_at_0.7"]
    results["overall_accuracy_line"] = results["line"]["accuracy_at_0.7"]
    results["config"] = {
        "pipeline_model": CONFIG["model"],
        "judge_model": CONFIG["judges"][judge_name]["model"],
        "score_threshold": 0.7,
        "dataset": "symptom-only logs v2",
        "run_date": datetime.now().isoformat(),
    }
    results["judge"] = judge_name
    
    print(f"\n{'='*70}")
    print(f"RESULTS ({judge_name.upper()}) — narrative: acc@0.7 {results['narrative']['accuracy_at_0.7']*100:.1f}%, "
          f"acc@0.5 {results['narrative']['accuracy_at_0.5']*100:.1f}%, mean {results['narrative']['mean_score']:.3f}")
    print(f"{'':19}line:      acc@0.7 {results['line']['accuracy_at_0.7']*100:.1f}%, "
          f"acc@0.5 {results['line']['accuracy_at_0.5']*100:.1f}%, mean {results['line']['mean_score']:.3f}")
    if results["unscored_incidents"]:
        print(f"  ⚠ {results['unscored_incidents']} incidents unscored (judge failures) — excluded from accuracy")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=["qwen", "gpt", "groq", "all"], default="qwen")
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    judges = ["qwen", "gpt", "groq"] if args.judge == "all" else [args.judge]
    
    for j in judges:
        try:
            results = run_validation(j)
            with open(output_dir / f"results_{j}.json", "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Judge {j} failed: {e}")

if __name__ == "__main__":
    main()
