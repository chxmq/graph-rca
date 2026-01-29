import os
import json
import time
import random
import statistics
import ollama
from pathlib import Path
from typing import List, Dict

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2:3b"
# Discover directories relative to this script (experiments/evaluate_rag_accuracy.py)
SCRIPT_DIR = Path(__file__).parent.absolute()
INCIDENT_DIR = SCRIPT_DIR.parent / "data" / "real_incidents"
RESULTS_DIR = SCRIPT_DIR / "eval_final_results"

class RAGEvaluator:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_HOST)
        RESULTS_DIR.mkdir(exist_ok=True)
        
    def get_all_incidents(self) -> List[Dict]:
        incidents = []
        for inc_folder in sorted(INCIDENT_DIR.glob("incident_*")):
            inc_id = inc_folder.name.split("_")[1]
            try:
                with open(inc_folder / "metadata.json") as f:
                    meta = json.load(f)
                with open(inc_folder / "ground_truth.json") as f:
                    gt = json.load(f)
                with open(inc_folder / "postmortem.md") as f:
                    pm = f.read()
                
                # Use logs if they exist, otherwise use postmortem for RAG context
                logs_path = inc_folder / "logs.txt"
                logs = logs_path.read_text() if logs_path.exists() else ""

                incidents.append({
                    "id": inc_id,
                    "metadata": meta,
                    "ground_truth": gt,
                    "postmortem": pm,
                    "logs": logs
                })
            except Exception as e:
                print(f"Error loading incident {inc_id}: {e}")
        return incidents

    def run_benchmark(self, train_set: List[Dict], test_set: List[Dict]):
        print(f"🚀 Starting Benchmark: {len(train_set)} Knowledge base, {len(test_set)} Tests")
        
        results = []
        
        for i, test_case in enumerate(test_set):
            print(f"[{i+1}/{len(test_set)}] Testing Incident {test_case['id']} - {test_case['metadata']['company']}")
            
            # 1. Baseline: Identify RCA without RAG
            baseline_rca = self._identify_rca(test_case['logs'], context="")
            
            # 2. RAG: Retrieve top-3 similar incidents manually (simulating vector search)
            similar_incidents = self._get_similar_incidents(test_case, train_set)
            rag_context = "\n\n".join([
                f"Historical Incident ID: {inc['id']}\nCompany: {inc['metadata']['company']}\nCategory: {inc['metadata']['category']}\nRoot Cause: {inc['ground_truth']['root_cause']}"
                for inc in similar_incidents
            ])
            
            # 3. RAG: Identify RCA with retrieved context
            rag_rca = self._identify_rca(test_case['logs'], context=rag_context)
            
            # 4. Evaluate against Ground Truth
            baseline_score = self._verify_rca(baseline_rca, test_case['ground_truth']['root_cause'])
            rag_score = self._verify_rca(rag_rca, test_case['ground_truth']['root_cause'])
            
            print(f"    Baseline Accuracy: {baseline_score*100:.1f}%")
            print(f"    RAG Accuracy:      {rag_score*100:.1f}%")

            results.append({
                "incident_id": test_case['id'],
                "company": test_case['metadata']['company'],
                "baseline_rca": baseline_rca,
                "rag_rca": rag_rca,
                "ground_truth": test_case['ground_truth']['root_cause'],
                "baseline_correct": baseline_score,
                "rag_correct": rag_score,
                "improvement": rag_score - baseline_score
            })
            
        return results

    def _get_similar_incidents(self, test_case: Dict, train_set: List[Dict], count=3):
        """Simulates semantic search by picking incidents from the same category or random ones."""
        # Realistic simulation: prioritize same category
        category = test_case['metadata']['category'].split("/")[0]
        eligible = [inc for inc in train_set if inc['metadata']['category'].startswith(category)]
        if len(eligible) < count:
            remaining = [inc for inc in train_set if inc not in eligible]
            eligible += random.sample(remaining, min(len(remaining), count - len(eligible)))
        
        return random.sample(eligible, count)

    def _identify_rca(self, logs: str, context: str) -> str:
        if context:
            prompt = f"""[INSTRUCTIONS]
Identify the ROOT CAUSE of the CURRENT INCIDENT based ONLY on the provided logs.
We have provided historical incidents for reference. 
Use the historical context ONLY to understand similar failure patterns (e.g., if a similar DB timeout led to a pool exhaustion).
DO NOT attribute the historical root causes to the current incident unless the logs explicitly support it.

[CURRENT INCIDENT LOGS]
{logs[:2000]}

[HISTORICAL CONTEXT (FOR REFERENCE ONLY)]
{context}

[RESPONSE]
Return the Root Cause of the CURRENT INCIDENT in one concise sentence.
Root Cause:"""
        else:
            prompt = f"""Identify the ROOT CAUSE of the following incident based on the log evidence.
        
Logs:
{logs[:2000]}

Return the Root Cause in one concise sentence. If unknown, say 'Unknown'.
Root Cause:"""

        try:
            response = self.client.generate(model=MODEL, prompt=prompt, options={"temperature": 0.1})
            return response['response'].strip()
        except Exception:
            return "Unknown"

    def _verify_rca(self, prediction: str, ground_truth: str) -> float:
        """Evaluates if the prediction captures the meaning of the ground truth using LLM."""
        if not prediction or prediction.lower() == "unknown": return 0.0
        
        prompt = f"""Compare the 'Prediction' against the 'Ground Truth' root cause.
        
Ground Truth: {ground_truth}
Prediction: {prediction}

Does the Prediction capture the core issue described in the Ground Truth?
Return a JSON object with a single key 'score' between 0.0 and 1.0.
{{ "score": 0.0 to 1.0 }}"""

        try:
            response = self.client.generate(model=MODEL, prompt=prompt, format="json", options={"temperature": 0.0})
            data = json.loads(response['response'])
            return float(data.get('score', 0.0))
        except Exception as e:
            return 0.0

def main():
    evaluator = RAGEvaluator()
    all_incidents = evaluator.get_all_incidents()
    
    if not all_incidents:
        print("❌ No incidents found. Run curation first.")
        return

    # Deterministic split for reproducibility
    random.seed(42)
    random.shuffle(all_incidents)
    
    train_set = all_incidents[:45]
    test_set = all_incidents[45:]
    
    results = evaluator.run_benchmark(train_set, test_set)
    
    # Calculate aggregate metrics
    baseline_avg = statistics.mean([r['baseline_correct'] for r in results])
    rag_avg = statistics.mean([r['rag_correct'] for r in results])
    
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_tests": len(results),
            "baseline_avg_accuracy": baseline_avg,
            "rag_avg_accuracy": rag_avg,
            "improvement_pct": ((rag_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
        },
        "details": results
    }
    
    with open(RESULTS_DIR / "real_incident_bench.json", 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"\n✅ Benchmark Complete!")
    print(f"Baseline Accuracy: {baseline_avg*100:.1f}%")
    print(f"RAG Accuracy:      {rag_avg*100:.1f}%")
    print(f"Improvement:      {output['summary']['improvement_pct']:.1f}%")

if __name__ == "__main__":
    main()
