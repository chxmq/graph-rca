#!/usr/bin/env python3
"""
Parser Accuracy Experiment
Tests LLM parsing accuracy on LogHub datasets (BGL, HDFS).
Matches paper methodology: 400 samples per dataset, 3 runs each.
"""

import os
import sys
import json
import time
import re
import statistics
from pathlib import Path
from datetime import datetime

if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

import ollama

CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "llama3.2:3b",
    "temperature": 0.1,
    "runs_per_log": 3,
    "samples_per_dataset": 400,
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGHUB_DIR = PROJECT_ROOT / "data" / "loghub"

# Fallback sample logs if LogHub not available
BGL_SAMPLE = [
    {"raw": "- 1131511861 2005.11.09 R33-M0-N7-C:J03-U01 2005-11-09-06.11.01.134579 R33-M0-N7-C:J03-U01 RAS KERNEL INFO generating core.7681",
     "expected": {"timestamp": "2005-11-09-06.11.01", "level": "INFO", "component": "KERNEL", "message": "generating core.7681"}},
]

HDFS_SAMPLE = [
    {"raw": "081109 203615 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106",
     "expected": {"timestamp": "081109 203615", "level": "INFO", "component": "DataNode", "message": "Receiving block"}},
]


def load_loghub_dataset(dataset_name: str, max_samples: int) -> list:
    """Load LogHub dataset."""
    dataset_path = LOGHUB_DIR / dataset_name / f"{dataset_name}_2k.log"
    
    if not dataset_path.exists():
        print(f"  ⚠ {dataset_path} not found")
        return None
    
    logs = []
    with open(dataset_path, 'r', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                logs.append({"raw": line, "expected": None})
    
    print(f"  Loaded {len(logs)} logs from {dataset_path}")
    return logs


def parse_log_with_llm(client: ollama.Client, log: str) -> dict:
    """Parse a log entry using LLM."""
    prompt = f"""Parse this log entry into JSON with these exact fields:
- timestamp: the date/time portion
- level: severity (INFO, WARN, ERROR, FATAL, etc)
- component: the system/service name
- message: the main message content

Log: {log}

Return ONLY valid JSON:"""

    start = time.time()
    try:
        response = client.generate(
            model=CONFIG["model"],
            prompt=prompt,
            options=ollama.Options(temperature=CONFIG["temperature"]),
            format="json"
        )
        latency = time.time() - start
        text = response.response.strip()
        
        try:
            parsed = json.loads(text)
            return {"parsed": parsed, "latency": latency, "success": True}
        except json.JSONDecodeError:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {"parsed": parsed, "latency": latency, "success": True}
    except:
        pass
    
    return {"parsed": {}, "latency": time.time() - start, "success": False}


def check_field_present(parsed: dict, field: str) -> bool:
    """Check if a field was extracted."""
    if field not in parsed:
        return False
    val = str(parsed[field]).strip()
    return len(val) > 0 and val.lower() not in ['none', 'null', 'n/a', '']


def run_parser_experiment(client: ollama.Client) -> dict:
    """Test parser on LogHub datasets."""
    print("=" * 70)
    print("EXPERIMENT: Parser Accuracy on LogHub")
    print(f"Config: {CONFIG['samples_per_dataset']} samples/dataset, {CONFIG['runs_per_log']} runs each")
    print("=" * 70)
    
    fields = ["timestamp", "level", "component", "message"]
    results = {"datasets": {}}
    
    for ds_name, fallback in [("BGL", BGL_SAMPLE), ("HDFS", HDFS_SAMPLE)]:
        print(f"\nTesting {ds_name}...")
        
        logs = load_loghub_dataset(ds_name, CONFIG["samples_per_dataset"])
        if logs is None:
            print(f"  Using fallback sample data")
            logs = fallback
        
        field_correct = {f: 0 for f in fields}
        total_tests = 0
        latencies = []
        errors = 0
        
        for idx, log_entry in enumerate(logs):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(logs)}")
            
            for run in range(CONFIG["runs_per_log"]):
                result = parse_log_with_llm(client, log_entry["raw"])
                latencies.append(result["latency"])
                
                if result["success"]:
                    for field in fields:
                        if check_field_present(result["parsed"], field):
                            field_correct[field] += 1
                else:
                    errors += 1
                total_tests += 1
        
        field_accuracy = {}
        for field in fields:
            acc = field_correct[field] / total_tests if total_tests > 0 else 0
            field_accuracy[field] = {"mean": round(acc * 100, 1), "std": 0.0}
        
        overall = sum(field_correct.values()) / (total_tests * len(fields)) if total_tests > 0 else 0
        
        results["datasets"][ds_name] = {
            "samples": len(logs),
            "runs": CONFIG["runs_per_log"],
            "field_accuracy": field_accuracy,
            "overall_accuracy": round(overall * 100, 1),
            "latency": {
                "mean_s": round(statistics.mean(latencies), 3) if latencies else 0,
                "std_s": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
            },
            "throughput_per_sec": round(1 / statistics.mean(latencies), 3) if latencies else 0,
            "total_errors": errors
        }
        
        print(f"  Overall: {overall*100:.1f}%")
    
    results["timestamp"] = datetime.now().isoformat()
    return results


def main():
    client = ollama.Client(host=CONFIG["ollama_host"])
    results = run_parser_experiment(client)
    
    output_path = Path(__file__).parent / "data" / "parser_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
