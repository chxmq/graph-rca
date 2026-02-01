#!/usr/bin/env python3
"""
Parser Accuracy Experiment
Tests LLM parsing accuracy on LogHub datasets (BGL, HDFS).
Reports field-level accuracy for timestamp, level, component, message.
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
}

# Sample logs from BGL and HDFS datasets
BGL_LOGS = [
    {
        "raw": "- 1131511861 2005.11.09 R33-M0-N7-C:J03-U01 2005-11-09-06.11.01.134579 R33-M0-N7-C:J03-U01 RAS KERNEL INFO generating core.7681",
        "expected": {"timestamp": "2005-11-09-06.11.01", "level": "INFO", "component": "KERNEL", "message": "generating core.7681"}
    },
    {
        "raw": "- 1131511864 2005.11.09 R33-M1-N6-I:J18-U11 2005-11-09-06.11.04.524523 R33-M1-N6-I:J18-U11 RAS KERNEL INFO data TLB error interrupt",
        "expected": {"timestamp": "2005-11-09-06.11.04", "level": "INFO", "component": "KERNEL", "message": "data TLB error interrupt"}
    },
    {
        "raw": "- 1131512089 2005.11.09 R24-M0-N8-C:J14-U01 2005-11-09-06.14.49.930364 R24-M0-N8-C:J14-U01 RAS KERNEL FATAL data storage interrupt",
        "expected": {"timestamp": "2005-11-09-06.14.49", "level": "FATAL", "component": "KERNEL", "message": "data storage interrupt"}
    },
]

HDFS_LOGS = [
    {
        "raw": "081109 203615 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106",
        "expected": {"timestamp": "081109 203615", "level": "INFO", "component": "DataNode", "message": "Receiving block"}
    },
    {
        "raw": "081109 204230 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010",
        "expected": {"timestamp": "081109 204230", "level": "INFO", "component": "FSNamesystem", "message": "blockMap updated"}
    },
    {
        "raw": "081109 204432 38 WARN dfs.FSNamesystem: BLOCK* addStoredBlock: Redundant addStoredBlock request received",
        "expected": {"timestamp": "081109 204432", "level": "WARN", "component": "FSNamesystem", "message": "Redundant addStoredBlock"}
    },
]


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
        
        # Try to parse JSON
        try:
            parsed = json.loads(text)
            return {"parsed": parsed, "latency": latency, "success": True}
        except json.JSONDecodeError:
            # Try to extract JSON from response
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {"parsed": parsed, "latency": latency, "success": True}
    except Exception as e:
        latency = time.time() - start
    
    return {"parsed": {}, "latency": latency, "success": False}


def check_field_match(parsed: dict, expected: dict, field: str) -> bool:
    """Check if a field was correctly extracted."""
    if field not in parsed:
        return False
    parsed_val = str(parsed[field]).lower()
    expected_val = str(expected[field]).lower()
    # Partial match is OK for message field
    if field == "message":
        return expected_val in parsed_val or parsed_val in expected_val
    return expected_val in parsed_val or parsed_val in expected_val


def run_parser_experiment(client: ollama.Client) -> dict:
    """Test parser on LogHub datasets."""
    print("=" * 70)
    print("EXPERIMENT 6: Parser Accuracy on LogHub")
    print("=" * 70)
    
    datasets = {"BGL": BGL_LOGS, "HDFS": HDFS_LOGS}
    fields = ["timestamp", "level", "component", "message"]
    results = {"datasets": {}}
    
    for ds_name, logs in datasets.items():
        print(f"\nTesting {ds_name} dataset ({len(logs)} samples)...")
        
        field_correct = {f: 0 for f in fields}
        total_tests = 0
        latencies = []
        
        for log_entry in logs:
            for run in range(CONFIG["runs_per_log"]):
                result = parse_log_with_llm(client, log_entry["raw"])
                latencies.append(result["latency"])
                
                if result["success"]:
                    for field in fields:
                        if check_field_match(result["parsed"], log_entry["expected"], field):
                            field_correct[field] += 1
                total_tests += 1
        
        # Calculate per-field accuracy
        field_accuracy = {}
        for field in fields:
            acc = field_correct[field] / total_tests if total_tests > 0 else 0
            field_accuracy[field] = round(acc * 100, 1)
        
        overall = sum(field_correct.values()) / (total_tests * len(fields)) if total_tests > 0 else 0
        
        results["datasets"][ds_name] = {
            "samples": len(logs),
            "runs_per_sample": CONFIG["runs_per_log"],
            "total_tests": total_tests,
            "field_accuracy": field_accuracy,
            "overall_accuracy": round(overall * 100, 1),
            "mean_latency_s": round(statistics.mean(latencies), 3),
            "throughput": round(1 / statistics.mean(latencies), 2) if latencies else 0
        }
        
        print(f"  Overall: {overall*100:.1f}%")
        for f, a in field_accuracy.items():
            print(f"    {f}: {a}%")
    
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
