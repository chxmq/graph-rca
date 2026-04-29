# Experiment 2: Correlation Graph Construction Scalability

**Objective:** Measure O(n) linear complexity of correlation graph construction. This experiment times **only** the graph-building step—no LLM parsing.

## What It Measures

- **Correlation graph construction** only: linking pre-parsed log entries when explicit correlation keys match
- Results reflect graph construction cost, not evidence of semantic causality by itself

## What It Does NOT Measure

- LLM parsing throughput (~500ms per log) is measured in Experiment 1 (batch inference)

## How to Run

From project root with backend dependencies installed:

```bash
pip install -r backend/requirements.txt
python experiments/02_scalability/run_experiment.py
```

## Expected Results

| Log Entries | Time (ms) | Std Dev |
|-------------|-----------|---------|
| 10 | ~0.10 | ±0.02 |
| 100 | ~0.93 | ±0.03 |
| 1,000 | ~9.71 | ±0.51 |
| 2,000 | ~19.68 | ±0.53 |

## Files

- `run_experiment.py` - Experiment script
- `data/dag_scalability.json` - Raw measurements
