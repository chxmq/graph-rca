# Experiment 8: RAG Real-World Evaluation

**Objective:** Evaluate RAG vs baseline on 200 real-world incidents.

## Results (50 test incidents)

| Condition | Accuracy | Change |
|-----------|----------|--------|
| Baseline (no RAG) | 72.6% | - |
| With RAG | 64.9% | -7.7% |

> **Finding:** RAG shows heterogeneous effects - helps ambiguous cases, hurts clear ones.

## How to Run

```bash
python run_experiment.py
```

## Files

- `run_experiment.py` - Experiment script
- `data/02_rag_comparison.json` - Per-incident results
