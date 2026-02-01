# Experiment 2: Scalability Analysis

**Objective:** Demonstrate O(n) linear DAG construction complexity.

## Results

| Log Entries | Time (ms) | Std Dev |
|-------------|-----------|---------|
| 10 | 0.10 | ±0.02 |
| 100 | 0.93 | ±0.03 |
| 1,000 | 9.71 | ±0.51 |
| 2,000 | 19.68 | ±0.53 |

## How to Run

```bash
python run_experiment.py
```

## Files

- `run_experiment.py` - Experiment script
- `data/dag_scalability.json` - Raw measurements
