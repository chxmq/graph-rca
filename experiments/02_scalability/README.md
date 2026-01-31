# Experiment 2: Scalability Analysis

**Objective:** Prove O(n) linear complexity for log processing.

## Results

| Log Count | Total Time | Throughput |
|-----------|------------|------------|
| 50 | 123.5s | 0.41 logs/s |
| 100 | 244.6s | 0.41 logs/s |
| 250 | 608.4s | 0.41 logs/s |
| 500 | 1,211.6s | 0.41 logs/s |
| 1,000 | 2,420.1s | 0.41 logs/s |

**Constant 0.41 logs/s throughput = O(n) linear complexity confirmed**

## DAG Construction

| Log Entries | Time (ms) | Note |
|-------------|-----------|------|
| 100 | 0.93 | |
| 1,000 | 9.71 | |
| 2,000 | 19.68 | O(n) |

## Files

- `data/scale_results.json` - Scalability test data
- `data/dag_scalability.json` - DAG construction timing
