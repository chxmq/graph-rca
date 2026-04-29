# Experiment 1: Batch Inference Performance

**Objective:** Reduce per-log overhead through LLM batching.

## Results

| Batch Size | Throughput | Latency | Speedup |
|------------|------------|---------|---------|
| 1 (baseline) | 0.40 logs/s | 2,515 ms | 1.0× |
| 8 | 0.48 logs/s | 2,069 ms | 1.2× |
| 16 | 0.82 logs/s | 1,224 ms | 2.1× |
| **32** | **2.27 logs/s** | **441 ms** | **5.7×** |

## Paper Claim

> "Batch size 32 achieves 5.7× throughput improvement (0.40 → 2.27 logs/s)"

## Files

- `data/batch_results.json` - Raw experimental data (5 runs per batch size)
