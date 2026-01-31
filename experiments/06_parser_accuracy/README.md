# Experiment 7: Parser Accuracy

**Objective:** Evaluate LLM parser accuracy on LogHub datasets.

## Results

| Dataset | Overall Accuracy | Throughput |
|---------|-----------------|------------|
| BGL | 99.6% | 0.45 logs/s |
| HDFS | 99.2% | 0.36 logs/s |

### Field-Level Accuracy

| Field | BGL | HDFS |
|-------|-----|------|
| Timestamp | 100.0% ± 0.0 | 100.0% ± 0.0 |
| Severity Level | 99.8% ± 0.0 | 96.8% ± 0.1 |
| Component | 98.7% ± 0.0 | 100.0% ± 0.0 |
| Message | 100.0% ± 0.0 | 100.0% ± 0.0 |

## Paper Claim

> "GraphRCA achieves 99.6% parsing accuracy on BGL and 99.2% on HDFS without predefined templates."

## Files

- `data/01_drain_baseline.json` - Drain parser baseline
- `data/02_llm_parsing.json` - LLM parsing results
