# Experiment 3: RCA Baseline Comparison

**Objective:** Evaluate root cause identification accuracy across failure categories.

## Results (20 scenarios × 3 runs = 60 tests)

| Category | Scenarios | Success Rate | 95% CI |
|----------|-----------|--------------|--------|
| Database | 4 | 100.0% | (75.7–100%) |
| Security | 3 | 100.0% | (70.1–100%) |
| Application | 4 | 100.0% | (75.7–100%) |
| Monitoring | 2 | 100.0% | (61.0–100%) |
| Infrastructure | 4 | 91.7% | (64.6–98.5%) |
| Memory | 3 | 88.9% | (56.5–98.0%) |
| **Overall** | **20** | **96.7%** | **(88.6–99.1%)** |

## Paper Claim

> "GraphRCA achieves 96.7% RCA accuracy across 20 diverse failure scenarios spanning 6 categories."

## Files

- `data/04_pipeline_rca.json` - Full RCA results (60 tests)
- `data/06_statistical_analysis.json` - Statistical analysis with CIs
