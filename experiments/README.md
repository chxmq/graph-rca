# GraphRCA Experiments

Organized experiment results for the paper.

## Folder Structure

```
experiments_organized/
├── 01_batch_inference/      # LLM batching performance (5.7× speedup)
├── 02_scalability/          # O(n) linear complexity proof
├── 03_baseline_comparison/  # 96.7% RCA accuracy across 20 scenarios
├── 04_rag_noise/            # (Failed - sqlite version issue)
├── 05_doc_ablation/         # Documentation dependency analysis
├── 06_category_rca/         # Per-category breakdown
├── 07_parser_accuracy/      # 99.6% BGL, 99.2% HDFS
├── figures/                 # Publication-ready PDF & PNG
├── scripts/                 # comprehensive_overnight.py + run log
├── latex_tables.tex         # Parser & RCA tables
├── statistical_tables.tex   # Confidence interval tables
└── EVALUATION_REPORT.md     # Full summary report
```

## Key Results

| Metric | Value | Source |
|--------|-------|--------|
| Batch 32 Throughput | 2.27 logs/s (5.7× speedup) | 01_batch_inference |
| RCA Overall Accuracy | 96.7% (88.6–99.1% CI) | 03_baseline_comparison |
| Parsing Accuracy (BGL) | 99.6% | 07_parser_accuracy |
| Parsing Accuracy (HDFS) | 99.2% | 07_parser_accuracy |
| DAG Scalability | O(n) linear | 02_scalability |

## Run Date

- Comprehensive overnight: 2026-01-31 (5h 00m 56s)
- Eval final results: 2026-01-29
