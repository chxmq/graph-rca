# GraphRCA Experiments

Each folder contains `run_experiment.py` + `data/` with results.

## Run All Experiments

```bash
python run_all_experiments.py
```

## Experiments

| # | Folder | Experiment | Key Finding |
|---|--------|------------|-------------|
| 01 | `01_batch_inference/` | Batch Processing | 5.7× speedup (batch 32) |
| 02 | `02_scalability/` | DAG Scalability | O(n) linear, ~10μs/entry |
| 03 | `03_baseline_comparison/` | RCA Accuracy | 96.7% (20 scenarios) |
| 04 | `04_doc_ablation/` | Documentation Ablation | 4.8pp with docs |
| 05 | `05_noise_sensitivity/` | Noise Sensitivity | 95%→65% retrieval |
| 06 | `06_parser_accuracy/` | Parser Accuracy | 99.6% BGL, 99.2% HDFS |
| 07 | `07_multi_judge_validation/` | Multi-Judge | 77.3-84.9% (3 judges) |
| 08 | `08_rag_real_world/` | RAG Evaluation | 72.6% baseline vs 64.9% RAG |

## Quick Start

```bash
cd experiments/01_batch_inference
python run_experiment.py
```

## Requirements

- Python 3.13
- `pip install ollama chromadb`
- Ollama running with `llama3.2:3b`
