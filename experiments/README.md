# GraphRCA Experiments

Each folder contains `run_experiment.py` + `data/` with results.

## Run All Experiments

```bash
python run_all_experiments.py
```

## Experiments

| # | Folder | Experiment | Paper Claim |
|---|--------|------------|-------------|
| 01 | `01_batch_inference/` | Batch Processing | 5.7× throughput speedup (batch 32) |
| 02 | `02_scalability/` | DAG Scalability | O(n) linear, ~10μs/entry, <20ms for 2000 entries |
| 03 | `03_baseline_comparison/` | RCA Baseline | GraphRCA vs Simple LLM on 200 incidents |
| 04 | `04_doc_ablation/` | Documentation Ablation | +4.8pp with documentation |
| 05 | `05_noise_sensitivity/` | Noise Sensitivity | 95%→65% retrieval at 5× noise |
| 06 | `06_parser_accuracy/` | Parser Accuracy | 99.6% BGL, 99.2% HDFS (LogHub) |
| 07 | `07_multi_judge_validation/` | Multi-Judge RCA | 77.3-84.9% across 3 LLM judges |
| 08 | `08_rag_real_world/` | RAG Evaluation | RAG paradox: -14.4pp degradation |
| 09 | `09_latency_profiling/` | Latency Breakdown | LLM parsing 89.2% of total time |

## Dataset

- **200 real-world incidents** from 50+ companies (GitHub, Cloudflare, AWS, Google, etc.)
- **LogHub benchmarks**: BGL (2000 logs), HDFS (2000 logs)

## Quick Start

```bash
cd experiments/01_batch_inference
python run_experiment.py
```

## Requirements

- Python 3.13
- `pip install ollama chromadb pydantic`
- Ollama running with `llama3.2:3b` and `qwen3:32b` (for judging)

