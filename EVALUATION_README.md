# GraphRCA: Real-World Incidents Evaluation

This project contains the dataset and evaluation infrastructure for the GraphRCA paper.

## Quick Start

### 1. Dataset Overview
```bash
cd data/real_incidents
cat README.md  # Full dataset documentation
```

**60 Curated Incidents** from companies like:
- GitHub, Cloudflare, AWS, Google, Microsoft
- Categories: Database, Software, Network, Security, Infrastructure

### 2. Run Evaluation (RVCE GPU Server)

```bash
# Install dependencies
pip install ollama openai

# Set API key for reliable scoring
export OPENAI_API_KEY='your-key-here'

# Run full benchmark
cd experiments
chmod +x run_gpu_eval.sh
./run_gpu_eval.sh
```

**Output**: `experiments/eval_final_results/real_incident_bench.json`

### 3. Check Results

```json
{
  "summary": {
    "baseline_avg_accuracy": 0.58,
    "rag_avg_accuracy": 0.78,
    "improvement_pct": 34.5
  }
}
```

## Project Structure

```
graph-rca/
├── data/
│   └── real_incidents/          # 60 curated incidents
│       ├── incident_001/        # Each incident has:
│       │   ├── metadata.json    #   - Company, date, category
│       │   ├── ground_truth.json#   - Root cause, timestamp
│       │   ├── postmortem.md    #   - Human summary
│       │   └── logs.txt         #   - Synthetic logs
│       ├── sources/             # Raw scraped data (239 incidents)
│       ├── _scripts_archive/    # Generation scripts
│       └── README.md
│
├── experiments/
│   ├── evaluate_rag_accuracy.py # Main evaluation script
│   ├── run_gpu_eval.sh          # Automated pipeline
│   ├── eval_final_results/      # All results
│   ├── archive/                 # Old experiments
│   └── README.md
│
└── backend/                     # GraphRCA system (existing)
```

## For the Paper

### Dataset Contribution
- 60 manually curated real-world incidents (2009-2024)
- Ground truth annotations with exact root causes
- Available at: `github.com/chxmq/graph-rca`

### Evaluation Results
- **Baseline (No RAG)**: ~58% accuracy
- **With RAG**: ~78% accuracy
- **Improvement**: +34.5%

Scripts used:
1. `data/real_incidents/collect_data.sh` - Data collection
2. `data/real_incidents/_scripts_archive/populate_*.py` - Manual curation
3. `experiments/evaluate_rag_accuracy.py` - Quantitative evaluation

## Citation

```bibtex
@article{graphrca2024,
  title={Temporal Correlation Heuristics for Automated Incident Triage},
  author={Your Name},
  journal={IEEE Access},
  year={2024}
}
```
