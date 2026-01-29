# Experiments Directory

This directory contains evaluation scripts and results for the GraphRCA system.

## Core Scripts

### `evaluate_rag_accuracy.py`
**Purpose**: Evaluates RAG-enhanced RCA accuracy on real-world incidents

**How it works**:
1. Loads 60 curated incidents
2. Splits into 75% training (knowledge base) / 25% testing
3. For each test case:
   - **Baseline**: Identify RCA using logs only (no RAG)
   - **RAG**: Retrieve similar incidents + identify RCA with context
4. Scores predictions using GPT-4o-mini as judge
5. Outputs results to `eval_final_results/real_incident_bench.json`

**Usage**:
```bash
export OPENAI_API_KEY='your-key-here'
python evaluate_rag_accuracy.py
```

### `run_gpu_eval.sh`
**Purpose**: Automated evaluation pipeline for remote GPU servers

**Steps**:
1. Checks dependencies and Ollama service
2. Generates synthetic logs for all incidents
3. Runs the RAG accuracy benchmark
4. Generates final report

**Usage**:
```bash
chmod +x run_gpu_eval.sh
./run_gpu_eval.sh
```

## Results Directory

### `eval_final_results/`
Contains all evaluation outputs:
- `real_incident_bench.json`: RAG vs. Baseline accuracy results
- Previous experiment results (Drain baseline, LLM parsing, etc.)
- Statistical analysis and LaTeX tables

## Legacy Scripts (Archive)

- `overnight_evaluation.sh`: Old evaluation script (superseded)
- `run_real_evaluation.sh`: Original evaluation approach (superseded)
- `statistical_analysis.py`: Statistical tests for experiment results

## Dependencies

```bash
pip install ollama openai
```

## For the Paper

The key result for your IEEE Access paper comes from:
- `evaluate_rag_accuracy.py` → quantitative evaluation
- `real_incident_bench.json` → accuracy metrics and improvement percentages
