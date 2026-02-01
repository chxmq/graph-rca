# Reproducibility Checklist

## ✅ Data (Complete)
- **200 real incidents** in `data/real_incidents/incident_001` to `incident_200`
- Each incident contains:
  - `ground_truth.json` - Expected root cause
  - `logs.txt` - Synthetic log evidence
  - `metadata.json` - Company, category, date, severity
  - `postmortem.md` - Original incident description

## ✅ Experiment Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `comprehensive_overnight.py` | Main overnight runner (all experiments) | `experiments/scripts/` |
| `run_overnight.py` | GPU-optimized RCA + RAG experiments | `experiments/overnight_gpu/` |
| `run_noise_test.py` | Standalone noise sensitivity test | `experiments/overnight_gpu/` |
| `run_ieee_eval.sh` | Drain vs GraphRCA comparison | `experiments/scripts/` |
| `evaluate_rag_accuracy.py` | RAG accuracy benchmark | `experiments/scripts/` |
| `generate_paper_figures.py` | Regenerate all paper figures | `experiments/` |

## ✅ Results Data

| Experiment | Data File | Location |
|------------|-----------|----------|
| Batch Inference | `01_batch_inference.json` | `experiments/01_batch_inference/data/` |
| DAG Scalability | `dag_scalability.json` | `experiments/02_scalability/data/` |
| Drain Baseline | `01_drain_baseline.json` | `experiments/06_parser_accuracy/data/` |
| Noise Sensitivity | `03_noise_sensitivity.json` | `experiments/overnight_gpu/results/` |

## ✅ Figures
All in `experiments/figures/`:
- `fig1_throughput.png` + `.pdf`
- `fig2_field_accuracy.png` + `.pdf`
- `fig3_dag_scalability.png` + `.pdf`
- `fig4_rca_categories.png` + `.pdf`
- `fig5_ablation.png` + `.pdf`
- `fig6_summary.png` + `.pdf`

## How to Reproduce

```bash
# 1. Setup
cd graph-rca
pip install ollama chromadb drain3 pandas matplotlib

# 2. Start Ollama
ollama serve &
ollama pull llama3.2:3b
ollama pull qwen3:32b  # For scoring

# 3. Run experiments
cd experiments/scripts
python comprehensive_overnight.py  # Full suite (5-6 hours)

# 4. Regenerate figures
cd experiments
python generate_paper_figures.py
```

## Hardware Requirements
- **Minimum:** NVIDIA GPU with 8GB+ VRAM (Quadro GV100 tested)
- **Recommended:** A100-40GB for faster results
