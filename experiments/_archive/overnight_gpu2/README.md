# GraphRCA Overnight GPU Experiments

**Single folder, single script, fault-tolerant execution.**

## Quick Start

```bash
cd experiments/overnight_gpu
export OPENAI_API_KEY="your-key-here"
python run_overnight.py
```

## Prerequisites

1. **Ollama** running with models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

2. **ChromaDB** running on port 8000:
   ```bash
   docker run -p 8000:8000 chromadb/chroma
   # OR
   chroma run --port 8000
   ```

3. **OpenAI API Key** (for GPT-4o-mini judge):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## What It Does

| Experiment | Description | Time Est. |
|------------|-------------|-----------|
| **01_rca_accuracy** | RCA on 60 real incidents (3 runs each = 180 tests) | ~3 hrs |
| **02_rag_comparison** | Baseline vs RAG on 15 test incidents | ~1 hr |
| **03_noise_sensitivity** | Retrieval with 1000+ decoys | ~2 hrs |

## Fault Tolerance Features

- ✅ **Checkpointing**: Saves progress every 5 incidents
- ✅ **Auto-resume**: Continues from last checkpoint if crashed
- ✅ **Retries**: 3 attempts per API call with exponential backoff
- ✅ **Graceful degradation**: If one experiment fails, others continue
- ✅ **Full logging**: `results/experiment_log.txt`

## Output

```
experiments/overnight_gpu/results/
├── checkpoint.json           # Resume point
├── experiment_log.txt        # Full log
├── 01_rca_accuracy.json      # RCA results
├── 02_rag_comparison.json    # RAG vs baseline
├── 03_noise_sensitivity.json # Noise test
└── final_summary.json        # Quick summary
```

## If Something Goes Wrong

1. Check `results/experiment_log.txt` for errors
2. Just re-run `python run_overnight.py` - it will resume automatically
3. To start fresh, delete `results/checkpoint.json`
