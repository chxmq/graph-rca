# Real Incident RAG Evaluation

## Running on RVCE GPU Server

1. **Install OpenAI library**:
```bash
pip install openai
```

2. **Set your API key**:
```bash
export OPENAI_API_KEY='your-key-here'
```

3. **Run the benchmark**:
```bash
cd experiments
./run_gpu_eval.sh
```

## What It Does

- Uses local Llama-3.2:3b to generate RCA predictions
- Uses GPT-4o-mini to **score** how accurate those predictions are
- Compares Baseline (no RAG) vs. RAG-enhanced accuracy
- Generates report in `eval_final_results/real_incident_bench.json`

## Cost Estimate

- ~60 scoring calls × 2 (baseline + RAG) = 120 API calls
- Cost: ~$0.50-1.00 total (GPT-4o-mini is very cheap)

## Expected Improvement

With GPT-4o-mini as judge, you should see accurate scoring and likely see RAG **improving** accuracy by 15-30%.
