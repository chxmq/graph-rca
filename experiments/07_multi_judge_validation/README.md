# Experiment 7: Multi-Judge Validation

**Objective:** Cross-validate RCA accuracy using independent LLM judges.

## Results (200 incidents)

| Judge | RCA Accuracy |
|-------|--------------|
| Qwen3:32b | 77.3% |
| Llama-70B (Groq) | 81.5% |
| GPT-4o-mini | 84.9% |
| **Mean** | **81.2%** |

## How to Run

```bash
python run_experiment.py
```

## Files

- `run_experiment.py` - Experiment script
- `data/01_rca_accuracy.json` - Full 200-incident results
