# Experiment 07: Multi-Judge RCA Validation

Cross-validates RCA accuracy using multiple independent LLM judges to ensure results are not biased by a single evaluation model.

## Judges

| Judge | Model | API |
|-------|-------|-----|
| **qwen** | Qwen3 32B | Local (Ollama) |
| **gpt** | GPT-4o-mini | OpenAI |
| **groq** | Llama-3.3-70B | Groq (free tier) |

## Requirements

```bash
# For local Qwen judge
ollama pull qwen3:32b

# For GPT judge
pip install openai
export OPENAI_API_KEY="your-key"

# For Groq judge (free)
pip install groq
export GROQ_API_KEY="your-key"  # https://console.groq.com/keys
```

## Usage

```bash
# Run with local Qwen (default, no API key needed)
python run_experiment.py --judge qwen

# Run with GPT-4o-mini
python run_experiment.py --judge gpt

# Run with Llama-70B via Groq
python run_experiment.py --judge groq

# Run ALL judges (comprehensive validation)
python run_experiment.py --judge all
```

## Outputs

- `data/results_qwen.json` - Qwen judge results
- `data/results_gpt.json` - GPT judge results
- `data/results_groq.json` - Groq/Llama judge results
- `data/multi_judge_summary.json` - Combined summary (when using --judge all)

## Paper Reference

Used for Table VIII: Multi-Judge Cross-Validation showing consistent accuracy across independent judges.
