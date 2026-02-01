# Experiment 08: RAG Real-World Evaluation with Multi-Judge Validation

Compares baseline (no RAG) vs RAG-enhanced RCA accuracy on real-world incidents.

## Multi-Judge Support

Uses three independent LLM judges for cross-validation:
- **Qwen 32B** (local via Ollama) - default
- **Llama 3.3 70B** (Groq API) - requires `GROQ_API_KEY`
- **GPT-4o-mini** (OpenAI API) - requires `OPENAI_API_KEY`

## Usage

```bash
# Single judge (default: qwen)
python run_experiment.py --judge qwen

# Run all judges
python run_experiment.py --judge all

# API judges (require keys)
export GROQ_API_KEY="your-key"
python run_experiment.py --judge groq

export OPENAI_API_KEY="your-key"
python run_experiment.py --judge gpt
```

## Key Finding

RAG shows **heterogeneous effects**:
- Helps when logs are ambiguous (up to +100pp)
- Hurts when logs already contain clear evidence (up to -100pp)
- Average degradation of ~14pp across all judges

## Additional Scripts

- `enrich_with_companies.py` - Adds company names to results for analysis

## Output

- `data/rag_comparison_qwen.json` - Per-judge results
- `data/rag_comparison_all_judges.json` - Combined summary
