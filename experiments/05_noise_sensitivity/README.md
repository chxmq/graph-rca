# Noise Sensitivity Test

**Objective:** Evaluate retrieval accuracy when finding the correct document among many "decoy" documents.

## What This Tests

Your senior asked:
> "Report the retrieval accuracy when the system must find the correct document among 1,000 'decoy' or irrelevant documents in ChromaDB."

## How to Run

```bash
# Make sure Ollama and ChromaDB are running
cd experiments/06_noise_sensitivity
python noise_sensitivity_test.py
```

## Expected Output

The script will:
1. Index all 60 incident postmortems as "target" documents
2. Add 192 raw postmortems from `sources/raw/` as "decoy" documents  
3. For each incident, query with its logs and check if the correct postmortem is retrieved
4. Report Recall@1, Recall@3, Recall@5 metrics

## Results to Add to Paper

After running, add results to Section VI (around line 670):

```latex
\textbf{Noise Sensitivity:} Retrieval tested against 192 distractor documents 
achieved Recall@3 of X\% for finding relevant incident postmortems.
```
