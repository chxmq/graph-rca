# GraphRCA Comprehensive Experiment Results

**Generated:** 2026-02-01T06:56:16.705517  
**Model:** llama3.2:3b

---

## 1. Batch Inference Performance

| Batch Size | Throughput | Latency | Speedup |
|------------|------------|---------|--------|
| 1 (baseline) | 1.90 logs/s | 527 ms | 1.0× |
| 8 | 2.66 logs/s | 376 ms | 1.4× |
| 16 | 4.67 logs/s | 214 ms | 2.5× |
| 32 | 12.18 logs/s | 82 ms | 6.4× |

**Key Finding:** Batch size 32 achieves 6.4× speedup.

---

## 2. Scalability Analysis

| Log Count | Total Time | Throughput |
|-----------|------------|------------|
| 50 | 28.0s | 1.78 logs/s |
| 100 | 55.7s | 1.79 logs/s |
| 250 | 138.7s | 1.80 logs/s |
| 500 | 277.5s | 1.80 logs/s |
| 1000 | 553.3s | 1.81 logs/s |

---

## 3. Baseline Comparison

| Method | Accuracy | p-value | Significance |
|--------|----------|---------|-------------|
| **GraphRCA** | **47.2%** | - | - |
| Simple_LLM | 66.7% | <0.05 | * |
| Temporal_Heuristic | 58.3% | <0.1 | ns |
| Frequency_Anomaly | 41.7% | <0.1 | ns |
| Random | 36.1% | <0.1 | ns |

---

## 4. RAG Noise Sensitivity

| Decoy Docs | Total Docs | Accuracy |
|------------|------------|----------|
| 100 | 103 | 100.0% |
| 500 | 503 | 100.0% |
| 1000 | 1003 | 100.0% |
| 2000 | 2003 | 100.0% |

---

## 5. Documentation Ablation

| Configuration | Docs | Accuracy |
|---------------|------|----------|
| full_docs | 3 | 93.3% |
| partial_docs | 1 | 86.7% |
| no_docs | 0 | 100.0% |

---

## 6. Category-wise RCA

| Category | Scenarios | Accuracy |
|----------|-----------|----------|
| Database | 2 | 50.0% |
| Security | 2 | 50.0% |
| Application | 2 | 50.0% |
| Infrastructure | 2 | 50.0% |
| Memory | 2 | 33.3% |
| Monitoring | 2 | 50.0% |

---

## LaTeX Tables

See `all_latex_tables.tex` for copy-paste ready tables.
