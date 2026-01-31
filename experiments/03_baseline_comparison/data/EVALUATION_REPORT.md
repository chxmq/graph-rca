# GraphRCA Comprehensive Evaluation Report

**Generated:** 2026-01-29  
**Datasets:** LogHub BGL (2000 logs), HDFS (2000 logs)  
**Model:** Llama 3.2 3B  
**Statistical Runs:** 3 per configuration

---

## Executive Summary

| Metric | BGL | HDFS |
|--------|-----|------|
| **Parsing Accuracy** | 99.6% | 99.2% |
| **Throughput** | 0.45 logs/s | 0.36 logs/s |
| **Avg Latency** | 2.21s | 2.76s |

**RCA Success Rate:** 96.7% across 20 scenarios (60 total tests)

---

## 1. Parser Comparison

### Throughput

| Dataset | Drain (Baseline) | GraphRCA (Ours) | Ratio |
|---------|------------------|-----------------|-------|
| BGL | 131,154 logs/s | 0.45 logs/s | ~290,000x |
| HDFS | 12,954 logs/s | 0.36 logs/s | ~36,000x |

**Trade-off:** GraphRCA trades throughput for zero-configuration deployment (no templates required).

### Latency

| Dataset | Drain | GraphRCA |
|---------|-------|----------|
| BGL | 0.01ms | 2,210ms |
| HDFS | 0.08ms | 2,759ms |

---

## 2. Field-Level Parsing Accuracy

| Field | BGL | HDFS |
|-------|-----|------|
| Timestamp | 100.0% ± 0.0 | 100.0% ± 0.0 |
| Severity Level | 99.8% ± 0.0 | 96.8% ± 0.1 |
| Component | 98.7% ± 0.0 | 100.0% ± 0.0 |
| Message | 100.0% ± 0.0 | 100.0% ± 0.0 |
| **Overall** | **99.6%** | **99.2%** |

---

## 3. Root Cause Identification

### By Category (20 scenarios × 3 runs = 60 tests)

| Category | Scenarios | Success Rate |
|----------|-----------|--------------|
| Database | 4 | **100.0%** |
| Security | 3 | **100.0%** |
| Application | 4 | **100.0%** |
| Monitoring | 2 | **100.0%** |
| Infrastructure | 4 | 91.7% |
| Memory | 3 | 88.9% |
| **Overall** | **20** | **96.7%** |

### Detailed Scenarios

**Database (100%):**
- DB Connection Pool Exhaustion ✓
- Database Deadlock ✓
- Database Replication Lag ✓
- Query Performance Degradation ✓

**Security (100%):**
- Brute Force Attack ✓
- SSL Certificate Expired ✓
- Token Expiration Cascade ✓

**Application (100%):**
- Config Deployment Error ✓
- Cascading Microservice Failure ✓
- Thread Pool Exhaustion ✓
- Retry Storm ✓

**Monitoring (100%):**
- Alert Fatigue Cascade ✓
- Metrics Pipeline Failure ✓

**Infrastructure (91.7%):**
- Disk Space Exhaustion ✓
- Network Partition ✓
- CPU Throttling ✓
- DNS Resolution Failure (1/3 runs failed)

**Memory (88.9%):**
- Memory Leak OOM ✓
- Memory Fragmentation ✓
- Cache Eviction Storm (1/3 runs failed)

---

## 4. DAG Construction Scalability

| Log Entries | Time (ms) | Std Dev |
|-------------|-----------|---------|
| 10 | 0.10 | ± 0.02 |
| 25 | 0.24 | ± 0.01 |
| 50 | 0.47 | ± 0.02 |
| 100 | 0.93 | ± 0.03 |
| 250 | 2.33 | ± 0.04 |
| 500 | 4.71 | ± 0.06 |
| 1,000 | 9.71 | ± 0.51 |
| 2,000 | 19.68 | ± 0.53 |

**Complexity:** O(n) - Linear scaling confirmed

---

## 5. Ablation Study

All prompt configurations achieved 100% on test scenarios:

| Configuration | Success Rate |
|---------------|-------------|
| Full Pipeline (Ours) | 100% |
| No Structure | 100% |
| No Causal Chain | 100% |
| Verbose Engineer | 100% |

---

## Files Generated

**Data:**
- `01_drain_baseline.json` - Drain parser baseline metrics
- `02_llm_parsing.json` - LLM parsing with field-level accuracy
- `03_dag_scalability.json` - DAG construction scalability
- `04_pipeline_rca.json` - Full pipeline RCA results (60 tests)
- `05_ablation_study.json` - Ablation study results

**Publication Materials:**
- `latex_tables.tex` - 4 LaTeX tables ready for paper
- `figures/fig1_throughput.pdf` - Throughput comparison
- `figures/fig2_field_accuracy.pdf` - Field-level accuracy
- `figures/fig3_dag_scalability.pdf` - DAG scalability
- `figures/fig4_rca_categories.pdf` - RCA by category
- `figures/fig5_ablation.pdf` - Ablation study
- `figures/fig6_summary.pdf` - Summary dashboard

---

## Key Claims for Paper

1. "GraphRCA achieves **99.6%** parsing accuracy on BGL and **99.2%** on HDFS without predefined templates."

2. "Root cause identification achieves **96.7%** accuracy across 20 diverse failure scenarios spanning 6 categories."

3. "DAG construction exhibits **O(n) linear complexity**, completing in under 20ms for 2000 log entries."

4. "Four out of six failure categories achieve **100% RCA accuracy** (Database, Security, Application, Monitoring)."
