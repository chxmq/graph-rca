# Related work: honest RCA numbers across the field

Collected June 2026 to position GraphRCA's post-leakage-fix results.
Conclusion up front: **low absolute scores on honest RCA benchmarks are the
field-wide norm; our pre-fix 69.7% was the anomaly, not our post-fix numbers.**

## Benchmark reality check

| Work | Setup | Result |
|---|---|---|
| OpenRCA (Microsoft, ICLR 2025) | GPT-4o + telemetry, real failures | **9% accuracy**; best agentic setup ~13%; frontier 2026 models ~36% |
| ITBench (IBM, 2025) | SOTA agents, K8s incident diagnosis | **11.4% solve rate**; mid-2026 frontier with heavy agentic scaffolding still <50% |
| Zhang et al. (Microsoft, in-context RCA with GPT-4, arXiv:2401.13810) | GPT-4 + retrieved similar incidents, ~real Microsoft incidents | ROUGE-L **19.89**, BERTScore 84.91; human (incident-owner) correctness **2.47/5**; only 58.3% score >3 even after excluding confounded cases |
| **GraphRCA (ours, post-fix)** | llama3.2:3b on-prem, single-shot, symptom-only logs | mean judge similarity ~0.21 (gpt judge), real-log incidents 0.4–0.9 |

Notes:
- Zhang et al. zero-shot GPT-4: ROUGE-L 10.27 → 19.89 with 20 retrieved
  in-context incidents (**+93.7% relative**). Relevant retrieval beats random
  examples by **+41.2%**. This is direct independent support for our exp08
  RAG hypothesis (retrieval is the mechanism that bridges symptoms → cause).
- Zhang et al. found 10–20 retrieved examples optimal; performance degrades
  beyond 20 (context distraction — converges with our RAG-Information
  Paradox). Our exp08 uses top-1; a top-k ablation (k ∈ {1, 3, 10}) would
  directly connect to their finding.
- OpenRCA explicitly frames RCA as unsolved for LLMs; ITBench-AA is one of
  the least saturated agentic benchmarks.

## Positioning paragraphs (draft material)

1. Our measured drop from 69.7% (leaky logs) to ~2–20% (symptom-only logs)
   on identical incidents quantifies how answer leakage in synthetic
   telemetry inflates RCA evaluation — a methodological hazard for any
   study generating logs conditioned on known causes. Post-fix, our absolute
   numbers align with GPT-4-class systems on honest benchmarks (OpenRCA 9%,
   ITBench 11.4%, Microsoft incident-owner correctness 2.47/5) despite using
   a 3B on-premises model.
2. The evidence-explicitness stratification (real-log incidents 0.4–0.9 vs
   symptom-only ~0.2) shows the pipeline extracts what telemetry contains;
   absolute ceiling is set by what telemetry CAN contain.
3. Improvement roadmap grounded in the literature:
   - retrieval count ablation (top-k 1→10–20, per Zhang et al.)
   - agentic/iterative diagnosis loops (per OpenRCA findings)
   - model-scale ablation (3B vs 32B pipeline, same benchmark)
   - multi-modal telemetry (metrics/traces) — ITBench direction

## Links
- OpenRCA: https://github.com/microsoft/OpenRCA / https://openreview.net/forum?id=M4qNIzQYpd
- ITBench-AA: https://huggingface.co/blog/ibm-research/itbench-aa / https://openreview.net/forum?id=jP59rz1bZk
- Zhang et al.: https://arxiv.org/abs/2401.13810
