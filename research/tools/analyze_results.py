#!/usr/bin/env python3
"""
Statistical analysis for GraphRCA experiment results (exp07 + exp08).

Everything the paper reports must be derivable by running this script —
no ad-hoc notebook math.  Produces:

  - per-judge metrics with bootstrap 95% confidence intervals
  - inter-judge agreement (pairwise Spearman on per-incident scores)
  - stratification: real vs synthetic logs, category, parse quality
  - exp08 paired tests: Wilcoxon signed-rank for RAG vs baseline and
    baseline vs zero-shot, per judge and pooled
  - markdown report + machine-readable JSON

Usage:
  uv run --with scipy python research/tools/analyze_results.py
  (expects results in research/experiments/{07,08}*/data/)
"""

import json
import random
import statistics
from pathlib import Path

try:
    from scipy.stats import wilcoxon, spearmanr
except ImportError:
    raise SystemExit("scipy required: uv run --with scipy python research/tools/analyze_results.py")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP07_DIR = PROJECT_ROOT / "experiments" / "07_multi_judge_validation" / "data"
EXP08_DIR = PROJECT_ROOT / "experiments" / "08_rag_real_world" / "data"
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"
JUDGES = ["gpt", "groq", "qwen"]
BOOTSTRAP_N = 10_000
SEED = 42


def bootstrap_ci(values: list, stat=statistics.mean, n: int = BOOTSTRAP_N) -> tuple:
    """Percentile bootstrap 95% CI."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(SEED)
    stats_ = sorted(
        stat([rng.choice(values) for _ in values]) for _ in range(n)
    )
    return (round(stats_[int(0.025 * n)], 4), round(stats_[int(0.975 * n)], 4))


def load_real_log_flags() -> dict:
    flags = {}
    for d in sorted(INCIDENT_DIR.glob("incident_*")):
        gt = d / "ground_truth.json"
        if gt.exists():
            flags[d.name] = bool(json.loads(gt.read_text()).get("logs_available", False))
    return flags


def analyze_exp07(report: dict, lines: list) -> None:
    real_flags = load_real_log_flags()
    per_judge_scores: dict = {}

    lines.append("## Experiment 07 — RCA accuracy (200 incidents)\n")
    for judge in JUDGES:
        f = EXP07_DIR / f"results_{judge}.json"
        if not f.exists():
            lines.append(f"- {judge}: results missing, skipped\n")
            continue
        data = json.loads(f.read_text())
        rows = [r for r in data["incidents"] if r.get("avg_score") is not None]
        scores = [r["avg_score"] for r in rows]
        line_scores = [r["line_score"] for r in data["incidents"] if r.get("line_score") is not None]
        per_judge_scores[judge] = {r["id"]: r["avg_score"] for r in rows}

        mean_ci = bootstrap_ci(scores)
        acc7 = [1 if s >= 0.7 else 0 for s in scores]
        acc7_ci = bootstrap_ci(acc7)

        entry = {
            "n_scored": len(scores),
            "unscored": data.get("unscored_incidents", 0),
            "narrative": {
                "mean": round(statistics.mean(scores), 4) if scores else 0,
                "mean_ci95": mean_ci,
                "acc@0.7": round(sum(acc7) / len(acc7), 4) if acc7 else 0,
                "acc@0.7_ci95": acc7_ci,
                "acc@0.5": round(sum(1 for s in scores if s >= 0.5) / len(scores), 4) if scores else 0,
            },
            "line": {
                "mean": round(statistics.mean(line_scores), 4) if line_scores else 0,
                "acc@0.7": round(sum(1 for s in line_scores if s >= 0.7) / len(line_scores), 4) if line_scores else 0,
            },
        }

        # Stratification: real vs synthetic logs
        real = [r["avg_score"] for r in rows if real_flags.get(r["id"])]
        synth = [r["avg_score"] for r in rows if not real_flags.get(r["id"])]
        entry["strata"] = {
            "real_logs": {"n": len(real), "mean": round(statistics.mean(real), 4) if real else None},
            "synthetic_logs": {"n": len(synth), "mean": round(statistics.mean(synth), 4) if synth else None},
        }
        # Stratification: parse quality
        clean = [r["avg_score"] for r in rows if r.get("parse_stats", {}).get("parse_errors", 1) == 0]
        noisy = [r["avg_score"] for r in rows if r.get("parse_stats", {}).get("parse_errors", 0) > 0]
        entry["strata"]["clean_parse"] = {"n": len(clean), "mean": round(statistics.mean(clean), 4) if clean else None}
        entry["strata"]["noisy_parse"] = {"n": len(noisy), "mean": round(statistics.mean(noisy), 4) if noisy else None}
        # Per category means
        cats: dict = {}
        for r in rows:
            cats.setdefault(r.get("category", "Unknown"), []).append(r["avg_score"])
        entry["by_category"] = {c: {"n": len(v), "mean": round(statistics.mean(v), 4)} for c, v in sorted(cats.items())}

        report["exp07"][judge] = entry
        nar = entry["narrative"]
        lines.append(
            f"- **{judge}** (n={entry['n_scored']}): narrative mean {nar['mean']} "
            f"(CI95 {nar['mean_ci95']}), acc@0.7 {nar['acc@0.7']:.1%} (CI95 {nar['acc@0.7_ci95']}); "
            f"line mean {entry['line']['mean']} | real logs {entry['strata']['real_logs']['mean']} "
            f"vs synthetic {entry['strata']['synthetic_logs']['mean']}\n"
        )

    # Inter-judge agreement (Spearman on shared incidents)
    pairs = [(a, b) for i, a in enumerate(JUDGES) for b in JUDGES[i + 1:]
             if a in per_judge_scores and b in per_judge_scores]
    if pairs:
        lines.append("\n### Inter-judge agreement (Spearman, per-incident narrative scores)\n")
        report["exp07"]["inter_judge"] = {}
        for a, b in pairs:
            shared = sorted(set(per_judge_scores[a]) & set(per_judge_scores[b]))
            if len(shared) < 10:
                continue
            rho, p = spearmanr([per_judge_scores[a][i] for i in shared],
                               [per_judge_scores[b][i] for i in shared])
            report["exp07"]["inter_judge"][f"{a}-{b}"] = {"rho": round(float(rho), 3), "p": float(p), "n": len(shared)}
            lines.append(f"- {a} vs {b}: rho={rho:.3f} (p={p:.2g}, n={len(shared)})\n")


def analyze_exp08(report: dict, lines: list) -> None:
    lines.append("\n## Experiment 08 — zero-shot vs pipeline vs RAG (holdout set)\n")
    pooled_base, pooled_rag, pooled_zero = [], [], []
    for judge in JUDGES:
        f = EXP08_DIR / f"rag_comparison_{judge}.json"
        if not f.exists():
            lines.append(f"- {judge}: results missing, skipped\n")
            continue
        data = json.loads(f.read_text())
        tests = [t for t in data["tests"]
                 if t.get("baseline_score") is not None and t.get("rag_score") is not None]
        base = [t["baseline_score"] for t in tests]
        rag = [t["rag_score"] for t in tests]
        zero = [t["zero_shot_score"] for t in tests if t.get("zero_shot_score") is not None]
        pooled_base += base; pooled_rag += rag; pooled_zero += zero

        entry = {
            "n": len(tests),
            "zero_shot_mean": round(statistics.mean(zero), 4) if zero else None,
            "baseline_mean": round(statistics.mean(base), 4) if base else None,
            "rag_mean": round(statistics.mean(rag), 4) if rag else None,
            "rag_delta_mean": round(statistics.mean([r - b for r, b in zip(rag, base)]), 4) if base else None,
            "rag_delta_ci95": bootstrap_ci([r - b for r, b in zip(rag, base)]),
        }
        # Paired Wilcoxon (zero_diff handling: skip if all deltas are zero)
        deltas = [r - b for r, b in zip(rag, base)]
        if any(d != 0 for d in deltas):
            stat, p = wilcoxon(rag, base)
            entry["wilcoxon_rag_vs_baseline_p"] = float(p)
        zb = [(b, z) for b, z in zip(base, zero)] if len(zero) == len(base) else []
        if zb and any(b != z for b, z in zb):
            stat, p = wilcoxon(base, zero)
            entry["wilcoxon_baseline_vs_zeroshot_p"] = float(p)

        report["exp08"][judge] = entry
        lines.append(
            f"- **{judge}** (n={entry['n']}): zero {entry['zero_shot_mean']} -> base {entry['baseline_mean']} "
            f"-> RAG {entry['rag_mean']} | RAG delta {entry['rag_delta_mean']} (CI95 {entry['rag_delta_ci95']}), "
            f"Wilcoxon p={entry.get('wilcoxon_rag_vs_baseline_p', 'n/a')}\n"
        )

    if pooled_base and any(r != b for r, b in zip(pooled_rag, pooled_base)):
        stat, p = wilcoxon(pooled_rag, pooled_base)
        report["exp08"]["pooled"] = {
            "n": len(pooled_base),
            "wilcoxon_rag_vs_baseline_p": float(p),
            "rag_delta_mean": round(statistics.mean([r - b for r, b in zip(pooled_rag, pooled_base)]), 4),
        }
        lines.append(f"\n**Pooled across judges** (n={len(pooled_base)}): RAG vs baseline Wilcoxon p={p:.4g}\n")


def main():
    report = {"exp07": {}, "exp08": {}}
    lines = ["# GraphRCA results analysis\n",
             f"(generated by analyze_results.py; bootstrap n={BOOTSTRAP_N}, seed={SEED})\n"]
    analyze_exp07(report, lines)
    analyze_exp08(report, lines)

    out_json = Path(__file__).parent / "analysis_report.json"
    out_md = Path(__file__).parent / "analysis_report.md"
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text("".join(lines))
    print("".join(lines))
    print(f"\nWritten: {out_md} and {out_json}")


if __name__ == "__main__":
    main()
