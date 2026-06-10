#!/usr/bin/env python3
"""
Analyze filled human-evaluation sheets against the answer key.

Reads every rating_sheet_*.csv in research/tools/human_eval/, de-anonymizes
slots via answer_key.json, and reports:

  - per-condition human accuracy (rating==2) and partial credit (>=1)
  - Fleiss' kappa across raters (per condition and overall)
  - Spearman correlation between mean human rating and each LLM judge's
    similarity score (validates or indicts the LLM-as-judge protocol)

Usage:
  uv run --with scipy python research/tools/analyze_human_eval.py
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import spearmanr
except ImportError:
    raise SystemExit("scipy required: uv run --with scipy python research/tools/analyze_human_eval.py")

HERE = Path(__file__).parent
EVAL_DIR = HERE / "human_eval"
EXP08_DIR = HERE.parent / "experiments" / "08_rag_real_world" / "data"
CATEGORIES = (0, 1, 2)


def fleiss_kappa(matrix: list) -> float:
    """matrix[i][c] = number of raters assigning category c to item i.
    Standard Fleiss' kappa; returns float('nan') for degenerate input."""
    n_items = len(matrix)
    if n_items == 0:
        return float("nan")
    n_raters = sum(matrix[0])
    p_j = [sum(row[c] for row in matrix) / (n_items * n_raters) for c in range(len(CATEGORIES))]
    p_i = [
        (sum(x * x for x in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in matrix
    ]
    p_bar = statistics.mean(p_i)
    p_e = sum(p * p for p in p_j)
    if p_e == 1.0:
        return float("nan")
    return (p_bar - p_e) / (1 - p_e)


def main():
    key = json.loads((EVAL_DIR / "answer_key.json").read_text())["incidents"]
    sheets = sorted(EVAL_DIR.glob("rating_sheet_*.csv"))
    if not sheets:
        raise SystemExit(f"No filled sheets (rating_sheet_<name>.csv) in {EVAL_DIR}")
    print(f"Raters: {[s.stem.replace('rating_sheet_', '') for s in sheets]}")

    # ratings[(incident, condition)] = [rating per rater]
    ratings: dict = defaultdict(list)
    for sheet in sheets:
        with open(sheet) as fh:
            for row in csv.DictReader(fh):
                inc = row["incident_id"]
                if inc not in key:
                    continue
                for slot in "ABC":
                    raw = (row.get(f"rating_{slot}") or "").strip()
                    if raw not in {"0", "1", "2"}:
                        continue
                    ratings[(inc, key[inc][slot])].append(int(raw))

    n_raters = len(sheets)
    complete = {k: v for k, v in ratings.items() if len(v) == n_raters}
    dropped = len(ratings) - len(complete)
    if dropped:
        print(f"⚠ {dropped} item-condition pairs missing ratings from some rater — excluded")

    conditions = sorted({c for _, c in complete})
    print(f"\n{'condition':<12} {'n':>4} {'acc(=2)':>8} {'partial(>=1)':>12} {'mean':>6} {'kappa':>7}")
    for cond in conditions:
        items = [(k, v) for k, v in complete.items() if k[1] == cond]
        flat = [r for _, v in items for r in v]
        acc = sum(1 for _, v in items if statistics.median(v) == 2) / len(items)
        partial = sum(1 for _, v in items if statistics.median(v) >= 1) / len(items)
        matrix = [[v.count(c) for c in CATEGORIES] for _, v in items]
        kappa = fleiss_kappa(matrix)
        print(f"{cond:<12} {len(items):>4} {acc:>8.1%} {partial:>12.1%} "
              f"{statistics.mean(flat):>6.2f} {kappa:>7.3f}")

    overall_matrix = [[v.count(c) for c in CATEGORIES] for v in complete.values()]
    print(f"\nOverall Fleiss' kappa: {fleiss_kappa(overall_matrix):.3f} "
          f"({n_raters} raters, {len(complete)} item-condition pairs)")

    # Human vs LLM-judge agreement
    score_key = {"zero_shot": "zero_shot_score", "baseline": "baseline_score", "rag": "rag_score"}
    for judge in ["gpt", "groq", "qwen"]:
        f = EXP08_DIR / f"rag_comparison_{judge}.json"
        if not f.exists():
            continue
        by_id = {t["id"]: t for t in json.loads(f.read_text())["tests"]}
        humans, llms = [], []
        for (inc, cond), votes in complete.items():
            llm = by_id.get(inc, {}).get(score_key[cond])
            if llm is None:
                continue
            humans.append(statistics.mean(votes))
            llms.append(llm)
        if len(humans) >= 10:
            rho, p = spearmanr(humans, llms)
            print(f"Human vs {judge} judge: Spearman rho={rho:.3f} (p={p:.2g}, n={len(humans)})")


if __name__ == "__main__":
    main()
