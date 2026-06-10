#!/usr/bin/env python3
"""
Generate blind human-evaluation sheets from experiment 08 results.

For each holdout incident, the three system outputs (zero-shot, pipeline
baseline, RAG) are shuffled into anonymous slots A/B/C so raters cannot
favor a condition.  The slot->condition mapping is written to a separate
answer key that raters must NOT see.

Outputs (research/tools/human_eval/):
  rating_sheet.csv   — one row per incident: ground truth + predictions A/B/C
                       and empty rating columns.  Give an identical copy to
                       each rater; they fill rating_A/B/C with 0, 1, or 2.
  answer_key.json    — slot->condition mapping per incident.  KEEP HIDDEN.
  INSTRUCTIONS.md    — rating rubric for the raters.

Rubric: 2 = correct root cause, 1 = partially correct / right direction,
0 = wrong or unrelated.

Usage (after exp08 has produced rag_comparison_<judge>.json):
  python research/tools/make_human_eval_sheets.py
"""

import csv
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP08_DIR = PROJECT_ROOT / "experiments" / "08_rag_real_world" / "data"
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"
OUT_DIR = Path(__file__).parent / "human_eval"
SEED = 1337  # different from the experiment seed on purpose
CONDITIONS = ["zero_shot_pred", "baseline_pred", "rag_pred"]

INSTRUCTIONS = """# Human evaluation instructions

You are rating root-cause predictions for real production incidents.

For each row in rating_sheet.csv you see the incident's documented ground
truth and three anonymous predictions (A, B, C). For EACH prediction enter:

  2 — correct: identifies the actual root cause
  1 — partially correct: right direction/component but incomplete or vague
  0 — wrong: incorrect or unrelated to the actual cause

Rules:
- Rate each prediction independently; ties are fine.
- Do not discuss ratings with other raters until everyone has finished.
- Do not try to guess which system produced which prediction.
- Save your filled sheet as rating_sheet_<yourname>.csv in this folder.
"""


def main():
    # Use any judge's results file — predictions are judge-independent for
    # baseline/rag; zero-shot differs per judge, so we take each judge's
    # zero-shot only when consistent, preferring the gpt file.
    source = None
    for judge in ["gpt", "groq", "qwen"]:
        f = EXP08_DIR / f"rag_comparison_{judge}.json"
        if f.exists():
            source = f
            break
    if source is None:
        raise SystemExit(f"No exp08 results found in {EXP08_DIR} — run exp08 first.")

    data = json.loads(source.read_text())
    tests = [t for t in data["tests"] if all(t.get(c) for c in CONDITIONS)]
    if not tests:
        raise SystemExit("exp08 results contain no usable predictions "
                         "(rerun exp08 with the full-prediction format).")

    rng = random.Random(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    key = {"source_file": source.name, "seed": SEED, "incidents": {}}

    with open(OUT_DIR / "rating_sheet.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["incident_id", "ground_truth",
                         "prediction_A", "prediction_B", "prediction_C",
                         "rating_A", "rating_B", "rating_C"])
        for t in tests:
            gt_file = INCIDENT_DIR / t["id"] / "ground_truth.json"
            ground_truth = json.loads(gt_file.read_text())["root_cause"] if gt_file.exists() else ""
            conds = CONDITIONS.copy()
            rng.shuffle(conds)
            key["incidents"][t["id"]] = {slot: cond.replace("_pred", "")
                                         for slot, cond in zip("ABC", conds)}
            writer.writerow([t["id"], ground_truth,
                             t[conds[0]], t[conds[1]], t[conds[2]], "", "", ""])

    (OUT_DIR / "answer_key.json").write_text(json.dumps(key, indent=2))
    (OUT_DIR / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    print(f"Wrote {len(tests)} incidents to {OUT_DIR}/rating_sheet.csv")
    print("Answer key (do not share with raters):", OUT_DIR / "answer_key.json")


if __name__ == "__main__":
    main()
