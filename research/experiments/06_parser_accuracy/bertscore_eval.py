#!/usr/bin/env python3
"""
Full-dataset BERTScore evaluation of the LLM parser's free-text fields.

The paper previously reported exact-match on the full LogHub sets but
BERTScore on only a 30-entry sample — an inconsistency flagged in review.
This script closes it: it re-parses the full BGL and HDFS 2k sets with the
production LogParser, stores every (prediction, ground-truth) pair, and
computes BERTScore F1 on the free-text fields (message, component) across
ALL entries, alongside exact-match on the same pairs for consistency.

Ground-truth extraction is imported from run_experiment.py so the two
evaluations can never drift apart.

Usage:
  OLLAMA_HOST=... uv run --with ollama --with bert-score python bertscore_eval.py
  SMOKE_TEST=1 ... (32 lines per dataset)

Outputs (data/):
  parse_pairs_{BGL,HDFS}.jsonl   per-entry pairs (checkpointed, resumable)
  bertscore_report.json          aggregate report
"""

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2] / "app" / "backend"))
from app.log_parser import LogParser

# Import ground-truth extractors from the canonical experiment module.
spec = importlib.util.spec_from_file_location("exp06", HERE / "run_experiment.py")
exp06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp06)

OUT_DIR = HERE / "data"
DATASETS = {"BGL": exp06.extract_ground_truth_bgl, "HDFS": exp06.extract_ground_truth_hdfs}
MAX_SAMPLES = 32 if os.environ.get("SMOKE_TEST") else 2000
BATCH = 16


def parse_dataset(name: str, extract_gt) -> list:
    logs = exp06.load_loghub_dataset(name, MAX_SAMPLES)
    if not logs:
        raise SystemExit(f"dataset {name} unavailable")

    pairs_file = OUT_DIR / f"parse_pairs_{name}.jsonl"
    done = set()
    if pairs_file.exists():
        for line in pairs_file.read_text().splitlines():
            try:
                done.add(json.loads(line)["idx"])
            except json.JSONDecodeError:
                continue
    if done:
        print(f"{name}: resuming, {len(done)} entries already parsed")

    parser = LogParser(model="llama3.2:3b", timeout=180)
    todo = [(i, l["raw"]) for i, l in enumerate(logs)
            if i not in done and extract_gt(l["raw"]) is not None]
    print(f"{name}: parsing {len(todo)} lines (of {len(logs)} loaded)")

    for start in range(0, len(todo), BATCH):
        chunk = todo[start:start + BATCH]
        try:
            chain = parser.parse_log("\n".join(raw for _, raw in chunk))
        except Exception as e:
            print(f"  batch at {start}: failed ({type(e).__name__}) — skipped")
            continue
        # Align entries to inputs. When some lines failed validation, the
        # parser's parse_errors name their 1-based positions within this
        # chunk, so the surviving entries can still be paired correctly.
        import re as _re
        # "regex fallback used" lines still yield an entry — not failures.
        failed_pos = {int(m.group(1)) for e in chain.parse_errors
                      if "fallback used" not in e
                      for m in [_re.match(r"line (\d+):", e)] if m}
        kept_inputs = [pair for j, pair in enumerate(chunk, 1) if j not in failed_pos]
        if len(chain.log_chain) != len(kept_inputs):
            print(f"  batch at {start}: {len(chain.log_chain)} entries vs {len(kept_inputs)} "
                  f"survivors — skipped (unalignable)")
            continue
        with open(pairs_file, "a") as fh:
            for (idx, raw), entry in zip(kept_inputs, chain.log_chain):
                gt = extract_gt(raw)
                fh.write(json.dumps({
                    "idx": idx, "raw": raw,
                    "gt_message": gt["message"], "gt_component": gt["component"],
                    "pred_message": entry.message, "pred_component": entry.component,
                }) + "\n")
        print(f"  {name}: {min(start + BATCH, len(todo))}/{len(todo)}")

    if not pairs_file.exists():
        print(f"{name}: WARNING — no batches succeeded, no pairs file")
        return []
    return [json.loads(l) for l in pairs_file.read_text().splitlines()]


def main():
    OUT_DIR.mkdir(exist_ok=True)
    all_pairs = {name: parse_dataset(name, gt_fn) for name, gt_fn in DATASETS.items()}

    from bert_score import score as bert_score
    report = {"n_per_dataset": {k: len(v) for k, v in all_pairs.items()},
              "smoke_test": bool(os.environ.get("SMOKE_TEST"))}

    for name, pairs in all_pairs.items():
        report[name] = {}
        for field in ("message", "component"):
            preds = [p[f"pred_{field}"] or "" for p in pairs]
            gts = [p[f"gt_{field}"] or "" for p in pairs]
            usable = [(a, b) for a, b in zip(preds, gts) if a.strip() and b.strip()]
            if not usable:
                continue
            cands, refs = zip(*usable)
            # Raw F1 is comparable with the paper's earlier 30-sample figure;
            # baseline-rescaled F1 is more discriminative. Report both.
            _, _, f1_raw = bert_score(list(cands), list(refs), lang="en",
                                      rescale_with_baseline=False, verbose=False)
            _, _, f1_rs = bert_score(list(cands), list(refs), lang="en",
                                     rescale_with_baseline=True, verbose=False)
            exact = statistics.mean(a.strip().lower() == b.strip().lower() for a, b in usable)
            report[name][field] = {
                "n": len(usable),
                "bertscore_f1_raw": round(float(f1_raw.mean()), 4),
                "bertscore_f1_rescaled": round(float(f1_rs.mean()), 4),
                "bertscore_f1_rescaled_p25": round(float(f1_rs.quantile(0.25)), 4),
                "exact_match": round(exact, 4),
            }
            print(f"{name}/{field}: n={len(usable)} F1raw={float(f1_raw.mean()):.4f} "
                  f"F1rescaled={float(f1_rs.mean()):.4f} exact={exact:.4f}")

    (OUT_DIR / "bertscore_report.json").write_text(json.dumps(report, indent=2))
    print(f"Written: {OUT_DIR / 'bertscore_report.json'}")


if __name__ == "__main__":
    main()
