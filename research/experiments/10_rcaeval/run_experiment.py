#!/usr/bin/env python3
"""
Experiment 10: GraphRCA's structural core on RCAEval (real telemetry).

RCAEval (Pham et al., ASE'24/WWW'25; https://github.com/phamquiluan/RCAEval)
provides failure cases captured from real running microservice systems
(Online Boutique, Sock Shop, Train Ticket) with injected faults.  Each case
contains real logs (logs.csv), the fault injection timestamp, and the ground
truth root-cause service encoded in the directory name
({service}_{fault}/{instance}).

What this evaluates: the deterministic GraphRCA core — log structuring,
per-component correlation chains (GraphGenerator), and graph-aware root
attribution — extended from single root-cause selection to a ranked list of
suspect services.  No LLM is involved: parsing quality is evaluated in
exp06 and narrative generation in exp07/08; this experiment isolates the
structural attribution logic on REAL logs with objective exact-match
scoring, immune to LLM-judge circularity.

Ranking rule (a direct generalization of app.graph_generator.find_root_cause):
services are ordered by (1) owning the earliest in-degree-zero ERROR-level
node in the correlation graph, (2) earliest ERROR timestamp, (3) ERROR
volume.  Ties broken by symptom-line volume.

Metric: Accuracy@k (k=1..5) and Avg@5 = mean(Acc@1..Acc@5), the metric used
by RCAEval's published baselines, computed overall, per system, per fault.

Usage:
  python run_experiment.py                  # all unzipped RE* datasets
  python run_experiment.py --dataset RE2-SS
  SMOKE_TEST=1 python run_experiment.py     # 3 cases
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "app" / "backend"))
from app.models import LogEntry, LogChain
from app.graph_generator import GraphGenerator

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rcaeval"
OUT_DIR = Path(__file__).parent / "data"

CONFIG = {
    # Anomalous window after fault injection (RCAEval injects at t and
    # records the anomalous period afterwards).
    "window_seconds": 600,
    # Real logs are dominated by INFO noise; keep severity lines plus
    # INFO lines that carry symptom vocabulary (a slow-query INFO line is
    # evidence).  Cap per case to bound memory.
    "max_lines_per_case": 8000,
}

SEVERITY_RE = re.compile(r'\b(ERROR|CRITICAL|FATAL|SEVERE)\b|"s":\s*"E"', re.IGNORECASE)
WARN_RE = re.compile(r'\bWARN(ING)?\b|"s":\s*"W"', re.IGNORECASE)
SYMPTOM_RE = re.compile(
    r"slow|timeout|timed out|retry|refused|fail|error|exception|unavailable|"
    r"denied|reset|dropped|congestion|saturat|latency|5\d\d\b",
    re.IGNORECASE,
)


def infer_level(row: dict) -> str:
    explicit = (row.get("level") or "").strip().upper()
    if explicit in {"ERROR", "CRITICAL", "FATAL", "WARN", "WARNING", "INFO", "DEBUG"}:
        return "WARNING" if explicit == "WARN" else explicit
    text = row.get("message") or ""
    if (row.get("error") or "").strip() or SEVERITY_RE.search(text):
        return "ERROR"
    if WARN_RE.search(text):
        return "WARNING"
    return "INFO"


def load_case_entries(case_dir: Path) -> tuple:
    """Returns (post_entries, pre_symptom_rows, post_symptom_rows).

    post_entries feed the correlation graph (variant A).  The pre/post
    symptom rows — raw (component, lowercased message) pairs from equal
    windows before and after injection — feed the mention-differential
    ranking (variant B)."""
    inject_ts = int((case_dir / "inject_time.txt").read_text().strip())
    w = CONFIG["window_seconds"]
    entries, pre_rows, post_rows = [], [], []
    with open(case_dir / "logs.csv", newline="", errors="replace") as fh:
        for row in csv.DictReader(fh):
            try:
                ts_s = int(row["timestamp"]) / 1e9
            except (KeyError, ValueError, TypeError):
                continue
            if not (inject_ts - w <= ts_s <= inject_ts + w):
                continue
            message = (row.get("message") or "")[:500]
            component = (row.get("container_name") or "unknown").strip()
            level = infer_level(row)
            symptomatic = level != "INFO" or SYMPTOM_RE.search(message)
            if symptomatic:
                (pre_rows if ts_s < inject_ts else post_rows).append((component, message.lower()))
            if ts_s >= inject_ts and symptomatic and len(entries) < CONFIG["max_lines_per_case"]:
                entries.append(LogEntry(
                    timestamp=datetime.fromtimestamp(ts_s, tz=timezone.utc),
                    message=message,
                    level=level,
                    component=component,
                ))
    return entries, pre_rows, post_rows


def rank_by_mention_diff(pre_rows: list, post_rows: list) -> list:
    """Variant B: rank services by the INCREASE in symptomatic evidence
    pointing at them after injection, relative to the pre-fault baseline.
    Evidence for service S = symptom lines emitted by S itself, plus
    symptom lines from other services whose message names S (a caller
    logging 'payment timeout' blames payment, not itself)."""
    components = {c for c, _ in pre_rows} | {c for c, _ in post_rows}
    names = sorted(components, key=len, reverse=True)

    def evidence(rows):
        score = defaultdict(float)
        for src, msg in rows:
            score[src] += 1.0
            for svc in names:
                if svc != src and svc in msg:
                    score[svc] += 2.0  # being named in another's symptom line is stronger evidence
        return score

    pre, post = evidence(pre_rows), evidence(post_rows)
    diff = {svc: post.get(svc, 0.0) - pre.get(svc, 0.0) for svc in components}
    return [svc for svc, _ in sorted(diff.items(), key=lambda kv: -kv[1])]


def rank_services(entries: list) -> list:
    """Build the correlation graph and rank components as root-cause
    suspects using the graph structure (see module docstring)."""
    if not entries:
        return []
    dag = GraphGenerator(LogChain(log_chain=entries)).generate_dag(analyse_root=False)
    root_ids = set(dag.root_ids)
    nodes_by_id = {n.id: n for n in dag.nodes}

    stats = defaultdict(lambda: {"root_err_ts": None, "first_err_ts": None,
                                 "errors": 0, "lines": 0})
    for node in dag.nodes:
        e = node.log_entry
        s = stats[e.component]
        s["lines"] += 1
        if e.level.upper() in {"ERROR", "CRITICAL", "FATAL", "SEVERE"}:
            s["errors"] += 1
            ts = e.timestamp.timestamp()
            if s["first_err_ts"] is None or ts < s["first_err_ts"]:
                s["first_err_ts"] = ts
            if node.id in root_ids and (s["root_err_ts"] is None or ts < s["root_err_ts"]):
                s["root_err_ts"] = ts

    INF = float("inf")
    ranked = sorted(
        stats.items(),
        key=lambda kv: (
            kv[1]["root_err_ts"] if kv[1]["root_err_ts"] is not None else INF,
            kv[1]["first_err_ts"] if kv[1]["first_err_ts"] is not None else INF,
            -kv[1]["errors"],
            -kv[1]["lines"],
        ),
    )
    return [name for name, _ in ranked]


def service_hit(ranked_component: str, gt_service: str) -> bool:
    """A ranked container counts as the ground-truth service if it is that
    service or one of its sidecars/databases (e.g. carts-db -> carts)."""
    c = ranked_component.lower()
    g = gt_service.lower()
    return c == g or c.startswith(g + "-") or c.startswith(g + "_")


def evaluate_case(case_dir: Path, gt_service: str) -> dict:
    entries, pre_rows, post_rows = load_case_entries(case_dir)
    rec = {
        "case": str(case_dir.relative_to(DATA_DIR)),
        "gt_service": gt_service,
        "n_entries": len(entries),
    }
    for variant, ranking in (("graph", rank_services(entries)),
                             ("mention", rank_by_mention_diff(pre_rows, post_rows))):
        rec[f"top5_{variant}"] = ranking[:5]
        for k in range(1, 6):
            rec[f"{variant}_acc@{k}"] = any(service_hit(c, gt_service) for c in ranking[:k])
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="e.g. RE2-SS (default: all unzipped RE* dirs)")
    args = parser.parse_args()

    datasets = ([DATA_DIR / args.dataset] if args.dataset
                else sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("RE")))
    cases = []
    for ds in datasets:
        for inject in ds.rglob("inject_time.txt"):
            case_dir = inject.parent
            if not (case_dir / "logs.csv").exists():
                continue
            # dirname pattern: .../{service}_{fault}/{instance}
            fault_dir = case_dir.parent.name        # e.g. payment_disk
            gt_service = fault_dir.rsplit("_", 1)[0]
            fault = fault_dir.rsplit("_", 1)[1]
            cases.append((case_dir, gt_service, fault, ds.name))

    if os.environ.get("SMOKE_TEST"):
        cases = cases[:3]
        print(f"SMOKE TEST: {len(cases)} cases")
    print(f"Evaluating {len(cases)} cases from {[d.name for d in datasets]}")

    OUT_DIR.mkdir(exist_ok=True)
    checkpoint = OUT_DIR / "checkpoint.jsonl"
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["case"]] = rec
            except json.JSONDecodeError:
                continue

    results = []
    for i, (case_dir, gt, fault, system) in enumerate(cases, 1):
        rel = str(case_dir.relative_to(DATA_DIR))
        if rel in done:
            rec = done[rel]
        else:
            try:
                rec = evaluate_case(case_dir, gt)
            except Exception as e:
                print(f"[{i}/{len(cases)}] {rel}: FAILED {type(e).__name__}: {e}")
                continue
            rec["fault"] = fault
            rec["system"] = system
            with open(checkpoint, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        results.append(rec)
        print(f"[{i}/{len(cases)}] {rel}: gt={gt} graph@5={rec['graph_acc@5']} mention@5={rec['mention_acc@5']} (n={rec['n_entries']})")

    def agg(rows, variant):
        out = {f"acc@{k}": round(statistics.mean(r[f"{variant}_acc@{k}"] for r in rows), 4) for k in range(1, 6)}
        out["avg@5"] = round(statistics.mean(out[f"acc@{k}"] for k in range(1, 6)), 4)
        out["n"] = len(rows)
        return out

    report = {"config": {**CONFIG, "run_date": datetime.now().isoformat(),
                         "variants": {"graph": "graph-aware severity ranking (GraphGenerator roots)",
                                      "mention": "pre/post symptomatic-mention differential"}}}
    for variant in ("graph", "mention"):
        block = {"overall": agg(results, variant), "by_system": {}, "by_fault": {}}
        for key, field in (("by_system", "system"), ("by_fault", "fault")):
            groups = defaultdict(list)
            for r in results:
                groups[r.get(field, "?")].append(r)
            block[key] = {g: agg(rows, variant) for g, rows in sorted(groups.items())}
        report[variant] = block

    (OUT_DIR / "rcaeval_results.json").write_text(json.dumps(report, indent=2))
    for variant in ("graph", "mention"):
        print(f"\n{variant.upper()} overall: {report[variant]['overall']}")
        for fault, m in report[variant]["by_fault"].items():
            print(f"  {fault}: avg@5={m['avg@5']} acc@1={m['acc@1']} (n={m['n']})")
    print(f"Written: {OUT_DIR / 'rcaeval_results.json'}")


if __name__ == "__main__":
    main()
