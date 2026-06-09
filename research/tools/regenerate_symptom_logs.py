#!/usr/bin/env python3
"""
Regenerate synthetic incident logs as SYMPTOM-ONLY telemetry.

Why: the original generation prompt (data/real_incidents/manage.py
generate-logs) handed the model the ground-truth root cause with no
instruction to hide it, so many logs.txt files state the answer
verbatim.  Any RCA pipeline evaluated on them gets inflated accuracy.

This script regenerates logs so that they:
  - show only observable symptoms (errors, timeouts, retries, metrics),
  - never state or paraphrase the underlying cause,
  - pass the leakage validator in check_log_leakage.py before being
    accepted (retrying up to --attempts times per incident).

Originals are preserved as logs_leaky_v1.txt the first time an incident
is regenerated.  Incidents whose ground_truth.json has
"logs_available": true (real production logs) are never touched.

Usage:
  python regenerate_symptom_logs.py --dry-run          # preview 3 incidents
  python regenerate_symptom_logs.py                    # full run
  python regenerate_symptom_logs.py --model qwen2.5-coder:32b
  SMOKE_TEST=1 python regenerate_symptom_logs.py       # 3 incidents only

After a full run, re-audit and then rerun experiments 07 and 08:
  python research/tools/check_log_leakage.py
  python research/experiments/07_multi_judge_validation/run_experiment.py
  python research/experiments/08_rag_real_world/run_experiment.py --judge all
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_log_leakage import (
    banned_phrase_hits,
    content_words,
    leakage_ratio,
    validate_candidate,
)

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

import ollama

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"
BACKUP_NAME = "logs_leaky_v1.txt"

CONFIG = {
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11435"),
    "model": "llama3.2:3b",
    "temperature": 0.7,
    "attempts": 6,
    "max_leakage_ratio": 0.35,
    "min_lines": 10,
    "postmortem_context_chars": 600,
}


def build_prompt(gt: dict, postmortem: str, attempt: int) -> str:
    root_cause = gt.get("root_cause", "")
    forbidden = sorted(content_words(root_cause))
    start_ts = gt.get("root_cause_timestamp") or "2024-03-01T10:00:00Z"

    # On retries, harden the instructions rather than just rerolling.
    extra = ""
    if attempt > 0:
        extra = (
            "\nYour previous attempt revealed the diagnosis. Be stricter: "
            "describe only WHAT the monitoring systems observed, never WHY."
        )

    return f"""You are generating realistic production log lines for an incident
that operators have NOT yet diagnosed.

CONFIDENTIAL DIAGNOSIS (for your context only — it must NOT appear in the
logs in any form): {root_cause}

Incident category: {gt.get('category', 'Unknown')}
Background (do not quote from it): {postmortem[:CONFIG['postmortem_context_chars']]}

STRICT RULES:
1. Write 15-25 log lines showing only OBSERVABLE SYMPTOMS: error codes,
   timeouts, retries, queue depths, latency numbers, HTTP status codes,
   health-check failures, alerts firing.
2. NEVER state, name, hint at, or paraphrase the underlying cause. No log
   line may explain WHY anything is failing.
3. Do NOT use any of these words: {', '.join(forbidden) if forbidden else '(none)'}.
4. Never write phrases like "root cause", "caused by", "due to",
   "detected misconfiguration", or anything that announces a conclusion.
5. Format every line as: ISO-8601 timestamp [LEVEL] service-name: message trace_id=<id>
   Use realistic service names for this technology stack. Mix INFO, WARN,
   ERROR, and CRITICAL levels as the incident escalates.
6. Correlation keys: lines that belong to the same request flow MUST share
   the same trace_id (e.g. trace_id=req-4f2a). Use 2-4 distinct trace_ids
   across the incident so failures form connected chains, and keep
   service-name consistent for lines from the same component.
7. Timestamps start near {start_ts} and advance over several minutes.
8. Output ONLY log lines — no markdown, no commentary.{extra}"""


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    # Drop any non-log preamble lines the model sneaks in.
    kept = [
        l for l in text.splitlines()
        if re.match(r"^\s*\d{4}-\d{2}-\d{2}", l)
    ]
    return "\n".join(kept).strip() or text


def regenerate_incident(client: ollama.Client, folder: Path, model: str) -> dict:
    gt = json.loads((folder / "ground_truth.json").read_text())
    postmortem = (folder / "postmortem.md").read_text() if (folder / "postmortem.md").exists() else ""
    root_cause = gt.get("root_cause", "")

    best = None  # (ratio, logs, reasons)
    for attempt in range(CONFIG["attempts"]):
        response = client.generate(
            model=model,
            prompt=build_prompt(gt, postmortem, attempt),
            options={"temperature": CONFIG["temperature"]},
        )
        candidate = strip_fences(response["response"])
        ok, reasons = validate_candidate(
            root_cause,
            candidate,
            max_ratio=CONFIG["max_leakage_ratio"],
            min_lines=CONFIG["min_lines"],
            require_correlation=True,
        )
        ratio = leakage_ratio(root_cause, candidate)
        if best is None or ratio < best[0]:
            best = (ratio, candidate, reasons)
        if ok:
            return {
                "status": "accepted",
                "attempts": attempt + 1,
                "leakage_ratio": round(ratio, 3),
                "logs": candidate,
            }

    return {
        "status": "needs_review",
        "attempts": CONFIG["attempts"],
        "leakage_ratio": round(best[0], 3),
        "reasons": best[2],
        "logs": best[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate symptom-only synthetic logs")
    parser.add_argument("--model", default=CONFIG["model"], help="Ollama model for generation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate for 3 incidents and print results without writing files")
    parser.add_argument("--only", help="Regenerate a single incident, e.g. incident_001")
    args = parser.parse_args()

    folders = sorted(INCIDENT_DIR.glob("incident_*"))
    if args.only:
        folders = [f for f in folders if f.name == args.only]
        if not folders:
            print(f"Incident {args.only} not found")
            sys.exit(1)
    if os.environ.get("SMOKE_TEST") or args.dry_run:
        folders = folders[:3]
        print(f"Limited run: {len(folders)} incidents")

    # Optional Modal proxy auth (research/tools/modal_ollama_server.py with
    # requires_proxy_auth=True): forward the token headers when present.
    headers = None
    if os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET"):
        headers = {
            "Modal-Key": os.environ["MODAL_KEY"],
            "Modal-Secret": os.environ["MODAL_SECRET"],
        }
    client = ollama.Client(host=CONFIG["ollama_host"], headers=headers)
    report = {"model": args.model, "started": datetime.now().isoformat(), "incidents": []}
    accepted = skipped = needs_review = 0

    for i, folder in enumerate(folders, 1):
        gt_file = folder / "ground_truth.json"
        if not gt_file.exists():
            continue
        gt = json.loads(gt_file.read_text())

        if gt.get("logs_available"):
            print(f"[{i}/{len(folders)}] {folder.name}: real logs — skipping")
            skipped += 1
            continue

        t0 = time.time()
        result = regenerate_incident(client, folder, args.model)
        elapsed = time.time() - t0
        print(f"[{i}/{len(folders)}] {folder.name}: {result['status']} "
              f"(leakage {result['leakage_ratio']}, {result['attempts']} attempts, {elapsed:.0f}s)")

        if not args.dry_run:
            backup = folder / BACKUP_NAME
            logs_file = folder / "logs.txt"
            if logs_file.exists() and not backup.exists():
                backup.write_text(logs_file.read_text())
            logs_file.write_text(result["logs"] + "\n")

        entry = {k: v for k, v in result.items() if k != "logs"}
        entry["id"] = folder.name
        report["incidents"].append(entry)
        if result["status"] == "accepted":
            accepted += 1
        else:
            needs_review += 1

    report["summary"] = {
        "accepted": accepted,
        "needs_review": needs_review,
        "skipped_real_logs": skipped,
    }
    report_file = Path(__file__).parent / "regeneration_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"\nDone: {accepted} accepted, {needs_review} need review, {skipped} skipped (real logs)")
    print(f"Report: {report_file}")
    if needs_review:
        print("Incidents marked needs_review kept their best (lowest-leakage) candidate;")
        print("inspect them manually or rerun with --only <incident_id> and a stronger --model.")


if __name__ == "__main__":
    main()
