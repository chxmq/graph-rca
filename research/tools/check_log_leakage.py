#!/usr/bin/env python3
"""
Root-cause leakage checker for synthetic incident logs.

Measures how much of each incident's ground-truth root cause vocabulary
appears verbatim in its logs.txt.  Synthetic logs generated while the
model knew the root cause tend to restate it ("Misconfiguration detected
in resource management"), which inflates downstream RCA accuracy.

Usage:
  python check_log_leakage.py             # audit the whole dataset
  python check_log_leakage.py --json out.json

Also importable: regenerate_symptom_logs.py uses leakage_ratio() and
validate_candidate() to gate newly generated logs.
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"

# Generic words that appear in any incident description; matching these
# is not evidence of leakage.
STOPWORDS = {
    "with", "that", "this", "from", "were", "caused", "causing", "cause",
    "root", "which", "their", "when", "into", "during", "after", "because",
    "service", "services", "system", "systems", "server", "servers",
    "failure", "failures", "failed", "error", "errors", "issue", "issues",
    "incident", "outage", "production", "multiple", "resulted", "resulting",
    "leading", "across", "within", "between", "without",
}

# Phrases that mean the log line is diagnosing itself rather than
# reporting a symptom.  Real telemetry does not announce conclusions.
BANNED_PHRASES = (
    "root cause",
    "caused by",
    "due to misconfig",
    "misconfiguration detected",
    "misconfiguration confirmed",
    "configuration error confirmed",
    "human error",
    "the cause of",
)

# Leakage tiers used in the paper's stratified analysis.
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.40


def content_words(text: str) -> set:
    """Lower-cased words of 4+ letters, minus stopwords."""
    return set(re.findall(r"[a-z]{4,}", text.lower())) - STOPWORDS


def leakage_ratio(root_cause: str, logs: str) -> float:
    """Fraction of root-cause content words that appear in the logs."""
    rc_words = content_words(root_cause)
    if not rc_words:
        return 0.0
    return len(rc_words & content_words(logs)) / len(rc_words)


def banned_phrase_hits(logs: str) -> list:
    low = logs.lower()
    return [p for p in BANNED_PHRASES if p in low]


def tier(ratio: float) -> str:
    if ratio >= HIGH_THRESHOLD:
        return "high"
    if ratio >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def validate_candidate(
    root_cause: str,
    logs: str,
    max_ratio: float = 0.35,
    min_lines: int = 10,
    require_correlation: bool = False,
) -> tuple:
    """Gate a candidate synthetic log.  Returns (ok, reasons)."""
    reasons = []
    lines = [l for l in logs.splitlines() if l.strip()]
    if len(lines) < min_lines:
        reasons.append(f"only {len(lines)} non-empty lines (min {min_lines})")

    if require_correlation:
        # Without trace/request ids the graph builder degenerates to one
        # root per node and the evaluation never exercises edge creation.
        corr_lines = sum(1 for l in lines if re.search(r"\b(trace_id|request_id)=\S+", l))
        if lines and corr_lines / len(lines) < 0.5:
            reasons.append(f"only {corr_lines}/{len(lines)} lines carry trace_id/request_id")

    ratio = leakage_ratio(root_cause, logs)
    if ratio > max_ratio:
        reasons.append(f"leakage ratio {ratio:.2f} exceeds {max_ratio}")

    hits = banned_phrase_hits(logs)
    if hits:
        reasons.append(f"banned phrases present: {hits}")

    # Logs should look like logs: most lines need a timestamp-ish prefix
    # and a recognizable level token somewhere.
    ts_lines = sum(1 for l in lines if re.match(r"^\d{4}-\d{2}-\d{2}", l.strip()))
    if lines and ts_lines / len(lines) < 0.8:
        reasons.append(f"only {ts_lines}/{len(lines)} lines start with a timestamp")

    level_lines = sum(
        1 for l in lines
        if re.search(r"\b(INFO|WARN|WARNING|ERROR|CRITICAL|FATAL|DEBUG)\b", l)
    )
    if lines and level_lines / len(lines) < 0.8:
        reasons.append(f"only {level_lines}/{len(lines)} lines contain a log level")

    return (not reasons, reasons)


def audit() -> list:
    rows = []
    for folder in sorted(INCIDENT_DIR.glob("incident_*")):
        gt_file = folder / "ground_truth.json"
        logs_file = folder / "logs.txt"
        if not gt_file.exists() or not logs_file.exists():
            continue
        gt = json.loads(gt_file.read_text())
        logs = logs_file.read_text()
        ratio = leakage_ratio(gt.get("root_cause", ""), logs)
        rows.append({
            "id": folder.name,
            "logs_available": bool(gt.get("logs_available", False)),
            "leakage_ratio": round(ratio, 3),
            "tier": tier(ratio),
            "banned_phrases": banned_phrase_hits(logs),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Audit root-cause leakage in incident logs")
    parser.add_argument("--json", type=Path, help="Write full report to this JSON file")
    args = parser.parse_args()

    rows = audit()
    if not rows:
        print(f"No incidents found under {INCIDENT_DIR}")
        sys.exit(1)

    counts = {"high": 0, "medium": 0, "low": 0}
    for r in rows:
        counts[r["tier"]] += 1
    ratios = [r["leakage_ratio"] for r in rows]

    print(f"Incidents audited: {len(rows)}")
    print(f"Mean leakage ratio: {sum(ratios) / len(ratios):.2f}")
    print(f"  high   (>= {HIGH_THRESHOLD:.0%} of root-cause words in logs): {counts['high']}")
    print(f"  medium (>= {MEDIUM_THRESHOLD:.0%}):                          {counts['medium']}")
    print(f"  low    (<  {MEDIUM_THRESHOLD:.0%}):                          {counts['low']}")
    flagged = [r for r in rows if r["banned_phrases"]]
    print(f"Incidents with self-diagnosing phrases: {len(flagged)}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"Report written to {args.json}")

    # Non-zero exit when the dataset still contains high-leakage logs, so
    # this can run as a CI gate after regeneration.
    sys.exit(0 if counts["high"] == 0 else 2)


if __name__ == "__main__":
    main()
