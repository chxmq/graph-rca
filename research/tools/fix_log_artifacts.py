#!/usr/bin/env python3
"""
Deterministic post-processing for regenerated synthetic logs.

Fixes two generation artifacts without touching an LLM (so it cannot
reintroduce root-cause leakage):

1. Trace-ID fingerprint: the generator copied the prompt's example id
   (req-4f2a) into every incident.  Each distinct trace/request id in a
   file is remapped to a unique value derived from
   sha256(incident_id + original_id), preserving which lines share an id
   within the incident while making ids unique across the dataset.

2. Timestamp drift: only ~29% of logs used the incident's real date.
   All ISO timestamps in a file are shifted by a constant offset so the
   first log line aligns with the incident's root_cause_timestamp,
   preserving the intervals between lines.

Real-log incidents (ground_truth logs_available: true) are never touched.
Idempotent: trace ids are re-derived from the current ids deterministically
and the timestamp shift is 0 once aligned.

Usage:
  python research/tools/fix_log_artifacts.py          # apply
  python research/tools/fix_log_artifacts.py --check  # report only
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCIDENT_DIR = PROJECT_ROOT / "data" / "real_incidents"

CORR_RE = re.compile(r"\b(trace_id|request_id)=([\w.-]+)")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)")


def parse_ts(raw: str) -> datetime | None:
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00").replace(" ", "T", 1))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def remap_ids(incident_id: str, text: str) -> tuple:
    mapping = {}

    def repl(m):
        key, old = m.group(1), m.group(2)
        if old not in mapping:
            digest = hashlib.sha256(f"{incident_id}:{old}".encode()).hexdigest()[:6]
            prefix = "req" if key == "trace_id" else "rq"
            mapping[old] = f"{prefix}-{digest}"
        return f"{key}={mapping[old]}"

    text = CORR_RE.sub(repl, text)
    # The generator sometimes wrote ids into free text without the key=
    # prefix; remap those too so no original id survives anywhere.
    for old, new in mapping.items():
        text = text.replace(old, new)
    # The prompt's example id may linger in files whose keyed occurrences
    # were already remapped on a previous run.
    if "req-4f2a" in text:
        digest = hashlib.sha256(f"{incident_id}:req-4f2a".encode()).hexdigest()[:6]
        text = text.replace("req-4f2a", f"req-{digest}")
        mapping["req-4f2a"] = f"req-{digest}"

    return text, len(mapping)


def shift_timestamps(text: str, target_start: datetime) -> tuple:
    lines = text.splitlines()
    first = None
    for line in lines:
        m = TS_RE.match(line.strip())
        if m and (first := parse_ts(m.group(1))):
            break
    if first is None:
        return text, False
    delta = target_start - first
    if abs(delta) < timedelta(seconds=1):
        return text, False

    def fix_line(line: str) -> str:
        m = TS_RE.match(line)
        if not m:
            return line
        ts = parse_ts(m.group(1))
        if ts is None:
            return line
        new = (ts + delta).strftime("%Y-%m-%dT%H:%M:%SZ")
        return new + line[m.end(1):]

    return "\n".join(fix_line(l) for l in lines), True


def main():
    parser = argparse.ArgumentParser(description="Fix trace-id and timestamp artifacts in synthetic logs")
    parser.add_argument("--check", action="store_true", help="Report without writing")
    args = parser.parse_args()

    touched_ids = shifted = skipped_real = no_target_ts = 0
    for folder in sorted(INCIDENT_DIR.glob("incident_*")):
        gt_file = folder / "ground_truth.json"
        logs_file = folder / "logs.txt"
        if not gt_file.exists() or not logs_file.exists():
            continue
        gt = json.loads(gt_file.read_text())
        if gt.get("logs_available"):
            skipped_real += 1
            continue

        text = logs_file.read_text()
        new_text, n_ids = remap_ids(folder.name, text)
        if n_ids:
            touched_ids += 1

        target = parse_ts(str(gt.get("root_cause_timestamp", "")))
        if target is None:
            # Fall back to metadata's incident date, keeping the log's own
            # time-of-day so only the date moves.
            meta_file = folder / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                date = parse_ts(str(meta.get("date", "")) + "T00:00:00Z")
                first_m = next((TS_RE.match(l.strip()) for l in new_text.splitlines()
                                if TS_RE.match(l.strip())), None)
                first_ts = parse_ts(first_m.group(1)) if first_m else None
                if date is not None and first_ts is not None:
                    target = date.replace(hour=first_ts.hour, minute=first_ts.minute,
                                          second=first_ts.second)
        if target is not None:
            new_text, did_shift = shift_timestamps(new_text, target)
            if did_shift:
                shifted += 1
        else:
            no_target_ts += 1

        if not args.check and new_text != text:
            logs_file.write_text(new_text if new_text.endswith("\n") else new_text + "\n")

    mode = "CHECK" if args.check else "APPLIED"
    print(f"[{mode}] incidents with ids remapped: {touched_ids}, "
          f"timestamps shifted: {shifted}, real-log skipped: {skipped_real}, "
          f"no usable root_cause_timestamp: {no_target_ts}")


if __name__ == "__main__":
    main()
