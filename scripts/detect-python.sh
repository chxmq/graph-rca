#!/bin/bash
set -euo pipefail

# Prints the first supported Python executable and exits 0.
# Preferred range can be controlled with MIN_PY_MINOR/MAX_PY_MINOR.
MIN_PY_MINOR="${MIN_PY_MINOR:-11}"
MAX_PY_MINOR="${MAX_PY_MINOR:-13}"

for ver in python3.13 python3.12 python3.11 python3; do
  if command -v "$ver" >/dev/null 2>&1; then
    PYVER=$("$ver" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
    MAJOR="${PYVER%%.*}"
    MINOR="${PYVER##*.}"
    if [ "$MAJOR" = "3" ] && [ "$MINOR" -ge "$MIN_PY_MINOR" ] && [ "$MINOR" -le "$MAX_PY_MINOR" ]; then
      echo "$ver"
      exit 0
    fi
  fi
done

exit 1
