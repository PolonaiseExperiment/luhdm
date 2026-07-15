#!/usr/bin/env bash
# Run the full mediator-range limit scan set for one observed-event list.
#
#   scripts/run_all_scans.sh [DATA_FILE] [CACHE_DIR] [EXTRA_ARGS]
#
# DATA_FILE   event list, one impulse per line in eV (default notebooks/data.txt).
# CACHE_DIR   where scan7_<tag>.npz are written (default notebooks/computation_cache).
# EXTRA_ARGS  passed verbatim to every scan_grid.py call, e.g.
#             "--mode 1 --df 3 --q-min 1000 --m-min 1e5".
#
# Reusable across modes: point DATA_FILE at data_mode{1,2,3}.txt, give each mode
# its own CACHE_DIR, and pass the matching --mode in EXTRA_ARGS.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${1:-notebooks/data.txt}"
CACHE="${2:-notebooks/computation_cache}"
EXTRA="${3:-}"
PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$CACHE"

echo "events: $DATA   ->   cache: $CACHE   extra: [$EXTRA]   ($(date '+%F %T'))"
run() {
    local tag="$1"; shift
    echo "=== $(date '+%F %T') START $tag ==="
    "$PY" scripts/scan_grid.py "$@" $EXTRA --data "$DATA" \
          --out "$CACHE/scan7_$tag.npz"
    echo "=== $(date '+%F %T') DONE  $tag ==="
}

run 2m       --lamb 2.0
run 2cm      --lamb 2e-2
run 2mm      --lamb 2e-3
run 200um    --lamb 2e-4
run 20um     --lamb 2e-5
run 10um     --lamb 1e-5
run 2um      --lamb 2e-6
run massless --massless --lamb 2.0

echo "ALL DONE $(date '+%F %T')"
