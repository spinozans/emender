#!/usr/bin/env bash
# Detached leased driver for the 2700 s chunk-8 BOARD arm (cards 4/5).
#   usage: run_board_chunk8_leased.sh
# The extension lane's own verdict (RESULT_CHUNK_SEQUENTIALITY.md next-steps
# item 2) named this arm: the matrix_ops board arm at CHUNK 8, everything else
# identical to the banked board arm (seed 3401, transferred LR, K5B whole-scan
# path, honest 2700 active-second clock, flagship stream, 150M-board
# calibration eval, mb8/64 windows), with the per-chunk-size bit-equality +
# 0-decision-flip parity receipt BEFORE timing and the op-rate diagnostics on
# the trained model.  Acquires ONE lease from cards 4/5 (0/1 are the K6
# lane's, 2/3 the CMA-ES lane's -- never touched here), runs the arm in its
# own process, and COMMITS THE ARM'S ARTIFACTS before releasing so the board
# arm survives an attended-session ceiling.  Run this DETACHED (setsid +
# redirected stdio, the long-session survival playbook): the arm and its
# lease owner survive the session that launched them.
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender

export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-4,5}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none}"

status=0
if $PY -m experiments.expressivity_tasks.x45_matrix_emender.board_chunk8 arm; then
  # git add on this arm's OWN paths only: the K6 and CMA-ES lanes commit their
  # own files from their own sessions.
  paths=("$X45/board_chunk8.py" "$X45/run_board_chunk8_leased.sh" \
         "$X45/RESULT_BOARD_CHUNK8.md" \
         "$X45/receipts/board_chunk8.json" \
         "$X45/receipts/board_chunk8_parity.json" \
         "$X45/board_runs/matrix_ops_c8" \
         "$X45/board_runs/driver.console.txt")
  add_paths=()
  for p in "${paths[@]}"; do [[ -e "$p" ]] && add_paths+=("$p"); done
  summary=$($PY -c '
import json, sys
r = json.load(open(sys.argv[1]))
print(f"chunk-8 board arm: {r["eval_nll_nats_per_byte"]:.4f} nats/byte, "
      f"{r["exposure_bytes"]/1e6:.1f}MB in {r["walls"]["train_seconds"]:.0f}s "
      f"({r["bytes_per_s"]/1e3:.1f}k tok/s)")' \
      "$X45/receipts/board_chunk8.json") || summary="receipt unreadable"
  if [ "${#add_paths[@]}" -gt 0 ]; then
    attempt=0
    until git add "${add_paths[@]}" 2>/dev/null \
        && git commit -m "X45 chunk-8 board arm: the sequentiality dial at 150M board scale ($summary)" \
           >/dev/null; do
      attempt=$((attempt + 1))
      if [ "$attempt" -ge 12 ]; then
        echo "commit failed after $attempt attempts; artifacts remain on disk"
        status=1
        break
      fi
      sleep 10   # index.lock contention with the concurrent K6/CMA-ES lanes
    done
  fi
  echo "committed the chunk-8 board arm: $summary"
else
  echo "the chunk-8 board arm FAILED; artifacts (if any) remain on disk"
  status=1
fi
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
exit "$status"
