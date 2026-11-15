#!/usr/bin/env bash
# Detached leased driver for the chunk-size sequentiality sweep (cards 4/5).
#   usage: run_sweep_chunk_leased.sh [CHUNKS...]   (default: 128 64 32 8 4 2)
# Each chunk arg is either a bare chunk size (N) or N:BYTES to override that
# arm's exposure budget (e.g. 2:10000000 for the chunk-2 half-exposure row
# the wall rule allows -- the receipt labels it NOT comparable to the 20M
# rows).  Acquires ONE lease from the dev-pool cards 4/5 (0/1 are the
# matched-arms lane's, 2/3 the CMA-ES lane's -- never touched here), then runs
# each arm in its own process (fresh graph slots and dispatch ledger per arm)
# and COMMITS AFTER EACH ARM (receipts + run dir + the running gradient table)
# so the sweep survives an attended-session ceiling.  Run this DETACHED
# (setsid + redirected stdio, the long-session survival playbook): the arms and
# the lease owner survive the session that launched them.
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
CHUNKS=${*:-"128 64 32 8 4 2"}

export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-4,5}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none}"

commit_arm() {
  local chunk=$1
  # git add -A on the sweep's OWN paths only: the arms lane and the CMA-ES lane
  # commit their own files from their own sessions.
  local paths=("$X45/sweep_chunk_sequentiality.py" \
               "$X45/run_sweep_chunk_leased.sh" \
               "$X45/RESULT_CHUNK_SEQUENTIALITY.md" \
               "$X45/receipts/sweep_chunk_c$chunk.json" \
               "$X45/receipts/sweep_chunk_parity_c$chunk.json" \
               "$X45/receipts/sweep_chunk_gradient.json" \
               "$X45/sweep_runs/chunk_$chunk" \
               "$X45/sweep_runs/driver.console.txt")
  local summary
  summary=$($PY -c '
import json, sys
r = json.load(open(sys.argv[1]))
print(f"{r["eval_nll_nats_per_byte"]:.4f} nats/byte, "
      f"{r["exposure_bytes"]/1e6:.1f}MB exposure in "
      f"{r["walls"]["train_seconds"]:.0f}s "
      f"({r["bytes_per_s"]/1e3:.1f}k tok/s)")' \
      "$X45/receipts/sweep_chunk_c$chunk.json") || summary="receipt unreadable"
  local attempt=0
  until git add "${paths[@]}" 2>/dev/null \
      && git commit -m "X45 chunk-sequentiality sweep: chunk $chunk arm ($summary)" \
         >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 12 ]; then
      echo "commit for chunk $chunk failed after $attempt attempts; artifacts remain on disk"
      return 1
    fi
    sleep 10   # index.lock contention with the concurrent arms lane
  done
  echo "committed chunk $chunk: $summary"
}

status=0
for spec in $CHUNKS; do
  chunk=${spec%%:*}
  bytes=${spec##*:}
  if [ "$bytes" = "$spec" ]; then
    bytes_opt=""
  else
    bytes_opt="--bytes $bytes"
  fi
  echo "=== arm chunk $chunk (exposure override: ${bytes_opt:-none}) ==="
  if $PY -m experiments.expressivity_tasks.x45_matrix_emender.sweep_chunk_sequentiality \
      arm --chunk-tokens "$chunk" $bytes_opt; then
    commit_arm "$chunk" || status=1
  else
    echo "arm chunk $chunk FAILED; continuing with the remaining arms"
    status=1
  fi
done
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
exit "$status"
