#!/usr/bin/env bash
# Detached leased driver for the WHEELER-ORDER capacity probe (SPAN-RUN) --
# cards 4/5, the finished separator/growth lane's pool (0/1 the
# matched-arms/kernel lane, 2/3 the running SEP-KV lane -- never touched).
# The registered sweep grid (3 LRs x 3 seeds x 6 spans x 2 arms at
# 3000 steps x batch 1024, RESULT_WHEELER_CAPACITY.md §4 as amended) is split
# BY SEED into three concurrent processes on one leased card (the eager
# per-token oracle is latency-bound, so seed-concurrency is close to free),
# then merged into the canonical arm receipt + the mechanism pass.
#
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook):
#   usage: run_wheeler_sweep_leased.sh ARM      (ARM: pointwise | ordered)
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
ARM=${1:?"usage: run_wheeler_sweep_leased.sh ARM  (ARM: pointwise|ordered)"}
MOD=experiments.expressivity_tasks.x45_matrix_emender.wheeler_capacity_probe
LOGS=$X45/wheeler_runs
mkdir -p "$LOGS"

# CPU thread cap (operator housekeeping, 2026-11-16): the eager per-token
# oracle is GPU-dispatch-bound, so capping torch's host thread pools costs
# single-digit % walls while stopping ~30 spin-waiting cores per worker on
# the shared box.  Applies to every worker and merge launch from this driver.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WHEELER_CPU_THREADS=4

# pin the arms to separate cards of the 4/5 pool
if [ -z "${GPU_LEASE_VISIBLE:-}" ]; then
  if [ "$ARM" = "pointwise" ]; then
    export GPU_LEASE_VISIBLE="4,5"
  else
    export GPU_LEASE_VISIBLE="5,4"
  fi
fi
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none} (arm $ARM, pid $$)"

SEEDS="3401 3402 3403"
pids=""
fail=0
for seed in $SEEDS; do
  $PY -m "$MOD" --arm "$ARM" --seeds "$seed" --run-tag "_seed$seed" \
      --lrs 3e-3,1e-3,3e-4 --batch 1024 --steps 3000 --device cuda --resume \
      > "$LOGS/${ARM}_seed${seed}.console.txt" 2>&1 &
  pids="$pids $!"
  echo "launched seed $seed (pid $!)"
done
for pid in $pids; do
  wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "one or more seed sweeps FAILED for arm $ARM; artifacts remain on disk"
  GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
  exit 1
fi
echo "seed sweeps complete; merging"

$PY -m "$MOD" --arm "$ARM" --merge-seeds --seeds 3401,3402,3403 \
    --batch 1024 --steps 3000 --device cuda \
    > "$LOGS/${ARM}_merge.console.txt" 2>&1 || fail=1
if [ "$fail" -ne 0 ]; then
  echo "the merge/mechanism step FAILED for arm $ARM; artifacts remain on disk"
  GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
  exit 1
fi

attempt=0
until git add "$X45/wheeler_capacity_probe.py" \
      "$X45/run_wheeler_sweep_leased.sh" \
      "$X45/receipts"/wheeler_${ARM}_s2*.json \
      "$X45/receipts"/wheeler_${ARM}_s4*.json \
      "$X45/receipts"/wheeler_${ARM}_s8*.json \
      "$X45/receipts"/wheeler_${ARM}_s16*.json \
      "$X45/receipts"/wheeler_${ARM}_s32*.json \
      "$X45/receipts"/wheeler_${ARM}_s64*.json \
      "$X45/receipts"/wheeler_${ARM}_sweep*.json \
      "$X45/receipts"/wheeler_mechanism_${ARM}.json \
      "$X45/wheeler_runs" 2>/dev/null \
    && git commit -m "X45 WHEELER-ORDER sweep: the $ARM arm (SPAN-RUN, rank 8, 3 LRs x 3 seeds x 6 spans, 3000x1024)" \
       >/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 12 ]; then
    echo "commit failed after $attempt attempts; artifacts remain on disk"
    GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 10   # index.lock contention with the concurrent lanes
done
echo "committed the $ARM sweep"
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
