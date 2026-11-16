#!/usr/bin/env bash
# Detached leased driver for the WHEELER-ORDER mimetic-init extension (the
# operator-directed variant, pre-registered in RESULT_WHEELER_CAPACITY.md §4b;
# commits BEFORE these cells run).  GATED: it does not preempt the committed
# S=8..64 capacity cells -- it waits until the standard-arm sweep workers have
# exited, then runs on the freed card pool:
#   phase A: 3 seeds x 3 LRs at S=2 (the M-1/M-2 grid)
#   best-LR selection from phase A's own receipts
#   phase B: 3 seeds x 1 LR at S=32 (the M-3 spot-check)
#   merge + the mechanism pass + commit
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
MOD=experiments.expressivity_tasks.x45_matrix_emender.wheeler_capacity_probe
LOGS=$X45/wheeler_runs
mkdir -p "$LOGS"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WHEELER_CPU_THREADS=4

# ---- gate: wait for the committed sweeps (both arms' workers + drivers)
echo "gate: waiting for the committed sweep workers to finish"
while pgrep -f "wheeler_capacity_probe --arm pointwise --seeds" >/dev/null \
   || pgrep -f "wheeler_capacity_probe --arm ordered --seeds" >/dev/null; do
  sleep 120
done
echo "gate: committed sweeps finished at $(date)"

if [ -z "${GPU_LEASE_VISIBLE:-}" ]; then
  export GPU_LEASE_VISIBLE="4,5"
fi
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none} (ordered_mimetic, pid $$)"

run_phase () {   # run_phase SEEDS_CSV SPANS_CSV LRS_CSV TAG
  local seeds="$1" spans="$2" lrs="$3" tag="$4"
  local pids="" fail=0 pid
  for seed in 3401 3402 3403; do
    $PY -m "$MOD" --arm ordered_mimetic --seeds "$seed" --spans "$spans" \
        --lrs "$lrs" --run-tag "_seed${seed}${tag}" \
        --batch 1024 --steps 3000 --device cuda \
        > "$LOGS/ordered_mimetic_seed${seed}${tag}.console.txt" 2>&1 &
    pids="$pids $!"
    echo "launched ordered_mimetic seed $seed tag $tag (pid $!)"
  done
  for pid in $pids; do
    wait "$pid" || fail=1
  done
  return "$fail"
}

fail=0
run_phase "3401,3402,3403" 2 3e-3,1e-3,3e-4 "" || fail=1
if [ "$fail" -ne 0 ]; then
  echo "phase A FAILED; artifacts remain on disk"
  GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
  exit 1
fi
echo "phase A (S=2 grid) complete"

BEST_LR=$($PY - <<'PYEOF'
import json
from pathlib import Path
R = Path("experiments/expressivity_tasks/x45_matrix_emender/receipts")
best = None
for seed in (3401, 3402, 3403):
    data = json.loads((R / f"wheeler_ordered_mimetic_sweep_seed{seed}.json").read_text())
    for run in data["runs"]:
        if run["span"] == 2 and (best is None or run["metrics"]["value_acc"] > best[1]):
            best = (run["lr"], run["metrics"]["value_acc"])
print(f"{best[0]:.0e}")
PYEOF
) || fail=1
if [ "$fail" -ne 0 ] || [ -z "${BEST_LR:-}" ]; then
  echo "best-LR extraction FAILED; artifacts remain on disk"
  GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
  exit 1
fi
echo "phase B spot-check at the mimetic arm's own best S=2 LR: $BEST_LR"

run_phase "3401,3402,3403" 32 "$BEST_LR" "_phaseB" || fail=1
if [ "$fail" -ne 0 ]; then
  echo "phase B FAILED; artifacts remain on disk"
  GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
  exit 1
fi
echo "phase B (S=32 spot-check) complete; merging"

$PY -m "$MOD" --arm ordered_mimetic --merge-seeds --seeds 3401,3402,3403 \
    --batch 1024 --steps 3000 --device cuda \
    > "$LOGS/ordered_mimetic_merge.console.txt" 2>&1 || fail=1
if [ "$fail" -ne 0 ]; then
  echo "the merge/mechanism step FAILED; artifacts remain on disk"
  GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
  exit 1
fi

attempt=0
until git add "$X45/wheeler_capacity_probe.py" \
      "$X45/run_wheeler_mimetic_leased.sh" \
      "$X45/receipts"/wheeler_ordered_mimetic_s2*.json \
      "$X45/receipts"/wheeler_ordered_mimetic_s32*.json \
      "$X45/receipts"/wheeler_ordered_mimetic_sweep*.json \
      "$X45/receipts"/wheeler_mechanism_ordered_mimetic.json \
      "$X45/wheeler_runs" 2>/dev/null \
    && git commit -m "X45 WHEELER-ORDER mimetic-init extension: the ordered_mimetic arm (safe birth at the fixed-family geometry, free drift) -- S=2 grid + S=32 spot-check" \
       >/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 12 ]; then
    echo "commit failed after $attempt attempts; artifacts remain on disk"
    GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 10   # index.lock contention with the concurrent lanes
done
echo "committed the mimetic extension"
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release >/dev/null 2>&1 || true
