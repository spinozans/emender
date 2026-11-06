#!/usr/bin/env bash
# Recovery only: refuse duplicate launcher/worker ownership before invoking.
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=2 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 2 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_seed3402
for arm in budgeted unbudgeted; do
  state=absent
  if [[ -f "$r/$arm/training.json" ]]; then
    state=$(/home/erikg/emender/.venv/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$r/$arm/training.json")
  fi
  if [[ "$state" != completed_matched_exposure ]]; then
    args=()
    if [[ -f "/home/erikg/c14_dense_bytes/emender_c14_b_seed3402_$arm.pt" ]]; then args=(--resume); fi
    /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_seed3402.py "$arm" preflight >> "$r/$arm/recovery_preflight.txt" 2>&1
    /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_seed3402.py "$arm" train "${args[@]}" >> "$r/$arm/recovery_train_console.txt" 2>&1
  fi
done
for arm in budgeted unbudgeted; do
  state=absent;args=()
  if [[ -f "$r/$arm/eval_resume.json" ]]; then
    state=$(/home/erikg/emender/.venv/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$r/$arm/eval_resume.json")
    args=(--resume)
  fi
  if [[ "$state" != complete ]]; then
    /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_seed3402.py "$arm" eval "${args[@]}" >> "$r/$arm/recovery_eval_console.txt" 2>&1
  fi
done
