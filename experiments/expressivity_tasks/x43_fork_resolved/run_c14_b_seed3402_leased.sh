#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=2 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 2 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_seed3402
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/lease_initial.txt"
for arm in budgeted unbudgeted; do
  /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_seed3402.py "$arm" preflight > "$r/$arm/preflight.txt" 2>&1
  rc=0
  /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_seed3402.py "$arm" train "$@" > "$r/$arm/train_console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/$arm/train_terminal.json"
  [[ "$rc" == 0 ]] || exit "$rc"
done
for arm in budgeted unbudgeted; do
  rc=0
  /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_seed3402.py "$arm" eval > "$r/$arm/eval_console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/$arm/eval_terminal.json"
  [[ "$rc" == 0 ]] || exit "$rc"
done
