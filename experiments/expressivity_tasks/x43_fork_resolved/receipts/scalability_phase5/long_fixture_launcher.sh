#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/lease_long.txt"
for t in 512 1024; do
  [[ ! -e "$r/long_${t}.json" && ! -e "$r/long_${t}_terminal.json" ]]
  rc=0
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5_long_parity --tokens "$t" --output "$r/long_${t}.json" > "$r/long_${t}_console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/long_${t}_terminal.json"
  if [[ "$rc" != 0 ]]; then exit "$rc"; fi
done
