#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=5 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 5 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round2/selected
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/lease_validation.txt"
failed=0
for gate in oracle battery shape gradient long_512 long_1024 long_2048; do
  [[ ! -e "$r/${gate}_terminal.json" && ! -e "$r/${gate}.json" ]]
  rc=0
  args=(-m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r2_selected_validate "$gate")
  if [[ "$gate" == shape || "$gate" == gradient ]]; then
    args+=(--output "$r/${gate}.json")
  elif [[ "$gate" == long_* ]]; then
    args=(-m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r2_selected_long_parity --tokens "${gate#long_}" --output "$r/${gate}.json")
  fi
  limit=()
  if [[ "$gate" == long_2048 ]]; then limit=(timeout --signal=TERM 180); fi
  "${limit[@]}" /home/erikg/emender/.venv/bin/python "${args[@]}" > "$r/${gate}_console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/${gate}_terminal.json"
  if [[ "$rc" != 0 ]]; then failed=1; fi
done
exit "$failed"
