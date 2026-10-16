#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
ROOT="$PWD/experiments/expressivity_tasks/results_invertible_stage2b"
runs=(headline_e97_seed0 headline_e97_seed1 headline_e97_seed2 headline_gdn2_seed0 headline_gdn2_seed1 headline_gdn2_seed2 headline_e97_identity_seed0)
while :; do
  complete=0
  for run in "${runs[@]}"; do
    receipt="$ROOT/receipts/$run/exit.txt"
    if test -f "$receipt"; then
      grep -qx 'exit_code=0' "$receipt" || exit 1
      complete=$((complete+1))
    fi
  done
  test "$complete" -eq 7 && break
  sleep 30
done
eval_run() (
  run="$1"
  receipt="$ROOT/receipts/matched_$run"
  mkdir -p "$receipt"
  eval "$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"
  printf '%s\n' "host=$(hostname)" "gpu=$CUDA_VISIBLE_DEVICES" "start=$(date -Is)" > "$receipt/lease.txt"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
  printf '/usr/bin/python3 -m experiments.expressivity_tasks.eval_invertible_matched --root %q --run %q\n' "$ROOT" "$run" > "$receipt/command.txt"
  set +e
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/python3 -m experiments.expressivity_tasks.eval_invertible_matched --root "$ROOT" --run "$run" > "$receipt/console.txt" 2>&1
  rc=$?
  set -e
  printf '%s\n' "exit_code=$rc" "end=$(date -Is)" > "$receipt/exit.txt"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$receipt/gpu_after.txt"
  return "$rc"
)
pids=()
for run in "${runs[@]}"; do eval_run "$run" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
