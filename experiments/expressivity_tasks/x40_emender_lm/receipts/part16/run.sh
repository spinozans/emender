#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x40_emender_lm/receipts/part16
export GPU_LEASE_VISIBLE=2,3,4
bash scripts/gpu_lease.sh status > "$r/broker_before.txt"
lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait) || exit 1
eval "$lease"
printf '%s\n' "$lease" > "$r/lease_exports.txt"
cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$r/lease.txt"
export CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/home/erikg/emender OMP_NUM_THREADS=1
for tool in memcheck racecheck initcheck; do
  timeout 600 /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 86 /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x40_emender_lm/receipts/part15/single.py > "$r/${tool}.txt" 2>&1
  printf '%s %s\n' "$tool" "$?" >> "$r/status.txt"
done
