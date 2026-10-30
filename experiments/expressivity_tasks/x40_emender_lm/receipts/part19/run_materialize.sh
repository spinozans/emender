#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x40_emender_lm/receipts/part19
export GPU_LEASE_VISIBLE=2,3,4
bash scripts/gpu_lease.sh status > "$r/broker_before_materialize_$1.txt"
lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$lease"
printf '%s\n' "$lease" > "$r/lease_exports_materialize_$1.txt"
cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$r/lease_materialize_$1.txt"
export PYTHONPATH=/home/erikg/emender CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1
set +e
timeout 180 /usr/local/cuda/bin/compute-sanitizer --tool initcheck --print-limit 0 --error-exitcode 86 /home/erikg/emender/.venv/bin/python "$r/materialize_probe.py" "${@:2}" > "$r/materialize_$1.txt" 2>&1
rc=$?
printf '%s %s\n' "$1" "$rc" >> "$r/status.txt"
exit "$rc"
