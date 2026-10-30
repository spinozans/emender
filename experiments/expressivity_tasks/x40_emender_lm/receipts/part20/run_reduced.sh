#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x40_emender_lm/receipts/part20
export GPU_LEASE_VISIBLE=2,3,4
bash scripts/gpu_lease.sh status > "$r/broker_before_reduced_clone.txt"
lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait) || exit 1
eval "$lease"
printf '%s\n' "$lease" > "$r/lease_exports_reduced_clone.txt"
cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$r/lease_reduced_clone.txt"
export PYTHONPATH=/home/erikg/emender CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1
sha256sum "$r/reduced_clone.py" "$r/run_reduced.sh" > "$r/sources_reduced_clone.txt"
timeout 180 /usr/local/cuda/bin/compute-sanitizer --tool initcheck --print-limit 100 --error-exitcode 86 /home/erikg/emender/.venv/bin/python "$r/reduced_clone.py" > "$r/reduced_clone.txt" 2>&1
rc=$?
printf '%s %s\n' reduced_clone "$rc" >> "$r/status.txt"
exit "$rc"
