#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x40_emender_lm/receipts/part17
tool=${1:-racecheck}
mode=${2:-diagnostic}
case "$tool" in racecheck|initcheck) ;; *) exit 2 ;; esac
case "$mode" in diagnostic|baseline) ;; *) exit 2 ;; esac
export GPU_LEASE_VISIBLE=2,3,4
bash scripts/gpu_lease.sh status > "$r/broker_before_${tool}_${mode}.txt"
lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait) || exit 1
eval "$lease"
printf '%s\n' "$lease" > "$r/lease_exports_${tool}_${mode}.txt"
cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$r/lease_${tool}_${mode}.txt"
export CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/home/erikg/emender OMP_NUM_THREADS=1
if [ "$mode" = diagnostic ]; then
  target=("$r/diagnostic.py")
else
  target=(experiments/expressivity_tasks/x40_emender_lm/receipts/part15/isolated.py original)
fi
timeout 360 /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --print-limit 0 --error-exitcode 86 /home/erikg/emender/.venv/bin/python "${target[@]}" > "$r/${tool}_${mode}.txt" 2>&1
status=$?
printf '%s_%s %s\n' "$tool" "$mode" "$status" >> "$r/status.txt"
exit "$status"
