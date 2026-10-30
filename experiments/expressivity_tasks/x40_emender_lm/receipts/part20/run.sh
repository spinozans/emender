#!/usr/bin/env bash
main() {
set -uo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x40_emender_lm/receipts/part20
name=$1
tool=$2
shift 2
export GPU_LEASE_VISIBLE=2,3,4
bash scripts/gpu_lease.sh status > "$r/broker_before_${name}.txt"
lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait) || exit 1
eval "$lease"
printf '%s\n' "$lease" > "$r/lease_exports_${name}.txt"
cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$r/lease_${name}.txt"
export CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/home/erikg/emender OMP_NUM_THREADS=1
export X40_PART20_RUN_NAME="$name"
sha256sum "$r/gate.py" "$r/lifecycle.py" "$r/candidate.patch" > "$r/sources_${name}.txt"
if [[ $tool = plain ]]; then
  timeout 600 /home/erikg/emender/.venv/bin/python "$r/gate.py" "$@" > "$r/${name}.txt" 2>&1
else
  timeout 600 /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --print-limit 0 --error-exitcode 86 /home/erikg/emender/.venv/bin/python "$r/gate.py" "$@" > "$r/${name}.txt" 2>&1
fi
status=$?
printf '%s %s\n' "$name" "$status" >> "$r/status.txt"
exit "$status"

}
main "$@"
