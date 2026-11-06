#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round5
cp "$GPU_LEASE_DIR/4" "$r/cuda_profile_lease.txt"
failed=0
for arm in r4 r5; do
 [[ ! -e "$r/cuda_${arm}.json" && ! -e "$r/cuda_${arm}_terminal.json" ]]
 rc=0
 timeout 240 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r5_cuda_profile "$arm" "$r/cuda_${arm}.json" > "$r/cuda_${arm}_console.txt" 2>&1 || rc=$?
 printf '{"exit_code":%d}\n' "$rc" > "$r/cuda_${arm}_terminal.json"
 if [[ "$rc" != 0 ]]; then failed=1; fi
done
exit "$failed"
