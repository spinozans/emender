#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round5
cp "$GPU_LEASE_DIR/4" "$r/quick_lease.txt"
for label in reduction_fixed quick_fixed matched_r4; do
 [[ ! -e "$r/${label}_terminal.json" ]]
 module=c14_phase5r5_probe
 args=(--quick)
 if [[ "$label" == reduction_fixed ]]; then module=c14_phase5r5_reduction_probe; args=(--page-size 1024); fi
 if [[ "$label" == matched_r4 ]]; then args=(--only1024 --matched-r4); fi
 rc=0
 timeout 240 /home/erikg/emender/.venv/bin/python -m "experiments.expressivity_tasks.x43_fork_resolved.$module" "${args[@]}" --output "$r/$label.json" > "$r/${label}_console.txt" 2>&1 || rc=$?
 printf '{"exit_code":%d}\n' "$rc" > "$r/${label}_terminal.json"
 [[ "$rc" == 0 ]] || exit "$rc"
done
