#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round3
cp "$GPU_LEASE_DIR/4" "$r/lease_followups.txt"
failed=0
for gate in reduction matched_r2 compile2048; do
 [[ ! -e "$r/${gate}.json" && ! -e "$r/${gate}_terminal.json" ]]
 module=c14_phase5r3_${gate}_probe
 rc=0
 timeout --signal=TERM 180 /home/erikg/emender/.venv/bin/python -m "experiments.expressivity_tasks.x43_fork_resolved.$module" --output "$r/${gate}.json" > "$r/${gate}_console.txt" 2>&1 || rc=$?
 printf '{"exit_code":%d}\n' "$rc" > "$r/${gate}_terminal.json"
 if [[ "$rc" != 0 ]]; then failed=1; fi
done
exit "$failed"
