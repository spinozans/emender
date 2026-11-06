#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round4
cp "$GPU_LEASE_DIR/4" "$r/tune_lease.txt"
failed=0
for page in 256 512 1024; do
 for kind in tune reduction; do
  label=${kind}_${page}
  [[ ! -e "$r/${label}.json" && ! -e "$r/${label}_terminal.json" ]]
  module=c14_phase5r4_probe
  args=(--only1024 --matched-r2)
  if [[ "$kind" == reduction ]]; then module=c14_phase5r4_reduction_probe; args=(); fi
  rc=0
  timeout --signal=TERM 240 /home/erikg/emender/.venv/bin/python -m "experiments.expressivity_tasks.x43_fork_resolved.$module" "${args[@]}" --page-size "$page" --output "$r/${label}.json" > "$r/${label}_console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/${label}_terminal.json"
  if [[ "$rc" != 0 ]]; then failed=1; fi
 done
done
exit "$failed"
