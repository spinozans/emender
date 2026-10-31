#!/usr/bin/env bash
# Managed single-card queue; no timeout, no other cards eligible.
set -euo pipefail
cd /home/erikg/emender
out=experiments/expressivity_tasks/x42_pile_scale/receipts/velocity_baseline_anchors
mkdir -p "$out"
exec >> "$out/console.txt" 2>&1
export GPU_LEASE_VISIBLE=0 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/gpu_lease.sh
printf 'waiting for card 0 at %s\n' "$(date -u +%FT%TZ)"
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait)"
eval "$lease_exports"
[[ "$GPU_LEASE_HELD" == 0 && "$CUDA_VISIBLE_DEVICES" == 0 ]]
cp .wg/gpu_leases/0 "$out/lease.txt"
printf 'acquired card 0 at %s\n' "$(date -u +%FT%TZ)"
/home/erikg/emender/.venv/bin/python scripts/registration_guard.py check "$out/sources.sha256"
sha256sum -c "$out/sources.sha256"
failed=0
for arm in transformer gdn2 pure; do
  status=0
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.velocity_anchor_resume "$arm" "$out" || status=$?
  if [[ "$status" == 0 ]]; then continue; fi
  if [[ "$status" != 10 ]]; then exit "$status"; fi
  if ! /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.velocity_baselines --arm "$arm" --output "$out/$arm" --length-sweep; then
    failed=1
  fi
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.velocity_anchor_report
 done
exit "$failed"
