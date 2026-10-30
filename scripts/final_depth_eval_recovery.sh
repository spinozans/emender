#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=1,5,6 OMP_NUM_THREADS=1 CUDA_LAUNCH_BLOCKING=1
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait --timeout 3600)"
eval "$lease_exports"
out=experiments/expressivity_tasks/x42_pile_scale/receipts/operational_recovery_part13
printf '%s\n' "$GPU_LEASE_HELD" > "$out/final_eval_launch/lease.txt"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.final_depth_eval --checkpoint experiments/expressivity_tasks/x40_emender_lm/receipts/part12_checkpoint_managed/training/runs/depth.ckpt --registration "$out/depth_eval_registration.json" --amendment "$out/depth_checkpoint_rebind_amendment.json" --output experiments/expressivity_tasks/x40_emender_lm/receipts/ignition_v1
