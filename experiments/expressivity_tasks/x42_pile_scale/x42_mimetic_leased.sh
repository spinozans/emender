#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GPU_LEASE_VISIBLE=2,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait)"
eval "$lease_exports"
[[ "$GPU_LEASE_HELD" =~ ^[24]$ ]] || exit 90
out=experiments/expressivity_tasks/x42_pile_scale/receipts/mimetic_geometry
mkdir -p "$out"
printf '%s\n' "$GPU_LEASE_HELD" > "$out/lease.txt"
nvidia-smi > "$out/gpu_before.txt"
for arm in donor mimetic unfrozen; do
  set +e
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.x42_mimetic_geometry "$arm" 2>&1 | tee "$out/${arm}_console.txt"
  status=${PIPESTATUS[0]}
  set -e
  printf '{"exit_code":%d}\n' "$status" > "$out/${arm}_terminal.json"
  [[ "$status" == 0 ]] || exit "$status"
done
