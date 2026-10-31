#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
card="${1:?physical card required}"; shift
[[ "$card" =~ ^[124]$ ]] || exit 90
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GPU_LEASE_VISIBLE="$card"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait)"
eval "$lease_exports"
[[ "$GPU_LEASE_HELD" == "$card" && "$CUDA_VISIBLE_DEVICES" == "$card" ]] || exit 90
receipt_dir="${X42_LAUNCH_RECEIPTS:?}"
mkdir -p "$receipt_dir"
git rev-parse HEAD > "$receipt_dir/base_commit.txt"
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt_dir/lease.txt"
nvidia-smi > "$receipt_dir/gpu_before.txt"
set +e
/home/erikg/emender/.venv/bin/python "$@" 2>&1 | tee "$receipt_dir/console.txt"
status=${PIPESTATUS[0]}
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt_dir/terminal.json"
exit "$status"
