#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GPU_LEASE_VISIBLE=0,1,5,6
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait --timeout 60)"
eval "$lease_exports"
[[ "${GPU_LEASE_HELD:-}" =~ ^[0156]$ ]] || exit 90
printf 'X40 physical card %s\n' "$GPU_LEASE_HELD"
if (( $# )); then
  /home/erikg/emender/.venv/bin/python "$@"
else
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.runner
fi
