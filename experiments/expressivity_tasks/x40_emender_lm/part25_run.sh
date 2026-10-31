#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GPU_LEASE_VISIBLE=3,4
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait --timeout 60)"
eval "$lease_exports"
[[ "${GPU_LEASE_HELD:-}" =~ ^[34]$ ]] || exit 90
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part25_budget
