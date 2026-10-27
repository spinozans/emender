#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export GPU_LEASE_VISIBLE=0,1,5,6
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --wait --timeout 60)"
[[ "$GPU_LEASE_HELD" =~ ^[0156]$ ]] || exit 90
printf 'X34 physical card %s\n' "$GPU_LEASE_HELD"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x34_lm_pilot.cli "$@"
