#!/usr/bin/env bash
# opt-norm battery launcher: leases GPUs via the broker (waits/round-robins),
# runs the full phase-2 battery resumably, auto-releases on exit.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
N="${1:-4}"
SLOTS="${2:-3}"
eval "$(scripts/gpu_lease.sh "$N")"
echo "opt-norm: leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec python experiments/expressivity_tasks/run_opt_norm.py --slots_per_gpu "$SLOTS"
