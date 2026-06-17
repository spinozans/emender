#!/usr/bin/env bash
# Lease ONE idle GPU and re-score both references on the FUSED kernel.
# task: re-run-offline. Auto-releases the lease on shell exit.
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REPO_ROOT="$(pwd)"

export GDN2_PATH="${GDN2_PATH:-/home/erikg/GatedDeltaNet-2}"
export HELDOUT_EVAL_BS="${HELDOUT_EVAL_BS:-8}"

eval "$(scripts/gpu_lease.sh acquire 1)"
echo "[run_rerun_fused] leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python results/offline-eval-references/rerun_fused.py --batch-size "$HELDOUT_EVAL_BS"
echo "[run_rerun_fused] DONE"
