#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

eval "$(scripts/gpu_lease.sh 1)"
echo "[gpu] leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec python scripts/ref_emender_mlp_run.py "$@"
