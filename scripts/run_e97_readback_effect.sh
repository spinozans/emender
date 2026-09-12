#!/usr/bin/env bash
set -euo pipefail
umask 077
RUN=${1:?new output root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CONTROL"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_e97_activation_alignment.py tests/test_e97_head_precision_probe.py tests/test_e97_outcome_rl_candidate.py tests/test_e97_native_execution.py
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/diagnose_e97_readback_effect.py freeze --output "$RUN"
RECIPE_SHA=$(sha256sum "$RUN/recipe-private.json" | cut -d' ' -f1)
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$LEASE"
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/triton"
timeout --kill-after=30 3600 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/diagnose_e97_readback_effect.py run \
 --recipe "$RUN/recipe-private.json" --recipe-sha "$RECIPE_SHA" --output "$RUN" 2>&1 | tee "$RUN/console.log"
