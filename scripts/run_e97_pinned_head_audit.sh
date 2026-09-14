#!/usr/bin/env bash
set -euo pipefail
umask 077
RUN=${1:?new output root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CONTROL"
: "${FROZEN_SOURCE_INVENTORY:?required source inventory outside export}"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ ! -e "$RUN" ]]; mkdir -m 700 "$RUN"
audit_exit() {
 local rc=$?; trap - EXIT
 if ! sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$RUN/source-after.log"; then rc=91; fi
 if declare -F gpu_lease_release >/dev/null; then gpu_lease_release; fi
 exit "$rc"
}
trap audit_exit EXIT
sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$RUN/source-before.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_head_precision_probe.py tests/test_e97_pinned_head_audit.py tests/test_e97_outcome_rl_candidate.py tests/test_e97_recurrent_precision.py > "$RUN/cpu-tests.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.audit_e97_pinned_head freeze --output "$RUN"
RECIPE_SHA=$(sha256sum "$RUN/assay/recipe-private.json" | cut -d' ' -f1)
[[ $RECIPE_SHA == eb99ad750df1722d7cd58c36ab8844d4c9e9655d76ae9774eed88c069186cf96 ]]
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$LEASE"
declare -F gpu_lease_release >/dev/null; trap audit_exit EXIT
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/triton"
timeout --kill-after=30 3600 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/qualify_e97_native_rl_logprobs.py run \
 --recipe "$RUN/assay/recipe-private.json" --recipe-sha "$RECIPE_SHA" --output "$RUN/assay" 2>&1 | tee "$RUN/assay/console.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.audit_e97_pinned_head audit --output "$RUN"
