#!/usr/bin/env bash
set -euo pipefail
umask 077
RUN=${1:?new output root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CONTROL"
: "${FROZEN_SOURCE_INVENTORY:?required inventory outside export}"
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
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_recurrent_operands.py tests/test_e97_first_divergence.py tests/test_e97_numerical_policy.py > "$RUN/cpu-tests.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.diagnose_e97_recurrent_operands freeze --output "$RUN"
PLAN_SHA=$(sha256sum "$RUN/plan.json" | cut -d' ' -f1)
[[ $PLAN_SHA == d29f851e2b3beabc2b767617a8ce278ddfa143a573f03d7022a431d3f56eb4f8 ]]
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$LEASE"
declare -F gpu_lease_release >/dev/null; trap audit_exit EXIT
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/capture-triton"
timeout --kill-after=30 900 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/diagnose_e97_recurrent_operands.py capture \
 --output "$RUN" --plan-sha "$PLAN_SHA" 2>&1 | tee "$RUN/capture.log"
CAPTURE_SHA=$(sha256sum "$RUN/capture-summary.json" | cut -d' ' -f1)
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/replay-triton"
timeout --kill-after=30 300 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/diagnose_e97_recurrent_operands.py replay \
 --output "$RUN" --plan-sha "$PLAN_SHA" --capture-sha "$CAPTURE_SHA" 2>&1 | tee "$RUN/replay.log"
