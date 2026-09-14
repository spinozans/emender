#!/usr/bin/env bash
set -euo pipefail
umask 077
RUN=${1:?new output root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CONTROL"
: "${FROZEN_SOURCE_INVENTORY:?inventory outside export required}"
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
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_fixed_linear.py tests/test_e97_linear_operands.py tests/test_e97_early_prompt.py tests/test_e97_first_divergence.py tests/test_e97_packed_forward_probe.py tests/test_e97_numerical_policy.py tests/test_e97_recurrent_precision.py tests/test_e97_fp32_state_layer_cuda.py tests/test_e97_facade.py tests/test_e97_sft_precision_integration.py tests/test_e97_fixed_recurrent_kernel_audit.py tests/test_e97_outcome_rl_candidate.py tests/test_e97_recurrent_operands.py > "$RUN/cpu-tests.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.qualify_e97_fixed_linear freeze --output "$RUN"
PLAN_SHA=$(sha256sum "$RUN/plan.json" | cut -d' ' -f1)
[[ $PLAN_SHA == 9e4502cd7b37df43a3a0160667b0c57097b2462b6238613118e18a2eb6418159 ]]
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$LEASE"
declare -F gpu_lease_release >/dev/null; trap audit_exit EXIT
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/triton"
timeout --kill-after=30 300 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/qualify_e97_fixed_linear.py run --output "$RUN" --plan-sha "$PLAN_SHA" 2>&1 | tee "$RUN/worker.log"
