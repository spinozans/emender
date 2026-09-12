#!/usr/bin/env bash
set -euo pipefail
if [[ ${E97_ONPOLICY_BOUNDED:-0} != 1 ]]; then
 export E97_ONPOLICY_BOUNDED=1
 exec timeout --kill-after=30 7200 bash "${BASH_SOURCE[0]}" "$@"
fi
umask 077
RUN=${1:?new run root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
cd "$CONTROL"
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_e97_native_onpolicy_canary.py tests/test_e97_native_execution.py tests/test_e97_open_swe_native_runtime_protocol.py tests/test_e97_openhands_native_execution.py
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/e97_native_onpolicy_canary.py freeze --recipe configs/pi/e97-native-onpolicy-canary-v1.json --output "$RUN"
PANEL_SHA=$(sha256sum "$RUN/panel.json" | cut -d' ' -f1)
mkdir "$RUN/preflight" "$RUN/results"
CUDA_VISIBLE_DEVICES='' timeout --kill-after=30 900 "$PYTHON_BIN" scripts/e97_native_onpolicy_canary.py preflight \
 --panel "$RUN/panel.json" --panel-sha "$PANEL_SHA" --output "$RUN/preflight" 2>&1 | tee "$RUN/preflight.log"
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)
eval "$LEASE"
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/triton"
timeout --kill-after=30 5400 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/e97_native_onpolicy_canary.py run \
 --panel "$RUN/panel.json" --panel-sha "$PANEL_SHA" --output "$RUN/results" 2>&1 | tee "$RUN/rollouts.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/e97_native_onpolicy_canary.py aggregate \
 --panel "$RUN/panel.json" --panel-sha "$PANEL_SHA" --output "$RUN/results" | tee "$RUN/aggregate.log"
