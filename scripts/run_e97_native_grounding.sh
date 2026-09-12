#!/usr/bin/env bash
set -euo pipefail
if [[ ${E97_GROUNDING_BOUNDED:-0} != 1 ]]; then
  export E97_GROUNDING_BOUNDED=1
  exec timeout --kill-after=30 5400 bash "${BASH_SOURCE[0]}" "$@"
fi
umask 077
OUT=${1:?new output root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
R=/mnt/nvme2n1/erikg/e97_systematic_posttraining
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CONTROL"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/eval_e97_native_grounding.py freeze \
 --base "$R/native-execution-diagnostic-v2" --training "$R/native-training-50m-agent-lr1e5-v1" --output "$OUT"
PANEL_SHA=$(sha256sum "$OUT/panel.json" | cut -d' ' -f1)
mkdir -m 700 "$OUT/results"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_e97_native_execution.py \
 tests/test_e97_open_swe_native_runtime_protocol.py tests/test_e97_openhands_native_execution.py | tee "$OUT/cpu-tests.log"
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$OUT/triton"
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)
eval "$LEASE"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$OUT/hardware.csv"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/eval_e97_native_grounding.py run --panel "$OUT/panel.json" \
 --panel-sha "$PANEL_SHA" --output "$OUT/results" 2>&1 | tee "$OUT/execution.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/eval_e97_native_grounding.py aggregate \
 --panel "$OUT/panel.json" --panel-sha "$PANEL_SHA" --output "$OUT/results" | tee "$OUT/aggregate.log"
