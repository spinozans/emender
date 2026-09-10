#!/usr/bin/env bash
set -euo pipefail
if [[ ${E97_NATIVE_EVAL_BOUNDED:-0} != 1 ]]; then
  export E97_NATIVE_EVAL_BOUNDED=1
  exec timeout --kill-after=30 2400 bash "${BASH_SOURCE[0]}" "$@"
fi
umask 077
PHASE=${1:?completed segment root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PLAN="$CONTROL/configs/pi/e97-native-training-50m-agent-evaluation-v1.json"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PLAN_SHA=$(sha256sum "$PLAN" | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CONTROL/scripts/eval_e97_native_training_segment.py" prepare \
 --phase "$PHASE" --plan "$PLAN" --plan-sha256 "$PLAN_SHA"
OUT="$PHASE/evaluation"
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$OUT/triton"
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
EVAL_SOURCE=/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-learning-eval-v1/worktree
SOURCE_INVENTORY=/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-learning-eval-v1/source.sha256
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$PLAN" "$EVAL_SOURCE" "$SOURCE_INVENTORY" <<'PY'
import json,sys,hashlib
p=json.load(open(sys.argv[1]));assert p['evaluator_source']==sys.argv[2] and p['evaluator_source_inventory']==sys.argv[3]
assert hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest()==p['evaluator_source_inventory_sha256']
PY
cd "$EVAL_SOURCE"
sha256sum --check --quiet "$SOURCE_INVENTORY"
PANEL_SHA=$(sha256sum "$OUT/panel.json" | cut -d' ' -f1)
LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)
eval "$LEASE"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$OUT/hardware.csv"
PYTHONPATH="$EVAL_SOURCE" "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/eval_e97_native_learning.py run --panel "$OUT/panel.json" \
 --panel-sha "$PANEL_SHA" --output "$OUT" 2>&1 | tee "$OUT/evaluation.log"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$EVAL_SOURCE" "$PYTHON_BIN" scripts/eval_e97_native_learning.py aggregate \
 --panel "$OUT/panel.json" --panel-sha "$PANEL_SHA" --output "$OUT" 2>&1 | tee "$OUT/aggregate.log"
sha256sum --check --quiet "$SOURCE_INVENTORY"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$CONTROL" "$PYTHON_BIN" "$CONTROL/scripts/eval_e97_native_training_segment.py" gate \
 --phase "$PHASE" 2>&1 | tee "$OUT/gate.log"
