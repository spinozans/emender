#!/usr/bin/env bash
set -euo pipefail
if [[ ${E97_CORRECTION_BOUNDED:-0} != 1 ]]; then
 export E97_CORRECTION_BOUNDED=1
 exec timeout --kill-after=30 10800 bash "${BASH_SOURCE[0]}" "$@"
fi
umask 077
DATA=${1:?completed data root required}
RUN=${2:?new training run root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$CONTROL"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_e97_grounding_correction.py tests/test_e97_native_training_segment.py tests/test_e97_native_execution.py
AUTH_SHA=$(sha256sum "$DATA/authority/manifest.json" | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/build_e97_sft_packs.py --authority-root "$DATA/authority" \
 --output-root "$DATA/packs" --context-size 65536 --authority-manifest-sha256 "$AUTH_SHA" --boundary-aware --sampler-mode epoch-permutation
PACK_SHA=$(sha256sum "$DATA/packs/manifest.json" | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/audit_e97_training_mix_schedule.py --authority "$DATA/authority" \
 --packs "$DATA/packs" --authority-sha256 "$AUTH_SHA" --pack-sha256 "$PACK_SHA" --steps 32 --sampler-key 974223 --output "$DATA/expected-schedule.json"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/prepare_e97_grounding_correction.py prepare \
 --recipe configs/pi/e97-grounding-correction-v1.json --data "$DATA" --phase "$RUN"
timeout --kill-after=30 5400 bash "$RUN/run.sh"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/prepare_e97_grounding_correction.py evaluations --data "$DATA" --phase "$RUN"
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)
eval "$LEASE"
for KIND in execution learning; do
 OUT="$RUN/evaluation/$KIND"
 export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$OUT/triton"
 PANEL_SHA=$(sha256sum "$OUT/panel.json" | cut -d' ' -f1)
 timeout --kill-after=30 3600 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
  scripts/numa_local_rank_exec.py "scripts/eval_e97_native_${KIND}.py" run --panel "$OUT/panel.json" \
  --panel-sha "$PANEL_SHA" --output "$OUT" 2>&1 | tee "$OUT/console.log"
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "scripts/eval_e97_native_${KIND}.py" aggregate --panel "$OUT/panel.json" \
  --panel-sha "$PANEL_SHA" --output "$OUT" | tee "$OUT/aggregate.log"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/prepare_e97_grounding_correction.py gate --phase "$RUN"
