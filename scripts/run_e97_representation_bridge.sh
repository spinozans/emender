#!/usr/bin/env bash
# Separate bounded phases, no retries or automatic reopening/promotion.
set -euo pipefail
umask 077
MODE=${1:?data, prepare, train, or evaluate required}
DATA=${2:?new data root required}
RUN=${3:?new training root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${FROZEN_SOURCE_INVENTORY:?immutable controller inventory required}"
: "${E97_RUN_EVIDENCE_DIR:?external phase evidence directory required}"
[[ -z ${GPU_LEASE_HELD:-} ]]
mkdir "$E97_RUN_EVIDENCE_DIR"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RECIPE=configs/pi/e97-representation-bridge-training-v1.json
cd "$CONTROL"
finish() {
 local original=$?; local rc=$original; trap - EXIT
 if declare -F gpu_lease_release >/dev/null; then gpu_lease_release || rc=93; fi
 if ! sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$E97_RUN_EVIDENCE_DIR/source-after.log"; then rc=91; fi
 if [[ -e "$RUN/source.sha256" ]]; then
  if ! (cd "$RUN/worktree" && sha256sum -c "$RUN/source.sha256") > "$E97_RUN_EVIDENCE_DIR/trainer-source-after.log"; then rc=92; fi
 fi
 printf '{"phase":"%s","original_exit":%d,"audited_exit":%d}\n' "$MODE" "$original" "$rc" > "$E97_RUN_EVIDENCE_DIR/terminal.json"
 exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$E97_RUN_EVIDENCE_DIR/source-before.log"
case "$MODE" in
 data)
  [[ ! -e "$DATA" && ! -e "$RUN" ]]
  CUDA_VISIBLE_DEVICES='' timeout --kill-after=30 2400 "$PYTHON_BIN" -m scripts.build_e97_representation_bridge --recipe "$RECIPE" --output "$DATA"
  ;;
 prepare)
  [[ ! -e "$RUN" && ! -e "$DATA/packs" && ! -e "$DATA/result-audit.json" ]]
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_representation_bridge_training.py tests/test_e97_grounded_curriculum.py tests/test_e97_grounded_expansion_gate.py tests/test_e97_grounding_correction.py tests/test_e97_native_training_segment.py tests/test_e97_native_execution.py
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.audit_e97_representation_bridge --recipe "$RECIPE" --data "$DATA"
  AUTH_SHA=$(sha256sum "$DATA/authority/manifest.json" | cut -d' ' -f1)
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/build_e97_sft_packs.py --authority-root "$DATA/authority" --output-root "$DATA/packs" --context-size 65536 --authority-manifest-sha256 "$AUTH_SHA" --boundary-aware --sampler-mode epoch-permutation
  PACK_SHA=$(sha256sum "$DATA/packs/manifest.json" | cut -d' ' -f1)
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/audit_e97_training_mix_schedule.py --authority "$DATA/authority" --packs "$DATA/packs" --authority-sha256 "$AUTH_SHA" --pack-sha256 "$PACK_SHA" --steps 32 --sampler-key 974223 --output "$DATA/expected-schedule.json"
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.train_e97_representation_bridge exposure --data "$DATA"
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.train_e97_representation_bridge prepare --recipe "$RECIPE" --data "$DATA" --phase "$RUN"
  echo REPRESENTATION_BRIDGE_READY_FOR_TRAINING
  ;;
 train)
  : "${E97_REVIEWED_EXPOSURE_SHA:?attended review of actual source exposures required}"
  [[ $(sha256sum "$DATA/exposure-audit.json" | cut -d' ' -f1) == "$E97_REVIEWED_EXPOSURE_SHA" ]]
  [[ ! -e "$RUN/console.log" && ! -e "$RUN/checkpoints" && ! -e "$RUN/summary.json" ]]
  (set -o noclobber; printf '{"max_updates":32,"reviewed_exposure_sha256":"%s"}\n' "$E97_REVIEWED_EXPOSURE_SHA" > "$RUN/training-attempt.json")
  chmod 400 "$RUN/training-attempt.json"
  sha256sum -c "$RUN/inventory.sha256" > "$E97_RUN_EVIDENCE_DIR/launch-inventory.log"
  timeout --kill-after=30 5400 bash "$RUN/run.sh"
  ;;
 evaluate)
  [[ ! -e "$RUN/evaluation" ]]
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.train_e97_representation_bridge evaluations --data "$DATA" --phase "$RUN"
  export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
  LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)
  eval "$LEASE"
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for KIND in execution learning; do
   OUT="$RUN/evaluation/$KIND"
   export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$OUT/triton"
   PANEL_SHA=$(sha256sum "$OUT/panel.json" | cut -d' ' -f1)
   timeout --kill-after=30 7200 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 scripts/numa_local_rank_exec.py "scripts/eval_e97_native_${KIND}.py" run --panel "$OUT/panel.json" --panel-sha "$PANEL_SHA" --output "$OUT" 2>&1 | tee "$OUT/console.log"
   CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "scripts/eval_e97_native_${KIND}.py" aggregate --panel "$OUT/panel.json" --panel-sha "$PANEL_SHA" --output "$OUT" | tee "$OUT/aggregate.log"
  done
  CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.train_e97_representation_bridge gate --phase "$RUN"
  ;;
 *) echo "unknown phase: $MODE" >&2; exit 2;;
esac
