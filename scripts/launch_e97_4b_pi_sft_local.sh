#!/bin/bash
# One-node/eight-GPU masked Pi SFT with pinned-CPU Schedule-Free state.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
MODE=${MODE:-qualification}
WORLD_SIZE=8
PARENT=${PARENT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/checkpoint_step_024448_loss_2.3981.pt}
PARENT_SHA256=${PARENT_SHA256:-3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568}
SOURCE_ARGS=${SOURCE_ARGS:-$(dirname "$PARENT")/args.json}
AUTHORITY_ROOT=${AUTHORITY_ROOT:-/mnt/nvme1n1/erikg/sft/e97-4b-pi-instruction-mix-v2}
PACK_ROOT=${PACK_ROOT:-$AUTHORITY_ROOT/packs-4096-v1}
AUTHORITY_SHA256=${AUTHORITY_SHA256:-$(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}')}
PACK_SHA256=${PACK_SHA256:-$(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}')}
LR=${LR:-0.00001}
WARMUP_STEPS=${WARMUP_STEPS:-8}
CONTEXT_SIZE=${CONTEXT_SIZE:-4096}
BOUNDARY_AWARE_PACKS=${BOUNDARY_AWARE_PACKS:-0}
GRADIENT_CHECKPOINT_GROUP_SIZE=${GRADIENT_CHECKPOINT_GROUP_SIZE:-1}
EMPTY_CACHE_MIN_RECORD_TOKENS=${EMPTY_CACHE_MIN_RECORD_TOKENS:-0}
MLP_CHECKPOINT_CHUNK_SIZE=${MLP_CHECKPOINT_CHUNK_SIZE:-0}
SAMPLER_KEY=${SAMPLER_KEY:-974003}
SAMPLER_MODE=${SAMPLER_MODE:-hash-replacement}
DILOCO_K=${DILOCO_K:-8}
DILOCO_MERGE=${DILOCO_MERGE:-1}
SAVE_EVERY=${SAVE_EVERY:-}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-3}
RESUME=${RESUME:-}
NEW_STAGE_FROM=${NEW_STAGE_FROM:-0}
NEW_STAGE_WEIGHT_MODE=${NEW_STAGE_WEIGHT_MODE:-saved}
SOURCE_IDENTITY=${SOURCE_IDENTITY:-$(git rev-parse HEAD)}
SOURCE_ARCHIVE=${SOURCE_ARCHIVE:-}
SOURCE_ARCHIVE_SHA256=${SOURCE_ARCHIVE_SHA256:-}
case "$MODE" in
  qualification)
    [[ -z "$RESUME" ]] || { echo "qualification must start from the parent" >&2; exit 64; }
    [[ "$NEW_STAGE_FROM" == 0 || "$NEW_STAGE_FROM" == 1 ]] || { echo "NEW_STAGE_FROM must be 0 or 1" >&2; exit 64; }
    STEPS=${STEPS:-8}; SAVE_EVERY=${SAVE_EVERY:-8}
    ;;
  canary)
    [[ ${CONFIRM_CANARY:-0} == 1 ]] || { echo "canary requires CONFIRM_CANARY=1" >&2; exit 64; }
    [[ -r "$RESUME" ]] || { echo "canary requires RESUME naming the qualification checkpoint" >&2; exit 66; }
    STEPS=${STEPS:-64}; SAVE_EVERY=${SAVE_EVERY:-32}
    ;;
  stage)
    [[ ${CONFIRM_STAGE:-0} == 1 ]] || { echo "stage requires CONFIRM_STAGE=1" >&2; exit 64; }
    [[ -r "$RESUME" ]] || { echo "stage requires RESUME naming the qualification checkpoint" >&2; exit 66; }
    STEPS=${STEPS:?stage requires explicit STEPS}; SAVE_EVERY=${SAVE_EVERY:-64}
    ;;
  parity-control)
    [[ ${CONFIRM_PARITY_CONTROL:-0} == 1 ]] || {
      echo "parity-control requires CONFIRM_PARITY_CONTROL=1" >&2; exit 64;
    }
    [[ -z "$RESUME" && "$NEW_STAGE_FROM" == 1 ]] || {
      echo "parity-control must restart a fresh stage from the parent" >&2; exit 64;
    }
    STEPS=${STEPS:-16}; SAVE_EVERY=${SAVE_EVERY:-8}
    ;;
  *) echo "MODE must be qualification, canary, stage, or parity-control" >&2; exit 64;;
esac
[[ "$NEW_STAGE_WEIGHT_MODE" == saved || "$NEW_STAGE_WEIGHT_MODE" == train ]] || {
  echo "NEW_STAGE_WEIGHT_MODE must be saved or train" >&2; exit 64;
}
if [[ "$NEW_STAGE_FROM" == 0 && -z "$RESUME" && "$NEW_STAGE_WEIGHT_MODE" != saved ]]; then
  echo "NEW_STAGE_WEIGHT_MODE=train requires NEW_STAGE_FROM=1 or RESUME" >&2; exit 64
fi
if [[ -n "$SOURCE_ARCHIVE" || -n "$SOURCE_ARCHIVE_SHA256" ]]; then
  [[ -r "$SOURCE_ARCHIVE" && ${#SOURCE_ARCHIVE_SHA256} == 64 ]] || {
    echo "source archive and SHA-256 must be supplied together" >&2; exit 66;
  }
  [[ $(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}') == "$SOURCE_ARCHIVE_SHA256" ]] || {
    echo "source archive SHA-256 mismatch" >&2; exit 66;
  }
  [[ "$SOURCE_IDENTITY" == "source-tree-sha256:$SOURCE_ARCHIVE_SHA256" ]] || {
    echo "SOURCE_IDENTITY must bind the verified source archive" >&2; exit 66;
  }
fi
[[ "$BOUNDARY_AWARE_PACKS" == 0 || "$BOUNDARY_AWARE_PACKS" == 1 ]] || {
  echo "BOUNDARY_AWARE_PACKS must be 0 or 1" >&2; exit 64;
}
[[ "$SAMPLER_MODE" == hash-replacement || "$SAMPLER_MODE" == epoch-permutation ]] || {
  echo "SAMPLER_MODE must be hash-replacement or epoch-permutation" >&2; exit 64;
}
[[ "$DILOCO_MERGE" == 0 || "$DILOCO_MERGE" == 1 ]] || {
  echo "DILOCO_MERGE must be 0 or 1" >&2; exit 64;
}
(( STEPS > 0 && DILOCO_K > 0 && SAVE_EVERY > 0 && EMPTY_CACHE_MIN_RECORD_TOKENS >= 0 && MLP_CHECKPOINT_CHUNK_SIZE >= 0 )) || exit 64
(( STEPS % DILOCO_K == 0 && SAVE_EVERY % DILOCO_K == 0 )) || {
  echo "steps and checkpoint cadence must be K-aligned" >&2; exit 64;
}
[[ -r "$PARENT" && -r "$SOURCE_ARGS" ]] || { echo "parent or args.json missing" >&2; exit 66; }
[[ $(sha256sum "$PARENT" | awk '{print $1}') == "$PARENT_SHA256" ]] || {
  echo "parent SHA-256 mismatch" >&2; exit 66;
}
[[ $(sha256sum "$AUTHORITY_ROOT/manifest.json" | awk '{print $1}') == "$AUTHORITY_SHA256" ]] || exit 66
[[ $(sha256sum "$PACK_ROOT/manifest.json" | awk '{print $1}') == "$PACK_SHA256" ]] || exit 66
SOURCE_COMMIT=$SOURCE_IDENTITY
RUN_ID=${RUN_ID:-e97-4b-pi-sft-local-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/runs/$RUN_ID}
[[ "$RUN_ROOT" == /* && "$RUN_ID" != */* && ! -e "$RUN_ROOT" ]] || {
  echo "invalid or existing run root: $RUN_ROOT" >&2; exit 65;
}
mkdir -p "$RUN_ROOT"/{checkpoints,identity,logs,terminal}
cat > "$RUN_ROOT/identity/launch.json" <<EOF
{"schema":"emender-e97-4b-pi-sft-local-launch-v1","mode":"$MODE","source_commit":"$SOURCE_COMMIT","source_archive":"$SOURCE_ARCHIVE","source_archive_sha256":"$SOURCE_ARCHIVE_SHA256","parent_sha256":"$PARENT_SHA256","authority_sha256":"$AUTHORITY_SHA256","pack_sha256":"$PACK_SHA256","world_size":8,"context_size":$CONTEXT_SIZE,"boundary_aware_packs":$BOUNDARY_AWARE_PACKS,"sampler_mode":"$SAMPLER_MODE","gradient_checkpoint_group_size":$GRADIENT_CHECKPOINT_GROUP_SIZE,"empty_cache_min_record_tokens":$EMPTY_CACHE_MIN_RECORD_TOKENS,"mlp_checkpoint_chunk_size":$MLP_CHECKPOINT_CHUNK_SIZE,"steps":$STEPS,"diloco_k":$DILOCO_K,"diloco_merge_enabled":$DILOCO_MERGE,"keep_checkpoints":$KEEP_CHECKPOINTS,"new_stage_from":$NEW_STAGE_FROM,"new_stage_weight_mode":"$NEW_STAGE_WEIGHT_MODE","optimizer_state_storage":"pinned-cpu"}
EOF
RESUME_ARGS=()
BOUNDARY_ARGS=()
MERGE_ARGS=()
if [[ "$BOUNDARY_AWARE_PACKS" == 1 ]]; then
  BOUNDARY_ARGS=(--boundary-aware-packs)
fi
if [[ "$DILOCO_MERGE" == 0 ]]; then
  MERGE_ARGS=(--disable-diloco-merge)
fi
if [[ -n "$RESUME" ]]; then
  [[ "$NEW_STAGE_FROM" == 0 ]] || { echo "resume and new-stage-from are mutually exclusive" >&2; exit 64; }
  RESUME_ARGS=(--resume "$RESUME")
elif [[ "$NEW_STAGE_FROM" == 1 ]]; then
  RESUME_ARGS=(--new-stage-from "$PARENT" --new-stage-weight-mode "$NEW_STAGE_WEIGHT_MODE")
fi
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS+=(--new-stage-weight-mode "$NEW_STAGE_WEIGHT_MODE")
fi
COMMAND=(
  torchrun --standalone --nproc_per_node="$WORLD_SIZE"
  scripts/numa_local_rank_exec.py -- scripts/train_e97_4b_pi_sft.py
  "${RESUME_ARGS[@]}" "${BOUNDARY_ARGS[@]}" "${MERGE_ARGS[@]}"
  --parent-checkpoint "$PARENT" --parent-sha256 "$PARENT_SHA256"
  --source-args-json "$SOURCE_ARGS" --source-commit "$SOURCE_COMMIT"
  --authority-root "$AUTHORITY_ROOT" --authority-sha256 "$AUTHORITY_SHA256"
  --pack-root "$PACK_ROOT" --pack-sha256 "$PACK_SHA256"
  --output-root "$RUN_ROOT/checkpoints" --log-jsonl "$RUN_ROOT/logs/training.jsonl"
  --steps "$STEPS" --save-every "$SAVE_EVERY" --keep-checkpoints "$KEEP_CHECKPOINTS" --diloco-k "$DILOCO_K"
  --context-size "$CONTEXT_SIZE" --gradient-checkpoint-group-size "$GRADIENT_CHECKPOINT_GROUP_SIZE"
  --empty-cache-min-record-tokens "$EMPTY_CACHE_MIN_RECORD_TOKENS"
  --mlp-checkpoint-chunk-size "$MLP_CHECKPOINT_CHUNK_SIZE"
  --lr "$LR" --warmup-steps "$WARMUP_STEPS"
  --sampler-key "$SAMPLER_KEY" --sampler-mode "$SAMPLER_MODE"
  --island-size 8 --merge-bucket-numel 67108864
  --offload-schedulefree-state --schedulefree-offload-bucket-numel 67108864
)
printf '%q ' "${COMMAND[@]}" > "$RUN_ROOT/identity/command.txt"; printf '\n' >> "$RUN_ROOT/identity/command.txt"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi
if [[ ${ACQUIRE_GPUS:-1} == 1 ]]; then
  eval "$(scripts/gpu_lease.sh acquire 8 --no-wait)"
fi
export NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 TIKTOKEN_CACHE_DIR=/tmp/data-gym-cache
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX=/tmp/e97-4b-pi-sft-${RUN_ID}
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/logs/run.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/return-code.txt"
if (( rc == 0 )); then
  latest=$(readlink -f "$RUN_ROOT/checkpoints/latest.pt")
  python scripts/verify_e97_4b_pi_sft_checkpoint.py "$latest" \
    --output "$RUN_ROOT/terminal/checkpoint.reload.json"
  sha256sum "$latest" > "$RUN_ROOT/terminal/checkpoint.sha256"
  echo "LOCAL_PI_SFT_QUALIFICATION_COMPLETE checkpoint=$latest"
fi
exit "$rc"
