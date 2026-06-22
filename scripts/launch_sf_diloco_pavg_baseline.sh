#!/usr/bin/env bash
# launch_sf_diloco_pavg_baseline.sh  (task sf-diloco-p1)
#
# Short reproduction of the STABLE ScheduleFree-DiLoCo plain-average (local-SGD)
# baseline: --optimizer schedulefree --diloco --diloco_outer_lr 1.0
# --diloco_outer_beta 0.0. This is the UN-bugged averaging path (outer_state is
# None, so the sf-diloco-p1 x/z/y outer-update fix does not touch it); the run
# exists to confirm the avg path still descends smoothly across merge boundaries
# with NO post-merge loss spikes, under the fused-guard NO-eager guarantee.
#
# Geometry / optimizer / data / env are byte-identical to the proven
# launch_emender_8gpu_diloco.sh recipe (1.3B E97 emender, p50k_base, pile.txt).
# Differences vs that 48h run: fewer GPUs (default 2 -- enough for a real
# cross-rank average) and a bounded --steps for a SHORT reproduction.
#
# Self-leasing: acquires N exclusive GPUs via scripts/gpu_lease.sh (--no-wait)
# and auto-releases on exit. REAL data, REAL fused Triton kernel.
#
# Usage:
#   scripts/launch_sf_diloco_pavg_baseline.sh                 # K=250, 2 GPUs, 1100 steps
#   K=100 STEPS=1100 scripts/launch_sf_diloco_pavg_baseline.sh
#   GPUS=2 LOGFILE=/tmp/x.log scripts/launch_sf_diloco_pavg_baseline.sh
#   OUTER_BETA=0.9 OUTER_LR=0.1 NDM_DILOCO_DEBUG_ASSERT=1 EXTRA_ARGS='--final_heldout_eval ...' scripts/launch_sf_diloco_pavg_baseline.sh
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

K="${K:-250}"
STEPS="${STEPS:-1100}"
GPUS="${GPUS:-2}"
DATA="${DATA:-/home/erikg/elman/data/pile.txt}"
OUTPUT="${OUTPUT:-/tmp/sf_diloco_pavg_baseline/k${K}}"
LOGFILE="${LOGFILE:-/tmp/sf_diloco_pavg_baseline_k${K}.log}"
OUTER_LR="${OUTER_LR:-1.0}"
OUTER_BETA="${OUTER_BETA:-0.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
mkdir -p "$OUTPUT"

echo "[baseline] acquiring $GPUS GPU lease (--no-wait) ..."
eval "$(scripts/gpu_lease.sh acquire "$GPUS" --no-wait)"
echo "[baseline] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  K=$K  STEPS=$STEPS  OUTER_LR=$OUTER_LR  OUTER_BETA=$OUTER_BETA  LOG=$LOGFILE"

ENV_ARGS=(
  env
  NDM_DILOCO_DEBUG_ASSERT="${NDM_DILOCO_DEBUG_ASSERT:-0}"
  NCCL_P2P_DISABLE=1
  TORCH_NCCL_ENABLE_MONITORING=0
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

"${ENV_ARGS[@]}" \
  torchrun --standalone --nproc_per_node="$GPUS" train.py \
  --level E97 \
  --dim 1792 \
  --n_heads 216 \
  --n_state 32 \
  --depth 11 \
  --expansion 1.0 \
  --use_gate 1 \
  --gate_activation silu \
  --mlp_ratio 2.2623 \
  --mlp_multiple 64 \
  --use_triton 1 \
  --optimizer schedulefree \
  --lr 0.001007 \
  --bf16 \
  --batch_size 4 \
  --chunk_size 2048 \
  --data "$DATA" \
  --tokenizer p50k_base \
  --diloco \
  --diloco_k "$K" \
  --diloco_outer_lr "$OUTER_LR" \
  --diloco_outer_beta "$OUTER_BETA" \
  --steps "$STEPS" \
  --save_every 100000000 \
  --keep_checkpoints 1 \
  --log_every 25 \
  --output "$OUTPUT" \
  $EXTRA_ARGS 2>&1 | tee "$LOGFILE"

echo "[baseline] DONE K=$K -> $LOGFILE"
