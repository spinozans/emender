#!/usr/bin/env bash
# implement-diloco-periodic: 7-GPU DiLoCo throughput sweep over K in {100,250,500}
# for the emender-mlp 1.286B seed geometry. ONE 7-GPU lease held for all three
# runs (sequential; each releases GPUs back to the same shell lease). bf16+fused,
# real commapile_mainmix, ctx2048, bs6 (the DDP-max from preflight, also DiLoCo's).
#
# Each run is sized to include >=1 inter-worker merge inside the measured window so
# the reported effective global tok/s already reflects the amortized sync cost.
set -euo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1436
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
OUTROOT="${1:-/tmp/diloco_sweep}"
BS="${2:-6}"
cd "$REPO"
mkdir -p experiments/diloco_100b/logs

eval "$(scripts/gpu_lease.sh 7)"
echo "LEASED CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "NGPU=$NGPU BS=$BS"

EM=(--dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64)
COMMON=(--data "$DATA" --bf16 --batch_size "$BS" --chunk_size 2048 \
  --output "$OUTROOT" --optimizer schedulefree --seed 42 --save_every 999999 \
  --keep_checkpoints 1 --tokenizer p50k_base --log_every 25)

run_k () {
  local K="$1" STEPS="$2" LOG="$3"
  echo "=== DiLoCo K=$K steps=$STEPS -> $LOG ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node="$NGPU" train.py \
    "${COMMON[@]}" "${EM[@]}" --steps "$STEPS" --diloco --diloco_k "$K" \
    2>&1 | tee "$LOG"
}

# K, steps (>=1 merge inside the window + steady state past warmup)
run_k 100 300 experiments/diloco_100b/logs/diloco_k100.log
run_k 250 550 experiments/diloco_100b/logs/diloco_k250.log
run_k 500 650 experiments/diloco_100b/logs/diloco_k500.log

echo "SWEEP_DONE"
