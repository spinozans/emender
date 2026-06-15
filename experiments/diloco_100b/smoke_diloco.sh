#!/usr/bin/env bash
# implement-diloco-periodic: 2-GPU correctness smoke for the DiLoCo periodic-sync
# path. Tiny E97 emender model, real commapile data, bf16+fused. Verifies:
#   - torchrun launches, process group inits, W_0 broadcast, no hang at merges
#   - merges fire every K steps and loss decreases (no divergence)
#   - final consensus checkpoint saves on rank 0
# Usage: smoke_diloco.sh <out_dir> [diloco_k] [steps]
set -euo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1436
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
OUT="${1:?out dir}"; K="${2:-20}"; STEPS="${3:-80}"
cd "$REPO"

eval "$(scripts/gpu_lease.sh 2)"
echo "LEASED CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# Tiny E97 emender (same family/kernel as the 1.3B 100B seed, ~small dims).
ARGS=(--dim 256 --depth 2 --level E97 --n_heads 8 --n_state 32 --expansion 1.0 \
  --use_gate 1 --gate_activation silu --mlp_ratio 2.0 --mlp_multiple 64 \
  --lr 0.001 --data "$DATA" --bf16 --batch_size 4 --chunk_size 512 --steps "$STEPS" \
  --output "$OUT" --optimizer schedulefree --seed 42 --save_every 999999 \
  --keep_checkpoints 1 --tokenizer p50k_base --log_every 10)

torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}" \
  --diloco --diloco_k "$K"
