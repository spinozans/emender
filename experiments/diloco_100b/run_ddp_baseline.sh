#!/usr/bin/env bash
# implement-diloco-periodic: vanilla per-step DDP baseline for the loss-vs-tokens
# PARITY check. Same emender-1.286B geometry / data / seed / bs / ctx as the
# DiLoCo sweep, but WITHOUT --diloco (so per-step gradient all-reduce). At matched
# step count the two have processed identical global tokens (bs*ctx*world per step),
# so loss-at-step == loss-at-global-tokens -> a direct divergence check: DiLoCo
# (periodic weight sync) must track DDP (exact-SGD) without diverging.
set -euo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1436
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
OUT="${1:-/tmp/diloco_ddp_base}"; STEPS="${2:-300}"; BS="${3:-6}"
cd "$REPO"
mkdir -p experiments/diloco_100b/logs

eval "$(scripts/gpu_lease.sh 7)"
echo "LEASED CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

EM=(--dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node="$NGPU" train.py \
  --data "$DATA" --bf16 --batch_size "$BS" --chunk_size 2048 --output "$OUT" \
  --optimizer schedulefree --seed 42 --save_every 999999 --keep_checkpoints 1 \
  --tokenizer p50k_base --log_every 25 --steps "$STEPS" "${EM[@]}" \
  2>&1 | tee experiments/diloco_100b/logs/ddp_baseline.log
echo "DDP_BASELINE_DONE"
