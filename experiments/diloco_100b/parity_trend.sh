#!/usr/bin/env bash
# implement-diloco-periodic: loss-vs-tokens PARITY TREND. Run DDP and DiLoCo(K=100)
# fresh to 600 steps (continuous stream 0 -> 51.6M global tokens) with a checkpoint
# at step 300 and 600, under ONE 7-GPU lease. Held-out BPB is then evaluated at
# matched token counts (25.8M @ step300, 51.6M @ step600) to show whether the
# DiLoCo merged-model gap to synchronous DDP shrinks (tracks, no divergence) as
# training proceeds. Same emender-1.286B geometry / data / seed / bs6 / ctx2048.
set -euo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1436
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
cd "$REPO"
mkdir -p experiments/diloco_100b/logs

eval "$(scripts/gpu_lease.sh 7)"
echo "LEASED CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

EM=(--dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64)
COMMON=(--data "$DATA" --bf16 --batch_size 6 --chunk_size 2048 --optimizer schedulefree \
  --seed 42 --save_every 300 --keep_checkpoints 4 --tokenizer p50k_base --log_every 50 --steps 600)

echo "=== DDP 600 ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node="$NGPU" train.py "${COMMON[@]}" "${EM[@]}" \
  --output /tmp/diloco_parity_ddp 2>&1 | tee experiments/diloco_100b/logs/parity_ddp600.log

echo "=== DiLoCo K=100 600 ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node="$NGPU" train.py "${COMMON[@]}" "${EM[@]}" \
  --diloco --diloco_k 100 --output /tmp/diloco_parity_dil 2>&1 | tee experiments/diloco_100b/logs/parity_dil600.log

echo "PARITY_TREND_DONE"
