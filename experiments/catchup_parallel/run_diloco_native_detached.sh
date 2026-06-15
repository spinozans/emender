#!/usr/bin/env bash
# Launch target for catchup-parallel-diloco. Run this script under setsid/nohup;
# it owns the GPU broker lease and keeps the heartbeat alive after the agent exits.
set -euo pipefail

REPO=${REPO:-/home/erikg/ndm/.wg-worktrees/agent-1451}
DURABLE=${DURABLE:-/mnt/nvme1n1/erikg/catchup_parallel/diloco_native_emender_mlp}
DATA=${DATA:-/home/erikg/elman/data/pile.txt}
HELDOUT=${HELDOUT:-/home/erikg/ndm/.wg-worktrees/agent-1411/experiments/lb_compare_20260613/heldout_p50k_2048.pt}
GPU_COUNT=${GPU_COUNT:-6}
STEPS=${STEPS:-12210}
DILOCO_K=${DILOCO_K:-250}
SAVE_EVERY=${SAVE_EVERY:-250}
TAG=${TAG:-diloco_native_emender_mlp_bs4_lr1007e-3_k250}

mkdir -p "$DURABLE"/{train,logs}
cd "$REPO"

exec > >(tee -a "$DURABLE/logs/supervisor.log") 2>&1
echo "SUPERVISOR_START $(date -u +%FT%TZ)"
echo "REPO=$REPO"
echo "DURABLE=$DURABLE"
echo "DATA=$DATA"
echo "HELDOUT=$HELDOUT"

# Acquire one eval GPU first, then the training GPUs. The broker accumulates
# leases in this shell; the installed EXIT trap releases all of them together.
eval "$(scripts/gpu_lease.sh acquire 1)"
EVAL_GPU="$CUDA_VISIBLE_DEVICES"
TRAIN_COUNT=$((GPU_COUNT - 1))
eval "$(scripts/gpu_lease.sh acquire "$TRAIN_COUNT")"
ALL_GPUS="$CUDA_VISIBLE_DEVICES"
TRAIN_GPUS=$(printf '%s\n' "$ALL_GPUS" | tr ',' '\n' | grep -v "^${EVAL_GPU}$" | paste -sd,)
NGPU=$(printf '%s\n' "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)

if [ "$NGPU" -lt 1 ]; then
  echo "no training GPUs available after lease"; exit 1
fi

echo "$$" > "$DURABLE/run.pid"
printf '%s\n' "$EVAL_GPU" > "$DURABLE/eval_gpu.txt"
printf '%s\n' "$TRAIN_GPUS" > "$DURABLE/train_gpus.txt"
printf '%s\n' "$ALL_GPUS" > "$DURABLE/all_leased_gpus.txt"

CURVE="$DURABLE/heldout_curve.csv"
if [ ! -s "$CURVE" ]; then
  printf 'tag,step,tokens,heldout_bpb,heldout_ce,scored_tokens,bytes_per_token,checkpoint\n' > "$CURVE"
fi

cat > "$DURABLE/train_command.txt" <<EOF
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --standalone --nproc_per_node=$NGPU train.py \\
  --data "$DATA" --bf16 --batch_size 4 --chunk_size 2048 --steps "$STEPS" \\
  --output "$DURABLE/train" --optimizer schedulefree --seed 42 \\
  --save_every "$SAVE_EVERY" --keep_checkpoints 100 --tokenizer p50k_base --log_every 25 \\
  --diloco --diloco_k "$DILOCO_K" --diloco_outer_lr 1 --diloco_outer_beta 0 \\
  --dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \\
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \\
  --mlp_ratio 2.262336203876648 --mlp_multiple 64
EOF

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XMA_PATH=${XMA_PATH:-/home/erikg/xma}

rm -rf "$DURABLE/train"
mkdir -p "$DURABLE/train"
echo "TRAIN_START $(date -u +%FT%TZ) eval_gpu=$EVAL_GPU train_gpus=$TRAIN_GPUS ngpu=$NGPU"
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun --standalone --nproc_per_node="$NGPU" train.py \
  --data "$DATA" --bf16 --batch_size 4 --chunk_size 2048 --steps "$STEPS" \
  --output "$DURABLE/train" --optimizer schedulefree --seed 42 \
  --save_every "$SAVE_EVERY" --keep_checkpoints 100 --tokenizer p50k_base --log_every 25 \
  --diloco --diloco_k "$DILOCO_K" --diloco_outer_lr 1 --diloco_outer_beta 0 \
  --dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64 \
  > "$DURABLE/logs/train.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$DURABLE/train.pid"

declare -A SEEN
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  for ckpt in $(find "$DURABLE/train" -name 'checkpoint_step_*.pt' -type f 2>/dev/null | sort); do
    [ -n "${SEEN[$ckpt]:-}" ] && continue
    s1=$(stat -c %s "$ckpt" 2>/dev/null || echo 0)
    sleep 5
    s2=$(stat -c %s "$ckpt" 2>/dev/null || echo 0)
    [ "$s1" != "$s2" ] && continue
    SEEN[$ckpt]=1
    echo "EVAL_START $(date -u +%FT%TZ) $ckpt"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 experiments/catchup_parallel/eval_emender_mlp_checkpoint_bpb.py \
      --ckpt "$ckpt" --heldout "$HELDOUT" --tag "$TAG" --out-csv "$CURVE" \
      >> "$DURABLE/logs/eval.log" 2>&1 || echo "EVAL_FAILED $ckpt rc=$?"
  done
  sleep 20
done

wait "$TRAIN_PID"
RC=$?
echo "TRAIN_EXIT rc=$RC $(date -u +%FT%TZ)"
exit "$RC"
