#!/usr/bin/env bash
# implement-diloco-periodic: 7-GPU DiLoCo periodic-sync run for one arm. Leases 7
# GPUs via the broker, launches torchrun with nproc=7, real commapile_mainmix data,
# bf16 + fused, and --diloco (periodic model-weight averaging every K steps instead
# of per-step DDP gradient all-reduce). Measures the effective global tok/s, which
# (unlike vanilla DDP at 31,291 tok/s = 52% eff) should approach the ~62k tok/s
# independent ceiling because the 1.29B all-reduce is amortized over K local steps.
#
# Usage: run_diloco.sh <arm: emender|gdn2> <steps> <batch_size> <K> <outdir> [extra train.py args...]
set -euo pipefail
ARM="$1"; STEPS="$2"; BS="$3"; K="$4"; OUT="$5"; shift 5
REPO=/home/erikg/ndm/.wg-worktrees/agent-1436
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
cd "$REPO"

eval "$(scripts/gpu_lease.sh 7)"
echo "LEASED CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "NGPU=$NGPU  DILOCO_K=$K"

COMMON=(--data "$DATA" --bf16 --batch_size "$BS" --chunk_size 2048 --steps "$STEPS" \
  --output "$OUT" --optimizer schedulefree --seed 42 --save_every 999999 --keep_checkpoints 1 \
  --tokenizer p50k_base --log_every 25 --diloco --diloco_k "$K")

if [ "$ARM" = "emender" ]; then
  ARGS=(--dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
    --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
    --mlp_ratio 2.262336203876648 --mlp_multiple 64)
elif [ "$ARM" = "gdn2" ]; then
  ARGS=(--dim 2176 --depth 12 --lr 0.00047431158698290157 --level gdn2-mlp --expansion 1 \
    --n_heads 30 --gdn2_mlp_ratio 3.258732449079677 --use_conv 1 --d_conv 4)
else
  echo "unknown arm $ARM"; exit 1
fi

torchrun --standalone --nproc_per_node="$NGPU" train.py "${COMMON[@]}" "${ARGS[@]}" "$@"
