#!/usr/bin/env bash
# Matched-compute LM run for the complex-eigenvalue head on the FUSED Triton kernel
# (task complex-eig-lm-2). Same protocol as the prior run_ceig_lm.sh but pinned to
# THIS worktree (fused kernel is the head default: cplx_fused_triton=True).
# Usage: run_ceig_lm_fused.sh <gpu> <level> <mlp_ratio> <budget_flag> <budget_val> <outdir> [layer_kwargs_json]
#   budget_flag: --steps  (matched-tokens)  or  --train_minutes (matched-wallclock)
set -euo pipefail
GPU=$1; LEVEL=$2; MLP=$3; BFLAG=$4; BVAL=$5; OUT=$6; LK=${7:-}
DATA=/mnt/nvme2n1/erikg/pile.txt
VAL=/mnt/nvme2n1/erikg/complex_eig_lm_val.txt
cd /home/erikg/ndm/.wg-worktrees/agent-1293
ARGS=(--data "$DATA" --val_data "$VAL"
  --level "$LEVEL" --dim 512 --depth 6 --n_heads 8 --n_state 64
  --expansion 1.0 --mlp_ratio "$MLP" --use_gate 1 --use_conv 1
  --batch_size 16 --chunk_size 512 "$BFLAG" "$BVAL"
  --lr 2e-3 --weight_decay 0.0 --optimizer schedulefree --bf16 --seed 42
  --log_every 100 --val_every 100000 --save_every 100000 --keep_checkpoints 1
  --compile_warmup_steps 20 --timer_after_compile_warmup
  --final_heldout_eval --final_val_batches 200 --heldout_bytes_per_token 1.0
  --output "$OUT")
if [ -n "$LK" ]; then ARGS+=(--layer_kwargs "$LK"); fi
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
echo "[run_ceig_fused] GPU=$GPU level=$LEVEL mlp=$MLP $BFLAG=$BVAL out=$OUT lk=$LK"
exec python train.py "${ARGS[@]}"
