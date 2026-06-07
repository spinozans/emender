#!/usr/bin/env bash
# run_one.sh NAME GPU LEVEL LR STEPS -- <extra train.py args>
# Token-matched single candidate for the e97-raw-plus comparison.
set -u
REPO=/home/erikg/ndm/.wg-worktrees/agent-1169
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_smoke_1gb.txt
VAL=/tmp/e97_heldout_val.txt
OUT=${OUT:-/mnt/nvme1n1/erikg/e97_raw_mlp_runs}
name=$1 gpu=$2 lr=$3 steps=$4; shift 4
export GDN2_PATH=/home/erikg/GatedDeltaNet-2
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$gpu nohup python -u "$REPO/train.py" \
  --data $DATA --val_data $VAL --bf16 --batch_size 2 --chunk_size 2048 \
  --tokenizer p50k_base --optimizer schedulefree --seed 42 \
  --steps $steps --val_every 999999 --log_every 50 --save_every 999999 --keep_checkpoints 1 \
  --lr $lr --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1 &
echo $! > "$OUT/$name.pid"
echo "[launch] $name GPU$gpu lr=$lr steps=$steps pid $(cat $OUT/$name.pid)"
