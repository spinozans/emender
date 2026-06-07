#!/usr/bin/env bash
# Wave runner for the e97-raw-plus matched comparison.
# Each candidate -> its own GPU (4..7), token-matched via fixed --steps.
# REAL commapile data + REAL held-out slice. No mocks.
set -u
REPO=/home/erikg/ndm/.wg-worktrees/agent-1169
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_smoke_1gb.txt
VAL=/tmp/e97_heldout_val.txt
OUT=${OUT:-/mnt/nvme1n1/erikg/e97_raw_mlp_runs}
STEPS=${STEPS:-3000}
VALEVERY=${VALEVERY:-1000}
mkdir -p "$OUT"
export GDN2_PATH=/home/erikg/GatedDeltaNet-2

COMMON="--data $DATA --val_data $VAL --bf16 --batch_size 2 --chunk_size 2048 \
 --tokenizer p50k_base --optimizer schedulefree --seed 42 \
 --steps $STEPS --val_every $VALEVERY --log_every 50 --save_every 999999 --keep_checkpoints 1"

run() { # name gpu lr extra...
  local name=$1 gpu=$2 lr=$3; shift 3
  local odir="$OUT/$name"
  echo "[launch] $name on GPU$gpu lr=$lr -> $odir"
  CUDA_VISIBLE_DEVICES=$gpu nohup python -u "$REPO/train.py" $COMMON \
    --lr $lr --output "$odir" "$@" > "$OUT/$name.log" 2>&1 &
  echo $! > "$OUT/$name.pid"
}

# A: e97-raw mixer-only (baseline)
run e97raw_mixer 4 9e-4 \
  --level E97 --e88_raw_write 1 --n_heads 354 --n_state 32 --expansion 1.0 \
  --use_gate 1 --gate_activation silu --use_triton 1 --dim 1536 --depth 10 --mlp_ratio 0

# B: e97-raw + MLP bolt-on (small MLP, depth-matched)
run e97raw_mlp_bolt 5 9e-4 \
  --level E97 --e88_raw_write 1 --n_heads 323 --n_state 32 --expansion 1.0 \
  --use_gate 1 --gate_activation silu --use_triton 1 --dim 1536 --depth 10 --mlp_ratio 1.5

# C: e97-raw + MLP reallocated (shrunk mixer, deeper, bigger MLP)
run e97raw_mlp_realloc 6 9e-4 \
  --level E97 --e88_raw_write 1 --n_heads 128 --n_state 32 --expansion 1.0 \
  --use_gate 1 --gate_activation silu --use_triton 1 --dim 1536 --depth 21 --mlp_ratio 2.0

# D: gdn2-mlp reference (leaderboard rank-2 geometry)
run gdn2_mlp 7 2.45e-3 \
  --level gdn2 --n_heads 8 --expansion 2.0 --dim 2304 --depth 17 --mlp_ratio 2.854

wait
echo "[wave complete]"
