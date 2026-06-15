#!/usr/bin/env bash
# diloco-loss-parity-longhorizon PHASE 1 (one 7-GPU lease, sequential arms):
#   A. outer-momentum SWEEP: beta{0.5,0.9} x lr{0.7,1.0} at K=250, to 750 steps
#      (token points 21.5M/43M/64.5M) — does outer momentum close the matched-token gap?
#   B. local-SGD (beta=0,lr=1) K=250 LONG horizon to 2500 steps (215M tok, ~4x the
#      prior 52M) — does the gap close/flat/widen with more tokens?
#   C. vanilla per-step DDP reference to 2500 steps — matched-token baseline at every point.
# After each arm: eval held-out BPB on every saved (consensus) checkpoint via ckpt_bpb.py
# against the agent-1433 preflight heldout (bpt 3.938), append BPB_RESULT lines to
# results.txt, then prune the run dir. emender-1.286B, bs6 ctx2048 bf16+fused, seed 42.
# global tokens/step = 6*2048*7 = 86016 (DDP and DiLoCo identical at matched steps).
set -uo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1439
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
HELDOUT=/home/erikg/ndm/.wg-worktrees/agent-1433/experiments/preflight_100b/heldout_comma_p50k_2048.pt
LH="$REPO/experiments/diloco_100b/longhorizon"
RES="$LH/results.txt"
cd "$REPO"

# Prevent a cold Triton autotune (GIL-blocking on first compile of a new shape)
# from tripping the NCCL watchdog and aborting the run. Same geometry the prior
# 7-GPU runs used; monitoring off + long heartbeat is purely defensive.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

eval "$(scripts/gpu_lease.sh 7)"
LEASED="$CUDA_VISIBLE_DEVICES"
NGPU=$(echo "$LEASED" | tr ',' '\n' | wc -l)
EVAL_GPU=$(echo "$LEASED" | cut -d, -f1)
echo "PHASE1 LEASED=$LEASED NGPU=$NGPU EVAL_GPU=$EVAL_GPU" | tee -a "$RES"
echo "PHASE1_START $(date -u +%FT%TZ)" | tee -a "$RES"

EM=(--dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64)

# run_arm <tag> <out> <save_every> <keep> <steps> [extra diloco/ddp args...]
run_arm() {
  local tag="$1" out="$2" se="$3" keep="$4" steps="$5"; shift 5
  echo "ARM_START $tag $(date -u +%FT%TZ)" | tee -a "$RES"
  rm -rf "$out"; mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$LEASED" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node="$NGPU" train.py \
    --data "$DATA" --bf16 --batch_size 6 --chunk_size 2048 --steps "$steps" \
    --output "$out" --optimizer schedulefree --seed 42 \
    --save_every "$se" --keep_checkpoints "$keep" --tokenizer p50k_base --log_every 50 \
    "${EM[@]}" "$@" > "$LH/logs/${tag}.log" 2>&1
  local rc=$?
  echo "ARM_TRAIN_RC $tag $rc $(date -u +%FT%TZ)" | tee -a "$RES"
  # eval every checkpoint on a single leased GPU
  local rundir
  rundir=$(find "$out" -mindepth 1 -maxdepth 1 -type d | head -1)
  for ck in $(ls "$rundir"/checkpoint_step_*.pt 2>/dev/null | sort); do
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 "$LH/ckpt_bpb.py" \
      --ckpt "$ck" --heldout "$HELDOUT" --tag "$tag" 2>>"$LH/logs/${tag}_eval.log" \
      | tee -a "$RES"
  done
  rm -rf "$out"   # prune to bound disk
  echo "ARM_DONE $tag $(date -u +%FT%TZ)" | tee -a "$RES"
}

# --- A. outer-momentum sweep (K=250, to 500 steps; token points 21.5M + 43M) ---
run_arm dil_b0.5_lr0.7 /tmp/lh_b05_lr07 250 4 500 --diloco --diloco_k 250 --diloco_outer_beta 0.5 --diloco_outer_lr 0.7
run_arm dil_b0.5_lr1.0 /tmp/lh_b05_lr10 250 4 500 --diloco --diloco_k 250 --diloco_outer_beta 0.5 --diloco_outer_lr 1.0
run_arm dil_b0.9_lr0.7 /tmp/lh_b09_lr07 250 4 500 --diloco --diloco_k 250 --diloco_outer_beta 0.9 --diloco_outer_lr 0.7
run_arm dil_b0.9_lr1.0 /tmp/lh_b09_lr10 250 4 500 --diloco --diloco_k 250 --diloco_outer_beta 0.9 --diloco_outer_lr 1.0

# --- B. local-SGD (beta=0) long horizon to 2500 steps (215M tokens) ---
run_arm dil_b0_long /tmp/lh_b0_long 500 8 2500 --diloco --diloco_k 250 --diloco_outer_beta 0.0 --diloco_outer_lr 1.0

# --- C. vanilla per-step DDP reference to 2500 steps (matched-token baseline) ---
run_arm ddp_long /tmp/lh_ddp_long 250 12 2500

echo "PHASE1_DONE $(date -u +%FT%TZ)" | tee -a "$RES"
