#!/usr/bin/env bash
# diloco-loss-parity-longhorizon PHASE 2 (one lease, sequential): test candidate
# FIXES for the local-SGD consensus collapse Phase 1 found (beta=0 K=250 held-out
# BPB exploded 2.03@86M -> 3.19@215M while per-rank train loss stayed ~5.0). All
# arms run to 2500 steps (215M tok) so they are directly comparable to dil_b0_long.
#
#   M. momentum-long  7-GPU  K=250 beta=0.9 lr=1.0  (canonical DiLoCo outer Nesterov)
#   S. smallK-long    7-GPU  K=50  beta=0   lr=1.0  (frequent sync -> less drift)
#   H. hybrid-long    6-GPU  3 islands x 2 (per-step DDP within island + DiLoCo
#                     across islands every K=250, beta=0) (tighter sync, 3 pts to avg)
#
# After each arm: held-out BPB on every consensus checkpoint (agent-1433 preflight
# tensor, bpt 3.938), appended to results.txt, then prune. emender-1.286B bs6 ctx2048
# bf16+fused seed 42. 7-GPU tok/step = 86016; 6-GPU hybrid tok/step = 6*2048*6 = 73728
# (analyze.py converts step->tokens per arm via its own world size; see PHASE2_TOKSTEP).
set -uo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1439
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
HELDOUT=/home/erikg/ndm/.wg-worktrees/agent-1433/experiments/preflight_100b/heldout_comma_p50k_2048.pt
LH="$REPO/experiments/diloco_100b/longhorizon"
RES="$LH/results.txt"
cd "$REPO"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

eval "$(scripts/gpu_lease.sh 7)"
LEASED="$CUDA_VISIBLE_DEVICES"
NGPU_ALL=$(echo "$LEASED" | tr ',' '\n' | wc -l)
EVAL_GPU=$(echo "$LEASED" | cut -d, -f1)
echo "PHASE2 LEASED=$LEASED NGPU_ALL=$NGPU_ALL EVAL_GPU=$EVAL_GPU" | tee -a "$RES"
echo "PHASE2_START $(date -u +%FT%TZ)" | tee -a "$RES"

EM=(--dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64)

# run_arm <tag> <out> <nproc> <save_every> <keep> <steps> [extra train.py args...]
run_arm() {
  local tag="$1" out="$2" nproc="$3" se="$4" keep="$5" steps="$6"; shift 6
  local gpus
  gpus=$(echo "$LEASED" | cut -d, -f1-"$nproc")
  echo "ARM_START $tag nproc=$nproc gpus=$gpus $(date -u +%FT%TZ)" | tee -a "$RES"
  echo "PHASE2_TOKSTEP $tag $((6*2048*nproc))" | tee -a "$RES"
  rm -rf "$out"; mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpus" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node="$nproc" train.py \
    --data "$DATA" --bf16 --batch_size 6 --chunk_size 2048 --steps "$steps" \
    --output "$out" --optimizer schedulefree --seed 42 \
    --save_every "$se" --keep_checkpoints "$keep" --tokenizer p50k_base --log_every 50 \
    "${EM[@]}" "$@" > "$LH/logs/${tag}.log" 2>&1
  echo "ARM_TRAIN_RC $tag $? $(date -u +%FT%TZ)" | tee -a "$RES"
  local rundir
  rundir=$(find "$out" -mindepth 1 -maxdepth 1 -type d | head -1)
  for ck in $(ls "$rundir"/checkpoint_step_*.pt 2>/dev/null | sort); do
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 "$LH/ckpt_bpb.py" \
      --ckpt "$ck" --heldout "$HELDOUT" --tag "$tag" 2>>"$LH/logs/${tag}_eval.log" \
      | tee -a "$RES"
  done
  # record measured throughput for this arm
  grep -E "global_tok/s" "$LH/logs/${tag}.log" | tail -20 | \
    awk -F'global_tok/s' '{split($2,a," "); s+=a[1]; n++} END{if(n)printf "PHASE2_TOKS %s %.0f\n",T,s/n}' T="$tag" | tee -a "$RES"
  rm -rf "$out"
  echo "ARM_DONE $tag $(date -u +%FT%TZ)" | tee -a "$RES"
}

# Healthy-regime horizon (1000 steps = 86M tok): Phase 1 showed the shared LR-recipe
# collapse onsets ~64-86M (DDP held-out bottoms 1.571@64.5M then climbs to 3.234@215M),
# so a long-horizon comparison is confounded by the collapsing baseline (recipe fix =
# follow-up fix-long-horizon). Here we compare every DiLoCo variant against the CLEAN
# DDP floor at matched tokens (21.5/43/64.5/86M) to see if any closes the ~0.45 BPB
# local-SGD gap. save_every 250 -> token points 250/500/750/1000.
run_arm mom_b0.9_lr1.0 /tmp/lh_mom 7 250 6 1000 --diloco --diloco_k 250 --diloco_outer_beta 0.9 --diloco_outer_lr 1.0
run_arm smallK50_b0    /tmp/lh_smallk 7 250 6 1000 --diloco --diloco_k 50 --diloco_outer_beta 0.0 --diloco_outer_lr 1.0
run_arm hybrid_isl2_b0 /tmp/lh_hybrid 6 250 6 1000 --diloco --diloco_k 250 --diloco_island_size 2 --diloco_outer_beta 0.0 --diloco_outer_lr 1.0

echo "PHASE2_DONE $(date -u +%FT%TZ)" | tee -a "$RES"
