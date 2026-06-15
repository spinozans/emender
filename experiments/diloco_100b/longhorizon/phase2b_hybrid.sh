#!/usr/bin/env bash
# diloco-loss-parity-longhorizon PHASE 2b: re-run ONLY the hybrid arm (the Phase 2
# hybrid crashed on a concurrent NCCL subgroup-init deadlock at DDP construction;
# fixed in train.py by sequential subgroup-comm warmup). 6-GPU = 3 islands x 2:
# per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS
# islands every K=250, beta=0, to 1000 steps (healthy regime, before the recipe
# collapse). Held-out BPB per consensus checkpoint vs the clean DDP floor.
set -uo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1439
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
HELDOUT=/home/erikg/ndm/.wg-worktrees/agent-1433/experiments/preflight_100b/heldout_comma_p50k_2048.pt
LH="$REPO/experiments/diloco_100b/longhorizon"
RES="$LH/results.txt"
cd "$REPO"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# NCCL P2P over PCIe deadlocks when initializing per-island SUBGROUP communicators
# on this no-NVLink box (verified: a 2-rank subgroup all-reduce hangs 600 s with P2P
# on, completes instantly with it off). Required for the hybrid's intra-island groups.
export NCCL_P2P_DISABLE=1

eval "$(scripts/gpu_lease.sh 6)"
LEASED="$CUDA_VISIBLE_DEVICES"
NPROC=$(echo "$LEASED" | tr ',' '\n' | wc -l)
EVAL_GPU=$(echo "$LEASED" | cut -d, -f1)
echo "PHASE2B LEASED=$LEASED NPROC=$NPROC EVAL_GPU=$EVAL_GPU" | tee -a "$RES"

EM=(--dim 1792 --depth 11 --lr 0.0010071509461604343 --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64)
tag=hybrid_isl2_b0; out=/tmp/lh_hybrid2
echo "ARM_START $tag nproc=$NPROC $(date -u +%FT%TZ)" | tee -a "$RES"
echo "PHASE2_TOKSTEP $tag $((6*2048*NPROC))" | tee -a "$RES"
rm -rf "$out"; mkdir -p "$out"
CUDA_VISIBLE_DEVICES="$LEASED" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node="$NPROC" train.py \
  --data "$DATA" --bf16 --batch_size 6 --chunk_size 2048 --steps 1000 \
  --output "$out" --optimizer schedulefree --seed 42 \
  --save_every 250 --keep_checkpoints 6 --tokenizer p50k_base --log_every 50 \
  "${EM[@]}" --diloco --diloco_k 250 --diloco_island_size 2 \
  --diloco_outer_beta 0.0 --diloco_outer_lr 1.0 > "$LH/logs/${tag}.log" 2>&1
echo "ARM_TRAIN_RC $tag $? $(date -u +%FT%TZ)" | tee -a "$RES"
rundir=$(find "$out" -mindepth 1 -maxdepth 1 -type d | head -1)
for ck in $(ls "$rundir"/checkpoint_step_*.pt 2>/dev/null | sort); do
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 "$LH/ckpt_bpb.py" \
    --ckpt "$ck" --heldout "$HELDOUT" --tag "$tag" 2>>"$LH/logs/${tag}_eval.log" | tee -a "$RES"
done
grep -E "global_tok/s" "$LH/logs/${tag}.log" | tail -15 | \
  awk -F'global_tok/s' '{split($2,a," "); s+=a[1]; n++} END{if(n)printf "PHASE2_TOKS %s %.0f\n",T,s/n}' T="$tag" | tee -a "$RES"
rm -rf "$out"
echo "ARM_DONE $tag $(date -u +%FT%TZ)" | tee -a "$RES"
echo "PHASE2B_DONE $(date -u +%FT%TZ)" | tee -a "$RES"
