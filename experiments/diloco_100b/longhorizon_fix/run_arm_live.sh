#!/usr/bin/env bash
# fix-long-horizon: run ONE arm of the corrected long-horizon recipe and eval
# held-out BPB on every saved checkpoint LIVE (as each lands) so a mid-run
# held-out rollover is caught early. 7 GPUs train + 1 GPU evals concurrently.
#
#   bash run_arm_live.sh <tag> <out> <steps> <warmup> <lr> [extra train.py args...]
#
# emender-mlp 1.286B, bs6 ctx2048 bf16+FUSED, seed 42, schedule-free AdamW.
# global tokens/step = 6*2048*7 = 86016 (DDP and DiLoCo identical at matched steps).
# Held-out = agent-1433 preflight tensor (bpt 3.938, 65536 tok) — same as the
# broken-recipe longhorizon baseline → directly comparable.
set -uo pipefail
REPO=/home/erikg/ndm/.wg-worktrees/agent-1442
DATA=/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt
HELDOUT=/home/erikg/ndm/.wg-worktrees/agent-1433/experiments/preflight_100b/heldout_comma_p50k_2048.pt
LH="$REPO/experiments/diloco_100b/longhorizon_fix"
RES="$LH/results.txt"
cd "$REPO"

tag="$1"; out="$2"; steps="$3"; warmup="$4"; lr="$5"; shift 5

# Defensive: keep a cold Triton autotune from tripping the NCCL watchdog (same as
# the predecessor 7-GPU runs).
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Lease 1 GPU for eval FIRST, then 7 for training (accumulates; releases together).
eval "$(scripts/gpu_lease.sh acquire 1)"; EVAL_GPU="$CUDA_VISIBLE_DEVICES"
eval "$(scripts/gpu_lease.sh acquire 7)"; ALL="$CUDA_VISIBLE_DEVICES"
TRAIN_GPUS=$(echo "$ALL" | tr ',' '\n' | grep -v "^${EVAL_GPU}$" | paste -sd,)
NGPU=$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)
echo "ARM $tag EVAL_GPU=$EVAL_GPU TRAIN_GPUS=$TRAIN_GPUS NGPU=$NGPU steps=$steps warmup=$warmup lr=$lr $(date -u +%FT%TZ)" | tee -a "$RES"

EM=(--dim 1792 --depth 11 --lr "$lr" --level E97 --n_heads 216 \
  --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.262336203876648 --mlp_multiple 64)

rm -rf "$out"; mkdir -p "$out" "$LH/logs"
echo "ARM_TRAIN_START $tag $(date -u +%FT%TZ)" | tee -a "$RES"
# Optimizer (and any decay flags) come through "$@" so the same runner serves both
# the schedule-free and the AdamW+cosine recipes. Default to schedulefree if the
# caller does not pass --optimizer.
case " $* " in *" --optimizer "*) OPT_ARGS=() ;; *) OPT_ARGS=(--optimizer schedulefree) ;; esac
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun --standalone --nproc_per_node="$NGPU" train.py \
  --data "$DATA" --bf16 --batch_size 6 --chunk_size 2048 --steps "$steps" \
  --output "$out" --seed 42 --warmup_steps "$warmup" "${OPT_ARGS[@]}" \
  --save_every 250 --keep_checkpoints 100 --tokenizer p50k_base --log_every 50 \
  "${EM[@]}" "$@" > "$LH/logs/${tag}.log" 2>&1 &
TRAIN_PID=$!

# Live eval loop: eval each checkpoint as it lands, on the dedicated eval GPU.
declare -A SEEN
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  rundir=$(find "$out" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
  if [ -n "${rundir:-}" ]; then
    for ck in $(ls "$rundir"/checkpoint_step_*.pt 2>/dev/null | sort); do
      [ -n "${SEEN[$ck]:-}" ] && continue
      # wait for the file to finish writing (size stable)
      s1=$(stat -c %s "$ck" 2>/dev/null || echo 0); sleep 3
      s2=$(stat -c %s "$ck" 2>/dev/null || echo 0)
      [ "$s1" != "$s2" ] && continue
      SEEN[$ck]=1
      CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 "$LH/ckpt_bpb.py" \
        --ckpt "$ck" --heldout "$HELDOUT" --tag "$tag" 2>>"$LH/logs/${tag}_eval.log" \
        | tee -a "$RES"
    done
  fi
  sleep 15
done
wait "$TRAIN_PID"; rc=$?
echo "ARM_TRAIN_RC $tag $rc $(date -u +%FT%TZ)" | tee -a "$RES"

# Final sweep: catch any checkpoints written after the last poll (incl. final).
rundir=$(find "$out" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
for ck in $(ls "$rundir"/checkpoint_step_*.pt 2>/dev/null | sort); do
  [ -n "${SEEN[$ck]:-}" ] && continue
  SEEN[$ck]=1
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" python3 "$LH/ckpt_bpb.py" \
    --ckpt "$ck" --heldout "$HELDOUT" --tag "$tag" 2>>"$LH/logs/${tag}_eval.log" \
    | tee -a "$RES"
done
rm -rf "$out"   # prune to bound disk (checkpoints already evaluated)
echo "ARM_DONE $tag $(date -u +%FT%TZ)" | tee -a "$RES"
