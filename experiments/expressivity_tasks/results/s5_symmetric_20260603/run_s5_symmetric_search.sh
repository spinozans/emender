#!/usr/bin/env bash
# S5-SYMMETRIC CMA-ES SEARCH — 4-ARM, 300-step candidates (task s5sym-search, agent-1009).
#
# FOUR arms; linear_state is FIXED per arm and REMOVED from the CMA search space:
#   1. e88-tanh   : layer E88, linear_state=0 (tanh ON), use_gate=1
#   2. e88-linear : layer E88, linear_state=1 (linear state), use_gate=1
#   3. m2rnn      : M2RNN-CMA
#   4. gdn        : fla-gdn
# The tanh-vs-linear (BL-1) decision is made by full-fidelity eval of two
# separately-tuned E88 models (winner-eval phase), NOT inside the truncated
# 300-step search — so linear_state/use_gate do NOT vary inside CMA.
#
# Per-candidate: 300 steps, seq_len 128, batch 32. Candidate eval T=128 ONLY
#   (the fitness). 256/512/1024 extrapolation is held out for the winner-eval.
# pop 10, hard 7-generation cap, converge < 0.005 acc over 3 consecutive gens
#   (min 3 gens), sigma 0.35, seed 42.
# Each arm CMA's over lr/dim/depth/n_heads/n_state, seeded from its MODEL_CONFIG
#   center, ~8M real-param matched +/-10% (rematch_dim_s5).
# GPUs: 0-7 round-robin (8-wide); with pop 10 this is ~2 waves/gen. ONE arm at a
#   time uses all idle GPUs. The idle GPU subset is detected at launch and LOGGED.
#
# Driver: scripts/cmaes_search_s5.py --objective s5_acc@T128 (REAL train_hybrid
#   s5_permutation eval; fitness = 1 - mean eval_acc@T128 over the final window).
set -u

REPO_ROOT="/home/erikg/ndm/.wg-worktrees/agent-997"
cd "$REPO_ROOT" || exit 3
RESULTS=experiments/expressivity_tasks/results/s5_symmetric_20260603
SEEDS=$RESULTS/seeds_s5_symmetric.json
S5_STEPS=300         # per-candidate cap (task override; NOT the calibrated 2000)
LOGDIR=$RESULTS/driver_logs
mkdir -p "$LOGDIR"

# --- GPU GATE: detect the idle subset of 0-7 (<2GB used). Use it; LOG it.
#     No wait-spin: whatever is idle NOW is what we use. Abort only if none. ---
IDLE=""
while IFS=',' read -r idx mem util; do
  idx=$(echo "$idx" | tr -d ' '); mem=$(echo "$mem" | tr -d ' MiB')
  if [ "$mem" -lt 2048 ]; then
    IDLE="${IDLE:+$IDLE,}$idx"
  else
    echo "GPU_BUSY: GPU $idx has ${mem}MiB used (>=2048MiB) — excluded." >&2
  fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)

if [ -z "$IDLE" ]; then
  echo "S5_SEARCH_ABORTED: no idle GPU in 0-7. No training launched." >&2
  exit 2
fi
echo "GPU_GATE_PASS: idle GPUs = [$IDLE]. Using s5_steps=$S5_STEPS, T=128-only candidate eval." | tee "$LOGDIR/gpus_used.txt"

run_arm() {
  local label="$1"; local model="$2"; shift 2
  local extra=("$@")
  echo "==================== ARM: $label (model=$model, s5_steps=$S5_STEPS, gpus=$IDLE) ===================="
  CUDA_VISIBLE_DEVICES="$IDLE" python scripts/cmaes_search_s5.py \
    --model "$model" --run_label "$label" \
    --objective s5_acc@T128 --phase cmaes \
    --anchor_configs "$SEEDS" --anchor_only_cmaes \
    --params 8M --param_tolerance 0.10 \
    --popsize 10 --sigma 0.35 \
    --min_generations 3 --converge 0.005 --consecutive 3 --max_generations 7 \
    --s5_steps "$S5_STEPS" --s5_eval_lengths 128 \
    --gpus "$IDLE" --output "$RESULTS" "${extra[@]}"
  echo "==================== ARM $label DONE (exit $?) ===================="
}

run_arm e88-tanh   e88     --s5_linear_state 0 --s5_use_gate 1
run_arm e88-linear e88     --s5_linear_state 1 --s5_use_gate 1
run_arm m2rnn      m2rnn
run_arm gdn        fla-gdn

echo "ALL_ARMS_COMPLETE"
