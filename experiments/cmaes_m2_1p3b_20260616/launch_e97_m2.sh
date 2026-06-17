#!/usr/bin/env bash
# CMA-ES over M2 (multi-query-read E97) at 1.3B — HOMOLOGOUS to the 1.3B
# emender-mlp / gdn2-mlp / pure-E97 / m2rnn searches (docs/SCALE_PLAN.md §1).
#
# Same protocol as scripts/repro_cmaes_1300m/launch_e97_queue.sh:
#   --phase cmaes --params 1300M --param_tolerance 0.03 --train_minutes 15
#   --popsize 8 --sigma 0.8 --chunk_size 2048 --tokenizer p50k_base --data pile.txt
#   --min_generations 12 --use_triton_e88  (bf16 + FUSED, fused-guard asserts no eager)
#
# The ONLY difference vs pure-E97 (`e97`): model_type is `e97-m2`, whose search
# space adds one axis — multiquery_r (R, 1..8) — the M2 readout rank. R=1 is
# byte-identical to pure-E97. The CMA explores R jointly with dim/n_heads/n_state/
# depth/lr/batch under the SAME ~1.3B param target (real params verified == estimate
# to <1e-4 at the anchor), so the readout-rank-vs-width iso-param tradeoff is searched.
#
# GPUs are leased via the broker (CLAUDE.md NON-NEGOTIABLE), not hand-picked.
set -euo pipefail
cd /home/erikg/ndm/.wg-worktrees/agent-1514

export CMAES_MAX_VALID_ATTEMPTS=80

ROOT="experiments/cmaes_m2_1p3b_20260616"
ANCHORS="$ROOT/anchors_e97_m2.json"
NGPU="${NGPU:-2}"

# Reserve NGPU exclusive GPUs; CUDA_VISIBLE_DEVICES set to the leased absolute
# indices; lease auto-releases (+ background heartbeat keeper) on shell exit.
eval "$(scripts/gpu_lease.sh ${NGPU})"
echo "[m2-cma] $(date -u +%Y-%m-%dT%H:%M:%SZ) leased GPUs: ${CUDA_VISIBLE_DEVICES}"

# Each worker subprocess gets CUDA_VISIBLE_DEVICES=<absolute gpu_id> via
# prepare_worker_env, so pass the leased absolute indices straight through.
python -u scripts/cmaes_search_v2.py \
  --model e97-m2 \
  --gpus "${CUDA_VISIBLE_DEVICES}" \
  --output "$ROOT/e97-m2" \
  --phase cmaes \
  --params 1300M \
  --param_tolerance 0.03 \
  --train_minutes 15 \
  --popsize 8 \
  --sigma 0.8 \
  --chunk_size 2048 \
  --tokenizer p50k_base \
  --data /home/erikg/elman/data/pile.txt \
  --min_generations 12 \
  --use_triton_e88 \
  --anchor_configs "$ANCHORS" \
  --anchor_only_cmaes
status=$?
echo "[m2-cma] $(date -u +%Y-%m-%dT%H:%M:%SZ) e97-m2 exit=${status}"
