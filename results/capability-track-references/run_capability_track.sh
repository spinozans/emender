#!/usr/bin/env bash
# Capability-vs-token tracking on BOTH reference checkpoints (emender E97 +
# gdn2-mlp), on the multiple-choice QA/reasoning panel behind paper Fig 3.
# task: capability-track-references
#
# - Leases exactly ONE idle GPU via the broker (does NOT touch GPUs 0/1 held by
#   the still-running reference trainings, pids 522358 / 584124).
# - Scores ALL present checkpoints of both refs at the checkpoint token cadence,
#   forward-only, schedule-free y-mode swap applied.
# - FUSED kernel only (NON-NEGOTIABLE #1): emender E97 use_triton=1 (hard-imports
#   the split-edit Triton kernel, raises rather than eager); gdn2-mlp
#   GDN2ExternalMLPLayer mode="chunk" (NVIDIA GatedDeltaNet-2 chunked Triton
#   kernel, ImportError if GDN2_PATH absent -- no eager fallback).
# - Panels are built offline from the local HF cache by build_*_eval_panel.py.
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REPO_ROOT="$(pwd)"
OUT_DIR="$REPO_ROOT/results/capability-track-references"
PANELS="$OUT_DIR/panels"

EMENDER_DIR="/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750"
GDN2_DIR="/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627"

export GDN2_PATH="${GDN2_PATH:-/home/erikg/GatedDeltaNet-2}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CAP_EVAL_BS="${CAP_EVAL_BS:-16}"

# Panels (built offline, committed). Rebuild only if missing.
if [[ ! -f "$PANELS/knowledge_panel_s1.jsonl" ]]; then
  python scripts/build_racer_eval_panel.py \
    --out "$PANELS/knowledge_panel_s1.jsonl" --per_task 50 --seed 20260521 --keep_going
fi
if [[ ! -f "$PANELS/reasoning_panel_s1.jsonl" ]]; then
  python scripts/build_reasoning_eval_panel.py \
    --out "$PANELS/reasoning_panel_s1.jsonl" --per_task 160 --seed 20260522 \
    --limit_total 2048 --keep_going
fi

# Lease ONE idle GPU; auto-releases on shell exit (EXIT/INT/TERM trap).
eval "$(scripts/gpu_lease.sh acquire 1)"
export EVAL_CHECKPOINT_GPU_LEASED=1
echo "[run_capability_track] leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python scripts/capability_track_references.py \
  --model "emender=$EMENDER_DIR" \
  --model "gdn2=$GDN2_DIR" \
  --panel "knowledge=$PANELS/knowledge_panel_s1.jsonl" \
  --panel "reasoning=$PANELS/reasoning_panel_s1.jsonl" \
  --y-mode train \
  --batch-size "$CAP_EVAL_BS" \
  --out-csv "$OUT_DIR/capability_by_checkpoint.csv" \
  --out-items "$OUT_DIR/capability_items.jsonl" \
  --keep-going

echo "[run_capability_track] DONE"
