#!/usr/bin/env bash
# Seed-2 panel-resample robustness run (the "seeds" axis): identical pipeline as
# run_capability_track.sh but on the independently sampled s2 panels. task:
# capability-track-references.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REPO_ROOT="$(pwd)"; OUT_DIR="$REPO_ROOT/results/capability-track-references"; PANELS="$OUT_DIR/panels"
EMENDER_DIR="/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750"
GDN2_DIR="/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627"
export GDN2_PATH="${GDN2_PATH:-/home/erikg/GatedDeltaNet-2}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
eval "$(scripts/gpu_lease.sh acquire 1)"; export EVAL_CHECKPOINT_GPU_LEASED=1
echo "[s2] leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python scripts/capability_track_references.py \
  --model "emender=$EMENDER_DIR" --model "gdn2=$GDN2_DIR" \
  --panel "knowledge=$PANELS/knowledge_panel_s2.jsonl" \
  --panel "reasoning=$PANELS/reasoning_panel_s2.jsonl" \
  --y-mode train --batch-size "${CAP_EVAL_BS:-16}" \
  --out-csv "$OUT_DIR/capability_by_checkpoint_s2.csv" \
  --out-items "$OUT_DIR/capability_items_s2.jsonl" --keep-going
echo "[s2] DONE"
