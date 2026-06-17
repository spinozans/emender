#!/usr/bin/env bash
# Offline held-out BPB scoring of BOTH references on the SHARED pile-tail tensor.
# task: offline-eval-references
#
# - Leases exactly ONE idle GPU via the broker (does NOT touch GPUs 0/1 held by
#   the still-running reference trainings).
# - Scores ALL present checkpoints of both refs on the same held-out tensor,
#   y-mode swap applied (schedule-free train() weights), forward-only.
# - FUSED kernel only: emender E97 use_triton=1 (hard-imports Triton, no eager);
#   gdn2-mlp GDN2ExternalMLPLayer mode="chunk" (FLA fused chunked GDN-2). No eager.
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REPO_ROOT="$(pwd)"
OUT_DIR="$REPO_ROOT/results/offline-eval-references"

# Shared held-out tensor (byte-identical across all 3 copies, md5 8e1198ab...).
# Use the archived (non-running-dir) copy so we never read a file under an
# actively-written run directory.
HELDOUT="/mnt/nvme1n1/erikg/ref_emender_mlp.contaminated_2057/heldout_pile_tail_p50k_2048_1m.pt"

EMENDER_DIR="/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750"
GDN2_DIR="/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627"

export GDN2_PATH="${GDN2_PATH:-/home/erikg/GatedDeltaNet-2}"
export HELDOUT_EVAL_BS="${HELDOUT_EVAL_BS:-8}"

# Lease ONE idle GPU; auto-releases on shell exit (EXIT/INT/TERM trap).
eval "$(scripts/gpu_lease.sh acquire 1)"
export EVAL_CHECKPOINT_GPU_LEASED=1   # tell eval_checkpoint.py not to re-lease
echo "[run_offline_eval] leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_one () {
  local name="$1" dir="$2"
  echo "[run_offline_eval] scoring $name from $dir"
  python scripts/eval_checkpoint.py \
    --run-dir "$dir" \
    --scoring-tensor "$HELDOUT" \
    --y-mode train \
    --keep-going \
    --out "$OUT_DIR/${name}_heldout_bpb.csv"
}

run_one emender "$EMENDER_DIR"
run_one gdn2    "$GDN2_DIR"

echo "[run_offline_eval] DONE"
