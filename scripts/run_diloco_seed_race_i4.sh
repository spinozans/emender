#!/usr/bin/env bash
# DiLoCo seed-race I=4 — Frontier dress-rehearsal (task: diloco-seed-race-2).
#
# Seed ALL 4 islands from the newest mature single-GPU emender checkpoint
# (full state incl. schedule-free optimizer z), then continue as a 4-island
# DiLoCo run with PLAIN averaging (outer_lr=1.0, outer_beta=0.0) racing ahead
# ~3B tokens. This is the "train locally, then scale out on Frontier" sim and
# extends the DiLoCo scaling law to I=4 from a mature seed (SWA-style regime).
#
# Launched DETACHED via scripts/launch_detached_run.sh so it survives the
# launching agent's exit; the wrapper leases exactly 4 IDLE GPUs through the
# broker (references on GPUs 0-1 are left untouched) and the lease is owned by
# the detached process (heartbeat keeper tracks torchrun's PID across exec).
#
# CONFOUND GUARDS (must all hold for any "degrades" reading to be admissible):
#   - seed loads FULL state incl SF z   (--resume -> load_checkpoint)
#   - PLAIN averaging beta=0            (outer_beta=0.9 DIVERGES, do NOT use)
#   - gate_activation=silu PASSED       (train.py default sigmoid would be a
#                                        silent-correctness trap on resume)
#   - FUSED only, no eager              (--use_triton 1; fused-guard asserts)
#   - matched TOTAL tokens              (seed-aware piecewise token formula in
#                                        analyze_seed_race_i4.py)
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REPO_ROOT="$(pwd)"

# --- frozen geometry (byte-identical to the reference run's args.json) --------
SEED_CKPT="${SEED_CKPT:-/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750/checkpoint_step_150500_loss_3.0442.pt}"
LOGDIR="${LOGDIR:-/mnt/nvme1n1/erikg/diloco_sweep/seed_race_i4}"
DATA="${DATA:-/home/erikg/elman/data/pile.txt}"

[ -f "$SEED_CKPT" ] || { echo "ERROR: seed checkpoint not found: $SEED_CKPT" >&2; exit 1; }
mkdir -p "$LOGDIR"

# steps=240000 resumes from start_step=150500 -> 89500 DiLoCo steps.
#   per-step tokens (I=4) = 4 islands * bs4 * chunk2048 = 32768
#   additional tokens     = 89500 * 32768 = 2.93B  (~3B race)
# save_every=3000 = 12 * diloco_k(250) -> every periodic save lands on a merge
# step, so the saved rank-0 checkpoint is the post-merge cross-island CONSENSUS.
#
# Why 3000 (not the ~0.5B-token cadence in the loose spec): the single-GPU
# reference run STOPS at step 244141 = 2.0B total tokens, while the seed is at
# 1.233B. The matched-token head-to-head therefore lives entirely in the OVERLAP
# window [1.233B, 2.0B]. save_every=3000 lands 7 consensus checkpoints in that
# window (steps 153000..171000) so the comparison is an effect size over many
# points, not a single noisy point (which the task explicitly requires).
# keep_checkpoints=30 retains the whole curve (7 overlap + 23 beyond) to the end
# (~30 * 7.72 GB = 231 GB on a 2.4 TB-free volume) so nothing is pruned mid-race.

exec scripts/launch_detached_run.sh \
  --name diloco_seed_race_i4 \
  --gpus 4 \
  --logdir "$LOGDIR" \
  -- \
  env NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0 \
  torchrun --nproc_per_node=4 --master_port=29561 train.py \
    --level E97 --params 100m \
    --embed_dim 1024 --dim 1792 --depth 11 --n_heads 216 --n_state 32 \
    --expansion 1.0 --state_expansion 2 --n_groups 32 --n_slots 64 \
    --mlp_ratio 2.2623 --mlp_multiple 64 \
    --use_gate 1 --use_permutation 1 --gate_activation silu \
    --gdn_allow_neg_eigval 1 --mamba_expand 2 \
    --use_triton 1 --bf16 \
    --optimizer schedulefree --lr 0.001007 --warmup_steps 0 \
    --batch_size 4 --chunk_size 2048 \
    --data "$DATA" --tokenizer p50k_base \
    --resume "$SEED_CKPT" \
    --diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 \
    --steps 240000 --save_every 3000 --keep_checkpoints 30 \
    --seed 42 \
    --output "$LOGDIR"
