#!/usr/bin/env bash
# Detached orchestrator for the DiLoCo seed-race I=6 cell (task diloco-seed-race).
#
# Companion to orchestrate_seed_i4.sh (finalize-diloco-scaling). Given the PID of
# the already-launched 6-replica training run, it:
#   1. waits for that run to finish (the ~3B-token race, ~16h)
#   2. relaunches it once if it died WITHOUT a step-190000 checkpoint
#   3. scores the I=6 consensus checkpoints on the clean disjoint tensor (y-mode)
#   4. re-scores the single-GPU reference run-dir to EXTEND reference_curve.csv
#      with whatever newer checkpoints (129000, 150500, ...) appeared during the
#      race -> maximizes the matched-token overlap (dedup: already-scored skipped)
#   5. runs analyze_degradation.py + plot_seed_race.py
#   6. touches DONE
#
# GPU discipline: training holds GPUs 2-7 while it runs, so scoring (which leases
# exactly ONE GPU from {2,3,4,5,6,7}) only proceeds AFTER training exits and those
# GPUs free. References 0-1 are NEVER touched. Relaunch (if needed) re-leases 6 via
# launch_detached_run.sh, owned by the detached child (survives this orchestrator).
set -uo pipefail

REPO=/home/erikg/ndm/.wg-worktrees/agent-1944
SWEEP=/mnt/nvme1n1/erikg/diloco_sweep
HERE=$REPO/experiments/diloco_scaling_law
LOGDIR=$SWEEP/seed_race_i6
LOG=$LOGDIR/orchestrate.log
DONE=$LOGDIR/orchestrate.DONE
SEED=/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750/checkpoint_step_129000_loss_3.1436.pt
TRAIN_PID="${1:?need training run pid}"
export GPU_LEASE_VISIBLE=2,3,4,5,6,7

exec >>"$LOG" 2>&1
echo "==================================================================="
echo "[orch $(date -u +%FT%TZ)] start; training pid=$TRAIN_PID"
cd "$REPO"

TRAINARGS="--data /home/erikg/elman/data/pile.txt --tokenizer p50k_base --level E97 --dim 1792 --n_heads 216 --n_state 32 --depth 11 --expansion 1.0 --use_gate 1 --gate_activation silu --mlp_ratio 2.2623 --mlp_multiple 64 --use_triton 1 --optimizer schedulefree --lr 0.001007 --bf16 --batch_size 4 --chunk_size 2048 --resume $SEED --steps 190000 --seed 42 --save_every 2500 --keep_checkpoints 30 --log_every 25 --val_every 999999999 --diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0"

wait_pid() { local pid="$1"; while kill -0 "$pid" 2>/dev/null; do sleep 60; done; }
have_final() { ls "$LOGDIR/train"/*/checkpoint_step_190000_*.pt >/dev/null 2>&1; }

launch_train() {  # echoes ONLY the run pid to stdout
  echo "[orch $(date -u +%T)] (re)launching i6 training" >&2
  scripts/launch_detached_run.sh --name diloco_seed_race_i6 --gpus 6 --logdir "$LOGDIR" -- \
    env NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --standalone --nproc_per_node=6 train.py $TRAINARGS --output "$LOGDIR/train" >&2
  sleep 5; cat "$LOGDIR/run.pid"
}

echo "[orch $(date -u +%T)] waiting for training pid=$TRAIN_PID ..."
wait_pid "$TRAIN_PID"
echo "[orch $(date -u +%T)] training pid gone"

if ! have_final; then
  echo "[orch $(date -u +%T)] no step-190000 ckpt -> died early; relaunch once"
  npid=$(launch_train); echo "[orch] relaunched training pid=$npid"
  wait_pid "$npid"; echo "[orch $(date -u +%T)] relaunch pid gone"
fi

echo "[orch $(date -u +%T)] scoring I=6 consensus checkpoints"
bash "$HERE/autoscore.sh" swell_i6
echo "[orch $(date -u +%T)] extending reference curve (newer ref checkpoints)"
bash "$HERE/autoscore.sh" reference
echo "[orch $(date -u +%T)] degradation analysis + overlay plot"
python3 "$HERE/analyze_degradation.py" || echo "[orch] analyze rc=$?"
python3 "$HERE/plot_seed_race.py" || echo "[orch] plot rc=$?"

# Commit + push the result artifacts so the deliverables land even if the chat
# agent's session has ended. This worktree (agent-1944) is owned by this task's
# branch alone, so a surgical add of ONLY the result files is safe; retry around
# any transient index lock. (NEVER `git add -A`.)
echo "[orch $(date -u +%T)] committing result artifacts"
cd "$REPO"
RESULTS="experiments/diloco_scaling_law/swell_i6_curve.csv \
         experiments/diloco_scaling_law/reference_curve.csv \
         experiments/diloco_scaling_law/degradation_summary.json \
         experiments/diloco_scaling_law/seed_race_i6_stats.json \
         experiments/diloco_scaling_law/seed_race_i6_plot.png \
         experiments/diloco_scaling_law/VERDICT_seed_race_i6.md"
for try in 1 2 3 4 5 6; do
  if git add $RESULTS 2>/dev/null; then
    if git commit -q -m "results: diloco seed-race I=6 scored + plotted + verdict (diloco-seed-race)"; then
      git push 2>&1 | tail -3
      echo "[orch] committed results $(git rev-parse --short HEAD)"
      break
    fi
    echo "[orch] nothing to commit (or commit failed) try $try"
  fi
  echo "[orch] git busy, retry $try"; sleep 10
done

echo "[orch $(date -u +%FT%TZ)] ALL DONE"
date -u +%FT%TZ > "$DONE"
