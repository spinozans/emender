#!/usr/bin/env bash
# e97-wallclock-cma stage-2 orchestrator: after the wall sweep frees the GPUs,
# run (1) the longer-budget head-to-head on the wall-clock-best tanh config
# (C=1 per-step: most expressive AND -- since fused throughput is flat in C --
# the fastest tanh realization too), then (2) the capability guard (C-vs-expr).
set -u
D=/home/erikg/ndm/.wg-worktrees/agent-1238/experiments/e97_wallclock_cma
cd "$D" || exit 1
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

# 1) wait for the wall sweep to write its result file (all GPUs free again)
log "stage2: waiting for wall sweep to finish (bpb_sweep_wall.json)..."
for i in $(seq 1 240); do
  if [ -f results/bpb_sweep_wall.json ]; then log "wall sweep done."; break; fi
  sleep 15
done

# small settle for GPU memory release
sleep 20

# 2) longer-budget head-to-head, C=1, 2 seeds, 1100s wall (longer than 5M screen ~600s)
log "stage2: launching head-to-head C=1 (2 seeds, 1100s wall)"
python wc_headtohead.py --C 1 --ratio 0.5 --gpus 0,1,2,3,4,5,6,7 \
  --wall_seconds 1100 --seeds 0,1 --outer_timeout_s 1500 \
  > results/headtohead_C1.log 2>&1
log "stage2: head-to-head done."
sleep 15

# 3) capability guard (C in {1,64,2048}, tiny 4M probes -- fast)
log "stage2: launching capability guard (C=1,64,2048)"
python capability_guard.py --gpus 0,1,2,3,4,5,6,7 --seed 0 --steps 2000 --Cs 1,64,2048 \
  > results/capability_guard.log 2>&1
log "stage2: capability guard done. ALL STAGE-2 COMPLETE."
