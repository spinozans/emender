#!/usr/bin/env bash
# Launch the 3 promoted-config bounded pilots in parallel on GPUs 0,1,2.
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-1119
PD=experiments/e99_1p3b_cma/pilot_results
i=0
for cfg in "$PD"/cfg_0_*.json "$PD"/cfg_1_*.json "$PD"/cfg_2_*.json; do
  gpu=$i
  od="$PD/p$i"
  mkdir -p "$od"
  nohup setsid python experiments/e99_1p3b_cma/pilot.py \
    --configs_json "$cfg" --gpu "$gpu" --batch_size 2 \
    --pilot_steps 1800 --pilot_wall_minutes 45 --bpb_batches 40 \
    --outdir "$od" > "$od/pilot.log" 2>&1 < /dev/null &
  echo "launched pilot $i on GPU $gpu: cfg=$cfg outdir=$od pid=$!"
  i=$((i+1))
done
