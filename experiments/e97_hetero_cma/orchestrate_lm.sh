#!/usr/bin/env bash
# e97-lm-1p3b orchestrator: wall-matched arm -> derive N_H -> token-matched arm.
# Pinned to 4 idle GPUs (4,5,6,7) per the at-most-4 coordination policy.
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-1275
export PYTHONPATH=.
OUT=experiments/e97_hetero_cma/results/lm_verdict
mkdir -p "$OUT"
W=720                       # wall-clock-matched budget (seconds)
DRV=experiments/e97_hetero_cma/lm_verdict.py
G_A=4; G_B=5; G_C=6; G_D=7  # the 4 idle lanes

run() {  # arch proto seed gpu wall tokcap outtag
  local arch=$1 proto=$2 seed=$3 gpu=$4 wall=$5 cap=$6 tag=$7
  local caparg=""; [ "$cap" != "-" ] && caparg="--token_cap $cap"
  CUDA_VISIBLE_DEVICES=$gpu python "$DRV" --arch "$arch" --protocol "$proto" \
    --seed "$seed" --wall_seconds "$wall" $caparg \
    --out "$OUT/${tag}.json" > "$OUT/${tag}.log" 2>&1
}

echo "[$(date -u +%T)] WALL ARM (W=${W}s) on GPUs $G_A $G_B $G_C $G_D"
run H wall 0 $G_A $W - H_wall_s0 &
run H wall 1 $G_B $W - H_wall_s1 &
run G wall 0 $G_C $W - G_wall_s0 &
run G wall 1 $G_D $W - G_wall_s1 &
wait
echo "[$(date -u +%T)] wall H/G done; LSTM reference"
run L wall 0 $G_A $W - L_wall_s0 &
wait
echo "[$(date -u +%T)] wall arm complete"

# derive N_H = min tokens H reached across seeds (the matched-token budget)
NH=$(python - <<'PY'
import json, glob, os
OUT='experiments/e97_hetero_cma/results/lm_verdict'
toks=[]
for f in ('H_wall_s0','H_wall_s1'):
    p=os.path.join(OUT,f+'.json')
    if os.path.exists(p):
        d=json.load(open(p)); toks.append(int(d.get('tokens',0)))
print(min([t for t in toks if t>0]) if toks else 0)
PY
)
echo "[$(date -u +%T)] N_H (matched tokens) = $NH"

echo "[$(date -u +%T)] TOKEN ARM (cap=$NH) on GPUs $G_B $G_C $G_D"
run G token 0 $G_B $W $NH G_token_s0 &
run G token 1 $G_C $W $NH G_token_s1 &
run L token 0 $G_D $W $NH L_token_s0 &
wait
echo "[$(date -u +%T)] token arm complete. ALL DONE."
