#!/usr/bin/env bash
# Two predeclared initialization conditions against the exact same frozen replay code.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN=${1:?new output root required}
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
SOURCE_CONTROL=/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-rl-replay-diagnostic-v2-control
read -r COMMIT < "$SOURCE_CONTROL/commit.txt"
[[ $COMMIT == dc444435* ]]
test ! -e "$RUN"
mkdir -m 700 -p "$RUN"
printf '{"source_commit":"%s","python_hash_seeds":[0,123],"workers_per_condition":2,"optimizer_updates":0,"automatic_expansion":false}\n' "$COMMIT" > "$RUN/execution-config.json"
chmod 400 "$RUN/execution-config.json"
cd "$SOURCE_CONTROL/worktree"
sha256sum --check --quiet "$SOURCE_CONTROL/source.sha256"
for SEED in 0 123; do
 printf 'HASHSEED_CONDITION=%s\n' "$SEED"
 PYTHONHASHSEED="$SEED" timeout --kill-after=30 1900 bash scripts/run_e97_actor_logprob_repeatability.sh "$RUN/seed-$SEED"
 sha256sum --check --quiet "$SOURCE_CONTROL/source.sha256"
done
cmp "$RUN/seed-0/recipe-private.json" "$RUN/seed-123/recipe-private.json"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
from scripts.eval_e97_native_execution import publish,sha
root=Path(sys.argv[1]);results=[];maps=[]
for seed in (0,123):
    path=root/f'seed-{seed}';recipe=json.loads((path/'recipe-private.json').read_text())
    by_id={(r['id'],r['turn']):r for r in recipe['selected']}
    report=json.loads((path/'rank-0-private.json').read_text())
    rows=[r for r in report['records'] if r['mode']=='full-generator-forced' and r['repetition']==0]
    current={(r['id'],r['turn']):[s['logp'] for s in r['steps']] for r in rows};maps.append(current)
    error=max(abs(a-b) for key,values in current.items() for a,b in zip(values,by_id[key]['recorded_logprobs']))
    results.append(dict(python_hash_seed=seed,recorded_actor_max_delta=error,parameter_sha256=report['parameter_before'],summary_sha256=sha(path/'summary.json')))
if set(maps[0])!=set(maps[1]) or len(maps[0])!=4:raise ValueError('condition coverage')
cross=max(abs(a-b) for key in maps[0] for a,b in zip(maps[0][key],maps[1][key]))
result=dict(status='diagnostic-measurements-complete',conditions=results,cross_seed_max_delta=cross,
            identical_replay_recipes=True,same_parameters_across_conditions=results[0]['parameter_sha256']==results[1]['parameter_sha256'],optimizer_updates=0,rl_optimizer_ready=False,
            original_probability_gate='failed, unchanged',automatic_expansion=False)
publish(root/'summary.json',result);print(json.dumps(result,sort_keys=True),flush=True)
PY
