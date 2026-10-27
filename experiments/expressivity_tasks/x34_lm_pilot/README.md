# X34 LM pilot

Read RECEIPTS.md before interpreting evidence. `runs/` contains local data,
checkpoints and append-only curves and is excluded from Git. All commands run
from the repository root, with the requested interpreter only.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x34_lm_pilot.cli prepare
PYTHONDONTWRITEBYTECODE=1 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x34_lm_pilot.cli counts
experiments/expressivity_tasks/x34_lm_pilot/run_leased.sh parity
experiments/expressivity_tasks/x34_lm_pilot/run_leased.sh integration --arm tape_lm
experiments/expressivity_tasks/x34_lm_pilot/run_leased.sh train --arm base_lm --tag pilot --steps 512
experiments/expressivity_tasks/x34_lm_pilot/run_leased.sh train --arm tape_lm --tag pilot --steps 512
experiments/expressivity_tasks/x34_lm_pilot/run_leased.sh probe --arm base_lm --tag pilot --probe-smoke
experiments/expressivity_tasks/x34_lm_pilot/run_leased.sh probe --arm tape_lm --tag pilot --probe-smoke
```

The broker allowlist is 0,1,5,6; cards 2/3/4 and 7 are excluded. Training runs
sequentially. Repeating a command resumes `latest.pt`, including optimizer, RNG,
data position and cumulative wall time. Changing corpus/tokenizer is rejected.
For a later full run, use the SAME tag and `--steps 24415`; this resumes the
smoke instead of restarting. Commit pre-registration before doing this. Omit
`--probe-smoke` for the complete registered evaluation (including 4096 context).
Full-run conclusions await both arms and measured compute review.

The default 12-hour cumulative cap includes validation and saving. Each update
is indivisible, so an update may overshoot; the receipt records actual elapsed
time. At regular milestones checkpointing bounds loss of work after SIGKILL to
63 updates; graceful Python exceptions save the latest state and a failure receipt.
The saved wall clock cannot include downtime or work lost to SIGKILL: externally
account for any lost attempt before resuming a strictly capped scientific run.

`receipts/*.json` are small reviewable evidence, not checkpoints. `parameters.json`
counts frozen parameters too; trainable counts are also given. `parity.json`
runs X31's unchanged oracle/gradient assertions at X34 dimensions and budget.
Integration verifies a ReplayBackward node and finite nonzero value-projection
gradients in the LM loss. Unit tests check causality, shared initialization,
parameter counts, data ordering, and probe normalization. The depth probe has
only two paired corruptions per depth; its accuracy is descriptive, with no
independence or statistical significance claim.

`run_part1.sh` performs both smoke arms sequentially (baseline first resumes at
64 steps as a checkpoint check). `run_full.sh` continues those exact checkpoints
after verifying registration is committed. For detached process management:

```bash
setsid nohup experiments/expressivity_tasks/x34_lm_pilot/run_full.sh > experiments/expressivity_tasks/x34_lm_pilot/runs/full_console.txt 2>&1 < /dev/null &
echo $! > experiments/expressivity_tasks/x34_lm_pilot/runs/full.pid
```

Inspect that PID and append-only `runs/pilot_*/curve.jsonl`; do not launch a
second runner against the same checkpoint. Immutable `receipts/smoke_*.json`
preserve part-1 evidence while `pilot_*.json` is updated by later full runs.

Optional plotting dependency (installed only under ignored `runs/`, leaving the
shared virtual environment unchanged):

```bash
UV_CACHE_DIR=experiments/expressivity_tasks/x34_lm_pilot/runs/uv_cache uv pip install --python /home/erikg/emender/.venv/bin/python --target experiments/expressivity_tasks/x34_lm_pilot/runs/plot_deps -r experiments/expressivity_tasks/x34_lm_pilot/plot_requirements.txt
PYTHONDONTWRITEBYTECODE=1 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x34_lm_pilot.summarize
```
