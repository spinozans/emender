# Handoff: 1.3B CMAES Redos and GDN2-MLP Top-Up

Snapshot taken: 2026-06-05 09:38 UTC

## Current repository state

- Working repo: `/home/erikg/emender`
- Durable code lives here. The package path is `ndm/` inside this repo.
- Do not treat `/home/erikg/ndm` as the durable codebase for this work. It has separate worktrees from other agents and older side experiments.
- Branch: `wg/gdn2-mlp-cma-20260603`
- Pushed commit: `8220d1e feat: add GDN2-MLP CMA proxy`
- Git status before this handoff file was written: clean and tracking `origin/wg/gdn2-mlp-cma-20260603`.
- Raw experiment artifacts are intentionally local and ignored by git under `experiments/local/`.
- `.gitignore` has `experiments/local/`, so source/report files must be staged explicitly. Do not use `git add -A`.

## What was done

1. Took over the `emender` working tree after the old parameter accounting issue.
2. Rebased the GDN2-MLP work onto current `origin/main`.
3. Added local GDN2 support in the fair-comparison harness:
   - external GDN2 mixer integration
   - official-style GDN2 + SwiGLU MLP proxy called `gdn2-mlp`
   - exact parameter accounting for mixer-only `gdn2` and `gdn2-mlp`
   - `--gdn2_mlp_ratio` support in model construction
   - CMAES search space for `gdn2-mlp`
4. Validated the edited Python files with `py_compile`.
5. Ran `git diff --check HEAD`.
6. Committed and pushed the branch:
   - `8220d1e feat: add GDN2-MLP CMA proxy`
7. Set up and smoke-ran the official NVIDIA GDN-2 stack outside the repo:
   - external checkout: `/home/erikg/GatedDeltaNet-2`
   - local overlay/artifacts: `experiments/local/gdn2_official/`
   - checkpoint reached optimizer step 200
   - official run was stopped intentionally after the checkpoint to free GPUs
8. Ran the local 1.3B `gdn2-mlp` CMAES proxy.

## Important code files

- `ndm/models/external_gdn2.py`
- `ndm/models/ladder_lm.py`
- `scripts/calc_dim.py`
- `scripts/cmaes_search_v2.py`
- `train.py`
- `docs/experiments/gdn2_mlp_cma_20260603.md`

## Experiment artifact roots

Main CMAES redo root:

```bash
/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529
```

GDN2-MLP CMAES run:

```bash
/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/gdn2-mlp_20260602_201948
```

GDN2-MLP result summary:

```bash
/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/gdn2-mlp_20260602_201948/results.json
```

GDN2-MLP controller log:

```bash
/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/controller_20260602T201948Z_setsid.log
```

Official GDN-2 checkpoint:

```bash
/home/erikg/emender/experiments/local/gdn2_official/runs/outputs/tsz128x16k_20B_official_gdn2_1.3B_b128_long_20260602T190636Z/latest-model-ckpt.pth
```

There is also a small partial/failed early GDN2-MLP directory:

```bash
/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/gdn2-mlp_20260602_201842
```

Ignore that one for analysis. It only has partial eval directories.

## Current GPU/process status

As of 2026-06-05 09:38 UTC, `nvidia-smi` showed all eight GPUs idle:

```text
GPU 0: 2 MiB used, 0% util
GPU 1: 2 MiB used, 0% util
GPU 2: 2 MiB used, 0% util
GPU 3: 2 MiB used, 0% util
GPU 4: 2 MiB used, 0% util
GPU 5: 2 MiB used, 0% util
GPU 6: 2 MiB used, 0% util
GPU 7: 2 MiB used, 0% util
```

There were no active 1.3B CMAES controllers in `/home/erikg/emender`.

There are some old watcher shell processes from previous agents and one CPU-only expressivity process:

```bash
python experiments/expressivity_tasks/run_unified_learnability.py --max_gpus 0
```

Those do not occupy GPU memory. Do not kill unrelated processes unless Erik explicitly asks.

## Current CMAES leaderboard

Metric note: `loss` in these artifacts is the CMAES fitness, an average over the eval training trajectory. `final_loss` is the eval's last-window/final reported loss. Use `loss` for the CMAES ranking.

| Rank | Model | Evals | Status | Best avg loss | Final loss | Params M | Best eval |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `e97-raw` | 103 | complete | 5.9511 | 5.4738 | 1265.6 | 58 |
| 2 | `gdn2-mlp` | 64 | complete | 5.9613 | 5.5400 | 1285.2 | 57 |
| 3 | `e97` | 109 | complete | 5.9733 | 5.5386 | 1274.7 | 59 |
| 4 | `e88-linear` | 104 | recoverable | 5.9854 | 5.5263 | 1269.1 | 98 |
| 5 | `e88` | 92 | complete | 5.9900 | 5.5338 | 1289.1 | 54 |
| 6 | `e88-raw` | 106 | complete | 6.0390 | 6.0390 | 1288.0 | 55 |
| 7 | `e97-linear` | 109 | complete | 6.0516 | 6.0516 | 1315.8 | 95 |
| 8 | `fla-gdn` | 109 | recoverable | 6.0854 | 5.5951 | 1292.1 | 83 |
| 9 | `m2rnn` | 107 | complete | 6.1161 | 6.1161 | 1296.4 | 72 |
| 10 | `gdn2` | 96 | complete | 6.3850 | 5.8386 | 1262.6 | 13 |
| 11 | `transformer` | 128 | complete | 6.4606 | 6.4606 | 1318.8 | 104 |

Coverage target was strictly more than 96 completed evals per model. Under that rule:

- Met target: `e97-raw`, `e97`, `e88-linear`, `e88-raw`, `e97-linear`, `fla-gdn`, `m2rnn`, `transformer`
- Under target: `gdn2-mlp` at 64, `e88` at 92, old mixer-only `gdn2` at 96

The priority under-target run is `gdn2-mlp`. The old `gdn2` result is mixer-only and should not be used as the primary GDN-2 comparison.

## GDN2-MLP result details

The completed GDN2-MLP run:

- status: complete
- evals: 64
- generations: 8
- wall time: 9.30 hours
- best avg loss: 5.961312
- best final loss: 5.5400
- best actual params: 1,285,245,320

Best config:

```json
{
  "dim": 2304,
  "depth": 17,
  "expansion": 2,
  "n_heads": 8,
  "mlp_ratio": 2.854220752778522,
  "lr": 0.0003907570359771844,
  "batch_size": 2
}
```

Interpretation so far:

- `gdn2-mlp` is currently second overall, but it has only 64 evals.
- It is not yet fully comparable to the 100+ eval CMAES redos.
- The search is fairly flat around 5.96 to 5.98.
- Good configs tend to use `expansion=2`, dimensions around 2304 to 2816, shallow-to-mid depths, and small GDN2 head counts.

## What needs to happen next

Main next step:

1. Top up `gdn2-mlp` from 64 evals to at least 104 evals.

Reasoning:

- Popsize is 8.
- 64 evals = 8 generations.
- Target is `>96`.
- 4 more generations gives 96, which is not strictly over target.
- 5 more generations gives 104, which clears the target.

Secondary optional cleanup:

2. Top up `e88` from 92 to at least 104 evals.
3. Optionally top up old mixer-only `gdn2` from 96 to at least 104 evals for bookkeeping, but do not prioritize it over `gdn2-mlp`.
4. Rebuild the summary table and plots from the durable `results.json` and `.done` files.
5. If a paper/report figure is made, plot full CMAES trajectories, not only best final points.

## Recommended GDN2-MLP top-up procedure

Do not launch this until Erik gives a green light.

The cleanest top-up is to continue the existing CMAES state, not start a fresh timestamped run. The final CMAES state was preserved as:

```bash
/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/gdn2-mlp_20260602_201948/cmaes_state_final.pkl
```

The runner resumes from `cmaes_state.pkl`, so first restore the final state under that expected name:

```bash
cd /home/erikg/emender

RUN=/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/gdn2-mlp_20260602_201948
test -e "$RUN/cmaes_state.pkl" || cp "$RUN/cmaes_state_final.pkl" "$RUN/cmaes_state.pkl"
```

Then launch the top-up. This keeps eval IDs continuing from 64 and should run five additional generations when `--min_generations 13` is used:

```bash
cd /home/erikg/emender

RUN=/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/gdn2-mlp_20260602_201948
LOG=/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/topup_$(date -u +%Y%m%dT%H%M%SZ).log

GDN2_PATH=/home/erikg/GatedDeltaNet-2 \
setsid /home/erikg/emender/.venv/bin/python -u /home/erikg/emender/scripts/cmaes_search_v2.py \
  --model gdn2-mlp \
  --phase cmaes \
  --resume \
  --warm_start /home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/gdn2-mlp/warm_start_primary.json \
  --gpus 6,7 \
  --output "$RUN" \
  --params 1300M \
  --param_tolerance 0.03 \
  --train_minutes 15 \
  --popsize 8 \
  --sigma 0.5 \
  --min_generations 13 \
  --consecutive 3 \
  --chunk_size 2048 \
  --tokenizer p50k_base \
  --data /home/erikg/elman/data/pile.txt \
  > "$LOG" 2>&1 < /dev/null &
echo "$LOG"
```

GPU choice:

- Prior GDN2-MLP used GPUs `6,7`.
- As of the snapshot all GPUs were free, but using two GPUs keeps the top-up contained.
- If Erik assigns different GPUs, change only `--gpus`.

Expected time:

- Original 64 evals on 2 GPUs took 9.30 hours.
- Five more generations = 40 more evals.
- On 2 GPUs, expect roughly 5.5 to 6 hours plus overhead.
- On 4 GPUs, it should be closer to half that, but only use more GPUs if Erik wants that.

Operational caution:

- If `--resume` restores state correctly, the log should say something like:
  - `RESUME: recovered 64 completed evals`
  - `RESUME: restored CMA-ES state`
  - `continuing from gen 9`
- If it does not restore the state, stop and reassess before letting it run for hours. It may still recover `.done` files but reinitialize CMAES from the warm start, which is an exploration restart rather than a true continuation.

## How to rebuild the best table

Use this from `/home/erikg/emender`:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/local/cmaes_redo_1300m_20260529")
models = [
    "e97-raw", "gdn2-mlp", "e97", "e88-linear", "e88",
    "e88-raw", "e97-linear", "fla-gdn", "m2rnn", "gdn2", "transformer",
]

rows = []
for name in models:
    arch = root / name
    best = None
    eval_count = 0
    status = "missing"
    if arch.exists():
        for run in sorted(p for p in arch.iterdir() if p.is_dir()):
            evals = []
            res = run / "results.json"
            if res.exists():
                try:
                    evals += json.loads(res.read_text()).get("all_results", [])
                    status = "complete"
                except Exception:
                    pass
            for done in run.glob("eval_*/*.done"):
                try:
                    evals.append(json.loads(done.read_text()))
                except Exception:
                    pass

            seen = set()
            dedup = []
            for e in evals:
                k = e.get("eval_id")
                if k is None:
                    k = (json.dumps(e.get("params", {}), sort_keys=True), e.get("loss"))
                if k not in seen:
                    seen.add(k)
                    dedup.append(e)
            eval_count = max(eval_count, len(dedup))
            if not res.exists() and dedup and status == "missing":
                status = "recoverable"
            for e in dedup:
                if e.get("success", True) and isinstance(e.get("loss"), (int, float)):
                    if best is None or e["loss"] < best["loss"]:
                        best = e

    if best:
        rows.append((best["loss"], name, eval_count, status, best))

rows.sort()
for rank, (_, name, evals, status, best) in enumerate(rows, 1):
    params_m = best.get("actual_params", 0) / 1e6
    print(
        f"{rank:2d} {name:14s} evals={evals:3d} {status:11s} "
        f"loss={best['loss']:.4f} final={best.get('final_loss'):.4f} "
        f"params={params_m:.1f}M eval={best.get('eval_id')}"
    )
PY
```

## Safety rules

- Do not write large experiment artifacts to `/tmp`.
- Keep logs under `experiments/local/...`.
- Do not commit `experiments/local/...`; it is intentionally ignored.
- Do not kill or move unrelated `/home/erikg/ndm` worktree jobs without Erik's explicit instruction.
- Do not run `git add -A`.
- Stage specific files only.
- Before starting any long run, check:

```bash
cd /home/erikg/emender
git status --short --branch
nvidia-smi
ps -eo pid,ppid,stat,etime,cmd | rg -i 'cmaes_search|train.py|torchrun|accelerate|pretrain.py|gdn2|e88|e97|m2rnn|transformer' || true
```

## Bottom line

The core code is durable, committed, and pushed. The raw CMAES data is durable locally under `experiments/local/`, not `/tmp`. The main scientific hole is that `gdn2-mlp` looks very strong but only has 64 evals, while the comparison target is more than 96 evals. The immediate next experiment is therefore a controlled `gdn2-mlp` CMAES top-up to at least 104 evals, preferably by resuming the preserved CMAES state in the existing run directory.
