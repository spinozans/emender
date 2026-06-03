# S5-SYMMETRIC CMA-ES — LAUNCH COMMANDS (exact, verbatim-executable)

**Status:** PRE-FLIGHT COMPLETE (no GPU used, no training run). This file is the
hand-off to the search task `s5sym-search`, which executes the four commands in
§3 **verbatim**.

**Protocol:** `paper/review/S5_SYMMETRIC_PROTOCOL.md` §D. Every architecture is
CMA-ES'd at 8M **from its existing seed** on the **same** state-tracking
objective (S5 accuracy at the trained length T=128), to the **same** capped
budget, with identical eval — removing the single BL-1 selection-criterion
asymmetry. Report whichever way it falls (§D.7).

**Driver:** `scripts/cmaes_search_s5.py` — a REPO COPY of
`scripts/cmaes_search_v2.py`. The `--objective lm_loss` path (the §5 LM-loss
search) is **byte-for-byte unchanged**; the new `--objective s5_acc@T128` path
is purely additive. `scripts/cmaes_search_v2.py` is **not** modified.

---

## 1. What the S5 objective does (real harness, no mock)

Under `--objective s5_acc@T128`, every CMA candidate:

1. Runs the **real** state-tracking harness
   `experiments/expressivity_tasks/train_hybrid.py --task s5_permutation`
   (seq_len 128, batch 32, schedule-free AdamW) for the capped budget of
   **5000 steps** with the candidate's `dim/depth/n_heads/n_state/lr` (and, for
   **E88 only**, the searched `linear_state`/`use_gate`).
2. Reads the `eval_acc` series that `train_hybrid` already writes to its
   `<label>.json` (no new eval code, no synthetic fitness).
3. Fitness = **`1 − mean_S5_acc@T128`**, where `mean_S5_acc@T128` is the mean
   `eval_acc` over the **final 1000 training steps** of the capped run.
   NaN / divergence / missing JSON → fitness **1.0** (worst). CMA-ES minimizes.

S3 is **not** in the loop (held-out control, §D.1). The
`{128,256,512,1024}` length-extrapolation grid is recorded per candidate
(`length_extrap` in each JSON) but is **never** the selection target.

### Parameter re-match note (important deviation, by necessity)

The protocol names `estimate_params_for_config` / `PARAM_TOLERANCE=0.10` for the
~8M re-match. That analytic estimator is calibrated for the §5 **LM** models at
large scale and **mispredicts the 8M hybrid GDN layer** (it returns 6.7M at
expansion=1 / 13.3M at expansion=2 where the real model is **8.33M**). So the S5
path gates candidates on the **real constructed parameter count** — it builds
the exact `HybridLadderLM` that `train_hybrid` will train (on CPU, no GPU) and
counts its parameters, holding every arm at **8M ±10%**. Measured seed counts:

| arm | seed (dim/depth/H/N) | real params | analytic est (rejected) |
|---|---|---|---|
| E88 (`e88`) | 384 / 4 / 32 / 32 | **7.96M** | 8.01M |
| M2RNN (`m2rnn`) | 384 / 4 / 32 / 32 | **8.10M** | 8.15M |
| FLA-GDN (`fla-gdn`) | 640 / 4 / 32 / 32 | **8.33M** | 13.28M ✗ |
| M2RNN-paper (`m2rnn-paper`) | 608 / 4 / 32 / 32 | **8.09M** | 8.18M |

Both E88 gate settings stay reachable at ~8M via dim re-match (`use_gate=1` at
dim 352–416; `use_gate=0` at dim 448–528), so `use_gate`/`linear_state` are
genuinely searched, not pinned.

---

## 2. Seeds and search space (protocol §D.2)

**Seeds** (CMA warm starts) are the 8M `MODEL_CONFIG` centers, in
`experiments/expressivity_tasks/results/s5_symmetric_20260603/seeds_s5_symmetric.json`,
loaded by `load_anchor_configs`. `--anchor_only_cmaes` makes each arm's **only**
warm start its own seed, so CMA-ES starts exactly at the existing center.

**Varied (all arms, selected on S5@T128):** `dim` (int, mult-16, re-matched to
~8M), `depth` 3–6, `n_heads` 8–64, `n_state` ∈ {16,32}, `lr` (log 1e-4..3e-3).
**Varied (E88 ONLY):** `linear_state` ∈ {0,1} and `use_gate` ∈ {0,1} — the BL-1
component, now searched on S5 symmetrically.
**Fixed (all arms):** update-rule family, optimizer (schedule-free AdamW),
train length T=128, eval grid {128,256,512,1024}, ~8M param target,
per-candidate budget, population, sigma, generation cap, search seed.

## 3. THE LAUNCH COMMANDS (run all four; verbatim)

Budget (§D.3), **identical for every arm**: per-candidate cap 5000 steps /
seq128 / batch32; population 16; sigma 0.35; converge when best-fitness
improvement < 0.005 (accuracy units) over 3 consecutive generations **OR** hard
cap **12 generations**; search seed **42**. All launches pinned to GPUs
**2,3,4,5** (never 0,1).

```bash
# Run from the repo root. Each arm writes a timestamped subdir under
# experiments/expressivity_tasks/results/s5_symmetric_20260603/.
RESULTS=experiments/expressivity_tasks/results/s5_symmetric_20260603
SEEDS=$RESULTS/seeds_s5_symmetric.json

# --- ARM 1: Emender / E88 (also searches linear_state & use_gate) ---
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/cmaes_search_s5.py \
  --model e88 --objective s5_acc@T128 --phase cmaes \
  --anchor_configs $SEEDS --anchor_only_cmaes \
  --params 8M --param_tolerance 0.10 \
  --popsize 16 --sigma 0.35 \
  --min_generations 3 --converge 0.005 --consecutive 3 --max_generations 12 \
  --gpus 2,3,4,5 --output $RESULTS

# --- ARM 2: M2RNN-CMA ---
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/cmaes_search_s5.py \
  --model m2rnn --objective s5_acc@T128 --phase cmaes \
  --anchor_configs $SEEDS --anchor_only_cmaes \
  --params 8M --param_tolerance 0.10 \
  --popsize 16 --sigma 0.35 \
  --min_generations 3 --converge 0.005 --consecutive 3 --max_generations 12 \
  --gpus 2,3,4,5 --output $RESULTS

# --- ARM 3: FLA / Gated DeltaNet ---
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/cmaes_search_s5.py \
  --model fla-gdn --objective s5_acc@T128 --phase cmaes \
  --anchor_configs $SEEDS --anchor_only_cmaes \
  --params 8M --param_tolerance 0.10 \
  --popsize 16 --sigma 0.35 \
  --min_generations 3 --converge 0.005 --consecutive 3 --max_generations 12 \
  --gpus 2,3,4,5 --output $RESULTS

# --- ARM 4: M2RNN-paper ---
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/cmaes_search_s5.py \
  --model m2rnn-paper --objective s5_acc@T128 --phase cmaes \
  --anchor_configs $SEEDS --anchor_only_cmaes \
  --params 8M --param_tolerance 0.10 \
  --popsize 16 --sigma 0.35 \
  --min_generations 3 --converge 0.005 --consecutive 3 --max_generations 12 \
  --gpus 2,3,4,5 --output $RESULTS
```

Notes:
- `--min_generations 3` lets the "3 consecutive generations < 0.005" rule be the
  binding early-stop; the hard cap is `--max_generations 12` (§D.3). The first
  arm to satisfy either stops; no arm gets more search than another.
- `--train_minutes` is intentionally omitted — the S5 path uses the fixed
  5000-step `train_hybrid` budget, not a wall-clock budget.
- GPU pinning is doubly enforced: `CUDA_VISIBLE_DEVICES=2,3,4,5` on the launch
  **and** `--gpus 2,3,4,5`; each training worker is pinned to a single GPU in
  {2,3,4,5} by `prepare_worker_env`. GPUs 0,1 are never used.

## 4. Records committed (closes the A.2 NOT-FOUND gap)

Each arm's run dir `…/s5_symmetric_20260603/<model>_<timestamp>/` accumulates:
- `eval_<id>/params.json` — the candidate config + objective.
- `eval_<id>/s5cand_<model>_eval<id>.json` — the **real** `train_hybrid` output
  (per-step `eval_acc`/`eval_loss`, `final_acc`, `length_extrap`).
- `eval_<id>/cmd.txt`, `stdout.txt`, `stderr.txt`, `.done` (fitness summary).
- `generations.jsonl` — per-generation snapshot (fitnesses, best-so-far, sigma,
  mean) for post-hoc convergence diagnostics.
- `results.json` — final ranked results incl. `s5_acc_at_T128` per candidate.

Commit the whole `s5_symmetric_20260603/` tree after the searches finish.

## 5. After the search — final-winner eval (protocol §D.5, follow-up, not here)

For each arm's winning config, run the full battery exactly as the current probe
(`run_separation_suite.py`): 3 seeds {42,123,456}; S5 train T=128 / 20000 steps,
eval T∈{128,256,512,1024}; **S3 held-out control** (train T=128 / 10000 steps,
same eval grid); report both the original defaults-based numbers and the
symmetric numbers side by side (§D.7). The within-class magnitude hedges at
`main.typ:1248-1249, 1262-1265` and `tab_s5` (`main.typ:1284-1288`) are updated
with the symmetric result **whichever way it falls** — `main.typ` is NOT touched
by this pre-flight or the search task.

## 6. Pre-flight verification (this task)

`python scripts/s5_symmetric_preflight_check.py` (CPU only, no training) asserts:
LM path unchanged; each arm's anchor == MODEL_CONFIG center; every seed real-8M
±10%; per-arm search space matches §D.2 (E88 searches linear_state & use_gate);
the param gate genuinely filters off-target candidates; the fitness adapter runs
the real `train_hybrid s5_permutation` harness and computes
`1 − mean(real eval_acc over final 1000 steps)` (verified against a committed
real `train_hybrid` JSON); GPU workers pin to {2,3,4,5}. All checks pass.
