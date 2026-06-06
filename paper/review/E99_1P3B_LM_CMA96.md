# E99 1.3B-class LM-CMA top-up to 96 short-run candidates (run-e99-1-3b)

**Date:** 2026-06-06 · **Candidate:** E99 typed Emender `typed-gdn2-lm` (native
GDN-2 heads + nonlinear specialist mix, the typed-gdn-2-head 5:1 ratio preserved
at scale) · **Status:** COMPLETE — 96/96 candidate-evaluations, 0 failures.

This is the bounded SHORT-RUN LM-CMA top-up requested as *"CMA-ES topped up to 96
with LM training at 1.3B"*. 96 = total candidate-**evaluations** (each a short 15-min
1.3B-class training), **NOT** 96 full multi-day runs. All training is REAL (real
Pile tokens, real fwd/bwd, schedule-free AdamW, the production `train.py` +
`ndm/models/ladder_lm.py` path). No checkpoint was published; `paper/main.typ`
untouched; nothing pushed; idle-GPU-only.

---

## 1. Headline result

| | Value |
|---|---|
| Candidate-evals completed | **96 / 96** (popsize 8 × 12 generations) |
| Failures / NaN / OOM | **0** (every generation `inf_frac = 0.0`) |
| Best **AvgLoss** (CMA fitness) | **7.5075** (eval 65) |
| Best config | `dim=3328, n_heads=102, n_state=32, depth=17, lr=1.358e-3, batch_size=2` (1.166 B params) |
| End-of-window loss of best (instantaneous, step 580) | ~6.65 nats/tok |
| CMA convergence | AvgLoss 8.07 → 7.51 over 12 gens; σ 0.127 → 0.066 |
| Search compute used | **1430 / 1440** train-GPU-minutes (23.8 / 24.0 GPU-h) — under ceiling |
| Per-eval median | 14.88 min, 2453 tok/s mean (proj. 2422 → +1.3%) |
| Promoted pilot (top-3, 3× tokens) | round-trip **PASS** all 3 (Δ ≤ 4.3e-6); rank **flips** (short-run winner → last); held-out BPB best = **eval 87** (2.090) |
| Token-matched cross-walk | pilot Final **5.65–5.79** ≈ handoff bf16 pack (5.55–5.64) — raw 7.51 gap was the **fp32 token deficit**, not architecture |

The CMA moved the operating point off the warm-start anchor (`3072/22/96/32`,
1.278 B) to a **wider + shallower + slightly smaller** region (`3328/17/~102/32`,
~1.16 B) with **higher LR** (~1.3–1.5e-3 vs anchor 9.95e-4), improving AvgLoss by
~0.56 nats over the 15-min window. n_state stayed at **32** for every top config.

---

## 2. Budget caps (recorded BEFORE launch) and actual usage

From `experiments/e99_1p3b_sanity/budget_caps.json` (emitted by the upstream
wire-e99-e98 task) and this run's driver defaults:

| Cap | Value (recorded up front) | Actual |
|---|---|---|
| Per-candidate train budget | 15 train-minutes (handoff wallclock-match) | median 14.88 min |
| Per-candidate token cap | ≈ 2.18 M tok @ 2422 tok/s | ~2.30 M tok @ ~2700 tok/s |
| Per-eval walltime ceiling (hard kill) | 15·60 + 300 s = 20 min | none hit |
| Aggregate training ceiling | **1440 GPU-min = 24.0 GPU-h** (= 96 × 15) | **1430 GPU-min** |
| Eval-count cap | **96** | 96 (`completed_all_generations`) |
| Projection gate | stop if median eval > 1.6 × 15 min | not triggered (14.88 ≈ 15) |
| Instability gate | stop if > 50 % of a generation inf/NaN | not triggered (0 %) |

`stop_reason.json` → `{"reason": "completed_all_generations", "evals_completed": 96,
"generations": 12}`. **No hard-stop fired** because the run stayed on-budget and
stable; the guards were live throughout (see `experiments/e99_1p3b_cma/run_e99_cma.py`).

---

## 3. Method

- **Candidate / architecture identity (FIXED).** `--level typed-gdn2-lm`, fp32, with
  the discovered typed-head logits `[3.9995,-1.9008,-0.9211,-2.8866,2.4146]`
  (`gdn2_recall, e97_track, count, latch, nonlin`), `gdn_allow_neg_eigval=1`,
  `lam_max=1.585`, `beta_max=2.747`, `knob_lr_mult=1.0`. `softmax` of those logits
  → ~82 % GDN-2 / ~17 % nonlinear, i.e. the **40:8 / 5:1 GDN-2:nonlinear ratio**
  from typed-gdn-2-head, preserved at every `n_heads` by largest-remainder
  allocation (at `n_heads=102` → 84 GDN-2 / 17 nonlinear / 1 count). These knobs
  were **not searched** — only width/depth/head-shape/LR moved, so the E99 typed
  identity is held fixed exactly as required.
- **Search space (5-D).** `dim ∈ [2048,3584]` (mult-128), `n_heads ∈ [48,160]`,
  `n_state ∈ {16,32}`, `depth ∈ [16,30]`, `lr ∈ [3e-4,1.5e-3]` (log). `batch_size`
  **fixed = 2** (see §7 conversion notes). Param-tolerance ±10 % of 1.270 B.
- **Fitness.** Primary CMA objective = **AvgLoss** = mean over the train window
  (the handoff convention, `parse_average_loss`); **Final** (last-100-avg,
  `FINAL_LOSS_LAST100`) recorded alongside so columns line up with the prior
  table. Held-out BPB is computed for the **promoted** configs in the bounded
  pilot (§6), keeping the 96-eval fitness LM-centered and handoff-comparable
  rather than per-candidate-BPB (budget). No synthetic-capability term is in the
  objective.
- **Schedule.** popsize 8, warm-started from the typed-gdn-2-head anchor
  (`dim=3072, depth=22, n_heads=96, n_state=32, lr=9.95e-4`), σ₀=0.14, run to
  exactly **12 generations = 96 evals** (no early-converge). LHS was not used —
  pure CMA from the discovered anchor, matching the handoff `--phase cmaes` shape.
- **Harness reuse.** The driver `experiments/e99_1p3b_cma/run_e99_cma.py` IMPORTS
  the production CMA harness `scripts/cmaes_search_v2.py` and registers the new
  `typed-gdn2-lm` model_type **without editing the shared file** (other tasks use
  it). All `.done`/`cmaes_state.pkl` crash-resume, GPU-pool parallelism,
  AvgLoss/Final parsing, output-dir scheme, and top-3 checkpoint retention are the
  harness's, verbatim. Param counts come from instantiating one
  `TypedHeadMixtureLayer` (matches the measured 1.2775 B anchor to 1.4e-5).
- **Data / tokenizer / ctx.** `/home/erikg/elman/data/pile.txt`, `p50k_base`
  (vocab 50281), `chunk_size=2048` — identical to the handoff.

---

## 4. Counting: how "96" maps to the prior CMA-ES counting

`96 = popsize 8 × 12 generations` candidate-**evaluations**. The prior batch
(`docs/HANDOFF_E97_GDN2_CMAES_20260528.md`) ran each target at `--popsize 8
--min_generations 8` (8 × 8 = 64). This top-up **continues the same popsize-8
counting** to 12 generations (8 × 12 = 96). Because `typed-gdn2-lm` is a NEW
candidate that did **not** exist in the prior batch, there was no prior CMA state
to resume on the production path, so per the recorded budget plan this is a
**documented fresh popsize-8 restart run to 12 generations** (not a continuation
of an existing pickle). Confirmed candidate-evals, not full runs.

---

## 5. Results

### 5.1 CMA convergence (per generation)

| Gen | gen-best AvgLoss | overall-best | evals | cum train-GPU-min | σ |
|----:|---:|---:|---:|---:|---:|
| 1 | 8.0706 | 8.0706 | 8 | 119 | 0.127 |
| 2 | 7.8695 | 7.8695 | 16 | 238 | 0.111 |
| 3 | 7.6680 | 7.6680 | 24 | 357 | 0.104 |
| 4 | 7.7800 | 7.6680 | 32 | 476 | 0.093 |
| 5 | 7.5751 | 7.5751 | 40 | 595 | 0.097 |
| 6 | 7.6261 | 7.5751 | 48 | 715 | 0.095 |
| 7 | 7.6722 | 7.5751 | 56 | 834 | 0.080 |
| 8 | 7.6168 | 7.5751 | 64 | 953 | 0.078 |
| 9 | **7.5075** | **7.5075** | 72 | 1072 | 0.075 |
| 10 | 7.6346 | 7.5075 | 80 | 1191 | 0.065 |
| 11 | 7.5453 | 7.5075 | 88 | 1310 | 0.060 |
| 12 | 7.5630 | 7.5075 | 96 | 1430 | 0.066 |

### 5.2 Top-10 candidates (extends the handoff summary columns)

Columns: `dim, n_heads, n_state, depth, lr, batch_size` (handoff fields) | **AvgLoss**
(handoff "Best avg loss") | **Final** (handoff "Best final loss") | + LM-path extras.

| eval | dim | n_heads | n_state | depth | lr | bs | params_B | **AvgLoss** | **Final** | last-step | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65 | 3328 | 102 | 32 | 17 | 1.358e-3 | 2 | 1.166 | **7.5075** | 7.5075 | 6.65 | 2780 |
| 87 | 3328 | 102 | 32 | 17 | 1.471e-3 | 2 | 1.166 | 7.5453 | 7.5453 | 6.72 | 2686 |
| 95 | 3328 | 101 | 32 | 17 | 1.497e-3 | 2 | 1.157 | 7.5630 | 7.5630 | 6.69 | 2673 |
| 33 | 3328 | 104 | 32 | 17 | 1.286e-3 | 2 | 1.188 | 7.5751 | 7.5751 | 6.73 | 2661 |
| 64 | 3328 | 96 | 32 | 18 | 1.474e-3 | 2 | 1.163 | 7.5870 | 7.5870 | 6.49 | 2547 |
| 83 | 3328 | 100 | 32 | 17 | 1.402e-3 | 2 | 1.148 | 7.5983 | 7.5983 | 6.62 | 2730 |

Full machine-readable records: `candidates.csv`, `candidates.json`,
`results.json`, `generations.jsonl` in the run dir.

### 5.3 Distribution over the 96 evals

- AvgLoss: best 7.5075 · mean 7.8342 · worst 8.4496 (no diverged/inf runs).
- tok/s: 2138 – 2780 (mean 2453); matches the 2422 projection (+1.3 %).
- params: 1.144 – 1.390 B (all within ±10 % of 1.270 B).
- The optimum sits at the **lower-param / higher-throughput** edge: under a
  wallclock-matched budget, faster configs see more tokens in 15 min, so AvgLoss
  implicitly rewards throughput — the **same property as the handoff convention**
  (15 train-minutes, AvgLoss fitness). Noted for honesty; not a defect.

---

## 6. Promoted configs → bounded longer pilot (rank stability, round-trip, held-out BPB)

**Pilot budget (recorded BEFORE launch, `pilot_results/pilot_budget.json`):** each
promoted config trained for `min(1800 steps ≈ 3× the ~580-step short run, 45
wall-minutes)`, fp32, bs 2, ctx 2048 — a fixed multiple of the short-run token
budget AND a fixed walltime ceiling. This is a **rank-stability check, NOT a full
run**, and it doubles as the token-matched cross-walk to the handoff bf16 budget
(§7). Promoted: eval 65, 87, 95 (top-3 by AvgLoss).

### 6.1 Pilot results (all bounded at min(≤1800 steps, 45 wall-min))

| eval | short-run AvgLoss (15 min) | pilot steps | pilot AvgLoss | pilot **Final** (last-100) | held-out **BPB** | held-out nats/tok | round-trip |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 65 | 7.5075 (rank **1**) | 1620 | 6.6303 (rank 3) | 5.7671 | 2.1260 (rank 3) | 5.6805 | **OK** Δ=4.3e-6 |
| 87 | 7.5453 (rank 2) | 1796 | **6.5158** (rank **1**) | 5.7924 | **2.0899** (rank **1**) | 5.5839 | **OK** Δ=0.0 |
| 95 | 7.5630 (rank 3) | 1643 | 6.5498 (rank 2) | **5.6520** | 2.0999 (rank 2) | 5.6107 | **OK** Δ=1.9e-6 |

Held-out BPB on the canonical Pile held-out byte slice (`tokens/byte = 0.25942`,
p50k_base), 40 batches; `BPB = nats_per_token · tokens_per_byte / ln 2`.

### 6.2 Round-trip (HARD-BLOCKER gate) — PASS for all promoted configs

Each config: in-memory `l_pre` on a fixed held Pile batch → save (production
`model_state_dict`) → **reload in a fresh process** → `l_post`. All three reproduce
the loss to **Δ ≤ 4.3e-6** with **0 missing / 0 unexpected** keys — well inside the
1e-2 tolerance, and not key-match-alone (the wire-e99-e98 PILE_BPB forward-mismatch
failure mode is absent). Checkpoints deleted after the check (no-publish constraint).

### 6.3 Rank stability — the short-run order does NOT survive to 3× tokens

- Short-run (15 min) AvgLoss order: **65 → 87 → 95**.
- Pilot (3×) AvgLoss order: **87 → 95 → 65** (Spearman ρ = **−0.5** vs short-run).
- Held-out BPB order: **87 → 95 → 65** — **agrees with the pilot AvgLoss**, both
  reversing the short-run order.
- Pilot Final (last-100) order: 95 → 65 → 87 (a third order — the configs are
  near-ties so the last-100 window is noisier).

**Finding:** the 15-min short-run winner (eval 65, the lowest LR of the three) drops
to **last** on both pilot AvgLoss and held-out BPB once trained ~3× longer; the
higher-LR configs (87, 95) overtake it. At the very short fp32 budget, AvgLoss is
dominated by the warm-up window and favours the lower-LR config; with more tokens
the higher-LR configs converge faster. The three are a **tight near-tie band**
(pilot AvgLoss 6.516–6.630; BPB 2.090–2.126; Final 5.65–5.79), so the robust output
is the **top band**, not a single ranked winner — and within it the
**held-out-BPB-best is eval 87**, not the short-run winner eval 65.

### 6.4 Token-matched cross-walk to the handoff (resolves the dtype caveat)

At ~1800 steps the pilot reaches the bf16 15-min **token** budget (§7 note 1). The
pilot **Final** losses **5.65–5.79** are now in the range of the handoff **bf16**
arms (E88-delta 5.5529, Transformer 5.4683, FLA-GDN 5.6165, Mamba2 5.6441) — i.e.
**token-matched, typed-gdn2-lm sits in the pack**, slightly above the strongest
handoff arms. This confirms the raw short-run 7.51-vs-~6.0 gap was the **fp32 token
deficit, not an architectural deficit**.

---

## 7. Comparability / conversion notes vs `HANDOFF_E97_GDN2_CMAES_20260528.md`

The prior batch's best-config table (1.27 B, 2K-ctx, 15 train-min, p50k_base,
schedule-free AdamW), for reference:

| Target | Best avg loss | Best final loss | Best config |
|---|---:|---:|---|
| Transformer | 5.9046 | 5.4683 | dim1664, n_heads10, expansion6, depth19, lr5.164e-4, bs4 |
| E88 delta | 5.9974 | 5.5529 | dim2048, n_heads348, n_state32, depth10, lr9.973e-4, bs2 |
| E88 raw-write | 6.0395 | 5.5909 | dim1792, n_heads362, n_state32, depth11, lr9.413e-4, bs2 |
| M2RNN XMA | 6.0626 | 6.0626 | dim2304, n_heads612, n_state16, depth10, lr5.607e-4, bs5 |
| FLA-GDN | 6.1104 | 5.6165 | dim3456, expansion2, depth12, n_heads38, lr8.627e-4, bs2 |
| Mamba2 | 6.0560 | 5.6441 | dim1920, d_state64, expand4, depth27, lr1.417e-3, bs2 |
| **E99 typed-gdn2-lm (this run)** | **7.5075** | **7.5075** | **dim3328, n_heads102, n_state32, depth17, lr1.358e-3, bs2** |

**Preserved conventions:** candidate field names (`dim,n_heads,n_state,depth,lr,
batch_size`); run-ID/dir scheme (`benchmark_results/cmaes_1270M_ctx2k_e99_typed_gdn2_20260605/typed-gdn2-lm`);
data file + split + tokenizer (`pile.txt`, `p50k_base`); ctx (2048); per-candidate
budget unit (`train_minutes=15`); fitness = AvgLoss / Final over the train window;
popsize-8 counting; top-3 checkpoint retention.

**Irreducible differences (each with a conversion note):**

1. **dtype / token deficit — THE dominant caveat for cross-arm loss comparison.**
   The handoff arms ran **bf16** (the v2 harness hardwires `--bf16`); `typed-gdn2-lm`
   runs **fp32** (its sanity-validated dtype — the typed GDN-2 + negative-eigenvalue
   heads were verified fp32 in wire-e99-e98). fp32 1.3B is ~2.7 k tok/s vs bf16
   ~8.4 k tok/s, so in a wallclock-matched **15-min** window this candidate sees
   **~580 steps vs ~1800 (E88/bf16-class)** — roughly **3× fewer tokens**. AvgLoss
   also includes the high warm-up window, which is a *larger* fraction of a shorter
   run. **Therefore the raw 7.51 vs ~6.0 gap is confounded by ~3× less training and
   is NOT a clean architecture comparison.** *Conversion:* the §6 pilot trains the
   promoted configs to ~1800 steps (~3× tokens ≈ the bf16 15-min token budget),
   giving the **token-matched** loss that lines up with the handoff bf16 table.
   Both `train_minutes` (wallclock-matched) and `tok/s`+`tokens` (token-matched)
   are recorded per candidate so either cross-walk can be computed.
2. **batch_size fixed = 2** (handoff searched it). fp32 1.3B at ctx 2048 is
   memory-bound to bs≈2 on a 48 GB card (anchor peak 44 GB); fixing it removes a
   degenerate, OOM-prone search dimension and equalizes the per-eval token budget.
   Recorded as `batch_size=2` for column alignment. *Conversion:* none needed — it
   is a constant; the search is 5-D instead of 6-D.
3. **n_state snapped to {16,32}** (the typed GDN-2 head_dim supported set). The
   task's "N=64 only if headroom" was **not** included — 64 is outside the typed
   supported snap set and would change the head construction; every top config
   chose 32 regardless, so this did not bind.
4. **train-loss ↔ held-out BPB.** The handoff reported **train-window AvgLoss/Final
   in nats/token**, NOT held-out BPB. This run reports the same train AvgLoss/Final
   for all 96, and ADDS held-out BPB for the promoted configs (§6) on the canonical
   Pile held-out byte slice. *Conversion:* `BPB = nats_per_token · (tokens/byte) /
   ln 2`. The held-out slice's measured `tokens/byte` is recorded in
   `pilot_results.json`; train-window loss and held-out BPB are different
   distributions (train vs held-out) and must not be equated directly.
5. **Driver.** Search loop is a thin controlled wrapper (budget guards) around the
   handoff harness rather than `cmaes_search_v2.py main()` directly; it produces
   identical `.done`/output artifacts. The fla-gdn / E98-CMA controls are in the
   sibling `run-matched-1-3b`, not here.

---

## 8. Failure accounting (handoff taxonomy: schedule-free AdamW fragility)

The handoff flagged fragility/instability under schedule-free AdamW. In this run:
**0 / 96 evals failed** — no NaN, no OOM, no diverged (`success<10.0`) runs; every
generation reported `inf_frac = 0.0`. The fixed bs=2 + param-matched ±10 % window +
fp32 kept all candidates stable. (Had instability exceeded 50 % of a generation,
the instability hard-stop would have fired and been logged to `stop_reason.json`.)

---

## 9. Artifacts (committed; raw CMA state + results)

Run dir `benchmark_results/cmaes_1270M_ctx2k_e99_typed_gdn2_20260605/typed-gdn2-lm/`:
`results.json`, `candidates.json`, `candidates.csv`, `generations.jsonl`,
`cmaes_state.pkl`, `stop_reason.json`, `gpu_file.txt`, per-eval `eval_*/{.done,
params.json,batch_size.txt}` (large `.pt` checkpoints pruned — never staged/published).
Code: `experiments/e99_1p3b_cma/{run_e99_cma.py, orchestrate.py, gpufile_manager.py,
aggregate_results.py, prep_pilot.py, pilot.py}`. Pilot:
`experiments/e99_1p3b_cma/pilot_results/{pilot_budget.json, p*/pilot_results.json}`.

---

## 10. Constraints honored

No `paper/main.typ` edit · no push · no HF publish · no checkpoint published/staged
(sanity + pilot checkpoints deleted after round-trip) · no full multi-day 1.3B run
launched · **idle-GPU-only**: the search waited for the sibling `run-matched-1-3b`
controls to release GPUs (a descendant-aware manager only ever lists GPUs that are
idle or running this search's own processes — never preempting a neighbour).

---

## 11. Recommendation to `synthesize-e99-1-3b`

Carry forward the **top-3 band** `dim=3328, n_state=32, depth=17, n_heads≈101–102,
lr≈1.36–1.50e-3` (~1.16 B) as the E99 typed-gdn2-lm region. **Prefer eval 87**
(`lr=1.471e-3`) as the single front-runner: it is **best on both the token-matched
pilot AvgLoss (6.516) and held-out BPB (2.090)**, even though the 15-min short run
ranked eval 65 first. The §6.3 rank flip is the actionable result — **do not pick
the full-run config from the 15-min AvgLoss ranking**; the longer-budget AvgLoss and
held-out BPB agree on eval 87. **Do not compare the 15-min fp32 AvgLoss directly to
the handoff bf16 table** — use the token-matched pilot Final/BPB (§6.4), where
typed-gdn2-lm sits in the handoff pack. The full-run decision (dtype, token budget,
multi-day schedule) remains human-gated and out of scope here. See `run-matched-1-3b`
for the dense-GDN-2 / E98-CMA controls.
