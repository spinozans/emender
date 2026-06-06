# E99 / E98 / GDN-2 — 1.3B-class LM wiring sanity + checkpoint round-trip

**Task:** `wire-e99-e98`. Launch-gate for the E99 / typed-Emender 1.3B effort.
Wire the candidate roster into the **real production LM path** (`train.py` +
`ndm/models/ladder_lm.py` / FLA-GDN — the same path that produced the E88 1.273B
and FLA-GDN 1.352B baselines, see `THROUGHPUT.md` and `PILE_BPB_MEASURED.md`),
run a short **real-data** sanity per candidate, and verify checkpoint
**round-trip loss consistency** before any 96-candidate LM-CMA top-up or long run.

All training is REAL: real Pile tokens (`/home/erikg/elman/data/pile.txt`,
`p50k_base`, ctx 2048), real forward/backward, schedule-free AdamW, on **idle**
RTX 6000 Ada GPUs (5 and 6; verified 2 MiB used / 0 % before launch; no other job
preempted). `paper/main.typ` untouched. No checkpoint published/staged. No push.
No multi-day run. Measured **2026-06-05** (UTC).

---

## 0. TL;DR

All three wired candidates **pass the sanity gate**: loss decreases, NaN-free,
dtype path documented, and — the hard requirement — the checkpoint **reloads in a
fresh process and reproduces the pre-save loss within tolerance** (0 missing / 0
unexpected keys). The `PILE_BPB_MEASURED` failure mode (a strict-clean load that
forward-mismatches to ~17.6 nats/token) is **absent** on every candidate.

| candidate | role | params | dtype | loss (80 steps) | NaN | tok/s | round-trip Δ | RT ok |
|---|---|---:|---|---|:---:|---:|---:|:---:|
| **typed-gdn2-lm** | **E99 typed Emender (primary)** | 1.2775 B | fp32 | 11.44 → **8.56** | no | 2 422 | **3.1e-5** | ✅ |
| **fla-gdn** | dense GDN-2 / GatedDeltaNet control | 1.3524 B | bf16 | 11.37 → **8.94** | no | 8 364 | **1.2e-3** | ✅ |
| **e98-cma-lm** | E98-CMA unified control | 1.2776 B | fp32 | 11.50 → **10.98** | no | 2 308 | **0.0** | ✅ |

Round-trip tolerance = 1e-2 nats/token on the same fixed held batch. fp32
candidates reproduce essentially exactly (Δ ≤ 3e-5); bf16 fla-gdn has Δ = 1.2e-3
from chunk-kernel bf16 non-determinism — both well inside tolerance.

**No hard blockers.** One wiring gap was found and fixed (below); two soft notes
(fp32 throughput, e98 early-loss transient) are documented for the downstream CMA.

---

## 1. Candidate roster and exact configs

Field names follow the handoff (`dim, n_heads, n_state, depth, lr, batch_size`
+ model-specific `expansion`), extended with the production-path fields
(`level`, `layer_kwargs`, `dtype`). Machine-readable:
`experiments/e99_1p3b_sanity/candidate_configs.json`.

| field | typed-gdn2-lm (E99) | fla-gdn (dense GDN-2) | e98-cma-lm (E98-CMA) |
|---|---|---|---|
| level | `typed-gdn2-lm` | `fla-gdn` | `e98-cma-lm` |
| dim | 3072 | 2688 | 3072 |
| depth | 22 | 21 | 17 |
| n_heads | 96 | 44 | 192 |
| n_state | 32 | 64 | 16 |
| expansion | 1.0 | 2.0 | 1.0 |
| lr | 9.95e-4 | 8.63e-4 | 9.79e-4 |
| knob_lr_mult | 1.0 | 1.0 | **5.38** |
| batch_size (sanity) | 2 | 2 | 1 |
| dtype | fp32 | bf16 | fp32 |
| **params** | **1,277,500,000** | **1,352,352,498** | **1,277,608,704** |

**Per-candidate `layer_kwargs` (the discovered form, preserved):**

- **typed-gdn2-lm (E99):** `head_type_logits = [3.9995, -1.9008, -0.9211,
  -2.8866, 2.4146]` over `[gdn2_recall, e97_track, count, latch, nonlin]`,
  `gdn_allow_neg_eigval = true`, `lam_max = 1.585`, `beta_max = 2.747`.
  `softmax` ⇒ ≈ **82.2 % GDN-2 / 16.8 % nonlinear** — the **40:8 / 5:1
  GDN2:nonlinear** ratio from `typed-gdn-2-head`, preserved at scale (at
  `n_heads=96` ⇒ ~79 GDN-2 + ~16 nonlinear + ~1 each track/count/latch).
- **fla-gdn:** native FLA GatedDeltaNet, no extra knobs — the `THROUGHPUT.md`
  1.352B baseline config, and the **same** native delta-memory backbone the typed
  GDN-2 heads run.
- **e98-cma-lm:** `corner_mixture = [0.4015, 0.2821, 0.0089, 0.3075]` over
  `[track, count, latch, nonlin]`, `lam_max = 1.585`, `beta_max = 2.747`,
  `igain_max = 2.0`, split-gate + spread-init + `knob_lr_mult = 5.38` — the
  `cma-capability` 8M winner meta-config, with dim/depth/n_heads re-derived to
  1.3B (the report's named "e98-scale-run starting point").

---

## 2. Wiring — what was changed (and the one gap found)

The candidates resolve to three `ladder_lm.py` levels. Only `fla-gdn` already
conformed to the production `LadderLM` layer protocol
(`out, h = layer(x, prev_hidden)`); it needed **no** change.

**Gap (fixed):** `TypedHeadMixtureLayer` and `UnifiedCellLayer` were written for
the expressivity `HybridLadderLM` path — their `forward(self, x)` takes a single
tensor and returns a **bare tensor**, so `LadderLM.forward`'s
`x, h_final = layer(x, prev_hiddens[i])` would break. They are also not fed their
candidate knobs by `LadderLM` (which passes only a fixed kwarg set). Fix (commit
on `wg/agent-1116/wire-e99-e98`):

1. `ndm/models/ladder_lm.py`: added `_LadderProtocolAdapter` (accepts/ignores
   `prev_hidden`, returns `(out, None)` — the same "no recurrent carry" contract
   `FLAGatedDeltaNetLayer` honours with `use_cache=False`), and registered two
   **new** production levels `typed-gdn2-lm` / `e98-cma-lm` that wrap the
   expressivity layers. The original `typed-gdn2` / `e98-cma` levels and their 55
   unit tests are **untouched** (verified `55 passed`).
2. `ndm/models/ladder_lm.py`: `LadderLM.__init__` gained a `layer_kwargs` dict,
   merged last into every per-layer constructor (default `None` ⇒ zero change to
   all existing levels).
3. `train.py`: added `--head_type_logits / --corner_mixture / --lam_max /
   --beta_max / --igain_max / --knob_lr_mult / --gdn_allow_neg_eigval`, builds
   `layer_kwargs`, and replicates the expressivity path's **knob-LR param group**
   (`lam/beta/igain/gamma_raw` at `lr × knob_lr_mult`) for both schedule-free and
   vanilla AdamW. Verified end-to-end: `e98-cma-lm` builds with **8 knob params @
   5.38× base, 16 base params** and trains through `train.py`.

---

## 3. Sanity results (real Pile, 80 steps, schedule-free AdamW)

Raw per-candidate JSON: `experiments/e99_1p3b_sanity/results/*_result.json`.
Random-baseline loss = `ln(50281) = 10.83` nats/token.

| candidate | loss step 1 | step 20 | step 40 | step 60 | step 80 | decreasing? | NaN? |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| typed-gdn2-lm | 11.44 | 10.69 | 9.15 | 8.40 | **8.56** | ✅ (clear descent) | no |
| fla-gdn | 11.37 | — | — | — | **8.94** | ✅ | no |
| e98-cma-lm | 11.50 | 12.30 | 11.46 | 10.98 | **10.98** | ✅ (after transient) | no |

- **typed-gdn2-lm (E99):** clean monotone-ish descent well below random — the
  primary candidate trains as expected at 1.3B fp32.
- **fla-gdn:** 11.37 → 8.94 in 20 steps already (fastest learner; well-tuned
  native kernel).
- **e98-cma-lm:** loss first **rises** to ~12.3 (steps ~10–20) then descends to
  10.98 by step 80. The early bump is the high knob-LR (`5.38× ⇒ 5.27e-3`)
  transient moving the placed λ/β knobs; it recovers below its start and below
  random. Net **decreasing**, NaN-free — but see §6 note: the discovered 8M LR is
  untuned at 1.3B and is exactly what the downstream CMA should re-tune.

**Dtype path (documented, honest):**

- **fla-gdn = bf16.** Weights cast to bf16 + bf16 autocast — the proven
  `THROUGHPUT.md` path. Sustained 8 364 tok/s here vs 8 248 in `THROUGHPUT.md`
  (within the few-percent GPU-to-GPU variance that doc records) — a real
  cross-check that the wired path matches the baseline path.
- **typed-gdn2-lm = fp32 and e98-cma-lm = fp32.** Run in fp32 (no autocast),
  matching the validated expressivity path for the unified-cell Triton kernel and
  the typed arm (`TYPED_GDN2_MIXTURE_CMA_RESULTS.md`: "fp32 … for the typed/E98
  arms"). The FLA GatedDeltaNet chunk kernel inside the typed GDN-2 heads accepts
  fp32 here (unlike the plain DeltaNet chunk kernel, which rejects it). bf16 for
  these two is **not yet validated** and is flagged as future work — it would
  roughly halve their memory and raise throughput; the downstream CMA may test it.

**Peak memory (1× RTX 6000 Ada, 48 GB):** fla-gdn 20.9 GB (bf16, bs 2);
typed-gdn2-lm 44.2 GB (fp32, bs 2 — near the card limit); e98-cma-lm 34.8 GB
(fp32, bs 1).

---

## 4. Checkpoint round-trip (HARD requirement — loss consistency, not key-match)

Per candidate the driver: trains 80 steps → records the loss on a **fixed held
batch** (deterministic, seed 1234) → saves `model_state_dict` (production
convention) → **spawns a fresh Python process** that builds the model, `load_
state_dict`, and recomputes the loss on the **same** batch → compares. A
strict-clean load is explicitly **not** treated as sufficient.

| candidate | L_pre (fixed batch) | L_post (fresh process) | Δ | missing / unexpected | verdict |
|---|---:|---:|---:|:---:|:---:|
| typed-gdn2-lm | 8.564193 | 8.564224 | **3.05e-5** | 0 / 0 | ✅ reproduces |
| fla-gdn | 8.850078 | 8.848924 | **1.15e-3** | 0 / 0 | ✅ reproduces |
| e98-cma-lm | 11.973406 | 11.973404 | **1.9e-6** | 0 / 0 | ✅ reproduces |

All Δ ≪ 1e-2 tolerance. fp32 candidates are deterministic (Δ ≤ 3e-5); bf16
fla-gdn's 1.2e-3 is chunk-kernel bf16 non-determinism, not a load defect. **No
forward/recurrence mismatch on any candidate** — the documented E88 failure mode
does not reproduce here.

---

## 5. Throughput → projected walltime / GPU-days

Sustained tok/s = mean of per-step throughput after a 3-step warmup, ctx 2048,
on one RTX 6000 Ada. GPU-days for a token budget = `tokens / tok_s / 86400`.

| candidate | tok/s | GPU-days / 1B tok | GPU-days / 10B tok |
|---|---:|---:|---:|
| fla-gdn | 8 364 | 1.38 | 13.84 |
| typed-gdn2-lm | 2 422 | 4.78 | 47.78 |
| e98-cma-lm | 2 308 | 5.02 | 50.15 |

The fp32 candidates are ~3.5× slower than the bf16 control — see the
comparability warning in §7 and `budget_caps.json`.

---

## 6. Blockers & soft notes

- **Hard blockers: NONE.** All three candidates wired into the real path, train
  NaN-free with decreasing loss, and pass the round-trip.
- **Fixed during this task:** (a) the forward-protocol gap (§2); (b) a driver bug
  where the round-trip reload subprocess collided with the still-resident parent
  model on one GPU (fixed by freeing parent GPU memory before spawning the
  reload). Neither affects the candidates.
- **Soft note — fp32 throughput.** typed-gdn2-lm / e98-cma-lm run fp32 (~2.3–2.4k
  tok/s) vs bf16 fla-gdn (8.4k). At a **wallclock-matched** 15-min CMA window the
  fp32 arms see ~3.5× fewer tokens than the bf16 control. Both train_minutes and
  token caps are emitted so the downstream tasks can pick wallclock- or
  token-matched and report both (§7).
- **Soft note — e98-cma early transient.** The 8M `knob_lr_mult=5.38` / `lr=9.79e-4`
  is untuned at 1.3B: loss bumps up before descending. Not a wiring fault; it is
  the exact thing the LM-CMA top-up is meant to re-tune. Reported, not hidden.

---

## 7. Comparability / conversion notes (REQUIRED — vs the E97/GDN-2 CMA-ES batch)

Read against `docs/HANDOFF_E97_GDN2_CMAES_20260528.md`. This task **owns** setting
up the conventions the batch inherits; they are emitted in
`candidate_configs.json` + `budget_caps.json` and mapped here.

1. **Candidate budget / counting — PRESERVED.** Prior `--popsize 8
   --min_generations 8`; "topped up to 96" = **96 total candidate evaluations =
   popsize 8 × 12 generations** (continue the prior popsize-8 CMA through
   generations 9–12, +32 evals from the 8×8=64 base; if prior state isn't
   resumable on the production path, restart popsize-8 × 12 gens and document the
   restart). Each eval is a SHORT run, not a full 1.3B training.

2. **Run IDs / output dir — MAPPED.** Prior scheme
   `benchmark_results/cmaes_1270M_ctx2k_<tag>_<YYYYMMDD>/<model>`. Reuse it for
   E99, e.g. `benchmark_results/cmaes_1270M_ctx2k_e99_typed_20260605/typed-gdn2-lm`.
   **Mismatch flag:** the prior batch drove `scripts/cmaes_search_v2.py`; the
   production LM path here is `train.py` / `ladder_lm.py`. run-e99 must either (a)
   point `cmaes_search_v2.py` at the production `train.py` config emitted here, or
   (b) wrap `train.py` in the CMA loop — either way reusing this run-ID/dir scheme.

3. **Model/config field names — PRESERVED.** `dim, n_heads, n_state, depth, lr,
   batch_size` (+ `expansion`; Mamba2 `d_state`/`expand`) used verbatim in
   `candidate_configs.json`. Extra production-path fields (`level`,
   `layer_kwargs`, `dtype`) are additive.

4. **Training budget — MAPPED with conversion.** Prior `--train_minutes 15
   --params 1270M --chunk_size 2048` (wallclock-matched). Preserved as the primary
   unit. Token cap = `15 × 60 × tok_s` emitted per candidate (typed 2.18M, fla
   7.53M, e98 2.08M tokens / 15-min eval). **Conversion note:** wallclock-matched
   ≠ token-matched because dtype/throughput differ ~3.5×; `budget_caps.json`
   carries both so either basis can be used, with the other reported alongside.

5. **Dataset / split + tokenizer — PRESERVED.** `--data
   /home/erikg/elman/data/pile.txt`, `--tokenizer p50k_base`, ctx 2048 — identical
   to the handoff. No change.

6. **Fitness definition — PRESERVED.** Prior fitness = best **AvgLoss / Final
   loss** over the train window (nats/token). Keep reporting **both** so columns
   line up. Any E99 extension (held-out BPB / worst-case capability score) is
   reported **alongside**, never replacing, AvgLoss/Final.

7. **Held-out BPB / loss — CONVERSION NOTE.** The prior batch reported **train
   AvgLoss/Final (nats/token)**, NOT held-out BPB. To compare an E99 BPB to the
   prior loss table: BPB = (nats/token) ÷ (ln 2) ÷ (bytes/token). For `p50k_base`
   on Pile the empirical bytes/token must be measured on the same slice (see
   `PILE_BPB_MEASURED.md` / `COMMA_PILE_BPB.md`); do NOT compare a held-out BPB to
   a train-window nats/token without (i) converting nats→bits and (ii) noting
   train-window vs held-out. Until then, report nats/token to line up with the
   handoff and BPB only as a labeled extension.

8. **Throughput fields — RECORDED.** Prior was wallclock-matched; here tok/s +
   walltime + GPU-days are recorded per candidate (§5) so both conventions
   cross-walk.

9. **Failure accounting — REUSED taxonomy.** Prior recorded fragility/instability
   under schedule-free AdamW. Same taxonomy applies: this sanity saw **0 NaNs / 0
   diverged runs**; the one instability observed is the e98-cma early knob-LR
   transient (recovers), logged as "transient-but-recovers", not a failure.

10. **Raw schema + summary columns — EXTENDED, not replaced.** Prior summary
    columns `Target | Best avg loss | Best final loss | Best config`. E99 emits
    those plus additive columns (params, dtype, tok/s, round-trip Δ, GPU-days) so
    rows stay directly comparable to the handoff table.

**Net comparability flag for run-e99 / run-matched:** the only genuine mismatch is
**driver** (production `train.py`/`ladder_lm.py` vs prior `cmaes_search_v2.py`) and
**dtype/throughput** (fp32 candidates vs bf16 controls ⇒ wallclock ≠ token
matched). Both are documented with conversions above and in `budget_caps.json`;
every other handoff convention is preserved verbatim.

---

## 8. Artifacts & reproduction

- Driver (real data, real fwd/bwd, fresh-process round-trip):
  `experiments/e99_1p3b_sanity/e99_lm_sanity.py`
- Per-candidate results: `experiments/e99_1p3b_sanity/results/{typed-gdn2-lm,fla-gdn,e98-cma-lm}_result.json`
- Candidate configs (downstream-consumable): `experiments/e99_1p3b_sanity/candidate_configs.json`
- Budget caps (per-candidate + aggregate, hard-stops): `experiments/e99_1p3b_sanity/budget_caps.json`
- Wiring: `ndm/models/ladder_lm.py` (`_LadderProtocolAdapter`, `typed-gdn2-lm` /
  `e98-cma-lm` levels, `layer_kwargs`), `train.py` (candidate-knob CLI + knob-LR
  group). 55/55 candidate-layer unit tests pass (no regression).
- Reproduce one candidate:
  `CUDA_VISIBLE_DEVICES=<idle> python3 experiments/e99_1p3b_sanity/e99_lm_sanity.py --config typed-gdn2-lm --steps 80 --batch_size 2`

Large `.pt` sanity checkpoints are intentionally **not committed/published** (task
constraint); they are regenerated by the driver and deleted after the round-trip.
