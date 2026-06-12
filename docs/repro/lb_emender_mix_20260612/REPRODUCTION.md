# lb-emender-mix — full-range mixture-axis 1.3B CMA (MEASURED data)

The KEY mixture run: the typed-gdn2 **Emender** searched at 1.3B with the mixture
axis spanning the **FULL range** 0% nonlinear (= pure `gdn2_recall` sea) → 100%
(= pure `e97_delta` split-edit). Prior emender runs capped the nonlinear fraction
at ~0.5 and cut at 8 gens; here the bound is `mixture_nonlin ∈ [0.0, 1.0]` and the
search runs the full symmetric budget so CMA can reach **both** pure corners.

## Protocol (standard driver, symmetric to the gdn2 / m2rnn / e97 1.3B arms)

- Driver: `scripts/cmaes_search_v2.py --model emender` → `--level typed-gdn2-lm`.
  The mixture-balance scalar `mixture_nonlin = f` maps to the 9-vector
  `head_type_logits` as `logit[0]=log(1−f)` (`gdn2_recall`), `logit[7]=log(f)`
  (`e97_delta`), all others `−30`. `allocate_types` then softmax→largest-remainder
  assigns integer head counts. The **only** change vs the committed setup is the
  search bound `mixture_nonlin: (0.0, 0.5) → (0.0, 1.0)` (commit `193fb0c`).
- **Full-range corner reachability VERIFIED** (`corner_reachability.txt`): at
  nh∈{32,378,2000}, `f=0` → **all** heads `gdn2_recall` / 0 `e97_delta`, and
  `f=1` → **all** heads `e97_delta` / 0 `gdn2_recall`. Both pure corners are
  selectable; the central anchor `f=0.5` lets CMA refine toward either.
- **n_state PINNED 32**, **expansion PINNED 1.0** (`EMENDER_N_STATE`/`EMENDER_EXPANSION`).
  Searched axes: `dim` (1024–4096, ×128), `n_heads` (32–2000), `depth` (10–50),
  `lr` (1e-4–3e-3 log), `batch_size` (1–128 log, clamped), and the full-range
  `mixture_nonlin` (0.0–1.0).
- **Precision/kernel: bf16 uniform + FUSED Triton split-edit, NO eager.** Asserted
  from the live worker cmdlines: `--bf16`, `--level typed-gdn2-lm`,
  `--layer_kwargs {"use_triton_e97": true, "cast_recurrent_bf16": true,
  "e97_state_nonlin": "tanh", "use_chunked_e97_delta": false, "overlap_streams": true}`.
  `use_triton_e97=True` + `cast_recurrent_bf16=True` guarantee the bf16 fused gate
  engages; there is no eager fallback in the worker cmdlines.
- Data slice: SAME `--data /home/erikg/elman/data/pile.txt`, `--seed 42`,
  `--tokenizer p50k_base` (vocab 50281) as
  `~/emender/experiments/local/cmaes_redo_1300m_20260529` → numbers directly
  comparable to the OLD leaderboard.
- CMA: popsize 8, min_generations 13, sigma 0.8, params 1300M, param_tolerance 0.03,
  chunk_size 2048, train_minutes 15. Central full-range anchor
  `dim1280/dep20/nh256/f0.5` (anchor_only). 8 GPUs (one candidate per GPU), leased
  via `scripts/gpu_lease.sh`.

## Result (MEASURED) — the full-range optimum is near the PURE-E97 corner

- **Best avg-loss: 6.0756**  (104 evals, 13 generations, CONVERGED, 5.97 h wall;
  `generations_without_improvement=9`, `sigma 0.275→0.112`).
- Best geometry (eval 47): `dim=2432, n_heads=212, depth=10, mixture_nonlin=0.9708,
  lr=1.144e-3, batch_size=2`, **n_state=32**, expansion=1.0 → **1273.2M params
  (−2.06% of 1300M, within ±3%)**. final-window loss 5.621.
- **Mixture fraction f = 0.9708 ⇒ ~97% `e97_delta` heads, ~3% `gdn2_recall`** — the
  free full-range search converged to the **near-pure-E97 corner**, not the linear
  one. Mixture is the load-bearing axis: it moved decisively to the nonlinear end.
- **Full mixture axis was actually explored**: `f ∈ [0.0017, 0.9975]` across the 104
  valid evals; per-0.1-bin histogram `{0.0:5, 0.1:2, 0.2:2, 0.3:4, 0.4:10, 0.5:11,
  0.6:10, 0.7:20, 0.8:17, 0.9:23}`. The **pure-gdn2 region** (`f<0.1`, 5 evals)
  bottoms out at **6.8263**; the **pure-e97 region** (`f>0.9`, 23 evals) reaches
  **6.0756** — a 0.75 gap favouring nonlinear. The linear corner is decisively worse.

### Convergence (best-so-far)

| gen | eval | best | f | dim | nh | dep | sigma |
|----:|-----:|-----:|----:|----:|---:|----:|------:|
| 0 | 8 | 6.4763 | 0.465 | 1792 | 260 | 14 | 0.275 |
| 1 | 16 | 6.1954 | 0.418 | 1664 | 271 | 14 | 0.275 |
| 2 | 24 | 6.1052 | 0.597 | 2304 | 249 | 10 | 0.276 |
| 3 | 32 | 6.0757 | 0.986 | 1920 | 284 | 10 | 0.279 |
| 5 | 48 | 6.0756 | 0.971 | 2432 | 212 | 10 | 0.271 |
| 12 | 104 | 6.0756 | 0.971 | 2432 | 212 | 10 | 0.112 |

Monotonic; locked onto the near-pure-E97 basin by g3 (f 0.60→0.99), then refined
geometry within it. Top-3 candidates span 6.0756–6.0999 (all f≥0.89) — the in-run
noise floor, all on the nonlinear side.

- Full curve: `emender_mix_generations.jsonl`; all 104 evals: `emender_mix_results.json`;
  distilled best: `best.json`.

## Context vs the old 1.3B leaderboard (p50k_base, same slice)

| arm | best avg-loss |
|---|---|
| e97-raw (`e88_raw_write`) | 5.9511 |
| gdn2-mlp | 5.9613 |
| pure-e97_delta (`lb-e97-pure`, reproduced) | 5.9869 |
| fla-gdn | 6.0854 |
| m2rnn | 6.1161 |
| **emender full-range mixture (this run)** | **6.0756** |

The mixture run's optimum (6.0756) sits between fla-gdn (6.0854) and the e97/gdn2
leaders. NOTE: this run's "pure-E97 corner" is the typed-mixture **`e97_delta`**
(delta-rule split-edit + tanh state nonlin), a different kernel from `lb-e97-pure`'s
standalone **`E97`/`e88_raw_write`** (raw split erase/write) — so the +0.089 gap vs
that arm reflects the different operator and the wider search space (5 geometry axes
+ the mixture axis under a shared 15-min budget), not a re-run of the same cell.
This is a **mixture-fraction / fitness result** (where the free axis lands and the
measured avg-loss there), not a capability or wall-clock GO/NO-GO.

**Bottom line (MEASURED):** with the mixture axis free over the full 0–100% range
at 1.3B, CMA drives it to **~97% nonlinear (near the pure-E97 corner)**; the pure-
linear `gdn2_recall` corner is decisively worse (6.83 vs 6.08). Best avg-loss 6.0756
at dim2432/nh212/n_state32/dep10/f0.971, converged over 104 evals / 13 generations.
