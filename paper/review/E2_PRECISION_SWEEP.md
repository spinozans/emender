# E2 — Precision sweep (fp64 / fp32 / bf16), serial S5 eval of e88-linear (tests H1)

**Task:** `e2-precision-sweep` (model claude:opus). Implements experiment **E2** from
`paper/review/PRECISION_NONLINEARITY_RESEARCH.md` §7. **`paper/main.typ` NOT edited.
Committed, not pushed.** GPUs: **only 2,3** used (RTX 6000 Ada).

## What this tests (H1)

H1 (`PRECISION_NONLINEARITY_RESEARCH.md` §0): *serial finite-precision rounding is a
per-step nonlinearity that could lift the nominally-linear `e88-linear` recurrence above
its idealized TC⁰ ceiling at bounded length, partly explaining why the linear-state model
won the S5 slot.* The decisive readout, stated in the task and §7-E2:

- **accuracy RISES as precision DROPS** (bf16 > fp32 > fp64): rounding is *helping* →
  **supports H1**.
- **accuracy flat, or rises WITH precision** (fp64 ≥ fp32 ≥ bf16): the win is *not*
  rounding-driven; fp64-serial ≈ idealized-linear proxy → **H1 disfavored**.

## Regime: train-vs-eval-only (stated honestly)

**Both regimes were run** (REAL training + REAL eval, no mocks):

1. **Eval-only (primary, the task's required fallback).** A single model is trained per
   seed at **fp32** (params fp32, no autocast); the **same** serial model is then
   evaluated at **fp64 / fp32 / bf16**. This is the cleanest H1 test: the weights are
   identical across the three arms, so *only the eval-time arithmetic dtype (the per-step
   rounding) changes*. This is the controlled "does serial rounding, at inference, change
   what the fixed linear operator computes" measurement.
2. **Train-per-precision (secondary, the diagonal).** A model is *trained* at **bf16**
   (matches the published §6 recipe — bf16 autocast) per seed and evaluated at all three
   dtypes. The train=eval diagonal gives bf16-train→bf16-eval vs fp32-train→fp32-eval.

**fp64 *training* was NOT run: it is infeasible on the available hardware.** RTX 6000 Ada
has ≈1/64 fp32 throughput in fp64; a 12 000-step S5 run with an fp64 backward would take
days. Per the task's explicit fallback ("if compute-bound, at minimum eval a fixed
fp32-trained model under the three eval dtypes AND report which you did"), the fp64 arm is
**eval-only**, and the **fp32-train → fp64-eval** cell is used as the *idealized-linear
proxy* the experiment design calls for.

## Setup (identical except dtype)

- **Model:** the `e88-linear` CMA winner, verbatim from
  `…/s5_symmetric_20260603/winners/e88-linear.args.json`:
  `level=E88, dim=256, depth=5, n_heads=38, n_state=32, linear_state=1, use_gate=1,
  lr=0.0026571…`, schedule-free AdamW, **7.86 M params**.
- **Task / recipe:** `s5_permutation` (120-way, random = 0.0083), train T=128, batch 32,
  **12 000 steps** (the published curve is converged ≈0.99 by step 10–12 k; all 6 runs
  reached probe acc 0.991–1.000 at T=128). Eval grid **T ∈ {128, 256, 512, 1024}**,
  8 batches/length. **3 seeds {42, 123, 456}.**
- **Serial path:** eval always runs the dtype-driven **PyTorch serial recurrence**
  (`ndm/models/e88_fla_hybrid.py:1667`, the per-timestep `for t in range(T)` loop). The
  fast CUDA/Triton/fused kernels are gated on `x.dtype == bf16 and self.training`, so at
  eval *every* arm uses the same serial Python loop — only the arithmetic dtype differs
  (fp64 = `model.double()`, no autocast; fp32 = plain; bf16 = bf16 autocast, matching the
  published eval). Verified: fp64 eval produces `torch.float64` logits.
- **Honest caveat:** the Mamba-style decay is computed in fp32 inside the layer
  (`A_log.float()`, `softplus(.float())`) for *all* arms, then cast to the working dtype.
  This fp32 decay step is **constant across the three arms**, so it does not confound the
  comparison; the accumulating state recurrence — where H1's per-step rounding would live
  — runs in the target dtype.

Driver: `scripts/e2_precision_sweep.py` (train + 3-dtype serial sweep);
orchestrator `scripts/e2_run_sweep.py` (GPUs 2,3 only); aggregator `scripts/e2_aggregate.py`.

## Results — accuracy by precision (seed-mean ± SD over {42,123,456})

### Eval-only regime — train_dtype = fp32 (PRIMARY)

| eval dtype | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|
| fp64 | 0.9951 ± 0.0058 | 0.7287 ± 0.0816 | 0.3776 ± 0.0438 | 0.1960 ± 0.0222 |
| fp32 | 0.9955 ± 0.0055 | 0.7346 ± 0.0793 | 0.3740 ± 0.0453 | 0.1964 ± 0.0266 |
| bf16 | 0.9971 ± 0.0030 | 0.7385 ± 0.0868 | 0.3778 ± 0.0458 | 0.1934 ± 0.0243 |

### Train-per-precision diagonal — train_dtype = bf16 (SECONDARY)

| eval dtype | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|
| fp64 | 0.9968 ± 0.0026 | 0.8395 ± 0.0100 | 0.4431 ± 0.0071 | 0.2303 ± 0.0070 |
| fp32 | 0.9979 ± 0.0013 | 0.8445 ± 0.0102 | 0.4424 ± 0.0095 | 0.2321 ± 0.0063 |
| bf16 | 0.9989 ± 0.0008 | 0.8519 ± 0.0204 | 0.4456 ± 0.0159 | 0.2305 ± 0.0049 |

(random baseline = 0.0083 at every length.) Raw per-seed JSONs:
`experiments/expressivity_tasks/results/e2_precision_sweep_20260604/e88linear_train{fp32,bf16}_seed{42,123,456}.json`;
roll-up `…/summary.json`.

## Monotonicity vs precision — interpretation

Because the eval sweep re-runs the **same** trained weights at each dtype, the
bf16−fp64 difference is a **paired, per-seed** quantity (the most powerful test). Seed-mean
deltas (eval-only / fp32-train):

| T | fp64 | fp32 | bf16 | bf16−fp64 (paired, per-seed) | reading |
|---|---:|---:|---:|---:|---|
| 128 | 0.9951 | 0.9955 | 0.9971 | +0.0020 → [−0.001, +0.006, +0.001] | bf16 ≳ fp32 ≳ fp64 |
| 256 | 0.7287 | 0.7346 | 0.7385 | +0.0097 → [+0.006, +0.007, +0.017] | bf16 ≳ fp32 ≳ fp64 |
| 512 | 0.3776 | 0.3740 | 0.3778 | +0.0002 → [−0.001, −0.001, +0.003] | flat / sign-flips |
| 1024 | 0.1960 | 0.1964 | 0.1934 | −0.0025 → [−0.002, −0.007, +0.001] | flat / sign-flips |

The bf16-train diagonal shows the same shape (bf16−fp64 = +0.002 / +0.012 / +0.003 / +0.000
at T=128/256/512/1024, sign-flipping across seeds at the long lengths).

**Verdict: accuracy is FLAT vs eval precision → H1 DISFAVORED.**

1. **The precision effect is sub-1.5-percentage-point and not length-robust.** The largest
   seed-mean gap is +0.012 (T=256). At T=512/1024 the paired bf16−fp64 delta is ≈0 and
   **flips sign across seeds**. By contrast the seed-to-seed SD is 0.01–0.09. The dtype
   signal is *inside the noise* and is two orders of magnitude smaller than the quantity it
   would need to explain — the linear model's ~0.99-at-T=128 separation from the baselines
   (GDN 0.54, M²RNN 0.17 in `S5_SYMMETRIC_RESULTS.md`).

2. **fp64-serial ≈ bf16-serial at every length.** The idealized-linear proxy (fp32-train →
   **fp64**-eval) reproduces *both* the T=128 win (0.9951) *and* the length-decay
   (0.73 → 0.38 → 0.20) essentially identically to bf16 (0.9971 → 0.7385 → 0.3778 → 0.1934).
   Driving the recurrence to near-exact arithmetic does **not** remove the win and does
   **not** rescue length extrapolation. This is exactly the §7-E2 "fp64 ≥ / ≈ bf16 ⇒ H1
   disfavored, fp64-serial ≈ near-ideal linear" branch.

3. **There IS a faint, repeatable bf16 ≳ fp32 ≳ fp64 lean at short lengths (T=128, 256)** —
   the *direction* H1 predicts. Reported honestly rather than dismissed: at T=256 every
   seed shows bf16 > fp64 (paired +0.006/+0.007/+0.017). But it is <1.5 pp, vanishes and
   reverses by T=512/1024, and is far too small to be the mechanism behind a 45-point
   separation from baselines. At face value it is consistent with a *tiny* rounding
   perturbation, not with "rounding supplies the usable nonlinearity that lifts the linear
   model out of TC⁰."

**Consequence (consistent with `PRECISION_NONLINEARITY_RESEARCH.md` §1, §6.5).** The
linear-state model's S5 behavior is a property of the **idealized linear operator itself**,
not of serial rounding: an *input-dependent, gated* (`use_gate=1`) linear recurrence
solves a fixed-length S5 instance and then fails to extrapolate, and fp64 reproduces this
unchanged. This points to the research doc's *leading competing explanation* (§6.5 /
Thread 4: input-dependent / dense-transition / eigenvalue expressivity — Cirone 2024,
Grazzi 2025, Movahedi 2025), **not** to H1/H3's serial-precision-artifact framing. The
paper's asymptotic TC⁰/NC¹ argument is untouched and need not invoke rounding; E2 removes
rounding as the explanation for the finite-length win.

### Secondary observation (train-time precision; out of H1's scope, confounded)

bf16-**trained** models extrapolate *better* than fp32-trained ones at length (T=256: 0.85
vs 0.73; T=512: 0.44 vs 0.38; T=1024: 0.23 vs 0.20), while both are at ceiling at T=128.
This is a **training-precision** effect (a different SGD trajectory; bf16 gradient noise
acting regularizer-like), **not** the inference-time rounding nonlinearity H1 is about, and
it carries the confound that the two are independent optimization runs (3 seeds each). The
controlled H1 test is the *within-model* eval sweep, which is flat. If anything, this is a
"stochastic-rounding-as-regularizer-during-training" story (a *different* mechanism than
H1), and even there the gap is modest and the long-T accuracy still collapses toward chance.

## Validation checklist

- [x] Serial S5 evaluated at **fp64 / fp32 / bf16**, **3 seeds {42,123,456}**, length grid
  **{128,256,512,1024}**; raw per-seed JSONs committed (`…/e2_precision_sweep_20260604/`).
- [x] **Train-vs-eval-only regime stated**: eval-only on fp32-trained models (primary) +
  bf16-train diagonal (secondary); **fp64 training infeasible on RTX 6000 Ada → fp64 is
  eval-only**; fp32-train→fp64-eval is the idealized-linear proxy.
- [x] **Monotonicity-vs-precision interpretation stated**: accuracy **flat** vs precision
  (paired bf16−fp64 < 1.5 pp, sign-flips at long T, ≪ seed SD); fp64-serial ≈ bf16-serial
  → **H1 DISFAVORED**. Faint short-length bf16≳fp64 lean reported honestly as too small/
  non-robust to be the driver.
- [x] **Only GPUs 2,3 used** (`CUDA_VISIBLE_DEVICES` ∈ {2,3} on every process);
  **`paper/main.typ` untouched**; **not pushed**.

## Reproduce

```bash
# 6 runs (fp32-train + bf16-train) x 3 seeds, GPUs 2,3 only, 12000 steps:
python scripts/e2_run_sweep.py
# seed-mean acc-by-precision tables + H1 monotonicity verdict:
python scripts/e2_aggregate.py
```
