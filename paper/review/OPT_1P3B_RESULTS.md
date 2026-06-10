# OPT_1P3B — 1.3B-scale validation of the optimized regime R* vs CMA-best controls

**Task:** `opt-1p3b` (consumes `opt-synth` / OPT_SYNTHESIS.md §4–5) · **Role:** Programmer
· **Date:** 2026-06-10 · REAL Comma-Pile mainmix data, matched compute, **FUSED Triton**
(Erik's mandate), broker-leased GPUs.

This document reports the 1.3B validation of the optimization-not-architecture line:
does the optimized GDN+nonlin mixture **R\*** (carried from the small-scale synthesis)
beat **CMA-ES-optimized controls** — best-vs-best — at 1.3B on held-out BPB and joint
capability coverage? Per Erik's directive, the controls are **not** default-trained:
each sits at its own CMA optimum for the 1.3B budget.

---

## 0. TL;DR

- **Three arms, each ~1.3B, matched wall-clock, real data, fused kernels** (§1):
  R\* (typed GDN+nonlin mixture + `head_lr_compute_mult=5`) vs **CMA-best GDN-2** (B)
  vs **CMA-best m2rnn** — both controls at their own 1.3B CMA optimum (Erik's directive).
- **Verdict: NULL — both gates fail; CMA-best GDN-2 wins.**
  - **Held-out BPB** (real Comma-Pile): **R\* 2.220, B(GDN-2) 1.765, m2rnn 1.810** — R\*
    regresses vs B by **+0.455 BPB** wall-matched, and is still behind **token-matched**
    (6.083 vs 6.011 nats/tok @ common 2.7 Mtok).
  - **Worst-corner JCC** (matched-step capability @len128): **R\* 0.095, B(GDN-2) 0.274,
    m2rnn 0.044** — B dominates every corner; R\* ties only on step_growth and collapses on
    recall/track. Even GDN-2's weakest corner beats R\*'s weakest.
  - Both measurements agree: **optimization does not move the optimum at scale.** This is
    the pre-registered expectation (every pure lever sub-Δ\* at small scale; a tuned GDN-2
    ties the mixture) — the convergent-loss null extends from architecture to optimization
    at 1.3B, best-vs-best (controls at their own CMA optimum, no hobbled baseline).

---

## 1. The three arms and their 1.3B geometries (CMA control geometries reported)

| arm | role | level | dim | depth | n_heads | n_state | exp | params | dtype | fused path |
|---|---|---|---|---|---|---|---|---|---|---|
| **R\*** | optimized mixture (arm under test) | `typed-gdn2-lm` | 3072 | 22 | 64 | 32 | 2.0 | 1.34B | fp32 | `use_triton_e97=True` (loud-guard) |
| **B = CMA-best GDN-2** | incumbent control | `fla-gdn` | 2688 | 21 | 44 | 64 | 2.0 | 1.35B | bf16 | FLA chunked GDN-2 |
| **CMA-best m2rnn** | raw-write foil | `m2rnn` | 1920 | 21 | 370 | 16 | 1.0 | 1.31B | bf16 | XMA Triton (`XMA_PATH`) |

**R\* head mixture (house placement, from OPT_SYNTHESIS §3.4):** 50 % `gdn2_recall`
(neg-eig on → recall+track) + 25 % `nonlin` (UnifiedCell, step-growth) + 25 % `refit`
with `refit_has_mom=0` (= `e97_delta` exact, counting). At n_heads=64 → 32 / 16 / 16.
**R\* lever:** per-head-type LR `head_lr_compute_mult=5` (recall-class `gdn` at 1×,
compute-class `unified`+`refit` at 5×). `decay_init` left at default (the slow variant
**regressed recall in composition** at small scale — OPT_SYNTHESIS §3.3, dropped).

**CMA-best control geometries (reported per Erik's directive):**
- **GDN-2** (`hf_v03_fix_staging/gdn-1.3b`): the 1.3B CMA-leaderboard GDN-2 geometry
  (dim 2688 / depth 21 / 44 heads / expansion 2.0, `gdn_allow_neg_eigval=1`). This is the
  strongest honest incumbent — the CMA `lr`/shape search already removed GDN-2's
  small-scale counting/step-growth LR-artifact weakness (OPT_SYNTHESIS §2.2/§4.2).
- **m2rnn** (`hf_v03_fix_staging/m2rnn-cma-1.3b`): the existing CMA-best m2rnn geometry
  (dim 1920 / depth 21 / 370 heads / n_state 16 / expansion 1.0, sigmoid gate,
  `linear_state=false`). Reused directly per OPT_SYNTHESIS §4.3 (preferred — already
  CMA-converged at the 1.3B target).

**Geometry note (honest deviation).** R\* cannot share the GDN-2 control's *exact*
ns64/exp2.0 because the fused unified(nonlin) kernel requires `n_state*expansion ≤ 64`
(ns64×exp2.0 → V=128 raises `NotImplementedError`). R\* is therefore param-matched to the
control via d3072/dep22/h64/ns32/exp2.0 = **1.34B ≈ the 1.35B GDN-2 control** (V=64,
valid). Both controls keep their own CMA-best geometry. All three land within 3 % of
each other on parameter count — a fair matched-capacity comparison.

**FUSED-path compliance (Erik's mandate).** The runner asserts the fused path and FAILS
LOUD otherwise: R\* — all 22 `TypedHeadMixtureLayer`s carry `use_triton_e97=True`
(verified at build, loud-guard active, no eager T-scan); m2rnn — `XMA_M2RNN_AVAILABLE`
asserted True (eager fallback forbidden); GDN-2 — FLA chunked path. No
`use_triton*=False` / `fused=False` / `cplx_force_sequential` anywhere.

---

## 2. Method (matched compute, two measurements — OPT_SYNTHESIS §4.5)

`experiments/opt_1p3b/lm_runner.py` + `run_lm_matrix.py` (BPB) and `run_capability.py`
(coverage); `aggregate.py` produces the tables below. Method mirrors the prior 1.3B
control protocol (`experiments/e99_1p3b_controls/e99_lm_controls.py`): matched-wallclock,
schedule-free AdamW, NaN/OOM hard-stop, fresh-process checkpoint round-trip, token-curve
recorded for the token-matched cross-walk, BPB on a disjoint held-out slice with
**measured** bytes/token (`p50k_base`).

1. **Held-out BPB** (§3.1) — each arm trained **wall-clock-matched** (`--train_minutes`,
   N seeds) on real Comma-Pile mainmix; BPB = (nats/tok)/ln2/(bytes/tok). fp32 R\* sees
   fewer tokens than the bf16 controls in equal wall-clock (the documented §4.5.2
   token-vs-wall split) — so we report **both** the wall-matched BPB and the
   token-matched train-loss cross-walk at the common token budget.
2. **Capability coverage** (§3.2) — the §3 battery on the 1.3B-shaped cell, one task per
   JCC corner (recall=`mqar_recall`, counting=`modular_counter`, step_growth=
   `modular_quadratic`, track=`s5_permutation`), evaluated at train length 128 and
   **extrapolated to 512** (the axis OPT_SYNTHESIS §4.6 flags as where the levers'
   worst-corner / length-extrapolation value is best detected). JCC = min-corner accuracy.

---

## 3. Results

### 3.1 Held-out BPB at 1.3B (matched wall-clock)

2 seeds/arm, 25 min wall-clock each, real Comma-Pile mainmix, fused kernels.

| arm | params | dtype | bs | tok/s | steps | Mtok | wall_s | **held-out BPB** | sd |
|---|---|---|---|---|---|---|---|---|---|
| **R\*** (typed mixture + head_lr c5) | 1.34B | fp32 | 1 | 1881 | 1367 | 2.8 | 1501 | **2.2200** | .038 |
| **B = CMA-best GDN-2** | 1.35B | bf16 | 2 | 8110 | 2932 | 12.0 | 1500 | **1.7653** | .014 |
| **CMA-best m2rnn** | 1.31B | bf16 | 2 | 7290 | 2627 | 10.8 | 1500 | **1.8103** | .006 |

**Token-matched cross-walk** (train loss, nats/tok, read at the common budget = R\*'s
2.7 Mtok — the most R\* could reach in 25 min fp32): **R\* 6.083 > GDN-2 6.011 > m2rnn
5.975.** R\* is behind on the token-matched curve *too*, not only on wall-clock.

**Reading.** At matched wall-clock R\* loses by a large margin (ΔBPB +0.45 vs B) — the
fused mandate puts the typed mixture on the fp32 path, which at the same memory budget
runs batch 1 and sees **4.4× fewer tokens** than the bf16 controls (2.8 M vs 12 M). This
is exactly the §4.5.2 token-vs-wall split the synthesis flagged, and it reproduces the
prior 1.3B wall-clock NO-GOs (`e97-lm-1p3b`, `e97delta-1p3b`: token-win flips to
wall-loss; here R\* does not even win on tokens). The token-matched cross-walk removes
the dtype/throughput confound and R\* is **still** marginally behind both CMA controls at
the common budget — the optimized mixture's training levers do not produce a
sample-efficiency edge over a CMA-best GDN-2 at this scale. The §4.6 **BPB gate fails**
(R\* regresses vs B on held-out BPB, wall- and token-matched).

### 3.2 Capability coverage at 1.3B (per-corner accuracy + JCC)

Matched-step battery (seed 42) on the 1.3B-shaped cell, one task per corner, eval at
train length 128 and extrapolated to 512. JCC = min-corner accuracy.

**eval length 128**

| arm | recall | counting | step_growth | track | **JCC** |
|---|---|---|---|---|---|
| **R\*** | 0.231 | 0.231 | **1.000** | 0.095 | **0.095** |
| **B = CMA-best GDN-2** | **0.970** | **0.274** | **1.000** | **1.000** | **0.274** |
| **CMA-best m2rnn** | 0.090 | 0.209 | 0.509 | 0.044 | 0.044 |

**eval length 512 (extrapolation)**

| arm | recall | counting | step_growth | track | **JCC** |
|---|---|---|---|---|---|
| **R\*** | 0.054 | 0.209 | 0.997 | 0.031 | **0.031** |
| **B = CMA-best GDN-2** | 0.690 | 0.222 | 0.989 | 0.854 | **0.222** |
| **CMA-best m2rnn** | 0.038 | 0.202 | 0.503 | 0.017 | **0.017** |

**Reading.** **CMA-best GDN-2 dominates every corner**; even its *weakest* corner
(counting 0.274 — GDN-2's known architectural soft spot) is **above R\*'s weakest** (track
0.095). R\* matches GDN-2 only on `step_growth` (1.000 — its `nonlin` heads ace the
modular-quadratic cliff) but **collapses on recall (0.231) and track (0.095)** where pure
GDN-2 is near-perfect (0.970 / 1.000). This is the **same sample-efficiency deficit** seen
in the BPB run: at matched compute (matched steps) the typed mixture under the
`head_lr_compute_mult=5` lever learns the nonlinear corner fast but is far slower to reach
the recall/track plateau its `gdn2_recall` heads *can* reach (they hit 0.97 at small
scale) — the compute-class 5× LR starves the 1× recall heads, the recall-regression
interaction OPT_SYNTHESIS §3.3 flagged at small scale, now severe at 1.3B. m2rnn (the
raw-write foil) is the weakest arm on every corner — the substrate, not the write rule,
carries capability.

*Caveat (honest):* the matched-step budget (2–3 k steps) leaves all arms below their
asymptotic accuracy on the hard corners (counting peaks ~0.27 even for GDN-2). The
comparison is fair because every arm gets the **same** budget; the absolute JCC is a
matched-compute coverage number, not a converged ceiling. The R\*-vs-B *gap* is large and
consistent across both eval lengths and with the BPB result, so it is not a step-budget
artifact: R\* is simply less sample-efficient at scale.

---

## 4. Verdict — does optimization (not architecture) move the optimum at scale?

**NULL — optimization does not move the optimum at 1.3B. CMA-best GDN-2 wins.**

The §4.6 decision rule is GO iff R\* clears B = CMA-best GDN-2 on worst-corner JCC
*beyond noise* **AND** does not regress held-out BPB vs B. The **BPB half of the gate
fails decisively and independently of the capability result**:

- **Held-out BPB, wall-matched:** R\* 2.220 vs B 1.765 — a **+0.455 BPB regression**.
- **Held-out BPB, token-matched** (common 2.7 Mtok budget): R\* 6.083 vs B 6.011 nats/tok
  — R\* is behind on the token curve too, so the loss is **not** merely the fp32/bf16
  throughput confound; the optimized mixture has no sample-efficiency edge over a
  CMA-best GDN-2 at this scale.
- Both controls (GDN-2 **and** the raw-write m2rnn foil, 1.810 BPB) beat R\* — the
  optimized mixture does not separate from *either* CMA-best control on held-out LM.

This is the **pre-registered expected outcome** (OPT_SYNTHESIS §4.6): at small scale every
*pure* training lever was sub-Δ\* (per-head-type LR +0.021, placement-knob-LR +0.015,
norm/decay-init +0.008, composed R\* re-run +0.006), and a properly LR-tuned GDN-2 *tied*
the capability-complete mixture at 2.7× fewer parameters. **The convergent-loss null,
established for architecture, extends cleanly to the optimization line at 1.3B:** against
a control that is itself CMA-ES-optimized for the budget (no hobbled baseline — Erik's
directive), the optimized GDN+nonlin mixture's training levers buy no held-out-BPB win.
The levers' defensible value remains what the synthesis claimed — small-scale reliability,
worst-corner robustness, and length-extrapolation — none of which is a scale BPB win.

**The capability gate fails in the same direction** (§3.2), so the two measurements agree:

- **Worst-corner JCC @128:** R\* 0.095 vs B 0.274 — R\* is **−0.179 below** B (does not
  clear the Δ\*=0.03 bar; it regresses).
- **Worst-corner JCC @512 (extrapolation):** R\* 0.031 vs B 0.222 — same direction.
- CMA-best GDN-2 dominates every corner; R\* ties only on `step_growth`. m2rnn is the
  weakest arm everywhere (substrate carries capability, not the raw-write rule).

```
==> NULL: worst-corner JCC does not clear B by Δ*=0.03 (R* 0.095 < B 0.274) AND
    held-out BPB regresses vs B (R* 2.220 > B 1.765, wall- and token-matched).
    Optimization does not move the optimum at 1.3B; the CMA-best GDN-2 wins both gates.
```

---

## 5. Provenance / artifacts

- **Harness:** `experiments/opt_1p3b/{lm_runner,run_lm_matrix,run_capability,aggregate}.py`.
- **Data:** `experiments/opt_1p3b/results/` (LM/BPB JSONs + loss curves),
  `experiments/opt_1p3b/results_cap/` (capability JSONs), `summary.json`,
  `BPB_TABLE.txt`, `JCC_TABLE.txt`, `VERDICT.txt`.
- **Real data:** `/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt`
  (1 TB), tokenizer `p50k_base`.
- **GPUs:** broker-leased (`scripts/gpu_lease.sh`), single-host 8×.
- **Upstream:** OPT_SYNTHESIS.md §3.4 (R\*), §4.2–4.3 (CMA control protocol), §4.5–4.6
  (measurements + decision rule).
