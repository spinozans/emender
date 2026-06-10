# OPT_NORM_RESULTS — Lever 3: normalization placement, gate-bias init, decay/A_log/dt_bias init

**Task:** `opt-norm` · **Role:** Programmer · probe of the OPT_SPEC.md line (§5.3)
· small scale (dim 256 / 32 heads / N32 / depth 4 / mlp 2.0), fp32, schedule-free
AdamW, 3 seeds (lever arms 2 seeds), shared §3 battery + JCC metric (§1.3).

## TL;DR (the verdict)

**The parameterization lever is a NULL.** Holding the function class frozen, no
normalization/gate/decay **init** knob moves Joint Capability Coverage (JCC) beyond
the pre-registered noise band (`Δ* = 0.03`) **once the substrate is held fixed**
(`ΔvsB2`). The single best axis — **decay/A_log/dt_bias init → "slow" (retention→1)**
— is directionally correct and confirms the §5.3 hypothesis mechanism (recall AND
counting both want near-1 retention), but its pure contribution over the
substrate-default is **+0.0084 (combined_best, 3 seeds) to +0.021 (decay_slow,
2 seeds)** — **below `Δ*`**. The CMA-derived cell defaults (`lam_max=1.585`,
`beta_max=2.747`, gate default, head-norm on) are already near-optimal; every other
knob is neutral-to-harmful. **This extends the convergent-loss null (OPT_SPEC §0)
from architecture to the gate/decay/norm-init optimization lever.**

The probe forwards **`combined_best` = decay-init→slow (retention→1)** as its
candidate, flagged as a marginal/NULL lever (the win vs incumbent GDN-2 is the
*substrate*, not the lever — see §3).

## 1. Substrate, arms, controls

**Substrate `M` (frozen; OPT_SPEC §2.2 house mixture, fp32-clean variant).**
50% `gdn2_recall` (`allow_neg_eigval` → recall+track) + 25% `nonlin` + 25%
`refit-del`. `head_type_logits = 30,0,0,0,29.30685,0,0,0,29.30685` → exactly
16/8/8 heads on 32. **Why refit-del, not e97_delta:** §2.1 mandates fp32
(`--disable_autocast`); the `e97_delta` fused split-edit Triton kernel rejects fp32
(dtype-mismatch crash — the autocast/bf16 gate). `refit` with momentum **off** is
the gated-delta special case = the §2.2-named counting substrate the TTT battery
validated in fp32 ([[ttt-capability-convergent-null]]). It is the same algorithmic
write rule, fp32-clean.

**Lever knobs (NEW plumbing — init/placement only, NO kernels, function class
unchanged; `ndm/models/typed_head_mixture.py`):**
- `decay_init ∈ {default, slow, fast}` (+ `decay_init_delta`): offsets every
  delta-memory `A_log`/`dt_bias` in the layer. `slow` → `exp(A_log)·softplus(·)`
  small → retention `r→1`; `fast` → `r→0`.
- `gate_bias_init ∈ {None, +2, −2}`: bias on the silu output-gate pre-activation of
  the FLA-path heads (replaces `g_proj` with a bias-enabled Linear, weight copied;
  gate FUNCTION `silu(Wx+b)` unchanged, only `b`'s init moves).
- `unified_head_norm ∈ {on, off}`: RMSNorm head-norm placement on the unified
  (count/nonlin/track) heads.
- `lam_max`, `beta_max`: existing flags (decay-eigenvalue cap; reflection depth).

**Controls (OPT_SPEC §4.1):** `B` = GDN-2 fair-default (`fla-gdn`,
`allow_neg_eigval`, **LR sanity {3e-4, 5e-4, 1e-3}; best reported as B** — not
hobbled). `B2` = the house mixture trained with the default regime (no lever) —
isolates the lever's pure effect. `spec_refit` = all-`refit-del` (counting/
step-growth specialist → frozen ceilings).

**Battery (8 tasks, REAL generators):** recall `mqar_recall`; counting
`modular_counter K5` / `dyck_depth_unbounded` / `anbncn_viability`; step-growth
`modular_quadratic K64` / `iterated_nonlinear_map`; track `s5_permutation`; sanity
`parity`. Train T=128, eval-extrapolate {128,256,512}. **272 real runs.**

**Frozen ceilings `S_c`** (`opt_ceilings.json`, hash `74352db2810735da`): recall
0.9517 (B, LR=1e-3), counting 0.8962 (`spec_refit`), step_growth 0.9715
(`spec_refit`), track 0.9997 (B). LR sanity picked **lr=1e-3** as B (recall+track
score 0.9757 vs 0.9712/0.9714 at 5e-4/3e-4).

## 2. JCC leaderboard (worst-corner ratio `min_c r_c`, seed-averaged)

`B (GDN-2)` JCC = **0.9243**, bottlenecked on **counting (0.924)** — its documented
weakness (recall/step/track all ≈1.0). `B2 (house)` JCC = **0.9728**.
`SE_seed = 0.000` → `Δ* = max(0.03, 2·SE) = 0.03`.

| arm | JCC | ΔvsB | ΔvsB2 | held | recall | count | step | track | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **decay_slow** (2s) | 0.9940 | +0.0697 | **+0.0212** | 4 | 1.000 | 0.996 | 0.994 | 1.000 | WIN* |
| **combined_best** (3s) | 0.9812 | +0.0570 | **+0.0084** | 4 | 1.000 | 0.981 | 1.000 | 0.989 | WIN* |
| decay_slow3 (δ=3) | 0.9742 | +0.0499 | +0.0013 | 4 | 1.000 | 0.974 | 1.000 | 1.000 | WIN* |
| B2_default (house) | 0.9728 | +0.0486 | 0.0000 | 4 | 0.973 | 0.991 | 1.000 | 1.000 | — (B2) |
| lam_1.3 | 0.9662 | +0.0419 | −0.0066 | 4 | 0.979 | 0.967 | 1.000 | 1.000 | WIN* |
| beta_1.5 | 0.9559 | +0.0316 | −0.0170 | 4 | 0.974 | 0.962 | 1.000 | 1.000 | WIN* |
| lam_1.0 | 0.9524 | +0.0281 | −0.0205 | 4 | 0.955 | 0.959 | 1.000 | 1.000 | NULL |
| gate_open (+2) | 0.9517 | +0.0274 | −0.0212 | 4 | 0.970 | 0.952 | 1.000 | 1.000 | NULL |
| beta_2.0 | 0.9348 | +0.0105 | −0.0380 | 3 | 0.987 | 0.935 | 1.000 | 1.000 | NULL |
| gate_closed (−2) | 0.9284 | +0.0041 | −0.0444 | 3 | 0.928 | 0.967 | 1.000 | 0.996 | NULL |
| norm_off | 0.9278 | +0.0035 | −0.0451 | 3 | 0.928 | 0.963 | 0.987 | 1.000 | NULL |
| **B_gdn (GDN-2)** | 0.9243 | 0.0000 | −0.0486 | 3 | 1.000 | 0.924 | 1.000 | 1.000 | (B) |
| decay_fast | 0.8124 | −0.1118 | −0.1604 | 2 | 0.862 | 0.916 | 1.000 | 1.000 | LOSE |
| spec_refit (ceil) | 0.1277 | — | — | 2 | 0.128 | 0.997 | 1.000 | 0.411 | (specialist) |

*WIN = clears the §1.4 bar **vs the incumbent GDN-2**. See §3 — these "wins" are
dominated by the *substrate*, not the lever; the lever's pure effect (`ΔvsB2`) is
the honest measure and is **sub-`Δ*` for every arm**.

## 3. The B-vs-B2 decomposition (the crux)

The §1.4 GO/NULL bar is "vs B = GDN-2". `decay_slow`/`combined_best` clear it
(+0.057…+0.070). **But so does the bare house mixture `B2` (+0.049)** — because
`B2` simply *has* counting heads (`refit-del`) that GDN-2 lacks. That is a
**substrate** difference (an architecture move, out of scope for opt-norm), not the
opt-norm parameterization lever.

The lever's **pure contribution is `ΔvsB2`** (OPT_SPEC §4.1: "`JCC(R) − JCC(B2)` is
the lever's pure contribution"). On that honest axis:

- **decay-init→slow is the ONLY positive knob** (+0.0084 at 3 seeds, +0.021 at 2);
  **both estimates are below `Δ* = 0.03` → NULL.**
- `lam`, `beta`, `gate-bias`, `norm-placement` deviations are all **≤ 0** vs the
  defaults → the **CMA-tuned defaults are already at the optimum** for these axes.
- `decay_fast` (retention→0) is **strongly harmful (−0.160)**: it craters recall
  (0.86) and counting (0.92).

**Mechanism (confirms the §5.3 hypothesis, sub-threshold effect size).** The one
load-bearing parameterization signal is the **decay/retention direction**: pushing
retention →1 helps (recall needs to hold the binding; the mod-k counter needs to
hold its running residue), pushing it →0 hurts both. `decay_slow3` (δ=3.0) shows
**no headroom beyond δ=2.0** (+0.0013) — the retention basin saturates. So the
hypothesis "the basin a corner converges to is set at init by the decay" is
*directionally* validated, but the magnitude — once the substrate is fixed — does
not clear the noise band.

## 4. Length-extrapolation (train T=128; the algorithm-vs-memorization gradient)

| arm | corner | T=128 | T=256 | T=512 |
|---|---|---|---|---|
| B_gdn | recall | 0.998 | 0.975 | 0.882 |
| | counting | 0.994 | 0.855 | **0.476** |
| | step_growth | 0.979 | 0.975 | 0.973 |
| | track | 1.000 | 1.000 | 0.999 |
| B2_default | recall | 1.000 | 0.985 | 0.793 |
| | counting | 0.998 | 0.930 | 0.624 |
| | track | 1.000 | 1.000 | 0.998 |
| combined_best | recall | 1.000 | 0.995 | 0.898 |
| | counting | 0.991 | 0.930 | 0.612 |
| | track | 1.000 | 1.000 | 0.966 |
| decay_slow | counting | 0.999 | 0.946 | 0.633 |

The lever's clearest *relative* benefit is at extrapolation: at T=512 GDN-2's
counting collapses to **0.476** while the house mixture + decay-slow hold
**0.61–0.63**. Again, most of that gap is the counting-head substrate; decay-slow
adds a small, sub-`Δ*` increment on top.

## 5. Convergence (OPT_SPEC §1.5)

The hard counting/track corners (`modular_counter`, `s5_permutation`) were trained
to **16000 steps** (the others plateau within 8000) after the 8000-step pass showed
`modular_counter` still climbing (0.5→0.82, |Δacc|≈0.16). **Convergence gate:
accuracy-plateau** (|Δacc| over the final 20% < 0.02) — the spec's relative-loss
certificate `(L80−Lf)/L80` is a **loss-to-zero artifact** on these exact-algorithm
tasks (loss → 0 keeps relative improvement large even at saturated accuracy), so it
flagged ~every run spuriously; the accuracy-plateau gate is the faithful realization
of §1.5's "compare plateaus, not progress". Residual non-convergence is concentrated
on `modular_counter` for a few seeds even at 16000 (and `spec_refit`, which keeps
climbing on counting as a specialist) — **honestly flagged**; the JCC comparison is
at matched 16000-step compute, so the lever-vs-control ranking is fair.

## 6. Forwarded candidate & feed to `opt-synth`

- **Best parameterization config:** `decay_init=slow` (retention→1, δ=2.0) on the
  §2.2 house mixture, all other knobs at the CMA-tuned defaults. Emitted as the
  `combined_best` arm (3 seeds).
- **Per-lever verdict (§6.3):** **NULL** — the gate/decay/norm-init lever does not
  clear `Δ*` over the substrate-default. The defaults are near-optimal; the only
  signal is the (sub-threshold, saturating) retention direction.
- **For composition (§6.4):** forward `decay_init=slow` as a "free, harmless,
  directionally-correct" default to stack with the other probes' winners; do **not**
  carry `lam<1.585`, `beta<2.747`, gate-bias, or `norm_off` (all ≤0 here).
- **Schema:** per-(arm,seed) rows in `results_opt_norm/JCC_ROWS.jsonl` (probe
  `opt-norm`, frozen ceilings hash `74352db2810735da`), ready for the unified
  leaderboard. `B` appears in all probes as the harness-consistency check.

## 7. Caveats / scope

- **fp32 substrate** uses `refit-del` (not `e97_delta`) for the counting slice — the
  fp32-clean realization of the same gated-delta write rule (§1); the fused
  split-edit kernel is bf16-only.
- **Norm-placement axis** is the reachable subset: `unified_head_norm` on the
  unified heads. FLA-GDN's internal q/k normalization and the `o_norm`
  (FusedRMSNormGated, post-state) are baked into the fused kernel and not toggleable
  without a kernel change ("no new kernels"); reported as not-swept.
- **gate-bias** is applied to the FLA-path gates (the recall backbone — the
  hypothesis target). silu (not sigmoid) gating; +2 opens, −2 closes.
- Small scale only; the 1.3B validation (CMA-best controls) is `opt-1p3b`'s job.

## 8. Reproduce

```bash
eval "$(scripts/gpu_lease.sh 4)"
python experiments/expressivity_tasks/run_opt_norm.py --slots_per_gpu 4   # full battery
python experiments/expressivity_tasks/compute_opt_ceilings.py             # freeze S_c
python experiments/expressivity_tasks/aggregate_opt_norm.py               # JCC + verdict
```
Resumable (per-job `.json` skip + atomic `.claim` so multiple instances cooperate).
Artifacts: `run_opt_norm.py`, `opt_norm_common.py`, `compute_opt_ceilings.py`,
`aggregate_opt_norm.py`, `opt_ceilings.json`, `results_opt_norm/JCC_ROWS.jsonl`.
