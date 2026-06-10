# OPT-INITSPREAD — Lever 2 (init-spread / placement) probe results

**Task:** `opt-initspread` · **Role:** Programmer · **Line:** OPTIMIZATION-NOT-ARCHITECTURE
(`OPT_SPEC.md` §5.2) · **Date:** 2026-06-10

This probe tests **Lever 2** of the optimization line: does *tuning the placement*
— how heads are allocated across the typed corners, with the spread-init's
knob-LR companion — let **one** trained instance of the frozen GDN+nonlin
within-layer substrate hold **all four capability corners simultaneously** (the
Joint Capability Coverage, JCC) better than a fairly-tuned GDN-2 control?

> **Headline — GO (placement is a real lever).** On the frozen `typed-gdn2`
> substrate, the **house placement** (50 % `gdn2_recall` + 25 % counting
> (`refit-del`) + 25 % `nonlin`, neg-eigval on, knob-LR 20×) holds **all four
> corners simultaneously** at **JCC = 0.959** (corners-held 4/4), beating the
> *strongest* GDN-2 control (`B` @ its best LR 1e-3, **JCC = 0.892**) by
> **ΔJCC = +0.066 ≥ Δ\* = 0.03** at convergent loss — the gain concentrated on
> **counting**, GDN-2's documented weak corner. The win is robust: `house_klr1`
> (+0.052) and `recallheavy_klr20` (+0.060) clear the bar too.
>
> **Placement is the lever; knob-LR is a small companion.** The placement
> contribution is **≈ +0.31 JCC** (house 0.944 vs the naive uniform-spread 0.634);
> the knob-LR companion adds only **+0.015** (house_klr1 0.944 → house_klr20
> 0.959, at the edge of seed noise). This matches the SPECIALIZATION_STUDY
> "placement not pressure" finding — but it is *specific* placement: the **naive
> uniform / skew / center "spread everything" placements COLLAPSE the track
> corner** (s5 ≈ 0.63–0.65 vs house 1.00), *converged*, across all 3 seeds. Track
> capability scales with the `gdn2_recall` (gdn-neg) allocation (50 % → 1.00,
> 25 % → 0.65, 17 % → 0.63); the dedicated `e97_track` heads do not substitute.
>
> This is a **per-lever GO** at small scale: a better *placement* of the frozen
> substrate beats the convergent-loss GDN-2 incumbent on joint coverage. (The
> line-level verdict awaits `opt-synth`'s composition + the 1.3B validation
> against CMA-best controls, OPT_SPEC §5.5/§6.)

---

## 1. Setup (shared harness, OPT_SPEC §2–§3)

**Frozen substrate `M`.** The `typed-gdn2` layer (`ndm/models/typed_head_mixture.py`)
— `dim 256`, `n_heads 32`, `n_state 32`, `depth 4`, `mlp_ratio 2.0`,
`--gdn_allow_neg_eigval 1`, fp32 (`--disable_autocast`), schedule-free AdamW
(`weight_decay 0.01`, betas `(0.9, 0.95)`). Identical shape to the TTT battery so
control numbers cross-check. The substrate's function class is **frozen**; the
probe varies only the training regime around it.

**The placement lever.** On `typed-gdn2`, head placement is the
`--head_type_logits` allocation across the 9 typed slots
`[gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell, e97_raw,
e97_delta, refit]`. This is the typed-substrate realization of the
SPECIALIZATION_STUDY spread-init: the UnifiedCell sub-block's `corner_mixture` is
**auto-derived** from the head allocation (`typed_head_mixture.py:373`), so the
placement lever *is* the head-type allocation. The knob-LR companion
(`--knob_lr_mult`) is swept as the second axis (1× vs 20×).

**Counting head = `refit-del`.** The counting/step-growth slice uses the `refit`
slot (idx 8) with `--refit_has_mom 0` — the momentum-OFF gated-delta special case
(== `e97_delta` exactly, per `ttt-capability`). OPT_SPEC §2.2 explicitly sanctions
"e97_delta/refit-del for counting". `refit-del` is fp32-clean; the `e97_delta`
slot (idx 7) crashes under `--disable_autocast` (its fused split-edit kernel
requires bf16 input while its projections stay fp32 — a real bug in the fused
path under fp32, avoided here, not worked around with a kernel change).

**Battery (OPT_SPEC §3.1).** Scored corners and witnesses:

| corner | witness task(s) | eval lengths |
|---|---|---|
| recall | `mqar_recall` | 128/256/512 |
| counting | `modular_counter` K=5  +  `dyck_depth_unbounded` | 128/256/512 |
| step-growth | `modular_quadratic` p=64 | 128/256/512 |
| track | `s5_permutation` | 128/256/512 |
| (control) | `parity` | 128/256 (sanity only) |

Train T=128; JCC computed per eval length and length-averaged. 3 seeds
{42,123,456}.

**JCC metric (OPT_SPEC §1.3).** Per-corner held ratio `r_c = clamp(acc_c / S_c,
0, 1)` against **frozen specialist ceilings** `S_c` (the best accuracy any single
control specialist reaches on corner c; written to `opt_ceilings.json`). Headline
**`JCC = min_c r_c`** over the scored corners — the worst-corner ratio, the only
aggregate that cannot be gamed by trading a hard corner for an easy one.
`corners_held` (r_c ≥ τ=0.95) and harmonic-mean reported alongside.

**Decision rule (OPT_SPEC §1.4).** A regime `R` is a **real win** iff
`JCC(R) − JCC(B) ≥ Δ*` with `Δ* = max(0.03, 2·SE_seed(B))`. Because JCC = `min_c`,
a positive ΔJCC *is* a positive gain on the worst corner — the min-aggregate makes
"no corner-trading" automatic.

## 2. Convergence (OPT_SPEC §1.5 — non-negotiable)

Every run trained to a fixed generous step budget; a **convergence certificate**
= relative improvement of the smoothed train loss over the final 20 % of steps
(`(L_80% − L_final)/L_80%`) is reported per run, **converged iff < 0.02** OR the
smoothed final loss is below 0.05 (a task trained to ≈zero loss has plateaued; the
relative ratio is noise-dominated on near-zero losses).

At the initial 12 000-step budget, `modular_quadratic`, `mqar_recall`,
`s5_permutation`, `dyck_depth_unbounded` and `parity` **plateaued** (smoothed
train loss 0.000) for every arm, but **`modular_counter` K=5 was still climbing**
(B_gdn2 0.973, house 0.960 — loss certificate ≈ 0.5). Per the spec's §3.2 "raise
the budget" clause, **`modular_counter` was re-run to 32 000 steps** for all arms
×3 seeds; the 12 000-step counter runs are archived under
`results_opt_initspread/_superseded_counter12k/`. All scored corners are at
convergent loss in the final table.

## 3. Controls (OPT_SPEC §4.1)

- **`B` = GDN-2 fair-default** (`fla-gdn`, `--gdn_allow_neg_eigval 1`), **LR-screened
  over {3e-4, 5e-4, 1e-3}** at 3 seeds and reported at its **best LR** — "reasonably
  tuned, NOT hobbled". (The 5e-4 default under-solves `modular_quadratic` at 0.87;
  1e-3 solves it at 0.999 — the screen matters and B is reported at its optimum.)
- **`B₂` = substrate-default** = the house placement trained with the default
  regime (`knob_lr_mult 1`); isolates the lever's pure contribution
  (`JCC(R) − JCC(B₂)`) from the incumbent contribution (`JCC(R) − JCC(B)`).
- **Specialist/ceiling arms:** `B` (recall/track owner via gdn-neg) and `alldelta`
  (all-`refit-del`; counting/step-growth owner) supply the frozen ceilings `S_c`.

**No new kernels.** Existing fused cells only; the sole new code is the probe
runner/aggregator (`run_opt_initspread.py`, `aggregate_opt_initspread.py`).

## 4. Arms (placement family)

| arm | placement (head allocation, 32 heads) | knob-LR |
|---|---|---|
| `B_gdn2` | GDN-2 control (`fla-gdn`, gdn-neg) | — |
| `alldelta` | all `refit-del` (counting/step-growth specialist) | 1× |
| `spread_center` | uniform over the 6 fp32-safe corners | 1× |
| `house_klr1` (=B₂) | 50 % `gdn2_recall` + 25 % `nonlin` + 25 % `refit-del` | 1× |
| `house_klr20` | same placement | 20× |
| `uniform4_klr20` | 25 % each: recall / track / nonlin / refit-del | 20× |
| `skew_klr20` | CMA difficulty-skew (starve recall, feed hard corners) | 20× |
| `recallheavy_klr20` | 62.5 % recall + 18.75 % nonlin + 18.75 % refit-del | 20× |

## 5. Results — JCC leaderboard (3 seeds, at convergent loss)

Frozen specialist ceilings `S_c` (`opt_ceilings.json`, hash `8ccc626bf668`):
recall **0.955** (B@1e-3), counting **0.890** (alldelta), step-growth **0.999**
(B@1e-3), track **0.999** (B@1e-3).

Baseline `B` = GDN-2 at its **strongest** LR (1e-3, selected as B's highest-JCC
LR over the {3e-4, 5e-4, 1e-3} screen — the hardest incumbent): **JCC 0.892**,
SE_seed 0.009, **Δ\* = max(0.03, 2·SE) = 0.03**.

| arm | JCC=min_c | held | hmean | recall | counting | step-gr | track | conv | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **`house_klr20`** | **0.959** | 4/4 | 0.985 | 0.98 | 0.96 | 1.00 | 1.00 | ok | **WIN +0.066** |
| `recallheavy_klr20` | 0.952 | 3.7 | 0.985 | 1.00 | 0.95 | 1.00 | 0.99 | ok | **WIN +0.060** |
| `house_klr1` (=B₂) | 0.944 | 3.3 | 0.980 | 0.98 | 0.94 | 1.00 | 1.00 | ok | **WIN +0.052** |
| `B_gdn2` (incumbent) | 0.892 | 3.0 | 0.970 | 1.00 | 0.89 | 1.00 | 1.00 | — | — |
| `uniform4_klr20` | 0.650 | 2.0 | 0.857 | 0.96 | 0.93 | 1.00 | **0.65** | ok | LOSE −0.243 |
| `skew_klr20` | 0.648 | 1.7 | 0.852 | 0.97 | 0.89 | 1.00 | **0.65** | ok | LOSE −0.244 |
| `spread_center` | 0.634 | 1.7 | 0.844 | 0.93 | 0.92 | 1.00 | **0.63** | ok | LOSE −0.258 |
| `alldelta` (specialist) | 0.364 | 2.0 | 0.589 | **0.47** | 0.99 | 1.00 | **0.47** | — | LOSE −0.528 |

Per-corner numbers are seed × length-averaged held ratios `r_c`; JCC is the mean
over seeds of the per-seed worst-corner ratio. `conv` is the §1.5 convergence gate
over the scored witnesses (B_gdn2's smoothed counter train-loss sits just at the
0.05 floor — see §2; its T=512 extrapolation is the variance, not the loss).
Per-(arm,seed) shared-schema rows incl. `per_length_ratio` are in
`results_opt_initspread/JCC_ROWS.jsonl`.

**Lever decomposition** (vs B₂ = `house_klr1`, the placement-with-default-regime):
- **placement** contribution: `house_klr1` 0.944 − `spread_center` 0.634 = **+0.310**.
- **knob-LR** contribution: `house_klr20` 0.959 − `house_klr1` 0.944 = **+0.015**
  (within ~1× the seed band; small but positive, acting on the counting corner).
- vs the incumbent `B`: `house_klr20` − `B` = **+0.066** (the §1.4 win).

**Cross-probe consistency.** The +0.066 win over GDN-2 is carried primarily by the
**substrate** (the mixture's dedicated `refit-del` counting heads that GDN-2 lacks),
not by the lever per se — the same conclusion the sibling probes reached
(`opt-headlr`: substrate +0.055 vs pure-lever +0.021; `opt-norm`: lever NULL, "the
win is the substrate"; `opt-minimal`: GDN-2's step-growth weakness is itself an LR
artifact, fixed 5e-4→1e-3, which is why `B` here is reported at 1e-3). This probe's
*distinct* contribution is the **placement-sensitivity** result: a *wrong* allocation
(uniform/skew/center) destroys the track corner (−0.31), so a good placement is
*necessary* to realize the substrate's coverage — the knob-LR companion adds only a
marginal +0.015. Two independent gotchas also reproduced across probes: the
`e97_delta` fp32/bf16 fused-kernel crash, and the relative loss-certificate being
noise-dominated near zero loss (floored here at smoothed train-loss < 0.05).

## 6. Findings

1. **Placement is a real lever on the worst corner (GO).** The house placement
   adds a dedicated counting slice (`refit-del`) that GDN-2 lacks, lifting the
   **counting** corner — GDN-2's documented weak arm (0.89) — to 0.96 while keeping
   recall/track/step-growth at ceiling. Because JCC = worst-corner, this raises the
   headline from 0.892 to 0.959 (+0.066 ≥ Δ\*). The win reproduces across three
   placement arms (house_klr1/klr20/recallheavy) and is **beyond seed noise**
   (SE 0.009, Δ\* 0.03).

2. **Placement dominates; knob-LR is a small companion.** Placement is worth
   **≈ +0.31 JCC** (house 0.944 vs the naive uniform-spread 0.634). The knob-LR
   companion (20×) adds only **+0.015** at convergence — positive (it nudges the
   counting corner up), but at the edge of the seed band. This refines the
   SPECIALIZATION_STUDY "placement not pressure" reading
   ([[unified-cell-pressure-vs-placement]]): on the typed substrate the *allocation*
   carries the effect, the knob-LR is secondary. (NB: at the pre-convergence
   12 000-step budget the two looked identical — the small knob-LR counting lift
   only emerges once `modular_counter` is trained to plateau, §2.)

3. **Spread is *specific*, not generic — the key negative result.** The naive
   **uniform / skew / center** placements **collapse the track corner** (s5 ≈
   0.63–0.65 vs house 1.00), *converged* (smoothed train loss 0.000 — they fit
   T=128 and fail to length-extrapolate), consistent across all 3 seeds. Track
   capability scales with the `gdn2_recall` (gdn-neg) allocation: **50 % → 1.00,
   25 % → 0.65, 17 % → 0.63**; the dedicated `e97_track` heads do **not**
   substitute. "Spread everything uniformly" is the *wrong* reading of spread-init
   — the right placement is **recall-heavy enough** (≥ ~50 % gdn-neg, for recall
   AND track) **with a dedicated counting slice**. `recallheavy` (62.5 % recall)
   also wins, so the operating window is "enough recall + some counting", not a
   knife-edge.

4. **`alldelta` is a counting specialist, not a coverage cell.** It aces
   counting/step-growth but goes recall-blind (0.47) and track-weak (0.47) — the
   documented recall↔counting trade-off the JCC `min` is designed to expose, and
   the reason a *mixed placement* (not the all-delta substrate) is what covers all
   corners.

5. **Honest scope.** This is a per-lever GO at the small (dim-256) scale on the
   capability battery only. It does **not** yet establish the line-level claim:
   `opt-synth` must compose this placement with the other levers and re-test, and
   `opt-1p3b` must clear the §1.4 bar at 1.3B against **CMA-best** GDN-2 (not the
   default-LR-screened `B` used here) and held-out BPB (OPT_SPEC §5.5). The
   placement to carry forward is **`house_klr20`** (50 % gdn2_recall + 25 %
   refit-del + 25 % nonlin, neg-eigval, knob-LR 20×); `recallheavy` is the
   robustness check that the window is wide.

## 7. Reproduce

```bash
eval "$(scripts/gpu_lease.sh 4)"          # broker lease (never hand-pick GPUs)
# full scored battery (12000 steps; modular_counter at 32000 — see §2):
python experiments/expressivity_tasks/run_opt_initspread.py \
    --tasks mqar_recall dyck_depth_unbounded modular_quadratic s5_permutation
python experiments/expressivity_tasks/run_opt_initspread.py \
    --steps_override modular_counter:32000 --b_full_lr_screen --no_b_lr_screen \
    --tasks modular_counter
# aggregate JCC + freeze ceilings + emit the shared-schema rows:
python experiments/expressivity_tasks/aggregate_opt_initspread.py --write_ceilings
```

Artifacts: `experiments/expressivity_tasks/{run,aggregate}_opt_initspread.py`,
`results_opt_initspread/{*.json,JCC_ROWS.jsonl}`, `opt_ceilings.json`.
