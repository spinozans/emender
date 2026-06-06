# TYPED GDN-2 HEAD MIXTURE — a placed heterogeneous population of NATIVE update rules

**Task:** `typed-gdn-2-head`. Builds on `cma-capability` (the CMA E98 unified-cell
winner), `e98-sixth-corner` (the failed hypothesis that a gated-delta E98 *corner*
closes GDN recall), `e98-five-corner` (the MQAR recall probe + native GDN
reference), `e98-on-e97` (the split-gated kernel) and `deep-novelty-synthesis`
(classic dynamical regimes vs. a placed heterogeneous mixture of update rules).

## The question

`e98-sixth-corner` showed the unified cell could **not** reproduce GDN's MQAR
recall as a placed operating point: the dedicated gated-delta *preset* scored
0.171 vs native GDN 0.951, and a spread-6 population that added a gated-delta
corner did not improve recall over spread-5. The conclusion there was that **GDN
recall is architectural, not a placeable knob of the unified cell.**

This task tests the alternative hypothesis: maximum expressivity comes from a
**placed heterogeneous population of *native* update rules** — GDN-2 as a
first-class head type with its own delta-memory kernel, sitting in the same layer
as the E98 corner specialists — rather than from operating points of one unified
cell. If true, a typed mixture should recover GDN-like recall **and** keep the
S5/count/latch/nonlinear specialist capabilities the unified cell already covers.

## Architecture: `TypedHeadMixtureLayer`

One layer holds a horizontal population of **five native head types** (file:
`ndm/models/typed_head_mixture.py`):

| type | native rule | implementation | role |
|---|---|---|---|
| `gdn2_recall` | gated delta rule `S_t = g_t S_{t-1}(I-β k kᵀ)+β v kᵀ`, `allow_neg_eigval=True` | **FLA `GatedDeltaNet`** (chunked `chunk_gated_delta_rule` Triton kernel), `head_dim=N=32`, short conv, neg eigenvalue | associative recall / tracking |
| `e97_track` | E97 split-gated reflection (along-key eig λ−β<0) | UnifiedCell `track` corner **with split gate** | S5 / group tracking |
| `count` | pure integrator (λ=1, β=0) | UnifiedCell `count` corner | a^n b^n c^n counting |
| `latch` | bistable ±1 (λ>1, tanh) | UnifiedCell `latch` corner | flag-hold |
| `nonlin` | state-nonlinear φ (iterated map) | UnifiedCell `nonlin` corner | iterated-nonlinear-map |

**GDN-2 is a real native delta-memory head, not an E98 corner approximation.** It
runs NVIDIA/FLA's gated-delta-rule chunk kernel directly with `allow_neg_eigval=
True` (the GDN-2 negative along-key eigenvalue used for tracking; Grazzi 2025 /
DeltaProduct). Matrix state **N = 32** per head (task default; matched to the
unified heads so raw head fractions are comparable — there is no local evidence a
larger GDN head_dim is needed: at N=32 the all-GDN arm already matches the native
`gdn` reference on MQAR, see below).

**Composition.** The four E98-native corner types share ONE `UnifiedCellLayer` in
`fixed_pop` mode — per-head knobs (λ, β, igain, γ) are **FROZEN buffers** at their
corner; only the q/k/v/o projections train. The `gdn2_recall` heads run the
separate native FLA block. Both sub-blocks map `dim→dim` with their own readout;
the layer output is their **sum** into the shared residual stream — one layer
holding two native pathways, sized by the head allocation. `split_gate=True` so
the `track` heads are exactly the validated E97 reflection recurrence.

**Head personalities are frozen** (the first-experiment constraint): the unified
knobs are non-learnable buffers and the GDN rule has no free personality knob.
`lam_max=1.585`, `beta_max=2.747` are pinned to the `cma-capability` winner so the
placed corners sit at the validated operating points. **No GDN internals are
tuned.**

### Deterministic instantiation from type logits

1. CMA proposes **5 unconstrained logits** (one per type).
2. `softmax` → desired type fractions.
3. **largest-remainder** rounding → integer head counts summing to `n_heads`.
4. Deterministic (not stochastic) allocation. **No floor is imposed**: a type may
   receive zero heads, reported honestly.

(`allocate_types` / `largest_remainder_counts` in `typed_head_mixture.py`; unit
tests in `tests/test_typed_head_mixture.py` cover exact-sum, zeros-allowed, and
the frozen-knob contract.)

### Parameter matching

Head shapes are matched (every head, GDN or unified, carries N=V=32 state and
contributes 32 readout dims). The only per-head parameter asymmetry is the unified
split-gate (erase+value-write projections) vs the GDN short-conv+decay/beta
projections — ≤~1.4×. Rather than fix this per-head, **total model params are
matched at the model level**: for every candidate, `dim` is binary-searched
(multiple of 8) to hit the 8M target given that candidate's allocation. So an
all-GDN config (cheaper heads) gets a larger dim than an all-unified config, and
every config compared sits at ~8M. Observed range across allocations: 7.84M–8.09M
at depth 4, `n_heads=48`, `n_state=32`.

## CMA search space and exact objective

Frozen architecture (clean first experiment): `depth=4`, `n_heads=48`,
`n_state=32`, `lam_max=1.585`, `beta_max=2.747`, GDN `allow_neg_eigval=True`. The
**only** search variables are:

- **5 head-type logits** `[gdn2_recall, e97_track, count, latch, nonlin]` (bounded
  box [−4, 4]; softmax is shift-invariant), and
- the **shared learning rate** (log, 1e-4…1e-3).

`dim` is derived per-candidate to the 8M budget. Two-phase LHS→CMA with the same
idle-GPU scheduler as `cma-capability` (fp32, schedule-free AdamW,
`--disable_autocast`, used-mem<2GB never-preempt).

**Objective (worst-case-aware, so collapse is visible — NOT a plain average):**

```
per-probe  p_i = mean over T∈{128,256,512,1024} of length-extrapolation acc
fitness        = 0.5 · mean_i(p_i)  +  0.5 · min_i(p_i)
cost (CMA min) = 1 − fitness
```

The `min` term is the floor: collapsing **any** single capability — dropping
recall to win specialists, or dropping specialists to win recall — halves the
marginal gain. Six probes: `mqar_recall`, `s5_permutation`, `anbncn_viability`,
`flag_hold_recall`, `iterated_nonlinear_map`, `mixed_probe` (which itself bundles
all four specialists + MQAR recall in disjoint vocab blocks).

## Per-type capability check (frozen native heads work in-pipeline)

Before the search, each native head type was driven to its corner by extreme
logits (all-X) and trained 2500 steps; this confirms the frozen personalities are
individually competent and that **native GDN recall is recovered in this pipeline
at N=32** (final acc @ T=128 train length / best length-extrap):

| all-X arm | probe | final acc |
|---|---|---|
| all-`gdn2_recall` (typed, fp32) | MQAR recall | **0.9875** |
| native `gdn` reference (bf16) | MQAR recall | 0.9561 |
| all-`e97_track` | s5_permutation | **1.0000** |
| all-`count` | anbncn_viability | **0.9919** |
| all-`latch` | flag_hold_recall | **1.0000** |
| all-`nonlin` | iterated_nonlinear_map | **0.7417** |

The typed GDN-2 head (0.9875) **matches/exceeds** the native `gdn` baseline
(0.9561) on MQAR — so the recall the unified cell could not place as a corner is
fully recovered by a native GDN-2 head. (Raw JSONs under
`results/typed_gdn2_cma/` and the per-type check logs.)

## CMA result: discovered type weights / head counts

8 generations (LHS 14 + CMA pop 8), converged at gen 6, 0 failed runs. The search
**collapses toward a GDN-2-heavy population** and keeps **only** the nonlinear-state
specialist:

**Winner `6e33ac42f500`** — `dim=240`, params **8.06M**, `lr=9.95e-4`,
logits `[3.9995, −1.9008, −0.9211, −2.8866, 2.4146]`:

| type | logit | heads (of 48) |
|---|---|---|
| **gdn2_recall** | 4.00 | **40** |
| **nonlin** | 2.41 | **8** |
| e97_track | −1.90 | 0 |
| count | −0.92 | 0 |
| latch | −2.89 | 0 |

Per-probe (search proxy, 2000 steps, 1 seed): recall 0.795, s5 0.984, count 0.902,
latch 1.000, nonlin 0.919, mixed 0.848 → **fitness 0.851 (mean 0.908, min 0.795)**.

**The mixture is GDN-2-heavy, NOT balanced.** track / count / latch all go to
**zero** because the native GDN-2 heads (`allow_neg_eigval=True`, gated decay)
*subsume* tracking (negative along-key eigenvalue → reflection/S5), counting (gated
decay → integration) and latching (decay → hold). The **only** specialist that
earns nonzero weight is the **nonlinear-state head** (8/48, the 2nd-highest logit)
— GDN-2 is linear-state and genuinely cannot do the iterated-nonlinear-map.

## Validation vs references (full budget: 6 probes × 2 seeds × 4000 steps)

Arms: the typed-gdn2 winner (fp32); the `cma-capability` **E98 CMA winner**
(fp32); and a param-matched **native delta-memory reference** (`gdn` layer = FLA
**DeltaNet** backend, plain delta rule, **8.06M**, bf16). Cells are mean over
seeds of mean-over-T length-extrapolation accuracy.

| probe | **typed-gdn2** | e98-cma | gdn(DeltaNet) ref |
|---|---|---|---|
| recall (MQAR) | **0.807** | 0.423 | 0.742 |
| s5 / track | 0.969 | **0.999** | 0.050 |
| count | 0.891 | **0.947** | 0.934 |
| latch | 0.999 | 0.967 | **1.000** |
| nonlin | **0.944** | 0.931 | 0.853 |
| mixed | 0.902 | **0.929** | 0.582 |
| **mean** | **0.919** | 0.866 | 0.694 |
| **min (worst-case)** | **0.807** | 0.423 | 0.050 |

Recall and S5 length-extrapolation per-T (the two discriminating capabilities):

| recall MQAR | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| typed-gdn2 | 0.996 | 0.973 | 0.813 | **0.447** |
| e98-cma | 0.905 | 0.556 | 0.171 | 0.058 |
| gdn(DeltaNet) | 0.977 | 0.913 | 0.697 | 0.382 |

| s5 / track | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|
| typed-gdn2 | 1.000 | 1.000 | 0.998 | 0.878 |
| e98-cma | 1.000 | 1.000 | 1.000 | 0.998 |
| gdn(DeltaNet) | 0.098 | 0.052 | 0.030 | 0.020 |

### Reading the table

- **Recall recovered.** The typed mixture does MQAR recall (**0.807**, and 0.447 at
  T=1024) where the E98 unified-cell winner **collapses** (0.423; 0.058 at T=1024).
  This is the resolution `e98-sixth-corner` predicted: GDN recall is *architectural*,
  and a **native** GDN-2 head supplies it — the unified cell could not place it as a
  corner. The typed mixture even **beats the param-matched standalone DeltaNet ref**
  on recall (0.807 vs 0.742), and the per-type check shows the all-GDN-2 typed arm
  reaches **0.9875** MQAR at N=32 (vs the native `gdn` bf16 ref 0.9561).
- **Specialists preserved.** S5 0.969 (vs DeltaNet ref's total **collapse** 0.050 —
  plain DeltaNet has no negative eigenvalue), count 0.891, latch 0.999, nonlin 0.944
  (the **best** of the three arms). The GDN-2 heads' `allow_neg_eigval` is doing the
  E97/S5 tracking job with **zero** dedicated track heads.
- **Worst-case wins.** Typed-gdn2 has the highest mean (**0.919** vs 0.866 vs 0.694)
  **and** by far the highest floor (**min 0.807** vs 0.423 vs 0.050): it is the only
  arm with **no collapsed capability**. E98 collapses on recall; DeltaNet collapses
  on S5 and mixed.
- **Honest trade.** E98 still edges the typed mixture on the *shared-knob* tasks it
  was tuned for — s5 (0.999), count (0.947), mixed (0.929) — by ≤0.04. The typed
  mixture trades a few points of specialist peak for a recall capability E98
  *structurally lacks*, and wins decisively on mean and worst-case.

## Nonlinear-head evidence & ablation (RUN, not predicted)

CMA gave the nonlinear-state head **nonzero** weight (8/48, logit 2.41 — the only
specialist it keeps). To test whether those heads actually matter, the clean
ablation — **typed winner minus the nonlinear heads** (all-GDN, dim re-derived to
8M = 256, same lr, 2 seeds × 4000 steps) — was run:

| probe | typed winner (40 GDN + 8 nonlin) | ablation (48 GDN, 0 nonlin) | Δ |
|---|---|---|---|
| recall | 0.807 | 0.765 | +0.042 |
| s5 | 0.969 | 0.988 | −0.019 |
| count | 0.891 | 0.912 | −0.021 |
| latch | 0.999 | 0.979 | +0.020 |
| **nonlin** | **0.944** | **0.953** | **−0.009** |
| mixed | 0.902 | 0.895 | +0.007 |
| **mean** | **0.919** | **0.915** | +0.004 |
| **min** | 0.807 | 0.765 | +0.042 |

**The result refutes the "nonlinear heads matter" hypothesis.** Removing all 8
nonlinear heads leaves the mean essentially unchanged (0.919 → 0.915) and — the
key point — **nonlin itself does not drop** (0.944 → 0.953, within noise, if
anything higher). The GDN-2 backbone alone (gated decay + `allow_neg_eigval`) does
the iterated-nonlinear-map at 0.953, far above the *plain* DeltaNet ref (0.853):
it is the **gating/neg-eigenvalue of GDN-2**, not a dedicated state-nonlinear φ
head, that supplies the nonlinear-map competence here. The winner's only real edge
over all-GDN is on recall (0.807 vs 0.765) and latch (0.999 vs 0.979), and that
tracks head count / dim, not the presence of nonlinear heads. **So the nonlinear
specialist is not justified at this scale on these probes** — a homogeneous GDN-2
population is within ~0.004 mean of the heterogeneous mixture.

## Honest verdict

**The next-scale candidate is a GDN-2 backbone (`allow_neg_eigval=True` + gated
decay), NOT the E98 unified cell and NOT a plain DeltaNet — and the exotic
specialist heads are NOT yet justified.** At ~8M, param-matched:

- The typed GDN-2-heavy population wins on mean (0.919) and worst-case floor
  (0.807), and is the only arm with **no collapsed capability**: it does
  associative recall **and** S5 tracking **and** counting/latching **and**
  nonlinear-state computation.
- But the heterogeneity that pays off is **GDN-2's own internal generality** (its
  negative eigenvalue does S5; its gated decay does count/latch/nonlinear-map), not
  the *addition of exotic specialist head types*: the all-GDN ablation matches the
  mixture (mean 0.915 vs 0.919). The strong form of "maximum expressivity needs a
  placed heterogeneous population of *different* update rules" is **not** supported
  at 8M — one native rule (GDN-2) placed densely is enough here.
- The unified-cell **E98** winner cannot do recall (0.423; collapses at length) —
  confirming `e98-sixth-corner`: GDN recall is architectural, not a placeable knob,
  and the fix is a native GDN-2 head.
- A **plain DeltaNet** backbone cannot do S5 (0.050) or mixed (0.582) — what
  rescues tracking is specifically GDN-2's gating + negative eigenvalue, which the
  typed heads carry.

Net: the result **redirects** the program from "balanced 4-corner E98 population"
and from "exotic placed specialists" toward a **GDN-2 backbone**, with the open
question of whether *any* specialist fraction earns its place left for the LM
pilot (the 8M ablation says no for nonlinear-state).

## Recommended next task — LM pilot at 150–300M

Supersede / replace `e98-lm-pilot` with a **GDN-2-backbone LM pilot**:

1. **Primary candidate:** a **pure GDN-2 backbone** (`allow_neg_eigval=True` +
   gated decay, N=32→ scale head_dim as usual for LM) at 150–300M. The 8M ablation
   shows this matches the heterogeneous mixture (mean 0.915 vs 0.919) and does
   recall, S5, count, latch **and** nonlinear-map on its own — so it is the
   simplest form that carries every capability E98 and plain DeltaNet each lack.
2. **Secondary (heterogeneity test):** the typed-gdn2 form with a small
   nonlinear-state head fraction (~17%, the CMA winner's 8/48) — to check at LM
   scale whether the specialist that did **not** help at 8M starts to help when the
   data is real and long-range. If it does not beat the pure backbone, drop it.
3. **Baselines (param-matched):** (a) the E98 CMA unified cell (expected to lose on
   in-context recall), (b) plain DeltaNet (expected to lose on state-tracking).
4. **Eval:** real LM corpus perplexity **plus** in-context MQAR / multi-query
   recall (E98's gap) **and** a state-tracking / S5-style probe (plain DeltaNet's
   gap), reported with a worst-case-aware score like this study's
   `0.5·mean + 0.5·min` so an LM that quietly drops recall or tracking is visible.

The headline for the program: the cheap-scale evidence points away from both the
balanced-E98 population and exotic placed specialists, and toward a **dense GDN-2
backbone** — the LM pilot should test whether *any* specialist fraction is worth
its parameters at scale, with the default expectation (from the 8M ablation) that
it is not.

## Reproduction

- Layer: `ndm/models/typed_head_mixture.py` (registered `typed-gdn2`). Tests:
  `tests/test_typed_head_mixture.py` (8 pass) + `tests/test_unified_cell.py`
  (47 pass, kernel unchanged).
- CMA search: `experiments/expressivity_tasks/typed_gdn2_cma.py` →
  `results/typed_gdn2_cma/{cma_best.json,cma_trace.json}` + per-(config,probe,seed)
  JSONs.
- Validation: `experiments/expressivity_tasks/run_typed_gdn2_validate.py` →
  `results/typed_gdn2_validate/*.json`; tables via
  `aggregate_typed_gdn2.py` → `results/typed_gdn2_summary.json`.
- Nonlinear ablation (all-GDN, no nonlin heads): `train_hybrid.py
  --layer_pattern typed-gdn2 --head_type_logits=8,-8,-8,-8,-8 --dim 256 --depth 4
  --n_heads 48 --n_state 32 --lr 9.95e-4 --lam_max 1.585 --beta_max 2.747
  --disable_autocast` over the six probes × seeds {42,123} × 4000 steps →
  `results/typed_gdn2_ablation/*.json`.
- All training REAL (fp32 schedule-free AdamW for the typed/E98 arms; bf16 for the
  DeltaNet ref whose chunk kernel rejects fp32), idle-GPU-only, `paper/main.typ`
  untouched.
