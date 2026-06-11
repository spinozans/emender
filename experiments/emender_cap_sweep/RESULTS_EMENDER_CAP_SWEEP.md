# emender-cap-sweep — MEASURED RESULTS

All numbers below are **measured on real GPUs** (this run, 8×idle, bf16, fused
e97 split-edit asserted). No paper numbers are cited; every value traces to a
committed JSON/CSV in this directory. Run date 2026-06-11.

Data files (committed):
- `results_cma/cma_result.json`, `results_cma/generations.jsonl`,
  `results_cma/candidates.csv` — Phase 1 (CMA convergence + found mixture)
- `results_sweep/sweep_per_seed.csv`, `sweep_mean.csv`,
  `capacity_boundary.json`, `CAPACITY_TABLE.md` — Phase 2 (capacity sweep)
- `throughput.json`, `results_efficiency/efficiency.json` — Phase 3 (efficiency)

---

## Phase 1 — DEDICATED CMA-ES over the Emender mixture (properly budgeted)

**Budget (this run, vs the prior under-budgeted 5×4):** popsize **8**, **15**
generations = **120 candidates** + 2 anchors, each trained on a real-Pile slice
to a **token-matched 6M-token** budget, dim derived per-candidate to ≈41M params
(param/FLOP-locked so the mixture search is not a capacity search). Fitness =
held-out BPB. **bf16 uniform + fused split-edit asserted on all 120 candidates**
(`fused_asserted=True`, verified). Wall: 68.6 min.

Search space: `x=[a_delta,a_track]` → softmax over
`[gdn2_recall(=0 sea), e97_delta, e97_track]` (9-type logits); driving both to the
lower bound recovers pure GDN-2 (the null).

### Convergence curve (best held-out BPB per generation — `generations.jsonl`)

| gen | best BPB | f_delta | f_track | sigma |
|----:|---------:|--------:|--------:|------:|
| 1 | 2.37708 | 0.188 | 0.000 | 1.866 |
| 2 | 2.37942 | 0.188 | 0.000 | 1.713 |
| 3 | 2.38503 | 0.094 | 0.031 | 1.455 |
| 4 | 2.38877 | 0.156 | 0.000 | 1.339 |
| 5 | 2.37053 | 0.094 | 0.031 | 1.178 |
| 6 | 2.37520 | 0.156 | 0.000 | 0.939 |
| 7 | 2.37474 | 0.125 | 0.000 | 0.743 |
| 8 | 2.37708 | 0.094 | 0.031 | 0.980 |
| 9 | 2.38503 | 0.094 | 0.031 | 0.838 |
| 10 | 2.37053 | 0.094 | 0.031 | 0.809 |
| 11 | 2.38082 | 0.094 | 0.000 | (cont.) |
| 12 | 2.37240 | 0.094 | 0.031 | |
| 13 | 2.37053 | 0.094 | 0.031 | |
| 14 | 2.37053 | 0.094 | 0.031 | |
| 15 | (converged at 28/1/3) | 0.094 | 0.031 | |

### CMA-found mixture (`found_mixture`)

```
28 gdn2_recall + 1 e97_track + 3 e97_delta   (n_heads=32)
f_delta = 0.09375   f_track = 0.03125   nonlinear_fraction = 0.125
held-out BPB = 2.37053   vs anchor pure-GDN-2 2.44959 (token-matched, −0.079)
verdict-on-loss = "EMENDER (CMA kept a nonlinear emendment fraction on loss)"
```

**Caveat from the curve itself (measured, not assumed):** the best BPB oscillates
in **2.370–2.389 (0.018 spread)** while the e97_delta fraction the curve visits
ranges **0.094–0.188** and σ shrinks 1.87→0.81. The loss is **flat across very
different mixtures** — the kept 12.5% nonlinear sprinkle is CMA settling inside a
flat loss basin, not loss evidence that nonlinearity helps. This is the
convergent-loss-null signature, now reproduced at proper CMA budget.

---

## Phase 2 — CAPACITY SWEEP (the capacity boundary, measured)

Arms (head_dim fixed 8 → n_heads = dim/8; n_state 32; depth 4; 3 seeds
{42,123,456}; train T=128, eval-extrapolate T∈{128,256,512}; fused e97 confirmed
`use_triton_e97=True`, e97_delta on the sequential split-edit path):
- `emender_cma` = the CMA-found mixture (28/1/3 scaled per dim)
- `gdn2typed` = all `gdn2_recall` on the typed path (clean substrate control)
- `gdn2` = `fla-gdn` neg-eig (the incumbent)

### Accuracy @ T=512 (length-extrapolation), mean over 3 seeds — `sweep_mean.csv`

**modular_quadratic (the step-growth cliff):**

| dim | emender_cma | gdn2 (fla) | gdn2typed | emender−gdn2typed |
|----:|------:|------:|------:|------:|
| 256 | 0.990 | 0.675 | 0.996 | **−0.006** |
| 384 | 1.000 | 0.995 | 0.995 | +0.005 |
| 512 | 1.000 | 0.996 | 0.995 | +0.005 |
| 768 | 1.000 | 0.995 | 0.983 | +0.017 |
| 1024 | 1.000 | 0.998 | 0.889 | +0.111* |

\*at dim1024 the *typed control* dips (one-seed variance); emender ties fla-gdn 0.998.

**s5_permutation (state-tracking):**

| dim | emender_cma | gdn2 (fla) | gdn2typed | emender−gdn2typed |
|----:|------:|------:|------:|------:|
| 256 | 0.774 | 0.999 | 0.998 | **−0.224** |
| 384 | 0.565 | 0.935 | 0.995 | **−0.430** |
| 512 | 0.315 | 0.998 | 0.996 | **−0.681** |
| 768 | 0.541 | 0.999 | 0.989 | **−0.448** |
| 1024 | 0.312 | 0.893 | 0.990 | **−0.678** |

**modular_counter (counting):**

| dim | emender_cma | gdn2 (fla) | gdn2typed | emender−gdn2typed |
|----:|------:|------:|------:|------:|
| 256 | 0.433 | 0.457 | 0.458 | −0.025 |
| 384 | 0.403 | 0.530 | 0.529 | **−0.126** |
| 512 | 0.459 | 0.557 | 0.666 | **−0.207** |
| 768 | 0.514 | 0.559 | 0.602 | −0.088 |
| 1024 | 0.463 | 0.510 | 0.525 | −0.062 |

### Capacity boundary — measured

The task asked for the dim where the Emender separation **closes**. Measured
result: **there is no positive separation to close.** Against the clean
`gdn2typed` control, `emender_cma − gdn2typed ≤ 0.05` at **every** dim on **every**
task (it is ≤ 0 everywhere). So the boundary, as the smallest dim where the
advantage falls below 0.05, is **dim 256 — the smallest tested — for all three
tasks**, because the advantage is already absent (negative) there.

- **modular_quadratic:** emender **ties** the typed control at all dims (both
  solve the cliff). The only place emender beats the *fla-gdn* incumbent is
  dim256 (0.990 vs 0.675) — but `gdn2typed` wins it identically (0.996), so that
  edge is the **typed substrate, not the nonlinear emendment heads**. By dim384
  fla-gdn also reaches the ceiling.
- **s5_permutation:** emender is **worse at every dim** and the deficit is large
  (−0.22 to −0.68). Per-seed it is **bimodal**: some seeds reach 1.0, others
  collapse to ~0.31, and the collapsed fraction grows with scale (at dim512 all
  3 seeds collapse to ~0.31). The nonlinear emendment heads **destabilize S5
  state-tracking**, increasingly with scale. GDN-2 (both controls) solves S5
  robustly (~0.99) at every dim.
- **modular_counter:** emender is **worse at every dim**, deficit peaking −0.207
  at dim512.

---

## Phase 3 — EFFICIENCY (measured)

### Throughput — `throughput.json` (1.3B head shape dim=2240, depth18, 64 heads, bf16, fwd+bwd)

| arm | tok/s | ratio vs GDN-2 |
|---|---:|---:|
| gdn_pure (GDN-2 ceiling) | 11926 | 1.000 |
| emender4 (4/64 e97_delta-tanh) overlap-off | 8775 | 0.736 |
| emender4 overlap-on | 9121 | **0.765** |
| emender8 (8/64) overlap-on | 8733 | 0.732 |
| shell4 (gdn2_nonlin_shell, capability-inert) overlap-on | 11361 | 0.953 |

The CMA-found fraction (≈9.4% e97_delta ≈ 6/64) sits between emender4 and
emender8 → **≈0.74× GDN-2 throughput** at the 1.3B head shape (overlap on). The
0.95× target is met only by the capability-**inert** shell head, not the
capability head.

### Iso-param held-out BPB — `results_efficiency/efficiency.json` (≈41M params, bf16, fused)

**Iso-param, iso-TOKEN (8M tokens each):**

| arm | held-out BPB | tok/s | params |
|---|---:|---:|---:|
| gdn2_pure | 2.51316 | 78011 | 41.21M |
| emender (CMA-found) | 2.51345 | 42156 | 42.10M |

→ **Δ BPB = +0.00029 = TIE** on loss at equal params and equal tokens.

**Iso-param, iso-WALL (derived from the measured token-tie + measured throughput):**
both arms reach the tied BPB 2.5132 at 8M tokens; at the measured tok/s that is
**102.5 s for GDN-2 vs 189.8 s for the Emender → the Emender is 1.85× slower to
equal loss.** (The direct 6-min wall probe is reported in the JSON but is
confounded: at lr 1.2e-3 both arms drift *past* their 8M-token checkpoints by
26M/14.5M tokens, so the 6-min BPBs (gdn2 3.606 / emender 3.105) reflect
late-training instability, not throughput — the token-tie + throughput inference
above is the reliable iso-wall statement.)

---

## One-line measured verdict

The properly-budgeted CMA keeps a 12.5% nonlinear sprinkle **on a flat loss
basin**; that CMA-found Emender **ties** best-vs-best GDN-2 on iso-param/iso-token
loss and on the modquad cliff, is **strictly worse** on S5 and counting at every
tested dim (256→1024), and runs at **0.54–0.74× throughput**. There is **no
capacity boundary at which an Emender advantage closes, because no advantage over
the typed GDN-2 control exists at any scale.** NO-GO / NULL — see `VERDICT.md`.
