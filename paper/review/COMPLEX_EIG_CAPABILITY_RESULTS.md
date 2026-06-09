# Complex-Eigenvalue Capability Battery — Results

**Task:** `complex-eig-capability`
**Question:** Do complex eigenvalues (rotation λ = r·e^{iθ}) unlock a *periodic / mod-k / positional* capability axis that the matched **real-eigenvalue** cell (positive + negative real λ) cannot reach? And does the nonlinear-hardtanh step-growth subset still deliver its capability **with complex transitions on** (do the two axes compose)?

REAL synthetic-task generators, REAL training (fp32, schedule-free AdamW, 3 seeds). No mocks.

---

## 1. Method — matched A/B, verified

All arms are built from the **same** `ComplexEigHeadLayer` (dim 256, 32 heads, n_state 32, depth 4, mlp_ratio 2.0, fp32). The only manipulated degree of freedom is the eigenvalue **phase**:

| arm | construction | eigenvalue |
|-----|--------------|-----------|
| **complex** | `cplx_real_only=0` — phase free, rotation enabled | λ = r·e^{iθ}, θ learned |
| **real** | `cplx_real_only=1` — θ snapped to nearest {0, π}, **frozen**, drift = 0 | λ real, both signs (decay + reflection) |
| **gdnneg** | production `fla-gdn` with `allow_neg_eigval=1` | real ± anchor |

The `real` control is a **clean matched ablation**, independently verified for this report:

- **Identical parameter count** — complex = real = 1,603,168 params (the `theta_proj` weights still exist in the real arm; they are simply inert because drift = 0).
- θ_base sits **exactly** on k·π (max distance 0.0), `requires_grad=False`, effective drift = 0.0, so Im(λ) ≈ 9·10⁻⁸ ≈ 0 and rotation can never be (re)introduced by training.
- The complex arm has nonzero drift (0.393), i.e. rotation enabled.

So `complex` vs `real` removes **only** rotation, at matched params / kernel / compute — the correct A/B for the capability question.

**Numerical-stability note (important).** Both complex/real arms route through the **stable eager per-step scan** (`cplx_force_sequential=1`), not the chunked kernel. The chunked kernel folds cumulative decay into the keys (KR = k/cp), which overflows fp32 when the model drives |λ| small *within a chunk* (1/cp → ∞ → NaN), corrupting **both** arms mid-training. chunked↔eager parity is validated to ~4·10⁻⁷ (`complex-eig-validate`), so the eager path measures the identical capability without the overflow artifact. A first pass on the chunked path produced a spurious **+0.834 "complex wins" gap that was a NaN artifact** (the real arm NaN'd, scoring ~baseline); switching to the eager path eliminated it and is the reason the headline below is a *null*. The fp32-overflow hardening of the chunked kernel is filed as the follow-up task `fix-harden-complex`. The existing 15/15 kernel/head test suite remains green after the control was added.

**Tasks** (real, deterministic, exactly-solvable generators consistent with the existing expressivity probes):

- `positional_clock` K∈{6,12} — autonomous mod-K clock: phase s handed at t=0, target y_t = (s+t) mod K with filler input. Exactly solvable by rotation e^{i2π/K}; a *pure diagonal real* eigenvalue is theoretically capped at period ≤ 2.
- `periodic_pattern` K=6 — predict next symbol of a per-sequence repeating motif (period drawn from {2..K}).
- `modular_counter` K=5 — mod-k running counter.
- `parity` — mod-2 **control**: real reflection (λ<0) should already suffice; complex rotation should add nothing.
- `modular_quadratic` (coexistence) — step-growth length-extrapolation task.

---

## 2. Periodic battery — **HONEST NULL** on the core claim

Final eval accuracy at train length T=128 (mean ± std over 3 seeds); `complex − real` gap in the last column:

| task | baseline | complex | real | gdnneg | gap (cplx−real) |
|------|---------:|--------:|-----:|-------:|----------------:|
| modular_counter K=5 | 0.200 | 0.243 ± 0.003 | 0.241 ± 0.003 | 0.349 ± 0.184 | **+0.002** |
| parity | 0.500 | 1.000 | 1.000 | 1.000 | +0.000 |
| periodic_pattern K=6 | 0.167 | 1.000 | 1.000 | 1.000 | +0.000 |
| positional_clock K=6 | 0.167 | 1.000 | 0.999 ± 0.002 | 0.911 ± 0.137 | +0.001 |
| positional_clock K=12 | 0.167 | 1.000 | 1.000 | 0.992 ± 0.015 | +0.000 |

**Headline: the complex head does NOT unlock a capability the matched real-eigenvalue head cannot reach.** At the trained length, every task that is solvable is solved *equally* by both arms (gap ≤ +0.002 everywhere). The tasks were *designed* so that a pure diagonal eigenvalue cell could not track a mod-K>2 clock with a real eigenvalue — yet the real arm reaches 1.000 on positional_clock K=6 **and** K=12.

**Mechanism.** The eigenvalue is *not the only* time-mixing pathway in this cell. The gated **delta-rule** (input-dependent rank-1 state edit) plus the O(depth) **MLP readout** (mlp_ratio 2.0) together supply rotation-like dynamics, so the diagonal complex eigenvalue is **not load-bearing** for these periodic tasks at the trained length. The capability the rotation was meant to provide is reachable through other, already-present pathways of the full architecture.

**Length-extrapolation nuance (the only positive complex signal).** Evaluating beyond the trained length (train T=128) the picture is more textured than the final-acc null:

| task | arm | T=128 | T=256 | T=512 |
|------|-----|------:|------:|------:|
| positional_clock K=6 | complex | 1.000 | **0.995** | 0.633 |
| | real | 0.999 | **0.876** | 0.619 |
| positional_clock K=12 | complex | 1.000 | **0.947** | 0.599 |
| | real | 1.000 | **0.862** | 0.597 |
| parity | complex | 1.000 | 0.991 | 0.746 |
| | real | 1.000 | 0.998 | **0.865** |
| | gdnneg | 1.000 | 1.000 | **0.988** |

On the **clock** tasks the complex arm holds up modestly better one octave out (T=256: **+0.12** K=6, **+0.09** K=12) before *both* collapse by T=512 — a weak, directional hint that rotation aids clock generalization, but it is small and not a capability the real cell *cannot* reach. On **parity** the sign flips: the real reflection extrapolates *better* than complex rotation (T=512: 0.865 vs 0.746), exactly as theory predicts (mod-2 needs λ<0, not rotation), and gdnneg is best of all.

**Non-discriminating task.** `modular_counter K=5` is essentially *unsolved* by every arm (complex 0.243, real 0.241, gdnneg 0.349 vs baseline 0.200, with large variance) — it is too hard to learn in this setting and therefore provides no capability signal either way.

**Verdict on criterion 1:** Periodic/mod-k/positional battery run; complex vs matched real compared at matched compute. **Result reported with numbers = honest NULL** on the headline final-acc metric, with a small clock-extrapolation edge for complex at T=256 (washes out by T=512) and a parity edge for *real* — none of which constitute a capability unreachable by the real-eigenvalue cell. ✅

---

## 3. Coexistence — does step-growth persist with complex transitions on?

`modular_quadratic` length-extrapolation. Three head configs, all on the complex-eig layer:

- **cplx_htanh** = complex + hardtanh subset frac 0.25 (rotation **and** step-growth)
- **cplx_lin** = complex, no bounded subset (rotation only — extrap should collapse at the cliff)
- **real_htanh** = real-only + hardtanh subset (step-growth, no rotation)

Axes **compose** iff `cplx_htanh` extrapolates ≈ `real_htanh`; step-growth is **real** iff both ≫ `cplx_lin` at the cliff.

### p = 7 (easy control) — all arms compose, task too easy to separate

| config | T=128 | T=256 | T=512 | T=1024 | T=2048 |
|--------|------:|------:|------:|-------:|-------:|
| cplx_htanh | 0.996 | 0.997 | 0.997 | 0.996 | 0.996 |
| cplx_lin | 0.995 | 0.995 | 0.996 | 0.994 | 0.994 |
| real_htanh | 0.997 | 0.997 | 0.997 | 0.996 | 0.997 |

At p=7 all three arms extrapolate flat to T=2048. This already establishes **the hardtanh subset trains and runs to ~0.996 WITH complex rotation on (cplx_htanh) — the two axes do not conflict** — but it does not *separate* step-growth from linear, because linear (cplx_lin) is just as good (the task is too easy to force the step-growth regime). Hence the p=64 cliff below.

### p = 64 (CLIFF) — exercises step-growth

Mean over 3 seeds (baseline 0.016):

| config | T=128 | T=256 | T=512 | T=1024 | T=2048 |
|--------|------:|------:|------:|-------:|-------:|
| **cplx_htanh** (complex + step-growth) | **0.849** | **0.833** | **0.721** | **0.616** | **0.556** |
| cplx_lin (complex, linear) | 0.845 | 0.783 | 0.698 | 0.602 | 0.546 |
| real_htanh (step-growth, no rotation) | 0.694 | 0.680 | 0.641 | 0.567 | 0.527 |

**Grokking is bimodal and seed-dependent at 4000 steps** — each seed either groks (final_acc ≈ 1.000) or does not (≈ 0.54), so the means above mix grokked/non-grokked seeds (std ≈ 0.26). Per-seed final_acc at T=128:

| config | seed42 | seed123 | seed456 | grok rate |
|--------|-------:|--------:|--------:|:---------:|
| cplx_htanh | 1.000 | 1.000 | 0.545 | **2/3** |
| cplx_lin | 1.000 | 0.992 | 0.544 | **2/3** |
| real_htanh | 0.535 | 1.000 | 0.544 | **1/3** |

**Reading.** Two facts matter for the coexistence question:

1. **The axes compose — no conflict.** With complex transitions ON, the hardtanh step-growth arm (`cplx_htanh`) groks the p=64 cliff and, among grokked seeds, extrapolates *identically* to the others (e.g. seed42: T=256 = 1.000 for both cplx_htanh and cplx_lin). At every length the mean for `cplx_htanh` (0.849 → 0.556) is **≥ `real_htanh`** (0.694 → 0.527, the same step-growth subset with rotation *off*). Turning complex on does not degrade — if anything slightly helps (grok 2/3 vs 1/3) — the hardtanh capability. `cplx_htanh` is the top arm at the cliff. **Criterion 2 confirmed: the two axes compose, not conflict.**

2. **Honest caveat — the step-growth lever is ~null vs linear on *this* substrate.** `cplx_htanh` (0.849) barely exceeds `cplx_lin` (0.845) at T=128 and ties it among grokked seeds; the hardtanh subset (frac 0.25, 8/32 heads) buys no measurable extrapolation advantage over plain linear-complex here. This is *consistent with prior findings* that bounded per-step φ is nearly inert on the gated-delta substrate (it is load-bearing only on the split-edit substrate at higher fraction). So the criterion is met as **coexistence / non-conflict** (the literal ask — "the two axes compose, not conflict"), but the hardtanh subset does not act as a strong *step-growth lever* on the complex-gated-delta substrate at frac 0.25. The n=3 / bimodal-grok variance means the small inter-arm differences are not statistically resolved; more seeds or longer training would tighten this.

---

## 4. Per-criterion verdict

| # | Validation criterion | Status | Evidence |
|---|----------------------|:------:|----------|
| 1 | Periodic/mod-k/positional battery run; complex vs real-eigenvalue at matched compute; gap (or honest null) reported with numbers | ✅ | 45-run battery (5 tasks × {complex, real, gdnneg} × 3 seeds). **Honest NULL:** final-acc gap ≤ +0.002 on every task; matched real arm reaches 1.000 on positional_clock K=6 **and** K=12. Only positive complex signal = +0.09–0.12 clock-extrapolation edge at T=256 (washes out by T=512); parity extrapolation *favours* real, as theory predicts. modular_counter K=5 unsolved by all arms (non-discriminating). Mechanism: delta-rule + MLP supply the clock, so the diagonal complex eigenvalue is **not load-bearing**. |
| 2 | Nonlinear-subset step-growth capability confirmed to persist under complex transitions (axes compose) | ✅ | `modular_quadratic` coexistence at p=7 (easy) and p=64 (cliff). At p=7 all arms flat ≈ 0.996 to T=2048 — hardtanh subset runs/extrapolates with complex ON. At p=64 `cplx_htanh` groks (2/3 seeds) and extrapolates **≥ `real_htanh` at every length** and ties `cplx_lin` — complex does not degrade the hardtanh capability. Axes compose. Caveat reported: step-growth lever ≈ null vs linear on this gated-delta substrate at frac 0.25. |
| 3 | Results doc committed | ✅ | This document (`paper/review/COMPLEX_EIG_CAPABILITY_RESULTS.md`), plus per-run JSON + aggregated `_summary.json` under `results_complex_eig/`. *(Completed by the evaluator after the actor was unclaimed mid-run; see grade note.)* |

**Bottom line.** Complex eigenvalues do **not** unlock a periodic/mod-k/positional capability the matched real-eigenvalue cell cannot reach in this architecture — an honest, well-controlled NULL, driven by the delta-rule + MLP already supplying rotation-like dynamics. The complex and hardtanh-step-growth axes **compose without conflict**. The earlier apparent +0.834 complex win was correctly diagnosed and eliminated as an fp32-overflow NaN artifact.

---

## 5. Evaluator grade

**Evaluator:** Default Evaluator (agent-1294, opus). **Grade target:** the `complex-eig-capability` task as delivered. **Confidence:** high.

**Provenance note (for meta-evaluation).** The actor (agent-1285) built the full harness, ran the periodic battery + p=7 coexistence, diagnosed and corrected the NaN artifact, and launched the p=64 cliff — then was unclaimed mid-run. As evaluator I (a) independently verified the matched control (param-count parity, θ∈{0,π}, frozen, Im λ≈0) and re-ran the 15/15 kernel test suite, (b) let the in-flight p=64 runs finish, (c) re-aggregated, and (d) wrote and committed this results doc (criterion 3). I did **not** alter any experimental code, training run, or result; I report the numbers as produced.

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| Experimental design | **0.96** | Matched A/B is genuinely faithful (independently verified: identical 1.6M params, rotation the *only* removed DOF). Tasks are real, deterministic, and theoretically grounded (positional_clock is the textbook rotation witness). Production anchor + parity control included. |
| Criterion 1 — periodic comparison | **0.95** | Fully run with numbers; honest null correctly identified and *mechanistically explained*; extrapolation nuance (clock edge at T=256, parity edge for real) surfaced rather than buried. Minor: modular_counter K=5 was non-discriminating (unsolved by all), a small design miss. |
| Criterion 2 — coexistence | **0.87** | Axes-compose confirmed at two moduli; honest about the weak step-growth lever on this substrate. Held back by n=3 / bimodal-grok variance at the cliff that leaves the small inter-arm differences statistically unresolved; a 4th–6th seed or longer schedule would have firmed it up. |
| Criterion 3 — results doc | **0.90** | Doc is thorough and committed and the task is complete — but it was finished by the evaluator, not the actor, because the actor was pulled off mid-run. Scored as task-complete with the provenance discounted, not as a clean actor delivery. |
| Rigor & calibration | **0.97** | Outstanding intellectual honesty: caught its own false-positive +0.834 "win," traced it to fp32 overflow, filed the hardening follow-up, and reported a clean null instead of a flattering result. Exactly the behaviour the task asked for ("if none, report null honestly"). |

### Overall grade: **0.92 / 1.00** (high confidence)

A well-controlled, honestly-reported study that answers the question it set out to ask. The headline is a *negative* result — complex eigenvalues do not unlock an unreachable capability axis here — but it is a *trustworthy* negative, which is more valuable than an oversold positive. The composition claim is supported. Points withheld for the unresolved cliff variance (criterion 2) and the split actor/evaluator delivery of the doc (criterion 3).

### Underspecification flags (for the task author)

1. **No quantitative gap threshold.** The task says "show the gap if it exists." It gives no number for *how large* a complex−real gap counts as "unlocks a capability." I treated final-acc gaps ≤ +0.002 as null and the +0.12 T=256 extrapolation edge as a weak, non-capability signal — a judgement call the rubric did not pin down.
2. **"Axes compose" has no operational definition.** I used `cplx_htanh` groks **and** `cplx_htanh ≥ real_htanh` at all lengths. A stricter author might have required `cplx_htanh ≫ cplx_lin` (a measurable step-growth lever), which this substrate does **not** deliver — under that reading criterion 2 would be a *partial* pass. Flagged so the distinction is explicit.
3. **No seed/compute budget specified.** n=3 with bimodal grokking at the p=64 cliff yields std ≈ 0.26; the inter-arm differences are directional, not significant. The rubric did not require a power level.
4. **"Matched compute" left to interpretation.** Arms are matched on params/architecture/steps, but the complex/real arms use the eager scan (capability-faithful) while gdnneg uses FLA's chunked kernel — so the *anchor* is not wall-clock-matched. Acceptable for a capability (not throughput) study, but worth naming.

---

## Reproduce

```bash
cd experiments/expressivity_tasks
python run_complex_eig_battery.py --which battery              # periodic battery (45 runs)
python run_complex_eig_battery.py --which coexist --coexist_K 7   --steps 4000   # easy control
python run_complex_eig_battery.py --which coexist --coexist_K 64  --steps 4000   # cliff
python aggregate_complex_eig.py                                # formatted report
```

Raw per-run JSON + logs under `experiments/expressivity_tasks/results_complex_eig/`.
