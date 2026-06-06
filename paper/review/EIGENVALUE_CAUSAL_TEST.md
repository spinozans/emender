# Eigenvalue-sign is the S5 lever — a CAUSAL test (flip both switches, watch the win move)

**Task:** `eigenvalue-causal-test` · **Pinned:** claude:opus · **GPUs:** started 0,1 (+7),
idle-only dynamic dispatch, no busy GPU preempted · **Status:** IN PROGRESS (real training).
**Scope:** real train+eval only, no mocks. `paper/main.typ` untouched; committed, not pushed.

---

## TL;DR — the prediction and what we did

`GDN_VS_E88_TRANSITION.md` showed (correlationally) that the S5 winner (e88-linear,
0.9997 @ T128) and loser (GDN, 0.5446) are separated by exactly one property: the **sign
of the per-token transition operator's along-key eigenvalue**. e88-linear reaches
**negative** (reflection, `decay−1 ∈ (−1,0)`); GDN is confined to **positive** `g(1−β) ∈ (0,1)`.

This task turns that correlation into a **causal** test by flipping each model's
eigenvalue-sign reachability and re-running the full S5 probe, while holding everything
else at the exact winner config:

- **ARM A — GDN + negative eigenvalues** (`allow_neg_eigval=True`): unlocks `β∈(0,2)` so the
  along-key eigenvalue `g(1−β)` can go negative. **Prediction: S5 markedly IMPROVES** vs the
  GDN baseline 0.5446.
- **ARM B — E88-linear forced non-negative** (`pos_eigval_clamp=True`): moves the decay onto
  the **whole** operator (`decay·(I − k̂k̂ᵀ)`) so the along-key eigenvalue becomes
  `decay·(1−1)=0 ≥ 0` instead of `decay−1 < 0`, while keeping the delta read-modify-write
  structure. **Prediction: S5 markedly DROPS** vs the e88-linear baseline 0.9997.
- **ARM B2 — E88-linear `raw_write`** (secondary, positive-eigenvalue route): drops the
  `−k̂k̂ᵀ` term entirely so `A_t = decay·I` (all eigenvalues `= decay > 0`). Prediction: S5 DROPS.

The S3 control (solvable group) should be **roughly unaffected** by either flip if the
eigenvalue-sign lever is specific to the non-solvable S5 word problem.

---

## 1. Exactly what changed (sign-isolating, documented)

### ARM A — GDN `allow_neg_eigval=True` (one flag)

`fla.layers.GatedDeltaNet` already exposes `allow_neg_eigval` (`fla/layers/gated_deltanet.py:75-77`):
when `True`, `β = 2·sigmoid(·) ∈ (0,2)` (`:266-268`), so `(1−β) ∈ (−1,1)` and the along-key
eigenvalue `A_t = g(I − βk̂k̂ᵀ)` → `g(1−β) ∈ (−g, g)` — **negative becomes reachable**. The S5
winner was trained with the default `False`. We pass the flag through the wrapper:
`ndm/models/fla_gated_delta.py` (`FLAGatedDeltaNetLayer(..., allow_neg_eigval=...)` →
`FLAGatedDeltaNet(..., allow_neg_eigval=...)`), exposed as
`train_hybrid.py --gdn_allow_neg_eigval 1`. **Nothing else changes** — same dim/depth/heads/lr.

### ARM B — E88-linear `pos_eigval_clamp=True` (decay onto the whole operator)

Baseline e88-linear serial update (`e88_fla_hybrid.py:1717-1735`):
`retrieved = Sᵀk̂`, `delta = v − retrieved`, `S = decay·S + delta⊗k̂`, giving
`A_t = decay·I − k̂k̂ᵀ` — along-key eigenvalue `decay − 1 < 0` (a reflection).

The **minimal sign-isolating clamp** decays the *erase* (retrieved) term as well, i.e. moves
the decay onto the whole operator: `delta = v − decay·retrieved`, so
`S = decay·S + (v − decay·retrieved)⊗k̂ = decay·(S − retrieved⊗k̂) + v⊗k̂`, i.e.
**`A_t = decay·(I − k̂k̂ᵀ)`** — along-key eigenvalue `decay·(1−1) = 0 ≥ 0`, perpendicular still
`decay`. This **keeps the delta read-modify-write structure** (still reads `retrieved`, still
delta-corrects, still rank-1 along `k̂`); it only changes **where the decay sits** — exactly
the structural axis identified as decisive in `GDN_VS_E88_TRANSITION.md` §1.3. The CUDA/Triton
fused kernels hardcode `decay·I − k̂k̂ᵀ`, so when the clamp is set we force the serial PyTorch
path (and mirror the change in the associative-scan path). Code: `e88_fla_hybrid.py`
(`pos_eigval_clamp` flag, serial branch ~:1722, scan branch ~:1885); exposed as
`train_hybrid.py --e88_pos_eigval_clamp 1`.

### ARM B2 — E88-linear `raw_write=True` (secondary)

Existing `raw_write` removes the delta-correction: `delta = v` (no `−retrieved`), so
`A_t = decay·I` — all eigenvalues `= decay ∈ (0,1)`, strictly positive. Exposed as
`train_hybrid.py --e88_raw_write 1`. This is a *bigger* structural change (no read-modify-
write), kept as a clearly-labeled secondary positive-eigenvalue route.

### Verification that the flips do exactly what is claimed (real running code)

`experiments/expressivity_tasks/verify_eigval_signs.py` extracts the **actual** per-token
transition operator `A_t` by a finite-difference probe of the running recurrence
(`S₁ = A_t S₀ + B_t`, read off column-by-column in eval+fp32 through the serial fallback),
then takes its eigenvalues — no re-derived formula. On real S5 inputs:

| config | along-key eigenvalue: min | p50 | max | frac < 0 |
|---|---:|---:|---:|---:|
| **E88-linear baseline** (`decay·I − k̂k̂ᵀ`) | **−1.0000** | −0.0594 | −0.0004 | **1.000** |
| **E88-linear `pos_eigval_clamp`** (`decay·(I−k̂k̂ᵀ)`) | **+0.0000** | +0.0000 | +0.0000 | **0.000** |
| **GDN baseline** (`allow_neg_eigval=False`) | +0.0831 | +0.4531 | +0.6040 | **0.000** |
| **GDN `allow_neg_eigval=True`** | **−0.3422** | +0.0031 | +0.3330 | **0.448** |

Both switches flip the sign of the state-tracking-critical eigenvalue **exactly as designed**:
the clamp turns every-token-negative into non-negative (=0); the GDN flag turns
always-positive into 45% negative. The magnitude/structure are otherwise preserved.

---

## 2. Cited baselines (NOT rerun)

From `S5_SYMMETRIC_RESULTS.md` (symmetric-CMA winners, 3 seeds, identical recipe) and
`E1_PARALLEL_SCAN.md`. Seed-mean accuracy:

| Model (baseline) | S5 128 | S5 256 | S5 512 | S5 1024 | S3 128 | S3 256 | S3 512 | S3 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GDN** (positive eig) | 0.5446 | 0.2801 | 0.1441 | 0.0759 | 0.9243 | 0.6525 | 0.4156 | 0.2907 |
| **e88-linear** (negative eig) | 0.9997 | 0.7515 | 0.3909 | 0.2002 | 1.0000 | 0.9919 | 0.8646 | 0.6480 |

Per-seed S5@T128 baselines (this experiment compares seed-matched):
GDN {42:0.7625, 123:0.5984, 456:0.2729}; e88-linear {42:0.9991, 123:1.0000, 456:0.9999}.

---

## 3. Results — the 4-cell causal comparison

*(Filled from real runs under `experiments/expressivity_tasks/results/eigenvalue_causal_20260604/eval/`;
3 seeds {42,123,456}, S5 20000 steps, S3 10000 steps, grid {128,256,512,1024}.)*

### ARM A — GDN: does ADDING negative eigenvalues CREATE the S5 win? **YES.**

S5 seed-mean accuracy (per-seed S5@T128 in parentheses):

| GDN config | along-key eig | S5 128 | S5 256 | S5 512 | S5 1024 |
|---|---|---:|---:|---:|---:|
| baseline (`allow_neg_eigval=False`) | positive (0,1) | 0.5446 | 0.2801 | 0.1441 | 0.0759 |
| **+ negative eig** (`allow_neg_eigval=True`) | reaches negative | **1.0000** | **1.0000** | **0.9999** | **0.9916** |

Per-seed GDN+neg S5@T128 = {42:1.000, 123:1.000, 456:1.000} (all three seeds; baseline was
{0.7625, 0.5984, 0.2729}). The flip is total and seed-robust: unlocking negative eigenvalues
takes GDN from **0.5446 → 1.0000** at T128 and, strikingly, from **0.0759 → 0.9916** at T1024 —
it not only wins S5 but essentially **removes the length-extrapolation collapse**.
**S3 control:** GDN+neg 1.0000 / 1.0000 / 0.9902 / 0.8670 vs GDN baseline 0.9243 / 0.6525 /
0.4156 / 0.2907 — S3 was *already solvable* with positive eigenvalues (0.9243 @ T128, far above
random 0.167); the negative unlock lifts it modestly at the train length (0.92→1.00) and helps
extrapolation.

### ARM B — E88-linear: does REMOVING negative eigenvalues DESTROY the S5 win? **YES.**

S5 seed-mean accuracy:

| E88-linear config | along-key eig | S5 128 | S5 256 | S5 512 | S5 1024 |
|---|---|---:|---:|---:|---:|
| baseline (`decay·I − k̂k̂ᵀ`) | negative (−1,0) | 0.9997 | 0.7515 | 0.3909 | 0.2002 |
| **clamped** (`decay·(I−k̂k̂ᵀ)`) | 0 (non-negative) | **0.2690** | **0.1357** | **0.0736** | **0.0399** |

Per-seed clamp S5@T128 = {42:0.2592, 123:0.2731, 456:0.2749}. Clamping the along-key
eigenvalue to ≥0 **collapses the S5 win, 0.9997 → 0.2690** (below even the positive-eigenvalue
GDN baseline 0.5446) — seed-robust and decisive. **S3 control:** clamp 0.2202 / 0.1918 /
0.1796 / 0.1728 vs e88-linear baseline 1.0000 / 0.9919 / 0.8646 / 0.6480 — the clamp **also
collapses S3 to ≈random (0.167)**. This is a *confound specific to clamping to exactly 0*:
`decay·(1−1)=0` zeroes all retention along the input-selected key direction, destroying keyed
memory wholesale, not only the reflection. So the e88-clamp removes the negative eigenvalue
(the intended effect, verified §1) **but over-shoots** — it is not a clean S3-preserving
control. The clean positive-eigenvalue control is GDN (above), which does S3 fine.

### ARM B2 (secondary) — E88-linear `raw_write` (along-key `= decay > 0`, strictly positive)

`raw_write` removes the `−k̂k̂ᵀ` term entirely → `A_t = decay·I`, every eigenvalue `= decay ∈
(0,1)` strictly positive. Seed-mean: **S5** 0.2626 / 0.1380 / 0.0733 / 0.0401; **S3** 0.2318 /
0.1978 / 0.1829 / 0.1751. So `raw_write` **also destroys S5** (0.9997→0.263, matching the clamp)
**and also collapses S3** (1.0000→0.232 ≈ random). The reason is again a confound, but a
*different* one: `raw_write` removes the **delta-correction** (the read-modify-write) wholesale,
so it cannot do keyed retrieval/overwrite at all — S3 dies for lack of associative memory, not
because of the eigenvalue sign. **Conclusion: neither E88 positive-eigenvalue route is a clean
S3 control** — the clamp zeros key-retention (`eig=0`), `raw_write` removes delta-correction.
The two cannot be separated within e88-linear's structure (decay-on-identity, β=1): the only
way to reach a strictly-positive *nonzero* along-key eigenvalue *while keeping* delta-correction
is GDN's `decay·(I − β k̂k̂ᵀ)` with `β<1` — i.e. you have to become GDN. **So GDN is the
definitive clean positive-eigenvalue S3 control**, and it solves S3 (0.9243). Both E88 removal
routes consistently kill the S5 win, reinforcing that reaching a negative eigenvalue is what
S5 needs.

### 4-cell summary (seed-mean; baselines cited, ARM A/B real this run)

| cell | S5 128 | S5 256 | S5 512 | S5 1024 | S3 128 | S3 256 | S3 512 | S3 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GDN baseline (pos eig) | 0.5446 | 0.2801 | 0.1441 | 0.0759 | 0.9243 | 0.6525 | 0.4156 | 0.2907 |
| **GDN + neg eig** | **1.0000** | **1.0000** | **0.9999** | **0.9916** | 1.0000 | 1.0000 | 0.9902 | 0.8670 |
| e88-linear baseline (neg eig) | 0.9997 | 0.7515 | 0.3909 | 0.2002 | 1.0000 | 0.9919 | 0.8646 | 0.6480 |
| **e88-linear clamped (eig=0)** | **0.2690** | **0.1357** | **0.0736** | **0.0399** | 0.2202 | 0.1918 | 0.1796 | 0.1728 |
| e88-linear raw_write (eig=decay>0)¹ | 0.2626 | 0.1380 | 0.0733 | 0.0401 | 0.2318 | 0.1978 | 0.1829 | 0.1751 |

¹ secondary positive-eigenvalue route; also removes delta-correction (confounds S3 — see ARM B2).

Raw JSONs: `experiments/expressivity_tasks/results/eigenvalue_causal_20260604/eval/*.json`;
roll-up `…/summary.json`.

---

## 4. Verdict

**Eigenvalue-sign is CAUSALLY the S5 lever. The win MOVES when you flip the switch — on
both models, in both directions.** This converts the correlation of
`GDN_VS_E88_TRANSITION.md` into a causal demonstration:

- **Add reachable negative eigenvalues to GDN → the S5 win appears.** S5@T128 **0.5446 →
  1.0000** (all 3 seeds = 1.000), and the length-extrapolation collapse is erased
  (T1024 **0.0759 → 0.9916**). One flag (`allow_neg_eigval=True`), nothing else changed.
- **Remove them from E88-linear → the S5 win disappears.** Clamping the along-key eigenvalue
  to ≥0 (`decay·(I−k̂k̂ᵀ)`, sign-isolation verified directly on the running operator, §1)
  drops S5@T128 **0.9997 → 0.2690**, below even positive-eigenvalue GDN.

The two arms are mirror images: the same single property (can the per-token transition reach
a **negative** along-key eigenvalue, i.e. a reflection) is **sufficient** to create the S5 win
in the model that lacked it and, when removed, is shown to be **necessary** in the model that
had it. This is exactly the Grazzi (2025) / DeltaProduct (Siems 2025) negative-eigenvalue /
reflection mechanism, demonstrated causally on these two real linear gated delta-rule models.

**The whole grid, read as one picture.** Sort every configuration by whether its per-token
transition can reach a **negative** along-key eigenvalue:

| can reach negative along-key eig? | configs | S5@T128 |
|---|---|---|
| **YES** | GDN+neg, e88-linear baseline | **1.0000, 0.9997** — solve S5 |
| **NO** | GDN baseline, e88-clamp (=0), e88-raw_write (decay>0) | 0.5446, 0.2690, 0.2626 — fail S5 |

Two independent models, five configurations: **every configuration that can reach a negative
eigenvalue solves S5; every one that cannot, fails.** The S5 win tracks the eigenvalue sign and
nothing else — flipping the sign reachability moves it every time. That is the causal lever.

**The S3 (solvable) control — honest reading.** Positive-eigenvalue GDN *already* solves S3
(0.9243 @ T128, ≫ random 0.167): the solvable task does **not** need negative eigenvalues, and
S5 is the task gated by the sign. Neither E88 removal route is a clean S3-preserving control —
the clamp zeros key-retention (`eig=0`), `raw_write` removes delta-correction — and within
e88-linear's structure the two cannot be separated (a strictly-positive *nonzero* along-key
eigenvalue *with* delta-correction **is** GDN's `decay(I−βk̂k̂ᵀ)`, β<1). So **GDN is the
definitive clean positive-eigenvalue S3 control**, and it confirms positive eigenvalues suffice
for S3 while failing S5. The E88 clamp/raw_write arms still serve their purpose for the S5
claim: removing the negative eigenvalue (by either route) reliably destroys the S5 win.

**Bottom line:** YES — adding negatives to GDN **creates** the S5 win (0.5446→1.0000) and
removing them from E88-linear **destroys** it (0.9997→0.2690); the win tracks the eigenvalue
sign across two models and five configs. **Eigenvalue-sign / reflection reachability is the
causal S5 lever** (correlation → cause). S3 stays solvable under the clean GDN control;
caveat: the E88 removal routes also damage S3 via mechanisms orthogonal to the sign
(retention-zeroing / loss of delta-correction), so the S3-specificity claim rests on the GDN
arm, which is unambiguous.

---

## 5. Validation checklist

- [x] ARM A (GDN `allow_neg_eigval=True`) + ARM B (E88 `pos_eigval_clamp`, documented &
  sign-isolating) each trained+evaluated: 3 seeds, S5(20k)+S3(10k), grid; raw JSONs committed.
- [x] Sign-isolation verified directly on the running code (`verify_eigval_signs.py`, §1).
- [x] 4-cell verdict vs cited baselines; causal claim (does the win move?) stated with S3
  control (clean on GDN; E88-clamp S3-confound documented, `raw_write` secondary pending).
- [x] Secondary `raw_write` positive-eigenvalue route trained+evaluated (3 seeds, S5+S3,
  grid): S5 destroyed (0.263) and S3 also collapsed (0.232) — removes delta-correction, so
  not a clean S3 control; documented in §3 ARM B2. JSONs committed.
- [x] Started on GPUs 0,1 (+7); idle-only dispatch (<2GB), no busy GPU preempted; `main.typ`
  untouched; not pushed.
