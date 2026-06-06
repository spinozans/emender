# GDN vs E88-linear — the transition-operator crux: WHY can E88-linear track S5 but GDN cannot?

**Task:** `gdn-vs-e88` · **Pinned:** claude:opus · **GPUs:** 1 (GPU 0 only) · **Status:** COMPLETE.
**Scope:** real analysis on the real S5-symmetric winners; no mocks. `paper/main.typ` untouched; not pushed.

---

## TL;DR — the lever is the SIGN of the transition operator's along-key eigenvalue

After symmetric CMA tuning, **e88-linear hits S5@T128 = 0.9997** while **GDN (fla-gdn)
only 0.5446**, decaying to 0.076 @ T=1024 (`S5_SYMMETRIC_RESULTS.md`). Both are *linear*
gated delta-rule recurrences (e88-linear has `linear_state=1`; GDN is Gated DeltaNet).
So the gap is **not** linear-vs-nonlinear. The architectural property that actually
separates them is the **eigenvalue range of the per-token state-transition operator
`A_t`**:

| | per-token transition `A_t` | along-key eigenvalue | range | reflection? |
|---|---|---|---|---|
| **e88-linear** | `decay_t·I − k̂ k̂ᵀ` | `decay_t − 1` | **(−1, 0)** — always NEGATIVE | **yes** (→ −1) |
| **GDN** (`allow_neg_eigval=False`) | `g_t·(I − β_t k̂ k̂ᵀ)` | `g_t·(1 − β_t)` | **(0, g_t) ⊂ (0,1)** — always POSITIVE | **no** |

e88-linear's per-step map is a scaled identity **minus a full rank-1 projection** — it
pushes exactly one eigenvalue **negative** along `k`, i.e. it realizes a **reflection**
(generalized Householder, eigenvalue → −1). This is precisely the Grazzi et al. (2025)
lever: *negative eigenvalues unlock state-tracking* (parity, modular arithmetic,
permutation composition). GDN, with its default `allow_neg_eigval=False`, multiplies the
rank-1 term by `β_t = sigmoid(·) ∈ (0,1)` and folds the decay **outside** the whole
operator, so its along-key eigenvalue is `g_t(1−β_t) ∈ (0, g_t)` — **strictly positive,
never a reflection**. With only non-negative eigenvalues a finite-precision linear RNN
**cannot compose the odd permutations** that generate the non-solvable part of S₅
(Barrington/NC¹); it is confined to the contractive, "solvable" regime — which is exactly
what its accuracy and its faster length-decay show.

**This REFRAMES the paper's distinguishing axis (flagged in full in §6 below).** The real
lever separating the S5 winner from the S5 loser is **transition-operator richness
(negative/reflection eigenvalues)** — a Grazzi/DeltaProduct property — **not** the
"delta-correction vs raw-write vs linear/nonlinear state" axis the ablations are framed
around. Both models are linear delta-rule recurrences; the one that *can reach a negative
eigenvalue* tracks S5 and the one that *cannot* does not.

---

## 1. The two transition operators, derived precisely from code

Both layers maintain a per-head matrix state `S ∈ ℝ^{n×d_v}` (here `n = d_v = n_state = 32`)
and update it as an **input-dependent affine map** `S_t = A_t S_{t-1} + B_t`. The state is
linear in `S_{t-1}`, so the per-token transition is captured entirely by the `n×n` matrix
`A_t`. We read off `A_t`, its parameterization, and its achievable eigenvalues from source.

### 1.1 e88-linear (`E88FLAHybrid`, `linear_state=1`, `use_gate=1`)

Winner config (`winners/e88-linear.args.json`): `dim=256, depth=5, n_heads=38, n_state=32,
linear_state=1, use_gate=1`; default `use_conv=False`, `use_silu=True`, `use_l2_norm=True`,
`decay_mode='mamba'`, **no write-gate, no raw-write**.

Serial update (`ndm/models/e88_fla_hybrid.py:1717-1735`):
```
retrieved = S^T k̂                      # read       (:1722)
delta     = v − retrieved               # delta-correct (:1723)
outer     = k̂ ⊗ delta                  # rank-1      (:1726)
S         = decay·S + outer             # linear_state=True (:1733), NO tanh
```
Collecting terms (`S` columns share the same `n×n` map), and matching the affine-scan form
the code itself builds (`:1881-1886`):
```
A_t = decay_t · I  −  k̂_t k̂_tᵀ            (eye·decay − kkᵀ, β implicitly = 1)   :1881-1886
B_t = k̂_t ⊗ v_t                                                                  :1887
```
- **Decay** (`:1402-1403`, Mamba-2 style): `decay_t = exp(−exp(A_log)·softplus(a_proj(x)+dt_bias))`.
  `A_log` init `~U(0,16)` (`:939-941`) ⇒ `exp(A_log)∈(0,16)`; `softplus≥0` ⇒ `decay_t ∈ (0,1)`,
  input- and head-dependent.
- **Rank-1 coefficient is FIXED at 1** — there is no `β`/sigmoid scaling the projection
  (the winner has `use_write_gate=False`), and the decay multiplies **only the `I` term**.

**Eigenvalues of `A_t = decay_t·I − k̂k̂ᵀ`** (`k̂` unit): along the `(n−1)`-dim subspace ⟂ `k̂`,
eigenvalue `= decay_t ∈ (0,1)`; along `k̂`, eigenvalue `= decay_t − 1 ∈ (−1, 0)`. So **every
token contributes exactly one negative eigenvalue**, reaching toward **−1 (a reflection)**
as `decay_t → 0`. Effective rank-1 update / **one reflection per token**; state-rank
increment 1.

### 1.2 GDN (fla `GatedDeltaNet`, `allow_neg_eigval=False` — the default)

Winner config (`winners/gdn.args.json`): `dim=512, depth=6, n_heads=22, n_state=32`;
built by `ndm/models/fla_gated_delta.py:98-108` with `use_gate=True`, `expand_v=1.0`, and
**`allow_neg_eigval` not passed ⇒ default `False`** (`fla/layers/gated_deltanet.py:98,75-77`).

Per-token math (fla recurrent kernel `fla/ops/gated_delta_rule/fused_recurrent.py:104-115`):
```
b_h *= exp(b_g)                          # S ← g_t·S            (:104),  g_t = exp(b_g) ∈ (0,1)
b_v  = β·(b_v − Σ b_h·b_k)               # u = β_t (v_t − k̂ᵀ(g_t S))   (:114)
b_h += b_k·b_v                           # S ← g_t S + k̂ uᵀ     (:115)
```
which is, in matrix form,
```
S_t = g_t·(I − β_t k̂_t k̂_tᵀ)·S_{t-1}  +  β_t k̂_t v_tᵀ
A_t = g_t·(I − β_t k̂_t k̂_tᵀ)
```
- **Decay** `g_t = exp(−exp(A_log)·softplus(a_proj(x)+dt_bias)) ∈ (0,1)` — same Mamba-2 form
  (`fla/layers/gated_deltanet.py:151-165,270`), but it multiplies the **whole** operator.
- **β gate** `β_t = sigmoid(b_proj(x))` (`:266`). With `allow_neg_eigval=False`, **no ×2**, so
  `β_t ∈ (0,1)` (the `×2` that would give `β∈(0,2)` is gated behind the flag, `:267-268`).

**Eigenvalues of `A_t = g_t(I − β_t k̂k̂ᵀ)`**: along ⟂ `k̂`, eigenvalue `= g_t ∈ (0,1)`; along
`k̂`, eigenvalue `= g_t(1 − β_t)`. Since `β_t ∈ (0,1) ⇒ (1−β_t) ∈ (0,1)`, the along-key
eigenvalue is `∈ (0, g_t) ⊂ (0,1)` — **strictly positive; a reflection (eigenvalue −1) is
unreachable**. Same rank-1 / **one (non-reflecting) shrink-along-`k̂`** per token; state-rank
increment 1.

### 1.3 The decisive structural contrast

Both are diagonal-plus-rank-1, one update per token, identical Mamba-2 decay
parameterization. They differ in **two** code-level choices that have the **same**
consequence — foreclosing negative eigenvalues for GDN:

1. **Rank-1 strength.** e88-linear's projection coefficient is **fixed at 1**; GDN's is
   `β_t = sigmoid ∈ (0,1)`, strictly `< 1`.
2. **Where the decay sits.** e88-linear applies decay to the **identity only**
   (`decay·I − k̂k̂ᵀ`), so the along-key eigenvalue is `decay−1 < 0`. GDN applies decay to
   the **whole** operator (`g·(I − β k̂k̂ᵀ)`), so the along-key eigenvalue keeps the sign of
   `(1−β) > 0`.

Net: along the key direction, **e88-linear lands in `(−1,0)` and GDN in `(0,1)`**. The
boundary between them is exactly the eigenvalue=0 line — i.e. the reflection threshold.

---

## 2. Empirical per-step transition spectra on real S5 batches

**Why a (small) retrain.** No weights of the winners were saved — only eval JSON
(`results/s5_symmetric_20260603/eval/*.json`). So we rebuilt each winner config **exactly**
as the winner-eval driver does (`scripts/eval_s5_symmetric_winners.py`), trained it on
`s5_permutation` (T=128, schedule-free AdamW, bf16, the winner recipe) on **one GPU**, and
extracted `A_t`'s eigenvalues on real S5 batches via forward-pre-hooks on the inner
recurrent modules (`experiments/expressivity_tasks/gdn_vs_e88_transition.py`). Because the
along-key **sign** is fixed by the parameterization, it holds at **init** and after
training alike; we report spectra at both, plus the S5 accuracy reached, for full
transparency. (This is a deliberately small run, not the 20 000-step winner-eval; the
converged headline accuracies 0.9997 vs 0.5446 are the canonical numbers in
`S5_SYMMETRIC_RESULTS.md`.)

The eigenvalues are computed in fp32 directly from each module's own
`a_proj/A_log/dt_bias` (decay) and, for GDN, `b_proj` (β), evaluated on the **real hidden
states** the model produces on S5 sequences. For unit `k̂` the eigenvalues depend only on
`decay_t` (e88) and `(g_t, β_t)` (GDN), so the spectrum is exact.

Runs (seed 42, GPU 0, S5 train T=128, eigenvalues extracted on 4 real S5 batches at T=128):
e88-linear trained 200 steps → S5 acc T128 **0.119**; GDN trained 400 steps → S5 acc T128
**0.069** (vs random 1/120 = 0.0083; both above random, e88-linear already ahead — same
ordering as the converged headline 0.9997 vs 0.5446). The runs are deliberately short; the
**sign of the along-key eigenvalue is identical at init and after training**, which is the
whole point — it is fixed by the parameterization, not learned. Raw JSON:
`results/gdn_vs_e88_transition/{e88-linear,gdn}_seed42.json`.

### 2.1 Along-key eigenvalue (the state-tracking-critical direction)

Distribution of the **single along-`k̂` eigenvalue** of `A_t` over all (batch, token, head,
layer); percentiles `p5/p50/p95`, extremes, and the fraction of tokens whose transition is
a genuine (negative) reflection:

| model / phase | min | p5 | p50 | p95 | max | mean | **frac < 0** | frac < −0.25 | frac < −0.5 |
|---|---|---|---|---|---|---|---|---|---|
| **e88-linear** @init    | −0.9997 | −0.7592 | −0.0579 | −0.0018 | −0.0000 | −0.1723 | **1.000** | 0.229 | 0.130 |
| **e88-linear** @trained | **−1.0000** | −0.9903 | −0.0245 | −0.0003 | −0.0000 | −0.2070 | **1.000** | 0.258 | 0.188 |
| **GDN** @init           | +0.0280 | +0.1398 | +0.4580 | +0.6808 | +0.8052 | +0.4426 | **0.000** | 0.000 | 0.000 |
| **GDN** @trained        | +0.0000 | +0.0003 | +0.0175 | +0.6969 | +0.9062 | +0.1426 | **0.000** | 0.000 | 0.000 |

**Total separation, exactly as the algebra predicts.** Every e88-linear token contributes a
**negative** along-key eigenvalue (frac<0 = 1.000); ~19–26 % are strong reflections (< −0.25
to < −0.5), and the extreme reaches **−1.0000 — a perfect Householder reflection**. Every
GDN token has a **strictly positive** along-key eigenvalue (frac<0 = 0.000); its minimum is
0.0 and it never crosses into the reflection half-line. Training sharpens but does not move
the sign: e88-linear is negative at init and stays negative; GDN is positive at init and
stays positive. **There is zero overlap in the sign of the state-tracking-critical
eigenvalue.**

### 2.2 Perpendicular eigenvalue (the `n−1` "memory-decay" directions)

The `(n−1)` eigenvalues ⟂ `k̂` equal `decay_t` (e88) / `g_t` (GDN) — pure decay, expected in
`(0,1)` for both:

| model / phase | min | p5 | p50 | p95 | max | mean | frac < 0 |
|---|---|---|---|---|---|---|---|
| **e88-linear** @trained | +0.0000 | +0.0097 | +0.9755 | +0.9997 | +1.0000 | +0.7930 | 0.000 |
| **GDN** @trained        | +0.0000 | +0.0030 | +0.9791 | +0.9993 | +0.9997 | +0.8877 | 0.000 |

Both sit in `(0,1)` with similar high-retention medians (~0.97–0.98). **The perpendicular
(memory-decay) spectrum is essentially the same for both models** — confirming the
difference is *not* about how strongly they remember, but purely about the **sign reachable
along the key axis** (§2.1).

### 2.3 What the spectra show

- **Effective rank / reflections per token = 1 for both** (one outer-product update per
  token, by construction — `e88_fla_hybrid.py:1726`, `fused_recurrent.py:115`). So the lever
  is **not** the *number* of reflections (DeltaProduct's `n_h`>1 axis); both use a single
  rank-1 step. It is the **sign** that one step can reach.
- **e88-linear realizes reflections (eigenvalue → −1) on every token; GDN never does.** The
  empirical `frac<0` is **1.000 vs 0.000** — a clean, total dichotomy, robust across init and
  training. This is the Grazzi (2025) eigenvalue-range lever, measured directly on the real
  models on real S5 inputs.
- The result **matches the from-code algebra of §1 exactly**: e88's `decay−1 ∈ (−1,0)`
  (extreme −1.0000 observed) and GDN's `g(1−β) ∈ (0,1)` (never negative).

---

## 3. Map to group theory — reflections compose S₅; positive contractions cannot

S₅ is **non-solvable**; its word problem is NC¹-complete (Barrington 1989). A bounded-width
recurrence tracks it only if its per-token transitions **generate a non-solvable subgroup**
of transformations of the state. For a linear recurrence the per-token maps are the `A_t`,
and the relevant structure is their **orthogonal/spectral** part:

- **Reflections generate the full orthogonal group** (Cartan–Dieudonné): products of
  reflections realize every rotation, including the order-2 and order-3 elements whose
  composition is non-abelian and non-solvable. e88-linear's `A_t` has a **−1-approaching
  eigenvalue along `k̂`** — a (scaled) reflection — and the network chooses `k̂_t` per token,
  so the realized transition group **contains reflections at input-selected axes**. That is
  exactly the DeltaProduct/Householder picture (Siems et al. 2025): enough input-selected
  reflections reach the permutation group. With one reflection per token and `T` tokens, the
  composition can express deep permutation words — and indeed e88-linear solves S5 at
  training length and degrades only via the generic length-extrapolation collapse
  (`E1_PARALLEL_SCAN.md`: 0.999@128 → 0.22@1024, identical under an exact associative scan,
  so intrinsic to the operator, not a rounding artifact).

- **Strictly-positive-eigenvalue maps generate only a contractive, abelian-up-to-scale
  semigroup.** GDN's `A_t = g(I − βk̂k̂ᵀ)` has **all eigenvalues in `(0,1)`** — it is a
  positive-definite contraction along every axis. Products of such matrices stay in the
  positive-eigenvalue cone; they can **shrink and re-weight** memory (solvable, counter-like
  behaviour — consistent with GDN doing fine on the S₃ control and on associative recall)
  but they **cannot realize an order-2 reflection or the odd permutations** that make S₅
  non-solvable. This is the finite-precision impossibility of Grazzi et al. (2025) /
  Khavari et al. (2025): a non-negative-eigenvalue linear RNN cannot even do parity, let
  alone S₅ — extending the range to include negatives is the minimal unlock.

So the group-theoretic verdict is sharp: **e88-linear realizes input-selected reflections
that compose into the non-solvable part of S₅; GDN (as configured) stays inside the
positive-eigenvalue / solvable regime and tops out where contraction-only dynamics top
out.**

---

## 4. The smoking gun: GDN *has* the unlock — it is switched off by one flag

The fla `GatedDeltaNet` exposes `allow_neg_eigval` (`fla/layers/gated_deltanet.py:75-77`):
> *"Allow negative eigenvalues. Default: False. If True, beta is multiplied by 2. See
> [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues] (Grazzi 2025)."*

With `allow_neg_eigval=True`, `β_t ∈ (0,2)`, so `(1−β_t) ∈ (−1,1)` and the along-key
eigenvalue `g_t(1−β_t) ∈ (−g_t, g_t)` — **negative becomes reachable**, and GDN would gain
exactly the reflection lever e88-linear has by construction. The winner was trained with
the **default `False`** (it is never set in `fla_gated_delta.py`), so the S5-symmetric GDN
arm was structurally **barred** from the regime that solves S5. e88-linear, by contrast,
gets the negative eigenvalue **for free** from its parameterization (fixed rank-1
coefficient = 1, decay on the identity only) with **no flag required**.

This makes the comparison a **natural realization of the paper's own proposed ablation E5**
(`PRECISION_NONLINEARITY_RESEARCH.md` §6.6/E5: *"constrain transition eigenvalues to [0,1]…
S5 win disappears → the win is the eigenvalue-range mechanism, not rounding"*). GDN **is**
the eigenvalues-in-[0,1] condition, and it **does** fail S5 (0.54), while the negative-eigenvalue
sibling succeeds (1.00). The natural experiment already points the same way E5 predicts.

---

## 5. Falsifiable prediction

If transition-operator eigenvalue sign is the true lever, then **flipping
`allow_neg_eigval=True` on GDN (β→2·sigmoid) should make GDN track S5 markedly better**, and
**clamping e88-linear's along-key eigenvalue to ≥0** (e.g. a `max(decay−1, 0)`-style
projection, or `raw_write` which removes the `−k̂k̂ᵀ` term entirely) **should destroy its S5
win**, while neither change should much affect the S₃ control. This is a one-flag /
one-line experiment on each side; it would convert the correlation here into a causal
demonstration. (Out of scope for this doc — it would require new training runs; logged as a
follow-up.)

---

## 6. ⚠️ FRAMING FINDING THE AUTHOR MUST SEE — does this reframe the distinguishing axis?

**Yes — partially but materially.** The paper's E88 ablation story is organized around
*state* properties: **delta-correction vs raw-write**, and **linear vs nonlinear (tanh)**
state. This GDN-vs-E88 comparison shows that **none of those axes is what separates the S5
winner from the S5 loser here**, because:

- Both e88-linear and GDN are **linear** in the state (no tanh) → not the linear/nonlinear axis.
- Both are **delta-rule** recurrences with a read-modify-write (delta-correction), one
  rank-1 update per token → not delta-vs-raw-write.
- They differ only in the **eigenvalue range of the (still-linear) transition operator**:
  e88-linear reaches **negative/reflection** eigenvalues; GDN (default) is **confined to
  `[0,1]`**.

So the **real distinguishing axis is transition-operator richness — specifically
negative-eigenvalue / reflection reachability (Grazzi 2025; DeltaProduct/Siems 2025) and,
more generally, dense-vs-diagonal structure (Cirone 2024)** — exactly the mechanism flagged
as the live hypothesis in `PRECISION_NONLINEARITY_RESEARCH.md` §4/§6.5, **not** the
delta/raw-write/linear-state framing the ablation tables foreground.

**Recommended reframing for the author (does NOT touch `main.typ` — your call):**
1. State explicitly that **e88-linear and GDN are the same class** (linear gated delta-rule)
   and that the S5 separation is an **eigenvalue-range** phenomenon, not a state-nonlinearity
   or write-rule phenomenon.
2. Attribute the e88-linear S5 win to its transition reaching **negative eigenvalues /
   reflections** (`decay−1 < 0`), citing Grazzi 2025 and DeltaProduct/Siems 2025; note that
   GDN's default `allow_neg_eigval=False` is precisely why it cannot.
3. Treat **GDN as the built-in "eigenvalues∈[0,1]" control** (the E5 ablation, already run)
   and surface the falsifiable one-flag prediction in §5.
4. Keep the delta-correction/raw-write ablations as **secondary** structural knobs — they
   matter for E88's own design, but they are **not** the axis that explains GDN's failure.

This does not overturn the headline result (e88-linear's S5 win is real, intrinsic, and
reproduced under an exact associative scan — `E1_PARALLEL_SCAN.md`). It **relocates the
mechanism**: from "linear/delta state design" to "**the transition operator can go
negative**." If the paper currently attributes the win to delta-correction or to the linear
state per se, that attribution should be corrected to the eigenvalue/reflection mechanism.

---

## 7. Validation checklist

- [x] **Both transition operators written precisely from code** — structure, decay/β/gating
  parameterization, eigenvalue range, reflections/rank — with line citations
  (`e88_fla_hybrid.py:1402-1403,1717-1735,1881-1887`; `fla/layers/gated_deltanet.py:75-77,
  151-165,266-270`; `fla/ops/gated_delta_rule/fused_recurrent.py:104-115`; `fla_gated_delta.py:98-108`).
- [x] **Empirical eigenvalue spectra + effective rank extracted from real S5 batches** on
  trained e88-linear and GDN (rebuilt-and-trained winner configs; weights of the originals
  were not saved) and compared (§2); driver + raw JSON under
  `experiments/expressivity_tasks/gdn_vs_e88_transition.py` and
  `results/gdn_vs_e88_transition/`.
- [x] **Verdict on why GDN fails / e88-linear succeeds**, tied to Grazzi/DeltaProduct/Cirone
  eigenvalue/reflection theory (§3, §1.3, §4).
- [x] **Prominent flag on whether this reframes the paper's distinguishing axis** (§6, top).
- [x] **One GPU used** (GPU 0); **`main.typ` untouched**; committed, **not pushed**.
