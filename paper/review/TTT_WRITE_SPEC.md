# Inner-optimization (test-time-training) WRITE rule — SPEC + kernel plan

**Task:** `ttt-spec` · **Type:** SPEC / DESIGN (no implementation) · **Date:** 2026-06-09
**Role:** Architect (system-design + dependency-analysis) · **Deliverable:** this document.

> One candidate Emender head-type whose **WRITE rule is inner optimization**: the per-token state
> update is one or more **inner-optimizer steps** on an inner reconstruction loss
> `L_inner(S; k_t, v_t)`. The delta-rule is exactly **one gradient step** of
> `L_inner = ½‖v − Sᵀk‖²`; this spec generalizes that single step along four axes — (a) **K inner
> steps**, (b) **inner momentum**, (c) **inner-MLP fast-weights** (nonlinear memory), (d) **alternative
> inner losses** — classifies each honestly as chunkable or sequential, and gives the **fused Triton
> kernel plan** for the chosen chunkable variant (modeled on `ndm/triton/e97_chunked_autograd.py`).
>
> Proposed taxonomy name: **`refit`** — the inner-optimization write (see §5). It is a *third axis*
> ("write-rule order / emendation depth") that composes with the existing eigenvalue×saturation grid;
> the delta-rule heads (GDN-2, `nonlin`, `rot`) are all the **order-1** corner of this axis.

---

## 0. Where this slots in — the delta-rule already in the repo IS test-time training

The grafting target is the same gated-delta substrate the rest of the Emender heads use (FLA
`GatedDeltaNet` projections / short-conv / L2-norm / output-gate / RMSNorm), realized three ways:

| Substrate | File | Write recurrence | Kernel |
|---|---|---|---|
| Native GDN-2 (linear) | `ndm/models/typed_head_mixture.py` (`gdn2_recall`) | `S_t = g_t(I−β_t k̂_t k̂_tᵀ)S_{t-1}+β_t k̂_t v_tᵀ` | FLA `chunk_gated_delta_rule` (diagonal+UT) |
| Nonlinear shell | `ndm/models/gdn2_nonlin_shell.py` | same write, then `S←φ(S)` at chunk bounds | `ndm/triton/gdn2_nonlin_fused.py` |
| E97 split-edit (linear) | `E88FLAHybrid` + `ndm/triton/e97_chunked_autograd.py` | asymmetric gated-delta | chunked Newton–Schulz UT inverse |

The central observation that motivates this head: **every one of these is already a one-step online
learner.** The gated-delta write is identically one SGD step (with weight decay) on an inner
least-squares loss — §1 derives this. Two facts about the *existing* code make the generalization
concrete rather than speculative:

1. The chunk kernel's **Newton–Schulz iterations are inner-optimizer steps.** The e97 kernel inverts
   the strictly-lower-triangular `(I+M)` with `NEWTON_STEPS = ceil(log2 C)` iterations of
   `X ← X(2I − MX)` (`e97_chunked_autograd.py:120-123, 375`). That iteration is the chunk-level inner
   solver: each step doubles the order of the Neumann series for the exact intra-chunk least-squares
   refit. **`NEWTON_STEPS` is therefore literally the inner-step-count knob `K` at chunk granularity**
   (§4) — exposing it is a one-line change, not a new algorithm.
2. The **decay gate `g_t` is the inner loss's ridge / weight-decay term**, and the **`raw_write` flag
   is the inner-loss choice** (Hebbian vs MSE). §2.4 makes both exact. So two switches already in the
   repo (`decay`, `raw_write`) are *already* points on the inner-optimization axis.

This head does not invent test-time training; it **promotes the inner optimizer from an implicit
1-step solver to an explicit, configurable one** and asks which richer settings stay chunkable.

---

## 1. The inner-optimization frame; delta-rule = one inner gradient step

### 1.1 State = fast-weights, write = inner-optimizer step

Frame the recurrent state `S ∈ ℝ^{N×V}` (key dim `N`, value dim `V`) as a **fast-weight matrix**: a
linear associative memory whose read is `read(S; k) = Sᵀ k ∈ ℝ^V`. Each token presents a
(key, value) training pair `(k_t, v_t)`. Define a per-token **inner loss**

```
L_inner(S; k_t, v_t) = ½ ‖ v_t − Sᵀ k_t ‖²                              (inner reconstruction loss)
```

The WRITE rule is one or more steps of an **inner optimizer** that descends `L_inner` from the
running state `S_{t-1}`. This is exactly the test-time-training view of linear attention
(@ttt_sun2024; @wang_test_time_regression2025; @longhorn2024): the memory is *trained at test time*,
one optimization step per token, on the stream of its own (k, v) pairs.

### 1.2 Delta-rule = one gradient step (the derivation)

The gradient of the inner loss w.r.t. the fast-weights is

```
∇_S L_inner = k_t (Sᵀ k_t − v_t)ᵀ = − k_t δ_tᵀ ,     δ_t ≜ v_t − S_{t-1}ᵀ k_t   (the residual)
```

One gradient-descent step with inner learning-rate `β_t`, started at `S_{t-1}`:

```
S_t = S_{t-1} − β_t ∇_S L_inner|_{S_{t-1}}
    = S_{t-1} + β_t k_t δ_tᵀ
    = S_{t-1} + β_t k_t (v_t − S_{t-1}ᵀ k_t)ᵀ
    = (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ                          (the DELTA RULE)
```

This is **identically the DeltaNet write** (@deltanet2024) and, with the decay gate added (§2.4), the
**Gated-DeltaNet write** (@gated_deltanet2024; @gated_deltanet2_2026) used as `gdn2_recall` in
`typed_head_mixture.py`. The delta-rule is **one step of inner gradient descent on
`L_inner = ½‖v − Sᵀk‖²`** — nothing more. This identity is the historical fast-weight-programmer
view (@schmidhuber_1992_fastweights; @schlag_irie_schmidhuber_2021), with the modern least-squares /
online-regression reading made precise in @wang_test_time_regression2025 and @longhorn2024.

The generalization program is now obvious: a single SGD step on a quadratic loss is the *crudest
possible* inner optimizer. §2 enriches it.

---

## 2. Generalization axes (real recurrences)

Four independent axes enrich the inner optimization. Each is given as an exact recurrence and tied
to its prior-art and (where it exists) its already-present repo realization.

### 2.1 Axis (a) — K inner steps

Run `K` inner gradient steps **per token** before committing the state. Two genuinely different
regimes, and the distinction is the crux of the chunkability story (§3):

**(a-i) K steps on a single token's pair `(k_t, v_t)`.** Iterate `S^{(0)} = S_{t-1}`,
`S^{(i+1)} = S^{(i)} + β k_t (v_t − S^{(i)ᵀ}k_t)ᵀ`. Because `L_inner` is a convex quadratic with a
**rank-1 Hessian** `k_t k_tᵀ`, this iteration is exactly solvable. The residual contracts
geometrically: `δ^{(i+1)} = (1 − β‖k_t‖²) δ^{(i)}`, so after `K` steps

```
S_t = S_{t-1} + k_t δ_t^{(0)ᵀ} · β_eff ,   β_eff = [1 − (1 − β‖k_t‖²)^K] / ‖k_t‖²
```

i.e. **a single rank-1 update with a rescaled step `β_eff`.** With L2-normed keys (`‖k_t‖=1`, the
GDN convention) `β_eff = 1 − (1−β)^K`. **Conclusion (important, honest): single-token K-step is
degenerate — it adds no expressivity over choosing `β`, and needs no kernel change.** Multi-step on
one token only "converges harder" to the same rank-1 projection.

**(a-ii) K steps on the chunk's pairs (the non-trivial regime).** Define the **chunk inner loss**
over the `C` causal pairs in a chunk and refit `S` to *all of them at once*:

```
L_chunk(S) = Σ_{t∈chunk} ½ ‖ v_t − Sᵀ k̂_t ‖²       (causally masked; cross-token key interactions)
```

Now the keys are **not** collinear, the Hessian is full-rank, and K steps genuinely improve the fit:
they reduce inter-key cross-talk that one step leaves as residual. **K→∞ is the exact chunk
least-squares refit**, which is precisely what the e97 kernel's UT inverse computes — and its
`NEWTON_STEPS` count is `K` (§0, §4). So axis (a) is real only *across tokens*, and it is *already
chunkable* because the inner loss is quadratic in the linear state.

### 2.2 Axis (b) — inner momentum (heavy-ball / Titans)

Carry a **momentum buffer** `M_t ∈ ℝ^{N×V}` — an exponential moving average of the per-token
surprise (gradient) — and step the fast-weights along it (@titans2025, "momentary + past surprise"):

```
δ_t = v_t − S_{t-1}ᵀ k̂_t                                     (surprise / residual)
M_t = μ_t M_{t-1} + β_t k̂_t δ_tᵀ                             (momentum = EMA of surprise)
S_t = g_t S_{t-1} + M_t                                       (decayed state + accumulated momentum)
```

Setting `μ_t ≡ 0` collapses `M_t = β_t k̂_t δ_tᵀ` and recovers the gated-delta write exactly. This is
the **heavy-ball generalization of the delta-rule**: the write integrates surprise over time instead
of reacting only to the current token. It introduces a **second `N×V` recurrent state** (the
momentum buffer) but the system `(S_t, M_t)` is **still linear** — see §3 for why it stays chunkable
as a second-order linear recurrence (Titans give a chunk-parallel training form). This is the chosen
new lever for the kernel (§4).

### 2.3 Axis (c) — inner-MLP fast-weights (nonlinear memory)

Replace the *linear* memory `read(S;k)=Sᵀk` with a **nonlinear MLP** fast-weight
`f_W(k) = W₂ φ(W₁ k)`, and let the inner optimizer update `W = (W₁, W₂)`:

```
L_inner(W; k_t, v_t) = ½ ‖ v_t − f_W(k_t) ‖² ;   W_t = W_{t-1} − η ∇_W L_inner   (K steps)
```

This is **TTT-MLP** (@ttt_sun2024) and the nonlinear arm of the fast-weight line. The gradient
`∇_W L_inner` is nonlinear in `W` (it passes through `φ′`), so the per-token update is **not** an
associative-scannable linear map. **This variant is sequential** (§3) and is the natural province of
the **nonlinear-memory (`nlmem`) sibling spec** — §8 fixes the interface so the two specs compose
rather than collide. It is reserved here as **`refit-nonlin`** (analogous to the taxonomy's reserved
`rot-nonlin`): named, scoped, not built by this task.

### 2.4 Axis (d) — alternative inner losses

The inner loss is a free choice; several already-named mechanisms are just different `L_inner`:

| Inner loss `L_inner(S;k,v)` | One-step write | Identity / repo realization |
|---|---|---|
| `½‖v − Sᵀk‖²` | `S + βk δᵀ` | **delta-rule** (DeltaNet) |
| `½‖v − Sᵀk‖² + (λ/2)‖S‖²` | `(1−ηλ)S + βk δᵀ` | **gated delta** — `(1−ηλ)=g_t` ⇒ **the decay gate IS ridge / weight-decay** (@longhorn2024 online-regularized view) |
| `−⟨v, Sᵀk⟩` (linear/Hebbian) | `S + βk vᵀ` | **raw-write** (`e97_raw`, `raw_write=True`) — Hebbian outer product, no residual |
| Huber / robust `ρ(v−Sᵀk)` | `S + βk (ρ′(δ))ᵀ` | robust write: per-step nonlinear residual reweight (outlier-robust memory) |
| normalized / cosine `½‖v̂ − (Sᵀk)/‖Sᵀk‖‖²` | reweighted delta | matches L2-normed-value recall |

Two of these (**ridge ⇒ decay**, **Hebbian ⇒ raw-write**) prove that the existing `decay` and
`raw_write` switches are *already* inner-loss choices — the strongest evidence the frame is the
right one. The quadratic losses stay chunkable; **non-quadratic losses (Huber/cosine) make the
per-token write a nonlinear function of the in-chunk residual**, which (like per-step `hardtanh`)
breaks the closed-form chunk solve — sequential (§3).

---

## 3. Chunkability — honest classification per variant

The governing principle, established for this repo's heads in
`fuse-2kernel-tanh-perp-chunkable` (memory) and `phi_shell` ("all per-step φ are
non-chunkable"): **a per-token update is chunk-parallelizable iff it is an affine (linear) map of the
state.** Chunking works by composing per-token transition operators into one per-chunk operator and
prefix-scanning across chunks; that composition is closed only for affine maps. The inner-optimizer
view makes the boundary crisp: **the update is affine ⇔ the inner loss is quadratic AND the memory is
linear.** Steps and momentum preserve affineness; nonlinear loss or nonlinear memory destroy it.

| Variant | Inner loss | Inner steps | Memory | Extra state | Update is affine in S? | **Chunkable** | Kernel |
|---|---|---|---|---|---|---|---|
| delta (baseline) | MSE | 1 / token | linear `S` | — | yes | **YES** | FLA `chunk_gated_delta_rule` / e97 |
| ridge-delta (gated) | MSE+‖S‖² | 1 | linear `S` | — | yes | **YES** | = GDN decay gate (already present) |
| Hebbian (raw) | linear | 1 | linear `S` | — | yes | **YES** | `e97_raw` |
| K-step, single-token | MSE | K | linear `S` | — | yes (collapses to `β_eff`) | **YES** (degenerate) | no change |
| K-step, chunk refit | chunk-MSE | K→exact | linear `S` | — | yes | **YES** | = e97 Newton–Schulz, `NEWTON_STEPS=K` |
| **momentum (heavy-ball/Titans)** | MSE | 1 + EMA | linear `S` | momentum `M [N,V]` | yes (order-2 linear) | **YES** | **e97 + momentum carry ← CHOSEN (§4)** |
| robust inner loss | Huber/cosine | 1 | linear `S` | — | **no** (nonlinear residual reweight) | **NO** | sequential per-step |
| inner-MLP (TTT-MLP) | MSE | K | **nonlinear MLP `W`** | hidden activations | **no** (∇ through `φ′`) | **NO** | sequential → `nlmem` spec (§8) |

**Why momentum stays chunkable.** Stack the state `X_t = [S_t ; M_t]`. The §2.2 recurrence is

```
[S_t]   [ g_t   μ_t ] [S_{t-1}]   [ β_t k̂_t δ_tᵀ ]
[   ] = [           ] [        ] + [             ]
[M_t]   [  0    μ_t ] [M_{t-1}]   [ β_t k̂_t δ_tᵀ ]
```

an **upper-triangular 2×2 companion per channel** with eigenvalues `g_t` and `μ_t`. Its
chunk-cumulative operator is again upper-triangular (two cumulative-product factors plus one cross
term) — structurally the **same shape as the complex-eig head's "two prefix sums"**
(`COMPLEX_EIG_HEAD_SPEC.md` §3.1), only with a real `(g, μ)` companion instead of a complex twiddle.
The surprise `δ_t` couples through `S_{t-1}` exactly as in plain delta, so the **intra-chunk solve is
the same nilpotent UT system**, with the score-matrix entries gaining a `μ`-discounted cumulative
factor alongside the existing `g`-decay factor. Associativity holds because the per-token map is
affine. ⇒ momentum is chunkable, at the cost of one extra carried `[N,V]` tile.

**The chosen variant for the fused kernel is the linear chunkable family: momentum-delta (axis b)
with the inner-step count `K` exposed via `NEWTON_STEPS` (axis a-ii), default inner loss = ridge-MSE
(axis d, = the existing decay gate).** It is the richest inner optimizer that remains GDN-2-class
throughput. The nonlinear arms (axes c, robust-d) are deliberately handed to the sequential
`nlmem`/`refit-nonlin` track.

---

## 4. Fused Triton kernel plan (chosen variant) — model: `e97_chunked_autograd.py`

Extend `ndm/triton/e97_chunked_autograd.py`. That kernel already fuses the chunked delta scan (fwd:
`_e97_fwd_save_kernel`; bwd: `_e97_bwd_kernel`, reverse chunk walk threading `dS` in registers,
recompute-forward intermediates, every heavy step a `tl.dot` on tensor cores). The `refit` head adds
**(i)** an exposed inner-step knob and **(ii)** a momentum buffer carried across chunks. Both are
local additions; the UT/Newton–Schulz machinery and the SMEM discipline are reused verbatim.

### 4.1 What changes vs the e97 kernel

| Change | e97 today | `refit` |
|---|---|---|
| Inner steps `K` | `NEWTON_STEPS = ceil(log2 C)` (always exact) | **expose `K`**: `NEWTON_STEPS = min(K, ceil(log2 C))`; `K<ceil(log2 C)` ⇒ *truncated Neumann* = approximate inner solve (cheaper, fewer matmuls). A `constexpr`. |
| Momentum buffer `M` | none | carry a second fp32 tile `M : [N,V]` per `(b,h)` alongside `S`; update per §2.2 |
| Cross-chunk carry | `S ← exp(G_last)·S + SdK` | carry **both**: `M ← μ̄·M + dM_chunk`, `S ← exp(G_last)·S + M_contrib` via the 2×2 `(g,μ)` companion (§3) |
| Score matrix | `M_score[t,j]=exp(G[t]−g[t]−G[j])·(k_j·u_t)` | add the `μ`-cumulative factor `exp(Gμ[t]−gμ[t]−Gμ[j])` as a second twiddle (one extra `cumsum(log μ)` prefix, like `Φ` in complex-eig) |
| New gradient sinks | dk,dq,dv,decay,erase,write | + `dμ` (momentum gate) flowing through the `Gμ` prefix, exactly as `dG` flows through `G` (`e97_chunked_autograd.py:296-308`) |

### 4.2 Forward kernel (per chunk, per `(b,h)`)

1. **Two prefix sums.** Keep `G[t]=cumsum(log g_t)` (present). Add `Gμ[t]=cumsum(log μ_t)`; `μ_t` is
   precomputed outside the kernel (a pointwise gate, like `g`) and passed in. Compute `exp` factors
   once per tile.
2. **State tiles.** Maintain `S [N,V]` (present) and `M [N,V]` (new), both fp32 in registers.
3. **Twiddled UT matrix.** Where e97 forms `M_score = decay_factor · (k_jᵀ u_t)` (lines 114-116),
   multiply in the momentum-cumulative factor: `M_score[t,j] = exp(G[t]−g[t]−G[j]) ·
   exp(Gμ[t]−gμ[t]−Gμ[j]) · (k_jᵀ u_t)` (strictly lower). Structure unchanged ⇒ still nilpotent.
4. **Inner-step solve (= Newton–Schulz).** Run `X ← X(2I − M_score·X)` for `NEWTON_STEPS = K` steps
   (lines 120-123). `K = ceil(log2 C)` ⇒ exact refit; `K<` ⇒ truncated (the explicit inner-step
   knob). This loop **is** the inner optimizer.
5. **Momentum-aware delta + read.** Form `Δ = Tmat·RHS` (lines 125-127) with `RHS` including the
   momentum carry-in from `M`; emit `out_t = γ_t·QS + A·Δ` (lines 129-131) unchanged.
6. **Cross-chunk carry of `(S, M)`.** Apply the 2×2 `(g,μ)` companion (§3): update `M` with the
   chunk's accumulated surprise discounted by `Gμ`, then `S ← exp(G_last)·S + (momentum contribution)`
   (generalizes line 136). Store per-chunk **entry** `S` *and* `M` to HBM (the only things the
   backward needs; mirrors `Sentry_ptr`, add `Mentry_ptr`).

### 4.3 Backward kernel

Reverse chunk walk (lines 181-331), threading **two** state gradients `dS` and `dM` in registers
(init from `dS_final`, `dM_final`). Recompute the forward per-chunk intermediates (Tmat via
Newton–Schulz, score matrices, `Δ`) exactly as e97 does (lines 200-239); the only new VJP work is:

- propagate `dM` through the `(g,μ)` companion to `dS_prev`, `dM_prev` and to `dμ` (the
  momentum-gate grad), structurally identical to how `dG` is assembled from the decay factors
  (lines 296-308): `dμ` accumulates from the `Gμ` cumulative factor in the score matrix plus the
  cross-chunk companion term;
- everything else (`dk,dq,dv,decay,erase,write`) is the existing VJP unchanged.

Work in **log-space for both gates** (`g` and `μ`): the kernel already returns grad-wrt-log-decay to
avoid the `dg/decay` blow-up as `decay→0` (`e97_chunked_autograd.py:315-320, 404`); the same `_GLOG_FLOOR`
discipline (line 47) applies to `log μ` to keep `exp(−gμ)` from overflowing fp32.

### 4.4 Tile budget, `C`, parity

- **Tiles.** One extra carried `[N,V]` fp32 tile (`M`) in fwd and `dM` in bwd; with `N,V ≤ 64` (the
  kernel's scope, line 346) these are ≤16 KB each. The dominant `[C,C]` UT tiles are **unchanged** in
  number, so the SMEM pressure that fixes `C=32` on Ada/Ampere (the fused bwd holds ~6-7 `[C,C]` fp32
  tiles, ~106 KB at `C=64` > the 100 KB limit; lines 424-431) is **not worsened** by the small extra
  `[N,V]` state. **Keep `C=32` default**, `C=64` on Hopper.
- **One extra prefix sum** (`Gμ` cumsum) per tile + a handful of pointwise `exp`/multiply ops. FLOPs
  ≈ unchanged (the `[C,C]` matmuls dominate and are unchanged); expect **GDN-2-class throughput**, the
  same band as e97 (0.33–0.53× GDN-2 wall = *faster*; `E97_CHUNKED_KERNEL_NOTE.md`). Momentum is a
  *constant additive* overhead, not a per-`C` penalty (unlike the per-step-tanh sequential tax that
  closed `e97-wallclock-cma-shell-flat`).
- **Identity fast-path.** Gate the momentum tiles on a `momentum: tl.constexpr`; when `μ≡0` and
  `K=ceil(log2 C)` the kernel compiles to the **exact e97 delta path** (zero overhead — the same
  `complex=False`-style fast-path discipline used in `gdn2_nonlin_fused`).
- **Parity tests** (mirror `tests/test_e97_chunked.py`, **REAL data only, no mocks**): (a) `μ≡0,
  K=exact` ⇒ bitwise-near FLA `chunk_gated_delta_rule` (fp32 rel-err < 1e-5). (b) `μ>0, K=exact` ⇒
  matches an eager `for t in range(T)` heavy-ball reference (§2.2) to fp32 rel-err < 3e-3, bf16/TF32
  < 5%. (c) `μ≡0, K<exact` ⇒ matches a truncated-Neumann reference (verifies the inner-step knob).
  (d) gradcheck on `μ`-gate (`dμ`), `decay`, `β` via the eager reference.

---

## 5. Naming — propose `refit`, a third taxonomy axis

The current taxonomy (`EMENDER_TAXONOMY.md`) names heads by the **state-map dynamics**: eigenvalue
placement (`decay`/`reflect`/`rot`) × saturation (linear / `-nonlin`). The inner-optimization write
is **orthogonal to both** — it is a property of the *write rule* (the input map / correction), not of
the state transition's eigenvalues or saturation. The delta-rule already lives inside *every* current
head; this axis asks **how rich the per-token correction is**.

> **Proposed third axis — write-rule order ("emendation depth").** The Emender's per-token correction
> is an inner-optimizer update on `L_inner`. Its order:
> - **order-1 (`delta`, implicit/default):** one gradient step, MSE loss. Every current head
>   (`decay`, `reflect`, `nonlin`, `rot`) is order-1. This is the existing GDN-2 / e97 write.
> - **order-K+momentum (`refit`):** K inner steps and/or momentum on the linear (quadratic) inner
>   loss — **chunkable**, this spec's head.
> - **nonlinear inner (`refit-nonlin`, reserved):** inner-MLP fast-weights or non-quadratic loss —
>   **sequential**, the `nlmem` sibling spec (§8). Reserved exactly as `rot-nonlin` is: added only
>   once the axis earns it.

`refit` composes with the eigenvalue×saturation grid: one can have `decay-refit`, `reflect-refit`,
`rot-refit` (complex eigenvalue + momentum inner optimizer). The name is read from the dynamics of
the *write*: the head **re-fits** its fast-weight memory to each token's (k,v) by inner optimization,
rather than nudging it once. (Alternatives considered: `relax` — iterative relaxation toward the
inner-loss minimum; `accel` — heavy-ball acceleration. `refit` chosen as the clearest umbrella for
"inner optimization / test-time training of the memory.") This proposal is offered for ratification
by the taxonomy-doc owner; it does not rename any code.

---

## 6. Capability hypotheses — what richer emendation should unlock that GDN can't

GDN's single delta step is a greedy one-shot online learner. The hypotheses below are the
capabilities a *better inner optimizer* should add, to be checked on the existing probe harness (MQAR
recall, S5, modular_quadratic, + the long-context-interference probes below). **REAL train+eval
only.**

1. **High-load / correlated-key recall (the K-step / chunk-refit win).** One delta step leaves
   residual cross-talk when stored keys are non-orthogonal; the exact chunk refit (K→exact) removes
   it. **Hypothesis:** `refit` solves MQAR at higher key-correlation and higher memory load (more
   pairs per state) than 1-step delta, because the inner optimizer actually *minimizes* the
   reconstruction loss instead of taking one step toward it. This is the in-context-regression /
   mesa-optimization capability (@von_oswald_mesa2023; @von_oswald_icl_gd2023;
   @wang_test_time_regression2025).
2. **Long-context retention under interference (the momentum win).** Heavy-ball integrates surprise
   over time, so a single noisy or contradictory token cannot overwrite a well-supported memory
   (@titans2025). **Hypothesis:** `refit`-momentum beats 1-step delta on "needle with distractors" /
   drifting-key streams and on long-range recall where the answer was written far back and
   interfered with since.
3. **Adaptive emendation strength (data-dependent `K`/`β_eff`, `μ`).** A learned per-token inner
   learning-rate / momentum lets the head correct hard on surprising tokens and barely on redundant
   ones — better sample efficiency on bursty/structured streams. **Hypothesis:** `refit` is more
   sample-efficient (steeper probe-accuracy-vs-tokens) on structured-recall tasks even where its
   converged accuracy ties GDN.
4. **Complement, not replacement.** `refit` enriches *how* memory is written (linear inner optimizer);
   the `rot` head enriches *what eigenvalues* the state carries (periodicity/position); `nonlin`/`-nonlin`
   add bounded-saturation depth. They are orthogonal — `refit` composes with all three, and the
   nonlinear inner-MLP capability (deep nonlinear memory) is the separate `nlmem` bet (§8).

---

## 7. The explicit prediction — TIE on OOS bpb (convergent-loss null)

Consistent with the entire E88→E99 line — **every exotic head ties the GDN-2 baseline on LM
bits-per-byte at matched params and tokens**, with capability showing only on probes
(`e97-nonlin-in-time-separates-modquad`: "LM-BPB tie ≠ capability equivalence";
`e97delta-1p3b-tie-split`; `e97-wallclock-cma-shell-flat`; `e99-1p3b-controls`; `COMPLEX_EIG_HEAD_SPEC.md`
§7).

> **Prediction (commit it):** at matched parameter budget and matched tokens, the `refit`
> (momentum / K-step linear-TTT) head **TIES** `gdn2-mlp` / `gdn-neg` on out-of-sample LM
> bits-per-byte (within seed noise, |ΔBPB| ≲ 0.02). **Separation appears only on the §6 capability
> probes** — high-load/correlated-key recall (chunk-refit), long-context-under-interference
> (momentum), and sample-efficiency (adaptive strength) — where `refit` is predicted to beat 1-step
> GDN.

This is the **convergent-loss null**: at convergence the MLP + attention layers absorb the
write-rule difference, so LM-BPB is the wrong instrument; the accept/reject decision is decided on the **probe panel**.
`refit`'s distinctive bet (shared with `rot`, unlike the per-step-tanh / inner-MLP heads): the chosen
variant is **chunkable** (§3), so a probe win costs **no wall-clock** — it does not reintroduce the
sequential-kernel tax that produced the wall-clock losses in `e97-wallclock-cma-shell-flat` and
`fuse-2kernel-tanh-perp-chunkable`. A `refit` head that *changed* LM-BPB would be a surprise to
double-check, not the expected result.

---

## 8. Coordination with the `nlmem` (nonlinear-memory) spec

Axis (c) — inner-MLP fast-weights — is **deliberately out of scope for this chunkable cell** and is
the subject of the sibling nonlinear-memory spec. The clean division of labor:

| | `refit` (this spec) | `nlmem` / `refit-nonlin` (sibling) |
|---|---|---|
| Memory | linear `S` (+ momentum `M`) | nonlinear MLP `W=(W₁,W₂)`, `f_W(k)=W₂φ(W₁k)` |
| Inner loss | quadratic (MSE/ridge/Hebbian) | MSE (or any) through `φ` |
| Update | affine in state | nonlinear in state |
| **Chunkable** | **YES** (§3) | **NO** — sequential fused scan (TTT-MLP, @ttt_sun2024) |
| Kernel | e97 + momentum carry (§4) | sequential per-step (the `phi_shell` / `gdn2_nonlin_shell` latency-bound family) |
| Throughput | GDN-2-class | latency-bound; needs stream-overlap (`hetero-kernel-hits-095x`) to hide its tax |

**Shared interface (so the two specs compose):** both are points on the **write-rule order axis**
(§5); both reuse the FLA-projection substrate (§0). A mixture layer can run `refit` heads on the
chunked tensor-core path and `refit-nonlin` heads on a side CUDA stream — exactly the hetero split
the complex-eig spec uses for its `hardtanh` subset (`COMPLEX_EIG_HEAD_SPEC.md` §5). The `nlmem`
spec should: (1) adopt the `L_inner`/inner-optimizer vocabulary and the §1 delta-rule derivation as
shared preamble; (2) own axis (c) and the non-quadratic-loss arm of axis (d); (3) name its head
`refit-nonlin` (or coordinate a name) under the §5 third axis. This spec owns the linear/chunkable
arm only.

---

## 9. Validation checklist (this spec)

- [x] **Delta-rule = one inner gradient step** — `∇_S L_inner = −k δᵀ`, one step ⇒ the gated-delta
      write, derived in §1.2; tied to DeltaNet/GDN/fast-weight prior art.
- [x] **Generalization axes with real recurrences** — (a) K steps: single-token collapses to `β_eff`,
      chunk-refit = exact least-squares (§2.1); (b) momentum: heavy-ball/Titans recurrence (§2.2); (c)
      inner-MLP: TTT-MLP, nonlinear (§2.3); (d) inner-loss table incl. ridge⇒decay, Hebbian⇒raw-write
      (§2.4).
- [x] **Chunkable-vs-sequential honestly classified per variant** — full table §3 with the affineness
      criterion; momentum chunkable via the `(g,μ)` companion proof; robust-loss & inner-MLP sequential.
- [x] **Fused Triton kernel plan for the chosen variant** — momentum-delta + exposed `K` on the e97
      chunked kernel: prefix sums, twiddled UT, Newton–Schulz = inner steps, `(S,M)` carry, backward
      VJP, tile budget, identity fast-path, parity tests (§4).
- [x] **Real citations** — TTT (@ttt_sun2024), Longhorn (@longhorn2024), Titans (@titans2025),
      DeltaNet-as-online-learning (@deltanet2024, @schlag_irie_schmidhuber_2021,
      @wang_test_time_regression2025), mesa-optimization (@von_oswald_mesa2023, @von_oswald_icl_gd2023),
      fast-weight ancestry (@schmidhuber_1992_fastweights, @widrow_hoff_1960) — §10.
- [x] **Capability hypotheses + convergent-loss-null prediction** — §6 (load/correlated recall,
      interference, adaptive strength) and §7 (TIE on bpb, separate on probes, chunkable ⇒ no
      wall-clock tax).
- [x] **Named per taxonomy** — `refit`, proposed third axis (§5); nonlinear arm reserved as
      `refit-nonlin`, coordinated with `nlmem` (§8).

---

## 10. Citations (real)

Present in `paper/refs.bib`:

- `@deltanet2024` — Yang, Wang, Zhang, Shen, Kim. *Parallelizing Linear Transformers with the Delta
  Rule over Sequence Length.* arXiv:2406.06484, 2024. (chunked delta-rule = the chunkable inner solve)
- `@gated_deltanet2024` — Yang, Kautz, Hatamizadeh. *Gated Delta Networks.* arXiv:2412.06464, 2024.
- `@gated_deltanet2_2026` — Hatamizadeh, Choi, Kautz. *Gated DeltaNet-2.* arXiv:2605.22791, 2026.
- `@titans2025` — Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.*
  arXiv:2501.00663, 2025. (inner momentum / surprise; chunk-parallel memory)
- `@schmidhuber_1992_fastweights` — Schmidhuber. *Learning to Control Fast-Weight Memories.* Neural
  Computation 4(1), 1992.
- `@schlag_irie_schmidhuber_2021` — Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast
  Weight Programmers.* ICML 2021. (delta-rule as fast-weight programming)
- `@widrow_hoff_1960` — Widrow, Hoff. *Adaptive Switching Circuits.* IRE WESCON, 1960. (the LMS /
  one-step-gradient ancestor of the delta-rule)

**To add to `paper/refs.bib`** (real, not yet in the file — flag for the bib owner):

- `@ttt_sun2024` — Sun, Li, Dalal, Xu, Vikram, Zhang, Dubois, Chen, Wang, Koyejo, Hashimoto,
  Guestrin. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* arXiv:2407.04620,
  2024. (TTT-linear and TTT-MLP — the inner-optimization frame and the nonlinear arm)
- `@longhorn2024` — Liu, Wang, Wu, Feng, Stone, Liu. *Longhorn: State Space Models Are Amortized
  Online Learners.* arXiv:2407.14207, 2024. (gated linear recurrence as regularized online regression)
- `@wang_test_time_regression2025` — Wang, Shi, Fox. *Test-time Regression: A Unifying Framework for
  Designing Sequence Models with Associative Memory.* arXiv:2501.12352, 2025. (delta-rule = one step
  of online least-squares; the umbrella for this spec)
- `@von_oswald_icl_gd2023` — von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov,
  Vladymyrov. *Transformers Learn In-Context by Gradient Descent.* ICML 2023, arXiv:2212.07677.
- `@von_oswald_mesa2023` — von Oswald, Schlegel, Meulemans, et al. *Uncovering Mesa-Optimization
  Algorithms in Transformers.* arXiv:2309.05858, 2023. (in-context inner optimization = mesa-opt)
