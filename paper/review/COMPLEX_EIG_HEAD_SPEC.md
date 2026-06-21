# Complex-eigenvalue gated-delta head — SPEC + kernel plan

**Task:** `complex-eig-spec` · **Type:** SPEC / DESIGN (no implementation) · **Date:** 2026-06-09
**Role:** Architect (system-design + dependency-analysis) · **Deliverable:** this document.

> One Emender head-type: a **complex-eigenvalue (rotation–scaling) gated-delta head**, plus the
> arrangement **"complex eigenvalues on ALL heads, per-step `hardtanh` on a SUBSET of heads."**
> Complex eigenvalues are a *known* primitive (S4 / S5 / LRU diagonal scan); this spec does **not**
> re-derive whether they work — it specifies exactly how to graft them onto the existing
> gated-delta head and the existing chunked hetero kernel, with enough precision that an
> implementer can follow it without further design decisions.

---

## 0. Where this slots in (substrate, verified against code)

The grafting target is the gated-delta head as realized three ways in the repo, all of which
share the SAME FLA `GatedDeltaNet` projections / short-conv / L2-norm / output-gate / RMSNorm:

| Substrate | File | Recurrence | Kernel |
|---|---|---|---|
| Native GDN-2 (linear) | `ndm/models/typed_head_mixture.py` (`gdn2_recall`) | `S_t = g_t(I−β_t k̂_t k̂_tᵀ)S_{t-1}+β_t k̂_t v_tᵀ` | FLA `chunk_gated_delta_rule` (diagonal+UT) |
| Nonlinear shell | `ndm/models/gdn2_nonlin_shell.py` | same write, then `S←φ(S)` at chunk boundaries | `ndm/triton/gdn2_nonlin_fused.py` |
| Per-step φ | `ndm/models/phi_shell.py` | `S_t = φ(diag(g_t)S_{t-1}+β_t(v_t−S_{t-1}k_t)k_tᵀ)` | sequential (non-chunkable) |
| E97 split-edit (linear) | `E88FLAHybrid` + `ndm/triton/e97_chunked_autograd.py` | asymmetric gated-delta | chunked Newton–Schulz UT inverse |

The current FLA transition's **diagonal part is a single real scalar per head per step**:
`g_t = exp(−exp(A_log)·softplus(a_proj(x)+dt_bias)) ∈ (0,1)` (see `gdn2_nonlin_shell._project`,
lines 196–199), and `allow_neg_eigval=True` doubles β so the *along-key* eigenvalue
`g_t(1−β_t)` can reach `(−g_t, g_t)` (negative ⇒ reflection — the validated S5 / tracking lever,
`paper/review/EIGENVALUE_CAUSAL_TEST.md`).

**This spec replaces that real scalar/per-channel decay with a per-key-channel COMPLEX eigenvalue.**
Everything else in the head is reused verbatim. The complex head is a new head type
`cplx_gdn` plus a config flag that makes the complex transition the *default for all head types*;
the per-step `hardtanh` subset is the `phi_shell` per-step map applied to a configurable fraction
of heads. The arrangement reuses the `typed_head_mixture` stream-overlap machinery
(`overlap_streams`, lines 404–457) because the complex bulk is chunkable (tensor-core bound) and
the `hardtanh` subset is not (latency bound) — exactly the hetero-kernel split that
`hetero-kernel-hits-095x` already exploits.

---

## 1. Transition: real decay → complex eigenvalue λ = r·e^{iθ}

### 1.1 State, channel pairing

Per head, key dim `N = d_k` (production `N = 32`), value dim `V = d_v` (`32`). Pair the `N` real
key channels into `N/2` **complex channels**. The recurrent state becomes complex:

```
S_t ∈ ℂ^{(N/2)×V}          (equivalently a real tensor [N/2, V, 2] holding (Re, Im),
                            or a 2×2-real-block tensor [N/2, V, 2, 2] — see §3.4)
```

The query/key are complexified by the same pairing: `q_t, k_t ∈ ℂ^{N/2}` formed from the `N`-dim
FLA projection output as `k_t[j] = k_t^{re}[2j] + i·k_t^{re}[2j+1]` (and likewise `q`). The value
`v_t ∈ ℝ^V` is lifted to `ℂ^V` with zero imaginary part (`v` carries content, not phase). L2-norm
on `q,k` is applied to the **complex magnitude** `‖k‖ = sqrt(Σ_j |k_j|²)` so `θ=0` reduces exactly
to the real head's `use_qk_l2norm_in_kernel` normalization.

### 1.2 The complex eigenvalue (the only changed mechanism)

The per-step diagonal transition is a complex diagonal `Λ_t = diag(λ_{t,1},…,λ_{t,N/2})` with

```
λ_{t,j} = r_{t,j} · e^{ i θ_{t,j} }          r ∈ (0,1) magnitude,  θ ∈ ℝ phase
```

equivalently the 2×2 real **rotation–scaling block**

```
       ⎡ r cosθ   −r sinθ ⎤
Λ_j =  ⎢                   ⎥        (acts on the (Re,Im) pair of channel j)
       ⎣ r sinθ    r cosθ ⎦
```

**Magnitude `r_{t,j}` — the decay gate (reuse GDN, per-channel).**

```
r_{t,j} = exp( − exp(A_log_j) · softplus( a_proj(x_t)_j + dt_bias_j ) ) ∈ (0,1)
```

This is the EXACT GDN decay formula (`gdn2_nonlin_shell._project` line 198), only **per-channel**
(`A_log`, `dt_bias`, and `a_proj` output dim become `N/2` per head instead of one scalar). `r<1`
guarantees the complex spectral radius `|λ| = r < 1` ⇒ a contraction ⇒ BIBO-stable scan (the
S5/LRU stability condition). No extra clamp needed; `r∈(0,1)` is automatic from the exp.

**Phase `θ_{t,j}` — base frequency grid + bounded input modulation.**

```
θ_{t,j} = θ_j^base  +  Δ · tanh( W_θ x_t )_{j}              (per head)
```

- `θ_j^base` — a **learnable per-channel base frequency** (the "clock grid"), one scalar per
  complex channel per head, initialized log-spaced (§1.3). This is the content-independent
  positional/periodic prior (the RoPE / S4 / S5 frequency ladder).
- `Δ · tanh(W_θ x_t)_j` — a **bounded** input-dependent perturbation. `W_θ: dim → (N/2)·n_heads`
  is a new projection; `tanh` bounds it to `(−1,1)` and the scalar `Δ` (default `Δ = π/8`) caps
  the per-step phase drift so the input can *retune* the clock (selective rotation, the Mamba-style
  data dependence) without letting θ run away. Setting `Δ=0` recovers a pure fixed-frequency LRU
  channel; setting `Δ=π` recovers full input control of the sign (subsumes `allow_neg_eigval`, §4).

### 1.3 Initialization of θ_j^base (S4 / RoPE-style log-spaced grid)

The grid must cover periods from the shortest resolvable (2 steps, θ=π) up to a long-context
period (~`T_max` steps, θ near 0), log-spaced so resolution is dense at low frequency (long
range) and present at high frequency (short range) — the S4/RoPE design:

```
for channel j = 0 .. N/2-1:
    # log-spaced angular frequencies, RoPE convention with base B (default B=10000)
    θ_j^base = θ_max · B^{ −2j / (N/2) }                     # θ_max = π  (Nyquist)
            ⇔ period_j = 2π / θ_j^base  ranges  [2, 2π·B^{(N/2-2)/(N/2)} ]
```

**Init refinement (preserve the real-positive subspace at init).** Reserve a configurable
fraction `f_dc` of channels (default `f_dc = 0.5`) at `θ_j^base = 0` (pure real-positive decay =
exact GDN behavior) and log-space the remaining `(1−f_dc)·N/2` channels over `(0, π]`, including a
few channels at/near `θ=π` (reflection = negative eigenvalue). Consequence: **at init the complex
head is a strict superset of the real GDN head** (half the channels behave identically to GDN), so
it cannot start worse; training moves channels off `θ=0` only where rotation helps. `θ_j^base` is
in the weight-decay-exempt group (like `A_log`, `dt_bias`).

### 1.4 Exact projection list (per head)

Reused verbatim from FLA `GatedDeltaNet` (no change): `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`b_proj` (β), `g_proj` (output gate), short-conv `q/k/v_conv1d`, `o_norm` (RMSNorm), L2-norm.

Changed / added:

| Symbol | Shape (per head) | Role | Status |
|---|---|---|---|
| `A_log` | `N/2` | per-channel log-decay base (was scalar) | widened |
| `dt_bias` | `N/2` | per-channel softplus bias (was scalar) | widened |
| `a_proj` | `dim → (N/2)·H` | per-channel magnitude gate (was `→H`) | widened |
| `θ_base` | `N/2` | log-spaced base-frequency grid (§1.3), learnable, WD-exempt | **new** |
| `W_θ` (`theta_proj`) | `dim → (N/2)·H` | bounded phase modulation `Δ·tanh(W_θx)` | **new** |
| `Δ` | scalar (buffer) | phase-drift cap, default `π/8` | **new** |

Added params per head ≈ `dim·(N/2)·H` (the `θ_proj`) + `N` (θ_base, A_log, dt_bias widening) —
i.e. one extra `dim→(N/2)H` projection, ~the size of `a_proj`. Matched in the CMA budget driver by
deriving `dim` to the target, identical to how `typed_head_mixture` already matches the split-gate
asymmetry (`typed_head_mixture.py` docstring lines 47–50).

---

## 2. Delta correction on the complex state + read

The gated-delta read-modify-write is kept; only the inner products become **Hermitian** (use the
conjugate transpose on the key axis) so they reduce to the real inner product at θ=0.

```
# read (delta retrieval) — conjugate on the key axis:
retrieved_t = k_t^H S_{t-1}            ∈ ℂ^V        (k_t^H = conj(k_t)ᵀ,  row vec · matrix)
delta_t     = v_t − retrieved_t        ∈ ℂ^V        (v_t real, lifted)

# write (gated-delta, rank-1 along the complex key), decay = complex diagonal Λ_t:
S_t = Λ_t ⊙_row ( S_{t-1} − β_t k̂_t (k̂_t^H S_{t-1}) ) + β_t k̂_t v_tᵀ
    = (Λ_t (I − β_t k̂_t k̂_t^H)) S_{t-1} + β_t (Λ_t k̂_t) v_tᵀ
```

where `Λ_t ⊙_row` scales row `j` of the state by `λ_{t,j}` (the diagonal complex eigenvalue acting
on the key axis). `β_t ∈ (0,2)` scalar per head per step is reused unchanged (with `allow_neg_eigval`
doubling). The **transition operator** `A_t = Λ_t(I − β_t k̂_t k̂_t^H)` has:
- *perpendicular* (to k̂) eigenvalues `= λ_{t,j} = r_{t,j} e^{iθ_{t,j}}` — the placed complex spectrum;
- *along-k̂* eigenvalue `= (k̂_t^H Λ_t k̂_t)(1−β_t)` — the delta/recall direction, β-controlled.

This is the principled **delta** form (`raw_write=False`). The `raw_write=True` variant
(`S_t = Λ_t S_{t-1} + β_t k̂_t v_tᵀ`, drop the read term) is also available, mirroring `e97_raw`.

### 2.1 Read-out: `Re` vs `[Re; Im]` (decision: `Re`, justified)

```
o_t = Re( q_t^H S_t )  ∈ ℝ^V                          (chosen)
```

**Justification for `Re` over `[Re;Im]` concat:**

1. **Parity / unification.** At `θ≡0` (all real-positive), `q,k` are real, `S` is real,
   `q_t^H S_t = q_tᵀ S_t`, and `o_t` is identically the existing real GDN read. `[Re;Im]` would
   emit a phantom `Im` channel that is zero in the real limit — it would not reduce to GDN.
2. **Head-shape matching.** The mixture (`typed_head_mixture`) requires every head to contribute
   `V=32` readout dims so type fractions stay comparable (docstring lines 45–50). `Re` gives
   exactly `V`; `[Re;Im]` gives `2V` and breaks the matching (and doubles `o_proj`).
3. **No information is lost.** `Re(q^H S) = Σ_j |q_j||S_j| cos(∠q_j − ∠S_j)`: the **real part of a
   complex inner product already encodes the phase alignment** between the (learned-phase) query and
   the state. Because `q`'s phase is itself a free projection, the network selects which phase
   component to read — the `Im` part is a 90°-rotated copy reachable by rotating `q`. This is exactly
   the **S5 / LRU convention**: those models read `Re(C·h)` from a complex diagonal state, never
   `[Re;Im]`. We follow it.

`[Re;Im]` is documented as a one-line ablation (`read_mode='reim'`, doubles `V`) for the
phase-regression probe only, not the default.

### 2.2 Kept verbatim from gated-delta

L2-norm on `q,k` (complex magnitude, §1.1); β gate with `allow_neg_eigval` doubling; output gate
`o = o_norm(o, g_proj(x))`; `o_proj`; RMSNorm; short-conv on q/k/v. The only deltas vs the linear
GDN head are: complex diagonal `Λ_t` instead of scalar `g_t`, conjugate `^H` on the two inner
products, and `Re(·)` on the read.

---

## 3. Chunkability — the complex diagonal scan is the standard S5/LRU associative scan

### 3.1 Cumulative complex transition = magnitude-product × phase-sum

The **diagonal** part of the transition (the eigenvalue placement, what governs cross-chunk state
carry) is multiplicative and per-channel. Over a span τ = a..b:

```
∏_{τ=a}^{b} diag(λ_τ) = diag( ∏_{τ} r_τ · e^{ i Σ_τ θ_τ } )
                      = diag( exp( Σ_τ log r_τ )  ·  e^{ i Σ_τ θ_τ } )
```

So the cumulative transition is **(cumulative log-magnitude) + (cumulative phase) →
exp/cis** — two independent prefix sums per channel:

```
G[t]  = Σ_{i≤t} log r_i           (cumulative log-magnitude;  EXISTS today as the GDN log-decay cumsum)
Φ[t]  = Σ_{i≤t} θ_i               (cumulative phase;          NEW — one extra prefix sum)
λ-product over (j,t] = exp(G[t]−G[j]) · ( cos(Φ[t]−Φ[j]) + i sin(Φ[t]−Φ[j]) )
```

This is **literally the S5 / LRU diagonal associative scan** (a diagonal linear recurrence whose
cumulative operator is an elementwise product, hence an associative prefix-scan). The complex case
adds exactly ONE more prefix-sum (`Φ`) and replaces the real decay factor `exp(G[t]−G[j])` with a
complex "twiddle" `exp(G[t]−G[j])·cis(Φ[t]−Φ[j])`. Associativity holds because complex
multiplication is associative — the LRU's defining property.

### 3.2 The intra-chunk delta system is the existing UT structure, over ℂ

The existing chunked kernel (`ndm/triton/e97_chunked.py`, `E97_CHUNKED_KERNEL_NOTE.md`) solves a
**unit lower-triangular** system per chunk for the delta corrections,

```
(I + M) Δ = P − diag(decay_prev)·U·S0c ,   M[t,j] = exp(G[t]−g[t]−G[j])·(k_j · u_t)   (strictly lower)
```

inverting `(I+M)` with **Newton–Schulz** `X ← X(2I − MX)` in `ceil(log2 C)` matmul steps (exact
because `M` is nilpotent, `M^C=0`). **The complex extension changes only the entries, not the
structure:**

```
M_ℂ[t,j] = exp(G[t]−g[t]−G[j]) · cis(Φ[t]−φ[t]−Φ[j]) · ( k_j^H · u_t )      (strictly lower, ℂ)
```

`M_ℂ` is still strictly-lower-triangular ⇒ still nilpotent ⇒ Newton–Schulz still inverts **exactly**
in `ceil(log2 C)` steps, now with **complex `tl.dot`**. No new algorithm; the scalar twiddle
`cis(Φ[t]−φ[t]−Φ[j])` multiplies the same decay factor that already lives in `M`, and the key inner
product becomes Hermitian `k_j^H u_t`.

### 3.3 Throughput: GDN-2-class

- **FLOPs:** complex mul = 4 real mul + 2 add (or Karatsuba: 3 mul + 5 add). Complex matmul =
  4 real matmuls of the same shape (or 3 with Karatsuba). So the complex kernel is **≈2× the FLOPs**
  of the real kernel, **same asymptotics** `O(T·C·N·V / C)` per chunk, **same launch structure**
  (`T/C` sequential cross-chunk steps, all intra-chunk work is matmuls).
- **Memory / state:** 2× (Re,Im) — `N/2` complex channels = `N` reals, so the *state size is
  unchanged* (`N/2 × V × 2 = N × V`); only the q/k twiddles add the `Φ` prefix tile.
- **Verdict:** same tensor-core-bound regime as GDN-2 / the e97 chunked kernel (97–100% util at the
  scale dims, `E97_CHUNKED_KERNEL_NOTE.md`). 2× FLOPs at constant util ⇒ expect ~0.5–0.9× GDN-2
  wall-clock for the complex bulk — **the same band the chunked e97 kernel already occupies**, and
  the hetero-overlap lever (`hetero-kernel-hits-095x`) recovers ≥0.95× when blended.

### 3.4 Kernel-extension plan (concrete, for the implementer)

Extend the **diagonal-scan** half of `ndm/triton/gdn2_nonlin_fused.py` (the boundary-φ scan) and
the **chunked UT** kernel `ndm/triton/e97_chunked_autograd.py`. Two representation options; pick
**(A) split Re/Im tiles** (simplest, recommended) — **(B) 2×2 real blocks** is mathematically
identical and noted for the tensor-core-packing micro-opt.

**State / data layout (A).** Carry `S` as two fp32 tiles `S_re, S_im : [N/2, V]` (or `[C, N/2]`,
`[C, V]` chunk tiles). All existing real tiles (q,k,v,β,decay) gain an Im partner for q,k (v stays
real, Im=0).

**Forward kernel changes (per chunk, per (b,h)):**
1. **Two prefix sums.** Keep `G[t] = cumsum(log r_t)` (already present). Add
   `Φ[t] = cumsum(θ_t)` where `θ_t = θ_base + Δ·tanh(theta_proj(x)_t)` is precomputed outside the
   kernel (cheap pointwise) and passed in like `g`. Compute `cosΦ, sinΦ` once per tile.
2. **Twiddle the UT matrix.** Where the real kernel forms `M[t,j] = decay_factor · (k_j·u_t)`,
   form the complex pair:
   `M_re = df·(cosΔΦ·KU_re − sinΔΦ·KU_im)`, `M_im = df·(sinΔΦ·KU_re + cosΔΦ·KU_im)`,
   with `df = exp(G[t]−g[t]−G[j])`, `ΔΦ = Φ[t]−φ[t]−Φ[j]`, and `KU = k_j^H u_t` (Hermitian:
   `KU_re = k_re·u_re + k_im·u_im`, `KU_im = k_re·u_im − k_im·u_re`). Each `tl.dot` becomes the
   standard 4-real-matmul complex product (3 with Karatsuba if register-bound).
3. **Newton–Schulz over ℂ.** Run the existing `X ← X(2I − MX)` loop with complex `tl.dot` (4 real
   matmuls per product). `M_ℂ` strictly-lower ⇒ exact in `ceil(log2 C)` steps, unchanged count.
4. **Cross-chunk carry.** Multiply the carried entry-state by the chunk-cumulative complex diagonal
   `exp(G[C-1]−G[-1])·cis(Φ[C-1]−Φ[-1])` (elementwise per channel) — the S5/LRU diagonal carry.
5. **Read.** `o_t = Re(q_t^H S_t) = q_re·S_re + q_im·S_im` summed over the key axis (the Im output
   is discarded under the `Re` read; not computed unless `read_mode='reim'`).

**Backward kernel.** The VJP over ℂ is the **conjugate** of the forward operator (Wirtinger
calculus): walk chunks in reverse threading a complex `dS = (dS_re, dS_im)`, reuse the
recompute-forward-intermediates pattern of `e97_chunked_autograd.py`. Two extra gradient sinks:
`∂/∂θ` accumulates from the twiddle (`dΦ` flows back through `cis` as a cross term
`dΦ = Σ Re(i·conj(λ-contrib)·dS)`), feeding `theta_proj` and `θ_base`; `∂/∂log r` flows through `G`
exactly as the existing real decay gradient. Tile budget: complex doubles the live `[C,C]` fp32
tiles vs the real kernel; at `C=32` the real kernel uses 4 KB/tile with wide margin
(`E97_CHUNKED_KERNEL_NOTE.md` "Why C=32"), so `C=32` stays comfortably under the ~100 KB/SM Ada
ceiling at 2× tiles. **Keep `C=32` default** (Hopper could do `C=64`).

**Reuse for the identity/linear path.** When `θ≡0` and `Δ=0`, the Im tiles are statically zero;
gate on a `complex=False` constexpr so the kernel compiles to the **exact real GDN path** (so the
real head is recovered with zero overhead — the `identity` fast-path discipline already used in
`gdn2_nonlin_fused.nonlinear_gated_delta_scan`).

**Parity tests (mirror `tests/test_e97_chunked.py`, REAL data only).** (a) `θ≡0,Δ=0` ⇒ bitwise-near
the FLA `chunk_gated_delta_rule` (rel-err < 1e-5 fp32). (b) Fixed `θ_base`, `Δ=0` ⇒ matches a pure
PyTorch complex-diagonal LRU reference scan. (c) Full complex + delta ⇒ matches an eager complex
`for t in range(T)` reference (the `nonlinear_gated_delta_torch_reference` analogue, complexified)
to fp32 rel-err < 3e-3, bf16/TF32 < 5%. (d) gradcheck on `theta_proj`, `θ_base`, `A_log` via the
complex reference.

---

## 4. Unification — eigenvalue placement in the unit disk is ONE knob

Real-positive, real-negative, and complex transitions are **not three mechanisms — they are three
regions of the same complex eigenvalue λ = r·e^{iθ}** placed in the open unit disk (`r<1`):

| Regime | θ | λ | Operator action | Existing realization |
|---|---|---|---|---|
| **real positive** | `θ = 0` | `λ = r ∈ (0,1)` | pure decay / leak | GDN default (`allow_neg_eigval=False`) |
| **real negative** | `θ = π` | `λ = −r ∈ (−1,0)` | reflection | `allow_neg_eigval=True` along-key eig (S5/tracking lever, `EIGENVALUE_CAUSAL_TEST.md`) |
| **complex** | `θ ∈ (0,π)` | `λ = r·e^{iθ}` | rotation–scaling | **this head** |

**Key structural claim:** the negative eigenvalue (the validated S5 / DeltaProduct tracking lever)
is *exactly the `θ=π` endpoint* of the complex phase, and the GDN positive decay is the `θ=0`
endpoint. The complex head with the §1.3 init (`f_dc` channels at `θ=0`, a few at `θ=π`, the rest
log-spaced between) therefore **subsumes both prior heads as boundary cases of one continuous
knob** — phase. There is no separate `allow_neg_eigval` switch in the complex head: negative is
just `θ→π`, reachable continuously, and the input modulation `Δ·tanh(W_θx)` lets the data move a
channel between decay (θ≈0), rotation (θ mid), and reflection (θ≈π) per step. This is the
cleanest possible statement of the E88/E97/E98 eigenvalue-placement program: **one disk, one knob.**

---

## 5. "Everywhere × nonlinear-subset" arrangement

### 5.1 The cell

```
ALL heads:        carry the complex diagonal transition Λ_t  (complex_eig default ON)
SUBSET of heads:  ADDITIONALLY apply a per-step bounded map  S_t ← hardtanh(S_t)
                  (the phi_shell per-step φ; the modquad depth lever)
```

Two physically distinct kernels, because **complex is chunkable but per-step `hardtanh` is not**
(memory: `fuse-2kernel-tanh-perp-chunkable` — *bounded per-step state ⊥ chunkable* is
fundamental; `phi_shell` docstring: "All per-step phi are non-chunkable"):

- **complex-only heads (the majority):** run the **chunked complex diagonal scan** of §3 —
  tensor-core bound, GDN-2-class throughput.
- **complex + hardtanh heads (the subset):** run the **sequential per-step** complex scan with
  `S_t ← hardtanh(S_t)` every step — latency bound, ~const time in `T` (like the `gdn2_nonlin_shell`
  sequential path).

This is **structurally the hetero-kernel split** (`hetero-kernel-hits-095x`): a small latency-bound
sequential fraction runs on a **side CUDA stream** concurrently with the tensor-core chunked bulk,
so the sequential subset's wall cost hides under the bulk (and its backward overlaps the bulk's
backward). The `typed_head_mixture` `overlap_streams` machinery (lines 404–457) already implements
exactly this for the `gdn2_nonlin_shell` head — the complex+hardtanh subset slots into the same
side-stream slot. Net: "everywhere complex" stays fast; "subset hardtanh" pays the sequential tax
**only on a small head fraction, hidden by overlap** — the same ≥0.95× regime.

> Why `hardtanh` specifically: `phi-explore-bounded-saturation-split-edit` found per-step **bounded**
> saturation (`tanh = hardtanh = softsign`, all perfect 1.000) is the depth lever on the modquad
> cliff; the mechanism is *boundedness, not the particular function*. `hardtanh` is the cheapest
> bounded map (a clamp, exact gradient `1` on `[−1,1]` else `0`) — ideal for the sequential kernel.

### 5.2 Config knobs

Added to `TypedHeadMixtureLayer.__init__` (and a standalone `ComplexEigHeadLayer` mirroring
`GDN2NonlinShellLayer`):

```python
complex_eig: bool = True            # complex diagonal transition on ALL heads (default ON)
n_heads: int = 48                   # existing
cplx_theta_base: int = 10000        # RoPE-style log-spaced base B (§1.3)
cplx_dc_frac: float = 0.5           # fraction of channels init at θ=0 (real-positive subspace)
cplx_theta_drift: float = math.pi/8 # Δ, the per-step phase-modulation cap (§1.2)
cplx_read_mode: str = 'real'        # 'real' (Re, default) | 'reim' (ablation)
nonlin_subset_frac: float = 0.0     # fraction of heads ALSO getting per-step φ (0 ⇒ pure complex)
nonlin_subset_phi: str = 'hardtanh' # the per-step bounded map on the subset (phi_shell PHI set)
overlap_streams: bool = True        # run the non-chunkable subset on a side stream (reuse)
```

Head-count split for the subset uses the existing **largest-remainder** allocator
(`largest_remainder_counts`, `typed_head_mixture.py:101`): `n_nonlin = round(nonlin_subset_frac ·
n_heads)` heads get the sequential complex+φ kernel; the rest get the chunked complex kernel. The
allocation is reported honestly in `head_alloc()` (a type may get 0).

### 5.3 Substrate integration (two routes, no new file conflicts)

1. **As a global transition swap (recommended first).** Add `complex_eig` to the FLA-projection
   heads (`gdn2_recall`, `gdn2_nonlin_shell`): when set, the diagonal `g_t` decay is replaced by the
   complex `Λ_t` (§1–2) and the chunked scan calls the complex kernel (§3.4). This makes "complex on
   ALL heads" a one-flag change to the *existing* mixture; the per-step-φ subset is the *existing*
   `gdn2_nonlin_shell`/`phi_shell` path with `state_chunk=1` and `state_nonlin='hardtanh'`, now over
   the complex state. File scope: `gdn2_nonlin_shell.py`, the two triton kernels, `typed_head_mixture.py`.
2. **As a first-class head type `cplx_gdn`.** Append `'cplx_gdn'` to `TYPE_NAMES`
   (`typed_head_mixture.py:90`) with the legacy right-pad-`-inf` contract (lines 136–145) so older
   logit vectors still allocate it 0 heads. Then CMA can search its fraction against the other 8
   types. This is the clean A/B but needs a 9th logit slot.

`phi_shell` integration: `phi_shell` already runs `S_t = φ(diag(g_t)S_{t-1}+…)` at `state_chunk=1`
per step. The complex subset is `phi_shell` with `g_t` → `Λ_t` (complex diagonal) and
`φ=hardtanh` applied to `(Re,Im)` independently — a localized change to `phi_shell`'s recurrence,
no new substrate.

---

## 6. Capability hypotheses (what the eigenvalue placement should unlock)

Complex eigenvalues are a *known* primitive (S4/S5/LRU); the hypotheses below are the **capabilities
the placement grafts onto gated-delta**, to be checked on the existing probe harness (S5
permutation, MQAR recall, modular_quadratic, + new periodic/positional probes). REAL train+eval
only, no mocks.

1. **Phase / relative position.** A complex diagonal recurrence is a *content-plus-position* memory:
   the cumulative phase `Φ[t]−Φ[j]` encodes relative position `(t−j)` natively (this is exactly why
   RoPE/LRU work). **Hypothesis:** the complex head solves position-sensitive probes
   (copy-with-offset, relative-position regression) that the real-decay GDN smears, because real
   decay is position-*magnitude* only (no phase to carry `t−j`).
2. **Periodicity.** Channels initialized at `θ_j^base` are tuned band-pass filters at angular
   frequency `θ_j`. **Hypothesis:** the head detects periodic structure (period-`p` token patterns)
   far more sample-efficiently than the real head, which has no oscillatory mode.
3. **Mod-k counting (the sharp one).** Setting `θ = 2π/k` makes a channel a **native k-step clock**:
   after exactly `k` steps the phase returns to 0, so the channel implements a **mod-k counter /
   period-k resonator** in one linear recurrence — no depth, no nonlinearity needed. **Hypothesis:**
   the complex head solves mod-k / modular-quadratic-period tasks (the `modular_quadratic` cliff,
   `e97-nonlin-in-time-separates-modquad`) by *placing a clock at θ=2π/k*, where the real GDN needs
   iterated depth and the per-step-tanh head needs the bounded-saturation depth lever. Prediction:
   on mod-k tasks **complex-only** approaches the per-step-`hardtanh` head's accuracy without the
   sequential-kernel cost.
4. **Complement, not replacement.** Complex placement gives *periodicity/position*; per-step
   `hardtanh` gives *bounded-saturation depth* (the modquad-cliff lever,
   `phi-explore-bounded-saturation-split-edit`). They are orthogonal capabilities — hence the
   "everywhere complex × hardtanh-subset" arrangement: the bulk gets cheap periodicity, the subset
   adds depth where rotation alone is insufficient (deep modular composition, non-period-aligned
   counting).

---

## 7. The explicit prediction: TIE on OOS bpb (convergent-loss null)

Consistent with the entire E88→E99 line — **every exotic head TIES the GDN-2 baseline on LM
bits-per-byte at matched params and tokens** (`e97-nonlin-in-time-separates-modquad`: "LM-BPB tie ≠
capability equivalence"; `e97delta-1p3b-tie-split`; `e97-wallclock-cma-shell-flat`; `e99-1p3b-controls`).
The capability shows on **probes**, never on convergent LM loss.

> **Prediction (commit it):** at matched parameter budget and matched tokens, the complex-eigenvalue
> gated-delta head **TIES** `gdn2-mlp` / `gdn-neg` on out-of-sample LM bits-per-byte (within seed
> noise, |ΔBPB| ≲ 0.02). The **separation** appears only on the §6 capability probes —
> phase/position, periodicity, and mod-k counting (`θ=2π/k` native clock) — where the complex head
> is predicted to beat the real-decay GDN, and on mod-k to approach the per-step-`hardtanh` head
> **without its sequential-kernel wall-clock tax**.

This is the **convergent-loss null**: LM-BPB is the wrong instrument (the MLP + attention layers
absorb the capability difference at convergence); the right instruments are the structured probes.
A complex-eig head that *changed* LM-BPB would be a surprise to be double-checked, not the expected
result. The accept/reject decision is therefore made on the **probe panel**, not BPB — unlike the prior heads,
which lost on wall-clock at a BPB tie. The complex head's distinctive bet: it is
**chunkable** (§3, unlike the per-step-tanh heads), so a probe win does **not** cost throughput —
removing the wall-clock objection that closed `e97-wallclock-cma-shell-flat` and
`fuse-2kernel-tanh-perp-chunkable`.

---

## 8. Validation checklist (this spec)

- [x] **Exact head update equations** — complex transition `Λ_t=diag(r e^{iθ})` (§1.2), gated-delta
      write with `k^H` (§2), read `o=Re(q^H S)` with Re-vs-[Re;Im] justified (§2.1).
- [x] **Projection list + θ init** — full table §1.4; log-spaced RoPE/S4 grid with `f_dc` real-DC
      reservation §1.3.
- [x] **Chunkability shown explicitly** — cumulative `∏diag(λ)=diag(∏r · e^{iΣθ})` = two prefix
      sums = S5/LRU diagonal associative scan (§3.1); intra-chunk = same nilpotent UT Newton–Schulz
      over ℂ (§3.2); GDN-2-class throughput (§3.3).
- [x] **Kernel-extension plan** — concrete forward/backward changes to `gdn2_nonlin_fused.py` +
      `e97_chunked_autograd.py`, layout, twiddle, tile budget, parity tests (§3.4).
- [x] **Unification** — real-pos (θ=0), real-neg (θ=π = `allow_neg_eigval`), complex (θ∈(0,π)) = one
      disk, one knob (§4).
- [x] **"Everywhere × nonlinear-subset" config + substrate integration** — cell, knobs, two
      integration routes, stream-overlap reuse (§5).
- [x] **Capability hypotheses + convergent-loss-null prediction** — §6 (phase/position, periodicity,
      mod-k native clock) and §7 (TIE on OOS bpb, separate on probes; chunkable ⇒ no wall-clock tax).

## 9. Open implementation questions handed to the implementer

1. **Per-channel vs per-head magnitude `r`.** §1.2 specifies per-channel (`A_log,dt_bias` widened to
   `N/2`). A cheaper variant keeps `r` per-head scalar and only `θ` per-channel — fewer params,
   loses per-channel decay. Default: per-channel; ablate per-head scalar.
2. **Karatsuba vs 4-mul complex `tl.dot`.** 4-mul is simplest; Karatsuba (3 mul) saves FLOPs if the
   kernel is compute-bound at scale. Start 4-mul, profile, switch if `<0.85×` GDN-2.
3. **`C=64` on Hopper.** The 2× tile budget fits `C=64` on Hopper's 228 KB/SM; gate `C` on
   device capability (keep `C=32` on Ada/Ampere).
4. **θ_base trainability.** Default learnable + WD-exempt; an ablation freezes it (pure fixed LRU
   grid) to test whether learned frequencies help beyond the init grid.
