# Nonlinear-memory (MLP fast-weight) head — SPEC + fused kernel plan

**Status:** spec only. No implementation in this task. This document pins down a
candidate Emender head-type whose recurrent **state is a small nonlinear memory** — a
1-hidden-layer MLP whose *parameters* are the carried state — in contrast to GDN's
**linear matrix associative memory**. It defines the exact recurrence, the write/read
rules, an honest chunkability classification, a **fused sequential-scan Triton kernel
plan** (modelled on the existing `e88_triton_forward` sequential path and the
`e97_chunked_autograd` reverse-replay backward), capability hypotheses, and a
convergent-loss-null prediction.

Proposed taxonomy name: **`mlp-mem`** (see §7). This is the *state-type* name for the
same physical cell that `ttt-spec` reaches from the *write-rule* angle (its "inner-MLP"
row); the boundary between the two specs is fixed in §7.3.

---

## 0. Where this slots in (substrate, verified against code)

The Emender layer is a within-layer pool of head-types over the dynamics grid
(`paper/review/EMENDER_TAXONOMY.md`). Every existing cell carries a **matrix** state
`S ∈ ℝ^{N×V}` and differs only in the *eigenvalue placement* of its per-token transition
(`decay` / `reflect` / `rot`) and whether the matrix update is passed through a
**saturating** map (`-nonlin`). The reference sequential kernel for a saturated matrix
state is `ndm/triton/e88_triton_forward.py`:

```
# e88 per (b,h), matrix state S [N,V], sequential time loop:
r_t     = S_{t-1}^T @ k_t                       # retrieve  [V]
delta_t = v_t - r_t                             # [V]
S_t     = tanh(decay_t * S_{t-1} + outer(delta_t, k_t))   # [N,V]
out_t   = S_t^T @ q_t                           # [V]
```

`mlp-mem` keeps the *substrate* (per-head, per-token, gated read/write/decay; same
projection list as GDN-2: `k,q,v` plus a write-strength gate and a forget gate) but
**replaces the state type**: the carried state is no longer a matrix `S` but the
parameter set of a small MLP. This is a new axis orthogonal to the 2-axis grid (§7.2).

---

## 1. The memory and its read

### 1.1 State = parameters of a 1-hidden-layer MLP

Carry, per (batch, head), the parameters of a small MLP `M_θ : ℝ^{d_k} → ℝ^{d_v}`:

```
θ_t = ( W1_t ∈ ℝ^{H×d_k},  W2_t ∈ ℝ^{d_v×H} )            (biases optional, §1.4)
M_θ(x) = W2 · σ( W1 · x )                                 σ = elementwise nonlinearity
```

- `d_k = d_v = N` is the per-head key/value width (kept ≤ 64, as in e88/e97).
- `H` is the inner hidden width — the **memory's capacity knob**, small (e.g. 16–64).
- `σ` is a bounded saturating map; **default `σ = tanh`** (so the hidden activations are
  bounded, connecting to the boundedness result that bounded state is what unlocks the
  depth capability on the modular-quadratic cliff). GELU is an alternative knob.

The *state* is `θ` = the `W1, W2` matrices. Their total size is `H·d_k + d_v·H = 2HN`
floats per head — e.g. `H=N=32 → 2048` floats, comparable to a matrix state `N·V = 1024`.

### 1.2 Read — `out_t = M_{θ_t}(q_t)`

The read is a **nonlinear** function of the query:

```
out_t = W2_t · σ( W1_t · q_t )                            # [d_v]
```

Contrast with every matrix-state head, whose read `out_t = S_t^T q_t` is **linear** in
`q_t` (bilinear in `(S_t, q_t)`). The retrieved value here can be any nonlinear function
of the query the inner MLP can express. (Read happens at the **post-write** weights
`θ_t`, mirroring delta-net's `out_t = S_t^T q_t` which reads after the update; this is a
single design knob, §6.2.)

---

## 2. The write — one inner gradient step toward `k_t ↦ v_t`

### 2.1 Inner loss and update rule (real math)

Per token, define the inner reconstruction loss of the memory at the *previous* state:

```
ℓ_t(θ) = ½ ‖ M_θ(k_t) − v_t ‖²                            # fit the new association k_t→v_t
```

The write is a single gradient-descent step on `ℓ_t`, with an **input-dependent inner
learning rate** `η_t` (the write-strength / β gate) and a **forget gate** `γ_t ∈ (0,1)`
(multiplicative weight-decay of the fast weights):

```
θ_t = γ_t · θ_{t-1} − η_t · ∇_θ ℓ_t( θ_{t-1} )           (★ the mlp-mem recurrence)
```

`η_t = softplus(W_η x_t)` or `sigmoid`-gated; `γ_t = sigmoid(W_γ x_t)`. Both are
per-token, per-head scalars (or per-row vectors, a knob).

### 2.2 The gradient in closed form (what the kernel computes)

Let, at `θ_{t-1} = (W1, W2)`:

```
pre = W1 · k_t        ∈ ℝ^H
h   = σ(pre)          ∈ ℝ^H
ŷ   = W2 · h          ∈ ℝ^{d_v}                           # memory's current guess for v_t
e   = ŷ − v_t         ∈ ℝ^{d_v}                           # inner error (the "delta", signed)
δ   = (W2^T · e) ⊙ σ'(pre)   ∈ ℝ^H                        # backprop of e through the hidden layer
```

Then the two parameter gradients are each **rank-1**, but **coupled through `σ` and the
current `W2`**:

```
∇_{W2} ℓ_t = e · hᵀ            ∈ ℝ^{d_v×H}                (outer product)
∇_{W1} ℓ_t = δ · k_tᵀ          ∈ ℝ^{H×d_k}               (outer product)
```

So the concrete update (★) is the pair of gated rank-1 corrections:

```
W2_t = γ_t · W2_{t-1} − η_t · e · hᵀ
W1_t = γ_t · W1_{t-1} − η_t · δ · k_tᵀ
```

Two rank-1 writes per token — but unlike GDN they are **not independent linear writes**:
`δ` depends on `W2_{t-1}` and on `σ'(W1_{t-1} k_t)`, so the `W1` write is a *nonlinear,
state-dependent* function of the current memory. That coupling is the whole point, and it
is exactly what destroys chunkability (§4).

### 2.3 Delta-rule (GDN) is the degenerate corner — proof

Remove the hidden layer (`H → d_k`, `σ = identity`, fold `W1,W2` into a single matrix
`W ∈ ℝ^{d_v×d_k}` so `M_θ(x) = W x`). Then:

```
ℓ_t(W) = ½‖W k_t − v_t‖² ,   ∇_W ℓ_t = (W k_t − v_t) k_tᵀ = −Δ_t k_tᵀ ,  Δ_t = v_t − W k_t
W_t = γ_t W_{t-1} − η_t (W_{t-1} k_t − v_t) k_tᵀ
    = γ_t W_{t-1} + η_t (v_t − W_{t-1} k_t) k_tᵀ
    = γ_t W_{t-1} + η_t Δ_t k_tᵀ
```

This is **exactly the gated delta-rule** (`reflect`/`decay` GDN-2) with `β_t = η_t`,
`S = Wᵀ`. So:

> **Proposition.** GDN-2's matrix memory is `mlp-mem` with the hidden layer removed and
> `σ` linear: it is the **no-hidden-layer, linear** corner of the `mlp-mem` axis. The
> delta-rule is one inner gradient step of a *linear* reconstruction loss; `mlp-mem`
> takes one inner gradient step of a *nonlinear* (MLP) reconstruction loss.

This places `mlp-mem` in the taxonomy as a genuine generalization, not a side-grade
(§7), and aligns the write rule with `ttt-spec`'s "delta-rule = 1 inner step" frame
(§7.3).

### 2.4 Contrast vs GDN linear memory (explicit)

| | GDN linear memory (`decay`/`reflect`) | `mlp-mem` nonlinear memory |
|---|---|---|
| State | matrix `S ∈ ℝ^{N×V}` | MLP params `θ = (W1,W2)`, `2HN` floats |
| Stored map `k→v` | **linear**: `M(k)=S k` | **nonlinear**: `M(k)=W2 σ(W1 k)` |
| Read in `q` | **linear** `S^T q` | **nonlinear** `W2 σ(W1 q)` |
| Write | rank-1, **linear in `S`**: `S_t=γS_{t-1}+βΔk^T` | two rank-1, **nonlinear in `θ`** (δ couples through `W2`,`σ'`) |
| Transition `θ_{t-1}↦θ_t` | **affine** `S ↦ (γI−βkk^T)S + βvk^T` | **nonlinear**, state-dependent |
| Associative? | **yes** → chunkable scan (WY/UT) | **no** → sequential scan (§4) |

---

## 3. Full per-token recurrence (one place)

For each (batch, head), state `θ = (W1 [H,N], W2 [V,H])`, time loop:

```
# inputs at step t: k_t,q_t ∈ ℝ^N , v_t ∈ ℝ^V , gates η_t, γ_t (scalars)
pre = W1 @ k_t ;  h = σ(pre) ;  ŷ = W2 @ h ;  e = ŷ − v_t          # forward at the key
δ   = (W2^T @ e) ⊙ σ'(pre)                                          # inner backprop
W2  = γ_t * W2 − η_t * outer(e, h)                                  # rank-1 write
W1  = γ_t * W1 − η_t * outer(δ, k_t)                                # rank-1 write
out_t = W2 @ σ(W1 @ q_t)                                            # read at the query (post-write)
```

`σ' (pre)` for `σ = tanh` is `1 − h²` (reuse `h`); no extra nonlinearity evaluation. All
operands are `≤ 64`-wide; every `@` is a tiny matrix–vector / outer product.

---

## 4. Chunkability — honest classification: **NON-associative → sequential**

### 4.1 Why it cannot be chunked (the real reason)

A chunked-parallel scan (the WY/UT structure used by `e97_chunked_autograd` and FLA's
gated-delta) exists **iff** the per-token state transition is **affine in the state**, so
the cumulative transition over a chunk collapses to a *product of per-token transition
matrices* `A_t` times the state plus an accumulated drive:

```
GDN:  S_t = A_t S_{t-1} + b_t ,  A_t = γ_t I − β_t k_t k_tᵀ   (affine ⇒ associative ⇒ chunkable)
```

The `mlp-mem` transition is **not affine in `θ`**. Writing it out, `W1_t` depends on `θ_{t-1}`
through

```
δ = (W2_{t-1}ᵀ (W2_{t-1} σ(W1_{t-1} k_t) − v_t)) ⊙ σ'(W1_{t-1} k_t)
```

which is **cubic** in the carried weights (a `W2·W2·σ(W1·)` composition) and passes
through the nonlinearity `σ` and its derivative `σ'`. There is no constant
per-token transition matrix `A_t`; the map `θ_{t-1} ↦ θ_t` is genuinely nonlinear and
state-dependent. Therefore the chunk composition does **not** factor into a fixed-rank
closed form, **no parallel/associative scan exists**, and the forward must run as a
**sequential per-token recurrence**. This is the same wall the saturated matrix head hit
(`tanh`-state never engaged the chunked path, `fuse-2kernel-nogo-tanh-perp-chunkable`):
**bounded/nonlinear state ⊥ chunkable** is fundamental, and it applies a fortiori here —
the nonlinearity is now in the memory *itself*, not just a post-update squash.

### 4.2 The one honest caveat — mini-batch / "TTT-chunk" relaxation (a *different* recurrence)

Sun et al. (2024) make TTT-MLP throughput-viable with **mini-batch TTT**: within a chunk
of size `C`, compute all `C` inner gradients **at the chunk-entry weights `θ_c`** (not
sequentially updated within the chunk) and apply one combined step at the chunk boundary.
This *is* chunk-parallelizable. But it is an **approximation with a real semantic cost**:
the per-token online write is replaced by a *stale-gradient, mini-batch* write — tokens
inside a chunk no longer see each other's updates. It is therefore a **coarser, different
recurrence**, not the exact (★) this spec defines. We state this honestly and **keep the
exact per-token semantics as primary** (the task asks for a per-token "correct the MLP
toward `k_t→v_t`" write). `C` becomes an optional knob trading token-granularity for
parallelism; `C=1` is the exact sequential cell. The chunkability boundary is thus:

| Variant | Recurrence | Chunkable? |
|---|---|---|
| GDN delta (linear inner step) | affine in `S` | **yes** (WY/UT scan) |
| `mlp-mem`, exact per-token (`C=1`) | nonlinear in `θ` | **no** → sequential fused scan |
| `mlp-mem`, mini-batch TTT (`C>1`) | stale within-chunk grads | yes, but **different (approx) recurrence** |

This division of labor matches `ttt-spec`: the *linear-inner-step* and *K-step/momentum*
chunkability questions belong to `ttt-spec`; the **inner-MLP ⇒ non-chunkable** verdict
belongs here.

---

## 5. FUSED Triton kernel plan (sequential scan; real/imag-free; NOT pure-torch)

Pure-torch is **not acceptable** for experiments. The plan mirrors the **sequential**
structure of `e88_triton_forward._e88_forward_kernel` (one program per `(b, head_block)`,
state resident in registers/SRAM, sequential time loop, sparse forward checkpoints) and
the **reverse-replay** backward of `e97_chunked_autograd` (a second fused kernel walking
the sequence in reverse, recomputing per-step intermediates from sparse checkpoints,
threading the state-gradient across steps in registers). State is real-valued — no
complex pairing, so **no real/imag split** (unlike `complex_eig_chunked`).

### 5.1 Forward kernel `_mlp_mem_fwd_kernel`

- **Grid:** `(B, ceil(H_heads / BLOCK_H))` — one program per (batch, head-block), exactly
  e88's launch shape. (`H_heads` = number of attention heads; do not confuse with inner
  width `H`.)
- **Resident state in SRAM/registers** per head: `W1 [H, N]`, `W2 [V, H]`. With
  `H,N,V ≤ 64`, both tiles are ≤ `64×64` fp32 → fit alongside e88's `[N,V]` budget. Inner
  width `H` is a `tl.constexpr`.
- **Per-step body** (all `tl.dot` on tensor cores, intermediates never touch HBM):
  1. load `k_t,q_t [N]`, `v_t [V]`, gates `η_t,γ_t` (scalars; or `[H]`/`[V]` row-gates).
  2. `pre = W1 @ k_t` `[H]`; `h = tanh(pre)`; `ŷ = W2 @ h` `[V]`; `e = ŷ − v_t`.
  3. `δ = (W2ᵀ @ e) * (1 − h*h)` `[H]`.
  4. `W2 = γ_t*W2 − η_t * e[:,None]*h[None,:]`; `W1 = γ_t*W1 − η_t * δ[:,None]*k_t[None,:]`.
  5. `out_t = W2 @ tanh(W1 @ q_t)` `[V]`; store `out_t`.
- **Sparse forward checkpointing** (identical scheme to e88, `CKPT_INTERVAL=K`, default
  16): save `θ = (W1,W2)` every `K` steps to `θ_ckpt[seg]` (the state *before* step
  `seg*K`), shrinking the saved-state buffer from `O(T)` to `O(T/K)`. Slot size per head
  `= 2HN` floats. Also write `θ_final`.
- **Autotune** `BLOCK_H ∈ {1,2,4,8}` (e88's high-head-count fix). bf16 inputs run the
  TF32 tensor-core path (`allow_tf32`); fp32 runs exact (mirror e88/e97).

### 5.2 Backward kernel `_mlp_mem_bwd_kernel` — fused reverse BPTT

This is the substantive part: backprop **through the inner gradient step** (the write
itself contains a gradient, so the backward needs the *second-order* terms of the inner
MLP — for a 1-hidden-layer MLP these are **closed-form**, no iterative inverse needed,
unlike the Newton–Schulz UT inverse in `e97_chunked`).

- **Grid:** same `(B, ceil(H_heads/BLOCK_H))`; walk segments in **reverse**.
- **Carry across steps** in registers: the adjoints `dW1, dW2` of the state (the
  BPTT state-gradient `dθ`), plus output grads accumulated into `dk,dq,dv,dη,dγ`.
- **Per-segment:** reload `θ_ckpt[seg]`, **recompute** the forward for the `K` steps to
  recover each step's `(pre, h, ŷ, e, δ, out)` (recompute, don't store — same trade as
  e97), then walk the `K` steps **in reverse** applying the per-step adjoint.
- **Per-step adjoint** (the new math; `σ=tanh`, `σ'=1−h²`, `σ''=−2h(1−h²)`). Forward step
  was `θ' = (W1',W2')` from `θ=(W1,W2)`; incoming adjoints `(dW1', dW2')` plus the read
  adjoint `do_t` (read uses `θ'` and `q_t`). Produce `(dW1,dW2)` and input grads:

  ```
  # (a) read path, at post-write θ' = (W1',W2'):
  preq = W1' @ q_t ; hq = tanh(preq)
  dW2'  += do_t[:,None] * hq[None,:]
  dhq    = W2'ᵀ @ do_t ; dpreq = dhq * (1 − hq²)
  dW1'  += dpreq[:,None] * q_t[None,:] ;  dq_t += W1'ᵀ @ dpreq

  # (b) gated rank-1 writes  W1'=γW1−η δ k^T ,  W2'=γW2−η e h^T :
  dγ += <dW1', W1> + <dW2', W2>                 # Frobenius inner products
  dW1 += γ * dW1' ;  dW2 += γ * dW2'            # decay path into previous state
  dη += −(<dW1', δ k^T> + <dW2', e h^T>)
  d_δ  = −η * (dW1' @ k_t) ;  dk_t += −η * (dW1'ᵀ @ δ)
  d_e1 = −η * (dW2' @ h)   ;  d_h1 = −η * (dW2'ᵀ @ e)

  # (c) differentiate δ = (W2ᵀ e) ⊙ σ'(pre) and the forward at the key (second order):
  #     pre=W1 k, h=σ(pre), ŷ=W2 h, e=ŷ−v.  Let s'=1−h², s''=−2 h s'.
  d_e   = d_e1 + W2 @ (d_δ ⊙ s')                                   # e feeds both writes and δ
  d_sp  = (W2ᵀ e) ⊙ (s'' * (W2ᵀ d_δ?))   …                         # σ'' term from δ (see note)
  dv_t += −d_e
  d_h   = d_h1 + W2ᵀ d_e                                           # h feeds W2-write and ŷ
  dpre  = d_h * s' + d_sp                                          # combine paths through pre
  dW2  += d_e[:,None]*h[None,:] + (d_δ⊙s')[:,None] ... (e-path)    # W2 appears in ŷ and in δ
  dW1  += dpre[:,None]*k_t[None,:] ;  dk_t += W1ᵀ @ dpre
  ```

  The `σ''` cross-terms (marked `…`) are the only genuinely new algebra vs a first-order
  delta backward; they are finite, closed-form, and bounded (tanh derivatives ≤ 1). The
  implementer expands the three `W2`-appearances (in `ŷ`, in `δ` via `W2ᵀe`, and in the
  `W2`-write) and the two `pre`-paths exactly; this spec fixes the structure and the
  non-obvious second-order couplings, matching the level of detail in
  `COMPLEX_EIG_HEAD_SPEC.md §3.2`. A finite-difference gradient check (per-step, then full
  sequence) against an fp64 torch reference is the parity gate, as in
  `tests/test_e97_chunked.py`.

### 5.3 Numerics

- `σ = tanh` keeps hidden activations and all backward derivatives **bounded** — no
  `exp(−g)` log-decay overflow path exists here (decay is multiplicative `γ_t∈(0,1)`, not
  `exp`), so no `E97_GLOG_FLOOR`-style guard is needed. The one stability knob is a **cap
  on `η_t`** (clamp the inner LR) so a single inner step cannot blow up `ℓ_t`; default
  `η_t = η_max · sigmoid(·)` with `η_max` a config scalar.
- fp32 exact path required for the parity test; bf16/TF32 for throughput.

### 5.4 Cost accounting (sets the wall-clock prediction)

Per token: `W1@k` (`HN`), `W2@h` (`VH`), `W2ᵀ@e` (`VH`), two outer products (`HN+VH`),
plus the read `W1@q, W2@·` (`HN+VH`) — i.e. **≈ 6 tiny matmuls per token vs GDN-2's
chunked scan amortizing one matmul per chunk**. The backward roughly doubles this and adds
the `σ''` terms. Combined with the **forced sequential scan** (no chunk parallelism),
expect materially lower throughput than every chunkable head — see §8.

---

## 6. Cell config and integration

### 6.1 Knobs

- `H` (inner hidden width) — memory capacity; `{16,32,64}`.
- `σ` — `tanh` (default, bounded) | `gelu`.
- biases on/off (§1.4: adding `b1∈ℝ^H, b2∈ℝ^V` to the state costs `H+V` more floats and
  one more rank-0 grad each; default **off** to keep the state matrix-only).
- `γ_t` forget gate, `η_t` inner-LR gate (with `η_max` cap), scalar vs row-vector.
- `C` mini-batch-TTT chunk (default `1` = exact; `>1` = approx, §4.2).

### 6.2 Read timing

Default **read after write** at `θ_t` (mirrors delta-net `S_t^T q`). Knob: read at
`θ_{t-1}` (pre-write) decouples the read from the current token's own write; default keeps
the e88/delta convention.

### 6.3 Substrate placement

Drop-in head-type in the within-layer typed pool (same route as `rot`/`nonlin`): a new
`mlp-mem` branch in `typed_head_mixture`, autograd.Function wrapping the two Triton
kernels, loud-guard that it only runs when its kernels are present (mirror the existing
heads). New kernel file `ndm/triton/mlp_mem_fused.py` — **no edit conflict** with
existing kernels.

---

## 7. Taxonomy name and placement

### 7.1 Proposed name: `mlp-mem`

In the lab's terse dynamics-naming style (`decay`/`reflect`/`rot`/`nonlin`), the
distinguishing fact is **the state is a small MLP fast-weight, not a matrix**. Proposed
head-type name: **`mlp-mem`**, with the suffix convention **`-mem` reserved for "state is
a fast-weight network."**

### 7.2 It opens a THIRD axis (honest taxonomy note)

The existing grid is *eigenvalue placement × saturation*, both of which are properties of
a **matrix** recurrence. `mlp-mem` does not move within that grid — it changes the **state
representation** itself. So it is not a new cell of the 2-axis grid; it proposes

> **Axis 3 — memory representation:** linear-matrix associative (`S`, all current heads)
> vs **nonlinear-MLP functional** (`θ`, `mlp-mem`).

GDN-2 is the degenerate corner of this axis too (no-hidden-layer + linear `σ`, §2.3), so
the "GDN-2 is a special case of the Emender" proposition extends cleanly: it is also the
matrix-state corner of the memory-representation axis. **Recommendation:** keep `mlp-mem`
as a *reserved* head-type (like `rot-nonlin`) — admit it into the named taxonomy **only
once a capability probe earns it** (§8), not speculatively.

### 7.3 Boundary with `ttt-spec` (no naming collision)

`ttt-spec` and this spec describe the **same physical cell** from two angles:

- `ttt-spec` owns the **write rule** family — write = inner optimization (delta = 1 step;
  generalize via K steps / momentum / inner loss / **inner-MLP**). Its "inner-MLP" row
  *is* this memory.
- `nlmem` (this spec) owns the **state type** — the memory is a 1-hidden-layer MLP — and
  fixes the recurrence, the non-chunkability verdict, and the fused sequential kernel.

The concrete head = **`mlp-mem` state trained by a `ttt`-style (1-inner-step) write**.
`ttt-spec`'s inner-MLP row should point here for the kernel plan; this spec points to
`ttt-spec` for the write-rule generalization (K-step/momentum) that would sit on top of
the same `mlp-mem` state. No two specs claim the same name for different things.

---

## 8. Capability hypotheses + convergent-loss-null prediction

### 8.1 What a nonlinear memory can store that a linear matrix cannot

1. **Non-bilinear / nonlinear `key→value` maps.** GDN's read `S q` is linear in `q`; the
   stored association is a linear map. `mlp-mem`'s read `W2 σ(W1 q)` is nonlinear, so it
   can represent associations a single linear slot provably cannot — canonically
   **XOR-like / parity associations**: with `k=[a,b]`, store `v = a XOR b`. XOR is not
   linearly separable, so no matrix `S` reproduces it from one key; a 1-hidden-layer MLP
   does. **Probe:** in-context boolean-function recall (parity, AND-of-XORs).
2. **Higher key capacity / less crosstalk.** A rank-bounded linear matrix addresses by
   fixed inner product `S k`, so non-orthogonal keys collide. The MLP hidden layer is a
   *learned, key-adaptive* addressing/hashing, expected to store more `(k,v)` pairs with
   less interference when keys are nonlinearly separable. **Probe:** MQAR-style recall at
   high key count with correlated keys; compare collisions vs GDN.
3. **In-context nonlinear regression.** After writing a stream of `(k,v)`, the read
   `M_θ(q)` is literally a 1-SGD-step-trained MLP regressor on the in-context data, so it
   should fit **nonlinear** in-context target functions (sinusoid, piecewise) that a
   linear-memory in-context regressor cannot. **Probe:** in-context function learning,
   nonlinear targets.
4. **Bounded nonlinear state on the known separators.** With `tanh` hidden, `mlp-mem` is a
   bounded nonlinear map — on the **modular-quadratic** and **unbounded-counting**
   separators where prior nonlinear-state heads separated from linear GDN
   (`e97-nonlin-in-time-separates-modquad`, `linvsnonlin-separator-is-counting`),
   `mlp-mem` should separate where linear GDN ties. **Probe:** modular_quadratic mod
   32–64, length-extrapolation, 3 seeds.

### 8.2 Convergent-loss-null prediction (explicit, the lab's standing bar)

Consistent with this lab's repeated finding across every exotic head (`rot`, `nonlin`,
`e97_delta`: token-win/wall-loss, convergent-loss-null), the prediction is:

- **Matched-token LM BPB on natural language → TIE with dense GDN-2** (convergent-loss
  null holds; natural-language next-token does not reward a nonlinear associative memory
  over a linear one — the MLP and MLP-memory are partially redundant).
- **Matched-wall-clock → `mlp-mem` LOSES**, and by more than the previous exotics: the
  scan is **non-chunkable** (forced sequential) and each token costs ≈6 small matmuls
  forward + a second-order backward. Predict throughput **well below** the 0.7–0.9×
  range that `rot`/`tanh`-state already lost at — plausibly **0.2–0.5×** of GDN-2 — so the
  wall-clock verdict is a clear loss absent a capability win.
- **NO-GO by default, GO only on a probe.** `mlp-mem` earns a place in the taxonomy
  **only** if one of the §8.1 probes (XOR-assoc / nonlinear in-context regression /
  modular-quadratic) shows a **real, reproducible separation that survives at matched
  wall-clock on a task that genuinely needs it** — the same bar every other exotic head
  was held to. The convergent-loss-null on LM BPB is *predicted to hold*; the head's case
  rests entirely on capability, not on loss.

---

## 9. Validation checklist (this spec)

- [x] **MLP-memory recurrence + write/read rules in real math; contrast vs GDN linear
  memory explicit.** §1–§3: state `θ=(W1,W2)`, read `M_θ(q)=W2 σ(W1 q)`, write =
  one gated inner gradient step (★) with closed-form `∇_{W1},∇_{W2}`; §2.3 derives
  GDN/delta-rule as the no-hidden-layer linear corner; §2.4 the explicit contrast table.
- [x] **Chunkability classified honestly; FUSED Triton plan (not pure-torch).** §4:
  non-affine-in-state ⇒ **non-associative ⇒ sequential**, with the mini-batch-TTT
  relaxation flagged as a *different (approximate)* recurrence. §5: fused sequential
  forward (e88-style, SRAM-resident state, sparse checkpoints) + fused reverse-replay
  BPTT backward (e97-style) including the **second-order** per-step adjoint; pure-torch
  explicitly excluded.
- [x] **Capability hypotheses + convergent-loss-null prediction.** §8: XOR/parity,
  capacity, nonlinear in-context regression, bounded-state separators; explicit
  TIE-on-BPB / lose-on-wall-clock / NO-GO-absent-probe prediction.
- [x] **Taxonomy name proposed.** §7: **`mlp-mem`** (reserved `-mem` suffix = MLP
  fast-weight state), opening a third *memory-representation* axis; boundary with
  `ttt-spec` fixed.

---

## 10. Open implementation questions handed to the implementer

1. Exact expansion of the §5.2(c) `σ''` cross-terms (three `W2`-appearances, two
   `pre`-paths) — fix by fp64 finite-difference parity, per-step then full-sequence.
2. SRAM budget at `H=N=V=64, BLOCK_H>1` — may need `BLOCK_H=1` or to tile `H`.
3. Whether the read should share the key forward (`pre` at `k`) when `q≈k` paths matter —
   default keeps read independent at `q`.
4. Whether `C>1` mini-batch-TTT is worth a separate kernel, or `C=1` sequential is the
   only configuration carried to experiments (recommend `C=1` first — it is the exact
   cell the capability hypotheses are about).
