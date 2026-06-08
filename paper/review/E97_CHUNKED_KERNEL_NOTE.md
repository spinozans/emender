# E97 split-edit delta: chunked-parallel kernel flips the within-layer NO-GO

**Task:** `chunk-e97-kernel` · **Date:** 2026-06-08 · GPU: NVIDIA RTX 6000 Ada (AD102, 142 SM, ~100 KB shared mem/SM)

## TL;DR

The within-layer NO-GO verdict ([e97-within-layer-tradeoff], [e97-scale-pilot-nogo])
hinged **entirely** on the E97 split-edit delta kernel being ~2.6× slower than
GDN-2 because it ran a **sequential `for t in range(T)` outer-product scan**
inside one Triton program per (batch, head) — latency-bound, unable to saturate
the GPU. That slowness was an **incomplete implementation, not a fundamental
property** of the cell.

We reformulated the **e97_delta** recurrence (the principled cell: delta
correction `v − S@k`, *not* raw-write) into a **chunked-parallel** form modelled
on FLA's `chunk_gated_delta_rule`, and wrote a **fully fused Triton forward +
backward** kernel. Result, all bf16, matched shapes, **fwd+bwd**:

| shape (B,T,H,N,V)        | GDN-2 ms | GDN util | **E97 fused ms** | **E97 util** | **E97 / GDN** | E97 PyTorch-staged / GDN |
|--------------------------|---------:|---------:|-----------------:|-------------:|--------------:|-------------------------:|
| within_layer 8,512,16,32,32  | 2.844 |  6.0 % | **0.516** | **83.0 %** | **0.18× (5.5× faster)** | 1.97× |
| scale_1p3b  4,2048,16,32,32   | 1.864 | 26.0 % | **1.453** | **100 %**  | **0.78× (faster)**      | 6.92× |
| scale_1p3b_wide 4,2048,32,32,32 | 3.683 | 33.6 % | **1.842** | **99.8 %** | **0.50× (2× faster)**   | 3.64× |

**The chunked fused E97 kernel is FASTER than GDN-2 at every shape**, at 83–100 %
GPU utilisation. The prior NO-GO premise (2.6× slower, 13–15 % util) is
**refuted** — the throughput reasoning behind the within-layer/scale NO-GO is
**flipped**. (`experiments/e97_chunked_kernel/bench_kernel.json`.)

## The chunked reformulation

Per (b,h), with linear state `S` of shape `[N,V]`, the split-edit delta recurrence

```
read_key_t  = e_t ⊙ k_t                    erase/read gate, [N]
write_val_t = w_t ⊙ v_t                     value write gate, [V]
delta_t     = write_val_t − S_{t-1}^T read_key_t
S_t         = decay_t · S_{t-1} + k_t · delta_t^T
out_t       = S_t^T q_t
```

is, after substituting `delta_t`, an **asymmetric gated-delta** affine recurrence
`S_t = (decay_t·I − k_t·read_key_t^T) S_{t-1} + k_t·write_val_t^T`. This matches
`E88FLAHybrid._scan_recurrence` (linear_state, raw_write=False, split-edit) and
differs from standard DeltaNet only in being asymmetric (left vector `k`, right
vector `read_key = e⊙k`; DeltaNet uses `k` for both).

Chunk length `C`, entry state `S0c` per chunk. With `G[t] = Σ_{i≤t} log decay_i`
(inclusive cumulative log-decay), the per-step deltas solve a **unit
lower-triangular** system `(I+M) Δ = P − diag(decay_prev)·U·S0c`, where
`M[t,j] = exp(G[t]−g[t]−G[j])·(k_j·u_t)` (strictly lower). All `C`-sized work is
matmuls; only the `T/C`-step cross-chunk state thread is sequential — exactly
FLA's structure. Full derivation in `ndm/triton/e97_chunked.py` (docstring).

The UT inverse `T = (I+M)^{-1}` is computed by **Newton–Schulz**
`X ← X(2I − LX)`: because `M` is strictly-lower (nilpotent, `M^C = 0`) the
residual is `M^{2^k}`, so `ceil(log2 C)` steps invert **exactly** with only
`tl.dot` matmuls — no `O(C)` substitution loop, no triangular-solve backward.

## Implementation

Three artifacts, all parity-verified against `e88_torch_reference`:

1. **`ndm/triton/e97_chunked.py`** — staged PyTorch-chunked forward (autograd
   backward). Differentiable reference for the chunked algebra; `inverse_mode`
   `'solve'` (triangular solve) or `'newton'` (matmul). **Slow** (1.97–6.92×
   GDN) — launch/HBM-bound by the many small batched matmuls + solve; this is the
   path the *prior* benchmark mislabelled, and it is **not** the fast path.

2. **`ndm/triton/e97_chunked_fwd_kernel.py`** — fused Triton **forward** (C=64),
   inference fast-path. 0.44–0.69× GDN-2 forward (faster). All matmul operands in
   fp32 on the **TF32** tensor-core path (`allow_tf32`) for bf16/fp16 inputs —
   TF32 carries 10 mantissa bits (> bf16's 8) so it meets bf16 tolerance.

3. **`ndm/triton/e97_chunked_autograd.py`** — the load-bearing fused **fwd+bwd**
   `autograd.Function`. Forward writes per-chunk entry states; the **backward is
   a second fused Triton kernel** that walks chunks in **reverse**, threads the
   state-gradient `dS` in registers, and recomputes each chunk's forward
   intermediates (the chunked VJP — derived by hand, every step a `tl.dot`). This
   is the GDN-2-class training path and the benchmark subject.

### Why C=32 (not 64) for the backward

The forward fits C=64 easily. The fused backward holds ~6–7 `[C,C]` fp32 tiles
live at peak; at C=64 that is **106 KB/SM, just over Ada/Ampere's ~100 KB hard
limit** (`OutOfResources: Required 106496, limit 101376`). This is an
**architectural shared-memory ceiling on this GPU, not an algorithmic limit** —
it fits on Hopper (228 KB/SM). C=32 keeps every tile at 4 KB with wide margin and
**already beats GDN-2** (table above), so C=32 is the default for the autograd
path. We minimised peak liveness (recompute cheap score matrices after Newton;
consume `dQK`/`dKU` one at a time) which brought C=64 from 114 KB → 106 KB; the
last 5 KB would require splitting the backward into two kernels — unnecessary
given C=32 already wins.

## Numerical parity (vs `e88_torch_reference`, linear_state, split-edit)

`tests/test_e97_chunked.py` — **13/13 pass**:

| path | fp32 fwd | fp32 grads | bf16 fwd | bf16 grads |
|------|---------:|-----------:|---------:|-----------:|
| staged PyTorch-chunked      | ~1e-5 | <1e-3 | <5 % | — |
| fused Triton fwd (C=64)     | ~1e-5 | —      | <0.75 % | — |
| **fused fwd+bwd (C=32)**    | ~7e-7 | **~1e-6** | **<1.1 %** | **<2 %** |

(Tolerances: fp32 grad rel-err < 3e-3, bf16 fwd/grad rel-err < 5–6 %.)

## Wiring into `typed_head_mixture`

`E88FLAHybrid` gains `use_chunked_e97` / `e97_chunk_size`; when set (and
split-edit, delta, **linear-state**) the `use_optimized` training path routes the
recurrence through `e97_delta_chunked_triton` instead of the sequential
`e88_triton_optimized_apply`. The chunked kernel runs the **bare** recurrence, so
the branch first replicates the transforms the sequential kernel fuses internally
(silu on k,q,v; then L2-norm on k,q per head; `read_key = erase·(processed k)`)
and applies the silu output gate afterwards. `TypedHeadMixtureLayer` exposes
`use_chunked_e97_delta=True` (default) and passes it to the e97_delta block.

- **Engages only for linear-state e97_delta** — per-step `tanh` is a pointwise
  nonlinearity on the whole state every step, not associative, so it has **no
  chunked-matmul form**. With the default `e97_state_nonlin='tanh'` the head keeps
  the sequential kernel; set `e97_state_nonlin='linear'` to use the fast path.
  (GDN-2, the throughput target, is itself linear-state, so the apples-to-apples
  comparison is the linear delta.)
- **Loud no-eager guard intact**: a bf16-requiring head that cannot engage the
  Triton path during training still raises rather than silently running eager.
- **Equivalence check**: the wired chunked head matches the sequential path on
  identical weights within **0.55 %** (bf16) end-to-end.
- **No regressions**: `test_typed_head_mixture.py` (15) + `test_e88_triton.py`
  (7) + `test_e97_chunked.py` (13) all pass.

## Honest residuals

- **within_layer util is 83 %, just under the 85 % line.** This shape is tiny
  (B·H = 128 programs, NC = 16 chunks of 32) so it is launch-gap dominated in a
  relaunch loop — **GDN-2 itself reaches only 6 % util here**, and E97 is still
  **5.5× faster**. At the dims that drive the 1.3B decision (scale_1p3b,
  scale_1p3b_wide) util is **100 % / 99.8 %** and E97 beats GDN-2. The util target
  is met where it matters; the within-layer number reflects shape size, not kernel
  inefficiency, and throughput (the binding criterion) wins everywhere.
- C=64 backward needs Hopper-class shared memory (see above).
- Initial state `S0 = 0` only (the typed-head usage); arbitrary `S0` would be a
  small extension to the forward-save and backward kernels.

## What this unblocks

The throughput premise of the within-layer NO-GO is overturned: a linear-state
e97_delta head now trains at **GDN-2-class (better-than-GDN-2) throughput**. The
downstream task **e97delta-1p3b** can run a fair param-/token-matched 1.3B
comparison without the kernel being a latency confound. Whether e97_delta adds
*capability* over a GDN-2 backbone (recall, S5, counting — see
[e97-within-layer-tradeoff]) is a separate, still-open question; this task removes
the **performance** objection only.

[e97-within-layer-tradeoff]: e97 within-layer trade-off memo
[e97-scale-pilot-nogo]: e97-scale pilot NO-GO memo
