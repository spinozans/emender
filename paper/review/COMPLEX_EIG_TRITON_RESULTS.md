# Complex-eigenvalue gated-delta — REAL FUSED TRITON kernel (fwd + bwd)

**Task:** `complex-eig-triton`. Close the kernel gap left by `complex-eig-validate`:
the chunked complex gated-delta scan was pure `torch.complex` (zero `@triton.jit`),
which produces meaningless throughput numbers (the standing rule: everything must be
a fused Triton kernel before any experiment). This task implements the scan as a
**real fused Triton kernel** — forward AND backward — modeled on
`ndm/triton/e97_chunked_autograd.py`.

## What was built

`ndm/triton/complex_eig_chunked_autograd.py`:
- `_cplx_fwd_kernel` (`@triton.jit`) — forward chunk scan, one program per
  `(batch, head)`; carries the complex state `(S_r, S_i)` in registers across
  chunks; writes per-chunk entry state for the backward.
- `_cplx_bwd_kernel` (`@triton.jit`) — reverse chunk scan; recomputes the forward
  per-chunk intermediates and applies the chunked VJP, threading the complex state
  gradient `dS` in registers.
- `ComplexEigChunkedFn` (`torch.autograd.Function`) + `complex_gated_delta_chunked_triton`
  wrapper (drop-in for `complex_gated_delta_chunked`).

**No `torch.complex` / `torch.polar` in the hot path.** Triton has no native complex
type, so every complex matrix is carried as a real/imag tile pair and every complex
matmul is the four real `tl.dot` products
`(A_r+iA_i)(B_r+iB_i) = (A_r B_r − A_i B_i) + i(A_r B_i + A_i B_r)`.
The eigenvalue `λ = r·e^{iθ}` is folded into per-channel cumulative log-magnitude `G`
and phase `Φ` (the S5/LRU diagonal scan); the intra-chunk delta system is the same
strictly-lower-triangular nilpotent `(I+M)` solved by **complex Newton–Schulz**
(`ceil(log2 C)` steps, exact for nilpotent `M`); the cross-chunk carry is a
per-channel complex diagonal. The backward's matrix-inverse adjoint is the conjugate
form `dY = −Xᴴ dX Xᴴ` (the complex generalization of e97's real `−Xᵀ dX Xᵀ`).

The cheap elementwise preprocessing (complex pairing, complex L2-norm, `1/√P` query
scale) is done in **real arithmetic** outside the kernel so autograd chains it — there
is no `torch.complex` anywhere on the training path. (`torch.complex` appears once, in
the wrapper, only to package the returned *final state* for parity comparison against
the reference; the head discards it and it is not in any loop.)

Wired into `ndm/models/complex_eig_head.py`: the chunked-bulk heads now run the fused
kernel on CUDA (`cplx_fused_triton=True` default); the pure-`torch.complex`
`complex_gated_delta_chunked` is kept ONLY as the parity reference / CPU fallback. The
per-step bounded (`hardtanh`) subset still runs the sequential reference (bounded
per-step state is not chunkable — by design).

## Validation (all PASS) — `scripts/check_complex_eig_triton.py`, `tests/test_complex_eig_triton.py`

REAL random data (`torch.randn`), no mocks. GPU: NVIDIA L40S-class (49 GB), Triton 3.5.1.

### Parity vs eager per-step reference — fwd + bwd (fp32, exact)

| T    | fwd rel | dq | dk | dv | dlog_r | dtheta | dbeta | S_final |
|------|---------|----|----|----|--------|--------|-------|---------|
| 128  | 3.9e-7  | 4.2e-7 | 6.9e-7 | 4.1e-7 | 8.0e-7 | 1.1e-6 | 3.3e-7 | 5.2e-7 |
| 512  | 3.7e-7  | 4.8e-7 | 5.8e-7 | 4.9e-7 | 6.0e-7 | 8.2e-7 | 2.6e-7 | 5.5e-7 |
| 1024 | 3.2e-7  | 4.8e-7 | 6.9e-7 | 4.7e-7 | 6.0e-7 | 7.1e-7 | 4.1e-7 | 5.8e-7 |

All grads finite; far inside the `3e-3` fwd / `2e-2` grad tolerance. C=16 and C=32 both
verified; C=64 exceeds the 100 KB/SM shared-memory limit for the complex (doubled-tile)
variant, so the kernel runs C≤32 (the head's default is C=32).

**bf16 inputs → TF32 tensor-core matmuls** (the production autocast path): fwd/bwd rel
err ~6e-3 (looser, expected for TF32).

### Reductions (still hold)
- `θ=0` → real-positive decay (GDN regime): fused-vs-ref **3.2e-7**.
- `θ=π` → reflection / negative eigenvalue: fused-vs-ref **4.8e-6**.

### Throughput vs FLA fused GDN-2 — the headline

Saturating workload `B=8, H=16, N=V=32` (128 programs, neither kernel launch-bound),
forward, mean over 30 iters:

| T    | fused-TF32 | fused-fp32 | torch-complex | **vs torch-complex** | FLA GDN-2 | **TF32 / FLA** |
|------|-----------|-----------|---------------|----------------------|-----------|----------------|
| 512  | 0.343 ms  | 0.917 ms  | 2.223 ms      | **6.5× faster**      | 0.501 ms  | **0.68× (faster)** |
| 1024 | 0.476 ms  | 1.809 ms  | 3.102 ms      | **6.5× faster**      | 0.490 ms  | **0.97× (parity)** |
| 2048 | 0.976 ms  | 3.591 ms  | 8.205 ms      | **8.4× faster**      | 0.493 ms  | 1.98× (slower) |

**Verdict:** the fused complex kernel is **competitive with a real (non-complex) FLA
GDN-2** — *faster* at T≤512, parity at T=1024 — despite doing ~4× the matmul FLOPs
(complex multiply = 4 real `tl.dot`). It is **6.5–8.4× faster than the pure-`torch.complex`
path**, so the "3× pure-torch penalty" flagged by `complex-eig-validate` is eliminated.
At T=2048 it falls to 1.98× FLA: FLA's GDN-2 stays flat (~0.49 ms across all T at these
dims — overhead/latency-bound), while the complex kernel scales with its (larger) FLOP
count. **Chunkable confirmed** — exact parity at every T with sub-linear cross-chunk work.

## Test coverage
- `tests/test_complex_eig_triton.py` (14 tests): structural mandate (≥2 `@triton.jit`,
  no `torch.complex` in kernels), fwd parity T=128/512/1024 + C=16/32, bwd parity
  T=128/512/1024 (all 6 grads), reductions, bf16/TF32 path, head fused-path train step,
  fused-vs-torch-chunked agreement.
- `tests/test_complex_eig.py` (15 pre-existing tests): all green with the fused kernel
  wired into the head (incl. the LadderLM substrate smoke-train step).

## Files
- `ndm/triton/complex_eig_chunked_autograd.py` — the fused fwd+bwd Triton kernel (new).
- `ndm/models/complex_eig_head.py` — wired to the fused kernel (`cplx_fused_triton`).
- `tests/test_complex_eig_triton.py` — fused-kernel test suite (new).
- `scripts/check_complex_eig_triton.py` — parity + throughput harness (new).
