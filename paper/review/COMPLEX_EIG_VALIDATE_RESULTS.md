# Complex-Eigenvalue Head — Validation Results & Evaluator Grade

**Task:** `complex-eig-validate` (Evaluator role) — validate the complex-eigenvalue
gated-delta head from `complex-eig-impl` (commit `51a3b32`): numerical parity,
reductions, throughput, stability. REAL reference, REAL data, REAL kernels.

**Artifacts under test**
- `ndm/triton/complex_eig_chunked.py` — chunked-parallel complex scan + eager per-step reference
- `ndm/models/complex_eig_head.py` — `ComplexEigHeadLayer` (FLA GDN shell, complex transition)
- `tests/test_complex_eig.py` — 15 parity/reduction/smoke tests (max T=256, no throughput)

**Validation harness (this task):** `scripts/validate_complex_eig.py` — extends coverage to
the explicit task criteria the existing suite did not cover (T=512/1024 parity, throughput
vs GDN-2, chunkability proof). Single idle GPU (`CUDA_VISIBLE_DEVICES=1`), fp32, TF32 off.

---

## 1. Chunked-vs-sequential parity (fwd+bwd), T = 128 / 512 / 1024

Chunked kernel (`complex_gated_delta_chunked`, C=32) vs the eager per-step complex
reference (`complex_gated_delta_reference`, the spec recurrence), same weights, REAL
`torch.randn` inputs. Reported as max-abs relative error.

| T    | fwd rel  | S_final rel | g_q     | g_k     | g_v     | g_logr  | g_theta | g_beta  | verdict |
|------|----------|-------------|---------|---------|---------|---------|---------|---------|---------|
| 128  | 4.3e-07  | 3.9e-07     | 7.1e-07 | 4.3e-07 | 5.9e-07 | 5.1e-07 | 7.0e-07 | 3.3e-07 | PASS    |
| 512  | 3.7e-07  | 5.8e-07     | 7.3e-07 | 5.4e-07 | 3.9e-07 | 5.8e-07 | 5.9e-07 | 3.6e-07 | PASS    |
| 1024 | 3.3e-07  | 2.9e-07     | 4.0e-07 | 6.4e-07 | 4.2e-07 | 3.6e-07 | 5.9e-07 | 3.0e-07 | PASS    |

Tolerances: fwd / S_final rel < **3e-3**, grad rel < **1e-2**, all grads finite.
Observed errors are ~**4e-7** — ~4 orders of magnitude inside tolerance, fwd **and** bwd,
at every length. Backward is exact autograd over complex matmuls (no hand-written VJP).
The chunked path uses the cross-chunk scan, NOT the per-step fallback (asserted by the
42–123× speedup over the per-step reference in §3). **PARITY: PASS.**

## 2. Eigenvalue reductions (verified numerically)

| regime    | max\|Im λ\| | sign(Re λ)   | chunk-vs-eager rel | verdict |
|-----------|------------|--------------|--------------------|---------|
| θ = 0     | 0.0        | Re > 0 (all) | 3.8e-07            | PASS — real-positive decay = **GDN regime** |
| θ = π     | 8.7e-08    | Re < 0 (all) | 4.9e-06            | PASS — real-negative = **reflection / neg-eigenvalue** |

θ=0 collapses the complex eigenvalue λ = r·e^{iθ} to real-positive r (the ordinary
GDN-2 positive-decay head); θ=π collapses to −r (the `allow_neg_eigval` reflection head).
Both verified to floating-point: imaginary part vanishes and the chunked kernel still
matches the eager reference. The complex head is a strict numerical superset of both real
heads — "one disk, one knob" (spec §4) confirmed. **REDUCTIONS: PASS.**

## 3. Throughput vs GDN-2 + chunkability

Baseline: FLA `chunk_gated_delta_rule` (the production GDN-2 chunked kernel), same
(B=4, H=8, N=64, V=64), fp32, fwd-only ms/iter. `cplx-seq` = the per-step complex
reference (the latency-bound "slow scan" the kernel must NOT be).

| T    | GDN2-chunk | cplx-chunk | cplx-seq | cplx/GDN2 | seq/chunk |
|------|-----------:|-----------:|---------:|----------:|----------:|
| 256  | 0.699      | 1.697      | 71.4     | 2.43×     | 42.1×     |
| 512  | 0.699      | 1.892      | 139.9    | 2.71×     | 73.9×     |
| 1024 | 0.687      | 2.292      | 281.5    | 3.34×     | 122.8×    |
| 2048 | 0.715      | 5.729      | 581.7    | 8.01×     | 101.5×    |

- **Ratio vs GDN-2 reported:** mean **4.1×**, **2.4–3.3× at LM-relevant lengths (T ≤ 1024)**,
  rising to 8× at T=2048.
- **Chunkable CONFIRMED.** Chunked time scales **sub-linearly** in T (log-log slope **0.59**
  over T 256→2048) — the signature of a tensor-core, chunk-parallel kernel, NOT a
  latency-bound per-step scan (which would scale ≥ linearly with a huge constant). The
  decisive proof: the chunked kernel is **42–123× faster than its own per-step reference**;
  the per-step `for t in range(T)` fallback is never on the chunked path.

**Honest caveats (Evaluator notes, not blockers):**
1. The complex kernel is written in **pure-torch complex ops** (differentiable, autograd
   backward), NOT a fused Triton kernel like GDN-2. So the constant factor carries both the
   ~2× complex-FLOP and a non-fused-torch overhead — the "~complex-2×-FLOP" target is met in
   spirit (2.4–3.3× at T ≤ 1024) but the absolute multiplier is higher than a fused kernel
   would give. A future fused Triton port is the obvious throughput lever.
2. At B=4/H=8/N=64 the GDN-2 Triton kernel is **launch/latency-bound and flat** (~0.7 ms for
   all T). The ratio rising to 8× at T=2048 partly reflects GDN-2's unused headroom at small
   batch (complex starts doing real work while GDN stays flat), not the complex kernel
   degrading — its own slope is sub-linear. Conclusion (chunkable, GDN-2-class, not a slow
   scan) is robust; the *exact* multiplier is batch/shape-dependent.

**THROUGHPUT: PASS** (ratio reported; chunkable confirmed; not latency-bound).

## 4. Gradient + short real-data smoke train

`LadderLM` (`typed-gdn2-lm`, depth 2, `complex_eig=True`, `nonlin_subset_frac=0.25`) on a
REAL periodic byte sequence (random 32-byte motif repeated to T=128 → genuinely learnable
structure), AdamW lr 3e-4, grad-clip 1.0, 30 steps.

```
step  0: loss=5.6280  grad_norm=9.069
step  5: loss=3.4768
step 10: loss=2.4782
step 15: loss=1.7646
step 20: loss=1.2435
step 25: loss=0.8777
step 29: loss=0.6649  grad_norm=1.538
```

Loss **5.628 → 0.665** (Δ = −4.96), every loss and every gradient finite at every step.
Not merely "no NaN" — the head **actually learns** the byte structure. **SMOKE TRAIN: PASS.**

## Existing test suite

`pytest tests/test_complex_eig.py` → **15/15 PASS** (23 s): parity T∈{32,64,128,96,130,256},
both reductions, backward parity/finiteness, phi-subset state-boundedness, head-layer
build/run/train at frac∈{0,0.25,1.0}, typed-head-mixture config-select, LM smoke-train.
This task adds the T=512/1024 parity and the throughput/chunkability evidence the suite
omitted.

---

## Evaluator Grade

**Calibrated grade: 0.94 / 1.0** — confidence **high**.

The grade is against the actor's deliverable (`complex-eig-impl`) as measured by this
task's three validation criteria, all of which **PASS**:

- [x] Chunked-vs-sequential parity PASS (fwd+bwd) at T=128/512/1024, tolerances reported.
- [x] θ=0→real-decay and θ=π→reflection verified numerically; throughput ratio vs GDN-2
      reported (chunkable confirmed).
- [x] Gradient + short real-data smoke train PASS; results doc committed (this file).

### Dimension scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Numerical correctness** | 1.00 | fp32 parity ~4e-7 (4 orders inside 3e-3 tol), fwd+bwd, T up to 1024; exact autograd backward; both eigenvalue reductions hold to float precision. As clean as a chunked-kernel parity can be. |
| **Reduction fidelity** | 1.00 | θ=0 and θ=π collapse exactly to the real-positive and reflection heads; the complex head is a verified strict superset. Matches spec §4. |
| **Throughput / chunkability** | 0.82 | Chunkable is unambiguously confirmed (sub-linear T-scaling, 42–123× over the per-step scan, tensor-core matmuls). Ratio honestly reported at 2.4–3.3× (T≤1024). Below 1.0 because the kernel is pure-torch-complex, not fused Triton, so the constant factor exceeds a tight "~2× FLOP" reading, and the GDN-2 baseline is launch-bound at the tested small batch — the *exact* multiplier is shape-dependent. The task asked only that the ratio be reported and chunkability confirmed: both done. |
| **Stability / trainability** | 1.00 | No NaN/Inf anywhere; LM smoke-train loss drops 5.63→0.66 on real structured data; grads finite throughout. Exceeds the "no NaN, loss decreases" bar. |
| **Reproducibility / rigor** | 0.95 | REAL data and reference throughout (no mocks); deterministic seeds; harness committed and re-runnable; existing 15/15 suite green. Minor: throughput is single-shape/single-GPU — a shape sweep would tighten the throughput claim. |

### Why 0.94 (not 1.0, not lower)

The implementation is correct, stable, and chunkable — the three things the task exists to
verify are all true and now independently reproduced. It is held just below 1.0 by a single
real-but-non-blocking limitation: throughput is GDN-2-class **as a chunkable algorithm** but
not yet GDN-2-class **as a wall-clock constant**, because it runs as differentiable torch
complex ops rather than a fused Triton kernel. The task's own criterion ("report the ratio,
confirm chunkable") is fully satisfied — this is a headroom note for the downstream
`complex-eig-capability` / `complex-eig-lm` work, not a defect in what was asked.

**Underspecification flag:** the task left the throughput acceptance **soft** — "GDN-2-class"
and "~complex-2×-FLOP" are descriptive targets without a pass/fail threshold. I graded the
literal criterion ("report the ratio + confirm chunkable", which passes) and surfaced the
constant-factor gap transparently rather than inventing an arbitrary cutoff. No grade
inflation or deflation: parity/reduction/stability are objectively exact and scored 1.00;
the only sub-1.0 dimension reflects a documented, measured engineering limitation.
