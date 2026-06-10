# ttt-validate — Validation of the fused `refit` (TTT inner-opt write) kernel

**Task:** `ttt-validate` · **Kernel:** `ndm/triton/refit_chunked_autograd.py` (the
`refit` momentum-delta inner-optimization WRITE head of
`paper/review/TTT_WRITE_SPEC.md`) · **Date:** 2026-06-10
**Env:** NVIDIA RTX 6000 Ada Generation, fp32, torch 2.9.1+cu128, triton 3.5.1,
repo `aade0e3`. All GPU runs via the gpu-broker lease (`scripts/gpu_lease.sh`).

## What this cell is (one line)

The recurrent state is a linear fast-weight matrix `S [N,V]` plus a momentum buffer
`M [N,V]`, updated by **one heavy-ball inner-optimizer step per token** on the inner
reconstruction loss `½‖v_t − Sᵀk_t‖²` (with erase/write/decay gates and a momentum gate
`μ`). The joint `(S,M)` recurrence is an upper-triangular 2×2 `(g,μ)` companion per
channel → affine in the state → **chunk-parallelizable**: the intra-chunk twiddle
factorizes as `Φ = A1 @ A2` (two `[C,C]` tensor-core dots). Two real `@triton.jit`
kernels (fwd + reverse-replay bwd), **no torch fallback in the hot path**. At `μ≡0` the
momentum buffer collapses (`M_t = r_t`) and the cell becomes **exactly** the gated-delta
/ e97 write — the delta-rule = one-inner-step special case (`has_mom=False` fast path).

## Verdict: PASS — all validate criteria met

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Parity fwd+bwd within tol, T=128/512/1024 | **PASS** — `tests/test_refit_chunked.py` + `test_refit_head.py` 18/18 |
| 2 | Throughput ratio vs GDN-2, no pure-torch penalty | **REPORTED** — refit is FASTER at every T (1.6–3.1x f+b); see table |
| 3 | Gradient + short real-data smoke train | **PASS** — loss 5.50→1.33 (momentum) / →1.58 (delta) on real repo bytes |
| 4 | Results doc committed | this file |

## 1. Parity (fwd + bwd), T = 128 / 512 / 1024 — PASS

`python -m pytest tests/test_refit_chunked.py tests/test_refit_head.py -q` → **18 passed**
(13 chunked-kernel + 5 head). Gates (fp32 unless noted):
- **fwd** rel err `< 5e-5` vs the eager heavy-ball recurrence `refit_eager_reference`,
  at T=128/512/1024 (momentum on); and `< 2e-4` vs the independent `e88_torch_reference`
  delta recurrence at `has_mom=False`, confirming the delta = one-inner-step special case.
- **bwd**: all 7 grad sinks (`k,v,q,decay,μ,e,w`) rel err `< 3e-3` fp32 / `< 6e-2` bf16
  vs eager autograd through the reference recurrence (an independent full BPTT, not
  self-check), momentum on AND off.
- log-decay grad path; the exposed inner-step knob `K` (truncated-Neumann: K=5 near-exact,
  K=1 a genuine ~1.87 approximation); strong-decay/gate-floor finite-grad regressions.

## 2. Throughput vs GDN-2 — REPORTED (no pure-torch penalty)

Honest fp32 (TF32 off), CUDA-event timed, 20 iters after 5 warmup. Fused chunked `refit`
vs **FLA GDN-2** (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`, the chunked
gated-delta baseline) at matched shapes B=4, H=8, N=V=32, chunk C=32. Both the full
momentum path (`refit-mom`) and the delta special case (`refit-del`, the FLA-class
algorithm — the apples-to-apples check) are reported. `ratio = GDN-2 time / refit time`
(>1 ⇒ refit FASTER). Reproduce: `python scripts/bench_refit_vs_gdn2.py --iters 20`.

**Forward + backward (the training-relevant number):**

| T | refit-mom f+b | refit-del f+b | GDN-2 f+b | mom ratio | del ratio |
|---|---|---|---|---|---|
| 128 | 0.79 ms / 0.65M tok/s | 0.82 ms / 0.62M | 2.03 ms / 0.25M | **2.58x** | **2.47x** |
| 512 | 1.24 ms / 1.66M tok/s | 0.86 ms / 2.37M | 1.96 ms / 1.04M | **1.59x** | **2.27x** |
| 1024 | 2.31 ms / 1.77M tok/s | 1.30 ms / 3.16M | 3.96 ms / 1.04M | **1.71x** | **3.05x** |

Forward-only: refit-mom 3.2x/3.2x/1.15x, refit-del 3.2x/2.4x/2.2x at T=128/512/1024.

**Reading.** Unlike the non-chunkable mlp-mem cell (which crosses over to ≈0.5x of GDN-2
at long T), `refit` is **chunk-parallel tensor-core** and stays **FASTER than FLA GDN-2 at
every measured T** — fwd+bwd 1.6–3.1x. This is the headline validate result: there is **no
pure-torch penalty**, because the hot path is two real fused Triton kernels, not an eager
fallback. The delta special case (`has_mom=False`) runs the same algorithmic class as
GDN-2 and is 2.3–3.1x faster fwd+bwd, so the speedup is the kernel's tensor-core
efficiency, not an apples-to-oranges artifact; the full momentum path adds the `μ`-gate
twiddle (`A2`) and remains ≥1.6x. (Refit's per-token tok/s also grows with T like any
chunked kernel; the absolute Mtok/s here is on the small N=V=32 head, launch-overhead
dominated at T=128.)

## 3. Gradient + real-data smoke TRAIN — PASS

`python scripts/refit_lm_train_smoke.py --steps 80`. A genuine optimization loop (not a
single-step grad-finiteness check): `LadderLM(level='typed-gdn2-lm')` with the typed-head
mixture forced to **ALL `refit`** (`head_type_logits` mass on the refit slot, idx 8 of
`TYPE_NAMES`), dim=256 depth=4 heads=8 n_state=32, 3.495M params, AdamW lr=3e-3, B=8 T=256,
**80 steps**. Every recurrent head in every layer is the momentum-delta inner-optimizer.

**REAL DATA:** the byte stream is the repository's own source (2,000,000 bytes from 310
real `.py`/`.md` files) — genuine in-tree UTF-8 text, next-byte prediction. No synthetic
corpus.

```
step   0  loss 5.5044  bpb 7.941  gnorm 9.94
step  10  loss 2.6123  bpb 3.769
step  40  loss 2.2763  bpb 3.284
step  79  loss 1.3305  bpb 1.920
mean loss first-5 = 4.1849   last-5 = 1.5320   drop = 2.653 (63.4%)   bpb 6.038 -> 2.210
```

The delta special case also learns: `--has_mom 0 --steps 60` → first-5 4.118, last-5 1.830,
55.6% drop. Loss falls from the 256-vocab uniform prior (ln256 ≈ 5.545) toward ~1.3–1.6
within a short run, with finite loss and finite clipped grad-norm at every step. The fused
chunked fwd+bwd kernels therefore carry **real learning signal** end-to-end through
LadderLM (both momentum and delta modes), not merely finite gradients. (The overfit-a-batch
wiring check from ttt-triton, 4.21→0.006, still passes too; this adds the real-data leg.)

## Artifacts

- `scripts/bench_refit_vs_gdn2.py` — throughput benchmark vs FLA GDN-2 (NEW)
- `scripts/refit_lm_train_smoke.py` — real-data smoke train, all-refit LadderLM (NEW)
- `paper/review/TTT_VALIDATE_RESULTS.md` — this doc (NEW)
- `tests/test_refit_chunked.py`, `tests/test_refit_head.py` — parity suites (from
  ttt-triton, re-run green: 18/18)
- `ndm/triton/refit_chunked_autograd.py`, `ndm/models/refit_head.py` — the validated
  kernel + head (from ttt-triton)

## Notes for downstream (`ttt-capability`)

- The head is correct (parity, all 7 grads), fast (FASTER than FLA GDN-2 fwd+bwd at every
  T — no sequential-scan or pure-torch penalty), and trainable (learns on real data in
  both momentum and delta modes). The open question is **capability**, not implementation.
- Because refit clears the wall-clock bar (it is a chunked tensor-core kernel, not the
  sequential mlp-mem story), the capability question is whether the inner momentum (the
  `μ`-gate / heavy-ball surprise EMA) buys any capability the plain gated-delta
  (`has_mom=False`) cell cannot reach — the convergent-loss-null hypothesis to test next.
