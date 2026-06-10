# nlmem-validate — Validation of the fused nonlinear-MLP-memory kernel

**Task:** `nlmem-validate` · **Kernel:** `ndm/triton/mlp_mem_fused.py` (the `mlp-mem`
head of `paper/review/NONLIN_MEMORY_SPEC.md`) · **Date:** 2026-06-10
**Env:** NVIDIA RTX 6000 Ada Generation, fp32, torch 2.9.1+cu128, triton 3.5.1,
repo `8fae70c`. All GPU runs via the gpu-broker lease (`scripts/gpu_lease.sh`).

## What this cell is (one line)

The recurrent **state** is the parameters `theta=(W1,W2)` of a tiny 1-hidden-layer
MLP `M(x)=W2·tanh(W1·x)`, updated by **one gated inner gradient step per token** on
`½‖M(k_t)−v_t‖²`. The transition is non-affine in `theta` → **non-associative → no
chunked scan**; the kernel is a fused sequential per-token scan with sparse forward
checkpoints + reverse-replay BPTT (exact closed-form 2nd-order per-step VJP). Two real
`@triton.jit` kernels; eager ref is parity-only, never on the train path.

## Verdict: PASS — all four validate criteria met

| # | Criterion | Result |
|---|-----------|--------|
| 1 | Parity fwd+bwd within tol, T=128/512/1024 | **PASS** — `tests/test_mlp_mem_triton.py` 8/8 |
| 2 | Throughput vs GDN-2 reported | **REPORTED** — sequential-scan crossover, see table |
| 3 | Gradient + short real-data smoke train | **PASS** — loss 5.55→2.43 on real repo-source bytes |
| 4 | Results doc committed | this file |

## 1. Parity (fwd + bwd), T = 128 / 512 / 1024 — PASS

`python -m pytest tests/test_mlp_mem_triton.py -q` → **8 passed**. Gates (fp32):
- **fwd** rel err `< 2e-4` on `out`, `< 2e-4` abs on final `W1`/`W2`, at T=128/512/1024.
- **bwd** grads `dk,dq,dv,deta,dgamma` rel err `< 3e-3` vs eager autograd, at T=128/512/1024.
- unaligned T (130, pads to ckpt multiple) → finite grads; varied HID=64 corner → parity.

The fused bwd is checked against PyTorch autograd through the eager reference recurrence
(`mlp_mem_torch_reference`), i.e. an independent full BPTT, not against itself.

## 2. Throughput vs GDN-2 — REPORTED

Honest fp32 (TF32 off), CUDA-event timed, 20 iters after 5 warmup. mlp-mem fused
sequential scan vs **FLA GDN-2** (`fla.ops.gated_delta_rule.chunk_gated_delta_rule`,
the chunked gated-delta baseline) at matched shapes B=4, H=8, N=V=HID=32.
`ratio = GDN-2 time / mlp-mem time` (>1 ⇒ mlp-mem faster; <1 ⇒ mlp-mem slower).
Reproduce: `python scripts/bench_mlp_mem_vs_gdn2.py --iters 20`.

| T | mlp-mem fwd | GDN-2 fwd | ratio | mlp-mem f+b | GDN-2 f+b | ratio |
|---|---|---|---|---|---|---|
| 128 | 0.16 ms / 3.19M tok/s | 0.64 ms / 0.80M | **3.96x** | 0.61 ms / 0.84M | 2.00 ms / 0.26M | **3.27x** |
| 512 | 0.57 ms / 3.58M tok/s | 0.63 ms / 3.25M | **1.10x** | 1.97 ms / 1.04M | 2.02 ms / 1.01M | **1.03x** |
| 1024 | 1.14 ms / 3.60M tok/s | 0.63 ms / 6.50M | **0.55x** | 4.26 ms / 0.96M | 2.03 ms / 2.02M | **0.48x** |

**Reading.** mlp-mem throughput is **flat ~3.6M tok/s fwd** (a constant-work-per-token
sequential scan: time ∝ T, tok/s ≈ const). GDN-2's chunked kernel has fixed launch
overhead amortized over the chunk, so its tok/s **grows** with T. The two cross over
near T≈512: mlp-mem is **faster at short T** (3.3–4.0x, GDN-2 dominated by overhead) and
**slower at long T** (≈0.5x at T=1024, where GDN-2's chunked parallelism wins). This is
exactly the spec's prediction (sec 4): a non-chunkable sequential scan pays a growing
wall-clock penalty vs a chunkable baseline as T grows. The number is reported here, not
asserted — it is a real measured crossover, and it tells the downstream capability task
that any LM-scale (long-T) deployment of this head carries a sequential-scan throughput
cost relative to GDN-2.

## 3. Gradient + real-data smoke TRAIN — PASS

`python scripts/mlp_mem_lm_train_smoke.py --steps 80`. A genuine optimization loop (not a
single-step grad-finiteness check): `LadderLM(level='mlp-mem-lm')`, dim=256 depth=4
heads=8 n_state=32 HID=32, 2.997M params, AdamW lr=3e-3, B=8 T=256, **80 steps**.

**REAL DATA:** the byte stream is the repository's own source (2,000,000 bytes from 304
real `.py`/`.md` files) — genuine in-tree UTF-8 text, next-byte prediction. No synthetic
corpus.

```
step   0  loss 5.5454  bpb 8.000  gnorm 10.39
step  10  loss 2.8696  bpb 4.140
step  40  loss 2.7976
step  79  loss 2.4339  bpb 3.511
mean loss first-5 = 4.1195   last-5 = 2.5414   drop = 1.578 (38.3%)   bpb 5.943 -> 3.667
```

Loss falls from the 256-vocab uniform prior (ln256 ≈ 5.545) to ~2.43 within 80 steps
with finite loss and finite clipped grad-norm at every step. The fused sequential
fwd+bwd kernels therefore carry **real learning signal** end-to-end through LadderLM, not
merely finite gradients. (The wiring-only random-batch check `scripts/mlp_mem_lm_smoke.py`
from nlmem-triton still passes too; this adds the missing "does it actually learn on real
data" leg.)

## Artifacts

- `scripts/bench_mlp_mem_vs_gdn2.py` — throughput benchmark (NEW)
- `scripts/mlp_mem_lm_train_smoke.py` — real-data smoke train (NEW)
- `tests/test_mlp_mem_triton.py` — parity suite (from nlmem-triton, re-run green)
- `ndm/triton/mlp_mem_fused.py` — the validated kernels (from nlmem-triton)

## Notes for downstream (`nlmem-capability`)

- The head is correct (parity) and trainable (learns on real data). The open question is
  **capability**, not implementation.
- Throughput is the expected sequential-scan story: a short-T win but a long-T (≈0.5x at
  T=1024) loss vs chunked GDN-2. A capability case for this head must clear that wall-clock
  bar at LM-relevant T, mirroring the convergent-loss-null / wall-loss-NO-GO pattern seen
  for the other exotic heads in this program.
