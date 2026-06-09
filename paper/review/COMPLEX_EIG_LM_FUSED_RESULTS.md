# Complex-Eigenvalue Head — Matched-Compute LM Verdict on the FUSED Triton Kernel

**Task:** `complex-eig-lm-2` (`complex-eig-lm-fused`) — RE-RUN the convergent-loss +
wall-clock LM comparison for the complex-eigenvalue gated-delta head, this time on
the **fused Triton kernel** (`complex-eig-triton`, `complex_gated_delta_chunked_triton`,
no `torch.complex` in the hot path), NOT the pure-torch path used by the prior
`complex-eig-lm` run. The prior run's matched-token bpb tie is function-valid, but
its wall-clock numbers were measured on the unfused `torch.complex` scan (a ~2–3×
Python-complex penalty) and were therefore meaningless. This redo measures
wall-clock on the REAL fused kernel.

REAL data (the Pile, byte-level), REAL kernels (the fused Triton complex scan +
FLA GatedDeltaNet), REAL training. No mocks.

---

## 0. TL;DR

- **Hot path confirmed fused.** The complex arm's recurrence runs
  `complex_gated_delta_chunked_triton` (fused @triton.jit, real/imag-decomposed,
  no `torch.complex`); instrumentation counts **60 fused-kernel calls, 0
  torch.complex calls** during a timed window (6 layers × 10 steps).
- **Matched-token → TIE re-confirmed.** complex 2.1356 vs baseline 2.0857 bpb,
  Δ = **+0.050 bpb** on the identical seed-42 98.3M-token stream at matched params
  — the predicted convergent-loss-null (complex never lowers the loss ceiling).
- **Matched-wall-clock → STILL A LOSS, and the "3× penalty eliminated" premise is
  REFUTED for trainable LM.** The fused kernel is only fast in **TF32**
  (0.72–0.85× baseline), but **TF32 NaN-diverges within <30 optimizer steps** at
  the run's lr (2e-3) — independent of the inverse-decay guard. The only
  **trainable** fused configuration is **full-fp32 matmuls**, which runs at
  **0.21× baseline (4.7× slower)** — *slower than the very `torch.complex` path it
  was meant to replace* (0.46× on the same hardware). At matched 9.2-min wall-clock
  the baseline sees 4.6× more tokens and wins by **+0.302 bpb**.

The convergent-loss bet is upheld (loss is not the differentiator: matched-token
tie). But the optimistic wall-clock expectation — "fused ⇒ matched-wall-clock now
fair" — does **not** hold: the fused kernel's throughput win lives entirely in a
TF32 regime that cannot train this head, and its trainable (fp32) regime is the
slowest option of all.

---

## 1. Setup — identical single-variable comparison (unchanged from the prior run)

Both arms share a **byte-for-byte identical FLA GatedDeltaNet shell** (q/k/v
projections, short-conv, L2-norm, output gate, RMSNorm, out-proj) and differ in
**exactly one thing**: the recurrent transition.

| arm | level | transition | kernel (this run) |
|-----|-------|-----------|-------------------|
| **baseline (real)** | `real-eig-gdn` | per-head **real** scalar decay `g_t` | FLA native chunked |
| **complex-everywhere** | `complex-eig-lm` | per-key-channel **complex** eigenvalue `λ = r·e^{iθ}` | **fused Triton** `complex_gated_delta_chunked_triton` |

**Shared config (both arms):** dim=512, depth=6, n_heads=8, n_state=64 (P=32 complex
channels), expansion=1.0, SwiGLU MLP/layer, vocab=256 (byte-level), seq_len=512,
batch=16, schedule-free AdamW, lr=2e-3, weight_decay=0, **grad-clip 1.0**, bf16,
seed=42 (**identical byte stream** across arms).

**Matched params** (within ~1%, baseline-favored, equalized via MLP width):
baseline `mlp_ratio=2.33` → **19,295,200**; complex `mlp_ratio=2.0` →
**19,103,200** (reproduces the prior run's counts exactly).

**Data:** train `/mnt/nvme2n1/erikg/pile.txt` (1.2 TB, byte-level, random
doc-aligned start). Held-out = disjoint 64 MB Pile slice
(`/mnt/nvme2n1/erikg/complex_eig_lm_val.txt`). bpb = (CE_nats/ln2)/1.0.

> **Eval correctness.** train.py's schedule-free *averaged-(x)* final eval is broken
> in this environment (it reports `FINAL_HELDOUT_BPB` ≈ 2.9–23, the known
> reconstruction bug). The verified metric is the **y-iterate** (the actual trained
> weights), recovered by loading the optimizer state and calling `optimizer.train()`
> (`scripts/eval_ceig_bpb.py`, 400 val batches = 3,276,800 tokens). All bpb below
> are y-iterate.

---

## 2. The fused kernel CONFIRMED on the hot path — and a numerical wall

`scripts/bench_ceig_lm_fused.py` wraps the three scan entry points
(`complex_gated_delta_chunked_triton`, `complex_gated_delta_chunked`,
`complex_gated_delta_reference`) with call counters and runs the real LM forward+
backward. Result for `complex-everywhere`:

```
HOT-PATH CONFIRMED: fused calls=60, torch.complex calls=0
```

So the recurrence is the fused Triton kernel, NOT the torch-complex reference.
✅ Validation item 1.

**But the fused kernel cannot train this LM in its fast (TF32) configuration.**
With the head's default (`cplx_fused_triton=True`, bf16 autocast ⇒ TF32 `tl.dot`),
training NaN-diverges within a few optimizer steps:

| config | NaN at step | note |
|--------|------------:|------|
| TF32, inv-decay-guard=80 (default) | 11 | — |
| TF32, inv-decay-guard=30 | 31 | guard lowered |
| TF32, inv-decay-guard=20 | 8 | guard lowered |

The non-monotonic dependence on the decay guard shows the instability is **TF32
matmul precision** (the Newton–Schulz `(I+M)` inverse and the decay-absorbed
inverse key `1/cumdecay = exp(−Gprev)`, which spans ~exp(80) of dynamic range,
lose too many bits in TF32's ~10-bit mantissa), **not** overflow at the guard. The
warmup steps (no optimizer update) are stable; divergence begins only once weights
start moving. Under the **identical** protocol the prior `torch.complex` path (full
fp32 complex GEMM) was stable — so the only changed variable, matmul precision, is
the cause.

**Stabilizing forces full-fp32 matmuls.** Added env knob `CPLX_ALLOW_TF32=0`
(`ndm/triton/complex_eig_chunked_autograd.py`) forces fp32 `tl.dot` even on the
bf16 path (inputs/states/accumulation were already fp32; only the heavy matmuls
change). fp32 matmuls are stable, but the fused **backward** at the LM shape
(V=64, C=32) **overflows shared memory** (requires 136 KB > 99 KB Ada/Ampere
limit; Triton pads any C∈(16,32] to a 32-tile, so the fix is C≤16). With
`cplx_chunk_size=16` the fp32 path trains cleanly.

---

## 3. Throughput (same hardware, steady-state fwd+bwd, LM config T=512 B=16)

| config | tok/s | × baseline | trains? |
|--------|------:|-----------:|:-------:|
| **real-eig-gdn (baseline)** | 184,000 | 1.00× | ✅ |
| **torch.complex (unfused, the prior run's kernel)** | 85,450 | **0.46×** | ✅ stable |
| fused **TF32**, C=16 | 155,550 | 0.85× | ❌ NaN |
| fused **TF32**, C=32 | 137,543 | 0.72× | ❌ NaN |
| **fused fp32, C=16 (the only trainable fused config)** | 38,700 | **0.21×** | ✅ stable |
| fused fp32, C=32 | — | OOM SMEM | — |

**Isolation:** at the *same* C=16, TF32 (155 k) vs fp32 (38.7 k) is a **4.0× gap** —
the slowdown is **purely fp32-vs-TF32 tensor-core**, not chunk size. The fused
kernel's entire speed advantage is the TF32 tensor-core path; in fp32 it has no
tensor-core benefit and is **2.2× slower than the cuBLAS `torch.complex` GEMM** it
replaces. (The 0.46× `torch.complex` number reproduces the prior run's ~85 k tok/s
on the prior hardware.)

This is the crux of the wall-clock story: **fast ⇒ TF32 ⇒ cannot train; trainable
⇒ fp32 ⇒ slowest option.** The microbench "3× penalty eliminated / GDN-2-class
throughput" claim from `COMPLEX_EIG_TRITON_RESULTS.md` was a TF32 forward+backward
measurement on random data that never enters the training-instability regime; it
does **not** transfer to a real trainable LM.

---

## 4. Results

### 4a. Matched **tokens** (12,000 steps = 98.3M tokens, identical seed-42 stream)

| arm | params | held-out bpb (y-iterate, 400 batches) | CE last-100 | tok/s | wall (12k steps, post-warmup) |
|-----|-------:|--------------------------------------:|------------:|------:|------------------------------:|
| real-eig-gdn (baseline) | 19,295,200 | **2.0857** | 1.3488 | 184 k | **9.2 min** |
| complex-everywhere (fused fp32-C16) | 19,103,200 | **2.1356** | 1.3758 | 38.0 k | **42.8 min** |

Δ(complex − baseline) = **+0.050 bpb** (complex marginally *worse*) on the same
98.3M-token stream at matched params — the predicted **near-tie on loss**. Complex
does not lower the LM loss ceiling. **Convergent-loss-null re-confirmed** (the
y-iterate baseline 2.0857 reproduces the prior run's 2.0955; complex 2.1356
reproduces the prior 2.1300 — as expected, since fused-fp32 is numerically the same
fp32 complex scan as `torch.complex`; the train-CE gap of +0.027 nats ≈ the held
+0.050 bpb gap, no generalization surprise).

### 4b. Matched **wall-clock** (9.2 min post-warmup, the baseline's 12k-step budget)

| arm | wall | steps | tokens seen | held-out bpb (y-iterate) |
|-----|-----:|------:|------------:|-------------------------:|
| real-eig-gdn (baseline) | 9.2 min | 12,000 | **98.3M** | **2.0857** |
| complex-everywhere (fused fp32-C16) | 9.2 min | 2,605 | **21.3M** (0.22×) | **2.3877** |

Δ(complex − baseline) = **+0.302 bpb** (complex worse). At equal wall-clock the
baseline trains **4.6× more tokens** (because the trainable fused config is 0.21×
throughput) and wins decisively. This is *worse* than the prior `torch.complex`
matched-wall-clock gap would be on this hardware (torch.complex is 0.46×, i.e. it
would see ~2.2× more tokens than fused-fp32 in the same budget).

---

## 5. Verdict — vs the convergent-loss-null

**Prediction (Erik's bet):** TIE on bpb (capability, not loss, is the
differentiator).

- **Matched tokens → TIE (a hair worse), re-confirmed on the fused kernel.**
  complex 2.1356 vs baseline 2.0857, Δ +0.050 bpb. The complex-eigenvalue structure
  does not lower the LM loss ceiling. **This is the predicted convergent-loss-null,
  now established with the fused Triton recurrence (not the torch-complex path).**

- **Matched wall-clock → LOSS, and the redo's premise is REFUTED.** The task
  expected that switching to the fused kernel would make matched-wall-clock *fair*
  (no 3× torch penalty). It does not. The fused kernel is fast only in TF32, and
  **TF32 cannot train this head** (NaN < 30 steps, any decay guard). The only
  **trainable** fused config is fp32 matmuls at C≤16 — **0.21× baseline, slower
  than the `torch.complex` path it replaces (0.46×)**. So at honest, trainable
  matched-wall-clock the complex head loses by +0.30 bpb, and the loss is now driven
  by the **fp32-vs-TF32 tensor-core gap forced by training instability**, a
  *different and more fundamental* mechanism than the prior run's Python-complex
  overhead.

**Bottom line.** The complex-eigenvalue head buys **no LM-bpb advantage at matched
compute** — it ties on loss with identical data (the null), and on the fused kernel
it loses matched-wall-clock *more* badly than before, because its competitive
throughput exists only in a TF32 regime that diverges in training. Any case for the
head must rest on **targeted capability** (the complex-eig capability / positional &
modular probes), **not** on LM loss or wall-clock. The convergent-loss bet is
upheld; the "fused makes it wall-clock-fair" hope is not.

---

## 6. Honest accounting & caveats

- **The throughput penalty is now a precision/stability wall, not a Python penalty.**
  Prior run: `torch.complex` ran ~2–3× slow because of Python-complex dispatch. This
  run: the fused kernel removes that, but its tensor-core speed requires TF32, which
  is too imprecise for this head's training dynamics (NaN). The stable fp32 fused
  path has no tensor-core advantage and is the slowest trainable option. Both
  accountings (token + wall) are reported above with full token/time numbers.
- **Why fp32-C16 ≈ torch.complex on loss.** Both compute the *same* fp32 complex
  chunked scan; fused-fp32 is a faithful, autograd-correct, lower-throughput
  re-implementation. Hence the matched-token bpb reproduces the prior run.
- **C=32 fp32 OOM is a real ceiling.** The fused backward holds many doubled
  (real+imag) `[C,·]` fp32 tiles; at V=64 the C=32 tile set is 136 KB > the 99 KB
  Ada/Ampere shared-memory limit. C=16 is the largest fp32-trainable chunk at this
  head dim. (TF32 stores tile operands more compactly and fits C=32 — but, again,
  does not train.)
- **Could a smaller lr let TF32 train?** Possibly, but that would break the
  matched single-variable protocol (the baseline and the prior run both use lr 2e-3
  and are stable). Within the controlled comparison, TF32 is simply not a trainable
  configuration here.
- **Scale.** 19M params, ~10²M-token streaming screen (no repeats) — the standard
  repo near-convergence methodology (cf. e97-within-layer / E99 controls), the
  affordable evidence at this size. The convergent-loss bet concerns the loss
  *ceiling*, which a screen of this size addresses.
- **bpb is a loss metric, not a capability metric.** A tie on bpb is *consistent
  with* "capability is the differentiator, not loss." Capability separation, if any,
  lives in the targeted probes, not LM bpb.

---

## 7. Artifacts

- `ndm/triton/complex_eig_chunked_autograd.py` — fused kernel; added `CPLX_ALLOW_TF32`
  env override (force fp32 `tl.dot` for stable training).
- `scripts/bench_ceig_lm_fused.py` — hot-path-confirmation + throughput benchmark
  (the call-counter proof that the fused kernel is on the hot path).
- `scripts/run_ceig_lm_fused.sh` — matched-compute LM driver (this worktree, fused
  default; `--timer_after_compile_warmup` so the wall-clock budget excludes compile).
- `scripts/eval_ceig_bpb.py` — y-iterate held-out bpb eval (unchanged).
- Run dirs under `/mnt/nvme2n1/erikg/ceig_lm2_fused/` (logs + checkpoints):
  `baseline_tok12k`, `complex_tok12k_fp32c16`, `complex_wall9m_fp32c16`, plus the
  stability/throughput probes (`complex_stab_fp32_c16`, `complex_tf32_g{20,30}`,
  `complex_tf32_c16_probe`, `complex_torchcplx_probe`).
