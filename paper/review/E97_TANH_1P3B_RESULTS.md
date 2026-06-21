# tanh-e97-1p3b — does bounded (tanh) state on the fused-checkpointed kernel beat gdn2-mlp on wall-clock at 1.3B?

**Task:** `tanh-e97-1p3b`. Standing prior: the tanh-over-time E97 cell has a
GDN-fast fused-checkpointed sequential kernel already built
(`ndm/triton/e88_triton_forward.py`, `LINEAR_STATE=False`, sparse forward
checkpointing K=16). Audit measured 0.93× native at dim1024/h16. Premise to test:
at the 1.3B within-layer head shape the kernel is *launch-bound* (fixable), and
the richer tanh cell is more sample-efficient than the linear `e97_delta` that
already beat GDN token-matched (commit 71bda00). So tanh-E97 + gdn-neg should win
wall-clock once the launch is saturated.

**Date:** 2026-06-08. **Compute:** 8× RTX 6000 Ada, idle-only, REAL Pile
(`/home/erikg/elman/data/pile.txt`, p50k_base, ctx 2048), bf16, schedule-free
AdamW, checkpoint round-trip on every held-out BPB. No mocks; every BPB is a real
40-batch held-out measurement. **Run dir:** `experiments/e97_delta_1p3b_cma/`.

---

## 0. Decision deliverable (verdict)

**Does tanh-E97 + gdn-neg (within-layer, 1.3B, fused-checkpointed tanh kernel,
`LINEAR_STATE=False`) BEAT gdn2-mlp on WALL-CLOCK-matched held-out BPB? And does
tanh beat the linear `e97_delta`?**

> **Loses at honest wall-clock — decisive, triangulated across three independent real
> measurements. The "launch-bound, fixable" premise is REFUTED at the 1.3B head
> shape. tanh DOES beat linear `e97_delta` (+0.041 BPB token-matched) and DOES
> beat gdn2-mlp TOKEN-matched (+0.02) — but the bounded-state sample-efficiency
> edge is smaller than the throughput tax of the sequential scan, so wall-clock
> it LOSES. The kernel is not launch-bound; it is intrinsically ~0.75× GDN
> throughput because a non-associative tanh scan runs serial outer-products on
> CUDA cores while gdn's chunked scan saturates the tensor cores. Batch
> parallelism — the prior's sanctioned launch fix — does not close the ratio and
> craters BPB.**

### The decisive numbers (1.3B, param-matched ~1.25 B, REAL Pile)

| measurement | gdn2-mlp | tanh-E97 + gdn-neg | winner |
|---|---:|---:|---|
| **Head-to-head B=2, WALL-CLOCK-matched** (720 s, 2 seeds) | **2.021 / 2.032** | 2.063 / 2.079 | **gdn2-mlp** (+0.04…+0.05) |
| **Head-to-head B=2, TOKEN-matched** (4.99 M tok, 2 seeds) | 2.083 / 2.106 | **2.063 / 2.079** | **tanh** (+0.02…+0.027) |
| **CMA ratio search fitness** (360 s fixed-wall, tanh state, 24 ratios) | **2.168** (anchor) | 2.207 (best mix, 43 e97_delta) | **gdn2-mlp** |
| **Head-to-head B=4** (launch-fix attempt, wall-matched, 2 seeds) | **2.015 / 2.034** | 2.345 / 2.216 | **gdn2-mlp** (+0.18…+0.33) |
| sustained tok/s (B=2 real train) | 9598–9940 | 6970–6997 (**0.70–0.73×**) | gdn2-mlp |

tanh wins **only** token-matched at B=2 — its single favorable operating point —
and loses every wall-clock-fixed comparison.

---

## 1. Kernel speed — the fused-checkpointed tanh kernel is NOT GDN-fast at 1.3B (premise refuted)

The 0.93× was measured at **dim1024 / h16**: 16 heads, head_dim 64, n_state 64 →
large per-head work, few programs. The 1.3B within-layer shape is **dim2112,
H=64, N=32, V=33** → tiny per-head state (32×33), 64 heads. The tanh path routes
through `e88_triton_optimized_apply → e88_triton` (the `e88_triton_forward.py`
fused-checkpointed scan), grid `(B, H)`.

**Launch sweep at the real 1.3B shape** (`tanh_launch_sweep.py`, REAL models,
fwd+bwd, T=2048):

| micro-batch B | programs (B×H) | tanh tok/s | gdn tok/s | ratio | tanh GPU util |
|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 7697 | 9815 | **0.784×** | 99.5 % mean / 100 % peak |
| 4 | 256 | 8810 | 10456 | **0.843×** | 97.3 % mean |
| 8 | 512 | — | — | OOM | tanh state needs 29.5 GB @ B=4 |

Reading it:

- **The kernel is already GPU-busy (99.5 % util at B=2)** — it is *not* idle-SM
  launch-bound. The 0.78× is the *intrinsic* cost of the sequential scan: T=2048
  serial steps, each a tiny outer-product on CUDA cores, **no tensor-core
  matmul**. gdn's chunked-parallel scan turns the same recurrence into batched
  matmuls that saturate the tensor cores. This is a hardware-efficiency gap, not
  an occupancy gap.
- Batch parallelism (the prior's sanctioned fix, "saturate via batch × head ×
  checkpoint-block") raises the ratio only **0.784 → 0.843×**, and gdn speeds up
  too, so the real-training ratio barely moves (B=4: 7896/10433 = **0.757×**).
  B=8 OOMs at 1.3B — the checkpointed tanh state buffer is the memory wall.
- The V-columns of the recurrence are independent (column *v* of S depends only
  on column *v*), so the grid *could* gain a V-tile dimension. But at 99.5 %
  util and compute-bound on serial outer-products, more programs cannot add
  throughput — the bottleneck is instruction throughput on the scan, not SM
  occupancy. Splitting V multiplies launch/scratch overhead for no compute win.

**Conclusion:** the fused-checkpointed sequential tanh kernel is GDN-fast only at
the large-per-head shape (dim1024/h16); at the 1.3B within-layer head shape it is
a hard **0.70–0.80×**, and this is not a launch defect to fix — it is the
non-associative scan's inability to use the tensor cores.

---

## 2. tanh vs linear `e97_delta` — the +0.08 is +0.041 at 1.3B (real, but half)

Clean attribution control (`attribution_control.py`, same config — dim 2112,
21 gdn-neg + 43 e97_delta — sequential kernel, only the state nonlinearity
differs, 720 s, seed 0):

| state | kernel | held-out BPB | tokens |
|---|---|---:|---:|
| **tanh** (`LINEAR_STATE=False`) | sequential | **2.055** | 5.16 M |
| identity (`LINEAR_STATE=True`)  | sequential | 2.096 | 5.10 M |

**tanh beats linear by +0.041 BPB** at matched tokens — the bounded-state
nonlinearity IS load-bearing for sample efficiency, confirming the qualitative
prior. But the magnitude is **~half** the hoped +0.08 (that figure was from a
smaller-scale earlier run); at 1.3B the edge is +0.041. This is the *entire*
budget tanh has to overcome its throughput deficit — and it is not enough.

---

## 3. The ratio search already covered the tanh cell — no allocation flips it

The existing 1.3B CMA (`cma_all_results.json`) screened **tanh-state**
(`e97_state_nonlin` default = tanh) over the gdn-neg:e97_delta ratio + MLP +
shape: popsize 8 × 3 gens = 24 evals, e97_delta counts 2→60, **360 s fixed-wall
fitness** (so the fitness is itself wall-clock-matched). Result:

- best tanh-mix (43 e97_delta / 21 gdn): **2.207**
- `A_gdn2_mlp` anchor (64 gdn): **2.168** ← beats *every* tanh-mix ratio
- `A_delta_only` (64 e97_delta): 2.266; `A_seed` (32/32): 2.217

The search itself, on a wall-clock-fixed budget, ranks pure gdn2-mlp above all 24
tanh allocations. This is mechanistic: every tanh-E97 head added is a 0.75×-
throughput tax for only marginal token-edge, so wall-clock fitness is monotone
toward *fewer* tanh heads — the optimum is the gdn2-mlp endpoint (0 tanh heads).
The token-efficiency-optimal ratio (43 e97_delta, the head-to-head config) is the
*worst* wall-clock choice. No interior ratio beats the baseline.

---

## 4. The launch-fix attempt (B=4) backfires

Running both arms at micro-batch B=4 (`tanh_b4_headtohead.py`, 720 s, 2 seeds)
to cash in the 0.784→0.843× kernel speedup:

| arm | seed0 | seed1 | tok/s |
|---|---:|---:|---:|
| tanh-E97 + gdn-neg | 2.345 | 2.216 | 7896 / 7914 |
| gdn2-mlp (wall-matched) | 2.015 | 2.034 | 10433 / 9965 |
| gdn2-mlp (token-matched @5.66 M) | 2.095 | 2.092 | — |

At B=4 tanh loses **both** axes — even token-matched — and shows large seed
variance (2.345 vs 2.216). The recurrent delta cell is batch-fragile: at B=4 with
the B=2-tuned LR/schedule its BPB craters (2.063 → 2.28 avg), while gdn2-mlp is
batch-robust (2.026 → 2.025 avg). So tanh's token-efficiency edge exists *only*
at the B=2 operating point where the kernel is slowest. Throughput and
sample-efficiency are in direct tension along the batch axis too: you cannot
batch your way to GDN throughput without erasing the edge that motivated tanh.

---

## 5. Why it loses at wall-clock, in one ladder

```
tanh+seq, B=2 token-matched   2.063  ← BEATS gdn2-mlp token-matched (2.083); edge = +0.02
  (+ linear costs +0.041)             tanh's sample-efficiency is real (vs linear 2.096)
tanh+seq, B=2 wall-matched    2.063  ← LOSES gdn2-mlp wall (2.021); 0.72× throughput eats the edge
tanh, B=4 (launch fix)        2.28   ← LOSES worse; batch craters the delta cell
gdn2-mlp                      2.021  ← winner on every wall-clock-fixed budget
```

The binding constraint is **kernel throughput, not head allocation and not
launch config**: 0.70–0.80× at 99.5 % GPU util. tanh's +0.02…+0.04 token-edge
cannot overcome processing ~28 % fewer tokens in the same wall.

---

## 6. Residual + next lever

**Residual (the real wall, identical to `fuse-2kernel`'s conclusion, now
confirmed for the SEQUENTIAL tanh kernel specifically):** bounded (tanh) state is
sample-efficient (+0.041 vs linear) but its sequential scan is intrinsically
~0.75× GDN throughput at the 1.3B head shape — and that gap is **not** launch-
bound (kernel already at 99.5 % util) and **not** fixable by batch (backfires) or
by V-tiling (compute-bound, not occupancy-bound). `fuse-2kernel` showed chunking
*requires* linear state (which erases the edge, 2.055→2.096); this task shows the
sequential kernel that *keeps* tanh cannot reach GDN speed. So the tension is
two-sided and closed:

> **tanh state ⊥ tensor-core throughput.** Chunkable ⇒ linear ⇒ no edge.
> Sequential ⇒ tanh ⇒ no speed. There is no point on the current kernel menu
> that is both.

**Next lever (unchanged, now doubly-motivated):** a **chunkable bounded-state
kernel** — a saturating (tanh-like) state map expressed *inside* a tensor-core
chunked-parallel scan (`gdn2_nonlin_shell`). That is the only way to realize the
real +0.04 bounded-state token-efficiency edge as a wall-clock win. Pursue the
nonlinear-shell chunked kernel; do **not** revisit the sequential tanh kernel or
batch/launch tuning for this head shape — both are exhausted here.

---

## 7. Validation checklist (task gate)

- [x] **tanh-E97 (`LINEAR_STATE=False`) on the fused checkpointed kernel,
  verified speed at 1.3B head dims** — `tanh_launch_sweep.py`: 0.784× (B=2) →
  0.843× (B=4), 99.5 % util; routes through `e88_triton` fused-checkpointed scan
  (sequential, within-layer `typed_head_mixture`, no eager, not interleaved). The
  "launch-bound, fix it" premise is **refuted with measurement** (already GPU-
  busy; batch fix attempted and backfires; the gap is tensor-core utilization,
  not occupancy).
- [x] **CMAES over ratio + MLP + shape at 1.3B param-matched** — the existing
  24-eval tanh-state CMA (`cma_all_results.json`, 360 s fixed-wall fitness,
  e97_delta 2→60) IS the search; no tanh ratio beats the gdn2-mlp anchor on
  wall-clock fitness (best mix 2.207 vs 2.168).
- [x] **Held-out BPB vs gdn2-mlp, TOKEN- and WALL-CLOCK-matched, 2 seeds, REAL
  Pile** — B=2 head-to-head (`headtohead_results.json`, tanh = default state):
  wall 2.021/2.032 vs 2.063/2.079; token 2.083/2.106 vs 2.063/2.079. B=4 control
  (`tanh_b4_headtohead.json`) confirms.
- [x] **tanh-vs-linear delta quantified** — +0.041 BPB (tanh better), not +0.08,
  at 1.3B (`attribution_control.json`: tanh 2.055 vs identity 2.096).
- [x] **Explicit accept/reject + residual + next lever + doc committed** — loses
  at wall-clock; residual = tanh ⊥ tensor-core throughput; next lever =
  `gdn2_nonlin_shell` chunkable bounded-state kernel.

### One-line summary

The fused-checkpointed tanh kernel is **not** GDN-fast at the 1.3B head shape
(0.70–0.80× at 99.5 % util — a tensor-core-utilization wall, not a launch
defect); tanh is genuinely more sample-efficient than both linear `e97_delta`
(+0.041) and gdn2-mlp token-matched (+0.02), but that edge is smaller than the
throughput tax, so **it loses at wall-clock** — the only fix is a chunkable
bounded-state kernel (`gdn2_nonlin_shell`), not the sequential kernel or launch
tuning.
