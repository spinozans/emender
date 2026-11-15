# PROFILE_STEP_VS_GDN2 — same-session step decomposition, x45 vs GDN-2

**Diagnosis only.** Nothing in the model, kernels or schedules was changed. Every
number below is from one same-session run on one card (RTX 6000 Ada, 47.38 GiB,
torch 2.9.1+cu128, triton 3.5.1, `NDM_PIN_TRITON_AUTOTUNE=1`), receipt:
[`receipts/profile_step_vs_gdn2.json`](receipts/profile_step_vs_gdn2.json),
script: [`profile_step_vs_gdn2.py`](profile_step_vs_gdn2.py).

Measurement boundary, every table unless stated otherwise: **one whole model
training step** — prepare + all microbatches' forward and backward + clip/AdamW,
flagship-stream read excluded, 64 windows × 2048 bytes, mb8, bf16 autocast, fp32
master weights, fused CE — the gates.py / cd_* registered regime. Steady walls
are the median of un-profiled steady steps; device decomposition is one
additional steady step under `torch.profiler` whose chrome trace is attributed
kernel-by-kernel (correlation → cuda_runtime → enclosing op chain). RECORDED
receipt numbers are labelled RECORDED; both models were re-measured fresh here.

## 1. Headline: the same-session comparison

| per step (mb8, 64 windows) | x45 registered | GDN-2 | ratio |
|---|---|---|---|
| wall, p50                    | **9.29 s**  | **2.91 s** | **3.19×** |
| tok/s (steady, un-profiled)  | 10,863      | 34,855     | 0.31× |
| device busy                  | 3.32 s      | 2.61 s     | 1.27× |
| **host gap (wall − busy)**    | **5.97 s (64 %)** | 0.30 s (10 %) | 20× |
| kernel launches per step      | **434,357** | 59,294     | 7.3× |
| mean kernel duration          | 7.6 µs      | 44.0 µs    | 0.17× |
| launch-to-launch interval     | 21.4 µs     | 49.1 µs    | — |
| peak allocated                | 24.31 GB    | 4.49 GB    | — |

RECORDED references (9,306 vs 28,124 tok/s = 3.02×) were measured with the full
CPU+CUDA profiler active on *every* timed step, which inflates both arms' walls;
the honest same-session ratio is 3.19–3.21×. The recorded comparison and this one
agree on the ratio; the recorded absolute walls carry ~30–40 % profiler overhead
(e.g. the registered arm's 13.17 s/step recorded vs 9.29 s/step here).

**The step is not scan-arithmetic-bound. It is launch-bound.** The x45 step
issues 434k kernels whose *device* work sums to 3.32 s; the host needs 5.97 s to
issue them, so the GPU sits idle 64 % of the step. GDN-2 issues 59k kernels that
keep the GPU busy 90 % of the time. Fusing the block scan to a ~1 ms kernel
attacked a component that was never the wall: the whole scan region is ~1.7 s of
device time per step, and the wall is host time.

## 2. Component decomposition (device ms per step, from the profiled step)

| component (device ms/step) | x45 registered | GDN-2 | Δ |
|---|---:|---:|---:|
| (a) scan kernels — triangular solve (`linalg_solve_triangular`, 24,576 trsm launches, fp32) | 604.9 | — | +604.9 |
| (a) scan kernels — compiled-graph fused triton (cumsum/exp/scatter chains) | 1,063.6 | — | +1,063.6 |
| (a)+(b) compiled-graph matmul (routing similarity + scan reads; inseparable by name) | 200.1 | — | +200.1 |
| (a) GDN-2 mixer kernels (`chunk_gdn2_*`, `chunk_kda`, `causal_conv`, `l2norm`) | — | 540.2 | −540.2 |
| (b) routing/schedule — GDN-2 | — | 0 (none exists) | 0 |
| (c) embedding | 0.9 | 1.0 | ~0 |
| (c) head/CE (fused) | 0.7 | 0.6 | ~0 |
| (c) matmul backbone (MLP + mixer projections) | 801.8 | 955.8 | −154.0 |
| norms (RMSNorm) | 122.7 | 120.4 | +2.3 |
| (d) optimizer (clip + fused AdamW) | 6.7 | 7.6 | −0.9 |
| other elementwise (residual/silu/copy) | 515.9 | 981.1 | −465.2 |
| **device busy total** | **3,317.6** | **2,607.5** | **+710.1** |
| (e) prepare (host grouping) | 3 ms | 3 ms | ~0 |
| (f) host gap | **5,970** | **300** | **+5,670** |

Notes: x45's mixer device total (graph triton + graph matmul + solve) is 1.87
s/step; GDN-2's whole mixer is 0.54 s/step. x45's *backbone* is cheaper than
GDN-2's (its SwiGLU is wide but its mixer projections are narrow: −154 ms matmul,
−465 ms elementwise) — the backbone difference *funds* part of the mixer's device
excess.

## 3. Routing vs scan: the same-session differential microbench

Routing lives **inside** the same `CompiledFxGraph` as the scan arithmetic
(`torch.compile` wraps the whole `routed_chunk_scan`, router included), so kernel
names cannot split them. The split below is measured differentially at the exact
registered shape (b8/T2048/chunk64/rank128, a real captured block-0 activation,
the arm's own recorded schedule): routing = compiled `routed_chunk_scan`
(fwd+bwd) − compiled scan-only over the same recorded schedule (fwd+bwd).
There are 192 mixer scans per step (24 layers × 8 microbatches).

| per scan (fwd+bwd) | host dispatch | device busy | kernels |
|---|---:|---:|---:|
| compiled routed scan (routing + scan) | 55.6 ms | 9.9 ms | 2,171 |
| compiled scan-only                   | 27.2 ms | 9.2 ms | 1,447 |
| **→ routing, compiled**              | **28.5 ms** | **0.7 ms** | **724** |
| eager routed scan                    | 200.7 ms | 30.3 ms | 13,849 |
| eager scan-only                      | 115.9 ms | 18.6 ms | 6,065 |
| **→ routing, eager**                 | **84.8 ms** | **11.7 ms** | **7,784** |

Cross-checks (the microbench and the model trace agree):

* compiled scan kernels/scan 2,171 × 192 + projections ≈ the trace's 434,357
  launches/step (2,262/scan measured in-step);
* compiled scan device 9.9 ms × 192 = 1.90 s/step ≈ the trace's mixer device
  total 1.87 s/step;
* scan-side device is 9.2 ms/scan (≈1.77 s/step) and routing-side device is only
  0.7 ms/scan (≈0.13 s/step) — **routing is not a device problem**;
* per chunk (32 chunks/scan): routing ≈ 23 kernels, scan ≈ 45 kernels, compiled;
eager routing is **243 kernels per chunk**.

## 4. Gap attribution: where the 3.19× goes

Per-step wall difference: 9.29 − 2.91 = **6.38 s**. Attribution:

| component of the gap | s/step | share of Δ |
|---|---:|---:|
| (f) host/launch gap — **routing region** (compiled: 28.5 ms × 192, scaled to the in-model host) | ≈ 2.9 | **≈ 45 %** |
| (f) host/launch gap — **scan region** (compiled: 27.2 ms × 192, same scaling) | ≈ 2.8 | **≈ 44 %** |
| (f) host gap — backbone/norms/CE/opt | ≈ 0.3 (same as GDN-2's whole gap) | ≈ 0 |
| (a) device — mixer (scan 1.74 s + routing 0.13 s vs GDN-2's 0.54 s) | +1.33 | +21 % |
| (c) device — x45's cheaper backbone (matmul −0.15, elementwise −0.47) | −0.62 | −10 % |
| (c,d,e) embedding/CE/opt/prepare | ~0 | ~0 |
| **total** | **6.38** | **100 %** |

(The microbench dispatch terms sum to 10.68 s/step because each standalone scan
call carries its own sync/clone overhead; the in-model host time is 5.97 s, so
both terms are scaled by the same factor and only their 51:49 ratio is used.)

**Verdict on the hypothesis.** "Chunked host routing, 24 layers × windows" is
*half* right for the registered arm: routing is ~45 % of the total gap — but it
is not routing *Python* (it is compiled away — `route_chunk` does not even appear
as a Python frame in the registered arm's cProfile) and it is not routing
*arithmetic* (0.13 s/step of device time). It is
routing's **launch count** (≈23 compiled kernels per chunk, 6,144 chunks/step),
and the scan region is an equal-sized co-defendant (≈45 kernels per chunk,
including 24,576 fp32 triangular-solve launches). In the *eager* regime (the
fused path) routing Python is exactly the killer — see §5.

## 5. The fused-ON mb2 arm (881 tok/s): what the extra slowdown is

Same-session pair at mb2 (32 microbatches/step, 768 scans/step):

| per step (mb2) | x45 off (compiled, fused OFF) | x45 on (fused, compile OFF) | ratio |
|---|---:|---:|---:|
| wall, p50            | 29.75 s | **122.2 s** | **4.11×** |
| tok/s steady        | 3,579 | 871 | — |
| device busy          | 6.28 s | 15.05 s | — |
| host gap            | 23.5 s (79 %) | **107.2 s (88 %)** | — |
| kernel launches      | 1,590,614 | **6,349,058** | 4.0× |
| peak allocated       | 8.11 GB | 32.96 GB | — |

(RECORDED pair under per-step profiling: 132.8 vs 50.1 s = 2.65×; without
profiler overhead the honest regression is **4.1×**, not 2.4×.)

**Cause 1 — the fused path ships with no torch.compile anywhere.**
`compile_scan=False` in the fused config, so `route_chunk` runs eager: 243
kernels per chunk (vs ≈23 compiled), 7,784 kernels per scan — and 768 scans/step
gives **5.98 M routing launches ≈ 94 % of the arm's 6.35 M launches**. The
microbench's eager routing dispatch (84.8 ms/scan × 768 = 65 s) matches the
cProfile of one fused step directly:

* step under cProfile (130.5 s traced): `route_chunk` — 24,576 calls, **cum
  62.6 s (48 % of the step)**, of which 25.2 s is pure Python `tottime`;
  `read_weights` 14.1 s; backward total 50.8 s.

**Cause 2 — "compile the routing around the fused node" is not available on this
stack.** Attempted and recorded (`x45_on_mb2_compile_route_failed` in the
receipt): with `fused_backward=True` *and* `compile_scan=True`, dynamo does not
graph-break at the K4 custom autograd node — it traces into it and inductor
refuses to lower a `bool→int8` view inside the traced region:
`InductorError: LoweringException: torch.bool is not supported by torch.iinfo`,
target `aten.view.dtype`. So the model code's "both configurations are
measurable" premise fails on torch 2.9.1+cu128; the fused arm cannot borrow the
compiled routing.

**Not the cause — trajectory write traffic.** The K4 persistent kernels are
2.77 s of the 122 s wall (2.3 %): `_persistent_scan_trajectory_kernel` 1.09 s +
`_persistent_bwd_kernel` 1.68 s. The trajectory's real cost is *memory*, not
bandwidth: 33 GB peak forces microbatch 2 (mb8 and mb4 OOM, RECORDED), which
quadruples the scans per step (768 vs 192) and multiplies the launch storm.

**Routing unchanged?** Yes: `fused_backward` swaps only the chunk's
forward+backward (`trainable_block_scan`); `route_chunk` still runs per chunk —
24,576 calls/step = 768 scans × 32 chunks, confirmed in the cProfile, and the
receipts' grad probes stay all-attached/non-zero.

So the "fused scan at 1 ms" optimized the wrong term twice over: the scan region
it replaces was ~0.29 ms of device work per chunk, while the step was
launch-bound — and by forcing compile off it *multiplied* the launch count 4×.

## 6. Ranked bottlenecks (evidence per row)

1. **Host launch overhead of the per-chunk mixer pipeline — 5.97 s/step (64 % of
   the wall; 89 % of the GDN-2 gap).** 434,357 launches/step at 21.4 µs
   launch-to-launch with 7.6 µs mean kernel; dispatch≈latency per scan (55.6 ms
   host vs 9.9 ms device). Split ≈ 51 % routing region / 49 % scan region
   (differential, §3). This is the whole game: at zero host gap the registered
   step would run at its 3.32 s device floor ≈ **32 k tok/s**, already at parity
   with GDN-2's 34.9 k.
2. **Scan-region device arithmetic — 1.77 s/step.** Even with a perfect launch
   fix, the fp32 per-chunk triangular solve (605 ms; 24,576 trsm launches) plus
   the reads/index arithmetic (1.13 s) is 3.3× GDN-2's entire mixer device time
   (540 ms). This is the second ceiling.
3. **Routing eager / the fused arm's configuration — 107 s/step of host at mb2.**
   5.98 M of the 6.35 M launches are eager routing; `route_chunk` cum 62.6 s of
   the 130.5 s traced step. And the escape hatch (compile the routing around the
   K4 node) raises `InductorError` on this stack (recorded).
4. **Routing device cost — not a bottleneck.** 0.13 s/step compiled (0.7
   ms/scan). No work here is justified on device grounds.
5. **Not bottlenecks (measured, for the record):** optimizer 6.7 ms/step; data
   prepare 3 ms/step; embedding + fused CE ≈ 1.6 ms/step; norms 123 ms/step;
   the x45 backbone is *cheaper* than GDN-2's (−0.62 s device/step).

## 7. What the numbers say the next stage is

(Diagnosis only — nothing was changed.) The step is a launch-rate problem, not
an arithmetic problem: the next optimization stage should target kernel count
per chunk in *both* the routing and the scan region — a fused routing kernel
(23 compiled kernels per chunk → 1–2) and a single-kernel scan path that fuses
the routing arithmetic in the same launch (the K1+K2+K3 lane already points
there, but it must stay *compilable* or itself absorb the routing), plus batched
triangular solves across chunks. Conversely, the K4 fused node as shipped is
anti-productive on this stack (compile-impossible, 4× the launches at the forced
microbatch) and should not be the vehicle until the compile break is fixed
(e.g. `torch.compiler.disable` around the node — to be evaluated, not done here).
Device-side, parity with GDN-2 additionally wants the scan arithmetic in fused
kernels (1.77 s → ~0.5 s class), but that is the *second* 1.3 s, not the first
5.7 s.
