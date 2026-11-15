# K5 — the launch-count stage: taking the registered step toward its device floor

The diagnosis this stage works from is `PROFILE_STEP_VS_GDN2.md` (same-session,
receipt `receipts/profile_step_vs_gdn2.json`): the registered x45 step at mb8 is
**9.29 s wall against a 3.32 s device floor**, because it issues **434,357
kernels/step** (GDN-2: 59,294) whose device work sums to 3.32 s — the host needs
5.97 s to issue them, so the GPU idles 64 % of the step.  The plan: three ranked
fixes, one receipted gate at a time.

| gate | fix | state |
|---|---|---|
| A | routing fusion — the decision chain as one Triton kernel, bit-identical decisions | **landed** (below) |
| B | K4 compile hosting — the fused scan as a `torch.library` op under `torch.compile` | in flight |
| C | trajectory recompute-over-store — mb8 for the fused path | pending |
| D | the step metric — honest un-profiled walls, kernels/step, distance to 32 k tok/s | pending |

Every number below is from this stage's own receipts (session metadata inside
each): `receipts/k5_routing_fusion.json` for gate A.

## Gate A — routing fusion, landed and honestly measured

**What was built** (`k5_routing_fusion.py`): the whole *discrete* decision chain
of `schedule.route_chunk` — the `-2.0` dead-row fill, the tie-break penalties
(`arange * tie_break`, fp32), best/second/worst selection (torch's first-index
rule on exact ties), the free-slot ordinal scan (the one sequential dependency
along the token axis), and the five-op code selection of both lanes — fused into
ONE Triton kernel per chunk (`_route_decisions_kernel`), one program per
`(batch, head)` row, rank axis in registers.  The kernel consumes the *eager
router's own* `similarity` tensor: the bmm's reduction order belongs to cuBLAS
and a 1-ulp difference in `similarity` flips argmaxes, so the decision inputs
are never recomputed.  Read weights and payloads stay the verbatim torch
expressions of `route_chunk` — they carry gradient, and staying verbatim is what
makes their parity exact by construction.  Thresholds are rounded to the
similarity's dtype exactly as eager's tensor-vs-scalar promotion does it (bf16
under the model's autocast), so the kernel evaluates the same predicates.

**Parity (the gate's demand, no noise-floor allowance):** decisions, payloads and
read weights are **bit-identical to eager `route_chunk`** on

* pinned chunk-entry states — the deterministic instrument, every router output
  bit-exact (rank 8, chunk 32, all rows);
* whole routed scans at five fixtures: five-ops small, POP-saturated,
  ragged 100/64, rank prefix 6-of-16, and the **registered shape** (batch 8,
  2048 tokens, chunk 64, rank 128) under bf16 autocast — plus the growth policy;
* the **cd-protocol training fixture**: the registered model's own block-0 mixer
  on a real flagship-stream activation, all 32 chunks bit-exact, and one whole
  training step per router whose gradient deviation (13.4 on a gradient scale of
  ~30,000) is *below the eager arm's own run-to-run deviation class* (8.1;
  through-`scatter_add_` atomics noise, both arms alike);
* a duplicate-row tie fixture (every direction stored twice): the lower index
  wins, exactly as torch's argmax and the tie-break penalty both demand.

**Compile hosting:** dynamo captures the user-defined Triton kernel as one
opaque `triton_kernel_wrapper` node — the one detail that makes that work is that
the kernel's liveness outputs are allocated **int8 and cast to bool on return**:
a `bool` tensor passed through `view(torch.int8)` is an `aten.view.dtype`
bitcast inductor cannot lower (`torch.iinfo` on `torch.bool`) — the *same*
`InductorError` that blocks the K4 node under `compile_scan=True`, and the fix
pattern gate B reuses.  (A `torch.library` custom op hosts the kernel too, but
its Python dispatch measured ~0.16 ms/chunk under compile — more host time than
the ~2 inductor kernels it replaces — so the captured raw launch is the hosted
form.)  In the compiled regime the fused router's decisions are **bit-identical
to the compiled eager router's** (0 differing lanes in 16,384, bf16 and fp32),
which is the registered arm's own compiled decision semantics, unchanged.

**The honest measurement — the task's premise was wrong, and the receipt corrects
the profile's attribution.**  The profile ranked routing "≈23 kernels/chunk" as
bottleneck 1 and attributed the whole region to the decision machinery.  Measured
at the registered shape (fwd+bwd, per 2048-token scan):

| routing region (compiled, fwd+bwd) | kernels/chunk | host ms/scan |
|---|---:|---:|
| eager `route_chunk` (before) | 41.0 | 28–35 (±20 % run to run) |
| fused router (after) | 39.9 | ~same (interleaved A/B: 32.4 vs 32.1 ms/scan) |
| of which the *discrete decision chain* | **~2–3** | — |

Inductor had already fused the decision chain's argmax/where/cumsum storm to
~2–3 kernels per chunk; the routing region's launches live in the
**differentiable channels** — read weights (softmax·strength) forward+backward
(~20/chunk), payload lanes forward+backward (~8/chunk), and the similarity
preprocessing (norms/divs/bmm, ~8/chunk).  Fusing the discrete chain is
bit-exact and landed, but moves the compiled step metric by ~0.  The remaining
routing fusion (read weights + payloads + their adjoints, one Triton fwd/bwd
pair, with the read weights landing within the lane's established
`NOISE_FACTOR` × measured floor per the `k23_read.py` precedent) is approved
but deferred behind gates B and D (supervisor decision: the scan-region fixes
are worth more — see gate B).

**Where the fusion does pay, measured:** the *eager* routing regime — which is
exactly the fused mb2 arm's configuration today (`compile_scan=False`,
`route_chunk` eager: 243 kernels/chunk incl. backward, cum 62.6 s of a 130.5 s
traced step, `PROFILE_STEP_VS_GDN2.md` §5):

| one chunk of routing, eager (b8/c64/rank128, fwd) | kernels/chunk | host ms/chunk |
|---|---:|---:|
| `route_chunk` | 136 | 2.26 |
| fused router | **65** | **1.36** |

**A pre-existing property, now on the record** (`cross_regime` in the receipt):
the *compiled* registered arm's decisions already differ from the eager router's
at bf16 autocast — 628/16,384 slot lanes and 10 op lanes at the measured fixture
— because inductor lowers the similarity bmm differently from cuBLAS.  This is
not K5's (the fused router matches the compiled eager router bit-exactly, 0
lanes), it predates the stage, and it bounds what "decisions bit-identical to
eager" can mean inside `torch.compile`: identical *within* a regime, by
construction; across regimes, whatever inductor's bf16 bmm does — unchanged by
K5 either way.

## The kernels/step table (updated per gate)

| per step (mb8, 64 windows, registered regime) | baseline | after gate A |
|---|---:|---:|
| step wall, p50 (un-profiled) | 9.29 s | unchanged (not yet re-measured; gate D) |
| kernels/step | 434,357 | unchanged (decision fusion is ~0 in the compiled regime) |
| routing region host ms/scan (compiled) | 28.5 (profile fixture) | ~same |
| routing region kernels/chunk (compiled) | 41.0 → 40.9 | 39.9 |
| route_chunk, eager fwd kernels/chunk | 136 | 65 |

## The honest distance to the 32 k target

To be measured at gate D after gates B and C.  The decomposition this stage will
attribute the residual with: routing's differentiable channels (the known
survivor), the scan region, the backbone, and the launch-independent host floor.
