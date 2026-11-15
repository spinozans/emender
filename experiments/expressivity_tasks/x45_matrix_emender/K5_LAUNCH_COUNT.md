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
| B | K4 compile hosting — the fused scan as a `torch.library` op under `torch.compile` | **landed** (below) |
| C | trajectory recompute-over-store — mb8 for the fused path | **landed** (below) |
| D | the step metric — honest un-profiled walls, kernels/step, distance to 32 k tok/s | in flight |

Every number below is from this stage's own receipts (session metadata inside
each): `receipts/k5_routing_fusion.json` for gate A.

## Gate B — the K4 fused scan, compiled: the receipted blocker removed

**The blocker** (`PROFILE_STEP_VS_GDN2.md` §5): `compile_scan=True` +
`fused_backward=True` raised `InductorError` (`torch.bool` is not supported by
`torch.iinfo`, on `aten.view.dtype`) because dynamo traces *into* a custom
`torch.autograd.Function` and inductor cannot lower the kernels' `bool -> int8`
views — forcing the fused arm to `compile_scan=False` and eager routing: 243
kernels/chunk, 5.98 M routing launches per mb2 step, `route_chunk` cum 62.6 s of
a 130.5 s traced step, and the whole arm at **122.2 s/step (881 tok/s)**.

**The fix** (`k4_fused_backward.py`): the block call is re-exposed as a proper
`torch.library` custom op (`x45::fused_block_scan`) with `register_autograd`, so
dynamo emits one opaque call per chunk and the routing compiles *around* it —
and, because `torch.compile` traces a registered backward *into* the joint
graph rather than dispatching it at runtime, the reverse walk is its own opaque
op (`x45::fused_block_scan_bwd`).  The Triton launches stay out of both traces;
the forward and backward arithmetic are byte-for-byte the K4 kernels, which is
why the grad gates re-run unchanged.

**Receipt** (`receipts/k5_k4_compile_hosting.json`, gate B):

* the configuration **compiles**: one dynamo graph for the routed scan with the
  fused path ON (`unique_graphs: 1`, no skipped frames), and the kernel census of
  a compiled forward+backward shows the persistent K4 kernels — one
  `_persistent_scan_trajectory_kernel` and one `_persistent_bwd_kernel` per
  chunk, 32 of each per scan — inside the compiled region, 947 kernels/scan
  fwd+bwd against the eager-scan compiled path's 2,262 (profile's own figure);
* the **K4 grad parity gates re-run and unchanged**: `test_k4_fused_backward.py`
  14 passed, subprocess, summary captured verbatim in the receipt;
* the **honest fused-mb2 step wall** (gates.py protocol, un-profiled steady
  walls): **28.5 s/step p50 (3,699 tok/s)** at 21.4 GB peak, against the recorded
  122.2 s (881 tok/s) fused compile-off baseline — a 4.3× step-wall improvement
  from removing the blocker alone, and the profiled step's kernel count drops
  6.35 M → 0.83 M (7.6×) with device busy 15.05 s → 5.57 s.

At mb2 the hosted fused arm now lands at parity with the compiled eager-scan
arm's own recorded 29.75 s — the fused path is no longer the 4.1× regression it
was; the question the scale proof left open (does the fused arm reach its own
registered twin once the integration is fixed?) is answered *yes at mb2*, and
**mb8 needs gate C** (the 2.15 GB/layer trajectory store still forces
microbatch 2: 21.4 GB peak at mb2 is mostly trajectory).

## Gate C — recompute-over-store: microbatch 8 for the fused path

**The problem**: the store-everything K4 kept every chunk's trajectory alive
until its backward — 67 MB per chunk at the registered shape, 2.15 GB per
layer, **51.5 GB over 24 layers at microbatch 8**, above the card's 47.4 GiB.
That is what forced the fused arm to microbatch 2 and the 4.1x launch
regression the profile diagnosed.

**The fix** (`k4_fused_backward.py`): the forward keeps only the
*chunk-boundary checkpoints* — the chunk-entry state and liveness, the very
tensors the routing already consumed and the op already receives as inputs
(1 MB per chunk against the 67 MB trajectory; 0.75 GB against 48 GB over the
whole model at mb8).  The backward re-walks the chunk forward from its
checkpoint into a one-chunk scratch trajectory — `persistent_scan_trajectory`'s
own kernel, so the recomputed states are bit-identical to the ones a storing
forward would have saved — then runs the same reverse walk.  The autograd
engine runs a scan's chunk nodes strictly in sequence (the dstate chain), so
one scratch is live at a time and the allocator reuses it.

**Receipt** (`receipts/k5_recompute_over_store.json`, gate C):

* **grad parity, the same multiples of the same floors**: every backward output
  bit-identical to the store form's (`bit_equal_to_store: true` on all four);
  against the autograd oracle, `dstate` and `dpayload` bit-exact, `dweights`
  0.43x and `ddecay` 0.65x the re-measured floors — the same class as the K4
  gates, because it is the same adjoint on a bit-identical trajectory;
* **memory**: whole-scan fwd+bwd peak at the registered shape
  **0.34 GB** (the store form's floor for the same scan is 2.0 GB of trajectory
  alone); the whole mb8 step peaks at **21.3 GB — fits the card** (24.31 GB is
  the registered arm's own peak; the fused arm's activation set is a little
  smaller because the scan's autograd graph is 32 opaque nodes, not the eager
  chunk arithmetic's);
* **the recompute cost, paid and stated**: one extra forward walk per chunk in
  backward, **0.095 s of extra device time per step at mb8** (24 layers x 32
  chunks);
* **the mb8 fused step wall** (gates.py protocol, un-profiled steady walls):
  **8.26 s/step p50 = 12,821 tok/s**, 207,976 kernels/step, 3.07 s device busy —
  the configuration that OOMed before this gate, and already faster than the
  registered eager-scan arm's 9.29 s baseline with 2.1x fewer launches.

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
