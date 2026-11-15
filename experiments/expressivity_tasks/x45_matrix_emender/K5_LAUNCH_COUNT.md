# K5 — the launch-count stage: taking the registered step toward its device floor

The diagnosis this stage works from is `PROFILE_STEP_VS_GDN2.md` (same-session,
receipt `receipts/profile_step_vs_gdn2.json`): the registered x45 step at mb8 is
**9.29 s wall against a 3.32 s device floor**, because it issues **434,357
kernels/step** (GDN-2: 59,294) whose device work sums to 3.32 s — the host needs
5.97 s to issue them, so the GPU idles 64 % of the step.  The plan: three ranked
fixes, one receipted gate at a time.  **Outcome, stated up front:** gates B and
C removed the receipted blocker and the memory wall — the fused arm runs at mb8
at **parity with its registered twin** (the scale proof's open question) with
**2.1× fewer kernel launches** and the device floor reached; the 32 k tok/s
target is **not** reached, because the launch storm the profile attributed to
routing's decision machinery actually lives in routing's *differentiable*
channels (gate A's measurement corrects the profile's attribution), and that
fusion — approved, K23-precedent — is the named next item, not a mystery.

| gate | fix | state |
|---|---|---|
| A | routing fusion — the decision chain as one Triton kernel, bit-identical decisions | **landed** (below) |
| B | K4 compile hosting — the fused scan as a `torch.library` op under `torch.compile` | **landed** (below) |
| C | trajectory recompute-over-store — mb8 for the fused path | **landed** (below) |
| D | the step metric — honest un-profiled walls, kernels/step, distance to 32 k tok/s | **landed** (below) |

Every number below is from this stage's own receipts (session metadata inside
each): `receipts/k5_routing_fusion.json` (gate A),
`receipts/k5_k4_compile_hosting.json` (gate B),
`receipts/k5_recompute_over_store.json` (gate C),
`receipts/k5_step_metric.json` (gate D).

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

## Gate D — the step metric, and the honest distance to the 32 k target

Same boundary as the diagnosis, all three arms in ONE session
(``receipts/k5_step_metric.json``): the registered arm's path unchanged
(``compile_scan=True``, autograd through the compiled chunk arithmetic), the
gate B+C fused arm, and the full K5 stack (fused + gate A's routing kernel).
Steady walls are un-profiled; the device decomposition is one profiled steady
step, attributed kernel-by-kernel.

| per step (mb8, 64 windows, this session) | registered | fused (B+C) | fused + K5 routing |
|---|---:|---:|---:|
| step wall, p50 (un-profiled) | 7.40 s | 7.99 s | 7.28 s |
| kernels/step | **434,366** | **207,987** | 219,493 |
| device busy | 3.33 s | 3.05 s | 3.54 s |
| host gap (wall − busy) | 4.07 s | 4.94 s | 3.74 s |
| tokens/s (steady) | 13,807 | 12,862 | 13,932 |
| peak allocated | 24.31 GB | 21.28 GB | 21.28 GB |

Session honesty: this session's card runs ~20 % faster than the profile session
(the registered arm measures 7.40 s here against 9.29 s there), but the
session-invariant numbers reproduce exactly — 434,366 vs 434,357 kernels/step,
3.33 vs 3.32 s device busy — so the matched pair is read same-session.

**The matched pair, answered.**  The scale proof's open question was whether
the fused path reaches its own registered twin once the integration is fixed.
At mb8, same session, same protocol: **yes, parity** — the three arms' walls
(7.40 / 7.99 / 7.28 s) sit inside the step-to-step spread at n=3, and the fused
path *no longer regresses* where it used to be 4.1× slower and mb8-impossible.
What the fused path definitively buys is the launch count: **434 k → 208 k
kernels/step (−52 %)**, the device floor reached (3.05 s busy), and 3 GB less
peak memory.  What it does not yet buy is wall time: the step is still
host-gap-bound.

**The honest distance to 32 k tok/s: not reached, and the residual is named.**
Best steady arm: 13,932 tok/s against the ~32 k device floor.  The device busy
is already at the floor class (3.05–3.54 s; at zero host gap the fused arm would
run ~35 k tok/s, *above* the target).  The residual is the **host gap, 3.7 s**,
and the receipt attributes it: the routing region's **differentiable channels**
(read-weights forward+backward, payload lanes, similarity preprocessing — ~40
compiled kernels per chunk, 6,144 chunks/step) plus the compiled backbone's own
dispatch.  Gate A's discrete-chain fusion moved ~0 of it (measured, gate A) and
its kernel as built *costs* 0.50 s of device time per step (6,144 launches at
~82 µs — one program per (batch, head) row, a latency-bound 64-token serial
walk), which is why ``fused_routing`` stays default-off: the A-phase extension
approved by the supervisor — one Triton forward/backward pair for the read
weights and payloads, K23-precedent floor discipline on the weights, decisions
and payloads still bit-exact — is the item that owns the remaining 3.7 s, and
the decision kernel should be folded into it (or parallelised over token
blocks) rather than shipped alone.

**Follow-up (K5A, landed):** that extension is built and gated in
`K5A_ROUTING_DIFF.md` -- decisions bit-exact by construction with **zero flips
in 8.65 M lanes**, the step's launch count down to **125,033 kernels/step**
(-71 % vs registered), gate A's 0.50 s decision kernel gone -- and the honest
step verdict: the wall does **not** improve, because the step is no longer
launch-bound; the residual host gap is owned by per-chunk Python op dispatch
(the custom-op boundary K4 and K5A cross four times per chunk), not by
launches.  The 32 k target is not reached and the owning component is named
there.

## The kernels/step table (final)

| per step (mb8, 64 windows) | profile session | this session |
|---|---:|---:|
| registered arm: kernels/step | 434,357 | 434,366 |
| registered arm: device busy | 3.32 s | 3.33 s |
| registered arm: step wall | 9.29 s | 7.40 s |
| fused arm (B+C): kernels/step | — | **207,987** |
| fused arm (B+C): device busy | — | **3.05 s** |
| fused arm (B+C): step wall | — | 7.99 s (8.26 s in gate C's session) |
| fused mb2 arm (the old 122.2 s / 881 tok/s) | 6,349,058 kernels | **832,558 kernels, 28.5 s/step** (gate B) |
| routing region kernels/chunk (compiled) | 23 (profile's differential) | 41.0 → 39.9 (gate A, this fixture) |
| route_chunk, eager fwd kernels/chunk | 243 (incl. backward) | 136 → **65** (gate A) |
