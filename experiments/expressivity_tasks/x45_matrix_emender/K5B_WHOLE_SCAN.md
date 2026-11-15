# K5B — the routing region at whole-scan granularity: one op per (layer, window), the per-chunk Python removed

K5A ended with the step's owning component named: the fused arms cross the
custom-op boundary once per *chunk* — four op bodies per chunk (K4's forward +
backward pair and K5A's), ~24.6k op dispatches per step — and the op boundary
itself cost more host time (202+179 µs/chunk for the routing pair, ~230 µs of
it the Triton launch binding path) than the 83k compiled launches the fusion
had saved.  The doc's own last line said it: *the next item is the per-chunk
Python, not more fusion*.  This stage removed it.  The whole routing+scan
region — the K5A router and the K4 scan it subsumes, forward and adjoint — is
now **one opaque `torch.library` op pair per (layer, window) scan**, whose
body is the per-chunk sequence walked in one function and, from its second
call, **captured into a CUDA graph and replayed**.  Everything is banked in
`receipts/k5b_whole_scan.json` (the parity gates, the floors, the host
instruments, the determinism/pathology/compile rows) and
`receipts/k5b_step_metric.json` (the matched four-arm step metric, the
dispatch counts, the per-scan host cost before and after).

**Outcome, stated up front.**  The parity gates all hold, and they hold in
their strongest form: the whole-scan path is **bit-identical to the per-chunk
K5A+K4 arm on every channel** — the recorded schedule, slots, op codes,
liveness, payloads, read weights, reads, state, and the three gradients
`dquery`/`dpayload`/`ddecay`, on the eager call, the capture call and every
replay, at the registered shape under bf16 autocast — because its body *is*
that arm's kernel sequence walked in one function.  One whole training step of
a depth-2 model on the whole-scan arm produces the **bit-equal loss and
bit-equal gradients for every one of the model's parameters**.  The mandatory
flip test on the real routed fixture (A-phase's bar, matched) shows **0 flips
in 8,650,752 lanes**.  The host cost the stage targeted is gone: the
routing+scan region's host issue time went from **42.0 ms/scan to 1.0 ms/scan**
(fwd+bwd, the registered mixer shape; 8.05 s → 0.19 s per step over the 192
scans), and the op dispatch count from **24,576/step to 384/step** (192 forward
+ 192 backward, every one a graph replay).  The step wall: **3.25 s** against
the same session's 8.45 s (K5A) and 9.81 s (registered), with the host gap at
**0.024 s** — the wall is device-bound now.  The steady cd-protocol number is
**31,046 tok/s in the matched four-arm session** and **33,300 tok/s in the
same-day single-arm draw** of the identical arm (the 32k line, crossed in the
faster draw, 3% short in the matched session's — the residual is decomposed
below and it is *device*, not host).

| gate | demand | state |
|---|---|---|
| A | decisions/slots/payload codes bit-exact, incl. the mandatory flip test at A-phase's bar (8.65M lanes) | **landed** — bit-equal by construction to the per-chunk arm, 0 flips in 8,650,752 lanes |
| B | read-value floors re-measured first, the walked router within them | **landed** — within `NOISE_FACTOR` in both regimes (fp32: 1.49e-8 vs floor 7.45e-9; bf16: at the 1.14e-3 floor, median 9.3e-9) |
| C | host ms/scan before/after, op dispatches/step before/after | **landed** — 42.0 → 1.0 ms/scan; 24,576 → 384 dispatches/step |
| D | the step metric, matched pair, honest distance to 32k | **landed** — 3.25 s wall, 0.024 s host gap, 31.0k steady this session (33.3k same-day draw) |
| E | determinism, pathology, K4 grad gates unchanged, compile hosting | **landed** — replays bit-identical; refusals by name; gradients bit-equal to the arm; no InductorError |

## What was built (`k5b_whole_scan.py`)

Two `torch.library` ops, the K4/K5A hosting pattern at a new granularity:

* `x45::whole_scan_fwd(query, payload, decay, rank, rank_max, chunk_tokens,
  taus, scale, cast_code, slot_id) -> (reads, state, live, slot_token)` — the
  whole 2048-token window's routing+scan region as ONE opaque call.  Its body
  walks the 32 chunks exactly as the per-chunk arm does: the eager
  preprocessing from detached inputs (the norms, the stored division, the
  similarity bmm — their reduction orders are the decision semantics),
  the K5A prep/emit pair consuming the preprocessing's own
  similarity/strength tensors, and the K4 persistent scan kernel over the
  emitted schedule, with the chunk-entry states kept as the checkpoints the
  adjoint needs (the same bytes the per-chunk arm's autograd graph keeps
  alive).  First call per slot: the eager walk.  Second call: the walk is
  captured into a CUDA graph (static input buffers, the `BlockGraph`
  discipline: eager warmups, side-stream warmups, sync, capture) and
  replayed.  Every later call: three device copies plus one replay.
* `x45::whole_scan_bwd(query, payload, decay, dreads, ...)` — the whole scan's
  adjoint as its own opaque op: per chunk in reverse, the K4
  recompute-adjoint (chunk-entry checkpoint in, `dstate`/`dweights`/
  `ddecay`/`dpayload` out) then the K5A routing adjoint, with the state chain
  carried exactly as autograd carries it in the per-chunk arm
  (`dstate_out(c-1) = dstate(c) + drows(c)`), captured once and replayed.

The wrapper `whole_scan(...)` is the drop-in for `routed_chunk_scan` at the
mixer's boundary (`MatrixMixer(fused_whole_scan=True)`, default off);
`compiled_whole_scan()` compiles it the way `compiled_scan()` compiles the
registered scan — one dynamo graph hosting the op pair as opaque calls, with
`dynamic=True` so the 24 mixers' slot ids share one compiled graph (a static
compile would specialize on `slot_id` and hit the recompile limit).

**The dispatch ledger** (`DISPATCH`) counts every impl entry, replay, eager
walk, capture and refusal — the receipts read it around whole steps, so the
op-dispatch numbers are measured, not derived.

## Why CUDA graphs and not a persistent whole-window kernel

The K5A doc offered two variants; the parity ruling decides between them.
Routing for chunk `c` consumes the chunk-*entry* state — the state carry is
the only sequential axis, so a whole-scan op must walk the chunks — and the
walk needs the decision-input preprocessing at every chunk, which the settled
ruling (K5A's probe receipts: a Triton reproduction of the bf16 norm/bmm chain
matched only ~48% of similarity entries, the payload norm 0/1024) keeps in
torch.  A persistent Triton kernel cannot call cuBLAS, so the "one kernel per
scan" variant is blocked by the decision-parity gate; what remains is the
other variant the doc named — **capture the per-chunk sequence into a CUDA
graph inside one op per scan** (`hostloop_free.BlockGraph`, extended from the
forward-only block call to the whole region with its adjoint).  A small fixed
Python kernel sequence inside one op was measured out in the probe before any
of this was built: the captured walk still issues ~10 host launches per chunk
(~57 µs each of Triton binding alone), which is seconds per step — the graph
is not an optimisation here, it is the only form that removes the host.

## Memory and lifetime (the fault `reduce-overhead` had, not repeated)

`torch.compile(mode="reduce-overhead")` broke on this model because 24 shared
compiled modules drew from one cudagraph-trees pool and invalidated each
other's outputs between a forward and its backward
(`RESULT_MATRIX_EMENDER.md`).  K5B's graphs are **per-mixer slots**: each
`MatrixMixer` owns a slot id, and a slot's static buffers and forward-pool
saves are rewritten only by its own next microbatch — which the training
protocol (`fwd(m), bwd(m), fwd(m+1), ...` strictly interleaved) always
schedules after the backward that consumes them.  The forward saves live in
its capture pool (~86 MB per scan at the registered shape — the same bytes
the per-chunk arm's autograd graph keeps alive); the 24 backward graphs share
ONE pool handle (they replay strictly sequentially and their intermediates —
the 67 MB recompute trajectory, the adjoint partials — die inside each own
replay).  Peak allocation at mb8: **22.12 GB** (K5A arm: 20.27; the card is
47 GB).

## The compile-hosting discovery: impls run with autocast disabled

The K5B op compiles inside the registered graph — no `InductorError`, the
compiled wrapper bit-identical to the uncompiled one (receipt rows) — but
getting there surfaced a trap worth receipting on its own: **under
`torch.compile`, a custom-op impl runs with ambient autocast DISABLED**
(inductor bakes the casts into the traced graph; an opaque impl never sees
them — measured with a probe op in the receipt's trail).  Left implicit, the
whole-scan body's preprocessing would have silently run fp32 under the
compiled wrapper and changed the decision inputs' dtype regime.  The wrapper
therefore captures the caller's autocast regime (dynamo guards the state, so
the traced value is the caller's) and passes it into the op **explicitly**
(`cast_code`); the impl runs its body under exactly that context, eager or
compiled, and the slot key carries the code instead of an ambient read.  The
registered eager router's regime is what the flip test holds; the compiled
*registered* arm's own inductor-vs-eager preprocessing delta (K5A's probe row:
130979/131072 entries bit-equal at the registered shape) predates this stage
and is re-measured in the K5B receipt so the two cannot be confused.

## Gates A–E (all banked in `receipts/k5b_whole_scan.json`)

**Gate A — bit-exactness and the flip test.**  At the registered shape (batch
8, 2048 tokens, rank 128, chunk 64, bf16 autocast): the recorded schedule
fields, weights, reads, state and liveness are bit-equal to the per-chunk
K5A+K4 arm's, and the wrapper's three call forms (eager body, capture+replay,
replay) produce bit-equal reads and bit-equal `dquery`/`dpayload`/`ddecay`.
The **flip test**: the router the walk runs vs the registered `route_chunk`
on real chunk-entry states from a real routed scan — **0 differing lanes in
8,650,752**, every schedule field of all 32 chunks.  The **model-level form**:
one whole training step of a depth-2 model, the whole-scan arm vs the
per-chunk arm — the loss (5.710614204406738) and all 23 parameter gradients
bit-equal.

**Gate B — the floors, re-measured first.**  fp32 (rank 8, distinct
directions): floor 7.45e-9, the walked router's max deviation 1.49e-8 (median
9.3e-10) — within `NOISE_FACTOR`.  Registered bf16 (rank 64): floor 1.14e-3,
the deviation sits at the floor (the regime's own bf16 rounding — eager's
scores are bf16) with median 9.3e-9.  The adjoint floors (the eager autograd
vs the fp64 adjoint oracle): fp32 3.15e-6, bf16 0.898 on that fixture's scale
— and the whole-scan adjoint is *bit-equal* to the arm's (gate A), so the
floor that bounds the arm bounds it.

**Gate C — the host instruments.**  Per scan (fwd+bwd, the registered mixer
shape, warmed): **41.9 → 1.0 ms** of host issue time; over the 192
scans/step, **8.05 s → 0.19 s**.  Op dispatches per step, counted: before,
the K5A arm's profiled step shows `x45::routing_diff_fwd` 6,144 +
`x45::fused_block_scan` 6,144 + `x45::fused_block_scan_bwd` 6,144 +
`x45::routing_diff_bwd` 6,144 = **24,576**; after, the K5B ledger around a
whole step shows **192 forward + 192 backward, every one a graph replay, 0
eager walks, 0 fallbacks** = **384**.  A 64× cut, and the Triton launch
binding path (~57 µs × ~250 captured launches per scan) no longer exists on
the host at all.

**Gate D — the step metric, four arms, ONE session.**

| per step (mb8, 64 windows, one session) | step wall p50 | kernels/step | device busy | host gap | steady tok/s | peak |
|---|---:|---:|---:|---:|---:|---:|
| registered arm | 9.81 s | 434,366 | 3.34 s | 6.47 s | 10,316 | 24.31 GB |
| fused arm (gates B+C) | 8.51 s | 207,983 | 3.07 s | 5.45 s | 11,586 | 21.28 GB |
| fused + K5A routing | 8.45 s | 125,032 | 3.07 s | 5.38 s | 11,970 | 20.27 GB |
| **fused + K5B whole scan** | **3.25 s** | 193,315 | **3.23 s** | **0.024 s** | **31,046** | 22.12 GB |

The identical arm's same-day single-arm draw (same harness, same protocol,
same card, hours earlier) measured **3.20 s and 33,300 tok/s steady**.  The
honest distance to 32k is therefore a session statement, not an arm
statement: this session's box drew slow on the host side for *every* arm (the
registered arm: 9.81 s against its banked 7.82 s, at identical kernel counts —
434,366 — and identical device busy), and the K5B arm, being host-immune,
tracks the device draw (3.23 s busy here, 3.17 s in the faster draw).

**Gate E — determinism, pathology, the K4 grad gates.**  Two replays on the
same pinned inputs: bit-identical reads, state, `dquery`, `dpayload`,
`ddecay` (no atomics anywhere in the walk).  The refusals are the per-chunk
stack verbatim and bit-equal to calling it directly: the growth policy, the
append-only control, rank 512 (K5A's register-tile budget and the persistent
kernel's, refused by name), an explicit initial state, non-CUDA inputs, and
the registry overflow.  The K4 grad gates hold in their strongest form: the
gradients are bit-equal to the arm's, not merely within its floors.  All five
lane suites re-run verbatim as subprocesses: `test_k4_fused_backward`,
`test_k5_recompute`, `test_k5_routing_fusion`, `test_k5a_routing_diff`,
`test_k5b_whole_scan` — green.

## The ranked residual (31.0k this session, 3% under the 32k line)

The host program is finished: the gap is **0.024 s of a 3.249 s wall**.  The
residual is device-side, ranked with the same attribution discipline:

1. **The whole-scan body's own glue — the owning component of the shortfall.**
   The K5B arm's device busy is 3.23 s against the fused arms' 3.07 s:
   **+160 ms/step**, and the kernel count shows exactly where it lives
   (193,315 vs 125,032, ~68k small kernels at ~2.3 µs average).  Two parts,
   both consequences of moving the region inside one op: (a) the
   parity-mandated preprocessing now runs *eager and unfused* — ~10 kernels
   per chunk against the compiled arms' ~5 inductor-fused ones — visible as
   +480 ms of unfused elementwise in the profile (partly offset in the
   attribution by kernels the compiled arms hid inside their graph buckets);
   (b) the body's assembly copies (the per-chunk reads slices, the backward's
   `dquery`/`dpayload`/`ddecay` slice writes, the state-chain adds, the
   adjoint's contiguous copies).  Part (a) is not removable: the settled
   ruling keeps the decision inputs torch-computed and the chain verbatim,
   and the probe rows are why.  Part (b) is removable in principle
   (strided-output launcher variants for the scan and adjoint kernels),
   worth ~40–60 ms — measured, not estimated by us: the `aten::copy_` and
   elementwise rows of the profile.
2. **The session draw.**  Every arm's host measured ~25% slower this session
   than the banked one (registered 9.81 s vs 7.82 s at identical kernels and
   device busy); the device drew ~1.5% slower than the same-day smoke (3.23 s
   vs 3.17 s busy).  The K5B arm is host-immune, so its number follows the
   device draw: 33.3k in the faster same-day draw, 31.0k here.
3. **The K4 doctrine's device cost, unchanged by this stage**: the backward's
   recompute-over-store trajectory (586 ms/step) plus the reverse walk
   (429 ms) plus the K5A adjoint kernels (191 ms).  Storing the trajectory
   instead was K4's option 1, rejected at 51.5 GB for mb8; the recompute is
   the price of the memory fit and it is the same price the K5A arm pays.
4. **The remaining 24 ms of host** — two op dispatches, four input/gradient
   copies and two graph replays per scan, 192 scans.  At the measured
   1.0 ms/scan it would take a device draw ~2% faster for the wall to cross
   the 32k line in a slow session.

## The default, and why

`fused_whole_scan` is landed behind the flag and **default off**, like
`fused_routing_diff` before it — but with the opposite verdict on the step
metric: this flag's arm *wins* the step (3.25 s against the registered arm's
9.81 s same-session, 8.45 s for the K5A arm), its decisions are bit-identical
to the registered eager router's, its gradients bit-identical to the
receipted K5A+K4 stack's, and its host gap is 24 ms.  Flipping the default is
justified by this receipt the moment the registered arm's own semantics are
to be served by the stack it already receipted; the flag stays off only
because the registered path is untouchable by this stage's rules, not because
the number says so — the number says the opposite.
