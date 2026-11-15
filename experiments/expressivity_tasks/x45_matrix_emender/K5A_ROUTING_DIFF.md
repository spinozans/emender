# K5A — the routing region's differentiable channels, fused; and the honest end of the launch-count program

K5 named the item that owned the step's remaining host gap: the routing
region's *differentiable* channels — read weights (softmax·strength, forward +
backward), payload lanes, similarity preprocessing, ~40 compiled kernels per
chunk — plus gate A's own decision kernel, which cost 0.50 s of device time per
step as a separate launch.  This stage built the approved fusion: **one Triton
forward/backward pair covering the whole routing region**, with the settled
parity rules enforced gate by gate.  Everything is banked in
`receipts/k5a_routing_diff.json` (parity, floors, launch counts, the cd
protocol) and `receipts/k5a_step_metric.json` (the step metric, three arms, one
session).

**Outcome, stated up front.**  The parity gates all hold: decisions, slots, op
codes, liveness and payloads are **bit-identical to the registered path by
construction**, the mandatory decision-flip test on the real routed fixture
asserts **zero flips in 8,650,752 lanes**, and the read weights / routing
gradients are inside the measured floors.  The launch-count program is
**finished**: the step now issues **125,033 kernels/step** (registered: 434,366;
gates B+C: 207,985; GDN-2's class: 59k), and gate A's 0.50 s decision-kernel
device cost is **gone** (the whole four-kernel K5A pair costs 262 ms of device
time per step).  But the step wall did **not** improve — it got slightly worse —
because the step is no longer launch-bound: the residual host gap is owned by
**per-chunk eager Python op dispatch** (the custom-op boundary the pair must
cross, four op bodies per chunk including K4's), which costs more host time
than the launches it saves.  The 32 k tok/s target is **not reached**, and the
receipts name the owning component and the direction that would move it.  The
fused router is therefore landed, gated, and **default-off with the reason**.

| gate | demand | state |
|---|---|---|
| A | decisions/slots/payload codes bit-exact, incl. the mandatory flip test on the real fused path | **landed** — bit-exact by construction, 0 flips in 8.65 M lanes |
| B | weight/read floors measured first, fused within a stated multiple, max + median | **landed** — within `NOISE_FACTOR` on every fixture, both regimes |
| C | launch counts: routing before/after, total kernels/step | **landed** — 136→13 eager, 41→29.5 compiled, 434k→208k→**125k**/step |
| D | the step metric, matched pair, honest distance to 32 k | **landed** — 9.08 s wall, 125,033 kernels, 11.2 k tok/s: **not reached** |
| E | determinism, pathology, the 0.50 s gate-A device cost gone | **landed** — bit-stable reruns, rank 512 refused by name, kernel absent |

## What was built (`k5a_routing_diff.py`)

Four Triton kernels behind two `torch.library` custom ops (the K4 hosting
pattern: the forward `x45::routing_diff_fwd` is one opaque call, the adjoint
`x45::routing_diff_bwd` its own, because `torch.compile` traces a registered
backward into the joint graph and the Triton launches must stay out of both
traces):

* `_k5a_prep_kernel` — one program per `(row, token_block)`: the read weights
  (the unit-normalised `scale·cos` bmm as a `tl.dot` in the inputs' own dtype,
  rounded to bf16 exactly where eager rounds under the model's autocast, the
  live-masked fp32 softmax, the strength multiplier), the per-token selections
  from the similarity (best/second/worst with the tie-break penalty and torch's
  first-index rule), the MERGE blend, and the per-block need counts.
* `_k5a_emit_kernel` — one program per `(row, token_block)`: the free-slot
  ordinal resolution (the one cross-token dependency, resolved by loading the
  earlier blocks' counts — a handful of scalars — so the decision walk is
  **token-block parallel** where gate A's kernel was a latency-bound serial
  walk), the five-op code selection, and the payload lane select.
* `_k5a_bwd_scores_kernel` — the softmax/normalisation adjoint: `dquery`,
  `dpayload` (a masked passthrough, bit-exact), and the shared reduction
  partials.  The two cross-value-block reductions `du·uq` and `dur·ur` reduce
  algebraically to `Σ dscores·scores` per token/rank — one tile reduction
  instead of a second dot chain.
* `_k5a_bwd_rows_kernel` — the state-row adjoint per `(row, value_block)`: the
  score path through `unit(rows)`, the strength-multiplier path through
  `row_strength`, and the MERGE blend scatter written as its true algebra — a
  one-hot matmul, not a serial masked-add walk — with token blocks accumulated
  in-kernel so each program writes its own columns of `drows` directly.  No
  atomics anywhere in the pair.

The wrapper `routing_diff_route_chunk` is a drop-in for `schedule.route_chunk`
(`MatrixMixer(fused_routing_diff=True)`; default off), and refuses two shapes
to the eager router *by name*: `rank > 128` (the selection and softmax tiles
hold the whole rank axis in registers) and the append-only control
(`policy.ops=False`).

## Why the decision inputs stay torch-computed (the probe on the record)

The settled ruling allows the similarity/strength to be recomputed only if
reproduced exactly, with the flip test mandatory.  The receipt's
`similarity_probe` rows are the measurement: an in-kernel recomputation of the
eager bf16 chain matched the eager similarity on **~48 %** of entries at every
`tl.dot` blocking tried, and the bf16 payload norm on **0/1024** entries —
torch's `norm` reduction order and the cuBLAS/inductor MMA tiling are not
reproducible from Triton on this stack.  Recomputing the similarity in-kernel
would therefore have put decisions at flip risk on real fixtures, so the wrapper
keeps the eager preprocessing's own `similarity` and `strength` tensors (four
compiled kernels: the two norms, the stored division, the bmm) and the fused
kernels *consume* them.  Decisions are then bit-identical **by construction**
and the flip test is the proof, not a gamble.  The preprocessing is computed
from **detached** inputs: nothing differentiable ever consumed it (everything
similarity feeds ends in a comparison; the eager engine prunes the branch at
runtime), and under `torch.compile` the dead branch is *not* pruned from the
joint graph — leaving it attached would compile and run a dead backward for
provably zero gradients.

## Gate A — bit-exactness and the flip test

* **Pinned chunk states** (the deterministic instrument): every schedule field
  and the payloads bit-equal, every chunk, fp32 and bf16 autocast.
* **Whole routed scans**: five-ops small, POP-saturated, ragged 100/64 and
  1-token tails, rank prefix 6-of-16, the growth policy, and the **registered
  bf16 shape** (batch 8, 2048 tokens, chunk 64, rank 128) — schedule bit-equal
  on all of them, the continuous channels inside the measured floor.
* **The tie fixture** (every direction stored twice): the lower index wins,
  exactly as torch's argmax and the tie-break penalty demand.
* **THE FLIP TEST** (mandatory per the ruling): the real routed fixture — real
  chunk-entry states from a real routed scan at the registered shape under bf16
  autocast — **0 differing lanes in 8,650,752** across every schedule field of
  all 32 chunks.
* **The cd-protocol training fixture**: the registered model's own block-0
  mixer on a real flagship-stream activation — schedule bit-parity on every
  chunk, and one whole eager-scan training step per router whose gradient
  deviation (4.96 on a gradient scale of ~30,200) is within
  `NOISE_FACTOR` of the eager arm's own rerun deviation (2.02; the
  `scatter_add_` atomics noise both arms share).
* **The compiled regime**: `torch.compile` hosts the op pair with no
  `InductorError` (the bool→int8 views stay inside the opaque impls, the gate-B
  lesson), the compiled fused router's decisions are bit-identical to the
  compiled *eager* router's at bf16, and equal to the eager router's at fp32.

## Gate B — the measured floors (floor first, then the deviation)

| fixture | weights floor | fused max | fused median | within 8× floor |
|---|---:|---:|---:|---|
| fp32 (rank 8, distinct directions) | 7.45e-9 | 1.49e-8 | 9.3e-10 | yes |
| registered bf16 (rank 64) | 1.14e-3 | 1.14e-3 | 9.7e-6 | yes |

The floor is the K23 discipline: two orderings of the same math (torch's
softmax vs the written-out expression) plus the fp64 oracle, measured on the
same fixture before the fused path is judged.  The bf16 floor is the regime's
own rounding — eager's scores are bf16 — and the fused max sits at it with a
median three orders below.

The **gradients** follow the same discipline against the fp64 adjoint oracle
(the recorded discrete channel pinned from the eager schedule, as the doctrine
demands): fp32 floor 5.3e-6 with dquery/dstate deviations 6.0e-7/6.7e-6; the
bf16 regime's floor is the eager arm's own bf16 backward rounding (0.898 on
that fixture's scale) and the fused adjoint sits at it.  **`dpayload` is
bit-exact in both regimes** — it is a masked passthrough of the incoming
gradient, so any difference would be a bug, not a floor.

The **K4 grad gates re-run unchanged** (receipt, `gate_suites_rerun`):
`test_k4_fused_backward.py`, `test_k5_recompute.py`,
`test_k5_routing_fusion.py` and the new `test_k5a_routing_diff.py` — all four
green, run verbatim as subprocesses.

## Gate C — the launch counts

| one chunk of routing | kernels/chunk | note |
|---|---:|---|
| eager `route_chunk` (before) | 136 | uncompiled, the drop-in regime |
| fused K5A router (after) | **13** | the op bodies launch 2+2 kernels; the rest is the compiled preamble |
| compiled routing region (before) | 41.0 | K5's differential instrument, this session |
| compiled routing region (after) | 29.5 | same instrument; see the caveat below |

The compiled differential (routed scan minus scan-only over the same recorded
schedule) understates the fusion because the *scan's* kernels fuse differently
around an opaque op than around eager ops — the subtraction is polluted by the
scan side.  The clean instrument is the compiled router in isolation (same
harness, same loss glue, both routers): **eager 41 kernels/chunk → K5A 25**,
of which the pair itself is 4 Triton launches and the remainder is the
decision-input preamble (two norms, the stored division, the bmm with its
pointer-array setup) plus harness glue.  The end-to-end number is unambiguous
though — it is the step:

| per step (mb8, 64 windows, ONE session) | step wall p50 | kernels/step | device busy | host gap | peak |
|---|---:|---:|---:|---:|---:|
| registered arm | 7.82 s | 434,366 | 3.35 s | 4.47 s | 24.31 GB |
| fused arm (gates B+C) | 8.53 s | 207,985 | 3.06 s | 5.46 s | 21.28 GB |
| fused + K5A routing | 9.08 s | **125,033** | 3.10 s | 5.98 s | 20.27 GB |

(A first run of the same three arms in an earlier same-day session reproduced
the session-invariant numbers — 434,366 / 207,985 / 125,032 kernels — with
walls 7.79 / 8.65 / 8.82 s; the kernel counts and the ranking are stable, the
±0.3 s walls are the step-to-step spread.)

## Gate D — the step metric, and the honest distance to 32 k

**32 k tok/s is not reached.**  This session's protocol draws give 11.2 k
tok/s steady for the K5A arm (12.0 k for the B+C arm, 10.6 k registered; the
session-1 draw gave 11–14 k across arms).  The device floor is unchanged and
healthy — 3.10 s busy, the same class as every fused arm — and at zero host
gap this step would run at the ~35 k the K5 receipts projected.  The host gap
is the whole distance, and it **grew** with this stage.

**The ranked residual, decomposed with the same attribution:**

1. **Per-chunk eager Python op dispatch — the owning component.**  Every
   custom-op boundary the fused arms cross runs its Python body once per
   chunk at runtime, opaque to the compiled graph: K4's forward + backward
   pair and now K5A's.  Measured in isolation, the K5A pair costs
   **202 µs (forward) + 179 µs (backward) of host time per chunk** — of which
   ~230 µs is the four Triton launches' Python binding path (~57 µs each,
   barely sensitive to argument count), the rest allocation and dispatch
   machinery.  At 6,144 chunks/step that is seconds of host time that no
   launch-count reduction can touch: **the op boundary itself is the cost
   now**.  This is why the K5A arm's host gap (5.98 s) exceeds the B+C arm's
   (5.46 s) even while removing 83 k launches: the 83 k compiled launches it
   saved were worth ~2.2 s of host issue time, and the pair's Python bodies
   gave most of it back.
2. **The decision-input preamble** — 4–6 compiled kernels per chunk (the two
   norms, the stored division, the similarity bmm with cuBLAS's
   pointer-array setup), ~25–30 k launches/step.  Bit-exact decisions require
   them (the probe rows); they are the floor of the fused router's launch
   count.
3. **The compiled backbone and scan region** — the remaining ~95 k
   kernels/step (matmuls, norms, MLP, optimizer, the persistent scan
   kernels), whose dispatch the profile attributes as before.

The direction that would move it is named and is *not* more fusion of this
kind: remove the per-chunk Python — one op per *scan* instead of per chunk,
CUDA-graphing the step, or hosting the launches from inductor-generated code
(raw captured launches carry no per-chunk Python, but they cannot carry
gradient — the K23 lesson — so the differentiable channels need an op
boundary somewhere; the open question is *where*).

## Gate E — determinism, pathology, and the gate-A device cost

* **Determinism**: two fused runs on the same pinned inputs are bit-identical
  on every channel (no atomics anywhere in the pair).
* **Pathology**: rank 512 is refused to the eager router *by name* (the
  register-tile budget), and the refusal is the registered router verbatim —
  bit-identical by construction; ragged chunks and 1-token tails are in the
  gate-A fixtures.
* **The 0.50 s gate-A device cost is gone**: the K5A step profile contains no
  `_route_decisions_kernel` at all (the walk is folded into the
  token-block-parallel prep/emit pair), and the whole four-kernel pair costs
  **262 ms of device time per step** (prep 51.5, emit 17.9, bwd-scores 82.1,
  bwd-rows 110.4) — half of what gate A's kernel alone cost, with the backward
  included.

## The default, and why

`fused_routing_diff` is landed behind the flag and **default-off**, like
`fused_routing` before it.  The reasons, both receipted: the fused router's
decisions/payloads are bit-identical and its floors hold, so it is *not* a
semantics knob — but at the registered mb8 step it buys the launch count
(208 k → 125 k kernels/step, the program's headline) and **not the wall**
(8.53 → 9.08 s), because the per-chunk op dispatch costs more host time than
the launches it saves.  Flipping the default would be justified only by a step
metric that wins, and the honest number says it does not — the next item is
the per-chunk Python, not more fusion.
