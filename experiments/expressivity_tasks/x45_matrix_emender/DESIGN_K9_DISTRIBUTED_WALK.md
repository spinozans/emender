# K9 design — THE DISTRIBUTED-PERSISTENT WALK (written BEFORE the build)

**Status: the design of K9, resolved by the gate-0 receipts before any kernel was
written, and approved by the supervisor before the build.** The barrier economics
are measured and receipted in [`receipts/k9_barrier_economics.json`](receipts/k9_barrier_economics.json)
(`k9_barrier_economics.py`, gate 0); this document states the design those
receipts force and the exactness accounting of every channel.

## 0. The constraint, receipted

The K7 walk at rank 2048 holds its 16 MB store in HBM and walks it with ONE
program per row: per token it sweeps the store twice in slot tiles, and the
sweeps are dependent defined-order tree folds — fold-bound at 275 GB/s
effective, 374.3 ms per (layer, window) scan (the K8 gate-D table: 414 ms with
the recorded channels), 103 GB of DRAM traffic per scan. The FLOPs are trivial.
The K9 brief: keep the state ON-CHIP, distribute it across CTAs, and combine
per token by the Lean one-tree reduction — exact, no sketch, no new semantics.

## 1. The impossibility receipt (a first-class deliverable, not an apology)

The brief's design — each CTA permanently owns `rank/BLOCK` slots in shared
memory (BLOCK=128 slots: 64 KB at `d_v` 128), with a grid-wide barrier per
token — is **not expressible on this stack, and at the probe shape not
physically possible on this card**. Both facts are measured, not assumed
(gate 0, `receipts/k9_barrier_economics.json` → `impossibility`):

1. **No Triton grid sync.** Triton 3.5.1's `triton.language.extra.cuda`
   exposes only the PDL utilities (`gdc_wait`/`gdc_launch_dependents`) and
   diagnostics; there is no grid-wide barrier primitive, and the NVIDIA launch
   path has no cooperative-launch entry point (a source scan of the installed
   backend is in the receipt). A software grid barrier (spin on a global flag)
   is expressible in principle but is only deadlock-free if every CTA of the
   grid is co-resident — which fails next.
2. **The residency arithmetic.** RTX 6000 Ada: 142 SMs, 100 KB smem per SM.
   A 64 KB CTA fits once per SM (two would need 128 KB) — **142 co-resident
   CTAs, 9.3 MB of addressable on-chip state**. The probe-shape store is 16
   rows × 1 MB = **16 MB in 256 CTAs**: it does not fit the chip, so no barrier
   — cooperative or software — can exist for the smem-resident form at the
   shape the speedup is for. (It would not even fit at one row per grid launch
   without re-streaming, which defeats the design.)
3. **The unlock, named for a future stack:** a Triton with cooperative launch
   (or a card with ≥16 MB of co-addressable smem per scan's rows) could build
   the smem-resident form; the per-token work decomposition, the per-block
   exact partials and the one-tree reduction specified below carry over
   verbatim. On THIS stack the standing directive holds: no new CUDA C++.

## 2. The design the receipts force: distributed, L2-resident, graph-synchronized

The pre-stated decision rule — build the distributed design iff the projected
per-token barrier overhead stays under ~40 ms/scan — fires on the only
Triton-expressible per-token barrier, the **kernel-launch boundary**: eager it
costs 19.2 µs (118 ms/scan, over budget), inside a whole-scan CUDA graph (the
lane's own K5B discipline) it costs **0.56 µs per node**, so 2048 tokens × 3
syncs = **3.4 ms/scan**. Decision: BUILD THE DISTRIBUTED WALK, with the store
**L2-resident on-chip** (16 MB < 96 MB L2; measured hot: the 256-CTA sweep
reads the 16 MB store in 14.2 µs ≈ 1.2 TB/s, and reads+writes it in 14.3 µs)
and the per-token barrier the graph-captured launch boundary.

Grid and ownership (the brief's shape, the substrate swapped):

* **grid (row, slot-block)**, one CTA per (batch·head row, `NT` slot blocks of
  `BR`=64 slots × 128 values — the lane's own proven 8 192-float walk tile,
  `K7_WALK_TILE`, now ONE CTA per tile instead of one program's serial loop
  over all 32 of them; measured the fastest width at the probe shape, 56.1 ms
  vs the brief's smem-design 128-slot 64.2 ms — a measurement, not a
  preference, the K7 §6.1 precedent). The state itself lives in one contiguous
  fp32 global buffer that the L2 holds hot for the whole scan.
* **per token, three kernels** (the launch boundaries ARE the barrier):
  1. **score sweep** (grid rows×NT): every CTA loads its tile from the L2, does
     the per-row DEFINED value-axis tree folds — `nr`, the read score, the
     decision cosine — and writes the three whole-rank scratch channels
     (`score`, `sim`, `strength`; dead rows masked to −inf/−2 exactly as K7's
     sweep A). Program (row, 0) additionally performs the previous token's
     read-numerator one-tree combine (below). This is the fold work K7 does
     serially in one program, distributed.
  2. **decision** (grid rows): ONE program per row runs K7's decision phase
     VERBATIM on the whole-rank scratch vectors — the softmax max (exact),
     the denominator as the defined whole-axis tree IN ONE PROGRAM (no split:
     the weights channel stays bit-exact against the oracle), the registered
     selection chain with the first-index tie rules, the MERGE blend from two
     gathered rows, the lane. It writes the decision channel in the scan's
     `(b, T, h, lanes)` layout, the whole-rank weights scratch for the apply
     sweep, the lane vector, and six decision scalars.
  3. **apply sweep** (grid rows×NT): every CTA re-loads its tile, computes its
     block's read-numerator partial (a native per-tile sum — the floor channel,
     K7 sweep B's own form), applies DECAY in place (`mul.rn`), and the OWNING
     block alone executes the op in its tile: the erasing write or the
     `old_row + lane` carry (the decayed row captured in-tile, the exact
     single-nonzero gather), the MERGE source zeroing, and the liveness update.
     A NONE op writes nothing, exactly as K7.
* **the one-tree reduction** (the Lean full-population law, the K7 chunked
  path's own proven mechanism): the read numerator's per-block partials are
  combined in ONE defined balanced tree over the slot-block axis — a tile's
  internal tree plus one tree over the tile results IS the whole-axis tree,
  zeros padding exactly — by the next token's score-sweep program (row, 0),
  with one final tiny combine kernel after the last token. The softmax
  denominator needs no split at all (the decision program is one CTA per row).

**Whole-scan CUDA graph** (the K5B discipline, the lane's own): every buffer
is static and per-(shape, snap, policy) cached; the 3·T+1 launches are captured
once into one graph (~6 145 nodes at T=2048) and replayed per scan — one
replay, zero host syncs, the decision data flowing GPU-side through the static
buffers. Warmup runs the whole body eagerly once (compiles the two score
variants), then once on the capture side stream, then captures. The eager body
stays available (`graph=False`) and is the graph-vs-eager determinism
cross-check.

## 3. What is bit-exact and what is at the floor, stated once

* **Bit-exact (gate: `torch.equal`)** — every decision channel (slot, op,
  erase, live_after, the payload lane, pre-decision liveness), the read
  WEIGHTS channel, and the exit state and liveness, against the K7 eager oracle
  (`k7_high_rank_store.sequential_oracle`, K6's precedent) AND against
  `k7_sequential_walk` itself. Why: the per-row value-axis folds are row-local
  (a tile never splits a row), identical expressions in K9's score sweep and
  K7's sweep A; the scratch handoff is fp32-exact; the decision program is
  K7's whole-rank code verbatim (the denominator tree unsplit); the lane
  application is K7's, with the same capture-then-write order.
* **At the measured floor** — the read OUTPUT: K7's kernel sums per-tile
  partials sequentially across tiles; K9 combines the same per-tile partials
  in one tree over the slot-block axis. Both are orders no torch expression
  reproduces, both feed no decision (the K1/K23/K6/K7 corner); the gate is the
  measured fp32-vs-fp64 floor × `NOISE_FACTOR` (8), floors re-measured first.

## 4. The visibility analysis (the K7 nondeterminism class, closed)

K7's first walk kernel was nondeterministic because one warp's global store
was not guaranteed visible to another warp's subsequent load without a CTA
barrier. K9's granularity moves the hazard up one level and closes it by
construction, plus one deliberate reordering:

* **Cross-CTA** (K9's own class): every buffer written by one kernel launch
  and read by another is ordered by the launch boundary — inside a CUDA graph
  the stream order is the barrier (an implicit device-scope fence between
  dependent launches). No spin, no atomics, no acquire/release reasoning.
* **Intra-kernel**: each apply program loads and stores only its own private
  tile region; the score and decision kernels never read back what they write
  in the same launch.
* **The one deliberate reordering**: K7's single-program walk updates the
  liveness in its decision phase (after its own barrier-terminated sweeps). A
  K9 decision program that BOTH loads the whole liveness vector and stores
  `live[slot0]` would race a warp's store against another warp's load of the
  same byte inside one CTA — so the liveness update MOVES to the apply sweep's
  owning block (no program reads liveness in that launch; the next token's
  score sweep reads it across a launch boundary). The final bytes and their
  order relative to the state writes are exactly K7's, and the gate holds the
  liveness channel bit-equal.

## 5. Rank routing and growth composition

* **Rank ≤ 128 stays on the existing paths** (bit-pinned, already fast): the
  K9 walk REFUSES `rank × value_dim ≤ MAX_SEQ_TILE` (16 384 — the on-chip
  register budget, K6's own gate) by name; `k9_auto_walk` routes those ranks to
  `k7_sequential_walk` (imported, never edited) and everything above to the
  distributed walk.
* **Growth re-dispatch composes at the span-API level** (K8's machinery,
  imported): `k9_span_walk` replays a RECORDED span table — spans at or below
  128 on the existing K7 path, grown spans on the K9 walk, the state chained
  through K8's own zero-injection pads (`_pad_state`/`_pad_live`, imported).
  Every snap is its own cached graph (more slot blocks per snap, the new rows
  exact zeros — the prefix-reveal invariant), so a growth event re-dispatches
  to the next snap's graph with a state copy-in at the boundary. Full
  integration into `k8_growth.growing_scan`'s dispatch would need an edit to
  `k8_growth.py` (out of scope by the standing rule); the span-API composition
  is the receipted evidence the re-dispatch is exact.

## 6. What is honestly NOT here

* **No smem-resident state** — impossible per §1; the L2 is the on-chip
  substrate, and the honest expectation is a **4–6× scan speedup** (the
  projection band from gate 0: ~62–90 ms vs the receipted 374.3/414 ms), not
  the 10–30× the smem design projected.
* **Forward only** (K7's walk is forward-only; K8's hi backward is untouched,
  out of scope); `check_no_grad` refuses a graph through every entry point.
* **No whole-arm claim** — the receipt projects the whole-arm tok/s from the
  measured per-scan walls and the K8 arms' own accounting, stated with its
  assumptions.
