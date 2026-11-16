# K9 — THE DISTRIBUTED-PERSISTENT WALK: the high-rank per-token state kept on-chip

Stage 9 of the X45 kernel lane. The diagnosis this stage answers is receipted
twice over: at rank 2048 the K7 global-tiled walk streams the whole 16 MB store
from HBM twice per token (103 GB/window) and pays its dependent defined-order
tree folds serially — fold-bound at 275 GB/s effective, **374.3 ms per (layer,
window) scan** (`receipts/k7_high_rank.json`; 414 ms with the recorded
channels, `receipts/k8_gates.json` gate D), while the FLOPs are trivial. The
brief's design — the state distributed across CTAs' shared memory, combined per
token by the Lean one-tree reduction — is **not expressible on this stack and,
at the probe shape, physically impossible on this card**; the gate-0 receipts
measure both facts, the pre-stated decision rule fires on the only
Triton-expressible per-token barrier, and the supervisor approved the variant
built here before the build (the escalation is on record).

**Headline, as narrowly as the evidence supports.**

* **The smem-resident design is impossible, receipted as a first-class
  deliverable** (gate 0, `receipts/k9_barrier_economics.json`): Triton 3.5.1
  has no grid-wide barrier (only PDL utilities in `tl.extra.cuda`) and no
  cooperative launch; and a 64 KB CTA fits once per SM on this RTX 6000 Ada —
  **142 co-resident CTAs, 9.3 MB of addressable on-chip state against the 16 MB
  the probe shape needs**. The unlock for a future stack is named in the design
  doc (a Triton with cooperative launch, or a card with ≥16 MB of co-addressable
  smem per scan's rows); the standing no-new-CUDA-C++ directive held.
* **The distributed walk runs anyway, on the on-chip substrate the card does
  have: the L2.** The store lives in one contiguous fp32 buffer the 96 MB L2
  holds hot for the whole scan (measured: the 256-CTA sweep reads it at
  ~1.2 TB/s), the sweeps distribute across (row, slot-block) CTAs exactly as
  the brief specified, and the per-token barrier is the **graph-captured
  kernel-launch boundary** — 0.56 µs/node (the lane's own K5B whole-scan
  discipline), so 2048 tokens × 3 syncs = **3.4 ms/scan of barrier**, under the
  pre-stated 40 ms budget by 12×.
* **The number (gate C, probe shape batch 8 × 2048 tokens × 2 heads, fp32,
  forward only):** the rank-2048 scan falls **374.5 → 56.2 ms (6.7×)**; rank
  1024: 191.2 → 36.1 ms (5.3×); rank 512: 97.1 → 30.0 ms (3.2×). The honest
  scaling law: per-token ≈ **10.4 µs + 1.07 µs × slot-blocks** — a fixed
  decision/barrier cost that dominates at low rank, plus the sweep compute.
  One-time capture cost ~0.8–1.2 s per (shape, snap, policy), ~6 145 graph
  nodes replayed per scan.
* **Parity is total (gate B): 0 flips at ranks {256, 1024, 2048} against BOTH
  references** — the K7 global-tiled walk (kernel vs kernel) and the K7 eager
  oracle — decisions, payload lanes, read WEIGHTS, pre-decision liveness, the
  exit state and the liveness bit-equal; reads at 1.0–1.1× the re-measured
  fp32-vs-fp64 floor; the graph replay bit-equal to the eager launches;
  determinism across runs (the K7 cross-warp visibility class closed by
  construction at this granularity and tested). All five ops bit-exact on
  crafted fixtures; all-dead token 0 reads exactly zero at rank 2048.
* **Growth re-dispatch composes at the span-API level (gate D):** a recorded
  span table across snaps {128 → 256 → 512 → 1024} with ragged boundaries runs
  0 flips span by span against the oracles, the zero-injection receipts hold
  (the grown rows dead at entry), the whole-chain exit state bit-equal — every
  snap its own cached graph, the state copy-in at the snap change.
* **The honest expectation revision:** this is a **5–7× scan speedup**, not the
  10–30× the smem design projected. The projected whole-arm effect (from the
  banked K8 receipts' own accounting, stated in the receipt): the fixed-2048
  arm **821.7 → ~1 667 tok/s**, the grow arm **774.9 → ~1 485 tok/s** — and the
  C3 saturation probe at rank 2048 becomes affordable (~3.3 h per 180-step arm
  instead of the banked ~7.4 h), which is the purpose this speedup exists for.

---

## 1. What was built

| file | role |
| --- | --- |
| `k9_barrier_economics.py` | GATE 0, measured before the design: the sync survey, the residency arithmetic, the barrier economics (eager vs graph node), the L2 streaming rates → `receipts/k9_barrier_economics.json` |
| `DESIGN_K9_DISTRIBUTED_WALK.md` | the design, written before the build; the impossibility receipt and the exactness accounting live here |
| `k9_distributed_walk.py` | the walk: `_k9_score_kernel` (the distributed sweep A + the one-tree read combine), `_k9_decide_kernel` (K7's decision phase verbatim, one program per row), `_k9_apply_kernel` (sweep B + the owning-block op execution), `_k9_read_combine_kernel`, the whole-scan graph slots, `k9_sequential_walk` / `k9_auto_walk` / `k9_span_walk` |
| `test_k9_distributed_walk.py` | 14 tests: the 0-flip parity vs both references, every op branch, the graph-vs-eager equality, determinism, ragged windows, the refusals and the auto routing, the growth re-dispatch |
| `k9_gate_receipt.py` | gates B/C/D → `receipts/k9_gates.json` |
| `K9_DISTRIBUTED_WALK.md` | this document |

Zero edits to `k6_*`/`k7_*`/`k8_*` sources, the preregistrations and
`lattice.py` — imported, never modified.

### 1.1 The design, in one paragraph

Per token, three kernels; the launch boundaries are the barrier (the decision
data flows GPU-side through static buffers; the host never sees it):

1. **score sweep** (grid rows × NT): every CTA loads its `BR`=64-slot tile
   from the L2-resident store and does the per-row DEFINED value-axis tree
   folds — norms, read scores, decision cosines — into whole-rank scratch
   channels. These are K7's sweep-A expressions verbatim, row-local (a tile
   never splits a row), so the scratch is bit-identical to what K7's serial
   sweep computes; this is the fold work that made K7 fold-bound, now
   distributed. Program (row, 0) additionally combines the PREVIOUS token's
   read-numerator partials in ONE defined balanced tree over the slot-block
   axis — the Lean full-population law, the K7 chunked path's own proven
   mechanism (a tile's internal tree plus one tree over the tile results IS
   the whole-axis tree, zeros padding exactly).
2. **decision** (grid rows): ONE program per row runs K7's decision phase
   VERBATIM on the whole-rank scratch: the softmax max (exact), the
   denominator as the defined whole-axis tree IN ONE PROGRAM (no split — the
   weights channel stays bit-exact against the oracle by construction), the
   registered selection chain with the first-index tie rules, the MERGE
   blend from two gathered rows, the lane. It writes the decision channel in
   the scan's `(b, T, h, lanes)` layout and six scalars for the apply sweep.
3. **apply sweep** (grid rows × NT): every CTA computes its block's
   read-numerator partial (a native per-tile sum — the floor channel, K7
   sweep B's own form), applies DECAY in place (`mul.rn`), and the OWNING
   block alone executes the op in its tile: the erasing write or the captured
   decayed row + lane (the exact single-nonzero gather), the MERGE source
   zeroing, the liveness update. A NONE op writes nothing, exactly as K7.

The whole `3·T+1` launch sequence is captured into ONE whole-scan CUDA graph
per (shape, snap, policy) — static buffers, one replay per scan — after an
eager warmup; the eager body stays available and is the graph-vs-eager
determinism cross-check (bit-equal, receipted).

### 1.2 The one deliberate reordering (the visibility analysis)

K7's single-program walk updates the liveness in its decision phase, after a
CTA barrier terminates its sweeps. A K9 decision program that BOTH loads the
whole liveness vector and stores `live[slot0]` would race one warp's store
against another warp's load of the same byte inside one CTA — the K7
nondeterminism class at K9's granularity. So the liveness update MOVES to the
apply sweep's owning block, where no program reads liveness at all; the next
token's score sweep sees it across a launch boundary (an implicit device
fence). The final bytes and their order relative to the state writes are
exactly K7's, and the liveness channel is held bit-equal by the gates.

## 2. Gate B — parity (receipt `gate_b_sequential_walk`)

| rank | flips vs oracle | flips vs the K7 walk | state / live | read floor | read dev | × floor | regs (score/decide/apply/combine) | spills |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 256 | **0** | **0** | bit-equal | 8.9e-8 | 8.9e-8 | 1.0 | 80/27/104/18 | 0/0/0/0 |
| 1024 | **0** | **0** | bit-equal | 3.7e-8 | 4.1e-8 | 1.1 | 80/47/104/40 | 2/0/0/0 |
| 2048 | **0** | **0** | bit-equal | 2.3e-8 | 2.5e-8 | 1.08 | 86/64/104/40 | 0/0/0/0 |

Graph-vs-eager bit-equal and determinism-across-runs at every rank (receipted
per row). The random fixtures saturate in APPEND (the fresh-store regime, the
lane's own K6/K7 finding — the histograms are receipted honestly); the
five-op coverage is the crafted-fixture suite's (MERGE on both lanes, POP,
NONE, all-dead zero attention at rank 2048), every branch bit-exact.

## 3. Gate C — the number (receipt `gate_c_the_number`)

Probe shape (batch 8, 2 048 tokens, 2 heads, `d_v` 128, fp32, forward only,
median of 5, wall + CUDA-event device):

| rank | K7 walk (before) | K9 walk (after) | speedup | per-token | slot blocks | capture (one-time) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 97.1 ms | **30.0 ms** | 3.2× | 14.7 µs | 8 | ~0.8 s |
| 1024 | 191.2 ms | **36.1 ms** | 5.3× | 17.6 µs | 16 | ~1.2 s |
| 2048 | 374.5 ms | **56.2 ms** | **6.7×** | 27.4 µs | 32 | ~1.2 s |

Against the K8 gate-D banked 414 ms (the recorded-channel wall the whole arms
pay), the rank-2048 forward scan is **7.4× faster**. The scaling law, fitted
and receipted: **per-token ≈ 10.4 µs + 1.07 µs × NT** — the fixed term is the
decision kernel plus the 3 launch-boundary syncs (3.4 ms/scan of barrier at
0.56 µs/node, the gate-0 receipt), the linear term the distributed sweeps.
The tile-width measurement: the lane's proven 64-slot walk tile (`K7_WALK_TILE`,
8 192 floats, now ONE CTA per tile instead of one program's serial loop over
all of them) measures 56.2 ms against the brief's smem-design 128-slot 64.2 ms
— a measurement, not a preference (the K7 §6.1 precedent).

Two honest observations receipted with it: (i) the bare graph replay in a
tight loop measures ~65 ms — the full call's state copy-in re-populates the
L2 reuse set and the deployed path (copy-in, one replay, clone-out) measures
56.2 ms, which is the number reported; (ii) the capture is a one-time
~0.8–1.2 s per (shape, snap, policy) — amortised over a scan-per-step
schedule, receipted, not hidden.

**The projected whole-arm effect** (from the banked K8 receipts' own
accounting, stated in the receipt: 192 rank-2048 forward scans/step at 414 ms
= 79.5 s of the fixed-2048 arm's 135.4 s step; the backward, 270 ms/scan, is
K8's own kernel path and untouched by K9):

| arm | banked | projected with the K9 walk |
| --- | ---: | ---: |
| fixed-2048 | 821.7 tok/s, 135.4 s/step | **~1 667 tok/s**, ~66.7 s/step (2.0×) |
| grow | 774.9 tok/s, 143.6 s/step | **~1 485 tok/s**, ~74.8 s/step (1.9×) |

After K9, the hi backward becomes the dominant scan cost — the natural
follow-up. The C3 saturation probe at rank 2048 becomes affordable at ~3.3 h
per 180-step arm against the banked ~7.4 h.

## 4. Gate D — the pathology (receipt `gate_d_growth_pathology`)

The growth re-dispatch composes at the span-API level: a recorded span table
with RAGGED boundaries across snaps — `[(0,192)@128, (192,448)@256,
(448,700)@512, (700,960)@1024]` — replays through the auto route (the
existing K7 path at/below the on-chip budget, the K9 walk above), the state
chained through K8's own zero-injection pads (imported `_pad_state`/
`_pad_live`): **0 flips span by span** against the oracles given the chained
entry state, reads at 0.5–1.0× the re-measured floors, the grown rows dead at
every span entry and until the first schedule write that targets them (the
zero-injection receipts), the whole-chain exit state and liveness bit-equal.
Ragged single windows (T = 1, 33, 130): 0 flips. Per-snap occupancy
(sweep CTAs on 142 SMs): 64 CTAs at 256 (0.45 waves) → 512 CTAs at 2048
(3.6 waves) — the smem-residency impossibility is gate 0's.

Full integration into `k8_growth.growing_scan`'s dispatch would need an edit
to `k8_growth.py` (out of scope by the standing rule); the span-API
composition is the receipted evidence the re-dispatch is exact.

## 5. What is honestly not done

1. **No smem-resident state** — impossible per gate 0; the honest expectation
   was revised from the smem design's 10–30× to the measured 5–7×. The L2 is
   the on-chip substrate, and the working set (state 16 MB + streams ~52 MB)
   fits it at the probe shape; a store that outgrew the L2 would degrade
   toward the K7 wall — stated, not hidden.
2. **Forward only** — K7's walk is forward-only and K9 matches it; the hi
   backward (K8's kernels, 270 ms/scan at rank 2048) is untouched and becomes
   the dominant scan cost after this stage. `check_no_grad` refuses a graph
   through every K9 entry point.
3. **No whole-arm run** — the arm-level numbers are projections from the
   banked K8 receipts' own accounting (stated in the receipt), not measured
   steps; the natural follow-up is the C3 saturation probe, now affordable.
4. **The rank ≤ 128 paths are untouched** — the distributed walk REFUSES the
   on-chip register budget by name and `k9_auto_walk` routes those ranks to
   the existing K7 path; K6's registered wrapper remains the fastest route
   there and nothing in K9 competes with it.
5. **The capture cost is real** (~1 s per shape/snap/policy) and is amortised
   only by scan-per-step schedules; a shape zoo would pay it per distinct
   shape — the K5B slot discipline bounds the memory (16 slots, FIFO).

## 6. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k9_distributed_walk.py -q   # 14 passed
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/ -q                           # 224 passed
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k9_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k9_gates.json
# gate 0 (already banked; re-runnable):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k9_barrier_economics \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k9_barrier_economics.json
```

## 7. Recommended next step

Take the C3 saturation probe (the K8 verdict's named follow-up) at rank 2048
with the K9 walk: a probe whose stream forces the store past 128 live rows —
the one regime where the operator's transformer-parity target is testable in
value — now costs ~3.3 h per arm instead of ~7.4 h. The engineering twin is
the hi backward at the same distributed granularity (the dominant scan cost
after this stage), which needs the same span-level re-grid K8's backward
already has and the same launch-boundary barrier this stage receipts.
