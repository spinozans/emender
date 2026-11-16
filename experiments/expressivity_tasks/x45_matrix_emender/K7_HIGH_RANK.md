# K7 — THE HIGH-RANK STORE: the rank ceiling removed

Stage 7 of the X45 kernel lane, run in parallel with K6 (its files, cards and the
CMA-ES lane untouched). The one-program-owns-the-store designs refused the growth
snap set {1024, 2048} **by name** — `hostloop_free.persistent_available` (8 192
register floats, so the persistent block scan stops at rank 512 ≈ 4×
transformer-order), K4's fused backward through the same gate, and K6's per-token
kernel at `rank × d_v` ≤ 16 384 (rank 128 at `d_v` 128), all falling back to slower
twins. This module removes the ceiling on **both** sides of the scan with zero edits
to any registered semantic: the read becomes a **slot-blocked chunked value scan**
whose softmax denominator spans slot blocks under the Lean truncation law, and the
per-token **decision walk** moves to a global-tiled state (the design fork resolved
in [`DESIGN_HIGH_RANK_WALK.md`](DESIGN_HIGH_RANK_WALK.md), written first).

**No whole-arm result is claimed here.** Every number is one chunk, one block call
or one (layer, window) scan, forward only, fp32, one RTX 6000 Ada (48 GB), seed-stated.

**Headline, as narrowly as the evidence supports.**

* **Rank 2048 executes.** Both new paths run the full snap set — the chunked value
  scan (`k7_block_scan_routed`, 16 chunks × 7 kernels) and the per-token decision
  walk (`k7_sequential_walk`, ONE launch per scan) — with the by-name refusal
  receipted alongside (`persistent_available(2048, 128) == False` in the same JSON).
* **Gate A (the chunked read): state and liveness bit-exact at every rank** (K1 is
  the walk, shared with the k23 twin), reads at **0.28–0.45× the measured fp32
  floor** vs the k23 twin given identical torch weights — a pure reduction-order
  deviation — and at **5.77× the floor** vs the per-token oracle on the canonical
  routed fixture (the same 6.59× the k23 twin itself measures at that chunk length).
* **The split-softmax follows the Lean law, as a number**: the in-kernel weights
  land **1–2× the fp32 floor of `read.read_weights`' own fp64 twin** at every rank
  (max deviation 1.9e-8 at rank 128 down to 1e-9 at rank 2048), and the
  obstruction fixture (two equally-attractive rows in *different* slot blocks)
  shows the full-population value ~0.025 where a per-block renormalisation would
  give ~0.46 — an O(1) difference the gate would catch (`truncation_sharpens_survivor`).
* **Gate B (the walk): 0 flips in 279 040 compared lanes at rank 2048** — decisions,
  payload lanes, read WEIGHTS and pre-decision liveness `torch.equal` against the
  K7 oracle (written to the kernel's reduction order, the K6 precedent), the exit
  state and liveness bit-equal, the read output at **1.2–1.6× its measured floor**,
  255 registers, **0 spills**. The oracle itself is pinned to the K6 draft
  registration's oracle at rank 128: **0 decision flips, weights, liveness and
  state bit-equal**. Crafted fixtures force every one of the five ops (MERGE on both
  lanes, POP, NONE, CORRECT, APPEND), all bit-exact.
* **The honest walls**: the walk is 28.9 / 191.2 / 374.3 ms per (layer, window) scan
  at rank 128 / 1024 / 2048 (K6's register-resident kernel: 26.5 ms at rank 128 —
  and it refuses beyond). The chunked k7 block call at rank 2048: 7.5 ms given a
  weights channel, 13.1 ms computing the weights in-kernel — **not faster than the
  k23 twin (5.4 ms), which was never refused**; what K7 removes is the ceiling, not
  the twin. Stated with the memory arithmetic.

---

## 1. What was built

| file | role |
| --- | --- |
| `k7_high_rank_store.py` | Part A: `_k7_scores_kernel`, `_k7_mass_kernel`, `_k7_weights_kernel` (the split softmax), `_k7_inter_kernel`, `_k7_intra_kernel` (the re-gridded K3/K2), `k7_read_weights`, `k7_read`, `k7_chunk(_routed)`, `k7_block_scan(_routed)`; Part B: `_k7_walk_kernel`, `k7_sequential_walk`, and the K7 eager oracle `sequential_oracle` |
| `DESIGN_HIGH_RANK_WALK.md` | the Part-B design fork, written BEFORE the build; option (ii) refused from the receipts |
| `test_k7_high_rank.py` | 13 tests: gates A and B, the Lean-law fixture, op-branch coverage, the K6-oracle pin, determinism, the refusals |
| `k7_gate_receipt.py` | the receipts into `receipts/k7_high_rank.json` |
| `K7_HIGH_RANK.md` | this document |

### 1.1 Part A — the slot-blocked chunked value scan

The read of the registered chunked arm is K2+K3's two kernels; their rank axis was
an in-program loop, so at rank 2048 the grid did not grow with the rank. K7 re-grids
them: the **inter** kernel's grid is `(row, token-block, slot-block)`, each program
holding a rank tile (`BR` = 64 slots × the value tile, 8 192 floats — the lane's
measured no-spill tile), partials combined in **one fixed-order reduction** (the
defined balanced tree over the slot-block axis — bit-reproducible, no atomics, one
writer per element). The **intra** kernel is re-gridded over **step blocks** (see
§6.1: blocking the intra term by the steps' addressed slot multiplies its dot work by
the slot-block count — measured, then fixed). K1 is reused verbatim for the state
walk — it is already slot-blocked at any rank and bit-exact.

**The read softmax moves into the kernels with the grid, under the Lean law**
(`formal/lean/OPERATOR_ALGEBRA_STATUS.md`, EMENDER_ATTENTION_ALGEBRA: per-operator
normalisation does NOT make truncation harmless — `truncation_sharpens_survivor`
doubles a survivor's coefficient — and the exact resolution is to *normalise by the
full-population exact mass*). Implemented, stated exactly once:

```
per slot block:  m_b = defined tree over the tile of exp(score − gmax)   (explicit scalars)
one reduction:   denom = ONE defined balanced tree over the slot-block axis
                  (bit-identical to the K6-defined whole-axis tree: a tile's internal
                   tree plus one tree over the tile results IS the whole-axis tree,
                   zeros padding exactly)
one normalisation: w_j = mul.rn(div.rn(exp(score_j − gmax), denom), strength_j)
                   — the registered expression, ONE division by the full mass,
                   never a per-block renormalisation
```

The max combine is exact (max is order-free). `read.py` stays the single definition
of the weights; the gate holds the kernel channel to `read_weights` at the measured
fp32 floor of that channel against its own fp64 twin — torch's softmax order is not
reproducible from a kernel (the banked K5A probe), so this gate is floors, never
bits.

**The gate wording, honestly.** The brief asks for "bit-parity vs the existing k23
chunked path … (identical reads, same floors discipline)". Bit-identical READS are
impossible by construction once the rank axis is split across programs — a split
fp32 reduction is a different, equally valid order, and no path in this lane has
bit-exact reads (K23's own gate A is 1.65× a measured floor; K6's read is at 0.25×).
The gate implemented is the lane's established form, with the strongest available
extra evidence: **state and liveness `torch.equal`** (both paths share K1), **reads
within `NOISE_FACTOR` × the measured sequential floor vs the k23 twin *given
identical torch weights*** (which isolates the reduction-order change: 0.28–0.45×
floor) and **vs the per-token oracle** (5.77× on the canonical fixture — the same
number the k23 twin itself measures there, because at one slot block the k7 kernels
*are* k23's), plus the in-kernel-weights end-to-end form at the weights' floor.

### 1.2 Part B — the high-rank sequential decision walk

The fork is resolved in `DESIGN_HIGH_RANK_WALK.md` (written first): option (ii) —
compressed decision state — is refused because the strength recurrence is not exact
in the defined fp32 order (DECAY's multiplicative norm is a real-arithmetic fact,
not a defined-order one), APPEND breaks it outright (`||S_j + p||` is not a function
of the two norms), and the query-alignment term admits no recurrence at all — only a
new registered semantic (a sketch) could approximate it, which is K5A's
~48%-bit-equal failure class, not a parity gate. **Option (i) — the global-tiled
state walk — is chosen**: the store lives in global memory, one program per
(batch·head) row walks the window, and per token sweeps the store in slot tiles:

* **sweep A** (read-only): per-row defined value-axis tree folds — norms, read
  scores, decision cosines — into whole-rank scratch channels, plus the running
  softmax max (exact);
* **the decision phase**: K6's own code on whole-rank register vectors — the
  softmax denominator is the defined whole-axis tree **in one program** (no split:
  the weights channel is bit-exact against the oracle), the registered selection
  chain, the first-index tie rules, the MERGE blend from two gathered rows;
* **sweep B**: the read numerator (native per-tile sums, sequential across tiles —
  the one order no torch expression reproduces, held to the measured floor: the
  K1/K23/K6 corner, and the read feeds no decision), then DECAY in place, and the
  capture of the lane's target row (the exact single-nonzero gather);
* **the lanes** (after the DECAY, the registered order): one row write plus the
  MERGE-source zeroing; a NONE op writes nothing (its payload lane is exactly zero
  and its target is provably live — see the kernel comment).

Decisions, payload lanes, read weights, pre-decision liveness, the exit state and
the liveness are **bit-exact** against `sequential_oracle` (the K7 eager oracle,
written to the kernel's reduction order — the K6 precedent), and the oracle is
**pinned to the K6 registration's own oracle** at rank 128: 0 decision flips,
weights/liveness/state bit-equal. The read output is at 1.2–1.6× the measured
fp32-vs-fp64 floor.

## 2. Gate A — the chunked read (receipt `gate_a_chunked_value_scan`)

| fixture | floor | k7 vs k23 (same torch weights) | × floor | k7 vs oracle | state / liveness |
| --- | ---: | ---: | ---: | ---: | --- |
| canonical routed (rank 5, all five ops) | 5.79e-7 | 9.5e-7 | — | 3.34e-6 = **5.77×** | bit-equal / equal |
| routed chunk rank 128 (2 slot blocks) | 5.37e-7 | 1.49e-7 | 0.28× | 7.5e-8 | bit-equal / equal |
| routed chunk rank 512 (8) | 3.32e-7 | 1.34e-7 | 0.40× | 3.7e-8 | bit-equal / equal |
| routed chunk rank 1024 (16) | 3.81e-7 | 1.56e-7 | 0.41× | 2.4e-8 | bit-equal / equal |
| routed chunk rank 2048 (32) | 3.12e-7 | 1.42e-7 | **0.45×** | 2.2e-8 | bit-equal / equal |

The in-kernel weights channel vs the registered `read_weights`: 1.9e-8 at rank 128,
1e-9 at rank 2048 — against that channel's own fp64 floor of 2e-8 / 2e-9 (i.e. at
the floor). The end-to-end routed chunk (in-kernel weights) vs the k23 twin fed the
router's torch weights: 1.19e-7 … 1.71e-7, inside the allowance at every rank. The
Lean-law fixture: two ±e₀ rows in different slot blocks carry the whole mass;
the kernel returns the full-population ~0.025 where a per-block renormalisation
would return ~0.46 — the obstruction, receipted as a test.

## 3. Gate B — the walk's decisions (receipt `gate_b_sequential_walk`)

| rank | compared lanes | flips | state / live | read floor | read dev | × floor | regs / spills |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 128 | 33 280 | **0** | bit-equal | 1.19e-7 | 1.79e-7 | 1.5× | 255 / 0 |
| 1024 | 147 968 | **0** | bit-equal | 3.7e-8 | 4.5e-8 | 1.2× | 250 / 0 |
| 2048 | 279 040 | **0** | bit-equal | 2.3e-8 | 3.7e-8 | 1.6× | 255 / 0 |

The oracle pinned to the K6 registration's oracle at rank 128: **0 decision flips,
weights bit-equal, live_pre bit-equal, state bit-equal.** The receipt's random
fixtures saturate in APPEND (the fresh-store regime); the *tests* force the other
four ops with crafted stores (a repeated-direction store forces MERGE on both lanes,
a full store with an orthogonal stream forces POP, a full store whose every row sits
at cosine 0.3 to the payload forces NONE) — every branch bit-exact. All-dead token 0:
read and weights exactly zero (the certified zero attention), at rank 2048.

## 4. The refusal removed (receipt `refusal_removed`)

Block-call boundary: batch 8, 2 048 tokens, 2 heads, `d_v` 128, chunk 128,
synthetic five-op schedule (the rank sweeps' own fixture), 5 reps, wall
(`perf_counter`+sync) and device (CUDA events):

| rank | persistent refusal | k7 routed block (weights in-kernel) | k7 given-weights block | k23 twin block | torch `read_weights` per block |
| ---: | --- | ---: | ---: | ---: | ---: |
| 512 | no (the last rank it fits) | 12.19 ms | 6.14 ms | 5.31 ms | 5.58 ms |
| 1024 | **yes, by name** | 11.90 ms | 6.58 ms | 5.43 ms | 5.47 ms |
| 2048 | **yes, by name** | 13.15 ms | 7.50 ms | 5.37 ms | 5.58 ms |

**What this says, honestly.** The k23 twin was *never* refused (its rank axis is an
in-program dot loop; `NO_HOST_LOOP.md` gate E already ran it at rank 2048) — the
refusal this module removes is the persistent/by-name one that caps the
one-launch designs, and the k7 block scan executes the snap set that refusal
blocked, at ~1.2–1.4× the twin's wall for the given-weights form. That premium buys
the grid whose rank axis is a *grid* axis (parallelism that scales with rank instead
of one program's serial loop), the in-kernel split-softmax (the registered weights
channel produced on-device at any rank under the Lean law — the routed form,
13.1 ms, replaces the twin's 5.4 ms + torch's 5.6 ms of host-launched weights), and
the any-rank guarantee. The memory arithmetic, per chunk (T_c = 128, receipted):
scores/strength/weights channels 4.19 MB each, inter partials 8.39 MB, intra
partials 4.19 MB, state 4.19 MB — ~29 MB of transient channels per chunk at rank
2048, against a 48 GB card.

**The walk's scan walls** (one launch per scan, 5 reps):

| rank | wall / device | traffic per scan | tok/s (scan only) | the K6 register kernel |
| ---: | ---: | ---: | ---: | --- |
| 128 | 28.9 / 28.9 ms | 6.4 GB | 1.13 M | 26.5 ms (fits; refuses beyond) |
| 1024 | 191.2 / 191.2 ms | 51.5 GB | 171 k | refused → eager oracle |
| 2048 | 374.3 / 374.3 ms | 103.1 GB | 88 k | refused → eager oracle |

The walk is fold-bound, not bandwidth-saturated (103 GB in 374 ms ≈ 275 GB/s
effective): the sweeps are dependent defined-order tree folds, the same latency
class as K6's 13 µs/token register walk. That is the price of the registered
semantics at 16× the state, stated rather than optimised away.

## 5. Growth readiness (design requirement, NOT wired)

Both parts read every shape from the tensors at call time; the only rank-derived
constexprs are `RP`/`SB`/`NT`, so **each snap of `keys.ORTHONORMAL_RANKS`
(64 … 2048) is its own cached kernel binary** and a growth event that widens the
rank at a window boundary re-dispatches to the next snap's kernels — or to its
pre-captured graph, if a future stage captures one per snap — with the new rows
exact zeros (the prefix-reveal invariant the chunked arm's growth already relies
on). The eager boundary alternative (re-dispatch per segment) needs no capture at
all and is what the current wrappers do. No growth wave is wired, by scope.

## 6. What failed, and what is honestly not done

1. **The intra kernel's first grid was wrong, and the measurement caught it.** The
   brief's grid `(row, token-block, slot-block)` applied to the intra term
   multiplies its dot work by the slot-block count — every step belongs to exactly
   one slot tile, so all but one program zero the same coefficient matrix. Measured:
   10.86 ms vs the twin's 5.38 ms at rank 2048 (2×). The intra term's natural
   decomposition is the **step** axis: re-gridded over step blocks (k23's
   expressions unchanged, one fixed-order tree over the per-block partials), the
   wall fell to 7.50 ms. The inter kernel keeps the brief's slot-block grid — the
   inter term *is* rank-decomposable with no waste. This is the one deviation from
   the brief's grid wording, and it is a measurement, not a preference.
2. **The first walk kernel was nondeterministic, and determinism testing caught
   it.** Two runs of the same inputs differed at rank 128 (12 slot flips): the
   in-place global state and the scratch channels created cross-warp store→load
   hazards — one warp's global store is not guaranteed visible to another warp's
   subsequent load without a CTA barrier (Triton inserts none for global
   store→load). Fixed three ways: `tl.debug_barrier()` between the scratch writes
   and reads and at the token boundary; the lane's target row is *captured* in
   sweep B (the exact single-nonzero gather) instead of re-loaded after the decay
   stores; sweep B is deliberately not software-pipelined (the pipeliner's
   load/store alias analysis is not a thing to bet the walk on). Determinism is now
   asserted across two runs in the tests, at every gated rank.
3. **The walk's per-token cost is fold-bound**, and the design doc's pure-bandwidth
   estimate (~150 ms at rank 2048) was a floor, not the wall — corrected there with
   the measured 374 ms and the reason.
4. **The random gate-B fixtures saturate in APPEND** (the fresh-store regime — the
   sweep's own finding at K6, inherited). Op coverage comes from the crafted
   fixtures in the tests; the receipt records the histograms honestly.
5. **No backward, forward only** — `check_no_grad` refuses a graph through every
   K7 entry point, exactly as K1/K23/K6 do; the registered chunked arm still
   trains on the eager path, untouched.
6. **No whole-arm measurement and no 2700 s claim** — the K6 probe's result (the
   per-token arm beats the dial's best) is the motivation for a high-rank arm, and
   the honest whole-model number (tokens/s at the 150 M board with rank ≥ 1024) is
   a later stage's to take, gated on the operator's sign-off of the sequential
   registration.
7. **The k7 chunk path is not the fastest chunked read at bench shapes** (§4) —
   k23 remains the right choice where it fits; K7 is the any-rank ceiling-remover,
   not a speed claim. The `block_scan(path='auto')` dispatch in `hostloop_free.py`
   is untouched (its file is not K7's to edit); a future stage may route ranks >
   512 to `k7_block_scan_routed`.

## 7. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=4,5 scripts/gpu_lease.sh acquire 1 --wait)"
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k7_high_rank.py -q   # 13 passed
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/ -q                       # 185 passed
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k7_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k7_high_rank.json --reps 5
```

## 8. Recommended next step

Take the high-rank arm to the operator's pending decision with the two numbers that
matter: the walk's scan wall (374 ms at rank 2048 — fold-bound, stated) and K6's
banked quality result (the per-token arm beats the dial's best at rank 128). The
natural follow-up is a probe of the sequential arm at rank 1024 under the draft
registration's protocol once the operator signs it off — K7's walk is the only
per-token kernel that can run it — with the tuned-LR confirmation as its twin. The
cheap engineering follow-up is routing `block_scan`'s auto path to
`k7_block_scan_routed` at ranks the persistent kernel refuses, which needs one
line in `hostloop_free.py` and this lane's gate discipline.
