# K1 — the schedule-decoded state scan, as one fused Triton kernel

Stage 1 of the X45 matrix store's kernel lane, forward only.
Companion to `RESULT_MATRIX_EMENDER.md`; that document's §6 is the number this stage
attacks (matrix `matrix_ops` block: **21.7 ms of device time inside 113 ms of wall**).
Nothing here is a whole-arm result: no 2700 s arm was run and no tokens/s number is
claimed.

**Headline, as narrowly as the evidence supports.**

* One program per `(row, head, slot_block)` walks a chunk's 256 steps in registers with
  round-per-operation arithmetic and emits the injection list and the exit channels.
  **One kernel launch** replaces the eager chunk's **87**: state, liveness and therefore
  every death boundary are **bit-for-bit identical** to the sequential oracle on real
  routed schedules with all five ops (gate A), and reads/content sit at **0.0–1.66×** the
  measured fp32 sequential floor (gate B).
* The drop-in path (`chunk_scan(..., fused=True)`) is **3.6× fewer kernels** (1 392 → 384
  per 2048-token block call) and **2.3× faster** (24.06 → 10.59 ms, same boundary,
  forward only). **The remaining 23 kernels per chunk are the read reduction**, which is
  stage K2; K1's own stage is done.
* Kernel *occupancy* per chunk falls 0.302 → 0.046 ms (6.6×): the eager path's
  `[steps × steps]` coupling matrix, triangular solve, gather and scatter-add are gone,
  not merely fused.
* Default-off. `chunk_scan`/`chunked_scan`/`routed_chunk_scan`/`MatrixMixer` take
  `fused=False` and the eager reference **is** the gate-B oracle, so the flag cannot
  hide a divergence from itself.
* One real bug found by gating and fixed: every slot block stored each step's injection,
  so a block that did not own the addressed slot raced the owner with zeros and the
  **reads were run-dependent**. Exactly one block owns a step's injection now.

---

## 1. What was built

| file | role |
| --- | --- |
| `k1_schedule_scan.py` | the kernel, `k1_state_scan` (one chunk, one launch) and `k1_chunk` (the `chunk_scan` drop-in) |
| `test_k1_schedule_scan.py` | 20 tests = gates A, B, E and the flag |
| `k1_gate_receipt.py` | measures gates A and B into `receipts/k1_gates.json` |
| `bench_k1_scan.py` | measures gates C and D into `receipts/k1_schedule_scan.json` |
| `scan.py`, `model.py` | the `fused` flag on `chunk_scan` / `chunked_scan` / `routed_chunk_scan` / `MatrixMixer` / `MatrixModel`, default off |

### 1.1 Parallelism and what lives in registers

One program per `(row, head, slot_block)`, where `row` indexes `(batch, head)`. At the
brief's shape that is `8 × 2` rows and `128 / BS` blocks of slots (`BS ∈ {8, 16, 32}`,
autotuned on `R`). Each program holds a `[BS, V]` tile of the store in registers and
walks the chunk's `tokens × lanes` steps **in order**, because the liveness channel is
per-slot sequential: which lane touches a slot last is what decides whether it is live at
the chunk exit. The schedule arrives as scalar data with its own strides (no host-side
permute, no copy of the decision channel) and the addressed slot is a *predicate against
the lane index*, never a materialised `[rows, steps, rank]` mask or trajectory.

### 1.2 Operation order — and why it is exact

Per token the reference (`algebra.sequential_matrix_scan`) applies: decay every row, then
every lane in order, `S[j] = erase ? v : S[j] + v`. The walk performs the same operations
with the same rounding:

```
v    <- mul_rn(v, lam)                 every step, whole tile
v    <- erase ? p : add_rn(v, p)       at the addressed slot only
live <- live_after                     at the addressed slot only
inj  <- erase ? sub_rn(p, v) : p       the injection, at the addressed slot only
```

`mul.rn.f32` / `add.rn.f32` / `sub.rn.f32` are inline asm, so the compiler cannot contract
`v*lam + p` into an FMA. That is not defensive style: `x44/RESULT_CHUNK_SCAN.md` §6.2
records the concrete failure — a Triton scan that let the compiler contract the same
expression differed from the reference by 1 ulp, one slot crossed `free_threshold` a token
early, and the read moved from 3.0e-6 to 5.1e-4. This store has no `free_threshold` (see
§1.4), but it has the same class of channel: the entry liveness decides which rows the
next chunk's router can address.

### 1.3 The injection list

For every step the kernel emits the relaxation payload and the step's cumulative log
decay, so the read can be reduced *outside* the walk exactly as `scan.chunk_scan` does it:

```
S_s = exp(gc_s) H + sum_{u<=s} exp(gc_s - gc_u) k_{j_u} d_u^T ,  gc_u = sum_{w<=u} log lam_w
o_t = exp(gc_read) w_t^T H + sum_{u<read} exp(gc_read - gc_u) w_t[j_u] d_u
```

`delta` is `(batch, heads, steps, V)` and `log_gate` is `(batch, heads, steps)`; the
injection list is step-major, so it is dense with exact zeros at steps that write nothing
(compacting it would need an atomic prefix sum and would put the read's own matmul back
together from a sparse list — no gain). `_reads_from_injections` performs the reduction
in torch **in the reference's own expression**; that reduction is the 23 remaining kernels
per chunk and is stage K2's target, deliberately not attempted here.

### 1.4 The brief's terms against this algebra (a mapping, not an invention)

The task brief was written partly in the **tape** store's vocabulary. The matrix store's
reference has no such channels, and inventing them would have changed the semantics the
gates are defined against, so the mapping is stated instead:

| brief | this algebra |
| --- | --- |
| "strength `s *= pop_factor`, `s *= decay`, death at `s <= free_threshold`, APPEND sets `s = 1.0`" | the store has **no separate strength scalar**: `strength_j = ‖S[j]‖` (`read.row_strength`), so it decays because the *row* decays and a POP (`S[j] := 0`) removes it in the same step. There is no `pop_factor` and **no threshold death** — death is the schedule's `live_after=False`, which is a decision, exactly as `state_dependent_death_not_affine` requires. The kernel therefore emits `‖S[j]‖ · live` as the exit strength channel (§2.3), derived from the same registers the exit state is stored from, and `read.py` stays the single definition. |
| "age 0 on APPEND" | the matrix store has **no age channel** at all (no ring, no LRU, no `usage`); nothing consumes one. Not invented. |
| "pop factor / decay factor / append gate / beta" | `erase` (bool), `decay` (the learned diagonal gate), and the payload; the APPEND/CORRECT/POP distinction is the `op`/`erase`/`live_after` triple, and `beta` does not exist because the coefficient form is an exact overwrite (`S[j] = v'`), not a blend — the blend lives in *MERGE's payload*, computed by the router. |
| "per-token death codes (pop/decay)" | pop deaths exist and are emitted through the liveness channel; decay deaths do not (no threshold). Gate A therefore holds the *liveness channel* bit-exact, which is the whole of the death information this algebra has. |

Everything else in the brief was implemented as written: one program per
`(row, head, slot_block)`, the chunk in registers, no `[rows, C, rank]` masks, the
injection list plus the exit state, forward only, default-off behind a flag, Python
reference retained.

---

## 2. Gate A — the scalar channels are bit-exact

`python -m pytest experiments/expressivity_tasks/x45_matrix_emender/test_k1_schedule_scan.py -q`
→ **20 passed** (all gates). The module needs CUDA; without a device every test skips.

Oracle: `algebra.sequential_matrix_scan`, the per-token semantics of record. Tolerance:
**none** — `torch.equal`.

### 2.1 On a real routed schedule, with all five ops

`receipts/k1_gates.json`, fixture: batch 2, 64 tokens, 2 heads, rank 5, `d_v` 8, two
lanes, decay 0.95, seed 23, routing = `schedule.route_chunk` on real state with the value
stream drawn from 6 distinct directions (that is what makes MERGE reachable at all — a
uniform random stream never merges).

Op histogram over 512 lanes: `none 295, append 26, correct 125, pop 32, merge 34`.

| statement | measurement |
| --- | --- |
| state after every chunk of a 4-boundary walk | **bit-identical** (`torch.equal`, 4/4) |
| liveness after every chunk boundary | **bit-identical** |
| liveness at *every* token (one call per token, 24 tokens) | **bit-identical** (a separate test) |
| death codes derived from the schedule alone (`algebra.update_liveness`) | **bit-identical** to the walk's exit liveness |
| dead slot ⇒ exactly zero row | holds at every boundary (a POP writes exact zero) |
| strength channel vs `read.row_strength(state, live)` | 4.8e-7 absolute, on rows that are themselves bit-identical — i.e. the norm's reduction order, nothing else |

### 2.2 Why "bit-exact" is the right instrument and not a flourish

The only thing that consumes liveness is the next chunk's router, and it addresses rows
by `argmax` over a cosine score with masked-out dead rows. A slot that dies one token
early or late changes **which row a CORRECT writes** and therefore the trajectory, not by
an ulp but by a decision. That is `state_dependent_death_not_affine` in its runtime form,
and it is why the walk reproduces the reference's rounding rather than merely its formula.

### 2.3 The strength channel

`‖S[j]‖ · live` is emitted per slot; it is *derived from the bit-identical rows*, so it
agrees with `read.row_strength` to the fp32 reduction order (4.8e-7 absolute above) and
can flip nothing downstream (nothing thresholds it). `read.py` remains the single
definition of the read; the emitted channel is the exit diagnostic and the next chunk's
router input shape.

---

## 3. Gate B — content and reads at the measured floor

Floor: `test_matrix_emender.sequential_noise_floor` on the *same* fixtures (two reduction
orderings of the sequential math, plus fp32 against a float64 fold); the fused path is
held to `NOISE_FACTOR = 8×` it, the discipline the chunked path already lives under.
Measured floor on the routed fixture: **5.76e-7**, allowance **4.60e-6**.

| chunk | read dev vs oracle | ratio of floor | state dev vs oracle | read dev vs `chunk_scan` | liveness |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0 | 0.00× | 0.0 | 2.4e-7 | equal |
| 2 | 6.0e-7 | 1.04× | 0.0 | 1.4e-6 | equal |
| 3 | 9.5e-7 | 1.66× | 0.0 | 7.2e-7 | equal |
| 4 | 7.2e-7 | 1.24× | 0.0 | 4.8e-7 | equal |
| 16 | 9.5e-7 | 1.66× | 0.0 | 9.5e-7 | equal |

Two things worth reading off that table:

* **`state dev vs oracle = 0.0` at every chunk length** — the exit state of a whole
  sequence driven through the fused path is bit-identical to the per-token oracle's,
  which is a stronger statement than gate B asks for and is what the round-per-op
  arithmetic buys.
* The reads are *not* bit-identical, and the reason is structural rather than sloppy: the
  read is recomposed from the emitted injection list, and an erasing step's injection is
  `p - v` (one rounded subtraction), while the reference's `delta` comes out of a
  triangular solve. The difference is at the fp32 noise floor, 1.66× of it.

Other fixtures (in the test file): the `random_schedule` fixture over chunk
`1, 2, 3, 4, 16`; a chunk length that does not divide the token count; rank 64 and 2048.

**The injection list rebuilds the exit state**: `exp(gc_last) H + Σ_u exp(gc_last − gc_u)
k_{j_u} d_u` from the emission gives the kernel's own exit state to **1.3e-6 on a state
scale of 2.52** (5.2e-7 relative) — the contract K2 will consume.

**Re-routing drift, measured rather than assumed away.** Under `routed_chunk_scan(...,
fused=True)` the router reads the *fused* chunk-entry state, which differs from the
reference's by fp noise, so a decision sitting exactly on a threshold could move.
Measured on the routed fixture: **0 of 512 lane decisions moved**, reads differed by
9.5e-7 on a scale of 5.94. That is one fixture at one seed — it is evidence, not a proof,
and it is the risk to watch when the fused path is promoted to the default.

---

## 4. Gate C — kernel count per chunk

`receipts/k1_schedule_scan.json`, same inputs and the same recorded schedule for every
path (`torch.profiler`, CUDA activity, device-attributed events). Boundary: **one chunk of
128 tokens = 256 steps, forward only, fp32**, schedule and read weights pre-built, entry
state = the reference trajectory's state at chunk 4 (populated store: the first chunk's
read is *exactly zero* by the certified semantics and would be a flattering measurement).

| path | CUDA kernels per chunk | per 2048-token block call (16 chunks) |
| --- | --- | --- |
| `chunk_scan` (eager reference) | **87** | **1 392** |
| `k1_state_scan` (the fused walk only) | **1** | **16** |
| `chunk_scan(..., fused=True)` (drop-in, incl. read reduction) | **24** | **384** |

The eager count is 87 rather than the brief's "~58" because the brief's figure is per
chunk at a different chunk length; 87 is this shape (128 tokens × 2 lanes = 256 steps,
one `[256 × 256]` coupling matrix, one triangular solve, two `delta` reads, a gather, a
scatter-add and an `amax`).

**K1 removes 86 of 87 kernels in its own stage.** The 23 that remain in the drop-in path
are `_reads_from_injections` — a handful of small index ops plus two matmuls over the
emitted list. That is K2, and it is now the whole of the remaining launch count.

---

## 5. Gate D — isolated K1 timing

Boundary, stated once and repeated in the receipt: **one chunk of 128 tokens (256 steps),
forward only, fp32, batch 8 × heads 2 × rank 128 × `d_v` 128**; the schedule and the read
weights are inputs; the chunk-entry state is the reference trajectory's state after 4
chunks (store populated); device = CUDA events bracketing the call, wall = `perf_counter`
around the call plus `synchronize`; 2 warm-up + 5 measured reps.

### 5.1 `matrix_ops` routing (append/NONE in the timed chunk; see §4 of the receipt)

| path | kernels | device mean | device p50 | wall mean | wall p50 | kernel occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| `chunk_scan` eager | 87 | 1.581 ms | 1.556 ms | 1.548 ms | 1.527 ms | 0.302 ms |
| `k1_state_scan` | 1 | 0.217 ms | 0.217 ms | 0.185 ms | 0.185 ms | **0.046 ms** |
| `chunk_scan(fused=True)` | 24 | 0.683 ms | 0.684 ms | 0.653 ms | 0.654 ms | 0.115 ms |

### 5.2 `matrix_append` routing (a write on every step — the expensive case)

| path | kernels | device mean | device p50 | wall mean | wall p50 | kernel occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| `chunk_scan` eager | 87 | 1.519 ms | 1.531 ms | 1.488 ms | 1.499 ms | 0.274 ms |
| `k1_state_scan` | 1 | 0.256 ms | 0.252 ms | 0.186 ms | 0.182 ms | **0.084 ms** |
| `chunk_scan(fused=True)` | 24 | 0.677 ms | 0.680 ms | 0.646 ms | 0.650 ms | 0.153 ms |

### 5.3 The whole 2048-token block call (16 chunks carrying the state)

Boundary: the 2048-token block call, forward only, fp32 — `bench_blocks.py`'s block row
adds the backward and is a *different* boundary, so those 113.0 ms / 21.7 ms numbers are
not differenced against these.

| routing | path | kernels | wall mean | wall p50 | device mean | occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| `matrix_ops` | eager | 1 392 | 24.03 ms | 23.97 ms | 24.06 ms | 4.45 ms |
| `matrix_ops` | fused drop-in | 384 | **10.56 ms** | 10.59 ms | 10.59 ms | 1.93 ms |
| `matrix_append` | eager | 1 392 | 24.23 ms | 24.22 ms | 24.26 ms | 4.38 ms |
| `matrix_append` | fused drop-in | 384 | **10.89 ms** | 10.91 ms | 10.93 ms | 2.45 ms |

**How to read the three columns.** "Device" brackets the whole call with CUDA events,
so for a single-launch call it contains the host dispatch latency (the GPU is idle
waiting); "occupancy" is the profiler's summed kernel execution time. The eager chunk has
**0.302 ms of real device work inside 1.548 ms of wall** — the launch count *is* the cost,
which is the `RESULT_MATRIX_EMENDER.md` §6 story. K1's own stage is 0.046–0.084 ms of
occupancy and one launch; the drop-in path is then dominated by the 23-kernel read
reduction (0.115 ms of occupancy, 0.65 ms of wall). **K1 is no longer the launch-bound
stage; the read is.**

---

## 6. Gate E — pathologies

All of these are in `test_k1_schedule_scan.py`, all compared to the oracle with
`torch.equal` where the semantics are exact.

| pathology | result |
| --- | --- |
| a slot that dies and is re-allocated inside one chunk (POP then APPEND, same slot) | state and liveness bit-identical for every chunk length 1, 2, 3, 6; unwritten rows exactly zero; the reallocated slot live at the exit |
| a whole chunk of POPs | state **exactly zero**, nothing live, bit-identical to the oracle. The eager reference reaches the same place by cancelling exponentials and leaves ~1e-8 behind — the fp floor, and the reason the oracle is the gate here |
| a chunk with no writes | pure decay; injection list exactly zero; state and liveness bit-identical; reads exactly zero |
| chunk length not dividing the token count (13 tokens, chunk 2/3/4/5/13) | state and liveness bit-identical at every boundary; reads within 1e-5 and within 8× the measured floor |
| rank at both ends of the precompiled set (64 and 2048) | bit-identical; at rank 2048 the grid is 64 slot blocks of 32, so the block mask and the ownership rule are exercised at a full store |
| determinism across slot blocks | two runs bit-identical on every channel, at rank 64 (several blocks) |

---

## 7. What did not work, and what is honestly not done

1. **Every slot block wrote the step's injection.** The first version stored
   `tl.sum(delta_tile, 0)` at the step's delta row from *every* block. Only the block that
   owns the addressed slot has a non-zero contribution, so the other blocks raced the owner
   with **zeros**, and reads came out different from run to run (the bug surfaced as a
   0.865 read deviation at chunk ≥ 2 that disappeared when the same call was repeated).
   Fixed by an ownership predicate — exactly one block stores a step's injection, and only
   block 0 writes the shared `gate` channel. `test_the_fused_path_is_bit_deterministic_
   across_slot_blocks` now guards it. **This is the class of bug the gates exist for**: it
   is invisible at chunk length 1 and at a zero entry state.
2. **`tl.arange` needs a power of two.** The obvious `tl.arange(0, V)` fails to compile for
   `d_v = 5` (the small test fixtures). The value axis is now a power-of-two tile masked
   down to `V`; at `d_v = 128` the mask is all-true and costs nothing measurable.
3. **`@triton.autotune` keyed on `R` with `BS ∈ {8,16,32}`** recompiles and re-benchmarks
   per rank; that is what the tests pay for (20 tests in ~10 s), and it is why `R`/`T` are
   `do_not_specialize`d so a new sequence length does not trigger a compile. Note that
   `ndm/triton/pin_autotune.py` falls back to `configs[0]` for kernels it has not measured,
   so the arms will run `BS = 8` unless a config is registered; nothing about the math
   depends on `BS`, and the rank-64/2048 tests run multi-block grids.
4. **The read reduction is still torch**, 23 kernels per chunk. Reducing it in a second
   kernel is K2 and is deliberately not in this stage. Two cheap levers are visible and
   unimplemented: hoisting the constant index tensors (`order`, `read_index`, `inside`,
   `before`) out of the per-chunk call would remove ~6 of the 23, and the whole reduction
   is one `tl.dot`-shaped kernel per `(row, head, token_block)`.
5. **No CUDA graph, no `torch.compile`, no backward.** K1 is one eager launch per chunk;
   `MatrixMixer` refuses `fused_scan=True` together with `compile_scan=True` rather than
   composing two opaque paths. The backward of the fused walk does not exist yet; the arm's
   default path keeps autograd through the eager scan. This is stage 1 by scope, not by
   oversight.
6. **The timed `matrix_ops` chunk is write-sparse and has no CORRECT/POP/MERGE.** That is
   the real routing's regime at this seed: the store fills, then most tokens are NONE under
   capacity pressure. `matrix_append` (a write on every step) is measured alongside as the
   expensive case, and it bounds the per-step cost of any write mix, since the walk's cost
   depends on *whether* a step writes, not on which op it is.
7. **Growth was not exercised through the kernel.** Rank growth is a zero-injection at a
   chunk boundary and the kernel walks the full allocated width regardless, so a grown store
   is just a wider `R` (covered by the rank-2048 test), but no growth *wave* was run with
   the fused path on.

---

## 8. Deviations from the brief

1. **The strength/age/free-threshold channel was mapped, not implemented** (§1.4): this
   algebra has no separate strength scalar, no age channel and no threshold death. Gate A
   holds the channels that do exist (state, liveness) bit-exact, which is stronger on the
   channel that decides routing and denies nothing that has a definition.
2. **The injection list is dense (step-major), not compacted** to "≤ C entries per
   row/head"; empty steps are exact zeros. A compacted list needs an atomic prefix sum and
   would force the read to rebuild a dense matmul anyway.
3. **The read reduction stayed in torch.** The brief scopes K1 to the state scan plus the
   injection list, so the drop-in path keeps the reference's read expression; it is the
   remaining 23 kernels per chunk and the reason the drop-in is 2.3× and not 16× faster.
4. **No 2700 s arm, no tokens/s claim, no registered runner change.** `x42_pile_scale/
   lattice.py` is untouched; `fused=False` is the default everywhere.

---

## 9. Commands

```bash
python -m pytest experiments/expressivity_tasks/x45_matrix_emender/ -q            # 53 passed
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
python -m experiments.expressivity_tasks.x45_matrix_emender.k1_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k1_gates.json
python -m experiments.expressivity_tasks.x45_matrix_emender.bench_k1_scan \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k1_schedule_scan.json
```

## 10. Recommended next step

**K2: the read reduction as one kernel**, because it is now the whole remaining launch
count (23 of 24 kernels per chunk) and more than half of the drop-in path's device and
wall time. It is a favourable shape: per `(row, head, token_block)` it is
`o = exp(gc_read)·(w_t H) + coeff · (w[j_u] ⊙ d_u)` — two `tl.dot`s over the emitted list,
which is already in the layout it needs, with no new semantics. The same conservation law
applies and is already tested: the emitted list must rebuild the exit state. Only after
that does a fused backward or a CUDA graph make sense; both stay out of scope here.
