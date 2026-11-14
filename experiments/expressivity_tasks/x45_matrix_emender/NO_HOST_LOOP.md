# No host loop — the matrix store's 2048-token forward in one launch

Stage 3 (and last) of the X45 matrix store's *forward* kernel lane. `K1_SCHEDULE_SCAN.md`
took a chunk from 87 kernels to 1; `K23_READ.md` took the read to 2 more launches and a
block call from 1 392 kernels to 48 — and left the wall/occupancy ratio where it was
(5.16x → 5.69x), because 83 % of the remaining wall was **16 Python dispatch steps of 3
launches each**. This stage removes the Python loop over chunks.

**No whole-arm result is claimed here.** No 2700 s arm was run, no tokens/s is reported.
Every number below is one 2048-token block call, forward only, fp32.

**Headline, as narrowly as the evidence supports.**

* **Option A — one persistent kernel for the whole block** (`persistent_scan`): the store
  of a `(row, value_block)` lives in one program's registers and the program walks all
  2048 tokens in order. **1 launch, 1 kernel, 0.98 ms** at the brief's shape, against the
  48-launch baseline's 5.66 ms — **5.8x** — and the wall/occupancy ratio goes **4.60x →
  1.12x**, past GDN-2's 1.5x.
* **Option B — the same pipeline as one captured graph** (`BlockGraph`): 1 replay per
  block call, **1.27–1.39 ms** at C = 128/256/512, ratio **1.02–1.03x**. Also a ~4x win,
  ~0.3 ms behind A.
* **Option C — a wider chunk** costs nothing to adopt and needs no new kernel: C = 512
  takes the plain loop from 48 launches to 12 and from 5.66 ms to **1.56 ms** (ratio
  1.23x). It is the cheapest 3.6x in the lane.
* **Parity held, and A is the most accurate of the three**: state and liveness
  **bit-exact** against `sequential_matrix_scan` at every rank, read at **0.30–0.96x** of
  the measured fp32 floor (the K2+K3 chunked path is at 1.5–1.8x, and at 2.4–5.8x when the
  chunk is widened). **Nothing about the algebra, the routing or the read changed.**
* **C is not free in the read**: widening the chunk leaves the state bit-exact but moves
  the read's fp32 grouping — 1.65x the floor at C ≤ 16, 3.30x at C = 32, 6.59x at C = 64
  on the lane's canonical fixture (§2.3). Inside the 8x allowance, outside the 1.65x
  headline, so it is stated rather than sold.
* **Rank 2048 cannot use A** (a 1 MB register tile): `value_block` refuses it by name and
  `block_scan(path='auto')` takes the chunked path. That gate is tested, not assumed.
* **The winner is A at rank 64 and 128, B at rank 512** (§6) — and the difference between
  them at rank 128 (0.98 vs 1.27 ms) is smaller than the difference between either of them
  and doing nothing.

---

## 1. What was built

| file | role |
| --- | --- |
| `hostloop_free.py` | `persistent_scan` (Option A), `BlockGraph` (Option B), `chunked_block_scan` (the baseline and Option C), the size gate (`persistent_available`, `value_block`, `warp_count`) and `block_scan(path=…/auto)` |
| `test_hostloop_free.py` | 17 tests: parity against the oracle and against the chunked twin, graph bit-identity and input isolation, the size gate, rank 64/512, chunk 16/128/256, a chunk that does not divide the token count, no-write/POP-only blocks, determinism, the shape audit and the forward-only refusal |
| `bench_hostloop_free.py` | gates C and D into `receipts/hostloop_free.json`: launches, kernels, latency, dispatch, device, occupancy, and parity for every path at rank 64/128/512 |
| `NO_HOST_LOOP.md` | this document |

### 1.1 One interface, three sequencing strategies

Every path takes a whole 2048-token block at the boundary `K23_READ.md` §5.3 used —
`state (batch, heads, rank, d_v)` and `live (batch, heads, rank)` in, the **recorded**
`MatrixSchedule` and read weights `(batch, heads, tokens, rank)` as inputs, the tokens'
`reads` and the exit state out — and returns `(reads, state, live)`. They differ only in
how the sequence is sequenced:

```
chunked_block_scan(state, live, weights, decay, schedule, chunk_tokens=C)
    T/C chunks x 3 launches.  = scan.chunked_scan(fused=True) bit for bit (tested), so the
    numbers below are on the same pipeline K23_READ.md measured.  Option C is this at a
    larger C.

persistent_scan(state, live, weights, decay, schedule)                  # Option A
    ONE kernel.  grid = (batch*heads, d_v/BV).  One program owns every slot of its row for
    a block of BV value columns and walks the sequence token by token:
        read  (pre-write)  ->  DECAY  ->  every lane in order
    No chunk axis, no host loop, no cross-program synchronisation.

BlockGraph(...).capture() / .run(...)                                   # Option B
    The same per-chunk pipeline, captured once with static buffers; one replay per block
    call.  `run` copies the inputs into the static buffers (9 small device copies) and
    returns the static outputs.
```

### 1.2 The asymmetry this lane had to respect

GDN-2's forward is 4 launches for the whole sequence because its chunk axis is a **grid**
axis: `recompute_w_u_fwd_gdn2_kernel[(NT, B*H)]`, `chunk_gdn2_fwd_kernel_intra_token_parallel`
`grid = (B*T, cdiv(H, BH))` (`/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py`
lines ~209–250 and ~930). It can do that because its write
`S <- S(I - beta k kᵀ) + beta v kᵀ` needs **no selection**: every per-token quantity is a
function of the inputs, never of the state trajectory, so chunks are independent.

**This arm's five ops need selection** (which slot to CORRECT / POP / MERGE), and the
router takes that decision against the chunk-entry state (`schedule.route_chunk`), so chunk
`c`'s decisions depend on chunk `c-1`'s exit state. The chunked fold is equal to the
sequential fold only along the *state* axis — `chunked_fold_equiv` and
`combined_chunked_scan` in `formal/lean/EMENDER_OPERATOR_ALGEBRA.pdf` — never along a
parallel one, and `injection_propagation` is what says an in-chunk write has to be carried
as an injection rather than re-derived. Making the chunk axis a grid axis would therefore
be a *different computation*, and this stage did not do it. Instead:

* **A** moves the sequential chain onto the device (a program that owns the whole store can
  walk it with no synchronisation at all);
* **B** dispatches it once (a graph is the device-side form of the host loop — legitimate
  precisely *because* the pipeline is sequential);
* **C** pays the dispatch `T/C` times fewer times.

The routing itself is still 16 host steps in the arm, and it is outside this boundary (§8).

---

## 2. Parity — the gate that had to hold first

Two references, as in every other stage of this lane: the **per-token oracle**
`algebra.sequential_matrix_scan` (the semantics of record) and the **chunked twin**
`chunked_block_scan` at the same chunk length (the kernel comparison). The floor is the
**measured** fp32 sequential floor `sequential_noise_floor` on the same fixture, never a
fitted constant; the allowance is `NOISE_FACTOR = 8x` it.

### 2.1 The canonical fixture (rank 5, 64 tokens, 2 heads, all five ops)

`routed_fixture()` — real routing, op histogram `none 295 / append 26 / correct 125 /
pop 32 / merge 34` over 512 lanes — floor **5.786e-7**, read scale 5.94:

| path | read max vs oracle | × floor | median | state vs oracle | liveness |
| --- | --- | --- | --- | --- | --- |
| `persistent_scan` (**A**) | **4.768e-7** | **0.82x** | 5.96e-8 | **0.0 (bit-exact)** | equal |
| `chunked_block_scan` C = 8/16 | 9.537e-7 | 1.65x | — | 0.0 | equal |
| `chunked_block_scan` C = 32 | 1.907e-6 | 3.30x | — | 0.0 | equal |
| `chunked_block_scan` C = 64 | 3.815e-6 | 6.59x | — | 0.0 | equal |

`test_persistent_scan_reproduces_the_oracle_on_a_real_routed_schedule` asserts
`torch.equal(exit_state, oracle_state)` and `torch.equal(exit_live, oracle_live)` — not a
tolerance. That is not luck: the kernel applies the recorded schedule in the oracle's own
order (read, then DECAY, then every lane), with `mul.rn.f32`/`add.rn.f32` inline asm so no
FMA contraction can move the trajectory by one ulp, and it does so token by token rather
than through an injection list. The read differs from the oracle only by *reduction order*
(the rank axis in fp32 registers instead of cuBLAS's GEMV), which is why it sits at 0.82x
of the measured floor.

### 2.2 At the bench shape (batch 8, 2048 tokens, 2 heads, rank 128, d_v 128)

Floors measured on the bench shape itself: `matrix_ops` **2.060e-7** (read scale 1.72,
store scale 0.25), `matrix_append` **1.570e-6** (read scale 3.42, store scale 8.31).

| path | `matrix_ops` read vs oracle | × floor | `matrix_append` read vs oracle | × floor | state vs oracle |
| --- | --- | --- | --- | --- | --- |
| `persistent_scan` (**A**) | **1.788e-7** | **0.87x** | **4.768e-7** | **0.30x** | 0.0 (bit-exact) |
| `chunked` C = 16 | 3.576e-7 | 1.74x | 9.537e-7 | 0.61x | 0.0 |
| `chunked` C = 128 (baseline) | 3.725e-7 | 1.81x | 2.384e-6 | 1.52x | 0.0 |
| `chunked` C = 256 | 1.192e-6 | 5.79x | 3.700e-6 | 2.35x | 0.0 |
| `chunked` C = 512 | 1.192e-6 | 5.79x | 3.700e-6 | 2.35x | 0.0 |
| `graph` C = 128/256/512 (B) | identical to its chunked twin, byte for byte | | | | 0.0 |

Liveness is equal for every row, at every rank, and in every case (`live_equal: true` in
the receipt). **Every path's exit state is bit-exact against the oracle** — the chunked
path's state differs from the *eager* path's by ~1.5e-7/2.4e-6 (`state_bit_equal_to_eager:
false`), which is the eager scatter-add's own rounding, and is the same statement
`K1_SCHEDULE_SCAN.md` §3 makes.

The graph rows are byte-identical to their chunked twins because the graph captures the
same launches; `test_the_captured_block_is_bit_identical_to_the_uncaptured_loop` holds that
at C = 16/32/64 and across two replays.

### 2.3 Option C is a numerical choice, not a free knob — measured

The state is bit-exact at every chunk length. **The read is not**, because a chunk length
changes how much of the answer arrives through the entry term and how much through the
in-chunk injection list. Table §2.1 is that curve on the canonical fixture. At the bench
shape the same effect appears as 4.77e-7 (C = 128) → 1.31e-6 (C ≥ 256) for `matrix_ops`,
and the *eager* path shows the effect is the grouping's and not the kernels': the eager
chunked path at C = 16/256/512 differs from itself at C = 128 by **7.5e-9** (`matrix_ops`,
nothing at all for `matrix_append`), i.e. three orders below the fused path's movement.

So: **parity holds at every C** in the sense the gate needs (state and liveness bit-exact,
read inside the 8x allowance of the measured floor, every path equal to its own twin), but
**C = 256/512 buys wall time with read precision** (1.81x → 5.79x of the floor at rank 128,
and 6.59x of the floor at C = 64 on the small fixture). The lane's 1.65x headline holds at
C ≤ 16, not at C = 512. If the last 1e-6 of the read matters, C = 128 with A or B is the
honest choice; if only the ratio matters, C = 512 is free.

---

## 3. Gate C — launches per block call

Boundary: one block call as in §1.1. "Launches" counts what the **host** issues (a graph
replay is one); "kernels" is the profiler's count of device kernels inside the call, which
is why a graph row still shows 30–81.

| path | host launches per block call | kernels per block call | of which copies (the `reads` assembly) |
| --- | --- | --- | --- |
| `chunk_scan(fused=False)` — the eager reference | **1 392** | 1 410 | 18 (16 reads + 2 entry-state clones) |
| `chunked_block_scan` C = 16 | 384 | 512 | 128 |
| `chunked_block_scan` C = 128 (**baseline**) | **48** | **64** | **16** |
| `chunked_block_scan` C = 256 (**C**) | 24 | 32 | 8 |
| `chunked_block_scan` C = 512 (**C**) | 12 | 16 | 4 |
| `BlockGraph` C = 128 (**B**) | **1** | 78 | 16 (+ 9 input + 5 output copies) |
| `BlockGraph` C = 256 (**B+C**) | **1** | 46 | 8 (+ 14 copies) |
| `BlockGraph` C = 512 (**B+C**) | **1** | 30 | 4 (+ 14 copies) |
| `persistent_scan` (**A**) | **1** | **1** | 0 — the kernel writes `reads` in place |

The three pipeline kernels per chunk are `_k1_state_scan_kernel`, `_k23_inter_kernel`,
`_k23_intra_kernel`; the fourth distinct kernel in the chunked rows is
`at::native::direct_copy_kernel` — `chunked_block_scan` assembling `reads` from the
per-chunk results (`reads[:, :, start:stop] = chunk_reads`). **A pays none of those copies**
because one kernel produces the whole `reads` tensor, and the graph pays them inside the
replay as 16 of its 78 kernels. A production caller that keeps the graph's output must also
clone it out of the static buffers: measured at **+0.02 ms latency / +0.037 ms dispatch**
(`graph_C*_clone` rows), which changes nothing about the ranking but is not assumed away.

---

## 4. The persistent kernel's measured sweep

"One persistent kernel" is not one kernel: the value block `BV` (the value columns one
program owns), the warp count and the loop's software pipelining are all free parameters,
and this is the sweep the defaults came from. All rows: rank 128, `matrix_ops` routing, one
block call, p50 latency in ms (the receipt has the same rows at every rank).

| `BV` | warps | `num_stages` | latency | occupancy | latency/occupancy |
| --- | --- | --- | --- | --- | --- |
| 16 | 2 (rule) | 3 | 1.186 | 1.083 | 1.09 |
| **16** | **4 (default at this tile)** | **3** | **0.978** | **0.873** | **1.12** |
| 16 | 8 | 3 | 1.494 | 1.392 | 1.07 |
| 32 | 4 | 3 | 1.277 | 1.177 | 1.09 |
| 64 | 4 | 3 | 1.547 | 1.445 | 1.07 |
| 128 | 4 | 3 | 2.964 | 2.862 | 1.04 |
| 16 | 4 | **1** | 1.395 | 1.290 | 1.08 |

Three things the table says:

1. **Narrow beats wide at every shape.** The per-token cost of the serial walk is set by the
   rank axis and by the global loads, not by the value width, so a wider block only takes
   registers and program slots away: `BV` = 128 is 3x slower than `BV` = 16 at the same
   work. Hence `value_block` returns `MIN_VALUE_BLOCK = 16` and the value axis is blocked,
   not walked.
2. **`num_stages` is load-bearing (1.4x).** The walk is a serial chain, so each iteration's
   first act is a dependent global load (the weight row, the DECAY gate). Pipelining the
   loads ahead (`tl.range(…, num_stages=3)`) hides that latency; without it the kernel is
   1.40 ms instead of 0.98 ms. This was the single largest win inside A after the loops were
   removed.
3. **4 warps at a 2 048-float tile, 2 below it, never 8** — `warp_count(tile) =
   clamp(tile/512, 1, 4)`, which is what the default takes at each rank (rank 64 → tile
   1 024 → 2 warps, and there 2 and the default are the same 1.01 ms; rank 512 → tile 8 192
   → 4 warps, 2.17 vs 8 warps' 2.40).

**The register budget is the gate, and it is real.** At rank 512, `BV` = 64 is 90.5 ms and
`BV` = 128 is 250.5 ms — that is not slow code, it is spilling to local memory.
`MAX_PERSISTENT_TILE = 8 192` floats (32 KB) is where the curve turns, and rank 2048
(`next_pow2(2048) x 16 = 32 768` floats) is refused before a launch:

```
ValueError: the persistent kernel needs next_pow2(2048) * 16 = 32768 register-resident
floats and its budget is 8192; rank 2048 at value dim 128 does not fit, so it must take
the chunked path
```

`block_scan(..., path="auto")` catches that case by *policy* rather than by exception: it
asks `persistent_available` and takes the chunked path, which is the fallback the gate
exists to make automatic. At rank 2048 the chunked path still gets B and C (1.33–2.0 ms by
extrapolation of the C sweep, not measured here — §8).

There is deliberately **no `@triton.autotune`** on this kernel. `Autotuner.run` keys its
cache on the arguments named in `key`, and the value block is a `constexpr` that a key
cannot see (verified in the installed Triton 3.5.1 source), so one measured config would be
handed to every block shape; worse, tuning *launches the kernel* several extra times on the
real buffers, which is the opposite of what a zero-host-launch path is for. The rule is one
line and this table is its evidence.

---

## 5. Gate D — timing, and the ratio the lane has been chasing

Boundary, stated once: **one whole block call** (batch 8, 2048 tokens, 2 heads, rank 128,
`d_v` 128, lanes 2, fp32, forward only); entry state and liveness in, the recorded schedule
and read weights as inputs, the tokens' reads and the exit state out. The schedule and weights
are pre-built, so **no path pays for routing here**.

Four columns, and the distinctions matter:

* `latency` — `perf_counter` around the call **plus a synchronise**: what a step waits for;
* `dispatch` — `perf_counter` around the call alone: the host's share. For the eager loop
  latency ≈ dispatch because the host is the bottleneck; for one launch it is 0.11 ms
  against 0.98 ms of kernel;
* `device` — CUDA events bracketing the call (for a launch-bound call this contains the host
  latency, which is why it tracks wall);
* `occupancy` — the profiler's summed kernel time: real device work, and the denominator of
  the ratio the receipts have tracked since `RESULT_MATRIX_EMENDER.md` §6 (5.2x eager, 1.5x
  for GDN-2's fused kernels).

### 5.1 `matrix_ops` routing — the brief's shape

| path | launches | latency mean (p50) | dispatch | device | occupancy | **latency/occupancy** |
| --- | --- | --- | --- | --- | --- | --- |
| eager chunked (`fused=False`) | 1 392 | 20.694 (20.692) | 20.670 | 20.716 | 4.571 | 4.53x |
| `chunked` C = 16 | 384 | 44.272 (44.247) | 44.250 | 44.292 | 2.385 | 18.55x |
| `chunked` C = 64 | 96 | 11.084 (11.071) | 11.061 | 11.10 | 1.386 | 7.99x |
| **`chunked` C = 128 (baseline)** | **48** | **5.655 (5.592)** | 5.634 | 5.675 | 1.214 | **4.60x** |
| `chunked` C = 256 (**C**) | 24 | 2.864 (2.819) | 2.843 | 2.882 | 1.187 | 2.37x |
| `chunked` C = 512 (**C**) | 12 | 1.557 (1.558) | 1.438 | 1.576 | 1.268 | 1.23x |
| `graph` C = 128 (**B**) | 1 | 1.389 (1.391) | 0.088 | 1.405 | 1.354 | 1.03x |
| `graph` C = 256 (**B+C**) | 1 | 1.270 (1.270) | 0.085 | 1.286 | 1.240 | 1.02x |
| `graph` C = 512 (**B+C**) | 1 | 1.359 (1.360) | 0.085 | 1.374 | 1.324 | 1.03x |
| `graph` C = 256 + output clone | 1 | 1.292 (1.292) | 0.121 | 1.308 | 1.259 | 1.03x |
| **`persistent` (A)** | **1** | **0.979 (0.979)** | 0.110 | 0.995 | **0.873** | **1.12x** |

The baseline row is the *whole-call* form of `K23_READ.md` §5.3's 48-launch row (6.57 ms
there, 5.66 ms here): that receipt timed a sequence of chunk calls that discarded each
chunk's reads, this one assembles them into the block's `reads` tensor (16 extra copy
launches, §3). Run-to-run spread on a 5-rep mean is 5–10 % at these shapes; the
min–max columns are in the receipt.

### 5.2 `matrix_append` routing (a write on every token — the expensive case)

| path | launches | latency mean (p50) | dispatch | occupancy | latency/occupancy |
| --- | --- | --- | --- | --- | --- |
| eager chunked | 1 392 | 21.149 (21.198) | 21.123 | 4.465 | 4.75x |
| `chunked` C = 128 (baseline) | 48 | 5.617 (5.622) | 5.597 | 1.760 | 3.19x |
| `chunked` C = 512 (**C**) | 12 | 2.064 (2.063) | 1.476 | 1.863 | 1.11x |
| `graph` C = 512 (**B+C**) | 1 | 1.923 (1.923) | 0.087 | 1.893 | 1.02x |
| `graph` C = 512 + clone | 1 | 1.949 (1.947) | 0.120 | 1.911 | 1.02x |
| **`persistent` (A)** | **1** | **1.378 (1.376)** | 0.106 | **1.275** | **1.08x** |

### 5.3 Rank 64 and rank 512 (the ends of A's range)

Same block, synthetic schedule at the same shape (the routing fixture is built at rank 128;
the rank sweep is a timing and parity sweep, `synthetic_rank*` in the receipt).

| rank | path | launches | latency mean (p50) | occupancy | latency/occupancy |
| --- | --- | --- | --- | --- | --- |
| 64 | `chunked` C = 128 (baseline) | 48 | 5.528 (5.525) | 1.487 | 3.71x |
| 64 | `chunked` C = 512 (**C**) | 12 | 1.702 (1.697) | 1.497 | 1.13x |
| 64 | `graph` C = 256 (**B+C**) | 1 | 1.589 (1.590) | 1.548 | 1.03x |
| 64 | `graph` C = 256 + clone | 1 | 1.611 (1.611) | 1.570 | 1.03x |
| 64 | **`persistent` (A)** | **1** | **1.012 (1.011)** | **0.907** | **1.11x** |
| 512 | `chunked` C = 128 (baseline) | 48 | 5.621 (5.602) | 1.897 | 2.95x |
| 512 | `chunked` C = 512 (**C**) | 12 | 2.142 (2.141) | 1.944 | 1.10x |
| 512 | **`graph` C = 256 (B+C)** | **1** | **1.940 (1.939)** | 1.913 | **1.01x** |
| 512 | `graph` C = 256 + clone | 1 | 1.963 (1.963) | 1.934 | 1.02x |
| 512 | `persistent` (A) | 1 | 2.174 (2.170) | 2.064 | 1.05x |

### 5.4 What the ratio says

* The **wall/occupancy ratio moved from 4.60x to 1.02–1.12x** — past GDN-2's 1.5x — for
  both A and B, at every rank, in both routings. That is the number this lane was chasing
  through three stages; it moved when the host loop was removed and not before.
* A's ratio is slightly *higher* than B's (1.12 vs 1.02) because A pays one 0.11 ms launch
  and does **less** device work (0.873 ms of occupancy against the pipeline's 1.24–1.35 ms)
  — the
  ratio is close to 1 because there is very little left to overlap, not because A is hiding
  a cost.
* A beats B on latency at rank 64 and 128 (0.98 vs 1.33; 1.01 vs 1.44 with the clone) and
  loses at rank 512 (2.17 vs 2.03), where the register tile is at the budget and the
  pipeline's 128 programs of `(row, value_block)` shape balance better.
* **Option C alone is worth doing today**: 48 → 12 launches and 5.66 → 1.56 ms with no new
  kernel, no graph and no new lifetime rules — at the cost of the read precision in §2.3.

---

## 6. Which option won, honestly

| question | answer |
| --- | --- |
| Fewest host launches | A and B tie at 1; C = 512 gets to 12 with no new code |
| Fewest kernels | A (1) — it is the only path that is a single kernel |
| Lowest latency, rank 128, `matrix_ops` | **A: 0.979 ms** (B 1.270–1.389, C-only 1.557, baseline 5.655) |
| Lowest latency, rank 128, `matrix_append` | **A: 1.378 ms** (B 1.883–1.923, C-only 2.064, baseline 5.617) |
| Lowest latency, rank 64 | **A: 1.012 ms** (B 1.589–1.621, C-only 1.702) |
| Lowest latency, rank 512 | **B: 1.940 ms** (A 2.174, C-only 2.142) |
| Least device work | **A** (0.873 ms occupancy at rank 128 vs 1.24–1.35 for B/C) |
| Most accurate read | **A** (0.30–0.87x of the floor; the chunked pipeline is 1.5–5.8x) |
| Least machinery | **A** — no static buffers, no capture, no output-lifetime contract, no per-shape graph to hold |
| Easiest to adopt | **C** — one keyword argument to code that already exists |

**Option A wins**, and it wins on the axis that matters most to the next stage: it removes
the host loop *and* the read's numerical cost at once, needs no capture and no lifetime
discipline, works for any input tensor (the caller's tensors are read directly), and is the
simplest of the three to reason about at a call site. B is a genuine 5x and is the better
choice at rank 512 or for the code that already has the chunked pipeline (it needs no new
kernel at all). C is a free 4x that any caller of the existing path can take today.

The honest caveat on A: **it is one kernel because the whole store of a row fits one
program's registers, and that stops being true above rank 512 at `d_v` 128.** A is not a
general answer to the loop; it is an answer for the ranks this arm runs at, with a refusal
and a measured fallback beyond them.

---

## 7. Gate E — determinism and pathology

All in `test_hostloop_free.py`, all in the same run as the parity tests (17 tests, 89 in the
package):

| pathology | result |
| --- | --- |
| rank 64 (tile 1 024) and rank 512 (tile 8 192) | read 0.96x / 0.50x of the measured floor, median 7.5e-8 / 6.0e-8, **state bit-exact** against the oracle at both |
| rank 2048 | refused by name by `value_block`/`persistent_scan`/`block_scan(path='persistent')`; `block_scan(path='auto')` takes the chunked path and returns the right `reads` |
| chunk 16 / 128 / 256 | state bit-equal to the persistent kernel at every length, read inside the allowance, liveness equal |
| a chunk that does not divide the token count (13 tokens, chunk 2/3/5/13) | matches the oracle and the persistent kernel at every length |
| a block with no writes | the read at token `t` is the entry store decayed `t` times (checked token by token against the closed form); the exit state is the store decayed 9 times |
| a POP-only block | the liveness channel is empty at the exit and every read from token `rank+1` on is **exactly** zero |
| determinism across runs | two `persistent_scan` calls bit-identical; two graph replays bit-identical; no atomics, one writer per output element |
| the graph does not keep the caller's tensors | a second `run` with a different entry state returns the second answer, and the first state tensor can be zeroed afterwards without changing it |
| shapes and gradients | the shape audit refuses mismatches before any launch; all three paths refuse to build a graph (`FORWARD_ONLY`) and work under `torch.no_grad()` |

---

## 8. Deviations from the brief, and what is honestly not done

1. **The chunk axis was not turned into a grid axis, and no attempt pretends otherwise.**
   §1.2 states why: this arm's routing is state-dependent per chunk, so a per-chunk-parallel
   grid would be a different computation. The brief asked for the host loop to be removed
   without changing the semantics, and it is — by moving the sequence onto one program (A),
   into one replay (B), or into fewer dispatch steps (C).
2. **Option A's grid is `(batch*heads, d_v/BV)`, not `(rows, heads)`.** The brief's A holds
   the whole `rank x d_v` store in one program; this implementation additionally blocks the
   **value** axis, which costs no communication (every quantity in the walk, the read
   `w_t @ S[:, v]` included, is column-independent; each block keeps its own copy of the
   liveness channel, which is the same function of the schedule). Without it, A at rank 128
   runs 16 programs and takes 2.96 ms instead of 0.98 ms (§4, `BV` = 128 row). The rank axis
   is *not* blocked — that is the one that would need a cross-program reduction per token.
3. **`match -> decide` is still outside the kernel.** The decision channel is an input, as it
   is for K1/K2/K3, because routing is chunk-granular against the chunk-entry state
   (`schedule.route_chunk`), and re-deriving it per token inside the kernel would be the
   *per-token* routing the registered arm does not implement (`K23_READ.md` §7 measures what
   that difference is). So A removes the `T/C` scan loop, not the `T/C` routing step; a
   production block call still has 16 host steps for routing unless that is graphed too.
4. **Not measured: the routed block call, i.e. the arm's real boundary.** Routing is 16
   Python steps of torch ops between the scans; it is capturable (all shapes static, no host
   sync when `collect=False` and `growth=False`) and B would cover it, but that is a
   different boundary and is left to the next stage rather than measured here. Every number
   in this document is the scan boundary the receipts have used since `K23_READ.md` §5.3.
5. **Not measured at rank 2048.** §5.3 covers 64/128/512; at 2048 only the chunked paths are
   available and the C sweep is untested there. The extrapolation in §4 is labelled as one.
6. **No backward.** All three paths are forward only and refuse a graph, for the reason K1
   and K2/K3 do: the walk has no backward, so autograd through it would keep the read's
   gradient and silently drop the state's. The registered arm still trains on the eager path.
7. **The graph needs a `BlockGraph` object per (shape, chunk).** It is not a drop-in
   function; holding one per block shape is the caller's job, and the output buffers are
   reused across replays (`clone=True` exists and is measured). A at the same call site
   needs nothing.
8. **`torch.compile` of the scan was not attempted here.** `model.py` compiles the chunk
   loop today (`COMPILE_MODE = "default"`, no CUDA graph); whether the persistent kernel
   composes with inductor is a separate question and out of this scope.
9. **A's per-token cost is latency-bound, and the sweep shows where it goes** (`num_stages`
   1.4x, warps, block width) — but the walk is still a serial chain of 2048 dependent
   iterations at ~430 ns each. A further 2x would need the read to be block-wise
   (`tl.dot` over a token tile with an in-block correction), which is a different kernel with
   the injection-list structure of K2/K3 inside it. Not attempted; it is the natural next
   optimisation if A is adopted.

---

## 9. Commands

```bash
source scripts/frontier/activate_emender_frontier.sh   # or use .venv/bin/python
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/ -q                      # 89 passed
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/hostloop_free.json \
    --repeats 5
```

## 10. Recommended next step

Wire the persistent kernel where it fits and measure it in the arm, not on a block: the
scan is 0.98 ms of a forward pass whose routing is 16 host steps and whose MLP dominates
everything else, so the 5.8x on this boundary will not be a 5.8x on a step — the honest
next number is `tokens/s` at the 150 M board with `persistent_scan` in place of the compiled
chunk loop. Two things are needed first, and neither is in this lane's scope: a **backward**
(the arm trains, and the walk has none), and a decision on whether **routing** joins the same
per-launch path (it is capturable, and it is 16 host steps of its own). Option C is the one
thing that can be taken today with no new machinery at all — one keyword argument — with the
read-precision cost in §2.3 stated at the same time.
