# K2 + K3 — the matrix store's read as two fused Triton kernels

Stage 2 (and last) of the X45 matrix store's *forward* kernel lane. Companion to
`K1_SCHEDULE_SCAN.md`, which removed 86 of the 87 kernels an eager chunk launched and left
exactly one thing in torch: the read reduction over the emitted injection list, 23 kernels
per chunk. This stage is that reduction as two `tl.dot` kernels.

**No whole-arm result is claimed here.** No 2700 s arm was run, no tokens/s number is
reported; every number below is a chunk or a 2048-token block call, forward only.

**Headline, as narrowly as the evidence supports.**

* The read is **two launches per chunk** instead of 23 (`_k23_inter_kernel` then
  `_k23_intra_kernel`), and the whole drop-in chunk is **three** (K1 + K3 + K2) instead of
  24. A 2048-token block call went **1 392 → 48 kernels**.
* Read parity holds **at every token**, not just at chunk ends: on the arm's own routed
  schedule with all five ops present, the maximum deviation is **9.5e-7 = 1.65×** the
  measured fp32 sequential floor (5.79e-7; the allowance is 8×, 4.63e-6), the median is
  **1.2e-7**, and the state stays **bit-exact** against the per-token oracle.
* The block call went **23.5 → 6.6 ms** (3.6×) and `matrix_append` **23.7 → 6.1 ms**.
  Occupancy (real device work) went 4.46 → 1.07 ms.
* **The wall/occupancy ratio did not move toward GDN-2's 1.5×: it stayed at 5.2–5.7×.**
  That is the honest result and §5.3 says why with numbers: after the launches are gone the
  wall is Python-side dispatch (≈0.31 ms per chunk of host time against 0.061–0.098 ms of
  device work), so removing kernels lowers both sides of the ratio at once. Only in-kernel
  or graph-level sequencing can move it, and neither is in scope here.
* The default is now the fused path (`FUSED_SCAN_DEFAULT = True`), the eager reference is
  retained as the oracle **and still the arm's training path** — the fused path is forward
  only and refuses a graph rather than silently dropping the state's gradient.

---

## 1. What was built

| file | role |
| --- | --- |
| `k23_read.py` | the two kernels (`_k23_inter_kernel`, `_k23_intra_kernel`), `k23_read`, the `k23_chunk` drop-in, and the behaviour-neutral `union_normalised_reads` diagnostic |
| `test_k23_read.py` | 17 tests = gates A, B, E, the flag, the graph refusal and the diagnostic's identity |
| `k23_gate_receipt.py` | measures gates A and B and the diagnostic into `receipts/k23_gates.json` |
| `bench_k23_read.py` | measures gates C and D into `receipts/k23_read.json` |
| `scan.py` | `chunk_scan(..., fused=True)` now routes to `k23_chunk`; `FUSED_SCAN_DEFAULT` flipped to `True` |
| `test_matrix_emender.py`, `test_k1_schedule_scan.py`, `k1_gate_receipt.py`, `bench_k1_scan.py` | every call site that means *the eager reference* now says `fused=False`; `bench_k1_scan`'s `k1_chunk` row calls `k1_schedule_scan.k1_chunk` so it keeps its stage-K1 meaning |

### 1.1 The two kernels

Per `(row = batch·head, token_block)`, with `steps = tokens × lanes` and
`gc_read(t) = log_gate[t·lanes − 1]` (and `0` for token 0, where there is no previous
step):

```
K3  inter:  reads[t]  = exp(gc_read(t)) · (w_t @ H)                       one tl.dot chain
K2  intra:  reads[t] += Σ_{u < t·lanes} exp(gc_read(t) − gc_u) · w_t[j_u] · d_u
```

K3 walks the rank axis in `BK` blocks, accumulating `(BT, V)` in fp32 registers, so no
`[rows, tokens, rank]` tensor is ever materialised. K2 walks the step axis in `BU` blocks,
gathers `w_t[j_u]` for the block's token tile, applies the causal predicate
`u < t·lanes` (a mask, not a loop), and accumulates `(BT, BU) @ (BU, V)`. The two terms are
joined by **one fp32 load-add-store in the second kernel** — exactly the eager path's
`entry_read + injected`.

Both autotune over block shapes keyed on `(rank, value_dim)` —
`INTER_CONFIGS` = `BT ∈ {16,32,64} × BK ∈ {32,64}`, `INTRA_CONFIGS` = `BT ∈ {16,64} ×
BU ∈ {32,64}` — and both are `@triton.jit(do_not_specialize=['T','N'])`, so a new sequence
length is not a recompile. `V` is a constexpr padded to a 16-multiple (`VP`) and masked
down, because `tl.dot` wants every operand axis a multiple of 16.

### 1.2 Two things that are not style choices

**`input_precision='ieee'`.** Triton's default for fp32 `tl.dot` is TF32 on tensor-core
hardware, and TF32 would put the read three orders of magnitude above the fp32 floor.
Measured, in the receipt (`precision_probe`, one fp32 `tl.dot` at rank 128 against torch's
true-fp32 matmul, `allow_tf32` is False): **`ieee` 1.5e-5 absolute = 3.8e-7 relative,
`tf32` 3.4e-2 = 8.4e-4 relative** — 2 236× worse, per dot and before the read's own
reduction. Gate A (allowance 4.6e-6 on a read of scale 5.9) could not survive that, so the
setting lives in both kernels and in the receipt rather than in a comment.

**`restore_value=['reads']` on the intra kernel's autotune.** K2 accumulates into the inter
term (`load · add · store`), and `@triton.autotune` *benchmarks* every config by running the
kernel on the real buffers. Without the hook each trial would add the intra term again and
the reported read would be a multiple of the truth. `restore_value` snapshots `reads`
before each trial and restores it after. K3 stores, so it needs nothing.

### 1.3 The normalisation, stated once

The certified read (`x30_tape_cell/naive_reference.py` docstring line 19) is
`Σ_i softmax_i(score over live slots) · strength_i · value_i` with **no** division by the
strength mass. In this arm the softmax is taken **once per chunk against the chunk-entry
state** by `read.read_weights` and handed to the scan as the `weights` channel
(`schedule.route_chunk`), and the in-chunk injections are added **linearly with those same
weights**:

```
o_t = exp(gc_read_t) · w_t @ H  +  Σ_{u ≤ read_t} exp(gc_read_t − gc_u) · w_t[j_u] · d_u
```

So the denominator is over the entry-live slots, it is inside `w`, and it multiplies the
intra block too. `read.py` remains the single definition of the weights; these kernels only
reduce them, exactly as K1 does. §8 records why the brief's "the denominator must include
both blocks" is not implemented — it describes a different read.

---

## 2. Gate A — read parity at every token

`python -m pytest experiments/expressivity_tasks/x45_matrix_emender/ -q` → **72 passed**
(17 new). Fixture: batch 2, 64 tokens, 2 heads, rank 5, `d_v` 8, lanes 2, decay 0.95,
seed 23, routing = `schedule.route_chunk` on real state with values drawn from 6 distinct
directions. Op histogram over 512 lanes: `none 295, append 26, correct 125, pop 32,
merge 34` — all five ops present. Floor: `sequential_noise_floor` on the same fixture
(two reduction orderings plus fp64), **5.786e-7**; allowance `NOISE_FACTOR = 8×` =
**4.629e-6**.

| chunk | read max vs eager | read median vs eager | read max vs oracle | read median vs oracle | ratio of floor | state vs oracle | liveness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.8e-7 | 1.2e-7 | 4.8e-7 | 1.2e-7 | 0.82× | 0.0 | equal |
| 2 | 1.43e-6 | 3.6e-7 | 4.8e-7 | 1.2e-7 | 0.82× | 0.0 | equal |
| 3 | 1.19e-6 | 3.0e-7 | 9.5e-7 | 1.2e-7 | 1.65× | 0.0 | equal |
| 4 | 9.5e-7 | 2.4e-7 | 9.5e-7 | 1.2e-7 | 1.65× | 0.0 | equal |
| 16 | 9.5e-7 | 2.4e-7 | 9.5e-7 | 2.4e-7 | 1.65× | 0.0 | equal |

Read scale 5.94. Three things worth reading off that table:

* **Every deviation is ≤ 1.65× the measured floor, at every token, at every chunk length**
  — that is the whole gate, and the median says the max is not hiding a drift that most
  tokens see.
* **`state vs oracle = 0.0` everywhere**: the state is K1's, and K1 is bit-exact, so the
  fused read did not move a death boundary by construction. Liveness equal at every
  boundary.
* The deviation *vs eager* is sometimes larger than *vs oracle* (chunk 2: 1.43e-6 vs
  4.8e-7) and that is the right way round: the eager chunked path reaches the same state
  through a scatter-add whose rounding differs from the oracle's, so the fused path is
  closer to the semantics of record than the reference it replaces.

The pipeline is also structurally pinned: `test_the_fused_read_is_the_sum_of_the_two_reference_terms`
holds `k23_read` to `inter + intra` recomputed in the reference's own expression, and the
per-term isolation is gate B's job (§3).

---

## 3. Gate B — the four fixtures, and which failure each catches

| fixture | measured | catches |
| --- | --- | --- |
| (i) **no writes** (pure inter) | injection scale **exactly 0.0**, intra term **exactly 0.0**, read vs the entry term **5.96e-8** on a scale of 2.70 | a stray intra term, or a coefficient that fires on a zero injection |
| (ii) **a write on every lane**, from an empty store (pure intra) | inter term **exactly 0.0**, read vs the intra term **1.64e-7** on a scale of 2.35 | a missing or mis-scaled intra block — the term is nil in (i) and dominant here |
| (iii) **a slot dies mid-chunk** | read before the POP 1.5; residue after it **5.96e-8 = 4.0e-8 of the pre-pop scale**; slot not live at the exit | a wrong decay weight or a double-counted slot: the decayed entry term and the POP's injection must cancel *exactly* |
| (iv) **a token whose only support is an in-chunk injection** | entry term **exactly 0.0**; token 1 read `[0.75, 1.5, 2.25]` vs closed form `[0.75, 1.5, 2.25]`, deviation **0.0** | a wrong coefficient: with nothing in the entry state to hide behind, any missing `exp(gc_read)`/`exp(−gc_u)` factor is a pure scale error |

Case (iii) is the one that is not a coincidence: for a token after the POP the read is
`exp(gc_read) w[j] H[j] + exp(gc_read − gc_u) w[j] (−H[j]·exp(gc_u))`, whose two terms are the
same number with opposite signs. A denominator or coefficient error anywhere in the pair
leaves a visible fraction of 1.5 behind; the measured residue is the fp32 rounding of
`gc_read − gc_u` against `gc_read`, nothing else.

Case (iv) is the closest thing this algebra has to the brief's "only support is an
injection": the slot is live at chunk entry (so it *has* a normalised weight) but its
content is zero, so everything the read returns arrived through the intra term.

---

## 4. Gate C — kernel count

`receipts/k23_read.json`, same inputs and the same recorded schedule for every path
(`torch.profiler`, CUDA activity, device-attributed events). Boundary: **one chunk of 128
tokens = 256 steps, forward only, fp32**, schedule and read weights pre-built, entry state =
the reference trajectory's state at chunk 4 (populated store: the first chunk's read is
*exactly zero* by the certified semantics and would be a flattering measurement).

| path | kernels per chunk | per 2048-token block call (16 chunks) |
| --- | --- | --- |
| `chunk_scan(fused=False)` (eager reference) | 87 | 1 392 |
| `k1_state_scan` (the walk alone) | 1 | 16 |
| `k1_chunk` (K1 + **torch** read reduction) — post-K1 | 24 | 384 |
| `k23_chunk` (K1 + K3 + K2) — **this stage** | **3** | **48** |

The profiler's first kernels for the fused path are, in order,
`_k1_state_scan_kernel, _k23_inter_kernel, _k23_intra_kernel, _k1_state_scan_kernel, …` —
one of each per chunk, no hidden fallback. **K2+K3 removes 21 of the 24 kernels the post-K1
drop-in launched**; the remaining three are the walk and the two read terms.

---

## 5. Gate D — timing, and the ratio that did not move

Boundary, stated once: **one chunk of 128 tokens (256 steps), forward only, fp32, batch 8 ×
heads 2 × rank 128 × `d_v` 128**; the schedule and the read weights are inputs; the
chunk-entry state is the reference trajectory's state after 4 chunks. `wall` =
`perf_counter` around the call plus `synchronize`; `device` = CUDA events bracketing the
call; `occupancy` = the profiler's summed kernel execution time. 2 warm-up + **9** measured
reps, mean and p50, min–max spread reported in the receipt.

### 5.1 `matrix_ops` routing (append/NONE in the timed chunk)

| path | kernels | wall mean (p50) | wall spread | device (events) | occupancy | wall / occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| `chunk_scan` eager | 87 | 1.425 ms (1.432) | 1.406–1.433 | 1.454 ms | 0.276 ms | 5.19× |
| `k1_state_scan` | 1 | 0.178 ms (0.177) | 0.170–0.189 | 0.211 ms | 0.046 ms | 3.83× |
| `k1_chunk` (K1 + torch read) | 24 | 0.629 ms (0.627) | 0.612–0.650 | 0.658 ms | 0.115 ms | 5.46× |
| `k23_chunk` (K1 + K3 + K2) | **3** | **0.369 ms (0.368)** | 0.356–0.398 | 0.392 ms | **0.061 ms** | 6.03× |

### 5.2 `matrix_append` routing (a write on every token — the expensive case)

| path | kernels | wall mean (p50) | wall spread | device (events) | occupancy | wall / occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| `chunk_scan` eager | 87 | 1.431 ms (1.429) | 1.415–1.476 | 1.460 ms | 0.273 ms | 5.23× |
| `k1_state_scan` | 1 | 0.179 ms (0.176) | 0.171–0.187 | 0.249 ms | 0.084 ms | 2.10× |
| `k1_chunk` (K1 + torch read) | 24 | 0.631 ms (0.632) | 0.618–0.643 | 0.660 ms | 0.153 ms | 4.14× |
| `k23_chunk` (K1 + K3 + K2) | **3** | **0.371 ms (0.372)** | 0.351–0.389 | 0.394 ms | **0.098 ms** | 3.78× |

### 5.3 The whole 2048-token block call (16 chunks carrying the state)

| routing | path | kernels | wall mean (p50) | device (events) | occupancy | wall / occupancy |
| --- | --- | --- | --- | --- | --- | --- |
| `matrix_ops` | eager | 1 392 | 23.49 ms (22.99) | 23.52 ms | 4.455 ms | 5.16× |
| `matrix_ops` | **K1+K3+K2** | **48** | **6.57 ms (6.09)** | 6.60 ms | 1.070 ms | 5.69× |
| `matrix_append` | eager | 1 392 | 23.74 ms (23.11) | 23.77 ms | 4.438 ms | 5.21× |
| `matrix_append` | **K1+K3+K2** | **48** | **6.14 ms (6.14)** | 6.17 ms | 1.723 ms | 3.57× |

The event-bracketed device column tracks the wall column (ratio 0.84–1.00) precisely
because these calls are host-dispatch bound: the GPU is idle waiting for the host, so an
event bracket around the call measures the host. **Occupancy is the only honest device
number here**, and it is the denominator the 5.2× in `RESULT_MATRIX_EMENDER.md` §6 uses.

**What the ratio says, honestly.** The brief's goal was to move the wall/device ratio from
5.2× toward GDN-2's 1.5×. **It did not move.** Per chunk it is 5.19× eager and 6.03× fused;
per block call 5.16× and 5.69×. The reason is measurable and not mysterious:

* the host cost of a call is what is left after occupancy: `k1_state_scan` — one launch and
  five tensor allocations — is 0.178 − 0.046 = **0.132 ms** of host against 0.046 ms of
  device work, and `k23_chunk` — three launches and six allocations — is
  **0.369 − 0.061 = 0.308 ms** (83 % of its wall) against 0.061 ms;
* so a Triton launch plus its share of allocation costs roughly **60–90 µs** of host here,
  against 15–35 µs of device work per kernel at these shapes;
* the eager chunk spends 1.425 − 0.276 = **1.15 ms** of host time (81 %) against 0.276 ms.

So the *fraction* is unchanged and the ratio is set by it: removing kernels removes host
time and device time together. What this stage bought is the absolute numbers — host time
per chunk fell **1.19 → 0.34 ms** and the block call fell **3.57×** — and the ratio can only
fall when the per-chunk Python dispatch is itself removed (one launch for the whole block,
or a CUDA graph). That is §11's next step, not a claim here. The one datapoint in the
direction the brief wanted is a *single* fused kernel: `k1_state_scan` alone is 3.83× /
2.10×, i.e. the ratio tracks launches-per-call, and three launches per chunk cannot reach
1.5× while each launch costs ~60–90 µs of host and ~15–35 µs of device.

### 5.4 Parity at the bench shape

| routing | read deviation | read scale | relative | state deviation (K1 vs eager) |
| --- | --- | --- | --- | --- |
| `matrix_ops` | 3.0e-9 | 9.6e-3 | 2.7e-7 | 8.3e-7 of 2.01 |
| `matrix_append` | 1.8e-6 | 2.60 | 6.9e-7 | 2.4e-6 of 7.77 |

The `matrix_ops` read at this chunk is small (9.6e-3) because the store is saturated and the
weights are correspondingly diffuse — the relative column is the one to read. The state
deviation is K1's against the eager scatter-add, not against the oracle, where it is exact
(`K1_SCHEDULE_SCAN.md` §3).

---

## 6. Gate E — pathologies

All in `test_k23_read.py`, all against the eager reduction.

| pathology | result |
| --- | --- |
| chunk 16 / 128 / 256 at the arm's `d_v = 128` | deviation ≤ 1e-6 relative at every length |
| rank 64 and rank 2048 | 1e-6 and 1.1e-6 relative; 2048 is 2048 fp32 terms in a different blocking order than cuBLAS, which is why the bound is K1's 1e-5 for that case |
| a chunk length that does not divide the token count (13 tokens, chunk 2/3/4/5/13) | inside `NOISE_FACTOR × floor` per token vs the eager path and vs the oracle; liveness equal |
| determinism across runs | two `k23_read` calls bit-identical — no atomics, one writer per output element |
| value dim / weight-channel mismatches | refused before any launch (`MAX_VALUE_DIM` and the shape audit) |
| a graph through the fused path | refused (`FORWARD_ONLY`); accepted under `torch.no_grad()` |

---

## 7. The union-normalised read, as a number

`union_normalised_reads` recomputes `read.read_weights` **at every token** against the state
as it is at that token — a softmax over the slots live *now* rather than over the
chunk-entry set — and then reads. That is the tape store's per-token granularity; the
registered matrix arm weights a whole chunk from its entry state, which is what keeps the
scan affine and chunk-parallel. The diagnostic is **not on any runtime path** and changes no
default; it exists so "does chunk-constant weighting cost anything measurable?" is a number.

Fixture A is the gate-A routed fixture (rank 5, saturated, every chunk re-writes slots);
fixture B is the arm's shape (rank 128, `d_v` 128, chunk 128, real routing, 256 tokens,
decay on a 64-token horizon). Delta is normalised by the row's own read scale.

| fixture | row | read scale | max Δ | median Δ | max relative | median relative |
| --- | --- | --- | --- | --- | --- | --- |
| routed rank 5 | from an empty store | 5.94 | 8.25 | 1.92 | 1.39 | 0.32 |
| routed rank 5 | from a populated store | 5.94 | 6.14 | 1.85 | 1.03 | 0.31 |
| arm shape rank 128 | from an empty store | 1.51 | 34.5 | 0.43 | 22.9 | 0.29 |
| arm shape rank 128 | from a populated store | 1.51 | 0.48 | 0.20 | 0.32 | **0.13** |

Two readings, and the distinction matters:

* **From an empty store the number is a start-up artefact, not a quality statement.** The
  registered read of chunk 0 is *exactly zero* (no entry slot is live, so every weight is
  zero and that chunk's own writes are invisible to it) while a per-token read sees them.
  That is the 22.9× entry in the table, and it is why the populated row is the headline.
* **At the arm's shape from a populated store the median difference is 13 % of the read
  scale and the max 32 %.** So chunk-constant weighting is *not* a second-order effect at
  this granularity, and it is a real question for the claim set — but it is a question about
  the registered architecture (chunk-parallel routing was chosen for the affine scan), not
  something this stage introduced or may quietly change.

The two effects inside the diagnostic are separable and one of them is pinned by a test:
with **no in-chunk write** the live set and the cosines never move, so the per-token read
differs from the registered one by exactly the strength channel it re-reads from the decayed
rows, `union_t = exp(gc_read_t) · registered_t` — verified to 1e-6 relative. The remaining
difference (the softmax's live set) can only move when a write lands inside the chunk, which
is what the populated row measures.

---

## 8. Deviations from the brief

**1. There is no union denominator, and this stage did not add one.** The brief asks for
`denom_t = Σ_inter exp(score) + Σ_intra exp(score)` with
`out = (Σ_inter exp·v + Σ_intra exp·payload) / denom`. That is **not the read this arm
implements**, and implementing it would have changed the semantics and made gate A
unpassable by construction:

* `read.read_weights` (`read.py`) takes the softmax **once per chunk** against the
  chunk-entry state and returns `softmax_j(live entry rows) · ||S[j]||` as the `weights`
  channel; `schedule.route_chunk` records that as a fixed decision channel.
* `scan.chunk_scan` (the oracle) then reduces **linearly**: `(exp(gc_read)·w) @ H +
  (before · exp(gc_read − gc_u) · w[:, j_u]) @ delta`. The denominator is inside `w` and it
  multiplies both terms; there is no per-token renormalisation anywhere.
* `read.py`'s own docstring is explicit that the read magnitude carries the stored strengths
  and that there is **no** division by the strength mass — "do not fix that".

So the kernels take `weights` as an input channel exactly as K1 does, `read.py` stays the
single definition of the weights, and §7 measures what the brief's alternative would change.
Everything else in the brief was implemented as written: two `tl.dot` kernels, `autotune`
keyed on `(rank, value_dim)` with `do_not_specialize=['T','N']`, masks as constexpr-shaped
predicates, no host loops, no `[rows, C, rank]` materialisation, the eager reference
retained and the default flipped only after A and B passed.

**2. Gate B(iv) is a different fixture than the brief imagined.** Under the certified
semantics a slot that is *dead at chunk entry* has `w = 0`, so an in-chunk write into it
contributes nothing to that chunk's read: "a token whose only support is an in-chunk
injection" is realised as a **live entry row with zero content** (and a non-zero weight),
which makes the read `w[j] · d` and nothing else. §3 reports it as measured.

**3. The timed `matrix_ops` chunk has no CORRECT/POP/MERGE** (`none 2810, append 1286`):
that is the real routing's regime at this seed (the store fills, then most tokens are NONE
under capacity pressure). `matrix_append` (a write on every step) is measured alongside as
the expensive case. Gate A's fixture is the one that covers all five ops.

**4. `bench_k1_scan.py` changed meaning in one row, deliberately.** Its `k1_chunk` row used
to be `chunk_scan(..., fused=True)`, which now includes K2+K3. It calls
`k1_schedule_scan.k1_chunk` directly so the K1 receipt keeps measuring stage K1, and
`bench_k23_read.py` measures the new drop-in.

---

## 9. What did not work, and what is honestly not done

1. **TF32 is the default, and it is unusable here.** Triton's fp32 `tl.dot` defaults to TF32,
   so both kernels pass `input_precision='ieee'`. The receipt's `precision_probe` measures
   what that buys: at rank 128 the `ieee` dot is 3.8e-7 relative from torch's fp32 matmul,
   the `tf32` dot 8.4e-4 — 2 236× worse, per dot. This was closed before the first
   measurement rather than discovered: a kernel written with the default per-dot error
   carried through the read's scale (5.9) is ~5e-3, a thousand times the gate-A allowance,
   which is at least a *loud* failure. The receipt is the point — the next kernel in this
   lane should not have to rediscover it.
2. **The autotune hooks are load-bearing.** K2 accumulates into the inter term, and
   `@triton.autotune` benchmarks every config by running the kernel on the real buffers, so
   without `restore_value=['reads']` each trial would add the intra term again and the
   read would be a multiple of the truth that depends on how many configs had been
   benchmarked — i.e. a *cache-dependent* answer, which is the class of bug that survives a
   shape test and shows up as "the reads are a bit large sometimes". Also closed before
   measuring; the test suite compares against the eager path on the first call after a
   fresh compile, so a regression here fails gate A rather than hiding.
3. **The union diagnostic's first self-check was wrong, and that was the useful part.** I
   expected the per-token read to equal the registered read whenever there are no in-chunk
   writes. It does not: with no writes the state only decays, so a per-token
   `read_weights` re-reads the **strength channel** from the decayed rows and the read comes
   out scaled by a further `exp(gc_read)` — `union_t = exp(gc_read_t) · registered_t`, now a
   pinned identity in the test suite and the reason §7 reports two effects instead of one.
   The registered read applies the decay once (to the values); a per-token read applies it
   twice (strength and values).
4. **The intra gather reads the `slot` channel in its own layout.** The step axis is
   `t·lanes + lane`, so the address is affine in `u` only at one lane; the kernel decomposes
   `u // lanes` and `u % lanes` itself rather than paying a host permute + copy of the
   decision channel.
5. **The wall/occupancy ratio did not move** (§5.3). Stated as a measured non-result.
6. **No fused backward.** Forward only, enforced: the read is differentiable and the walk
   is not, so autograd through the fused path would keep the read's gradient and silently
   drop the state's. `k23_read`/`k23_chunk` raise `FORWARD_ONLY` when any input requires
   grad while autograd is enabled, and `MatrixMixer` refuses `fused_scan=True` together
   with `compile_scan=True`.
7. **The registered arm still trains on the eager path** (`MatrixMixer(fused_scan=False)`).
   Flipping the module default does not flip the arm, because the arm needs the backward.
   Training on the fused forward would need K4 (the backward) or a forward-only use.
8. **Rank growth was not exercised through the kernels.** A grown store is a wider `R`
   (covered by the rank-2048 test); no growth *wave* was run with the fused read on.
9. **Autotune cost is paid per distinct `(rank, value_dim)`.** The test suite compiles six
   inter and four intra configs per shape pair; that is the ~30 s the package now takes.
   `ndm/triton/pin_autotune.py` falls back to `configs[0]` for kernels it has not measured,
   so an arm would run `BT=16, BK=32` unless a config is registered — nothing about the
   math depends on the block shape.

---

## 10. Commands

```bash
source scripts/frontier/activate_emender_frontier.sh   # or use .venv/bin/python
python -m pytest experiments/expressivity_tasks/x45_matrix_emender/ -q        # 72 passed
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
python -m experiments.expressivity_tasks.x45_matrix_emender.k23_gate_receipt \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k23_gates.json
python -m experiments.expressivity_tasks.x45_matrix_emender.bench_k23_read \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k23_read.json --repeats 9
```

## 11. Recommended next step

The forward is now three kernels per chunk and the wall is 83 % Python-side dispatch, so the
next lever is **sequencing, not arithmetic**: either a CUDA graph over the whole block call
(the shapes are static and the schedule is a recorded input; `model.py` already has the
compiled-scan machinery and knows the graph-fault history) or one kernel for the whole block
whose grid carries the chunk axis with the state walk kept sequential inside it. Both are
strictly forward-only and would show up in exactly the metric that did not move here
(wall/occupancy, 5.7× → ?). **Taken, and answered with numbers, in
[`NO_HOST_LOOP.md`](NO_HOST_LOOP.md)**: one persistent kernel for the whole block call is
1 launch and 0.98 ms at 4.60× → 1.12×, a captured graph is 1.27–1.39 ms at 1.02×, and a
wider chunk (C = 512) alone takes the loop to 12 launches and 1.56 ms at 1.23×. The fused
**backward** is the other large stage and is independent of this one; attempt it only with
its own gate discipline, and not in the same lane as either of the above.
