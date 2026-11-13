# X44 — chunked parallel scan of the Emender memory state update

Result document for the chunked parallel-scan implementation of the token loop,
its parity evidence against the certified sequential path, and its measured
speed at the shapes the 150M arms run.

**Headline.** The chunked scan reproduces the sequential state trajectory
bit-for-bit at chunk length 1 and to `3.5e-6` max absolute deviation at chunk
lengths 32–128 (the sequential-vs-sequential noise floor on the same fixtures is
`1.5e-8`–`1.2e-7`), including the derived death schedule (identical to the
traced frees). With the chunk loop captured in a CUDA graph it is **4.6×–8.9×
faster than the same-semantics sequential per-token loop** and **17×–37× faster
than the certified `x31` Triton kernel** on the production shapes. The whole
memory block of one 150M-arm step (four heads) drops from `25.3 s` (certified
kernel) / `5.9 s` (sequential replay) to `0.85 s`. The win is **serialisation,
not bandwidth**: the chunked path moves 32–64× less slot-store traffic, yet it
runs at 1–22 GB/s against a measured 836 GB/s copy ceiling, because both paths
are dominated by per-token/per-op kernel count.

## 1. Placement

Implementation lives in a new `experiments/expressivity_tasks/x44_chunk_scan/`
rather than inside `x40_emender_lm/`, because

* it is a re-implementation of the **X30 `naive_reference` semantics** (the
  certified semantic reference), not of any X40 narrative arm;
* `x40/` and `x42/` sources are pinned by SHA-256 in
  `x42_pile_scale/velocity_regime.json` and `velocity_registration.json`
  (``verify_registration`` fails on any byte change), so a performance path
  belongs outside those trees;
* four 150M arms are running from the `x42/x43` trees right now; nothing in
  them is touched.

Files:

| file | role |
| --- | --- |
| `affine.py` | division-free chunked prefix scan of a diagonal affine recurrence (`compose`, `step_summaries`, `prefix_compose`, `scan_with_entry`, `chunked_prefix_scan`) |
| `schedule.py` | `DecisionSchedule` (the fixed decision channel) and its builder from a certified routing trace |
| `chunked_cell.py` | the Emender slot-store chunk scan: strength / content / age / usage channels and the chunk-parallel read |
| `triton_scan.py` | one Triton kernel for the intra-chunk affine scan (`mul.rn.f32`/`add.rn.f32`, no FMA contraction) |
| `graph_scan.py` | CUDA-graph capture of one fixed-shape chunked scan |
| `sequential_replay.py` | sync-free per-token loop over the same schedule (the speed baseline and a third parity implementation) |
| `test_chunk_scan.py` | 36 tests: affine algebra, parity, noise floor, pathological cases, GPU shapes, kernel/graph determinism |
| `bench_chunk_scan.py` | the speed / traffic measurement |

## 2. Formulation

The proved statement implemented is
`EmenderMemory.step_decomposition` (`s' = D ⊙ s + Σ_j e_j ⊗ b_j`) and its scan
corollaries in `formal/lean/EMENDER_OPERATOR_ALGEBRA.typ`:
`AffineOperator` composition, `chunked_fold_equiv`, `prefix_scan_equiv`,
`chunk_prefix_equiv`, `injection_propagation` (propagation needs no inverses),
`ordered_tree_equiv`.

**The schedule.** All discrete decisions are inputs, exactly as
`MemoryAlgebra.ScanCompatibleRelativeTo` requires: append slot and gate,
correction slot and `beta`, pop slot and pop multiplier, and the death masks.
Nothing is recomputed from rounded state. `TokenDecision.ValidFor`'s
obligation (death masks consistent with the supplied numeric state) is
discharged by deriving the deaths from the strength channel with the *same*
arithmetic as the reference (see §2.3).

**The channels.** Per token and slot the step is one diagonal affine map per
lane. The strength lane runs **two affine steps per token** so that the
operation order matches the reference exactly
(`strength * (1-pop)` then `* decay`, or `1` injected on append then `* decay`):

```
pop phase    d = 0, b = 1            on the appended lane   (erase + inject)
             d = 1-pop, b = 0        on the popped lane
             d = 1,     b = 0        elsewhere
decay phase  d = decay if live else 1, b = 0
```

Content lanes are erase-and-inject (`d = 0`, `b = payload`) on a write and identity
otherwise; age is `+1` per live token with `d = 0` on append; usage is
`* usage_decay + read weight` with a reset on append. Every one of these is
handled by the same engine, and **every zero diagonal is ordinary data**: POP
death and overwrite erasure both produce `d = 0`, and the composition
`(P₂,I₂) ∘ (P₁,I₁) = (P₂P₁, P₂I₁ + I₂)` never divides by an accumulated decay.
No `cumprod`/division formulation is used anywhere.

**The read** is the only part that is not an affine scan, and it is where the
chunk-parallelism pays. The sequential path does one matvec per token over all
live slots. Inside a chunk the same softmax is computed as

```
scores = q_chunk @ K_entry^T                        (C x S)
         with the column of every slot appended inside the chunk replaced by
         q_chunk @ k_event^T                        (C x C)
read   = W @ V_entry + (valid * W_event) @ (payload_event - V_entry[slot])
W      = softmax(scores masked by liveness) * strength_at_read * live
```

so the chunk-entry store (`S x (K+V)`) is streamed **once per chunk** instead of
once per token, and every token's read is one row of a matmul. `valid` selects,
for each (token, event) pair, the event that is the last write to its slot
before that token, so repeated writes to one slot inside a chunk are honoured.

**Renormalisation.** The intra-chunk pass keeps the running state and applies the
entry state through the recurrence itself; the multi-chunk composition
(`chunked_prefix_scan`) accumulates the `(P, I)` summaries in **fp64 at chunk
boundaries** and then re-multiplies the entry state by the local prefix product.
`test_affine_prefix_compose_matches_sequential_entries` shows the composed
entries match an fp64 exact fold to `32×` the fp32 cast noise, including a chunk
whose diagonal is entirely zero. At `S≈2200`, `rows=64`, `C=64` the per-chunk
trajectory tensors are already 36 MB each, so all-chunk trajectories at once
(`32 × ~10 × 36 MB`) do not fit; the runtime therefore carries the exact state
chunk to chunk, and the fully chunk-parallel form exists only as the tested
engine. This is stated as a limit, not hidden.

### 2.3 Death derivation (no masks in the schedule)

Deaths are not part of the input schedule in the production path; they are
derived in two passes per chunk. Pass A ignores in-chunk deaths (a lane's
strength dynamics depend only on its own lane, so the *first* threshold crossing
per life is unaffected) and the pop check is evaluated as
`strength_at_read * pop_factor <= free_threshold` — directly, because an append
at the same token erases the lane in pass A while the reference still records
that POP death before the reviving append. Pass B re-runs the strength channel
with the true liveness. The derived `(death_pop, death_decay)` masks are compared
**bit-for-bit against the reference trace's `freed` codes** in
`test_parity_with_sequential_reference` for every parametrised chunk length.

## 3. Parity evidence

Oracle: `x30_tape_cell.naive_reference.tape_scan` (the certified semantic
reference), replay mode. Second oracle: `x30_tape_cell.cell.py`'s checkpointed
replay path. Instances below are `naive.tape_scan` fixtures with `budget`
1–8/64/32, `heads` 2, decay `0.5`/`0.9`/`0.999995`/`0.999`, `free_threshold`
`0.1`/`0.15`/`0.2`/`1e-3`, key_dim 4, value_dim 5, up to 300 tokens; the
production-slot case runs on the GPU.

| gate | measurement |
| --- | --- |
| sequential-vs-sequential noise floor (two orderings of the same math) | read `1.5e-8`, values `1.5e-8`, strength/keys/age/usage `0` |
| chunked vs reference, chunk 1 | strength `0`, keys `0`, age `0`, usage `0`, values `3.0e-8`, read `6.0e-8` |
| chunked vs reference, chunk 2/3/4/5/19/64 | read `1.2e-7`, state `<= 1.2e-7` (max over decay 0.9/0.5/0.999995) |
| chunked vs `x30 cell.py` replay | `< 1e-5` on every channel |
| derived death masks vs traced `freed` codes | identical (pop and decay codes, all chunks, all fixtures) |
| chunked vs sequential, production shapes (2048 tokens, 2196 slots, real routed schedule) | max abs deviation `3.5e-6`, median `~1e-7` |
| tolerance rule | `<= 8×` the measured sequential noise floor; never tuned to pass |

Not bitwise: the chunked and sequential paths reassociate the read matmul
(`(rows,1,S)@(rows,S,V)` per token vs `(rows,C,S)@(rows,S,V)` per chunk) and the
value delta is computed as a cancellation `payload - V_entry`. That is the
`1e-7…3.5e-6` above. The *strength channel is bit-exact*, which is what keeps the
death boundaries and therefore the routing identical.

## 4. Pathological cases (all passing)

| case | test |
| --- | --- |
| zero diagonal entries: mid-chunk erasure, chunk-boundary erasure, a whole zero chunk, sparse zeros | `test_affine_zero_diagonal_matches_fold` |
| POP death then re-allocation of the same slot (and the append erasing the pre-death strength) | `test_pop_death_then_reallocation_of_the_same_slot` |
| decay death freezing strength and age at the death token | `test_decay_death_freezes_strength_and_age` |
| overwrite erasure plus a chain of three merges on one slot inside one chunk | `test_overwrite_erasure_and_correction_chain` |
| chunk length not dividing the token count (13 tokens, chunks 1/2/3/4/5/13/64), mostly-empty chunks | `test_chunk_length_not_dividing_token_count_and_empty_writes` |
| single-token sequence, every chunk size | `test_single_token_sequence_and_every_chunk_size` |
| no writes at all (pure decay, strength 0, read 0) | `test_no_writes_at_all_is_a_pure_decay` |
| single token per chunk (degenerate chunking) | `test_parity_with_sequential_reference[chunk=1]` |
| schedule that appends and corrects at the same row-token (rejected, not guessed) | `test_schedule_rejects_append_and_correct_collision` |
| affine composition associativity | `test_affine_compose_is_associative_and_matches_scan` |

## 5. Speed

Device: RTX 6000 Ada (card 0 or 1 via `scripts/gpu_lease.sh`, never 2–5), torch
2.9.1 cu128, fp32, `torch.backends.cuda.matmul.allow_tf32 = False`. Median of 3
runs after one warm-up. Inputs and the decision schedule are real: the schedule
is the routing the certified `x31` kernel produced for those inputs
(append rate 0.17–0.41, correction rate 0.29–0.53, pop rate 0.78 per token).

`certified` = `x31_tape_cell_fused.tape_scan` (the Triton kernel the X40/X42 arms
run). `sequential` = `sequential_replay.sequential_tape_scan`, identical
semantics and schedule to the chunked path. `chunked graph` = the chunked scan in
a CUDA graph; the best chunk length is shown (`c64` except `c32` at 2196 slots).
`state GB` is the analytic slot-store traffic
`rows · tokens · slots · (K+V) · 4` (sequential) versus
`rows · ⌈tokens/C⌉ · slots · (K+V) · 4` (chunked).

| shape (slots, tokens × rows) | certified ms | sequential ms | chunk graph ms | state GB seq | state GB chunk | speedup vs sequential | vs certified | max dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head0 s117, 2048×64 | 6181 | 1463 | **165** | 11.8 | 0.18 | 8.9× | 37.5× | 3.0e-6 |
| head1 s311, 2048×64 | 6343 | 1459 | **172** | 31.3 | 0.49 | 8.5× | 36.8× | 2.6e-6 |
| head2 s826, 2048×64 | 6369 | 1463 | **203** | 83.1 | 2.60 | 7.2× | 31.4× | 3.5e-6 |
| head3 s2196, 2048×64 | 6376 | 1481 | **313** | 221.1 | 6.91 | 4.7× | 20.3× | 3.5e-6 |
| short s2196, 256×64 | 687 | 184 | **40** | 27.6 | 0.86 | 4.6× | 17.3× | 2.1e-6 |
| short s117, 256×64 | 685 | 184 | **21** | 1.5 | 0.02 | 8.9× | 33.1× | 2.4e-6 |
| batch8 s2196, 2048×8 | 5481 | 1489 | **167** | 27.6 | 0.43 | 8.9× | 32.7× | 2.4e-6 |
| **four-head memory block** | **25269** | **5866** | **853** | **347** | **10.2** | **6.9×** | **29.6×** | |

Tokens/s at `T=2048, rows=64`: sequential `89 600`, chunked graph
`795 500` (s117) / `418 200` (s2196); the certified kernel `20 600`.

Step time of the whole 150M arm: the arms log `34 s/step` for 12 layers with
this memory block; the measured block is `25.3 s/step` of it, i.e. the memory
update is ~75 % of a training step and is the thing that was worth attacking.

### 5.1 Where the win is: serialisation, not bandwidth

* Measured copy ceiling on the card: **836 GB/s**. The sequential path's
  analytic state traffic at head3 (`221 GB`) would take `264 ms` at that
  ceiling; it takes `1481 ms`, i.e. it runs at `149 GB/s` (18 % of peak), and at
  head0 it runs at `8 GB/s` (1 % of peak). **The sequential path is not
  bandwidth-bound** — it is bound by 2048 sequential dependent per-token steps.
* The chunked path removes 32–64× of that traffic (`347 GB → 10.2 GB` for the
  four-head block) but converts almost none of it into wall time: it runs at
  1–22 GB/s.
* Instrumentation: the eager chunked scan issues **236 068 device kernels** for a
  2048-token scan (32 chunks × ~7 400) and spends `210 ms` of device time inside
  `~1.8 s` of wall time — **~70 % of wall time is host-side launch overhead**.
  CUDA-graph capture collapses that to one host launch and gives `165 ms–313 ms`
  ([the eager PyTorch chunked path is *slower* than the sequential loop:
  `1 685 ms` vs `1 463 ms` at head0 — see §6.1]). After capture the replay time
  tracks device time, i.e. it is now bound by the *count* of small kernels
  (~2.4 µs per graph node), not by bytes.
* **Conclusion: the chunked formulation is the right form and the graph is what
  makes it win; the remaining ceiling is kernel count, not traffic.** A fused
  intra-chunk kernel (bookkeeping loop + read in one program per row tile) is the
  next lever, and it is where the 32× traffic reduction would finally become
  wall time.

Graph capture cost is a one-off: `7.5 s` for capture (plus the warm-up passes) at
`T=2048, rows=64, s117`; `~9 s` at s2196. It is amortised over all replays of
that shape, and re-capture is needed only when a shape changes.

## 6. What did not work / limits

### 6.1 The eager PyTorch chunked scan is slower than the sequential loop

`chunked_tape_scan` without a graph: `1 685 ms` (head0) / `1 662 ms` (head3)
against `1 463 ms` / `1 481 ms` for the per-token loop. The reason is measured,
not guessed: 236 068 kernel launches, 70 % of wall time in launch overhead,
device time only 210 ms. A chunked scan written as Python loops over the chunk
steps has the *same* O(T) launch count as the sequential loop plus extra
trajectory-tensor work, so it cannot win without either a graph or a kernel.

### 6.2 A Triton scan that let the compiler contract `m*s+b` into an FMA flipped a death boundary

The first Triton version of the intra-chunk scan was FMA-contracted and differed
from the reference by `9.5e-7` (1 ulp). That is enough to move a strength across
`free_threshold` one token early: the chunked-vs-sequential read deviation went
from `3.0e-6` to **`5.1e-4`** at chunk 32, because one slot died at a different
token and its content left the read. The fix is explicit
`mul.rn.f32`/`add.rn.f32` inline PTX, which restores bit-exactness
(`test_triton_affine_scan_is_bit_exact`). This is the concrete instance of the
`state_dependent_death_not_affine` obstruction: the death mask is only
reproducible if the numeric channel is reproduced exactly.

### 6.3 `torch.compile` on the chunk loop

Attempted with `mode='max-autotune-no-cudagraphs'`; it produced graph breaks in
the schedule validation and was abandoned in favour of the CUDA graph (one host
launch, no compile risk). Killed after the compile budget.

### 6.4 The inter-chunk prefix composition is implemented and tested but not on the runtime path

`chunked_prefix_scan` composes per-chunk `(P,I)` summaries with a Hillis-Steele
scan over the chunk axis and materialises the chunk entries by renormalised
re-multiplication — the Lean `prefix_scan_equiv`/`injection_propagation` form.
At production slot counts the all-chunk trajectories do not fit in device memory
(§2), so the runtime carries the exact state chunk-to-chunk and the composition
is exercised by the affine tests only. Note also that it *cannot* be made fully
chunk-parallel without giving up exact death boundaries: the entry state of chunk
`c` depends on the deaths inside chunks `< c`, and those depend on the entry
state. That is the runtime shadow of `state_dependent_death_not_affine`.

### 6.5 Scope limits (stated, not silently dropped)

1. **Decisions are inputs.** The chunked scan accelerates the *replay* path —
   the differentiable backward pass and any forward pass that already has a
   trace. The routing/decision pre-pass, which must score every live slot to pick
   the pop target and compute the write residual, remains sequential. This is
   exactly `ScanCompatibleRelativeTo`'s hypothesis and the X30 `Decision`/`Trace`
   design; it is not a workaround, but it bounds the end-to-end win to the
   fraction of a step that is state update rather than decision.
2. **The append gate is taken from the schedule.** The reference computes
   `sigmoid(append_logit + append_err_coef · error)` where `error` is the
   write-residual magnitude — a state function. The chunked scan consumes gate
   magnitudes (as `cell.py`'s recorded `Decision.error` does). Production sets
   `error_coef = 0` (a zero buffer in the X42 arms), where the gate is a pure
   input function; a non-zero coefficient is rejected rather than silently
   dropped.
3. **Forward only.** No backward pass, no gradient parity, no training run was
   added here.
4. **One head at a time, `n_heads = 1` rows.** The X42 arms already scan each
   head separately, so this matches production, but cross-head batching of the
   matmuls is unexplored.
5. **Memory.** The per-chunk trajectory tensors are `rows × C × S`; at
   `rows=64, C=64, S=2196` that is 36 MB per tensor with `~10` live at once.
   `row_block` exists to keep this bounded; the benchmark uses `row_block=rows`.
6. **`usage` is carried for parity with `naive_reference` only**; `cell.py`'s
   efficient store drops it.

## 7. Exact commands

```bash
source scripts/frontier/activate_emender_frontier.sh   # or the local .venv
"$EMENDER_PYTHON" -m pytest experiments/expressivity_tasks/x44_chunk_scan/test_chunk_scan.py -q
# 36 passed

eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
"$EMENDER_PYTHON" -m experiments.expressivity_tasks.x44_chunk_scan.bench_chunk_scan \
    --repeats 3 --chunks 32 64 --out experiments/expressivity_tasks/x44_chunk_scan/receipts/bench_b.json
```

Raw measurements: `receipts/bench_a.json`, `receipts/bench_b.json`.

## 8. Recommended next step

Fuse the per-token bookkeeping (liveness, last-append/last-write indices, value
payload chain) and the read into one Triton program per (row-block, chunk), so a
chunk costs O(1) kernels instead of ~7 400. That is what converts the already
real 32–64× state-traffic reduction into wall time instead of leaving the path
bound at ~2.4 µs per graph node; the measured ceiling from that change alone is
the device-time figure in §5.1 (`210 ms` at `s117`, `383 ms` at `s2196` before
fusion, both of which then become bandwidth-limited work).
