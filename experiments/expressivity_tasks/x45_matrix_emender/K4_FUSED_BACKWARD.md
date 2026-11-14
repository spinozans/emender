# K4 — the fused backward of the matrix store's block scan

Stage 4 of the X45 matrix store's kernel lane: the **backward** of the
persistent block scan, as one reverse-walk kernel.  K1 fused the state scan,
K2+K3 fused the read, and the no-host-loop stage made the whole 2048-token
forward **one launch** (`NO_HOST_LOOP.md`) — and every one of those kernels is
forward only, `check_no_grad` refusing a graph through them, so the registered
arm still trains on ordinary autograd through the *eager chunk scan*
(`scan.chunk_scan(fused=False)`): ~1 400 kernels forward and ~2 000 backward
per block call.  The tape arm's profile is the warning this lane is not allowed
to reintroduce — 78 % of its step in checkpoint-replay kernels
(`ReplayBackward`/`reverse_kernel`, `PREREGISTRATION_MATRIX_EMENDER.md`) — and
the matrix arm's registered backward has no replay kernels but pays the eager
pipeline instead.  This stage replaces that pipeline with the adjoint of the
walk, in the same persistent form the forward just won with.

**No whole-arm result is claimed here.**  No 2700 s arm was run, no tokens/s, no
nats/byte.  Every number below is one 2048-token block call, forward and
backward, fp32, on one RTX 6000 Ada (48 GB).

**Headline, as narrowly as the evidence supports.**

* **One kernel walks the adjoint.**  The backward is 1 reverse-walk kernel plus
  2 deterministic reductions — **3 device kernels**, against the eager
  autograd path's **2 027** (chunk 128, the receipts' boundary; 3 579 at the
  model's own chunk 64).
* **The whole block call, forward + backward, goes 44.8 ms → 6.0 ms (7.5x)** at
  chunk 128, and **103.6 ms → 6.0 ms (17x)** against the chunk length the
  registered model actually trains at.  Wall/occupancy 3.06x → **1.03x**.
* **Grad parity holds at the noise floor, and half of it is bit-exact.**  Every
  grad output (dstate, dweights, ddecay, dpayload) matches the eager per-token
  autograd oracle within a stated multiple of a *measured* fp32 floor;
  dstate and dpayload are **bit-exact** against it, at every shape measured.
  The oracle itself is validated by an fp64 `gradcheck` first.
* **The forward did not move.**  The trajectory-recording forward is
  bit-identical to `persistent_scan` (reads, state, liveness — tested, and
  re-asserted after a backward has run), whose state and liveness are bit-exact
  against the per-token oracle (`NO_HOST_LOOP.md` §2).
* **The design decision — store the trajectory — costs 2.00 GiB and ~2.1 ms of
  forward time at the bench shape** (b8/T2048/2h/rank128/dv128: 31x the block's
  every other tensor, 4.2 % of the card; the forward without the store is
  1.0 ms, with it 3.1 ms).  The eager autograd baseline saves the same order of
  bytes as activations for its per-token multiplies (its peak during a step is
  3.1 GB), so this is not a new cost class — and recompute-over-store
  (`NO_HOST_LOOP`'s established doctrine) was **not needed**: option (2) below
  stays unbuilt until a measurement asks for it.
* **No replay kernels anywhere**: the profiler's kernel names for both the
  eager and the fused path contain no `replay`/`checkpoint` event (receipt
  `replay_kernels_absent: true` for every row) — the matrix arm's registered
  backward never had them; what this stage kills is the 2 027-kernel eager
  backward and the untrainability of the 1-launch forward.
* **Rank 2048 is refused by name and falls back** to the eager autograd path,
  through the same `persistent_available` rule the forward uses; the fallback
  is tested to produce flowing gradients, not just to not crash.

---

## 1. What was built

| file | role |
| --- | --- |
| `k4_fused_backward.py` | `persistent_scan_trajectory` (the forward + the per-token pre-decay state store), `persistent_scan_backward` (the reverse walk: 1 kernel + 2 deterministic reductions), `fused_block_scan` (the whole call as one `torch.autograd.Function`), `eager_block_scan` / `trainable_block_scan` (the flag, default off, with the size-gate fallback) |
| `test_k4_fused_backward.py` | 14 tests: the fp64 gradcheck of the oracle, gate A parity (floor-measured, per output), the five structural adjoint cases each naming its failure mode, determinism, rank 64/512, the size gate and fallback, the shape audit |
| `bench_k4_fused_backward.py` | gates C and D into `receipts/k4_fused_backward.json`: kernels forward/backward separately, the trajectory's bytes, peak memory, wall/dispatch/device/occupancy, grad parity at the bench shape, determinism, the rank sweep, the rank-2048 refusal |
| `K4_FUSED_BACKWARD.md` | this document |

### 1.1 The adjoint of the walk

The forward per token: read against the state as it stands, then DECAY, then
every lane in order.  The adjoint reverses exactly that, token by token, lanes
last-first, with the recorded schedule as a constant (the doctrine: no gradient
through decisions — there are no decision logits):

| forward op | adjoint |
| --- | --- |
| lane write, APPEND (`S[j] += p`) | `dpayload[t,l] = dS[j]`; `dS[j]` passes through |
| lane write, erase (CORRECT/POP/MERGE) | `dpayload[t,l] = dS[j]`; `dS[j] := 0` — the discarded content's gradient terminates here, which is why a POP's row is dead to everything before its death token |
| DECAY (`S <- lam S`) | `dlam_t = <dS, S_t>` against the **pre-decay** state; `dS <- lam * dS` |
| read (`o_t = w_t S_t`) | `dS[j] += w_t[j] dout_t` (direct) and `dw_t[j] = dout_t . S_t[j]` (through the weights) |

The two terms that need `S_t` — `dlam_t` and `dw_t` — are THE design decision of
the stage.  The options, in the order the brief gave them:

1. **Store the trajectory** (`T x rank x d_v` fp32, written by the forward kernel
   from the register tile it already holds).  Built.  Measured: 2.00 GiB and
   ~2.1 ms of forward time at the bench shape (§3).
2. Chunk-boundary checkpoints + in-kernel recompute.  **Not built** — option (1)
   measured fine at the arm's shape, and recompute would re-walk the block the
   backward is already walking.  It is the next lever if a measurement ever
   objects to the 2 GiB, not before.
3. An analytical WY-style adjoint avoiding the trajectory entirely (GDN-2's
   trick).  **Deliberately not attempted**: that is where
   `chunk_gdn2_bwd_kernel_wy_dqkg_fused` + `chunk_gdn2_bwd_kernel_intra`'s ~900
   lines live, and GDN-2 can do it only because its write needs no selection.

### 1.2 One grid, same size gate, two deterministic reductions

The backward kernel is the forward's persistent form walking the other way:
grid `(rows*heads, d_v/BV)`, the adjoint tile `RP x BV` in the program's
registers, `mul.rn`/`add.rn` so no FMA contraction moves the chain, loads
pipelined with `num_stages`, the same `value_block`/`warp_count` rules and the
same refusal (`persistent_available`: rank 2048 needs 32 768 register floats,
four times the budget — refused by name before any launch).

The value axis is the only axis blocked, as in the forward — but the adjoint's
two value-axis reductions (`dw_t[j] = sum_v dout . S_t[j]`, `dlam_t`) cross
value blocks.  The kernel therefore writes **per-block partials**
(`dweights_partial (rows, blocks, T, rank)`, `ddecay_partial (rows, blocks, T)`)
and the wrapper sums them over the block axis with torch — a fixed reduction
order, no atomics, bit-deterministic (§5).  `dpayload` needs no reduction: the
adjoint of a row write is column-independent, so each program writes its own
value columns of the addressed row — one writer per element, inactive lanes
written as exact zeros (§2.3).

### 1.3 The autograd boundary

`fused_block_scan` is the whole block call as one node: four continuous inputs
carry gradient (state, weights, decay, payload), the decision channel
(slot/erase/op/live_after) and the liveness are passed through for shape
checking and returned `None`.  `trainable_block_scan(fused=...)` is the flag —
**default off**, the eager autograd path, exactly as the doctrine requires; the
default flips only after the gates pass, and flipping it is the next stage's
decision, not this one's.  The backward is first-order (no `create_graph`):
the adjoint walk is not itself differentiable, which matches every kernel in
this lane and is stated in the module docstring.

---

## 2. Gate A — grad parity against the autograd oracle

The oracle is the **eager per-token walk under PyTorch autograd**
(`test_k4_fused_backward.eager_block_walk`), validated before it is trusted:
an fp64 `torch.autograd.gradcheck` on a fixture covering every adjoint branch
(NONE, APPEND, CORRECT-erase-with-payload, POP-erase-with-zero-payload, two
lanes) passes.  The floor is **measured** on the same fixture — two fp32
orderings of the walk (the read as an einsum and as a matmul) against each
other, and against the same walk in fp64 — never below one ulp of each output's
own scale; the allowance is the lane's `NOISE_FACTOR = 8x`.

On the canonical routed fixture (rank 5, 64 tokens, 2 heads, real `route_chunk`
routing, histogram none 295 / append 26 / correct 125 / pop 32 / merge 34):

| output | fused vs oracle (max) | vs floor | median | bit-exact |
| --- | --- | --- | --- | --- |
| dstate | 0.0 | 0.00x | 0.0 | **yes** |
| dweights | 9.537e-7 | 0.9x | 0.0 | no |
| ddecay | 3.815e-6 | 1.0x | 6.0e-8 | no |
| dpayload | 0.0 | 0.00x | 0.0 | **yes** |

At the bench shape (batch 2, the full-batch oracle's autograd graph saves the
whole trajectory and then some, so the parity case states its batch —
`grad_parity.batch` in the receipt): dstate and dpayload **bit-exact** (0.00x
floor), dweights 1.25x, ddecay 0.45x of floors measured on that fixture.  The
same relation holds at rank 64 and 512 (tested) and against the *registered*
backward (autograd through `chunk_scan(fused=False)`) within its chunk grouping.

The forward is asserted **bit-identical** to `persistent_scan` (all three
outputs), before and after a backward has run on the same inputs — the
"backward must not perturb the forward path" gate.

### 2.1 Why two of the four are bit-exact

The adjoint's state chain (`dS <- lam * dS`, the lane reversals, the read's
direct term) is elementwise: the same roundings in the same order as autograd's
own backward, so it reproduces the oracle bit for bit.  The two reductions
(`dw`, `dlam`) change grouping — the kernel's per-block partial sums against
cuBLAS's — and land at 0.5–1.3x of the measured fp32 floor.  The parity tests
assert the bit-exactness of the first pair explicitly, so a future refactor
cannot silently change their rounding class.

### 2.2 The structural adjoint cases (gate B), each with its failure mode

| case | test asserts | failure mode guarded |
| --- | --- | --- |
| (i) no-op block | dstate/ddecay equal the closed forms of a pure decay chain (`dlam_t` divides by the token's own gate — the off-by-one detector) | a DECAY weight applied one token early or late |
| (ii) append-only | every token's `dpayload` equals the single-write closed form (reads strictly after the write, decay-weighted; the entry rows' adjoint is the pure-decay chain) | the payload grad missing the write→read decay weighting, or including the token's own pre-write read |
| (iii) a slot POPped mid-block | upstream gradient strictly after the death token gives the earlier payload and the entry row **exactly zero**; gradient between write and death reaches the payload per the closed form | gradient leaking past an erase (a POP that does not kill the row's adjoint) |
| (iv) MERGE | the blend payload (content attached, weights detached — `route_chunk`'s own construction) carries gradient to **both** source rows; everything matches the oracle | a one-lane adjoint applied to a two-lane op |
| (v) CORRECT | `dpayload` is the rank-1 closed form, the discarded row's incoming gradient is **exactly zero**, other rows follow the pure-decay chain, no other lane produces a payload grad | the overwrite failing to terminate the old row's gradient |

Case (iii)'s boundary is named in its test because it is the one that bit us
during development: the read **at** the death token is pre-write, so it still
sees the popped row — "strictly after" is the claim, and the test says so.

### 2.3 The convention the brief could not have known: inactive lanes

`algebra.sequential_matrix_scan` applies a lane's write to every head of the
batch whenever *any* head's lane is active; for an inactive head that write is
`rows + 0` — the identity in value, but its autograd reports
`dpayload = dS[j]`.  The registered training path (`chunk_scan`) masks the
payload by `op != NONE`, so **its** payload gradient at an inactive lane is
exactly zero — and the kernels skip inactive lanes outright.  The fused
backward follows the registered path (zero), the grad oracle implements the
same masking, and `test_inactive_lane_payload_grads_are_zero_like_the_registered_chunk_path`
pins fused and registered grads together on a fixture with 295 inactive lanes.
This is the read-weights lesson again: the brief's "eager per-token reference"
and the code's registered semantics disagree in exactly one adjoint corner,
and the code is the contract.

---

## 3. Gate C — launches, and the trajectory's memory as a first-class number

Boundary: one whole block call, b8/T2048/2h/rank128/dv128, lanes 2, fp32.
"Kernels" is the profiler's device-kernel count per call; every kernel here is
also one host launch (a graph-free path).

| path | forward kernels | backward kernels | step (fwd+bwd) |
| --- | --- | --- | --- |
| eager chunk scan, C = 128 (the receipts' baseline) | 1 408 | **2 027** | 3 435 |
| eager chunk scan, C = 64 (the model's own default) | 2 560 | 3 579 | 6 139 |
| **fused** (`fused_block_scan`) | **1** | **3** (walk + 2 deterministic reductions) | **4** |
| `persistent_scan` (the forward, no trajectory — for comparison) | 1 | — | — |

**The trajectory.**  `T x rank x d_v` fp32, written by the forward kernel:

* **2.00 GiB** at the bench shape — **31x the block's every other tensor**
  (66 MiB: state, weights, decay, payload, and the step's two upstream
  gradients), **4.2 % of the card's 48 GB**;
* the store costs the forward **~2.1 ms** (1.0 ms without it, 3.1 ms with it,
  p50) — writing 2 GiB at the card's write bandwidth, i.e. the honest price of
  option (1), not an implementation artifact;
* peak allocated during a fused step: **4.85 GB**, against the eager path's own
  **3.10 GB** (autograd saves the per-chunk states of its pipeline) — the
  trajectory is ~1.7 GB on top of a baseline that already saves the same
  order, and it is released when the graph node frees.

The partial buffers the reductions need add 134 MiB (`dweights_partial`) +
1 MiB (`ddecay_partial`) at this shape, live only inside the backward.

---

## 4. Gate D — timing: the wall this stage kills

Same boundary, 5 reps, mean/p50 with min–max in the receipt; `dispatch` is the
host's share, `occupancy` the profiler's summed kernel time.

| path | step p50 (fwd+bwd) | fwd p50 | bwd p50 | dispatch | occupancy | wall/occ |
| --- | --- | --- | --- | --- | --- | --- |
| eager C = 128 (**the registered backward, before**) | **44.81 ms** | 20.55 | 26.94 | 44.78 | 14.67 | 3.06x |
| eager C = 64 (the model's own chunk) | 103.63 ms | 39.17 | 52.71 | 103.61 | 21.62 | 4.79x |
| **fused (after)** | **5.98 ms** | 3.08 | 2.92 | **0.94** | 5.79 | **1.03x** |

`matrix_append` (a write on every token, the expensive case): eager 54.0 ms →
fused **7.09 ms** (7.6x).  The fused backward's warp sweep (8 warps: 3.01 ms
vs the default's 2.92) confirms the forward's `warp_count` rule; a different
warp count is a different (equally valid) reduction order for `dw`/`dlam`, so
determinism is stated per configuration.

**The before/after step time, and the 78 % figure.**  The tape arm's 78 % of
CUDA time in checkpoint replay is what this lane's registration set out not to
reproduce; the matrix arm's registered backward has no replay kernels (the
receipt checks the profiler's kernel names for `replay`/`checkpoint` on every
path: absent).  What the matrix arm paid instead is the eager autograd
pipeline — 44.8 ms per block call at chunk 128, 103.6 ms at the model's own
chunk 64 — and this stage takes that to **6.0 ms**, of which 5.8 ms is device
work at a 1.03x wall/occupancy ratio.  The mixer's backward is no longer
host-bound, and the 1-launch forward is trainable for the first time.

**Rank sweep** (full step, same block shape, synthetic schedule):

| rank | fused step | eager step C = 128 | forward bit-equal to `persistent_scan` |
| --- | --- | --- | --- |
| 64 | 4.43 ms | 54.12 ms (12.2x) | yes |
| 128 | 5.98–7.09 ms | 44.8–54.0 ms | yes |
| 512 | 24.32 ms | 53.95 ms (2.2x) | yes |

Rank 512 is the honest boundary of option (1): the trajectory is 8.6 GB there,
and the backward's 11.1 ms (measured isolated) is bandwidth-bound on reading
it — not a spill, not a bug, the measured cost of storing over recomputing at 4x
the state.  It is still 2.2x faster than the eager path; the graph path (B)
that won the rank-512 forward has no fused backward and is not made trainable
by this stage.

---

## 5. Gate E — determinism, the size gate, the fallback, the audit

* **Bit-determinism**: two full fused forward+backward runs at the bench shape
  reproduce every gradient bit for bit (`determinism` in the receipt: all true;
  tested).  No atomics anywhere; the two cross-block reductions are summed
  over the block axis in a fixed order.
* **The size gate**: rank 2048 is refused by name by both the trajectory
  forward and the fused backward — `ValueError: the persistent kernel needs
  next_pow2(2048) * 16 = 32768 register-resident floats ... does not fit, so it
  must take the chunked path` — and `trainable_block_scan(fused=True)` falls
  back to the eager autograd path, which the receipt shows still producing
  finite, non-zero gradients (and which the tests show agreeing with the
  oracle).
* **The shape audit** refuses a backward whose inputs do not belong to the
  block (dreads shape, weights rank, dtype, contiguity of the directly-indexed
  tensors) before any launch, and the refusal fires through the public entry
  point too.
* **Reads unchanged**: the trajectory forward's three outputs are bit-identical
  to `persistent_scan`'s, before and after a backward (§2).

---

## 6. What failed, and what is honestly not done

1. **`torch.scatter` could not carry the oracle's payload gradient** (torch
   2.9.1: `ScatterBackward0 returned an invalid gradient` for a src of shape
   `(b, h, 1, V)` — a size-1 src dim).  The oracle's row write is
   `Tensor.index_put` instead, which is autograd-clean and value-exact.  This
   is an oracle-implementation note, not a semantics decision.
2. **The inactive-lane adjoint corner** (§2.3) is the one place the brief's
   named oracle and the registered code disagree; the registered path won, and
   the disagreement is receipted in a test rather than smoothed over.
3. **The death-token boundary** (§2.2 case iii) cost one test revision: the
   read at the death token is pre-write, so "the popped row's grad terminates
   at its death token" is a claim about *strictly later* tokens.  The test
   names the boundary.
4. **`dw` is computed even when the read weights do not require grad** (a
   `needs_input_grad` guard would save the 134 MiB partial buffer and its
   traffic).  Not built: in the arm every continuous input is attached, and a
   conditional backward is a correctness risk for a saving the arm cannot
   collect.  Stated here so it is a decision, not an oversight.
5. **Routing is still outside the boundary**, as in every stage of this lane
   (`NO_HOST_LOOP.md` §8.3): the schedule and read weights are recorded inputs.
   A production step still pays the 16 host steps of routing per block; that is
   the next boundary, not this one.
6. **No whole-arm measurement.**  The 6.0 ms is the mixer's block call, one of
   24 blocks of a forward whose MLP and routing are untouched.  The honest
   whole-arm number (tokens/s at the 150 M board) is the next stage's to take,
   and the flag (`trainable_block_scan(fused=True)`) is off by default until
   then.
7. **No double backward.**  The adjoint walk is not itself differentiable; a
   `create_graph` backward through `fused_block_scan` will refuse.  Every
   kernel in this lane is the same way, and the module docstring says so.

---

## 7. Commands

```bash
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/ -q     # 103 passed
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.bench_k4_fused_backward \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k4_fused_backward.json --repeats 5
```

## 8. Recommended next step

Flip the flag inside the arm and measure the whole model, not the block: wire
`trainable_block_scan(fused=True)` into `MatrixMixer` behind its own
`fused_backward` switch, run the 150 M board's step profile, and take the
tokens/s number with the 4-launch mixer (forward 1, backward 3) in place of
the 3 435-kernel eager pipeline.  The two things that number will surface —
routing's 16 host steps per block (now the dominant launch count in the mixer)
and the trajectory's 2 GiB per layer-call peak (24 layers share the card, and
the peak arrives per block, not summed) — are the next stage's measurements,
and neither is changed by anything in this file.
