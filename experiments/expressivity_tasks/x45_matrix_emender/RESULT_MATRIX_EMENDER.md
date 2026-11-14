# X45 — the matrix-state Emender: what was built, and what was measured

Result document for `PREREGISTRATION_MATRIX_EMENDER.md` (sha256 `20a163c9…74dec2`).
Every throughput number below carries its measurement boundary. Nothing here is a
whole-arm result: the registered 2700 s arms were **not** run in this lane (see §7).

**Headline, stated as narrowly as the evidence supports.**

* The five Emender ops are **exact** on an orthonormal key family: a POP zeroes its
  own direction and leaves every other row **bit-identical**, a CORRECT writes its
  row exactly and touches nothing else, and the registration's key form and the
  coefficient form agree for non-trivial families (§2).
* The chunked scan matches the per-token oracle within **1.6–2.8×** the measured
  fp32 sequential noise floor, and its liveness channel is **bit-exact at every
  chunk boundary** — so deaths, and therefore routing, are reproducible (§3).
* **No replay anywhere**: `ReplayBackward`, `reverse_kernel`, `Replay` and
  `scan_kernel` are absent from the device profile of both matrix arms, where the
  tape arm spent 78 % of its step (§4).
* Same-harness, same-config, same-stream comparison with the registered `gdn2` arm:
  `matrix_ops` **13.2 k tok/s**, `matrix_append` **15.9 k tok/s**, `gdn2`
  **21.8–28.1 k tok/s**. The matrix arms are therefore **1.6–2.1× slower** than
  GDN-2 in this harness, not at its class. Gate D is **not met** (§5).
* Block vs block at identical shapes and identical state size: the matrix ops block
  costs **21.7 ms** of device time against GDN-2's delta-rule block **5.97 ms**
  (3.6×), and is 13× slower in *eager wall* time because it is host-launch bound
  (§6). At the step level the memory block is no longer the binding cost of the arm.

---

## 1. What was built

`experiments/expressivity_tasks/x45_matrix_emender/`

| file | role |
| --- | --- |
| `keys.py` | the orthonormal key family (one-hot / Hadamard / seeded-QR), the active table as a prefix `K[0:r]`, `snap_rank`, `growth_target` |
| `algebra.py` | `MatrixSchedule` (the fixed decision channel), the five ops as one erase-flagged rank-1 update, the per-token oracle `sequential_matrix_scan`, `key_form_step` |
| `read.py` | the single-sourced read: softmax over live rows × strength, no renormalisation (**certified semantics**), plus `attention_probs`, `row_strength`, `read` |
| `schedule.py` | `RoutingPolicy` (frozen constants) and `route_chunk`: CORRECT / MERGE / APPEND / POP / NONE from the *chunk-entry* state, detached, plus the read-weight channel |
| `scan.py` | the chunked scan: division-free triangular solve of the delta rule, exact inter-chunk carry, `repeat_write_rate` (the growth statistic), growth at chunk boundaries, telemetry |
| `model.py` | the arm: GDN-2 block shape with the mixer swapped, exact capacity accounting, one shared compiled scan |
| `gates.py` | gates C and D (§4, §5) |
| `bench_blocks.py` | block vs block (§6) |
| `test_matrix_emender.py` | 33 tests = gates A, B, E |

**State and ops.** `S ∈ R^{r×d_v}` per head, row `j` the stored coefficient along
key direction `k_j` of the active prefix. With orthonormal keys `S k_j == S[j]`, so

| op | registration form | coefficient form |
| --- | --- | --- |
| APPEND | `S += v k_j^T` | `S[j] += v` |
| CORRECT | `S += (v' − S k_j) k_j^T` | `S[j] = v'` |
| POP | `S −= (S k_j) k_j^T` | `S[j] = 0` |
| DECAY | `S *= λ` | `S = λ S` |
| MERGE | two rank-1 ops | CORRECT(blend) on the target lane + POP on the source lane |

The three writes are one update with an *erase* flag, `S[j] = erase ? v' : S[j]+v'`.
All discrete choices arrive as the `MatrixSchedule` — never recomputed inside the
scan, never differentiated. `DECAY` is the differentiable diagonal channel (a learned
per-token, per-head gate, floored at `1e-4`), and the read weights are computed from
the chunk-entry state once per chunk, which is what keeps the scan affine.

**The scan.** With `gc_s = Σ_{u≤s} log λ_u` over the chunk's step axis (a token with
`L` lanes is `L` steps; DECAY sits on lane 0), the chunk update is

```
S_s = exp(gc_s) H + Σ_{u≤s} exp(gc_s−gc_u) k_{j_u} d_u^T
d   = (I + A)^{-1} (p − c ⊙ exp(gc) ⊙ H[j])      A[s,u] = c_s exp(gc_s−gc_u)[j_s=j_u][u<s]
```

`A` is **block-diagonal by slot**, so an erase in one direction cannot leak into
another: the cross-slot entries are structurally zero, not small. No division of an
accumulated quantity appears — the decay channel is a *sum* scan in log space and the
erase zeros live in a triangular forward substitution. The inter-chunk carry is the
exact state, one masked scatter-add per chunk (the composed prefix-product form has no
runtime path: the entry state of chunk `c` depends on the in-chunk deaths, which
depend on the entry state — the runtime shadow of
`state_dependent_death_not_affine`).

**Read (single-sourced, `read.py`).** `out_t = Σ_j softmax_j(score over live) ·
strength_j · S_{t−1}[j]`, with `strength_j = ‖S[j]‖` (the store has no separate
strength channel; a POP removes the row from the read in the same step) and
`score = scale · cos(q_t, S[j])` with the query a learned projection into the value
space. **No strength renormalisation** — that absence is by design, and
`test_read_matches_certified_reference` holds our read to the *executed*
`x30_tape_cell/naive_reference.py` read on identical state (deviation ≤ 1e-6 fp32).
Mask: store liveness, the same predicate `x30` uses (`live`) and `x35`'s
`causal_read.py` uses (`version >= 0`) — I found **no** different mask in the X40 path,
so there is no read-normalisation mismatch claim in this receipt.

**Rank is not a parameter.** The query lives in the value space and addresses *rows*,
so nothing about the projections depends on `rank`. `matrix_append` and `matrix_ops`
are parameter-identical (**149,981,808** each, −0.015 % of the 150,003,984 board);
growth changes the memory footprint and nothing else.

**Shape** (capacity-matched the way every arm in this project is matched): width 576,
depth 24 — the registered GDN-2 backbone — with 2 heads × 128 value dims, rank 128,
and the SwiGLU absorbing the remaining capacity (hidden 3352, against GDN-2's 1648).
That absorption is a **deconfound**: this mixer is much cheaper per layer than
GDN-2's, so at matched total capacity its MLP is twice as wide.

## 2. Gate A — orthonormality and exactness

`python -m pytest experiments/expressivity_tasks/x45_matrix_emender/test_matrix_emender.py -q` → **33 passed**.

| claim | measurement |
| --- | --- |
| `K[0:r] @ K[0:r]^T == I` for `r ∈ {64,128,256,512,1024,2048}`, families `standard`, `hadamard`, `qr` | deviation `≤ 4·r·eps_fp32` at every rank (the bound is the fp32 dot-product rounding, not a fitted constant) |
| prefixes are nested and never re-orthonormalised | `torch.equal` on the shared rows for every growth step |
| POP is exact | target row is **exactly** zero; every other row `torch.equal` to the pre-POP state |
| CORRECT is exact | target row is **exactly** the written value; every other row bit-identical |
| the registration's key form is the same algebra | `key_form_step` (`S += (v' − c S k_j) k_j^T`, a real contraction against the key table) matches the coefficient form for `hadamard` and `qr` families to 1e-5 relative |
| APPEND is rank-1 | exact |

This is the registration's exactness gate, at the level of the ops. The chunked path
reaches the same statements to fp32 noise (§3); only the cross-slot coupling is
*structurally* zero there, which is what makes it noise rather than leakage.

## 3. Gate B — scan parity, and bit-exact liveness

Oracle: `sequential_matrix_scan` (per-token, read → DECAY → every lane in order).
Tolerance discipline: the **sequential-vs-sequential noise floor is measured first**
(two reduction orderings of the same math, and fp32 against a float64 fold) and the
chunked path must stay within `8×` of it. Measured on three fixtures (batch 2, 48
tokens, 2 heads, rank 9, `d_v` 5, two lanes, all five ops):

| fixture | sequential noise floor | worst chunked read deviation | ratio (allowed 8×) |
| --- | --- | --- | --- |
| seed 7 | 2.81e-07 | 4.77e-07 | 1.70× |
| seed 13 | 1.73e-07 | 4.77e-07 | 2.75× |
| seed 21 | 2.97e-07 | 4.77e-07 | 1.61× |

worst state deviation over `chunk ∈ {1,2,3,4,5,16,64}`: **5.96e-07**.

* **Liveness is bit-exact**: the integer `live_after` channel propagated by the chunked
  scan equals the oracle's per-token channel at every chunk boundary
  (`torch.equal`, chunk 1–16). Death boundaries therefore cannot drift.
* Pathologies covered: single token, chunk length not dividing the token count,
  a schedule with no writes at all (pure decay), POP followed by re-allocation of the
  same slot in the same chunk, a correction chain, an inactive lane carrying a stray
  payload (masked explicitly).
* **Backward**: gradients through the chunked matmuls match the oracle's gradients to
  5e-3 relative on weights, payload and decay gates — ordinary autograd, no surrogate.

## 4. Gate C — no replay in the profile

Same step loop, `torch.profiler` CUDA attribution, forbidden kernel search over
`replaybackward`, `reverse_kernel`, `replay`, `scan_kernel`:

| arm | `replay_absent` | `forbidden_kernels_found` |
| --- | --- | --- |
| `matrix_ops` | true | `[]` |
| `matrix_append` | true | `[]` |
| `gdn2` (reference) | true | `[]` |

For contrast, the tape arm's registered step is 78 % replay:
`ReplayBackward` 29.5 %, `reverse_kernel` 29.4 %, `Replay` 9.6 %
(`x43_fork_resolved/runs_150m/profile_lattice_emender_bounded.json`). The matrix arms
retain activations and let autograd through the chunked matmuls; nothing in the arm
calls `torch.utils.checkpoint` (it is not imported anywhere in the package).
Top device kernels in the `matrix_ops` receipt: the two compiled-scan graph calls
(0.294 / 0.245 of device time), `aten::mm` 0.065, `aten::linalg_solve_triangular`
0.055, then cutlass bf16 GEMMs — i.e. dense matmul, a triangular solve, and the MLP.

## 5. Gate D — same-harness throughput, and the registered reference

**Harness (identical for all three rows below).** `x45/gates.py`: the registered
flagship stream, 64 windows × 2048 bytes per step, microbatch 8, seed 3401, AdamW with
the registered gdn2 nomination's hyperparameters, 1 warm-up step then **3 measured
steps**, `torch.profiler` running. Boundary: whole model, whole step = host window
grouping + every microbatch's forward and backward + clip/AdamW; the stream read
(`next(batches)`) is outside every phase, the same boundary `x43/profile_arm_step.py`
uses. `tokens` is the measured target count and is **344,027 in every receipt** (the
same windows), so the rows are directly comparable. Measured parameters: 149,855,670
(`gdn2`), 149,981,808 (both matrix arms).

| arm | scan mode | tok/s | step total (3 steps) | peak alloc | replay absent |
| --- | --- | --- | --- | --- | --- |
| `matrix_ops` | compiled, no CUDA graph | **13,189.7** | 26.083 s | 25.07 GB | ✓ |
| `matrix_append` | compiled, no CUDA graph | **15,855.9** | 21.697 s | 21.93 GB | ✓ |
| `gdn2` | eager (external Triton) | **28,124.0** | 12.233 s | 4.49 GB | ✓ |
| `gdn2` (first run of the same config, pre-correction labels) | eager | 21,795.3 | 15.784 s | 4.49 GB | ✓ |

**The spread matters.** The same `gdn2` configuration measured 21,795 / 28,124 /
29,098 tok/s on three runs of this 3-step harness (the third is quoted from its log;
receipts `cd_gdn2_c128.json` and `cd_gdn2_c128_rerun.json` are two of them). Short
runs are cold-start dominated (Triton autotune, first-iteration allocator growth), so
**the same-harness ratio is indicative, not precise**: `matrix_ops` is 1.65–2.13× slower
than `gdn2` and `matrix_append` 1.37–1.77× slower.

**Different harness, quoted for reference only — not comparable to the table above:**

| registered 2700 s arm (`x43_fork_resolved/runs_150m/`) | bytes / seconds | tok/s |
| --- | --- | --- |
| `lattice_gdn2` | 96,903,492 / 2701.1 | 35,875 |
| `lattice_transformer` | 105,844,355 / 2700.6 | 39,193 |
| `lattice_emender_bounded` (tape) | 36,499,275 / 2704.6 | 13,495 |

Those are 45-minute runs that include a held-out evaluation, on the registered arms;
the table in this section is a 3-step cold-start gate. **They are not the same
measurement and must not be differenced.** The only legitimate GDN-2 comparison this
lane can make is the three-row table above.

**Gate D verdict: not met.** The matrix arms do not reach GDN-2's performance class in
the same harness (1.6–2.1× slower), and the registered-rate comparison (13.2 k here
against 35.9 k registered) would be a boundary violation, so it is not made. What the
numbers do support: the memory block is no longer *the* binding cost (its device time
is 21.7 ms of a ~2.9 s step in the block measurement, §6), and the whole-arm gap is now
in per-token mixer arithmetic and in the twice-wider MLP that parameter matching
imposes, not in replay.

## 6. Block vs block (identical shapes, identical state size)

`x45/bench_blocks.py`, receipt `receipts/block_vs_block.json`. Boundary: **one memory
block, forward + backward of `out.pow(2).sum()`, eager, no `torch.compile`, no CUDA
graph**, batch 8 × 2048 × width 576, 1 warm-up + 3 CUDA-event reps, mean/p50/device
time. State: GDN-2 `d_k × d_v = 128 × 128` per head, 2 heads; matrix store `rank × d_v
= 128 × 128` per head, 2 heads — the same state size.

| block | wall mean | wall p50 | **device** | block parameters |
| --- | --- | --- | --- | --- |
| matrix `matrix_ops` | 113.04 ms | 113.12 ms | **21.70 ms** | 443,522 |
| matrix `matrix_append` | 97.38 ms | 97.53 ms | **13.37 ms** | 443,522 |
| GDN-2 delta-rule (external Triton) | 8.86 ms | 8.57 ms | **5.97 ms** | 1,101,442 |

Device ratio 3.63× (ops), 2.24× (append); eager wall ratio 13.2× — the difference is
host launch count in the eager matrix path, which is why the arm's training path
compiles the scan. Block parameters are not matched here (the block comparison is at
matched *state*), and parameter matching happens at the model level via the MLP.

Note `NDM_PIN_TRITON_AUTOTUNE=0` was required to measure the GDN-2 block: with the
pinned autotune registry its kernel raised `OutOfResources(122880, 101376, 'shared
memory')` at this shape. The unpinned sweep picks a config that fits. This is a
property of the pin, not of the kernel.

## 7. What did NOT work

1. **The read renormalisation.** A parent instruction (later retracted) said the
   certified read divides by `Σ_j a_j s_j`. I implemented it, wrote a test asserting it,
   and then verified the reference: `naive_reference.py` and `x35/causal_read.py` both
   read `Σ_j softmax_j · strength_j · v_j` with **no** renormalisation. Corrected in
   `f3ea30e3`; the test is now `test_read_matches_certified_reference` against the
   executed reference. No receipt ever claimed a read-normalisation mismatch, and the
   `x44` note about `denom` counting versions rather than live slots is **not** a
   strength-normalisation story.
2. **A block rate compared with a whole model.** My working notes carried a
   795 k tok/s *memory-block* figure next to GDN-2's 35.8 k *whole-arm* rate. That is
   a component against a model and is withdrawn; §6 measures block against block and
   §5 measures model against model, each with its boundary stated.
3. **Every graph-captured step timing taken before the fix.** Numbers measured while
   `Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards` was
   firing are discarded. The numbers that remained (≈ 36 k tok/s equivalent from a
   single-script probe) were taken on a configuration that then crashed in the gate
   and are **not** receipted.
4. **Per-module CUDA graphs.** `torch.compile(self._forward_impl, mode="reduce-overhead")`
   on each of the 24 mixers faults with `accessing tensor output of CUDAGraphs that has
   been overwritten by a subsequent run`: the compressed-region outputs live in the
   graph's private pool and the backward (or the next layer's replay) reads them after
   invalidation. `cudagraph_mark_step_begin()` per graphed invocation did not fix it,
   and a telemetry hold on those tensors made it worse. The correct form is a
   whole-step capture; that is not implemented.
5. **One graph for the whole 24-layer forward.** Did not finish compiling in 50 minutes
   (killed). The shipped mode is `torch.compile(mode="default")` on **one shared scan**
   (inductor fusion, no CUDA graph) — one compile instead of 24, no static-pool
   hazards. Compile is ~5–6 minutes per process and is paid before the measured loop;
   that is why each gate run's wall time is ~6 minutes while its measured segment is
   12–26 s.
6. **Dense one-hot addressing of the state.** An earlier directed design expressed the
   three addressing steps as `onehot @ state` and `onehot^T @ delta`. Measured: `rank`
   times the arithmetic (128× at rank 128) and 21.7 k → 9.2 k tok/s at the step level
   with no semantics change. Replaced by device-side gather/scatter onto freshly
   allocated temporaries (no in-place write to the state, no host sync, no dynamic
   shape), which is equally capturable and 2.3× faster. The ops themselves remain
   rank-1 updates; only the addressing changed.
7. **`torch.compile` on the eager chunk loop first:** 43 s/step against 8.8 s/step
   compiled at chunk 128 — the loop is launch-bound, and this is the measured reason
   the train path compiles.

## 8. Deviations from `PREREGISTRATION_MATRIX_EMENDER.md`

1. **Not built on `chunk_gdn2.py` (registration gate 1).** The scan is a PyTorch
   implementation of the same delta-rule WY decomposition; no new CUDA C++ was written,
   but the NVIDIA Triton kernel is **not** on the runtime path. Reasons: its triangular
   solve runs at tf32/bf16 precision (so the exactness gates could not be met through
   it), its key dimension is asserted `≤ 256` and it has no rank-growth form, and the
   one-hot/prefix structure makes the specialised scan simple. The measured cost of
   this deviation is §6 (block device time 3.6× GDN-2's block).
2. **Rank growth implemented but the growth wave was NOT run.** `growth_target`,
   the repeat-write statistic, chunk-boundary widening and the zero-injection are
   implemented and tested (snapping to `{64,…,2048}`, monotone, rows above the grown
   rank exactly zero, trigger reproducible from the recorded schedule), and the arm
   grows with zero new parameters. No growth *arm* was measured.
3. **The 2700 s arms were NOT run.** This lane hit its session ceiling before them, so
   `matrix_append` and `matrix_ops` have no registered 2700 s receipt and **no claim in
   C1–C4 is tested**. §5 is a 3-step gate, not the registered board.
4. **Not integrated into the registered runner.** The arms are built by
   `x45/gates.py`; `x42_pile_scale/lattice.py` was **not** modified, so no new arm name
   exists in it yet and no existing arm, receipt or number was touched.
5. **Read scores are cosines** (learned query in the value space) rather than the tape's
   raw dot against learned keys; the certified *structure* is reproduced and tested
   against the executed reference. §1.
6. **Chunk length 128 tokens** for the gate receipts (routing granularity), matching
   the harness config of the committed `gdn2` base receipt. The chunk sweep measured
   elsewhere in this lane (64 / 128 / 256 → slower / 13.2 k / 22.5 k tok/s in an earlier
   compiled-per-mixer mode) is **not** receipted and its numbers are from a mode that
   is no longer shipped.

## 9. Commands

```bash
python -m pytest experiments/expressivity_tasks/x45_matrix_emender/test_matrix_emender.py -q   # 33 passed
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
python -m experiments.expressivity_tasks.x45_matrix_emender.gates --arm matrix_ops \
    --steps 3 --warmup 1 --chunk-tokens 128 --seed 3401 --out receipts/cd_matrix_ops_c128.json
python -m experiments.expressivity_tasks.x45_matrix_emender.gates --arm matrix_append \
    --steps 3 --warmup 1 --chunk-tokens 128 --seed 3401 --out receipts/cd_matrix_append_c128.json
python -m experiments.expressivity_tasks.x45_matrix_emender.gates --arm gdn2 \
    --steps 3 --warmup 1 --chunk-tokens 128 --seed 3401 --out receipts/cd_gdn2_c128.json
NDM_PIN_TRITON_AUTOTUNE=0 python -m experiments.expressivity_tasks.x45_matrix_emender.bench_blocks \
    --out receipts/block_vs_block.json
```

## 10. Recommended next step

Nothing about the ops or the scan needs further work — gates A, B, C and E are met.
What is left is performance, and it is now a well-localised problem: in the shipped
mode the whole-arm gap to GDN-2 (1.6–2.1×) is per-token mixer arithmetic on an fp32
scan whose block costs 3.6× GDN-2's block in device time, against a state that is the
same size. Two levers, in the order I would take them:

1. **A block kernel.** The scan's device time is a triangular solve plus elementwise
   traffic over `(steps × steps)` coupling matrices that are 99 % structural zeros
   (only same-slot step pairs are non-zero). A slot-sorted, block-diagonal solve, or
   the GDN-2 Triton WY kernels fed one-hot keys, removes both. This is the deviation in
   §8.1, and closing it is where the 3.6× lives.
2. **Then re-measure the registered board**: two 2700 s arms under `runs_150m/` with
   the growth wave as its own lane, and C1–C4 tested against
   `lattice_gdn2`'s 1.1407 nats/byte rather than against a throughput number.
