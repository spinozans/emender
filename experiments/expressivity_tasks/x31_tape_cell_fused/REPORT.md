# X31 part 5 — training works; large-store performance and sparse storage remain unmet

## Implementation

`record_all=True` captures mutation-time before-images of every owned persistent
tensor the scan writes: key/value payloads, strengths/lazy references,
liveness, ages, birth/touch times, heap/inverse positions/due times/terminal
strengths/size, all free bitmap levels, birth links/head, pressure history and
score scratch. This follows physical HBM mutations, including delayed register
spills. `AllFieldJournal.undo_` consumes the final handle and restores a usable
pre-chunk handle. Fresh rollback drops newly created metadata; persistent
continuation restores all owned bytes. Snapshots rebase addresses onto independent
storage; parent journals support rollback across continuation chunks.

The log is **not compact logical-event storage**. Scalar writes skip unchanged
bits; vector spills record every valid element, including unchanged values.
Capture/undo uses ordered 128-element groups, preserving repeated-write
before-images without a chunk-boundary gather. Each scalar entry reserves an
8-byte address, 8-byte exact before-image and 4-byte width/group descriptor.
For N=B*H and C=(6*T+8)*S + T*(K+V+64) + (S+3*T)*(16*bit_length(S)+32),
reserved bytes are N*(20*C+16*T+8), O(T*S+(S+T)*log S). Dense age/score spills
remain. Reservations exceeding the checked int32 element-index range fail before
allocation. With gradients and both recording options disabled, no journal,
address map or replay copy is allocated. The existing payload-only option and
its mutation-time CAP fixture remain supported.

`backward.py` provides a first-order custom `Replay` Function and one reverse
kernel per batch/head store over the whole chunk. Case/target/allocation/pop and
death decisions are replay inputs (G10); routing is never selected again.
Physical writes are undone to each pre-token boundary. The kernel propagates
read key/query/strength weighting, age-1 delta-rule, append-content,
append-logit/error coefficient, pop and imported key/value/strength adjoints.
The append error remains detached per F12g. Read probabilities are differentiable
arithmetic on restored state. Usage instrumentation is unchanged and excluded
from functional-state parity, matching X30.

Autograd retains a private final-state copy and creates separate working storage
for each backward traversal. One binary-search mapping kernel rebases undo
addresses onto those copies. Retained first-order backward is repeatable and
returned forward state is not mutated. Public traces are copied independently,
including H=1 where a transpose can already be contiguous. Training continuation
imports materialized native state and rebuilds metadata; the inference
continuation path retains consuming zero-copy ownership. Private O(S) copies,
dense diagnostics/spills and sequential time remain costs. Higher-order
gradients are unsupported.

## Full parity gate — PASS

Final leased suite: **95 passed in 265.55 s**, lambda01 physical GPU 1,
Python 3.12.3, torch 2.9.1+cu128, Triton 3.5.1. Receipt:
`validation_part5.txt`, JUnit `validation_part5.xml`, and checked source/JUnit
hashes in `parity_part5.json`. No tests skipped. Coverage includes:

- 25 seeded imported-store forward/first-order cases: seeds 0..4,
  S={1,7,16,736,11776}, B=2/H=2/T=9/K=4/V=5.
- K64/V128 gradient cases at S37 (two seeds), S736 and S11776.
- All ten differentiable input paths against `naive_reference` with independent
  read and final-state cotangents, atol=rtol=1e-5; exact discrete traces.
- Imported age ties/dead/low-strength rows, repeated writes and CAP merges,
  mid-chunk pop/death, split continuation, retained-graph repeat backward and
  mutation of the public H1 trace without changing backward decisions.
- Exact byte rollback at S={1,7,37,736,11776}, snapshot/parent independence,
  pressure-ring wrap at 130 tokens and fail-before-allocation index bounds.

All numerical kernels used by the completed S736/H4 learning run are unchanged
in the final source. Later ownership hardening affects the H1 trace-alias case;
large-S forward launch geometry was subsequently matched at four warps for both
recording modes and revalidated before final timing. Large-S backward retains
its separately validated 16-warp configuration.

## Learning gates — PASS

MQAR achieved mean per-class recall **1.0 at update 500** on the registered
20,000-example validation pool, S736/H4/batch64, AdamW lr=1e-4. Receipt:
`mqar_part5.json`; full output `training_part5.txt`. This is an X31 run using the
fused reverse kernel, not an inherited v1 result.

The same process used an in-memory MQAR warm start and completed **all 2500
ladder updates**, k<=128. Final balanced accuracy **0.915**, accept recall
**1.000**, reject recall **0.830**; 1000 examples per class. The registered >0.9
signal was also cleared at update 2000 (0.922). Receipt: `ladder_part5.json`.

| Update | Balanced accuracy | Accept recall | Reject recall |
|---|---|---|---|
| 500 | 0.8975 | 1.000 | 0.795 |
| 1000 | 0.8605 | 1.000 | 0.721 |
| 1500 | 0.8575 | 1.000 | 0.715 |
| 2000 | 0.9220 | 1.000 | 0.844 |
| 2500 | 0.9150 | 1.000 | 0.830 |

Both stages use the X28E generators and frozen embedding/F12g projections.
Ladder stage 2 is omitted as in the v1 pilot. Execution retains right padding
(T128 MQAR, T386 ladder) to avoid shape recompilation; only registered query/EOS
positions are supervised. Ladder uses a physical batch64. These are single-seed,
in-range validation results, not OOD certification or a controlled causal
comparison with the operator-supplied v1 pilot. No model checkpoint was written.

## Registered cost gates — partial failure

Final receipt `cost_part5.json` / `cost_part5.txt`: physical card 1,
NVIDIA RTX 6000 Ada, fp32, B64, H4 unless specified, K64/V128, width256, T16.
Five warmups and 20 measured repetitions; memory accounting uses the fifth
warmup, with no extra forward call. Tables show median **end-to-end wall
milliseconds per token**, including cell projections, fresh-state allocation/
initialization and synchronization. Raw wall and CUDA-event samples are retained.
These are fresh-state forward measurements; training-step/steady continuation
speed is not established.

| Slots | X31 ms/token | X30 ms/token | X28E ms/token | X30/X31 | Registered target |
|---|---|---|---|---|---|
| 736 | 0.450551 | 4.544004 | 2.112711 | 10.085x | PASS |
| 11776 | 78.210305 | 24.970704 | 17.173224 | 0.319x | FAIL |

The large-store >=10x target **fails**. Head scaling at equal B64/per-head slot
budget also **fails** the <=2x target:

| Slots | H1 ms/token | H16 ms/token | H16/H1 | Target |
|---|---|---|---|---|
| 736 | 0.272476 | 1.545363 | 5.672x | FAIL |
| 11776 | 8.474946 | 342.053454 | 40.361x | FAIL |

Checkpoint forward is 77.964218
ms/token against 78.210305 without recording,
an observed delta of **-0.315%**. Both use
four forward warps. This **passes the <=5% timing threshold**; the observed
negative delta is not a claim that capture causally accelerates execution.
Capture, private replay copy and address mapping are included.

The sparse-storage/independence requirement **fails**. At this checkpoint point,
base state is 2,523,785,216 bytes; journal reservation
21,789,476,864; 253,904,384 entries use
5,078,087,680 record bytes; the private forward replay copy adds
2,511,726,592 bytes. These are component sizes,
not a measured peak. `summary_part5.json` records derived reserved sizes at
S736/S11776 and the guaranteed N*(5*T+4)*S dense-record lower bound. Actual
physical event counts and reserved capacity depend on S. A small relative time
delta against this slow forward does not establish compact checkpoints.

Gate 6's certified append mutation prototype measures **S=736: 1.086x / S=11776: 1.091x**, both
>1x and bitwise equal to its sequential mutation baseline. Receipt:
`append_cost_part5.json`. B64/H4, T128, K64/V128, five warmups/20 measurements;
launch/synchronization and a common live-mask reset are included. This excludes
reads, state allocation, routing/management, API certification and exact-curve
preparation; it is a mutation-phase result, not a full-scan speedup.

## Validation chronology and residual blockers

Intermediate completed suites (91/93/94 tests, and 95 before geometry matching)
are retained separately. Early literal/constexpr/atomic-broadcast/tile-width
compile failures and the 131072-byte shared-memory failure on 101376-byte
hardware are retained as failures. Five interrupted compilation attempts have
no completed-suite claim. The trial uniform 16-warp forward failed a checked
int32 reduction (1 failed, 74 passed); it was discarded. The final four-warp
forward passed all 95 tests. No failed or interrupted command is counted as a pass.

The first learning attempt passed MQAR at 500, then was interrupted after
ladder update160 was logged to replace 21 rebasing passes with one map and use
physical batch64. Its receipts remain separate; no 2500-update result is inferred
from it. The final completed learning run is reported above.

Two preliminary cost collections are retained but superseded. The first used
four forward warps without capture and sixteen with capture, confounding its
checkpoint delta. The next had matched geometry but one extra untimed accounting
forward. Final costs use matched four-warp geometry and exactly five warmups.
They collected without errors. `summary_part5.json` separates timing thresholds
from the failed storage-independence requirement.

Remaining acceptance blockers are large-store throughput, head scaling and
logical-event checkpoint compactness. Dense physical spills, conservative
reservation, private copies, padded single-query tensor-core scoring and
sequential time remain limitations. All source/artifacts are confined to X31;
X30/X28E references remain read-only. No .pt, WG, delegated agent,
Frontier/Slurm or resilient-training change is involved. Supervisor/Intercom
tool discovery returned no callable tool, so no coordination messages were sent.

## Historical parts 1–4

The following reports describe earlier implementations. Part 5 above supersedes
their missing-journal/backward/training statements.

X31 currently provides a working device-only sequential **inference baseline**,
a parallel append-mutation/rank-scan prototype, and a sparse rollback primitive.
It does not deliver the requested performance-first training implementation.
Training calls fail explicitly rather than silently using the X30 engine.
No training result, performance speedup or successful acceptance is claimed.

## Part 4 continuation — partial, backward half not delivered

This change adds an opt-in **key/value payload journal** to the live device
scan, not the complete requested training journal. `tape_scan(...,
record_payload=True)` returns it as `FusedState.write_journal`. The kernel
loads each old key/value payload immediately before the append, correction
or CAP write. It does not gather from a chunk boundary. Repeated writes to
the same slot therefore retain their intermediate before-images.
`WriteJournal.undo_payload_` reverses those writes exactly on detached owned
payload tensors. It neither evaluates routing nor restores a usable full
FusedState. Snapshot branching copies the journal as well as native metadata.
Recording is disabled by default and the disabled kernel specialization
does not allocate journal buffers.

Storage reserves one record per token and store, with -1 padding for no
write: B*H*T*(8+4*(K+V)) bytes, independent of slot budget. This is O(tokens),
**not compact O(actual events)** for sparse writes. Strengths, lazy references,
birth/touch times, liveness, ages, death heap, free bitmap, birth links and
pressure history are **not journaled**. In particular POP, strength touches
and deaths remain unjournaled. The payload journal cannot satisfy item 1 or
serve as a training checkpoint. There is **no custom autograd Function or
fused backward**, and training continues to fail explicitly. This is a
partial engineering change, not completion of the part 4 contract.

Seven new tests check before-images against successive one-token oracle
boundaries with S={1,7,736,11776}, two batches/two heads, popped/dead imported
rows and repeated writes, then verify bitwise payload rollback. A dedicated
CAP fixture requires old values [2,8,9.5,9.875], demonstrating that a boundary
gather would fail. Journal memory is checked against the formula above;
snapshot isolation, invalid-slot rejection and rollback across two consuming
continuation chunks are checked as well. The continuation test verifies that
recording preserves native payload storage identity.

Validation receipts: the first invocation incorrectly supplied an extra
pytest path to a wrapper that already supplies the test directory. It
exited 4 with zero tests collected (`validation_part4_attempt.txt`); it is
not a validation pass. The corrected leased-card subset completed **6 passed,
49 deselected in 83.26 s** including compilation, lambda01 physical card 1,
Python 3.12.3 (`validation_part4_journal.txt`). The complete suite then
completed **55 passed in 221.90 s**, including compilation, on the same
leased physical card (`validation_part4.txt`). That suite collected before
the final snapshot-isolation assertions and continuation test were added;
the final payload subset is recorded separately below. These checks are
forward and payload rollback checks only.

Final payload subset: **7 passed, 49 deselected in 6.02 s**, broker-leased
physical card 1, `validation_part4_final_journal.txt`. This run includes the
final snapshot assertions and continuation test. Scoped `git diff --check`
passed. There were no interrupted or compile-failure runs in this part;
the retained exit-4 invocation is a command-line collection error.

The full forward-and-first-order-gradient parity gate remains **UNMET**.
MQAR, the 2500-update ladder (both class recalls), and all cost gates remain
**NOT RUN / NOT MEASURED**. No timing or training claim is authorized by the
new journal checks. The missing all-field journal, ownership-safe autograd
and G10 reverse adjoints are residual blockers to the requested completion.
No X30/X28E files were edited. No .pt artifact, WG, delegated agent, Frontier
run or resilient-training change is involved. Tool discovery found no
`contact_supervisor` or Intercom tool in this session, so no supervisor
message could be sent. The historical sections below retain their original
receipts; this section supersedes their journal implementation status only.

## Part 3 implementation status — incomplete

This continuation implements a device birth-order index, persistent chunk
metadata and a tiled `tl.dot` scoring path. Mutation-time journaling, the
custom autograd Function and fused reverse replay remain **NOT IMPLEMENTED**.
The full forward-and-first-order-gradient gate therefore remains **UNMET**.
MQAR, ladder (including per-class recalls) and every registered cost gate
remain **NOT RUN / NOT MEASURED**. Preregistered targets are unchanged.

`birth_index.py` stably sorts imported live slots by age then slot ID once,
and maintains a doubly linked device order through appends, pop deaths and
scheduled deaths. Target selection no longer reduces over the slot vector.
Youngest fallback and lowest-slot birth ties are preserved. Ordinary appends
prepend and removals take constant pointer work. This is not X30's complete
birth-group index: imported age-zero ties or future births can require a
linear pointer walk. No bounded-depth lookup or performance claim is made.

`FusedState` owns its native buffers and metadata. Continuation reuses heap,
inverse positions, due times, terminal strengths, birth/touch timestamps,
free bitmap, birth links, heap size, lazy strength references and scoring
scratch. Native TapeState import still copies once and initializes metadata;
continuation does neither reconstruction nor full-state cloning. The kernel
still loads/spills dense read metadata and emits dense death diagnostics.
Lazy references remain unmaterialized internally across chunk boundaries.
The absolute token clock is a runtime argument to avoid compiling a distinct
kernel for every chunk position.

**API change:** passing a returned FusedState consumes that handle and mutates
its buffers in place. Reuse raises ValueError. Call `state.snapshot()` before
continuation to retain an independent branch; snapshot explicitly costs O(S).
Native TapeState imports remain unchanged. Historical native tensor aliases
from a consumed handle are not immutable snapshots. Do not concurrently use
the same owned state. This inference ownership model does not establish
autograd safety or imported-state gradients.

Scoring now tiles slot columns by 32 with `tl.dot`, using fp32 `tf32x3`
precision. Each recurrent query pads to M=16 and K pads to at least 32;
only its first output row is retained. Stores remain separate programs on
the B*H grid; time remains sequential. Value weighting remains a reduction.
The wasted query rows and scratch traffic may be costly. This establishes a
tiled scoring implementation, not an efficient cross-head GEMM or any speedup.

Final leased GPU receipt: **49 passed in 221.41 s**, including compilation,
on lambda01 physical card 1, Python 3.12.3. Full output:
`validation_part3.txt`. The existing 25 seeded imported-state cases at
S={1,7,16,736,11776} still satisfy forward/state atol=rtol=1e-5 and exact
discrete traces, including split continuation. Six added cases cover three
birth-order fixtures (ties, pop deaths, low imported strengths, reuse and
negative imported ages), persistent pointer identity and independent snapshot
branches, and key dimensions 31/64 with nonmultiple-of-32 slot count 37.
These are forward checks, not first-order-gradient validation.

Validation chronology: an intermediate birth-only implementation completed
43 tests in 251.79 s (terminal output); the first tiled build then failed
compilation, with 15 passed / 1 failed, because tile-width computation ran
inside JIT (`validation_part3_compile_failure.txt`). Moving the width to a
host constexpr fixed it. The next implementation completed 47 tests in
217.60 s (`validation_part3_intermediate.txt`). The final 49-test run includes
the runtime absolute clock and the two additional key-dimension cases. None
of these runs was interrupted. Pytest durations include compilation and
are not cost receipts. Manual inspection of a newly cached scan PTX found
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` instructions; this is
code-generation evidence only. Scoped `git diff --check` passed.

The implementation is a partial delivery of items 1–2, not completion of the
part 3 task. Items 3–6 remain outstanding. No .pt artifacts, WG services,
delegated agents, Frontier runs or resilient training changes are involved.

## Historical parts 1–2 record

The descriptions and ledger below record the state before part 3; the current
implementation status above supersedes their missing birth-index/tiled-read
and per-chunk-copy statements. Their training and gradient limitations remain.

## Built and checked

`cell_fused.py` supplies F12g projections and a functional-state scan adapter.
`kernels.py` launches one program per batch/head store and loops over the whole
input chunk on device. Read precedes pop/write. POP uses query argmax; WRITE
uses the lowest-index age-1 slot, with youngest-live fallback only for CAP.
Allocation is lowest free index. Keys, values and strengths use fp32 HBM.
The kernel emits discrete decisions, deaths and a rolling 128-token pressure
register. The host contains no token loop or scalar token routing.

Part 2 ports X30's integer-mantissa interval jumps to Triton (`lazy_decay.py`).
The scan keeps reference strengths, birth and touch timestamps; access jumps
match sequential fp32 multiplication, including nonunit popped/imported
references. A cancellable device min-heap (`device_schedule.py`) supplies death
events and their exact terminal strengths. Inverse positions bound its storage
to O(S), including repeated touches. Death processing visits scheduled events,
with scalar metadata updates; it does not sweep for decay deaths. Already-low
imported live strengths retain the oracle's first-read-then-pop-death behavior.

Lowest-free allocation uses the three-level 64-bit GPU bitmap in
`device_free.py`. Birth timestamps replace recurrent age increments, but
birth-group target selection still reduces over the store: X30's sparse birth
index is **not implemented**. Import reconstructs the heap and bitmap each
chunk, so continuation still pays O(S) initialization and full state clones.
Dense read metadata spills/reloads and dense diagnostic death traces remain.
Read scoring still uses dimension-wise reductions rather than tensor cores;
head stores are independent programs, not a demonstrated GEMM batch.

Real GPU validation on lambda01, leased physical GPU 1 (RTX 6000 Ada),
PyTorch 2.9.1+cu128, Triton 3.5.1, Python 3.12.3:

- 31 tests passed in 3.38 seconds via `run_leased.sh -q -x`.
- 25 randomized imported-store cases: 5 seeds x slots {1,7,16,736,11776}; batch=2,
  heads=2, chunk=9, K=4, V=5, decay=.8. Outputs and functional state match
  X30 naive_reference at atol=rtol=1e-5; all discrete trace fields match exactly.
- Continuation splits each random case at token 4 and checks read/state and
  pressure equivalence. Imported age ties, dead rows and low strengths occur.
- Dedicated exact-death boundary, lowest-index age tie, zero-residual append,
  CAP pressure and 128-token pressure expiry checks pass.
- Free-rank scan, parallel append writes and sparse reverse rollback with
  repeated slot IDs/padding pass. Journal fixture is 60 bytes independent of
  backing S=7 vs 11776.

These tests establish **forward baseline parity on the checked cases only**.
They do not establish the full first-order gradient gate or production-dimension training parity.
An initial CPU Triton-interpreter attempt failed on unsupported tensor `**2`;
the implementation was corrected to explicit multiplication before GPU checks.
The first GPU subset passed 16 tests; the intermediate expanded GPU suite passed 20, and the final suite passed 31.

## Acceptance ledger

| Requested gate/deliverable | Result |
|---|---|
| Device-side sequential chunk scan | Implemented baseline; exact forward cases pass |
| Tensor-core-friendly batched scoring | Not implemented |
| Lowest-free sparse allocator and lazy scheduler | Device bitmap and exact lazy heap implemented; sparse birth index and persistent chunk metadata missing |
| Trace/state/first-order gradient parity | Partial: forward only; gradient gate unmet |
| Sparse captured delta checkpoints + fused backward | Not implemented; standalone rollback primitive only |
| MQAR >=99% by 5000 updates | Not run; training unsupported |
| Ladder >0.9 signal by update 2500, class recalls | Not run; no accuracy/recalls to report |
| S=736/11776 cost vs X30/X28E | Not measured; no accepted performance claims |
| >=10x X30 at S=11776 | Not established |
| H=16/H=1 <=2x | Not measured |
| Checkpoint overhead <=5% at S=11776 | Not measured; forward still copies state densely |
| Parallel decomposition | Written with fp32/routing qualifications |
| Append-phase measured speedup | Not measured; mutation prototype correctness only |

## Limits and next engineering work

The core mission remains incomplete. Finish the sparse birth index and
persistent metadata, then implement the tiled read path before treating this
as a performance candidate. Forward must capture old event fields at mutation time;
a boundary gather is wrong for repeated writes. Then implement reverse replay
adjoints with fixed recorded decisions (G10), test learned content, pop, query,
append and imported-state gradients, and remove per-chunk store copies.
Only after that complete parity gate may the registered timing and training
runs start. O(events) checkpoint metadata does not eliminate dense read and
read-gradient work. Usage EMA is intentionally not updated by this baseline;
it is excluded from the functional-state gate, matching X30's scope.

All changed source is confined to X31. X30 and X28E references are imported
read-only. No resilient DiLoCo behavior or Frontier job is involved; resilient
requirement IDs and Slurm Partition/QOS evidence are not applicable. No WG
service or agent delegation was used. The requested `contact_supervisor` tool
was not present in the available tool catalog; no supervisor contact occurred.

## Part 2 scope and validation

This is a **partial continuation**, not completion of the part 2 contract.
The tiled read, mutation-time captured journal, fused reverse replay and
custom autograd Function remain absent. No first-order gradient parity is
claimed. Consequently MQAR, ladder (including class recalls), and all cost
measurements remain NOT RUN. The numeric S=736 target is now explicitly >=1x
X30; the existing >=10x at S=11776, H16/H1 <=2x, and <=5% checkpoint targets
remain registered before any measurement.

New checks compare device jumps, exact horizons and terminal strengths
bit-for-bit with X30 across 128 references for each of seven decay rates
(.125, .5, .8, .99, .9995, 1-2^-24, 1), plus popped references and immediate
threshold neighbors. Cancellable-heap tests exercise 180 events on each of
two stores for three decay rates, including rescheduling, ties and no-death
decay=1. A bitmap fixture exercises high bits at all three levels up to slot
262143, full-store exhaustion, and reuse. These complement the original
25 randomized forward/continuation cases, not the missing gradient gate.

Development diagnostics initially found an int32 overflow in the interval
normalization constant and a mismatch between raw Python decay constants
and their fp32 ratios; both were corrected. Initial multiwarp integration
also encountered illegal memory access. Device mutation helpers now include
program barriers between dependent device metadata updates, and the
death drain is capacity-bounded with enabled device assertions. Intermediate
full-suite runs were interrupted while replacing implementations; they do
not count as validation passes. Large-store compilation time is not a timing
receipt. Final validation output is recorded separately in
`validation_part2.txt`.

Final leased GPU run on lambda01 physical card 1: **43 passed in 259.84 s**,
including compilation. All 25 seeded randomized forward and continuation
cases still pass at 1e-5, with exact routing/death traces. The 12 new checks
pass. This validates the checked forward implementation and scheduler/
allocator primitives; the complete forward-and-gradient acceptance gate is
still **UNMET**. No performance or learning success is established.
