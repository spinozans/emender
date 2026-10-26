# X31 part 8 — checkpoint gate passes; large-store head scaling remains open

**Partial gate closure with a measured representation limit.** Final source passes
**109 tests in 15.52 s**, retaining all original 105 and adding four regressions.
Checkpoint overhead is **+4.874%, PASS**, versus part 7's +17.398%.
H16/H1 at S11776 is **4.775x, FAIL**, versus 5.135x. All other registered
throughput, small-store head, append and storage gates pass. The summary honestly
retains `all_registered_targets_passed=false`.

## Parts 5–8 convergence — unchanged gates and collection protocol

| Registered measurement / gate | Part 5 | Part 6 | Part 7 | Part 8 |
|---|---|---|---|---|
| Parity tests | 95 | 97 | 105 | 109 |
| X31 ms/token, S736 H4 | 0.450551 | 0.234388 | 0.213491 | 0.205282 |
| X31 ms/token, S11776 H4 | 78.210305 | 1.262202 | 0.298316 | 0.267132 |
| X30 speedup S736 (>=1) | 10.085x PASS | 20.735x PASS | 20.180x PASS | 22.352x PASS |
| X30 speedup S11776 (>=10) | 0.319x FAIL | 21.472x PASS | 111.947x PASS | 93.109x PASS |
| H16/H1 S736 (<=2) | 5.672x FAIL | 1.474x PASS | 1.271x PASS | 1.200x PASS |
| H16/H1 S11776 (<=2) | 40.361x FAIL | 21.618x FAIL | 5.135x FAIL | 4.775x FAIL |
| Checkpoint overhead (<=5%) | -0.315% PASS | +79.756% FAIL | +17.398% FAIL | +4.874% PASS |
| Append S736 (>1) | 1.086x PASS | 1.062x PASS | 1.059x PASS | 1.079x PASS |
| Append S11776 (>1) | 1.091x PASS | 1.063x PASS | 1.076x PASS | 1.057x PASS |
| Equal-stream storage independence | FAIL | PASS | PASS | PASS |
| All registered gates | FAIL | FAIL | FAIL | FAIL |

These are each part's own matched references, not a reused X30 denominator.
Part 8 H4 X31 time improves from 0.213491 to 0.205282 ms/token at S736 and
0.298316 to 0.267132 at S11776. The large-S speedup ratio falls from 111.947x
to 93.109x because its measured X30 baseline changes from 33.395531 to
24.872228 ms/token. X31 itself is faster at both registered H4 points.
Raw samples are retained; H1/S736 changes from 0.204663 to 0.209048 ms/token,
so this is not a claim that every head configuration improves in every sample.
Part 5 learning results remain historical; no learning rerun occurred.

## Changes and boundaries of the lazy path

Scoring is now a same-store fp32 key/query reduction inside `read_tiles`.
There are no unused cross-head output rows and no separate scoring launch on
the active cell path. Three stream-ordered launches per token serve all B*H
stores, with no host loop over heads. The older `score_heads` helper is retained
but unused. No tensor-core throughput claim is made for the new reduction.

Fresh empty-origin import bounds its expensive vector initialization to the
reachable prefix (rounded to a power of two). Inference continuation continues
to reuse owned metadata, without reconstruction. Inverse heap positions initialize
to -1 across the capacity so a newly reached row is safe in a later chunk.
This is **not a fully lazy dense representation**: native zero initialization,
several metadata fills/clones, free-bitmap setup and dense death diagnostics remain
O(H*S). Arbitrary native imports still clone native fields, construct their birth
index, and initialize against the full grid. Autograd continuation imports native
state again; it does not reuse inference metadata.

The heap audit finds O(log live) work, rather than a mandatory log(S) walk on
empty-origin streams: with T16 it cannot contain more than 16 newly allocated
live rows. Ordinary monotonic births insert at the head. Arbitrary imported
future/tied births can still walk O(S), and heap operations can reach O(log S)
when imported live count reaches S. No timer wheel or bounded birth-group
replacement was delivered. Those are explicit remaining design gaps.

## Why the S11776 head gate still fails

| Slots | H1 ms/token | H4 ms/token | H16 ms/token | H16/H1 |
|---|---|---|---|---|
| 736 | 0.209048 | 0.205282 | 0.250929 | 1.200x |
| 11776 | 0.190429 | 0.267132 | 0.909208 | 4.775x |

`structural_cost_part8.json` uses the same five warmups, twenty samples,
allocation/synchronization inclusion and T16 normalization to measure native
`init_state` alone at S11776: H1 **0.042732**, H4 **0.159212**, H16
**0.631451 ms/token**. The head gate allows the entire H16 call only
2 * 0.190429 = **0.380859 ms/token**. Measured native initialization alone is
**3.316x the entire H1 call**, before projections, scan or diagnostics.
The H16 native payload contains 9,261,023,232 bytes of keys/values alone.

This receipts a limiting component of the **current dense zero-filled public
state representation**, not a universal lower bound or proof that lazy/compact
backing could never pass. Slot-capacity-independent backing with explicit
materialization at public native-state access would be a larger representation
change; it is not delivered here. Replacing heap operations alone cannot remove
this measured native-initialization cost. We do not slow H1 or exclude state
initialization from the registered timing to manufacture a passing ratio.

## Lean capture and deferred replay

`journal_store` emits before-images at mutation sites and updates its device
count with ordinary single-writer loads/stores. Journal-only atomics and barriers
are removed. Event order is implicit in each store's program. Width/length tags
still batch reverse undo of a vector; they impose no forward synchronization.
Management-helper barriers and backward repeated-address ordering remain where
needed for correctness.

Review found that the old final materialization launch used parallel tiles per
store, violating the new non-atomic count invariant. It now uses one writer per
store and loops over its tiles. Two new decay=1, all-live, multi-tile import tests
at S97/S736 require exact rollback of every age field. This fixes an uncovered
race despite the original suite passing. Dense imported-state materialization
performance was not measured and may suffer from the serial tile traversal.

The training forward retains final owner references; it copies no replay owners
and performs no address rebasing. Backward clones and rebases only when consumed.
A consuming inference or training continuation freezes retained replay owners
before mutation. This deliberately moves a dense copy onto that continuation
when an earlier graph still needs it. Repeated backward remains isolated.
Public PyTorch in-place edits to retained owners now raise a version-check error
on backward instead of silently replaying edited data. This is an API restriction
compared with the previous eager private copy. Snapshot branches remain separate.

Backward retains the kernel's existing private routing/diagnostic tensors rather
than cloning public traces. Public trace construction copies only if its layout
would otherwise alias those tensors, including H1/singleton axes. The original
public-trace-tampering gradient regression remains green. Capture therefore adds
no private state-metadata or trace copies to the fresh forward path.

The registered S11776/H4 checkpoint point is **0.280151 versus 0.267132 ms/token**:
**+4.874%, PASS with a narrow margin**. The matched projected-input scan ablation
measures capture off/on at **0.245162 / 0.249189 ms/token**, **+1.643%**.
These are component experiments, not additive attribution or an irreducible
HBM event-emission floor. Backward still scans fixed journal capacity to rebase;
no backward-time gate or elimination of that scan is claimed.

Base state is 2,523,785,216 bytes. Checkpoint retains 113,444,864 journal bytes and
691,200 entries. Retained owner address space is 2,511,726,592 bytes, aliasing the
forward state; actual forward replay-copy bytes are **zero**. The legacy raw
`cost_part8.json` field `private_replay_copy_bytes` reports that retained address
space; `summary_part8.json` explicitly separates it from actual copied bytes.
Equal-stream storage still records 855,040 entries at both budgets and reserves
92,473,344 / 113,444,864 bytes (ratio 1.227). There is no total-memory-independence,
peak-memory, tight actual-event reservation, or arbitrary-import sparsity claim.

## Validation, provenance and residual risks

Final parity and all collection used broker-leased **physical GPU 1**, lambda01,
RTX 6000 Ada, Python 3.12.3 at `/home/erikg/emender/.venv/bin/python`,
torch 2.9.1+cu128 and Triton 3.5.1. Original B64/T16/K64/V128/width256,
fp32, fresh-state end-to-end wall medians, five warmups and twenty samples are
unchanged. Source/JUnit attestation precedes sequential timing/storage collection
on a single lease. No collection errors. GPU7 was never requested or used.

Final evidence: `validation_part8.txt/.xml`, `parity_part8.json`,
`parity_attestation_part8.txt`, `performance_part8.txt`, `cost_part8.txt/.json`,
`append_cost_part8.txt/.json`, `storage_part8.txt/.json`,
`structural_cost_part8.txt/.json`, and `summary_part8.txt/.json`.
The four added cases cover deferred replay across inference continuation and
repeated backward, public owner in-place mutation detection, and two multi-tile
exact-rollback fixtures. Full original forward, all ten first-order adjoints,
import, continuation, snapshot, pressure and trace parity remain green.

Chronology is retained rather than overwritten: the unchanged baseline passed
105 tests in 15.53 s; the capture prototype passed 105 in 319.87 s and is recorded
in `part8_lean_attempt.patch.json`. `_before_materialize_fix` passed 107 tests in
83.45 s but its collection was invalidated by the review-discovered multi-tile
race; collection had already completed before the attempted cancellation.
`_before_private_trace` passed 109 tests in 27.60 s and measured +5.036%
checkpoint overhead, motivating the final trace-copy removal. Only unsuffixed
receipts attest final source. No unsuccessful or superseded receipt is counted
as final validation. The narrow final checkpoint margin remains sensitive to
timing variability; large-store head scaling remains the one failed gate.

No .pt, WG, delegated agent, Frontier/Slurm or resilient-training change occurred.
Resilient requirement IDs and Partition/QOS evidence are inapplicable. Tool
catalog discovery found no callable `contact_supervisor`; no supervisor contact
was possible. Only files under `x31_tape_cell_fused` were changed.

---

# X31 part 7 — scalar recurrent metadata; two timing gates remain failed

**Partial delivery, not closure of the last two gates.** Final validation is
**105 passed in 338.18 s**, including all original 97 tests and eight new tests.
S736 and S11776 throughput, append-phase and the registered equal-stream storage
fixture pass. Large-store head scaling is **5.135x** (target <=2), and
checkpoint overhead is **+17.398%** (target <=5%).
`summary_part7.json` records `all_registered_targets_passed=false`.
Part 6 and earlier records below remain historical evidence.

## Implementation and precise limits

The steady recurrent mutation program no longer loads/spills S-length strength,
live, age, reference or touch vectors. POP uses a scalar winner produced by
`reduce_read`; emptiness uses the birth-list head. POP, allocation and each
scheduled death update their scalar metadata directly. Dense death traces are
zero-initialized once and receive event-indexed writes. Live strengths and ages
materialize once at chunk end. Fresh import still initializes O(S) metadata in
the first core launch and walks imported low-strength rows after their first read.

This is **not the full O(1)-amortized metadata design requested**: the existing
cancellable min-heap remains O(log S), and ordered birth insertion/selection can
walk O(S) with arbitrary imported future/tied births. No timer wheel or bounded
birth-group replacement is delivered. The ordinary steady path is scalar, but
those import and event-data-structure qualifications are acceptance limits.

`score_heads` packs up to 16 head queries into the 16 MMA rows, with batch,
head-group and slot-tile launch axes. K is processed in 32-wide tiles to fit the
Ada shared-memory limit. Each head has its own keys, so the grouped GEMM forms
cross-head scores and retains only each store's diagonal. This is a shared
multi-query GEMM, **not a proof of H-independent useful arithmetic**: off-diagonal
outputs are unused. Local weighting and parallel reduction follow scoring;
time remains sequential with four launches per token. Known empty-origin states
carry a conservative cumulative append bound, limiting score/read/materialization
grids to reachable rows; unknown imports retain full grids. Empty birth indexes
initialize directly, avoiding two unnecessary full-store sorts.

Capture emits vector before-images directly at mutation sites, including one
vector key-write group instead of DK scalar journal groups. There is no vector
spill scan or proposed-value staging. Scalars still compare their own old/new
bits in O(1), avoiding unchanged management records. Device atomics maintain
counts. A host-known conservative live upper bound selects fixed capacities;
there are **no `.item()`/`.cpu()` live-count or final-sizing reads in the training
path**. `compact_` is a compatibility no-op. Address-map transfer uses pinned
memory and a nonblocking device copy. This removes the identified sizing
synchronizations; it does not certify an absence of every allocator/runtime stall.

Internally created empty training imports consume their private native buffers,
avoiding the former extra dense import clone. Private replay copies only allocated
payload rows in this case; inactive replay payload reads are masked. Repeated
backward remains isolated. Unknown/nonempty imported training states still copy
dense payloads; nonpayload replay owners are cloned, and address rebasing scans
the fixed capacity. Ordered capture groups still pay atomics and barriers.
These retained structures, plus fresh H*S native allocation/initialization and
dense diagnostics, are remaining costs, not isolated component-time attributions.

## Final registered receipts

All final GPU work used broker-leased physical **GPU 1**, lambda01, RTX 6000 Ada;
Python 3.12.3 at `/home/erikg/emender/.venv/bin/python`, torch 2.9.1+cu128,
Triton 3.5.1, fp32. The source/JUnit attestation preceded collection. Five warmups,
twenty measurements, median end-to-end wall time per token and raw CUDA samples
are retained. B64/T16/K64/V128/width256, fresh state, allocation and synchronization
are included. H1, H4 and H16 were all measured at both budgets; no collection errors.

| Slots | X31 ms/token | X30 ms/token | X28E ms/token | X30/X31 | Speed gate |
|---|---|---|---|---|---|
| 736 | 0.213491 | 4.308314 | 2.117148 | 20.180x | PASS |
| 11776 | 0.298316 | 33.395531 | 17.175472 | 111.947x | PASS |

| Slots | H1 ms/token | H4 ms/token | H16 ms/token | H16/H1 |
|---|---|---|---|---|
| 736 | 0.204663 | 0.213491 | 0.260184 | 1.271x |
| 11776 | 0.204322 | 0.298316 | 1.049276 | 5.135x |

H scaling passes at S736 and fails at S11776. At S11776/H4, checkpoint forward is
0.350217 ms/token versus
0.298316: **+17.398%, FAIL**.
The two blockers remain open despite improvement from part 6's 21.618x head ratio
and +79.756% checkpoint delta. Base/private replay address spaces remain dense.

Append-phase speedups: S736 **1.059x**;
S11776 **1.076x**, both PASS with bitwise equality.
This remains the registered mutation-phase fixture, with common reset/launch/sync,
not an end-to-end append scan or integrated rank-scan claim.

The equal-append storage fixture records **855,040 entries** at either S and
retains **92,473,344 / 113,444,864 bytes** (ratio 1.227), so the unchanged <=1.5
independence fixture passes. Capture staging is zero. Fixed retained capacities
are 18,048 / 22,144 entries per store. **This gives up part 6's tighter actual-count
compaction:** reservation is not guaranteed <= twice the largest actual event
count. Unknown imported states reserve against S. Explicit zero-valued payload
mutations now emit records even when their bits were already zero.

At the actual S11776/H4 checkpoint point: base 2,523,785,216 bytes; fixed journal
113,444,864 bytes; 691,200 records (13,824,000 record bytes). Private replay reserves
2,511,726,592 address-space bytes; this is no longer a count of bytes copied in
empty-origin forward. Its payload copy visits at most B*H*T allocated rows.
No peak-memory or total-memory-independence claim follows.

Receipts: `validation_part7.txt/.xml`, `parity_part7.json`,
`parity_attestation_part7.txt`, `performance_part7.txt`, `cost_part7.txt/.json`,
`append_cost_part7.txt/.json`, `storage_part7.txt/.json`, `summary_part7.txt/.json`.
The eight new checks cover empty-origin forward/first-order/repeated-backward
parity at H1/H4/H16 and S736/S11776, plus production K64/V128 H16 forward/state/
trace checks at both budgets. Original imported-state, all ten adjoints, exact
rollback, snapshots, continuation and trace-ownership checks remain passing.

## Chronology and residual work

All unsuccessful/superseded receipts remain explicitly named and are not passes
for final source. `_inprogress_source_attempt` failed after 13 passes because a
lazy import observed an in-progress dataclass change; `_score_ownership_failure`
failed byte-exact persistent score rollback after one pass. Scores now use
separate ephemeral scratch. `_before_sparse_replay` passed 97 tests.
`_stages_resource_attempt` passed 103 tests, but production H16 timing exceeded
shared memory (131,072 requested, 101,376 available). Merely setting one pipeline
stage still failed the added production H16 test after 96 passes; this is
`_h16_resource_failure`. K32 score tiles fixed it. `_before_owned_import` then
passed 105 tests and measured 8.597x / +91.478%; the final private-import ownership,
empty birth initialization and reachable-grid changes received the complete
105-test run before all final measurements above. No interrupted run is a pass.

Remaining work includes dense native/replay representation costs, efficient
same-store multi-head scoring without wasted cross pairs, ordered capture cost,
fixed-capacity rebasing and metadata clones, a timer-wheel/bounded birth index,
and a tighter device-only retained-capacity scheme. These are structural limits;
the last two gates are **not closed**. Learning was not rerun; part 5 remains
historical MQAR/ladder evidence. No .pt artifact, WG, delegated agent, Frontier,
Slurm or resilient-training change occurred; resilient requirement IDs and
Partition/QOS evidence are inapplicable. Tool discovery found no callable
`contact_supervisor`, so no Intercom contact was possible.

---

# X31 part 6 — speed and compact-log gates pass; head scaling and checkpoint time fail

This is a partial performance completion. **All 97 tests pass**, including
all 95 part 5 tests and two new compact-storage checks. Both registered
end-to-end speed targets and both append-phase targets pass. Large-store
H16/H1 scaling and checkpoint time remain acceptance blockers.

## Final implementation

`read_tiles` grids over `(B*H, ceil(S/32))`: tf32x3 score MMA, exact lazy
strength evaluation, local softmax statistics and weighted values run on slot
tiles. Empty tiles skip payload loads and MMA. `reduce_read` grids over stores
and eight-value output blocks to combine partials with stable softmax rescaling.
The recurrent core uses tile maxima/argmax for POP and evaluates exact decay
only for the selected POP row. A final slot-tile pass materializes live strengths
and ages once per chunk. The per-store O(S*V) weighted read and per-token
full-store exact-decay calculation are gone.

Time remains sequential: three stream-ordered launches per token plus one final
materialization launch. The core still holds O(S) metadata vectors and spills;
it is not an O(1) metadata-only core. Heads batch across the launch grid, but
each recurrent query still pads to 16 MMA rows. There is no shared multi-query
head GEMM, cooperative barrier, token-group amortization or integrated append
rank-scan path. Fresh owned inference state avoids an unnecessary native clone.
Free bitmap words initialize in parallel; heap import visits live birth links.
Inactive imports skip horizon work. Empty heap entries initialize to -1 and
heap sizes to zero, removing allocator-dependent capture events.

All-field capture compares exact bits and logs only changed elements, preserving
mutation-time before-images and ordered undo groups. Small groups (<=128 lanes)
pack directly in registers, including concurrent final-materialization tiles;
large vector spills use bounded staging and compaction. Score/partial-read
scratch is ephemeral; the persistent score owner remains unchanged and still
participates in byte-exact rollback. Every existing owned-state field retains
its rollback semantics. CAP before-images [2,8,9.5,9.875], repeated writes,
parent/snapshot rollback and retained first-order backward remain covered.

Capture reservation bounds initially live rows plus possible appends, with heap
depth dependence. After capture, buffers clone into at most twice the largest
store's actual event count per store, plus per-token routing records. Cloning
also releases oversized backing storage for a single-store log; this is tested.
All-unchanged vector elements reserve no retained records. Counting live rows
and final sizing introduce host synchronization. Capture work still scans vector
spills; compact retained storage does not imply O(events) execution time.

## Final parity and registered timing receipts

Final suite: **97 passed in 17.22 s**, no skips or failures, broker-leased physical
GPU 1 on lambda01. The final run reused compiled kernels.
The prior complete read/decay run passed 97 in 577.28 s including compilation;
the final change only initialized fresh heap metadata deterministically.
Receipts: `validation_part6.txt/.xml`, `parity_part6.json` and
`parity_attestation_part6.txt`. The 25 seeded forward/gradient cases, four
K64/V128 gradient cases, all ten differentiable paths, S through 11776, exact
traces and consuming continuation still pass at atol=rtol=1e-5.

All final performance/storage collection ran sequentially under one broker lease
on physical GPU 1, NVIDIA RTX 6000 Ada Generation; Python 3.12.3, torch
2.9.1+cu128, Triton 3.5.1, fp32. Every timing uses five warmups,
twenty measurements, wall-clock medians and retained CUDA-event/raw samples.
Allocation and synchronization are included. Dimensions remain B64/T16,
K64/V128/width256. Forward and checkpoint use matched four-warp geometry.
Checkpoint accounting uses warmup five; there is no extra accounting forward.

| Slots | X31 ms/token | X30 ms/token | X28E ms/token | X30/X31 | Speed gate |
|---|---|---|---|---|---|
| 736 | 0.234388 | 4.860111 | 2.364095 | 20.735x | PASS |
| 11776 | 1.262202 | 27.101754 | 17.227776 | 21.472x | PASS |

| Slots | H1 ms/token | H4 ms/token | H16 ms/token | H16/H1 | <=2 gate |
|---|---|---|---|---|---|
| 736 | 0.219299 | 0.234388 | 0.323192 | 1.474x | PASS |
| 11776 | 0.279248 | 1.262202 | 6.036703 | 21.618x | FAIL |

At S11776/H4, checkpoint forward is
**2.268884 ms/token** versus
**1.262202** without capture:
**+79.756%**, FAIL against <=5%.
Head scaling at S11776 is **21.618x**, FAIL against <=2.
S736 head scaling passes. No all-gates-passed claim is made.

Append mutation-phase speedups are **1.062x**
at S736 and **1.063x** at S11776:
both PASS (>1x), with bitwise state equality. B64/H4/T128/K64/V128, five warmups
and twenty measurements; common live-mask reset, launch and synchronization are
included. Reads, allocation, routing, certification and curve preparation remain
excluded. This is a mutation-phase result, not an end-to-end append scan claim.

## Journal storage — PASS within the registered log scope

The equal-stream fixture uses B64/H4/T16, initially empty stores, and 4096
appends at either budget. Both runs capture exactly **99,328**
changed-element entries and retain exactly **2,689,024 bytes**
(including routing/count arrays). Actual underlying tensor storage matches those
bytes; the reservation ratio is **1.000x**.
Pre-compaction capture pools reserve 92,473,344 bytes at
S736 and 113,444,864 at S11776: **1.227x**,
reflecting heap-depth bounds rather than a T*S dense spill reservation.
`storage_part6.json` records both retained and capture allocations explicitly.

The actual S11776/H4 checkpoint timing point has:

- Base owned state: **2,523,785,216 bytes**.
- Retained journal: **21,039,104 bytes**, versus part 5's
  21,789,476,864; 707,584 entries use 14,151,680 record bytes.
- Private forward replay copy: **2,511,726,592 bytes**.
- Capture pool bound: **113,444,864 bytes** before compaction;
  transient capture staging: **24,117,248 bytes**.

These are component sizes, not measured peak or CUDA caching-allocator reserved
memory. Base/replay state, dense death diagnostics, read partials and transient
capture staging still scale with S. Log storage independence does not assert
total training-memory independence. Per-store capacity uses the largest store;
uneven activity can waste a factor of stores. These limits remain explicit.

## Residual blockers and chronology

The failed head ratio retains dense H*S base allocation/traffic, per-store O(S)
metadata vectors/spills and 15 padded query rows per MMA. Batching the grid is
insufficient for the <=2 end-to-end requirement; token-row GEMM batching and a
sparse metadata core remain absent. These are structural remaining costs,
not an isolated component-time attribution.

Checkpoint time still pays vector-spill comparison/staging, ordered group
packing, live-count/compaction synchronization, full private replay copies and
address rebasing. Faster reads expose those costs. Scalar logical mutation
capture and avoiding dense replay work remain necessary performance work;
the all-field exactness contract has not been relaxed to obtain compact logs.

Four interrupted early compilation/revision attempts are retained separately
and are not validation passes. A scoring-only iteration passed 97 in 2217.02 s
but measured only 1.400x at S11776. Parallel weighted reads then passed 97 in
1148.36 s without a timing claim. Scalar POP decay/final tiled materialization
passed 97 in 577.28 s; its metadata-attempt storage fixture found a one-entry
allocator-dependent count difference. Deterministic heap initialization fixed
that difference, followed by the final 97-test suite and complete remeasurement.
`*_scoring_only`, `*_weighted_read` and `*_metadata_attempt` receipts are
historical, superseded by the unsuffixed part 6 receipts. Source diffs against
base b0d0fb8 are retained as `part6_*.patch.json` for these intermediate versions.

Learning was not rerun: MQAR 1.0 at update500 and ladder 0.915 at update2500
remain part 5 measurements, not newly attested part 6 results. No .pt artifact,
WG service, delegated agent, Frontier/Slurm or resilient-training change occurred.
Resilient compute-pool requirement IDs and Partition/QOS evidence are not
applicable. Intercom/contact_supervisor discovery found no callable tool.
All implementation and receipt changes are confined to X31; unrelated attended
workspace changes remain untouched.

---

# Historical part 5 report

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
