# X35 part 1 result: the entry-state two-pass thesis is false as stated

The delivered result is a negative structural receipt, with executable X31
counterexamples, conservative trace instrumentation and an integer allocation
subproblem. It is not a new training cell. Detached routing is state-dependent:
a prior append changes age-1 selection, residual, case and possibly gate.
Even a certified append-only mutation run can have these dependencies.
See CHUNK_PARALLEL.md for the dependency table, restricted algorithm,
segmentation conditions, proof obligations, timer-wheel design and backing work.

## Measured density (seed 0 initialization only)

C=64, S=2944, K=64, V=128, H=4, d_model=256; 8 positive k=128 ladder
words and 8 MQAR T128 validation-prefix examples, using read-only X31 TapeCell.
Density counts individual (example, head, chunk) stores; a whole batched GPU
chunk requires every participating store to qualify. No trained weights were
reconstructed from the historical summary JSONs. Repeated ladder examples
measure identical-token runs and head variability, not independent samples.

| Data | Store chunks | Append mutation candidates | Entry-pure structural candidates |
|---|---:|---:|---:|
| Ladder, all full chunks | 128 | 0% | 0% |
| Ladder, all-a full windows | 32 | 0% | 0% |
| Ladder, all-b full windows | 32 | 0% | 0% |
| MQAR, all full chunks | 64 | 50% | 50% |
| MQAR, all-padding windows | 32 | 100% | 100% |
| MQAR, non-padding windows | 32 | 0% | 0% |

Each ladder store excludes 2 trailing tokens from the full-chunk density;
case histograms include the complete word. All-a and all-b full windows each
contain 1024 intra-chunk matched-value references across their 2048 decisions.
There are no POPs at this initialized closed gate. The >90% a-run purity
prediction FAILS on this stated initialization protocol. This does not establish
its failure after learning. MQAR's 50% is entirely trailing padding, not useful
parallel append work. `density.json` retains event-token fractions, source/token
hashes and exact numerators; `density.txt` retains broker card evidence.

## Validation and scope

`run_leased.sh -q`: **12 passed in 10.74 s**, on physical GPU 0, lambda01,
Python 3.12.3, with only the mandated project interpreter. Three fixtures compare
X31 and X30 outputs, five functional fields and every trace field bit-for-bit.
One additional fixture checks the read gradient with respect to values bitwise.
Other tests falsify chunk-entry routing, expose a true append run's dependency,
check nonzero error-coefficient gating and causal read/gradient dependence,
verify fail-closed missing-death evidence and segment coverage, and check exact
integer rank allocation with holes, no-writes and capacity overflow.

These are counterexample and component tests. They are not full v3 pure/mixed
state/gradient parity, and do not replace the 109-test X31 suite. X31 usage EMA
is excluded consistently with its public functional scope. The integer scan
preserves lowest-free order; the obstacle is not an fp32 free-list ordering.
No defect in X31 is alleged and no reference source was modified.

## Registered gates and staged implementation

- (a) >=3x complete pure forward: NOT RUN; no valid general pure path built.
- (b) trace density: MEASURED on initialization; a-run >90% prediction FAIL.
- (c) >=2x mixed training loop: NOT RUN; no mixed training implementation.
- (d) zero full pure/mixed bitwise parity failures: NOT ESTABLISHED. The 12
  delivered tests pass, but the required implementation/suite does not exist.

Stage 1 instrumentation and the formalization are delivered. Stage 2 is limited
to the integer allocation subproblem; no complete batched decision/read/write
path is delivered. `segments` is retrospective analysis only, not stage 3
runtime composition. Stages 3–4 remain unimplemented. Stage 5 is receipted as
the part-2 work order. No mutation-only timing is substituted for cell speed.

A corrected research direction is to batch append/correction PAIRS or compute
dependency waves and causal row versions. It may work well even though the
proposed independent first pass fails. The negative result does not prove
chunk parallelization impossible, nor that a two-token event rate imposes an
absolute throughput bound. A trained trace study remains necessary before
claiming sparse events for deployment.

## Operational receipt and residual risks

Only X35 files are included in the commit. Existing unrelated workspace edits
are preserved. No .pt files, WG service, workers, agent delegation, Frontier,
Slurm submissions or resilient training changes were used; resilient requirement
IDs and scheduler Partition/QOS requirements are inapplicable. GPU broker
wrappers restrict cards to 0–6; GPU7 was not requested or used. The short runs
used cards 0/1, not the long-run card reservation policy.

`contact_supervisor` was searched for first in the callable tool catalog and
was unavailable. No supervisor decision/progress contact could be sent.

Residual risks: initialization-only and small data sample; retrospective
classification does not predict routes; arbitrary imported states require
additional time-projected age/liveness checks; general bitwise GEMM and replay
accumulation equivalence is unproved; timer-wheel and compact ownership-aware
backing are work orders. There is no speedup or learning-quality claim.

## Part 2 — incomplete engineering delivery; acceptance targets not reached

The requested Consequences section and structural-target preregistration are
added. The trained receipt is cited precisely: the actual prefix is EMPTY,
CONFLICT, NOMATCH, CONFLICT, ...; “EMPTY/CONFLICT alternation” is shorthand
for append/correct, not the literal long-run case codes. Sequential routing
remains settled. No X31 source defect is alleged or patched.

Two independently usable GPU components are implemented:

- `causal_read.py`: two launches for a complete chunk, using recorded
  pre-token logical-slot-to-payload-version IDs and time-indexed rounded
  strengths. It preserves X31's 32-slot tile/reduction arithmetic. Tested
  views contain mid-chunk append, correction, death, reuse, holes and empty
  stores. The same version can serve several tokens. Invalid IDs have device
  assertions and safe load masks. Dead nonfinite payloads are excluded.
- `timer_wheel.py`: eight radix-256 levels, intrusive cancellable links,
  absolute int64 deadlines, and journal-aware mutation primitives. Scheduling
  and cancellation do bounded work; each timer cascades at most seven times.
  Advancing still costs O(elapsed tokens + cascaded/fired timers), and a single
  crowded bucket can take linear time in its membership. No stale timer
  accumulation. Tests exercise radix boundaries through 2^56, clocks near
  INT64_MAX, long-future timers, rescheduling/cancellation, randomized
  deadline-model agreement, link invariants and exact typed-journal undo.
  The diagnostic driver rejects advancement through INT64_MAX to prevent
  clock overflow. Death ties use deterministic list order, not the X31
  heap's sorted-slot order; full scan integration must adjudicate this.

**These are components, not an integrated v3 TapeCell.** The causal views
must still be emitted by the sequential core without O(C*S*(K+V)) copies.
The forward component deliberately rejects gradient-bearing calls; its
separate differentiable specification has shared-version adjoint tests, but
those do not establish composition with G10 recorded-decision replay.
The timer wheel is not connected to the scan's metadata owners, journal
capacity calculation, continuation/snapshot layout, or replay rebasing.
Its journal test proves primitive undo only, not cell continuation/backward.
Occupancy-proportional backing is **NOT IMPLEMENTED**. Dense public fields,
cold dead payloads and every replay address still require a coordinated
representation change; no change to that representation is landed here.

### Gate ledger

| Preregistered part-2 target | Verdict |
|---|---|
| Complete C64/S2944/H4 forward >=2x | NOT RUN; no integrated candidate |
| Read first-order adjoints compose with G10 | NOT ESTABLISHED |
| O(H*live) initialization/backing | NOT IMPLEMENTED |
| S11776 H16/H1 <=2, original T16 protocol | NOT RUN; backing unchanged |
| Timer wheel in sequential core, full replay/undo | NOT INTEGRATED |
| 109-test X31 oracle suite against changed v3 representation | NOT RUN; no changed v3 representation |

Unchanged-X31 baseline validation and new component parity are distinguished
in `summary_part2.json`. No isolated kernel timing is substituted for an
end-to-end gate, and no timing methodology was changed to report a pass.
This delivery fails the requested integrated engineering scope.

### Validation chronology and operational limits

The first collection passed 127 tests (109 unchanged X31, 12 prior X35,
6 new causal-read tests). Wheel development then exposed a Triton constexpr
reassignment compile failure (7 failed/18 passed) and an expiry failure
(6 failed/19 passed); both raw logs are retained. Moving cascade-level
specialization into its own JIT function resolved the latter; the final
source is validated separately in `validation_part2_final.txt` and XML.
Earlier failures are not counted as final validation.

All GPU execution used `run_leased.sh` and physical card 1, with the exact
project interpreter. No card 7, .pt file, WG, delegation, Frontier/Slurm or
resilient training change was used. Resilient requirement IDs and scheduler
Partition/QOS evidence are inapplicable. Existing unrelated changes were
preserved. `contact_supervisor` was searched first in the callable catalog
and is unavailable, so no supervisor coordination call could be sent.

Residual risks are the unimplemented backing, causal-view construction,
recorded-replay read adjoints and complete timer-wheel integration; no
throughput, memory-scaling, or full v3 parity claim is made.

## Part 3 — live causal reads delivered; backing scope incomplete

The live v3 entry point is now `cell_fused.tape_scan` / `TapeCell`.
Sequential routing runs inside one store-owning chunk kernel. The core emits
pre-token version IDs and exact rounded strengths at the same boundaries
recorded in the G10 journal. Each append/correction emits one immutable payload
version; unchanged rows share versions. Two read launches then score every
query in the chunk. POP query argmax remains sequential inside the core when
its threshold is enabled. There are no per-token output-read launches.
`INTEGRATION.md` documents the representation, complexity and replay ownership.

Production backward composes the read adjoints with the typed journal's exact
rollback: it restores the pre-read store at each recorded boundary, applies
read adjoints there, and transfers write/allocation adjoints at their original
mutation boundaries. It does not recompute routing or death decisions.
Public trace mutation, replay-owner mutation, snapshots, consuming continuation
and exact undo are exercised by the unchanged X31 oracle tests against the
v3 public cell module. Standalone component differentiability is not used as
proof of this composition.

**The requested occupancy-proportional backing is not delivered.** Native
payloads, slot metadata, public views and the death trace retain dense backing;
training replay also retains dense owners. No live-row pool, cold-row store,
logical-to-physical ownership map, lazy public materialization or O(H*live)
initialization is claimed. Limiting read/version work to the reachable prefix
is a read optimization, not a backing representation change. Thus the part-3
engineering scope fails even if an individual timing gate passes.

The integrated core keeps the X31 heap. The wheel's deterministic tie ordering,
metadata ownership, journal capacity and replay layout have not been composed
with the cell. X31 reference files are unchanged. A local v3 journal class
constructs v3 continuation handles on undo and rebases snapshots in the v3
address space; otherwise a reference journal would return a reference handle.

Validation and gate numbers are recorded below and in `summary_part3.json`.
The first attempted run exposed a Triton vector/scalar name collision while
compiling a multi-slot core; `validation_part3_compile_failure.txt` retains
that failed attempt. Its scalar POP temporary was separated from the causal
view temporary before rerunning. Intermediate validation is not substituted
for the final source validation.

### Final part-3 validation and gate results

Final source: **141 passed, zero failures/errors/skips in 494.13 s**: the
unchanged 109-test X31 suite with public-cell redirection, plus 32 X35 tests.
The 32 include four new live-cell integration regressions. A separate clean
process without module redirection passed all four new regressions (3.71 s),
including real v3 journal undo and reuse of the returned v3 handle. This avoids
letting test redirection mask implementation ownership mistakes. The extreme
negative-score POP fixture also checks argmax initialization below the read
mask sentinel. The 109-pass intermediate receipt predates final source changes
and is retained separately, not used as the final attestation.

| Registered gate | Measurement | Verdict |
|---|---|---|
| B64/C64/S2944/H4 complete forward >=2x | X31 0.164254, v3 0.087380 ms/token; **1.879757x** | FAIL |
| B64/T16/S11776 H16/H1 <=2 | H1 0.124013, H16 0.918685 ms/token; **7.407990x** | FAIL |
| Occupancy-proportional backing / fresh init | Dense payload and metadata unchanged | FAIL, not implemented |
| B64/H16/S11776 live-proportional memory | 8,192 live / 12,058,624 logical rows; 10,095,140,864 base-state bytes; 10,658,721,792 incremental peak bytes | FAIL |

The H16 standalone native-zero initialization floor is **0.664066 ms/token**
under the original T16/five-warmup/twenty-sample method. It did not approach
zero. H16 retained allocation is 10,298,073,088 bytes despite only 0.06794%
live occupancy. The dominant remaining structure is the dense public native
payload and persistent logical-slot metadata; full death traces, dense version
maps and training replay owners also remain. Read batching reduces per-token
launches but does not remove these allocations. The small read/version working
set is not a live-proportional memory receipt. Component floor measurements
are not additive attribution of full-cell timing.

All samples and accounting are in `gates_part3.json` and raw stdout in
`gates_part3.txt`, measured on physical GPU 1 (NVIDIA RTX 6000 Ada Generation).
No samples were dropped or initialization subtracted to reach a target. The
new matched S2944 X31 denominator, not a nonexistent historical S2944 number,
is used. No second timing run was selected to cross the 2x threshold.

Only X35 files are included in this delivery. GPU7, .pt output, WG, delegation,
Frontier/Slurm and resilient training were not used. Scheduler Partition/QOS
and resilient requirement IDs are inapplicable. Reference sources/tests were
not patched; unrelated pre-existing workspace changes are preserved.
`contact_supervisor` was searched first in the callable catalog and was absent.

**Overall delivery remains incomplete:** causal-read live-scan and first-order
G10 composition are established by tests, but mandatory occupancy backing is
missing and every registered performance/backing/memory gate fails. No complete
part-3 acceptance or speed-target success is claimed.

## Part 4 — incomplete: prefix backing and causal emission, not live/cold pools

The requested last structural item remains **UNMET**. The implemented change
removes architectural-S zero-fill for empty-origin ordinary inference/training
by allocating a proven append horizon and deferring public dense tensors.
It does not implement the full part-9 live-row/cold-row/logical-map design.
Dead rows retain their exact bits in the prefix. Crucially, the prefix grows
with cumulative chunk lengths even when live occupancy is zero. The backing
and live-proportional-memory gates remain false; sparse fresh-start memory
numbers cannot establish either structural gate.

`INTEGRATION.md` records promotion, continuation, replay-version checks,
snapshots and recording behavior. Existing X31 and X35 tests are unchanged;
new tests cover lazy continuation, exhaustion/pressure, exact dead-row/native
parity against the dense scan, final-state adjoints across training chunks,
public native/metadata mutation detection, snapshot/undo after promotion,
orphan public views and RoutingTrace compatibility. Five additional fixtures
compare incremental fp32 causal strengths against timestamp-jump emission
bit-for-bit over five decay rates, imports, POP, death, reuse and continuation.
The timer wheel was not integrated; sequential routing still uses the heap.

The final parity and performance results are in summary_part4.json and the
raw validation/gate receipts. gates_part4.py retains X31 part-8's five-warmup,
twenty-sample synchronized timing loop and all samples, including outliers.
Physical-state accounting is performed outside timings without forcing public
dense materialization. No initialization is subtracted from registered timings.
Component structural_cost rows are independent floors, not additive attribution.
The pre-emission-change run is retained as gates_part4_before_emission.json/txt;
any final remeasurement follows changed source and renewed parity validation.

Earlier failures are preserved: an imported TapeState eligibility mistake
caused the interrupted initial validation; it was corrected before subsequent
complete runs. Three new fixture failures compared the intentionally excluded
usage EMA against the eager oracle; the fixture now compares usage bitwise to
the unchanged dense scan and compares the five functional fields to eager.
No existing oracle assertion was changed. Intermediate passes are not used
as final-source attestation; parity_part4.json binds final validation to source
hashes and gates refuse stale or failing receipts.

Operationally, contact_supervisor was searched first in the callable tool
catalog and was unavailable. No supervisor coordination call could be sent.
Only X35 changes are included. GPU work uses run_leased.sh, cards 0–6 only,
and the exact requested project Python. No .pt files, WG/dispatcher/workers,
agent delegation, Frontier/Slurm submissions, or resilient training changes
are involved. Partition/QOS and resilient requirement IDs are inapplicable.
Existing unrelated workspace edits are preserved.

Residual risks/blockers: no live/cold pools or occupancy-driven compaction;
capacity growth and continuation reconstruction depend on chunk history;
public views/snapshots/explicit journals deliberately become dense; no
occupancy-proportional undo representation; timer wheel still unintegrated.
The whole part-4 work order must not be accepted from these improvements alone.

### Final part-4 validation and registered verdicts

Final source: **158 passed, zero failures/errors/skips** on physical GPU
1, lambda01. The unchanged 109 X31 and prior 32 X35 tests pass,
with 17 new backing/emission tests. `parity_part4.json` binds the source hashes
to that validation. Frozen replay tolerates later mutation of a consumed
promoted handle, matching existing private-copy ownership. Public TapeState
field replacement also updates the owning materialized continuation.
The exhaustion fixture starts with a physical two-row prefix under S=3, then
observes logical slot 2 followed by five R-CAP events and pressure exactly 5/8.

| Registered gate | Measurement | Verdict |
|---|---|---|
| C64/S2944/H4 complete forward >=2x | X31 0.166433, v3 0.075833 ms/token; **2.194722x** | PASS |
| S11776/T16 H16/H1 <=2 | H1 0.110748, H16 0.260153 ms/token; **2.349044x** | FAIL |
| Occupancy-proportional backing | Prefix grows with chunk history, including at zero live occupancy | FAIL |
| Live-proportional memory | No live/cold pools; sparse fresh memory alone is insufficient | FAIL |

B64/H16/S11776 with 8,192 live rows out of 12,058,624 logical rows
now uses 14,270,464 physical base-state bytes,
23,543,808 retained allocation bytes and 102,813,696
incremental peak bytes. This receipts sparse-start improvement, not a live-
proportional backing claim. With **zero live rows**, two successive T16 no-write
chunks grow base state from 14,270,464 to 27,983,872 bytes: a direct counterexample
to the remaining structural target.

The actual T16 prefix-native initialization floor is H1
0.005169, H16 0.005191 ms/token.
The standalone scan without projections is H1
0.088993, H16 0.238227 ms/token.
Projection-only floors and every sample are also retained. These independent
component measurements are not additive and were not subtracted from either
registered gate. Dense fresh-init cost is removed for ordinary empty-origin
inference/training, but the registered head-scaling target remains unmet.

All different-source measurements are retained in the chronological
`benchmark_timeline` of summary_part4.json. Final remeasurement follows public
field-assignment support and renewed parity; gates_part4_pre_public_assignment
retains the previous source (2.171054x forward, 2.351331x heads). The registered
verdict above uses final source, not the most favorable run.
**Overall: incomplete.** The last structural work item remains undelivered.

## Part 5 — FAIL: diagnostic adjudication, pooling still unimplemented

**This does not complete the work order or close v3.** No production cell,
backing, kernel, or replay implementation was changed. The delivered work is
an exact fresh-preparation/store-core isolation diagnostic and a six-gate
remeasurement, with a successful final parity receipt. The required live/cold
pools and nonidentity logical map remain unimplemented. This is an incomplete
delivery, not the permitted adjudication of a floor after pooling completion.

Final validation: **160 passed**, zero failures/errors/skips;
the unchanged 109 X31 tests and 49 existing X35 tests, plus two diagnostic
regressions. The regressions compare captured production-core decisions and
payloads with the complete scan and exercise diagnostic accounting without
collecting timings during parity. The initial diagnostic fixture indexing
failure and two subsequent accounting failures are preserved with their
intermediate validation/hash receipts. They are not final-source attestation.

| Registered gate | Final measurement | Verdict |
|---|---|---|
| C64/S2944/H4 forward >=2x | 2.194987x | PASS |
| S736/T16 H16/H1 <=2 | 2.348039x | FAIL |
| S11776/T16 H16/H1 <=2 | 2.349795x | FAIL |
| Occupancy-proportional backing | Prefix backing unchanged | FAIL |
| Live-proportional memory | No live/cold pools | FAIL |
| Full oracle parity | 160 tests, zero failures/errors/skips | PASS |

### Residual cost isolation

The premise that an O(architectural S) fresh-state allocation explains the
remaining heads ratio is not supported by the existing implementation. In an
ordinary empty-origin T16 call, `backing.scan` passes `budget=min(S,16)` to
`_tape_scan`: native payloads, metadata, free words, death diagnostics and
version maps all have physical S=16. Both architectural S736 and S11776 share
that same physical horizon. This says nothing about long continuations,
imports, explicit journals or public promotion, which still have residuals.

`structural_cost_part5.py` builds its preparation function directly from the
production `_tape_scan` AST, stopping immediately before the core launch. It
runs the actual native initialization, metadata setup, layout copies, trace
allocation, free-word initialization and causal-version preparation. The
independent core measurement prepares a NEW exact state before every timed
interval and times just the real core dispatch. Its preparation is explicitly
outside that diagnostic timing, never outside registered whole-cell timing.
The diagnostic uses independent random K=Q/V inputs, writes enabled, POP and
append logits zero; it is not a matched additive attribution of registered
whole-cell samples. A separate writes-disabled scan is an ablation, not a gate.

Exact fresh preparation: H1 **0.031334**, H16 **0.035144 ms/token**.
Sequential store core: H1 **0.056782**, H16 **0.190631 ms/token**,
ratio **3.357267x**. The core launches one
program per store: 64 programs at H1 versus 1024 at H16, each running the
sequential T16 routing/scheduling loop. All allocations in this fresh diagnostic
use 16 physical rows. Every sample is retained. These are measured component
costs, not an irreducible hardware floor; they are neither added nor subtracted
to relabel a failed registered gate. Further kernel work could change them.

At **zero live rows**, 4 successive T16 chunks grow base-state
bytes from **14,270,464** to **55,410,688**,
and physical rows from 16 to 64.
This remains a direct counterexample to occupancy-proportional backing.
No favorable fresh-start footprint overrides either structural failure.

### Parts 1–5 convergence

| Part | Integrated parity | C64/S2944/H4 forward >=2x | S11776/T16 heads <=2 | Live/cold pools / memory |
|---|---|---|---|---|
| 1 | Counterexamples only, 12 tests | NOT RUN | NOT RUN | NOT IMPLEMENTED |
| 2 | Components; no changed representation | NOT RUN | NOT RUN | NOT IMPLEMENTED |
| 3 | 141 passed | 1.879757x, FAIL | 7.407990x, FAIL | FAIL / FAIL |
| 4 | 158 passed | 2.194722x, PASS | 2.349044x, FAIL | FAIL / FAIL |
| 5 | 160 passed; production unchanged | 2.194987x, PASS | 2.349795x, FAIL | FAIL / FAIL |

Part 1's original pure-chunk >=3x and mixed-training >=2x targets were not run;
this table tracks the later structural-forward preregistration without
retroactively changing those original verdicts. Part-5 timing differences are
remeasurements, not speedups from a new production implementation.

Residual risks: no live/cold row pools, no occupancy-driven compaction or
journal-aware nonidentity map; history-dependent continuation capacity;
dense imports/public materialization/snapshots/explicit replay; separate,
unintegrated timer wheel. No closure, new pooling parity, or universal
impossibility claim is made. Summary and raw receipts remain authoritative.

Operational scope: only X35 artifacts changed. Exact project Python and X35
broker, physical GPU 1; no card 7 or .pt output. The LM and
certified-arm reservations were respected by the broker. No WG service,
workers, delegation, Frontier/Slurm or resilient training changes; scheduler
Partition/QOS and resilient requirement IDs are inapplicable. Unrelated
workspace edits are preserved. `contact_supervisor` was searched first in the
callable catalog and is unavailable; no supervisor coordination was possible.

## Part 6 — FAIL / partial: constant score scratch, active metadata still prefix-sized

**The requested occupancy-proportional metadata implementation is not delivered.**
The only production reduction landed here is the unused legacy score workspace:
its independent owner is now a one-element tensor rather than one float per
physical row per store. Public promotion leaves this owner constant-sized.
This retains the metadata tuple layout and journal owner identity. One new
regression checks promotion and snapshot ownership; the 160 existing tests
(including all 109 unchanged X31 tests) pass unchanged: **161 passed**.

The supplied diagnosis conflates aggregate base-state storage with per-store
metadata. The historical 14,270,464 bytes covers **1024 stores**, native payload,
metadata and history together: 13,936 bytes per store. In the ordinary fresh
T16 path, metadata already follows physical S16, as part 5 documented; it is
not allocated at architectural S11776. The core's 3.357x isolation therefore
cannot be attributed to architectural-S metadata initialization from that
receipt. This correction does not excuse the remaining live-memory failure.

`metadata_part6.json` inventories every native and metadata tensor without
forcing promotion. At B64/H16, the final zero-live probe measures:

| T16 chunks | Physical rows/store | Metadata bytes, all 1024 stores | Metadata bytes/store | Native bytes/store |
|---|---:|---:|---:|---:|
| 1 | 16 | 819,204 | 800.004 | 12,560 |
| 2 | 32 | 1,605,636 | 1,568.004 | 25,120 |
| 3 | 48 | 2,392,068 | 2,336.004 | 37,680 |
| 4 | 64 | 3,178,500 | 3,104.004 | 50,240 |

History adds 512 bytes/store. The tiny fractional metadata term is the shared
four-byte score placeholder divided by 1024 stores. Before this change the
unused score owner alone occupied 65,536 bytes at prefix 16 and 262,144 at 64;
now it occupies four bytes in either case. Active metadata and native storage
still grow at zero occupancy. No occupancy or live-memory gate passes.

### Remaining structures and exactness constraints

- The free bitmap still uses eagerly initialized three-level prefix storage.
  Its root and word strides use physical S; a lazy representation must keep
  lowest-free results and distinguish physical growth from architectural
  exhaustion. No replacement is landed.
- The death heap, inverse positions, due and terminal vectors remain
  prefix-sized. Part 2's wheel remains separate; its documented tie order
  differs from sorted-slot heap order. No live-sized growth or wheel
  integration is claimed. Standalone wheel tests do not prove cell parity.
- Birth links and birth/touch vectors remain addressed by logical slot.
  F12g selection depends on lowest-index birth ties, while typed journals
  retain absolute mutation addresses and snapshots rebase their owners.
  A sparse owner representation needs coordinated forward/replay changes.
  No such change or impossibility proof is delivered here.
- References remain prefix-sized; scores are the sole constant-sized owner
  change. Dense imports/public promotion/snapshots and recorded replay retain
  their existing representation.

These are specific uncompleted dependencies, not evidence that a semantics-
preserving sparse representation cannot be built. This session must not be
accepted as closing any of the three requested active-metadata structures.

### Final registered gates and parts 1–6 convergence

The unchanged five-warmup/twenty-sample complete-cell protocol was rerun only
after parity. Every sample is retained in `gates_part6.json`; no favorable
rerun was selected. Final forward **1.940140x FAIL** and S11776 heads
**2.246183x FAIL**. The forward >=2x target was not held. These are measurements
of the small scratch change, not an implementation of the requested redesign.
The different timing from part 5 alone does not establish causation.

| Part | Integrated parity | C64/S2944/H4 forward >=2x | S11776/T16 heads <=2 | Occupancy / live-memory |
|---|---|---|---|---|
| 1 | Counterexamples only, 12 tests | NOT RUN | NOT RUN | NOT IMPLEMENTED |
| 2 | Components; no changed representation | NOT RUN | NOT RUN | NOT IMPLEMENTED |
| 3 | 141 passed | 1.879757x FAIL | 7.407990x FAIL | FAIL / FAIL |
| 4 | 158 passed | 2.194722x PASS | 2.349044x FAIL | FAIL / FAIL |
| 5 | 160 passed | 2.194987x PASS | 2.349795x FAIL | FAIL / FAIL |
| 6 | 161 passed | 1.940140x FAIL | 2.246183x FAIL | FAIL / FAIL |

The initial zero-dimensional placeholder failed five unchanged rollback tests
because byte views require a dimension. The retained scalar-failure log/XML
shows 155 passed / 5 failed; the final one-element owner fixes this without
editing oracle tests. Final log/XML and `parity_part6.json` attest the passing
source. `summary_part6.json` records incomplete scope and residual risks.

Operationally: exact project Python, X35 broker, physical GPU1; no card7,
.pt files, WG, delegation, Frontier/Slurm or resilient training changes.
Scheduler Partition/QOS and resilient requirement IDs are inapplicable.
`contact_supervisor` was searched first in the callable tool catalog and is
unavailable; no decision or progress contact could be sent. Unrelated workspace
changes were preserved. This result is partial and the engineering item remains open.
