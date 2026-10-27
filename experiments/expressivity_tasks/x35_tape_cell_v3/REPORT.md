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
