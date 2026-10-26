# X31 preregistration

Registered 2026-10-26 before executing X31 tests or timing measurements.

Acceptance gates, in order:

1. Exact integral routing/death traces; outputs, keys, values, strength and
   first-order gradients against X30 naive_reference at atol=rtol=1e-5.
   Seeded randomized imported stores must include age ties, popped/dead rows,
   threshold-boundary deaths, all routing cases and continuation.
2. MQAR validation >=0.99 by update 5000; slots=736, heads=4, batch=64,
   AdamW lr=1e-4. No inherited X30 result counts as an X31 result.
3. Ladder a^k b^k, k<=128, 2500 updates, slots=736, heads=4.
   Success signal balanced accuracy >0.9; report both class recalls regardless.
4. Full forward latency at batch=64, heads=4, slots in {736,11776},
   against X30 and X28E eager. Numeric targets: >=1x X30 at 736 and
   >=10x X30 at 11776, using end-to-end per-token wall time.
   Heads=16 / heads=1 latency ratio <=2 at equal batch and slot budget.
5. Sparse checkpoint overhead <=5% of forward at slots=11776;
   demonstrate events scaling independently of S. Report base state separately.
6. Append-phase parallel prototype faster than sequential formulation;
   no numerical speedup threshold beyond >1x, include launch/synchronization.

Timing protocol: GPU lease restricted to physical cards 0..6. Warm up 5,
measure 20 repetitions with CUDA events and synchronize before reading results;
report medians, dimensions, software, physical GPU and dtype. Training and
performance claims are forbidden until the complete parity gate passes.
CPU interpreter checks are correctness diagnostics, never GPU performance.

## Results

Current results are in the Part 6 ledger below. Parts 2–5 are historical;
no v1 result is substituted for an X31 result.

### Part 4 current acceptance ledger (partial delivery)

| Deliverable / registered gate | Current result |
|---|---|
| Mutation-time append/correction/CAP payload before-images | Implemented opt-in in live kernel; repeated writes retain intermediate values |
| Full all-field O(events) mutation journal | NOT IMPLEMENTED; payload-only journal reserves O(tokens), not O(actual writes); POP/touches/deaths/management missing |
| Exact rollback | Keys/values only; NOT a full-state rollback or usable continuation |
| Fused backward / custom autograd / G10 adjoints | NOT IMPLEMENTED; training fails closed |
| Gate 1, complete forward + first-order gradient parity | UNMET; no gradient evidence |
| Gate 2, MQAR >=0.99 by 5000 | NOT RUN |
| Gate 3, ladder >0.9 by 2500 | NOT RUN; accept/reject recalls unavailable |
| Gate 4, >=1x at S736 / >=10x at S11776 / H16:H1 <=2 | NOT MEASURED |
| Gate 5, checkpoint overhead <=5% | NOT MEASURED; payload byte counts are not timing evidence |
| Gate 6, append-phase speedup >1x | NOT MEASURED |

Validation chronology (lambda01, broker-leased physical GPU 1, Python 3.12.3):

- `run_leased.sh -q -x tests/test_write_journal.py`: exit 4, zero tests;
  extra path incorrect because the wrapper supplies its own test directory.
  Receipt `validation_part4_attempt.txt`; **not a pass**.
- `run_leased.sh -q -x -k 'payload or repeated_cap'`: **6 passed,
  49 deselected in 83.26 s**, including compilation. Receipt
  `validation_part4_journal.txt`. Checks before-images at oracle token
  boundaries and bitwise key/value rollback, including repeated slot writes.
- `run_leased.sh -q -x`: **55 passed in 221.90 s**, including compilation.
  Receipt `validation_part4.txt`. Collection preceded final snapshot-isolation
  assertions and the additional continuation case; their final subset is
  recorded separately below. This includes all 49 preexisting tests.
- Final `run_leased.sh -q -x -k 'payload or repeated_cap'`: **7 passed,
  49 deselected in 6.02 s**, physical card 1. Receipt
  `validation_part4_final_journal.txt`; includes final snapshot isolation and
  two-chunk payload rollback with persistent storage pointers. Scoped
  `git diff --check` passed. No interrupted/compile-failure run in this part.

Reserved payload bytes: B*H*T*(8+4*(K+V)), unchanged across tested
S={1,7,736,11776}. No O(actual events) compaction or training checkpoint
overhead claim follows. Part 4 is incomplete; all gates above remain as stated.

### Part 3 current acceptance ledger (partial delivery)

Targets above remain unchanged; this section records status, not registration.

| Deliverable / registered gate | Current result |
|---|---|
| Sparse birth index | Device linked birth order implemented; not X30's bounded-depth birth-group structure; imported future/age-zero ties can walk O(S) |
| Persistent metadata / no continuation clones | Implemented with consuming FusedState ownership and explicit snapshot branching; native import still copies/initializes once |
| Tiled tensor-core scoring | `tl.dot` slot tiles, fp32 tf32x3; padded single-query rows, no batching efficiency or speedup claim |
| Mutation-time captured O(events) journal | NOT IMPLEMENTED; standalone old rollback primitive is not integrated |
| Fused backward / custom autograd / G10 | NOT IMPLEMENTED; training still fails closed |
| Gate 1, complete forward + gradient parity | UNMET; forward checks cannot substitute for first-order gradients |
| Gate 2, MQAR >=0.99 by 5000 | NOT RUN |
| Gate 3, ladder >0.9 by 2500 | NOT RUN; accept/reject recalls unavailable |
| Gate 4, >=1x at S736 / >=10x at S11776 / H16:H1 <=2 | NOT MEASURED |
| Gate 5, checkpoint overhead <=5% | NOT MEASURED |
| Gate 6, append-phase speedup >1x | NOT MEASURED |

All training and timing remain withheld under gate 1. See REPORT.md for the
ownership API change, remaining dense operations and validation chronology.

Final part 3 forward-only validation: `run_leased.sh -q -x` completed
**49 passed in 221.41 s**, including compilation, lambda01 physical GPU 1.
Output: `validation_part3.txt`. Six new cases supplement the prior 43 tests;
exact traces and 1e-5 forward/state checks pass in the tested cases.
The earlier compilation failure and intermediate 47-test pass are retained
separately. No interrupted run is counted as a pass. This receipt does not
clear gate 1, establish gradients, or authorize a performance/training claim.

### Recorded validation (2026-10-26)

The initial small-store GPU suite passed 20 tests. The final expanded suite
passed **31 tests in 3.38 s**, on lambda01 physical GPU 1, acquired/released
through the broker. Full output: `validation.txt`. This elapsed pytest time
includes compilation and is **not a throughput measurement**.

Forward oracle comparisons: seeds 0..4, S={1,7,16,736,11776}, batch=2,
heads=2, T=9, K=4, V=5; plus exact-death, age-tie and pressure-window fixtures.
Outputs/state satisfy atol=rtol=1e-5 and discrete traces match exactly in these
cases. First-order gradients are unsupported, so gate 1 remains **UNMET**.

| Preregistered target | Measured / status |
|---|---|
| Complete parity including gradients | Forward checks pass; gradients NOT IMPLEMENTED |
| MQAR >=0.99 by update 5000 | NOT RUN |
| Ladder balanced accuracy >0.9 by update 2500 | NOT RUN; class recalls unavailable |
| >=10x X30 at S=11776, batch64/H4 | NOT MEASURED |
| H16/H1 latency <=2x | NOT MEASURED |
| Checkpoint overhead <=5% at S=11776 | NOT MEASURED; sparse journal not integrated |
| Append-phase speedup >1x | NOT MEASURED |

Sparse journal storage fixture: 3 IDs (int64) + 3x3 old fields (fp32) =
60 bytes, unchanged between S=7 and S=11776. This verifies the primitive's
storage shape only; it does not establish sparse training checkpoint costs.
All performance/training measurements remain withheld under the parity-first
rule. No v1 training result is substituted for X31 evidence.

The first expanded 31-test run took 20.00 s including new large-budget kernel
compilation; the final rerun after input-validation checks took 3.38 s with
cached kernels. Neither number is a performance receipt.

### Part 2 partial continuation (2026-10-26)

`run_leased.sh -q -x`: **43 passed in 259.84 s**, including Triton compilation,
lambda01 physical GPU 1. Output: `validation_part2.txt`. Exact lazy interval
jumps, cancellable death scheduling and hierarchical lowest-free allocation
are integrated; original forward/continuation checks remain passing. These
are correctness receipts only. The birth index, tiled reads, mutation-time
journal and fused backward remain incomplete, so all training and timing
gates in the table above remain UNMET / NOT RUN / NOT MEASURED as recorded.


### Part 5 current acceptance ledger

| Deliverable / gate | Final result |
|---|---|
| All-field mutation-time before-images / exact chunk rollback | Implemented for owned physical writes; byte-exact persistent rollback, snapshot/parent checks pass |
| Compact logical-event journal / S independence | FAIL: dense vector capture and conservative O(T*S+(S+T)*log S) reservation |
| Fused backward / custom autograd / G10 | Implemented first-order reverse kernel; private replay and trace ownership; no routing reevaluation |
| Gate 1: complete forward + first-order gradient parity | PASS: final 95 passed in 265.55 s; 1e-5 all adjoints/state and exact traces |
| Gate 2: MQAR >=0.99 by 5000 | PASS: 1.0 at update500 on 20k validation, B736/H4/batch64/AdamW lr1e-4 |
| Gate 3: ladder >0.9 by 2500 | PASS: 0.915 at2500, accept1.000/reject0.830; best0.922 at2000 |
| Gate 4: >=1x X30 at S736 | PASS: 10.085x |
| Gate 4: >=10x X30 at S11776 | FAIL: 0.319x |
| Gate 4: H16/H1 <=2x | FAIL: 5.672x at S736; 40.361x at S11776 |
| Gate 5: checkpoint time <=5% | PASS timing: -0.315% observed delta, matched four-warp forward geometry |
| Gate 5: sparse/event storage independence from S | FAIL: 21,789,476,864 reserved bytes, 253,904,384 entries at S11776; dense-record lower bound grows with S |
| Gate 6: append mutation prototype >1x | PASS scoped mutation phase: S=736: 1.086x / S=11776: 1.091x; includes common reset/launch/sync, excludes reads/management/certification/allocation |

Current receipts: `validation_part5.txt/.xml`, `parity_part5.json`,
`training_part5.txt`, `mqar_part5.json`, `ladder_part5.json`, `cost_part5.txt/.json`,
`append_cost_part5.txt/.json`, and `summary_part5.json`. Source/JUnit hashes are
checked before cost/learning collection. Final card: broker-leased physical1,
lambda01 RTX6000 Ada; Python3.12.3, torch2.9.1+cu128, Triton3.5.1, fp32.

Final forward costs (B64/H4/T16/K64/V128/width256, fresh state each call):

| Slots | X31 ms/token | X30 ms/token | X28E ms/token | X30/X31 | Target |
|---|---|---|---|---|---|
| 736 | 0.450551 | 4.544004 | 2.112711 | 10.085x | PASS |
| 11776 | 78.210305 | 24.970704 | 17.173224 | 0.319x | FAIL |

Timing uses exactly five warmups and twenty measurements; component accounting
uses warmup five. Raw CUDA-event and wall samples are retained. Checkpoint time
includes the private forward replay copy and map. Base state bytes:
2,523,785,216; journal reservation: 21,789,476,864;
private replay copy: 2,511,726,592. These are
component sizes, not peak-memory measurements. Storage independence fails even
though the relative time threshold passes.

All compile/runtime failures, five interrupted validation attempts and the
interrupted initial ladder are retained separately and are not passes. The
failed uniform16 forward and preliminary mixed-geometry/extra-accounting cost
attempts are superseded by the final matched4/exact5 protocol. See REPORT.md
for the complete chronology and scope limits. No v1 accuracy or speed result
is substituted. The completed learning run uses unchanged S736/H4 numerical
kernels; subsequent H1 trace-ownership and large-S geometry changes received
the final parity check before costs. One seed, in-range ladder validation,
stage2 omitted; no OOD certification. No .pt artifact or out-of-scope source edit.


### Part 6 current acceptance ledger

Part 6 supersedes part 5 implementation/cost status; targets above are unchanged.
Learning remains historical part 5 evidence and was not rerun.

| Deliverable / gate | Final part 6 result |
|---|---|
| Full parity / all-field exact rollback | PASS: 97 tests, including the original 95, no skips; all ten gradients at 1e-5 and exact traces through S11776 |
| Slot-tile-parallel score and weighted read | Implemented `(B*H, ceil(S/32))` scoring/local weighting, parallel value reduction, scalar POP decay and final tiled materialization |
| S736 >=1x X30 | PASS: 20.735x |
| S11776 >=10x X30 | PASS: 21.472x |
| H16/H1 <=2 | S736 PASS 1.474x; S11776 FAIL 21.618x |
| Checkpoint time <=5% | FAIL: +79.756% |
| Compact retained log / storage independence | PASS: equal 99,328 entries retain 2,689,024 bytes at both budgets; capture bounds ratio 1.227x |
| Append mutation prototype >1x | PASS: S736 1.062x; S11776 1.063x; bitwise equality |
| All registered targets | FAIL: large-store head scaling and checkpoint time remain blockers |

Final receipts: `validation_part6.txt/.xml`, `parity_part6.json`,
`parity_attestation_part6.txt`, `performance_part6.txt`, `cost_part6.txt/.json`,
`append_cost_part6.txt/.json`, `storage_part6.txt/.json`, `summary_part6.txt/.json`.
Final suite: 97 passed in 17.22 s with cached compilation. All timing/storage
collection used one broker lease, physical GPU 1 on lambda01,
NVIDIA RTX 6000 Ada Generation, Python 3.12.3, torch 2.9.1+cu128, Triton 3.5.1.
Five warmups, twenty measurements, median end-to-end wall time and raw CUDA
samples; fresh state/allocation/synchronization included. H1/H4/H16 are explicit.
Source/JUnit hashes were checked before collection. No collection errors.

Checkpoint component sizes: base 2,523,785,216; retained journal
21,039,104; 707,584 entries / 14,151,680
record bytes; private replay copy 2,511,726,592.
The equal-append fixture's capture bounds are 92,473,344 /
113,444,864 bytes at S736/S11776. Transient staging and
base/replay state remain O(S); no peak or total-memory-independence claim.

Remaining structure: O(S) recurrent metadata and spills, H*S dense allocation,
single-query padded MMA, three launches per token, vector-spill journal
comparison/staging/sizing and full replay copies/rebasing. No shared multi-query
head GEMM, token-group amortization or integrated append rank-scan is delivered.
See REPORT.md for precise scope, chronology and the two failed timing gates.

Four interrupted attempts are not passes. Completed superseded suites and
measurements have explicit `_scoring_only`, `_weighted_read` or
`_metadata_attempt` names, with reconstructable source diffs in
`part6_*.patch.json`. Final deterministic empty heap metadata removes the
allocator-dependent one-entry discrepancy seen in the metadata attempt.
No .pt, WG, delegated agent, Frontier/Slurm or resilient-training change.


### Part 7 final acceptance ledger — partial delivery

Final parity: **105 passed in 338.18 s**, original 97 retained plus eight new checks,
no skips. Final receipts are unsuffixed `*_part7` files; source/JUnit hashes were
checked before sequential collection on broker-leased physical GPU1. Python3.12.3,
torch2.9.1+cu128, Triton3.5.1, fp32, lambda01 RTX6000Ada. Five warmups, twenty
measurements; same fresh-state B64/T16/K64/V128/width256 protocol.

| Gate | Final result |
|---|---|
| S736 speedup >=1 | PASS 20.180x |
| S11776 speedup >=10 | PASS 111.947x |
| H16/H1 <=2, S736 | PASS 1.271x |
| H16/H1 <=2, S11776 | FAIL 5.135x |
| Checkpoint overhead <=5% | FAIL +17.398% |
| Append-phase speedup >1 | PASS S736 1.059x / S11776 1.076x |
| Equal-stream storage independence | PASS 855,040 entries at both S; reservation ratio1.227 |
| All registered targets | FAIL: two timing gates still open |

Steady metadata updates are scalar; POP tile reduction moved out of the core.
Fresh import remains O(S), heap O(log S), imported birth walks O(S); strict
O(1)-amortized maintenance is not delivered. Grouped 16-query/K32 GEMM retains
only the same-store diagonal and wastes cross-head arithmetic. Dense H*S native
initialization and diagnostics remain. Known empty-origin grids use an append
upper bound; unknown imports still use full grids.

Capture emits before-images at vector mutation sites and uses device counts,
fixed capacities and pinned nonblocking rebase-map transfer. No live-count/final-
sizing `.item()`/`.cpu()` read or capture staging remains in the training path.
Empty private native imports avoid cloning, and private payload replay copies
allocated rows only. Ordered atomic/barrier groups, capacity-wide rebasing,
nonpayload replay clones and dense nonempty-import copies remain.

Reservation is 92,473,344 / 113,444,864 bytes at S736/S11776, no longer part6's
actual-count compaction; the <=2x largest-actual-event-count bound is not retained.
Unknown imported bounds may reserve against S. Base/private replay address-space
bytes remain 2,523,785,216 / 2,511,726,592 at S11776/H4; the latter is not bytes
copied on the empty-origin forward path. No peak or total-memory claim.

See REPORT.md for all failed/superseded run names, exact remaining structures
and H1/H4/H16 tables. Final `summary_part7.json` records false for both blockers
and `all_registered_targets_passed`. No learning rerun, .pt, WG, delegation,
Frontier/Slurm or resilient-training change.


### Part 8 final acceptance ledger — one gate remains failed

Final source: **109 passed in 15.52 s**, all original 105 retained and four added
regressions; no skips. Source/JUnit attestation preceded sequential collection
on broker-leased physical GPU1, lambda01 RTX6000Ada. Python3.12.3, torch2.9.1+cu128,
Triton3.5.1, fp32. Unchanged B64/T16/K64/V128/width256, five warmups/twenty samples,
fresh-state allocations and synchronization included; no collection errors.

| Gate | Part 8 result |
|---|---|
| S736 speedup >=1 | PASS 22.352x |
| S11776 speedup >=10 | PASS 93.109x |
| H16/H1 <=2 S736 | PASS 1.200x |
| H16/H1 <=2 S11776 | FAIL 4.775x |
| Checkpoint overhead <=5% | PASS +4.874%, narrow margin |
| Append mutation >1 | PASS S736 1.079x / S11776 1.057x |
| Equal-stream storage independence | PASS 855,040 entries each; capacity ratio1.227 |
| All registered targets | FAIL: large-store head gate remains open |

Final evidence is the unsuffixed `*_part8` set, including `structural_cost_part8`.
REPORT.md contains the single convergence table for parts5–8 and all H1/H4/H16
measurements. H4 absolute X31 time improves at both S; X30's S11776 denominator
changes between collections, so its speedup ratio is lower despite faster X31.

Large-S native init alone: H1 .042732 / H4 .159212 / H16 .631451 ms/token.
Entire H16 allowance is .380859 ms/token (twice H1). This measures the limiting
current dense zero-filled native representation; it is not a universal hardware
floor or proof that compact/lazy backing cannot pass. Full native laziness is
not delivered. Empty-origin expensive import vectors use a reachable prefix;
unknown imports, native payload initialization, several metadata fills and dense
diagnostics remain O(S). Heap depth is O(log live); arbitrary imports can still
reach O(log S), and future/tied birth walks can remain O(S). No timer wheel.

Capture uses single-writer counts without atomic/barrier groups. Final
materialization also has one writer per store; two new multi-tile rollback cases
cover the review-discovered invariant gap. Replay copies and rebasing occur at
backward or freeze-before-continuation, not the fresh forward. Internal trace
buffers are retained without copying; public traces remain independent. Forward
replay-copy bytes=0, retained aliased owner address space=2,511,726,592. The raw
cost field retains its legacy name; the summary separates copied/retained bytes.
Public owner in-place edits now raise on backward. Continuation freezing and
serial imported materialization can defer/add costs outside the registered fresh
forward. Backward rebasing still scans capacity. No backward-speed claim.

Matched projected scan capture delta +1.643%; no irreducible HBM floor claim.
Base state2,523,785,216; journal113,444,864; checkpoint691,200 records. Equal-stream
reservation92,473,344 /113,444,864. No total/peak-memory-independence claim.

Superseded receipts are explicitly suffixed `_before_materialize_fix` (107-test
source with an uncovered materialization race) and `_before_private_trace`
(109 tests, checkpoint+5.036%). The final 109-test source fixes the race and removes
the extra trace copy. Baseline/prototype receipts and prototype patch are retained.
See REPORT.md for complete chronology and residual risks. No learning rerun, .pt,
WG, delegation, GPU7, Frontier/Slurm or resilient-training change. Supervisor tool
was unavailable. All edits are scoped to X31.
