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

Pending. Unrun gates must remain explicitly unrun, never inferred from v1.

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
