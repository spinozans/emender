# X36 part 6 — sparse continued adjoints implemented; overall target FAILS

The final source passes **179 tests**, including the original 143, 25
T512 capacity/entry byte fixtures in both imported and continued modes,
and adversarial cold signed zeros. The 50-point production-allocation sweep
completes without OOM, peaking at **26,478,391,808 allocated bytes**
(24.660 GiB) and **26,937,917,440 reserved bytes**.
Sparse adjoints and private replay payloads use the exact entry/touched union.

**The requested <=1.5x continued capacity gate is still FAILED.** Small
17-row and 256-row entries fail the first doubling. The old failing second
live256 doubling is improved, and live1024/live2800 controls pass, but that does
not close the expanded gate. Absolute improvements are reported alongside
ratios; no baseline is deliberately slowed to manufacture a pass.
B64/T512 is **1.541x PASS**;
B64/T4096 is **2.624x FAIL**.
Fresh inclusive launches remain **59/61: FAIL** (<=10).
The continued profile has **106 inclusive activities** and
**2 steady host syncs / 2 DtoH transfers**: FAIL.
No irreducible structural or hardware floor is established.

Canonical evidence: [machine ledger](receipts/summary.json),
[attempt index](receipts/part6_attempts.json),
[sizing arithmetic](receipts/part6_sizing_arithmetic.json),
[gate arithmetic and decomposition](receipts/part6_failure_arithmetic.json).
The unchanged historical evidence remains in [part 5](PART5_DIAGNOSTIC.md),
[part 4](REPORT_PART4.md), [part 3](REPORT_PART3.md) and their archived ledgers.

## Representation, ownership and exactness

For each store n, let E_n be its live rows at entry and A_n be the logical
allocation IDs recorded by the chunk. The pool is U_n = E_n union A_n,
ordered by logical slot. Writes, deaths, corrections and pops of existing rows
are already covered by entry membership or an earlier allocation. PS is
max(1, max_n |U_n|); actual float buffers have row pitch **PS**, never S or
next_power_of_two(S). Triton lane padding is separate from allocation.
Every store receipts entry count, allocation events, measured union, exact PS,
and the conservative bound min(S, |E_n| + T). Duplicate allocations count once.

Private key/value replay payloads also use PS. Full native owners remain
immutable cold backing. Journal addresses are translated from native logical
AoS rows to packed SoA rows through an inverse map; every payload target is
device-asserted to belong to the union. A finally block reverses this mapping
and restores the original journal address space. Required native gradients
scatter packed hot adjoints into dense cold-gradient backing, retaining
untouched output adjoints and inactive strength signed-zero arithmetic.
Consuming continuations retain the existing freeze/version validation.

The pool is selected for every nonempty-entry short chunk, including continued
states whose logical reach has saturated at S. Reach and cardinality are
separate quantities. T>512 keeps the previous dense/prefix fallback, and fresh
empty origins keep the independently validated prefix specialization.
Logical-S reduction operands reconstruct the oracle's original addition tree;
full integer inverse maps, live/strength metadata and these reduction operands
remain capacity dependent. This is not an O(live) end-to-end algorithm.
Exceptional cold -0/nonfinite strength selects a full cold-tail expression;
ordinary positive-zero cold rows keep the smaller resident expression.
Two host metadata transfers are inside short nonempty-entry timings: packed
entry maximum/exceptional flag in forward, union maximum in backward.

## OOM, sizing and allocation protocol

The production journal reservation now uses measured entry cardinality,
independently of logical reach:

`capacity/store = T*(DK+DV+64+16*h) + (entry_live_max+T)*(8*h+32)`

where `h = S.bit_length()`. Pool worst-case row count is
`min(S, entry_live_max+T)`; exact allocation uses the measured union maximum.
For N=256, T512, S11776, entry256, the journal reserves 356,352 entries/store
and 1,824,522,240 bulk bytes, versus 10,317,987,840 bytes when entry is
incorrectly replaced by capacity. Frozen-key packed value+strength adjoints
use 33,816,576 bytes; private hot key/value payloads use 50,331,648 bytes.
The full arithmetic table includes small, 1024, 2800 and S-32 entries.
These figures are reservations, not a claim that journal bulk equals total peak.

A new public-output lifetime test reproduces a separate raw-owner cycle on the
original implementation: **40,844,800 bytes retained per discarded chunk**,
even with GC. Of that, the T9/N4/S2944 journal bulk accounts for 30,561,280
bytes; other owners, traces and context account for the remaining 10,283,520.
Detached replay-owner handles retain pointers and shared version counters
without retaining autograd outputs through their own contexts. The candidate
standalone allocation series is flat at **9,319,424 bytes** over seven samples.
The complete suite's final lifetime series is also flat (a different reference
harness baseline). See `part6_lifetime_original.txt`, `part6_matrix_parity.txt`
and `part6_final_source_parity.txt`.

The sweep uses the actual production tape_scan/prepare path. It never changes
private candidate node references to force collection. Public outputs are
dropped and GC runs outside samples. Two warmups and five samples per point
retain wall, forward/replay stream events, and allocated/reserved peaks.
The peak includes persistent seed/template and snapshot baseline; construction,
validation, GC and separate profiling are excluded from timing. Each sample
asserts unchanged live set, zero allocations and a nonmissing production pool.
Near-budget entries use S-32; those comparisons also grow useful occupancy,
so their literal <=1.5x failures are retained and labeled unmatched occupancy.

The enlarged byte harness separately hit an oracle-owner OOM: its error
attributed 44.47 GiB to this process. Only discarded historical oracle nodes
now release private handles after independent byte snapshots. Candidate nodes
are never privately cleared. The raw OOM receipts are retained; no X37
interference is asserted. Final sweeps have no OOM and no measured co-run
corruption requiring a retry.

## Continued/imported capacity sweep

All points are B64/H4/T512, fixed live rows spread to S-1, frozen keys, zero
writes/pops, decay=1. Event replay includes its actual production setup.
The table reports continued medians; imported controls and all five raw samples
are retained in `part6_hot_live*.json` and `part6_hot_near_budget.json`.

| Entry live | S2944 step/replay ms | S5888 step/replay ms | S11776 step/replay ms | First doubling step/replay | Second doubling step/replay | Aligned step/replay |
|---|---:|---:|---:|---|---|---|
| live17 | 50.277/23.019 | 94.330/56.443 | 137.426/73.910 | 1.876/2.452 FAILED | 1.457/1.309 PASSED | 1.166/1.294 PASSED |
| live256 | 118.542/37.021 | 232.348/71.010 | 337.679/86.211 | 1.960/1.918 FAILED | 1.453/1.214 PASSED | 1.057/1.208 PASSED |
| live1024 | 431.748/293.803 | 492.178/275.905 | 686.888/288.265 | 1.140/0.939 PASSED | 1.396/1.045 PASSED | 1.021/1.027 PASSED |
| live2800 | 1178.341/847.677 | 1190.620/832.063 | 1329.302/863.032 | 1.010/0.982 PASSED | 1.116/1.037 PASSED | 1.013/1.016 PASSED |
| near_budget | 1220.738/877.957 | 2502.386/1815.950 | 6333.839/4957.911 | 2.050/2.068 FAILED | 2.531/2.730 FAILED | 0.997/0.994 PASSED |

The first small-entry cliff coincides with the oracle's 4→16-warp layout and
larger full logical reduction shape. Those are current implementation costs,
not proof of a lower bound. A narrow eight-warp attempt failed exact bytes
and was reverted before timing. The earlier full-S replay payload version
passed live256 ratios but was much slower absolutely; the final hot backing
retains absolute improvements even where a ratio worsens. Large-entry AoS
was rejected because its apparent ratio pass slowed the S2944 baseline to
20.6 seconds (276 compiler spills). It is not accepted as convergence.

## Re-measured remaining gates and decomposition

The matched T4096 wall is 6188.075 / 2357.990
= 2.624x. Reaching <=2 requires **1472.094 ms**
of wall improvement. Whole-LM profiles retain pinned nonblocking ingestion in the
registered ranges. Narrow custom launch counts remain seven for fresh and
8 for the continued profile, but inclusive activities include all
native setup, fill, copy, reduction and mapping work. Fresh zero steady
host sync/DtoH remains PASS; the new continued sizing path FAILS that gate.

| Profile | Scan kernel ms | Reverse kernel ms | Setup ms | Rebase ms | Other forward/replay activities ms | Inclusive/custom |
|---|---:|---:|---:|---:|---:|---:|
| fresh T512/S2944 | 43.150 | 60.739 | 1.870 | 1.873 | 3.252 | 59/7 |
| fresh T4096/S2944 | 1043.835 | 2746.885 | 1.870 | 12.934 | 21.322 | 61/7 |
| continued L256/T512/S11776 | 238.810 | 85.893 | 0.575 | 0.920 | 13.966 | 106/8 |

GPU component busy times are separately profiled and are not additive wall
samples. The continued backward range wraps Replay.backward on the actual
autograd thread; loss construction and leaf accumulation are outside it.
An earlier outer-range profile missed that thread and is explicitly superseded.
Journal ablation and allocator CPU time remain separately labeled in the
machine ledger. Long occupied pairs grow 16,777,216→1,073,741,824, with no
frees, and logical payload access counts grow 64x for 8x tokens. Those are
algorithmic counts, **not DRAM measurements or a bandwidth/GEMM lower bound**.
The arithmetic receipt decomposes failures and required savings without
waiving any gate. No gate is declared to have hit an irreducible floor.

Both checkpoint-free 512-step/2,097,152-token curves pass against historical
losses: base max error **9.53674316e-07**,
tape **9.53674316e-07**. Median training step times
are 24.045/86.335 ms
(base/tape), ratio 3.591; this is separate from the B64 wall gates.

## Parts 1–6 convergence

| Gate | Part 1 | Part 2 | Part 3 | Part 4 | Part 5 diagnostic | Part 6 final |
|---|---|---|---|---|---|---|
| B64/T512 <=3x | 3.823 FAIL | 2.655 PASS | 2.202 PASS | 1.558 PASS | Not rerun | 1.541 PASS |
| B64/T4096 <=2x | Not run | 3.311 FAIL | 2.883 FAIL | 2.754 FAIL | Not rerun | 2.624 FAIL |
| Fresh S doubling <=1.5x | Not validated | FAIL | FAIL | PASS | Not rerun | PASS |
| Continued S doubling <=1.5x | Not validated | Not validated | Not validated | Not validated | 1.781/1.818 FAIL | Expanded sweep FAIL; 1024/2800 pass |
| Inclusive launches <=10 | >=1536 FAIL | 100 FAIL | 59/61 FAIL | 59/61 FAIL | Not rerun | Fresh 59/61, continued 106 FAIL |
| Narrow custom phases <=10 | — | — | 7 PASS | 7 PASS | Not rerun | Fresh 7 / continued 8 PASS |
| Zero steady host syncs/DtoH | FAIL | PASS | PASS | PASS | Not rerun | Fresh PASS / continued FAIL |
| Historical 2M curve match | PASS | PASS | PASS | PASS | Not rerun | PASS |
| Exact fixtures | 114 | 123 | 138 | 141 | 143 PASS | 179 PASS |
| Production-allocation continued sweep survives | — | — | — | — | OOM/GC failures | 50/50 points PASS; peak 24.660 GiB |

Reproduction uses only `/home/erikg/emender/.venv/bin/python` and
`run_leased.sh`; physical cards 0/1/5/6 only. Final full-source parity command is in
`part6_final_source_parity.txt`; `part6_hot_payload_parity.txt` is the preceding
179-test gate passed before final performance; final sweeps/gates/curves are orchestrated by
`receipts/part6_run_final.sh` (the live17 sweep ran immediately before it).
Rebuild the ledger with `receipts/part6_analysis.py`, then the arithmetic and
this report with the corresponding stdlib receipt scripts. Compressed Chrome
traces retain content hashes and summary equivalence checks.
No .pt checkpoints, card7, certified cards2/3/4, WG or subagents were used.
One validation compiler tail briefly overlapped another validation after SIGINT
failed to stop it promptly; it was terminated before performance. All timed
runs were sequential. `contact_supervisor` was not exposed; coordination
attempts and discoveries were recorded in `part6_coordination.txt`.
