# X36 part 7 — event-driven adjoint liveness; closure FAILS

The requested all-gates-green or established-floors closure is **not achieved**.
The named dense-scatter hypothesis is disproved for the measured workload:
**zero dense native-gradient unpack calls** execute at every fixed-live point.
Replacing that absent operation cannot improve either capacity doubling or T4096.
The implemented improvement uses journal decisions to suppress **unused packed
value-adjoint traffic**. All **179 tests pass**, including cold signed zeros,
continued-capacity byte fixtures and a new raising-unpack guard in all 25
capacity cases (both imported and continued).

The two host sizing synchronizations remain; inclusive launches remain above
10. Failed capacity/long-context gates retain their measured ratios and required
savings in [part7_arithmetic.json](receipts/part7_arithmetic.json).
**No hardware/irreducible floor waiver is claimed.** This is a failed performance
closure with a narrower validated optimization and a corrected diagnosis.

## What the journal actually routes

`reverse_program` already reads `Cases`, `Matched` and `Alloc` and routes the
packed value/key adjoint directly into the writing token's `DVout`/`DKout`.
Corrections read only their matched row; allocation reads and clears only its
allocated row. There is no dense cold-gradient pass in this token loop.
Pop derivatives use strength adjoints and logical reductions, not a W_v/W_k
payload scatter. Read adjoints still cover resident live rows because query
gradients depend on their contents.

`native_grad` calls `unpack` only if the corresponding native-state **input**
requires a gradient. The capacity protocol requires query/value **projection
input** gradients and no native-state input gradients: `[false,false,false]`.
The new regression patches `unpack` to raise; all 25 shapes still match oracle
bytes in both modes. Receipts assert those three flags at every sample.
Even for native-gradient consumers, unpack runs once per chunk, outside the
reverse token loop. T4096's empty-origin fallback does not use pooled unpack.

The new device-only event-consumer mask scans recorded correction/allocation
decisions once per pooled store. If no writing token can consume value adjoints
and there is no native-value gradient consumer, it suppresses GValues loads and
stores during read replay. It preserves the exact per-token DVout zero stores.
Chunks with writes and native-value consumers keep the original recurrence.
For live17/256/1024/2800 this removes logically
2,281,701,376, 34,359,738,368, 137,438,953,472, 375,809,638,400 bytes
of packed read/write accesses per chunk. These are logical accesses, **not
measured DRAM bytes**. The branch remains on device and adds no launch or sync.

Cold output-adjoint behavior is unchanged: when native gradients are requested,
untouched key/value rows retain any supplied output adjoints, and cold strength
rows execute the existing `+0` recurrence. They cannot universally be replaced
by zero without losing nonzero output adjoints or signed-zero semantics.
The full cold-signed-zero fixtures passed; no exactness break was observed.

## Capacity remeasurement

B64/H4/T512, entry live rows spread to S-1, frozen keys, zero writes/pops,
decay=1, production allocation. Two warmups and five raw samples per point;
construction/snapshots/GC remain outside samples. Imported controls and all
medians, compiler statistics, pool counts, native-gradient flags and peaks are
retained in `part7_live17/256/1024/2800.json`. All 40 points completed.

Each cell below is **step/replay** ratio; limit <=1.5 for both.

| Entry live | 2944→5888 | 5888→11776 | 8192→8224 |
|---|---|---|---|
| 17 | 1.950/2.692 FAILED | 1.424/1.262 PASSED | 1.157/1.274 PASSED |
| 256 | 2.075/2.353 FAILED | 1.465/1.218 PASSED | 1.051/1.220 PASSED |
| 1024 | 1.261/0.910 PASSED | 1.632/1.238 FAILED | 1.021/1.057 PASSED |
| 2800 | 0.992/0.923 PASSED | 1.194/1.102 PASSED | 1.018/1.030 PASSED |

Absolute continued **step/replay milliseconds** (no baseline slowing):

| Entry live | S2944 | S5888 | S11776 |
|---|---:|---:|---:|
| 17 | 48.394/21.043 | 94.373/56.643 | 134.341/71.456 |
| 256 | 110.589/29.182 | 229.438/68.658 | 336.115/83.660 |
| 1024 | 259.654/121.747 | 327.468/110.756 | 534.540/137.069 |
| 2800 | 773.540/442.912 | 767.212/408.749 | 916.344/450.564 |

Per-point ratios to the part-6 medians are also receipted in JSON; the
removed logical traffic does not imply proportional wall savings.

Zero native unpack traffic contributes **0 ms** to every failed gate. The
remaining capacity-dependent operations are concrete and separately counted:

| S | Forward tile probes/chunk | Full logical lane width | Public freed backing bytes |
|---|---:|---:|---:|
| 2944 | 12,058,624 | 4,096 | 385,875,968 |
| 5888 | 24,117,248 | 8,192 | 771,751,936 |
| 11776 | 48,234,496 | 16,384 | 1,543,503,872 |
| 8192 | 33,554,432 | 8,192 | 1,073,741,824 |
| 8224 | 33,685,504 | 16,384 | 1,077,936,128 |

The forward loops over `ceil(S/32)` tiles at saturated logical reach, including
empty tiles. It evaluates 32 floating lanes per nonempty tile. Uniform live256
entries populate all 92 then all 184 tiles on the first doubling: **2x padded
read work at fixed live**. Uniform live1024 entries populate all 184 then all
368 tiles on the second doubling: another **2x padded read work at fixed live**.
Live17 occupies 17 tiles at every capacity but still probes the full tile grid.
The JSON receipts these nonempty-tile counts and padded FLOP arithmetic for
every capacity/live pair; they are source-level lane counts, not ISA counters. The reverse reconstructs full logical operands for DK+4=68
reductions per store/token (64 query sums plus pop, maximum, denominator and
dp sum). The first doubling changes the logical width 4096→8192 and the oracle
warp policy 4→16, even when PS stays 17 or 256. `freed` is dense public int8
telemetry initialized once per chunk; its public transpose can require another
allocation/copy of the same size. Private metadata and inverse maps also retain
logical-S shapes. None is a dense projection-gradient scatter.

For every pair, the JSON retains `forward_delta`, `replay_delta` (under
`device_delta_ms`) and `max(0,large-1.5*small)` separately for wall/replay.
Those identify how much of the measured increment is forward versus backward;
source counts identify the surviving O(S) mechanisms. For the continued misses:

| Live | S pair | Forward delta ms | Replay delta ms | Wall savings required ms | Replay savings required ms |
|---|---|---:|---:|---:|---:|
| 17 | 2944→5888 | 10.417 | 35.599 | 21.783 | 25.078 |
| 256 | 2944→5888 | 79.351 | 39.476 | 63.554 | 24.885 |
| 1024 | 5888→11776 | 180.830 | 26.313 | 43.338 | 0.000 |

Required savings are `max(0,large-1.5*small)`; a zero replay requirement
means only the whole-step ratio fails. These are separately measured ranges. Their individual
subexpression timings have **not** been causally isolated. These are current
implementation costs, not an established irreducible floor. No failure is
silently waived because its packed pool is small.

## Long context: useful work is not constant per token here

The fresh matched T512/T4096 ratios are **1.551x** and
**2.621x**. T4096 requires
**1460.043 ms** of wall savings to meet <=2x.
The proposed dense-unpack removal contributes exactly zero. The new pooled
liveness specialization does not execute for this fresh long-context fallback.

The routing trace counts 16,777,216→1,073,741,824
occupied slot-token pairs: **64x** for 8x tokens.
Mean live occupancy is 128→1024. Just score-dot plus weighted-value read work is
`2*(64+128)*pairs` = 6,442,450,944→412,316,860,416
FLOPs. This is flat at 384 FLOPs **per live-slot/token pair**; per input token
it rises 8x. Padding, nonlinearities, recurrence and backward are additional.
These kernels use vector reductions; the count is not a calibrated GEMM floor.

| Separate profile | T512 ms | T4096 ms |
|---|---:|---:|
| Scan kernel | 41.096 | 1043.523 |
| Reverse kernel | 61.135 | 2750.898 |
| Other scan/replay device activities | 7.103 | 34.912 |

Scan ns per occupied pair is 2.449→0.972;
reverse is 3.644→2.562.
Inclusive fresh activities are 59→61, not 8x as many.
Most device busy time is inside scan/reverse; the full per-name launch ledger
and other activity totals are in the JSON. Profile GPU busy times are not
additive wall samples. Launch/telemetry wall costs are not independently
calibrated, so this does not establish an honest compute lower bound or fully
explain the wall residual. The long-context gate remains failed when above 2x.

## Sync and launch residuals

The continued profile has **106 inclusive activities** versus <=10
(excess 96), **2 steady stream syncs** and
**2 DtoH transfers** versus zero. Their complete names/counts
and busy-time decomposition are retained in `part7_profile.json` and the
losslessly archived trace. Explicit terminal and profiler teardown device synchronizations are outside
the registered ranges; no steady transfer is excluded.

The current eager exact-allocation path has two serial shape dependencies:
forward needs device entry maximum/cold-tail metadata before host journal
reservation/dispatch; backward needs the post-scan union maximum before host
`torch.empty` allocates exact PS. Arithmetic is **2 > 0**, independent of event
scatter. This is a conditional constraint of the present host-sized allocation
strategy, not a hardware floor: device allocation, bounded reservation or a
changed specialization strategy could alter it. No such redesign or launch
fusion was implemented here. The inclusive-launch gate has no irreducible-floor
proof. Both requested improvements remain unmet.

## Parts 1–7 convergence

| Gate | Part 1 | Part 2 | Part 3 | Part 4 | Part 5 diagnostic | Part 6 final | Part 7 |
|---|---|---|---|---|---|---|---|
| B64/T512 <=3x | 3.823 FAIL | 2.655 PASS | 2.202 PASS | 1.558 PASS | Not rerun | 1.541 PASS | 1.551x PASSED |
| B64/T4096 <=2x | Not run | 3.311 FAIL | 2.883 FAIL | 2.754 FAIL | Not rerun | 2.624 FAIL | 2.621x FAILED |
| Fresh S doubling <=1.5x | Not validated | FAIL | FAIL | PASS | Not rerun | PASS | Part 6 PASS carried; not rerun |
| Continued S doubling <=1.5x | Not validated | Not validated | Not validated | Not validated | 1.781/1.818 FAIL | Expanded sweep FAIL; 1024/2800 pass | See new matched-entry table; failed ratios remain FAIL |
| Inclusive launches <=10 | >=1536 FAIL | 100 FAIL | 59/61 FAIL | 59/61 FAIL | Not rerun | Fresh 59/61, continued 106 FAIL | Fresh 59/61, continued 106 FAIL |
| Narrow custom phases <=10 | — | — | 7 PASS | 7 PASS | Not rerun | Fresh 7 / continued 8 PASS | Fresh 7 / continued 8 PASS (unchanged phases) |
| Zero steady host syncs/DtoH | FAIL | PASS | PASS | PASS | Not rerun | Fresh PASS / continued FAIL | Fresh PASS; continued 2 syncs/2 DtoH FAIL |
| Historical 2M curve match | PASS | PASS | PASS | PASS | Not rerun | PASS | Part 6 PASS carried; fresh path unchanged by pooled optimization |
| Exact fixtures | 114 | 123 | 138 | 141 | 143 PASS | 179 PASS | 179 PASS, including 25 raising-unpack byte controls |
| Production-allocation continued sweep survives | — | — | — | — | OOM/GC failures | 50/50 points PASS; peak 24.660 GiB | 40/40 fixed-live points survive; no near-budget rerun |

## Reproduction and remaining risk

`receipts/part7_run.sh` runs the complete parity suite before all timed work,
then the four fixed-live sweeps, continued profile and matched T512/T4096
profiles/routing traces. Only `/home/erikg/emender/.venv/bin/python` and
`run_leased.sh` were used for GPU work, on physical card 0, 1
from the broker-approved 0/1/5/6 set.
No .pt files, forbidden/certified cards, WG, workers or subagents were used.
Run archival as `/home/erikg/emender/.venv/bin/python -m
experiments.expressivity_tasks.x36_tape_perf.receipts.part7_archive` from the
repository root. It verifies byte-identical gzip round trips and profile-summary
equality; `part7_analysis.py` regenerates arithmetic, then `part7_report.py`
regenerates this report. Part 6 is archived in [REPORT_PART6.md](REPORT_PART6.md).

At S2944, live1024 replay improves 293.803→121.747 ms and live2800 replay
847.677→442.912 ms versus part 6. These absolute gains do not close the small
entry gates or justify a floor waiver.

The delivered change removes unused packed adjoint traffic; it does not replace
a dense scatter on the measured path because that operation is absent. Failed
ratios, two sizing syncs and excess launches remain risks and unmet work. The
historical curve and fresh capacity gates are explicitly carried from part 6;
they were not rerun. Supervisor escalation was unavailable because no callable
`contact_supervisor` tool was exposed; coordination attempts and discoveries
are in `part7_coordination.txt`. This report must not be used to attest that
X36 reached all-gates-green or a proven-floor closure.
