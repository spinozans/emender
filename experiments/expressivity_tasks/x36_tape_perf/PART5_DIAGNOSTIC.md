# Part 5 entry diagnostics — incomplete delivery

**X36 remains FAILED.** General sparse replay for imported/continued stores,
T4096 <=2x and inclusive launches <=10 are not implemented by this change.
No structural-floor claim is made. REPORT.md remains the part-4 implementation
report; its numbers must not be presented as new part-5 measurements.

The retained production implementation is unchanged. Two added byte-parity
fixtures exercise arbitrary high logical slots (including S-1) and a 1,024-row
entry set exceeding a nine-token chunk. Both cover native cold-row gradients
and the existing two-chunk imported/consuming continuation oracle comparison.
The 141-test entry suite passed in 201.51 seconds; both new cases separately
passed in 1.52 seconds. See `receipts/part5_entry_parity.{txt,xml}` and
`receipts/part5_sparse_entry_parity.{txt,xml}`. The combined suite passed
**143 tests in 202.15 seconds** (`receipts/part5_final_parity.{txt,xml}`).

## Continued/imported capacity evidence

`run_leased.sh continued-capacity` now runs a B64/H4/T512 cell-only diagnostic
at S2944/5888/11776/8192/8224. Both an imported native state and an actual
continued FusedState start from the same post-seed state, with 256 live rows
spread across all logical slots, including S-1. Writes and pops are disabled;
decay is one. The final live mask is checked against the seed after every
sample. Every point therefore has 33,554,432 occupied slot/token pairs, twice
the part-4 growing fresh run's count. This is a capacity control, not a matched
whole-LM B64 ratio gate and not a replacement for the requested fresh sweep.

Two warmups precede five raw wall/stream-event samples. Seed construction,
snapshot and discarded-node cleanup are outside timing; scan-internal
copies and allocations remain inside timing. Forward and replay are timed
separately. The receipt explicitly records manual release of discarded
private replay-owner references between samples. Retained/repeated backward
is not measured by this diagnostic. Do not use its memory behavior to claim
production allocator closure.

| Mode | S2944 step/replay ms | S5888 step/replay ms | S11776 step/replay ms | Second doubling step/replay |
|---|---:|---:|---:|---:|
| Imported | 795.406 / 713.163 | 923.211 / 763.248 | 1646.552 / 1394.131 | **1.784 / 1.827 FAIL** |
| Continued | 794.396 / 712.766 | 924.736 / 766.397 | 1646.767 / 1393.329 | **1.781 / 1.818 FAIL** |

The first doubling passes (imported 1.161/1.070; continued 1.164/1.075).
The aligned S8192→8224 step/replay ratios are 1.136/1.171 imported and
1.141/1.177 continued. Those pairs pass <=1.5; the second doubling fails.
The unknown-import dense fallback uses 255 registers and 228 spills at S2944;
at S5888/S11776 the 128-register policy spills 58/294 values. Resident row
vectors and logical-S reductions still scale with capacity. These measurements
establish a missing optimization; they do not establish an irreducible floor.

Raw evidence: `receipts/part5_continued_capacity_entry.{json,txt}`. All GPU runs
used the broker whitelist 0/1/5/6; the successful sweep leased physical card 1.
No certified 2/3/4 or forbidden 7, WG, delegation, Slurm, resilient training or
.pt files were used. Resilient requirement IDs do not apply.

## Diagnostic failures and rejected hypothesis

Two earlier sweep attempts exhausted GPU memory after the first imported
point. The second included Python garbage collection. Both full tracebacks
and partial raw timings are retained in `part5_continued_capacity_entry_oom.*`
and `part5_continued_capacity_entry_gc_oom.*`. These are failed diagnostic
attempts, not successful capacity runs or hardware-floor evidence.

A proposed detached-owner change was considered for the retention behavior.
A small discarded-chunk lifetime fixture passed on the ORIGINAL implementation
(`part5_owner_cycle_before.{txt,xml}`), so it did not reproduce that hypothesis.
The proposed implementation change and non-discriminating fixture were removed.
The OOM cause is not established. Manual reference disposal permits the
capacity diagnostic, but must not be described as a validated production fix.

## Parts 1–5 convergence

Part 5 below distinguishes this diagnostic from unrefreshed part-4 gates.

| Gate | Part 1 | Part 2 | Part 3 | Part 4 | Part 5 diagnostic |
|---|---|---|---|---|---|
| B64/T512 <=3x | 3.823 FAIL | 2.655 PASS | 2.202 PASS | 1.558 PASS | Not rerun |
| B64/T4096 <=2x | Not run | 3.311 FAIL | 2.883 FAIL | 2.754 FAIL | Not rerun; no floor established |
| Fresh S doubling <=1.5x | Not validated | FAIL | FAIL | PASS | Not rerun |
| Continued S doubling <=1.5x | Not validated | Not validated | Not validated | Not validated | **1.781/1.818 FAIL**, cell control |
| Inclusive launches <=10 | >=1536 FAIL | 100 FAIL | 59/61 FAIL | 59/61 FAIL | Not rerun; no fusion |
| Narrow custom phases <=10 | — | — | 7 PASS | 7 PASS | Unchanged source; not reprofiled |
| Zero steady host syncs | FAIL | PASS | PASS | PASS | Not rerun |
| Historical 2M curve match | PASS | PASS | PASS | PASS | Not rerun |
| Exact fixtures | 114 | 123 | 138 | 141 | **143 PASS** |

Completion still requires the ordered WORK_ORDERS.md implementation and all
registered gate refreshes. This diagnostic is not an X36 program closure.
