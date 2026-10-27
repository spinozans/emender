# X36 part 4 — fresh-capacity spill cliff closed; delivery remains partial

Part-5 entry diagnostics and the parts 1–5 convergence table are in
[PART5_DIAGNOSTIC.md](PART5_DIAGNOSTIC.md). Continued-state capacity still
fails; the requested part-5 implementation is unfinished. The measurements
below are retained part-4 evidence, not refreshed part-5 gates.

**Partial engineering delivery.** Both capacity doublings pass <=1.5x step
and replay growth. The aligned S8192->S8224 pair is **1.022x step /
1.092x replay**, versus part 3's 1.551x / 1.755x. The
registered spill cliff is removed, with residual full-width reduction cost
reported below. B64/T512 is **1.558x PASS**
(260.955 / 167.543 ms).
B64/T4096 remains **2.754x FAIL**
(6121.207 / 2222.592 ms; <=2).
Inclusive launches remain **59 / 61: FAIL** against <=10; the final path keeps
**seven custom phases** at both lengths. All **141 tests pass** (138 existing
plus three new byte fixtures), zero steady host synchronizations remain
**PASS**, and both 2M-token curves retain max error **9.5367431640625e-7: PASS**.

The complete arbitrary-slot touched/live pool and native launch fusion are
**unfinished**. This implementation bounds fresh prefix adjoints independently
of S, while unknown sparse imports, cold native owners, required native
cold-row gradients and logical-S reductions retain dense representations.
It does not establish a general O(live) replay algorithm. Long-context and
inclusive-launch failures are not waived by the capacity result.

The canonical machine ledger is [summary.json](receipts/summary.json).
[Part 3](REPORT_PART3.md) and [its summary](receipts/summary_part3.json) are archived.

## Representation and exactness mechanism

Lowest-free allocation from an empty origin bounds every read/written row by
entry reach plus chunk writes. At all five fresh B64/T512 points the principal
strength-adjoint, evaluated-strength, probability, weight and value-dot vectors
now have **512 lanes**, covering measured live/head=256, rather than padded
4096/8192/16384 lanes. This is a conservative prefix reservation, not an exact
256-row journal-indexed pool. Corrections, pops, strength writes and deaths
remain covered by the same stable logical-slot prefix.

Private global strength adjoints use this bound when no native-input strength
gradient is requested. Required native gradients retain cold rows. Long
strength/probability scratch uses the reachable bound; read coverage is checked
by a device assertion after every undo boundary. Native owner shapes and
journal address ownership remain unchanged, including cold payloads.

Constrained gather reconstructs only the logical-slot reduction operands,
retaining the historical reduction tree and inactive-lane signed-zero
arithmetic. **Global reduction scratch is removed entirely.** Full logical-S
operands and the byte occupancy vector remain; exactness is tied to the
validated Triton layout. Unconstrained truncation/gather failed bytes.
Eight warps apply only to the tested DK64/DV128 prefix at S>=2944 with entry
reach + T <=512; smaller capacities, unknown imports and long replay retain
historical warp counts.

The [X31 part-9 scoping](../x31_tape_cell_fused/REPORT.md) and
[X35 backing work order](../x35_tape_cell_v3/CHUNK_PARALLEL.md) require cold-row
storage, stable logical ordering and rollback ownership. The implementation
reuses those ownership constraints through identity prefix indexing; the
complete arbitrary-slot map remains future work. Historical X31/X35 sources
and floating expressions were not edited.

## Capacity sweep and decomposition

Two warmups and five raw wall samples per point, with separate stream-event
components and correlation-attributed profiles. Every point independently
measures **16,777,216 occupied slot/token pairs, zero frees and live/head=256**.
Sources are `receipts/part4_seven_s*.json`.

| S | Step wall ms | Replay event ms | Warps / regs / compiler spills / shared bytes |
|---|---:|---:|---|
| 2944 | 260.955 | 62.376 | 8 / 101 / 0 / 2048 |
| 5888 | 279.050 | 75.548 | 8 / 126 / 0 / 2048 |
| 11776 | 283.666 | 76.310 | 8 / 126 / 0 / 2048 |
| 8192 | 280.782 | 74.700 | 8 / 116 / 0 / 2048 |
| 8224 | 286.980 | 81.546 | 8 / 128 / 2 / 2048 |

S2944->S5888 is **1.069x step / 1.211x replay**;
S5888->S11776 is **1.017x / 1.010x**.
Both pass <=1.5x. At the aligned boundary, capacity grows 0.391% and registers
grow 116->128; compiler spills are 0->2 instead of part 3's 166->388, and
shared memory stays 2048 bytes. The old doubling of resident float adjoints is
gone. The remaining **9.2% event-replay boundary cost** is reported
explicitly; not every S-dependent instruction has disappeared.

GPU busy times are separate profiles, not the wall/event samples above.
Journal is a matched no-grad ablation estimate already included in scan;
allocator is overlapping **scan/replay CPU self time**, not additive GPU time.
See [part4_decomposition.json](receipts/part4_decomposition.json).

| S | Scan GPU ms | Journal estimate ms | Other forward GPU ms | Replay adjoint GPU ms | Setup GPU ms | Rebase GPU ms | Other replay GPU ms | Allocator CPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2944 | 41.511 | 10.025 | 2.544 | 58.445 | 1.865 | 1.901 | 0.689 | 0.551 |
| 5888 | 41.018 | 8.924 | 4.602 | 69.188 | 3.751 | 1.781 | 0.688 | 0.536 |
| 11776 | 40.504 | 9.590 | 8.640 | 68.582 | 7.504 | 1.851 | 0.700 | 0.507 |
| 8192 | 41.017 | 10.759 | 6.161 | 72.167 | 5.205 | 1.839 | 0.694 | 0.513 |
| 8224 | 39.988 | 7.545 | 6.201 | 74.641 | 5.233 | 1.920 | 0.687 | 0.608 |

Replay adjoint GPU time across the requested doublings is **58.445 ->
69.188 -> 68.582 ms**, versus part 3's
160.529 -> 322.914 -> 1000.665 ms. Forward remains essentially flat.

## T4096 remains failed

Long replay computes maximum recorded allocation reach on device and combines
it with entry reach. The registered fresh run selects 2048 resident rows;
unknown imports and larger reach retain the full fallback in the same reverse
launch. There is no host query or synchronization to select the branch.

The delivered long program still has **255 registers /
26 compiler spills**. Its full-width fallback remains in the compiled
program. Event replay is **2771.657 ms**; the profile attributes
**2757.794 GPU ms to reverse** versus **1047.658 ms to scan**.
This is still replay-dominated. It cannot honestly be called an established
read-GEMM FLOP floor. The matched <=2 target needs **1676.024 ms** less whole-step
time. The ratio has not approached T512's 1.558x.

[part4_t4096_arithmetic.json](receipts/part4_t4096_arithmetic.json) records
**1,073,741,824 occupied slot/token pairs**, 524,288 allocations, zero frees and
final live/head=2048. Against measured T512 pairs this is **64x occupied pairs
for 8x tokens**, hence 8x mean occupancy. These are growing-occupancy runs, not
a fixed-live per-token control. Logical accesses are 0.825 TB forward / 2.199 TB
replay. Active key scoring and query adjoints each require 137.439 GFLOPs;
value-dot and value-adjoint updates each require 274.878 GFLOPs. These figures
are not measured DRAM bytes, executed ISA counts or a hardware lower bound.

A split fallback removed compact-path spills but added an eighth custom phase
and measured **2.726x FAIL**. Its eight-warp compact variant passed all ten
affected exact fixtures but measured **2.700x FAIL**; capping it at 128 registers
also passed those fixtures but regressed to **2.763x FAIL / 24 spills**. These
variants were reverted: they missed <=2 and regressed the requested seven-phase
shape. Their raw timings/profiles remain retained. No irreducibility claim is made.

## Inclusive launches, semantic gates and evidence

Final profiles contain **42 forward + 17 reverse = 59** activities at T512 and
**44 + 17 = 61** at T4096. Seven custom kernels remain initialize-free, scan,
materialize, replay setup, two rebase/restore launches and reverse. The narrow
custom <=10 target and requested seven-phase shape pass. **Inclusive <=10 fails**:
the remaining 52/54 native activities, chiefly initialization, fills,
input/output copies/transposes and mapping transfers, have **not** been fused.

All six final registered profiles include nonblocking pinned ingestion and
have zero stream/event synchronizations and zero DtoH. Each contains two
explicit `cudaDeviceSynchronize` measurement/profiler boundaries. Traffic
reductions and validation loss extraction occur outside those steady profiles.
All **25** candidate/final Chrome traces are losslessly gzip archived;
[part4_trace_archive.json](receipts/part4_trace_archive.json) records raw,
decompressed and archive hashes. The current reader reproduces every retained
receipt profile dictionary exactly, including reverted split-kernel profiles.

The final suite is **141 passed in 253.90 seconds**, including the original
138 tests plus dense S736/T512 and both aligned S8192/S8224 T512 byte fixtures.
B64/S2944 and S11776/T512 bytes, dense/random/frozen-key T4096, signed zeros,
expanded/strided adjoints, imported/consuming/retained state, zeroed-log
ownership and complete private address restoration remain green. No tolerance
changed. See [part4_seven_phase_parity.txt](receipts/part4_seven_phase_parity.txt).

Both checkpoint-free **512-step / 2,097,152-token** arms match every historical
X34 loss sample, each with max error **9.5367431640625e-7**. B8 median base/tape
steps are **27.253 / 87.805 ms**
(3.222x); that curve-run ratio is distinct from the B64 gate.
Raw curves: [part4_seven_training_slice.json](receipts/part4_seven_training_slice.json).

## Rejected attempts and parts 1–4 convergence

[part4_attempts.json](receipts/part4_attempts.json) indexes source patches, raw
failure logs and parity-cleared performance failures. Direct truncation and
unconstrained gather failed S736/T9 by 5.9604645e-7. Scratch expansion repaired
numeric comparison but failed signed-zero bytes. Restoring tail signs then
failed S11776/T33 by 1.9073486e-6; retaining the byte occupancy operand repaired
aligned layouts but failed S8193/T17 by 1.1920929e-5. Correct scratch alignment
passed 138 tests; final constrained gather removes that scratch and passes all
141. The initial 8->16-warp capacity policy passed parity but still failed replay
growth at 1.686x. **No exactness-failed prototype was performance-timed.** The
first direct failure is a labeled manual transcription of tool stdout;
subsequent raw failure logs are retained.

| Gate | Part 1 | Part 2 | Part 3 | Part 4 |
|---|---|---|---|---|
| B64/T512 <=3x | 3.823 FAIL | 2.655 PASS | 2.202 PASS | **1.558 PASS** |
| B64/T4096 <=2x | not run | 3.311 FAIL | 2.883 FAIL | **2.754 FAIL** |
| Step/replay <=1.5x per S doubling | not validated | FAIL | FAIL | **PASS** |
| Aligned spill cliff, step/replay | not isolated | not isolated | 1.551 / 1.755 FAIL | **1.022 / 1.092** |
| Inclusive activities <=10 | >=1536 FAIL | 100 FAIL | 59 / 61 FAIL | **59 / 61 FAIL** |
| Zero steady host syncs | FAIL | PASS | PASS | **PASS** |
| Historical 2M curve match | PASS | PASS | PASS | **PASS** |
| Exact suite | 114 | 123 | 138 | **141** |

## Remaining work and operations

The complete arbitrary-slot live pool, long-lived-hole compaction, native cold
backing integration and exact reductions without full logical-S operands
remain unfinished. T4096 and inclusive native launch fusion remain failed.
[WORK_ORDERS.md](WORK_ORDERS.md) records those delivery blockers explicitly.

All GPU execution used `run_leased.sh`, physical card **0**, exact project
Python and whitelist 0/1/5/6. Certified 2/3/4 and forbidden 7 were excluded.
There were no .pt files, WG machinery, workers, delegation, Frontier/Slurm or
resilient training changes; scheduler and resilient requirement IDs do not apply.
AGENTS.md and CLAUDE.md remain synchronized; unrelated workspace edits were
preserved. No callable `contact_supervisor` tool was exposed; the search and
progress limitations are in [part4_coordination.txt](receipts/part4_coordination.txt).

Reproduce the numerical ledger using the exact project interpreter on
[part4_analysis.py](receipts/part4_analysis.py), using retained JSON/XML inputs
only. [part4_seven_source_sha256.json](receipts/part4_seven_source_sha256.json)
identifies the final implementation and test sources.
