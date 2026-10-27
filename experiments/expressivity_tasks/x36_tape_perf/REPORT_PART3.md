# X36 part 3 — S-cliff isolated and cost reduced; three gates remain failed

**Partial engineering delivery.** B64/T512 is **2.202x PASS**
(365.153 / 165.821 ms; <=3). B64/T4096 is **2.883x FAIL**
(6170.573 / 2140.605 ms; <=2). Inclusive launches are **59 at T512,
61 at T4096: FAIL** against <=10; the custom phase count is **7: PASS**.
Capacity growth also **FAILS** the requested <=1.5x step **and replay** per
capacity doubling. Zero steady host synchronizations and the 2M-token curve
match remain **PASS**. The complete final suite is **138 passed**.

The canonical machine report is [summary.json](receipts/summary.json).
[Part 2](REPORT_PART2.md) and its [summary](receipts/summary_part2.json) are archived.
No failed target is waived by the improvements below.

## Capacity: decomposition before tuning

Fixed B64/T512, two warmups and five raw wall samples per point. Component
stream-event runs and profiles are separate from those samples. Fresh before
measurements are in [part3_baseline_capacity.json](receipts/part3_baseline_capacity.json);
final completed points and their source files are indexed in
[part3_measurements.json](receipts/part3_measurements.json).

| S | Before step ms | Delivered step ms | Before replay ms | Delivered replay ms | Warps / regs / compiler n_spills |
|---|---:|---:|---:|---:|---|
| 2944 | 436.084 | 365.153 | 237.484 | 164.920 | 8 / 255 / 32 |
| 5888 | 837.549 | 528.386 | 638.848 | 329.601 | 16 / 128 / 78 |
| 11776 | 4078.613 | 1208.622 | 3947.227 | 1012.803 | 16 / 128 / 392 |

S2944->S5888 costs **1.447x step / 1.999x replay**; S5888->S11776
costs **2.287x step / 3.073x replay**. Thus the first step doubling passes,
both replay doublings and the second step doubling fail. Four times capacity
costs **3.310x step / 6.141x replay**, improved from part 2's 9.206x / 16.426x,
but still outside the new target.

GPU busy times below come from correlation-attributed separate profiles.
The allocator column is CPU self time and overlaps device work. The journal
column is the matched no-grad cell ablation estimate, **not** an additive
in-kernel timer; scan already includes journal capture. Full samples and
scope definitions are in [part3_decomposition.json](receipts/part3_decomposition.json).

| S | Scan GPU ms | Journal estimate ms | Other forward GPU ms | Replay adjoint GPU ms | Fused replay setup ms | Rebase ms | Other replay GPU ms | Allocator CPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2944 | 43.385 | 5.915 | 2.556 | 160.529 | 1.869 | 1.746 | 0.689 | 0.672 |
| 5888 | 43.582 | 8.358 | 4.641 | 322.914 | 3.747 | 1.719 | 0.690 | 0.687 |
| 11776 | 44.822 | 8.701 | 8.671 | 1000.665 | 7.506 | 1.758 | 0.695 | 0.436 |

### Exact cliff mechanism and measured floor

The reverse program keeps full-`BS` strength adjoints, evaluated strengths,
probabilities, weights and value-dot adjoints, where `BS=next_power_of_two(S)`.
The important boundary is **S8192->S8224**: both capacities stay 32-slot aligned,
physical capacity grows only **0.391%**, but BS doubles. B64/T512 and final
live/head=256 are fixed; warps=16, the explicit register budget=128/thread,
and shared memory=8192 bytes stay fixed.

| S | BS | Step ms | Replay ms | Regs/thread | Compiler n_spills | Shared bytes |
|---|---:|---:|---:|---:|---:|---:|
| 8192 | 8192 | 797.527 | 588.731 | 128 | 166 | 8192 |
| 8224 | 16384 | 1236.824 | 1033.313 | 128 | 388 | 8192 |

That small aligned capacity change costs **1.551x step / 1.755x replay**.
Five BS-wide float temporaries describe 81,920 float lanes at BS16384,
before index and exact-decay temporaries; the selected 16x32x128 register
budget is 65,536 registers per program. The compiler's spill jump, unchanged
shared memory, stable live count and replay-only time jump identify a
**padded full-S register/spill layout cliff**, rather than a host-allocation
or journal-copy bottleneck. Actual allocator self time stays below 0.7 ms;
replay setup/rebase/other copies total about 10 ms at S11776 while its
adjoint kernel takes about 1001 ms.

This is the **observed floor of the retained exact layout**, not a proof of
an irreducible hardware lower bound. Avoiding padded register state while
preserving the oracle's reduction tree remains implementation work. Naive
streamed temporaries regressed large S, and 32-warps failed exact gradients
(including the final SoA fresh-prefix trial: max error 1.9073486e-5).

## Long context: occupied payload and exactness constraints

| T, S2944 | Base ms | Tape ms | Ratio | Capture ms/token | Replay ms/token | Final live/head |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 40.396 | 87.658 | 2.170x | 0.0813 | 0.3478 | 64 |
| 512 | 165.821 | 365.153 | 2.202x | 0.0906 | 0.3221 | 256 |
| 1024 | 439.901 | 905.571 | 2.059x | 0.1124 | 0.3711 | 512 |
| 4096 | 2140.605 | 6170.573 | 2.883x | 0.2602 | 0.7156 | 2048 |

The T4096 profiler attributes about 1.06 s to capture and 2.93 s to replay;
the remaining step is close to the matched base cost. An independent,
unmeasured routing-trace reduction verifies **1,073,741,824 live-slot/token pairs**,
**524,288 allocations, zero frees**, and final live/head=2048.
The corresponding logical payload accesses are **0.825 TB forward +
2.199 TB replay**. They include native values, value-adjoint read/write and
two native-key reads; frozen key adjoints are omitted. These are logical
access volumes, **not measured DRAM bytes**. The trace and per-token occupied
curve are in [part3_t4096_traffic.json](receipts/part3_t4096_traffic.json).

A separate B2/H4 cell-only control holds live/head=64, disables writes/pops,
and verifies four warps at T512 and T4096. Scan/token grows **1.071x**;
adjoint/token is **0.595x**. See
[part3_fixed_live.json](receipts/part3_fixed_live.json). The production memory
schedules differ across T; this control is not a replacement B64 cost gate.
It separates growing occupancy from an intrinsic long-T penalty. Production
still performs per-token payload/adjoint streaming on the larger live set,
and its measured ratio remains 2.883x. Four-warps retain the long-context
oracle reduction; fresh eight-warps passed random T4096 but failed the new
dense T4096 byte fixture (2.6941299e-5), so that switch was reverted.
No nonsequential-device-T or flat-time-under-growing-occupancy claim is made.

## Delivered implementation and inclusive launches

* Replay setup fuses 21 owned-field copies/transposes and native-adjoint
  initialization into one kernel, including strided/expanded gradient inputs.
* Key adjoints are enabled when input keys or imported keys require gradients.
  The LM already detaches keys, so its unused key-adjoint matrix and stores
  are eliminated. Required key/native gradients retain the exact path.
* SoA replay is re-enabled at large S with an explicit 128-register budget;
  the old AoS fallback was slower with that resource policy. Small/short
  reverse offsets use int64; long/large offsets retain checked int32 to avoid
  the measured wide-address register regression. Debug, overflow sanitation,
  journal bounds, ownership checks and publication fences remain enabled.
* Long-T reverse uses bounded independent exact-decay tiles and streamed
  float temporaries without changing the full-S reductions. Forward tracks
  actual allocation reach, and long-T reads cache the previous exact strength.
  A touched row selects its authoritative reference before the next single
  rounded decay multiplication; chunk entry evaluates the original reference.

The final model profiles contain **42 forward + 17 reverse = 59** activities
at T512. T4096 has **44 + 17 = 61**; its >2^31-element freed trace splits
native fills/copies. Seven custom kernels are initialize-free, scan,
materialize, fused replay setup, two rebase/restore launches and reverse.
The remaining **52–54 native activities** are chiefly forward state/metadata
initialization and input/output materialization plus mapping transfers.
Thus the narrow phase target passes and the inclusive <=10 target fails.

Whole-step profiles include nonblocking pinned ingestion: **zero stream/event
synchronizations and zero DtoH** at every measured registered shape. Two
`cudaDeviceSynchronize` events per profile are the explicit measurement and
profiler-exit boundaries. CPU-local optimizer scalars are not GPU syncs.
Profiles are linked from [summary.json](receipts/summary.json). Raw Chrome traces are losslessly gzip-archived;
[part3_trace_archive.json](receipts/part3_trace_archive.json) maps the original
stdout names and records verified raw/decompressed SHA256 values. The profile
reader reproduces every canonical profile dictionary exactly from these archives.

## Exactness, curves and rejected experiments

The final suite has **138 passes**, including full B64/S11776/T512 bytes,
dense/random T4096, long imported/consuming continuation, frozen-key gradients,
strided/expanded native adjoints, aligned padding boundaries, zero-sign bytes,
zeroed address-log ownership and complete private-log restoration. The
historical X31 sources remain unchanged. A test-only wider address-rebase
copy tile lets the historical dense T4096 oracle fit CUDA's grid.y limit;
its floating-point expressions are unchanged.

Both checkpoint-free **512-step / 2,097,152-token** arms match all historical
X34 loss samples with max error **9.5367431640625e-7**. Base/tape medians are
**23.374 / 125.842 ms** at B8;
that 5.384x curve-run ratio is distinct from the registered B64 gate.
Raw curves: [part3_training_slice.json](receipts/part3_training_slice.json).

[part3_attempts.json](receipts/part3_attempts.json) indexes parity failures,
regressions and the corrected diagnostic OOM. Widening all reverse indices
reduced automatic overflow-assert sites 74->13 but increased long-T spills
68->332 and regressed its step to 7920 ms; only the measured bounded fast-path
use remains. Compiler IR/PTX and the Triton sanitizer-source excerpt are
retained. No failed prototype is timed after an exactness failure.

The first separate occupancy diagnostic requested a 23 GiB temporary while
counting a 3 GiB int8 trace and OOMed **after** valid timings/profiles. That
failed record remains in `part3_delivered_shape_gates.json`; the corrected
bounded integer reduction and complete retry are in `part3_t4096_traffic.json`.
The canonical measurement index uses the retry's matched T4096 pair.

All GPU work used the broker wrapper, physical card 0, exact project Python,
and whitelist 0/1/5/6. Certified 2/3/4 and forbidden 7 were excluded from
workloads. No .pt checkpoints, WG machinery, delegation, Frontier/Slurm or
resilient-training work was used; scheduler and resilient requirement IDs are
inapplicable. AGENTS.md and CLAUDE.md remain synchronized. No callable
Intercom/contact_supervisor tool was exposed; coordination limitations and
progress are recorded in `receipts/part3_coordination.txt`.
