# Round6 first receipt: traffic audit fails physical-counter gate

Read R1-R5 RESULT documents, R1 execution notes, and R5 design/profile/resume
before this audit. No fused-cycle design or implementation has been selected.

The conditional one-pass payload at the R5 T1024, peak257 fixture is
180,075,776 bytes/head: 230,015 live-slot visits * (64 key + 128 content)
FP32 scalars, plus 1024 * (64 query + 260 features) FP32 scalars and
1024 incoming/outgoing two-parity x128 FP32 carries. The 24h necessary
head budget is 0.502690909ms, requiring 358.224 GB/s for this payload alone.
The source model uses read-time occupancy min(t,257): no arrays/deaths,
write enabled through t256, and reads precede stream writes. Actual occupancy
verification is pending the next diagnostic. `audit.json` also models a
two-allocation ramp and a persistent full store; these are distinct scenarios.
Carry transfers here are an explicit model convention, not an irreducible
requirement: a persistent fused CTA could keep those registers resident.
Metadata, dirty-slot writeback, outputs, initialization and backward are extra.
This is a necessary payload model, not total execution cost.

Crucial qualification: slot-to-fast-tier traffic is NOT automatically physical
HBM traffic. Repeated loads can hit cache, and this Ada card has device DRAM;
L2/shared/register residency must be distinguished. A persistent store's
working set can remain cached across positions. Thus the requested per-step
slot payload cannot honestly be called an architecture-physics HBM lower bound.
Physical counters are needed to quantify the current gap.

Fresh Nsight Compute attempt on exclusively broker-leased physical4 exits1:
ERR_NVGPUCTRPERM. `/proc/driver/nvidia/params` reports
`RmProfilingAdminOnly: 1`. No privilege/driver change was attempted.
`traffic_console.txt`, `traffic_terminal.json` and `traffic_lease.txt` retain
the failure. Requested metrics were dram__bytes_read.sum and
dram__bytes_write.sum. The launch-count1 probe was a permission preflight;
even success would require a complete store-cycle capture afterward.
The enclosed parity sweep completes exactly but its heavily instrumented
times are excluded from all performance comparisons.

R5 CUPTI only measures durations, not physical bytes. It cannot supply the
missing number. Actual DRAM bytes and measured actual/floor ratio are null,
not inferred from source loads, compiler spills or wall-minus-kernel time.
Architecture affordable/unaffordable verdict: UNDETERMINED. R5 NO-GO stands.

The initial arithmetic receipt mislabeled a one-allocation uncapped ramp as
full reachable; `audit_attempt1.json` retains it as superseded. Corrected
`audit.json` explicitly names two-allocation and persistent-full scenarios.
No candidate design choice may be justified by a fabricated measured gap.

Supervisor transport discovery by name/description found no callable
contact_supervisor/intercom tool; no contact or approval is claimed.
No resilient DiLoCo changes; requirement IDs inapplicable. No training,
protected-card work, WG service, delegation, corpus changes or .pt artifacts.
