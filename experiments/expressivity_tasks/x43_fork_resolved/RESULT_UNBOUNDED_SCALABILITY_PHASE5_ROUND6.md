# Round6 IO doctrine: audit blocked, no fused candidate

**INCOMPLETE; real1.3B remains NO-GO.** The compulsory-traffic receipt landed
first in bb3997ac and was pushed before subsequent component instrumentation.
Fresh physical-counter collection fails ERR_NVGPUCTRPERM, exit1. The requested
measured physical-traffic gap is unavailable. No fused-cycle design, kernel,
backward change or production promotion was made from an invented gap.

## Traffic receipt

The T1024 peak257 fixture has exactly 230,015 observed read-time live-slot
visits. One-pass key/content + query/260-feature + incoming/outgoing carry
payload is **180,075,776 bytes/head**, requiring **358.224GB/s** to fit the
necessary 0.502690909ms/head 24h budget. A two-allocation-per-position ramp
requires 807,944,192bytes/1607.239GB/s; a persistent full2048 store requires
1,614,036,992bytes/3210.794GB/s. These are distinct occupancy models at64/128
key/content width, not measured full1.3B throughput. Metadata, dirty-state
persistence, initialization, outputs, dense and backward work are additional.

These payloads are conditional slot-to-fast-tier traffic, **not an unconditional
HBM floor**. Cache residency can avoid repeated device-DRAM fetches. Carries
can stay resident in a persistent kernel. A valid architecture-physics verdict
must specify tier capacity/residency and occupancy, not merely divide source
loads by nominal memory bandwidth. Architecture affordability is UNDETERMINED.

Fresh ncu permission preflight on broker physical4 requests
dram__bytes_read.sum/dram__bytes_write.sum and fails with driver
RmProfilingAdminOnly=1. R5 CUPTI measures kernel time, not these bytes.
Actual physical bytes and actual/floor ratio remain null in audit.json.
No privilege escalation or driver modification attempted.

Source-level payload enumeration gives another useful, explicitly different
number: with observed BS2048,1024 pre updates and257 enabled post updates,
update_value requests 2,686,451,712bytes of value loads/stores; READ requests
1,073,741,824 value bytes; scoring requests536,870,912 key bytes. Sum
4,297,064,448bytes,23.863x the conditional payload, before metadata/scratch.
This is a source operation count, NOT measured DRAM traffic or its gap; cache,
compiler elimination, duplicate lanes and transactions affect physical traffic.
Even disabled/empty corrections traverse value pages in the current source.

## Unchanged R5 component measurement

Corrected component_fixed.json: one warmed CUPTI iteration on physical4,
actual T1024/S2049/BS2048; two repeat captures agree closely. Control is split
by the alternating ordered pre/post launch schedule, checked for2048 calls.

| Component | calls | device ms |
|---|---:|---:|
| initialization |1|0.002336|
| pre-control (CORRECT/array path and metadata) |1024|40.533595|
| addressing |1024|3.728771|
| normalization |1024|1.855632|
| READ, gather and carries |2048|27.862962|
| post-control (POP/stream write/decay and metadata) |1024|16.094445|
| sum |6145|90.077741|

Instrumented device span229.594224ms, sum90.077741ms:139.516483ms idle
within this instrumented span.6145 cuLaunchKernelEx host calls total26.219161ms;
host and device intervals overlap and must not be added. Neither value is a
certified attribution of uninstrumented overhead. First capture incorrectly
searched only cudaLaunchKernel and reported zero CPU calls; retain it as a
superseded CPU-launch measurement, corrected by cuLaunchKernelEx discovery.

Separate five-sample synchronized wrapper-inclusive medians: unchanged
R4 255.994776ms and unchanged R5 211.404172ms. Complete read/state/raw-trace
and repeat fingerprints match. This is a fresh R5 observation, not an R6 gain.
It projects420.545days native-only with the standing22*2*2/T1024,2B-target
scenario; dense/replay/optimizer/source/checkpoint excluded. Historical R5
191.888684ms/381.723005days remains a separate run, not overwritten.

The inherited '~164ms' is wall total minus a separate instrumented READ time;
it is not an additive measured component bucket. This benchmark supplies
preprojected inputs and executes forward only: content projection and backward
are absent. NativeReplay already saves inputs/output and recomputes through
c14_b_control_graph.backward; that replay's autograd retains immutable versions.
Profiling its retained tensors/recompute tradeoff remains separate uncompleted
work. Gather versus carries and individual control suboperations remain fused
and unresolved. No backward saving/recompute improvement is claimed.

## Resume and limits

The operator's measured-traffic-first prerequisite is not satisfied. Work order
3-5 remains unrun: no fused pass, new backward, ten-shape R6 sweep or GO claim.
No training/checkpoint exists. RESUME.md gives exact outstanding work and
fresh-path commands. This stop is a hardware-counter permission blocker,
not a runtime ceiling or a candidate semantic failure. A supervisor decision
about restoring counter access or changing the prerequisite cannot be obtained:
tool name/description discovery exposes no contact_supervisor/intercom transport.
No contact/approval is claimed and no alternate messaging service was used.

Manual notes: serial attended work, only broker physical4, approved venv Python;
all leases released and physical4 idle on final check. No WG service, delegation,
training, protected-card workload, corpus change, sxgc or /mnt/nvme2n1 operation,
or .pt commit. AGENTS.md/CLAUDE.md unchanged. No resilient DiLoCo changes;
gap-matrix IDs inapplicable. Unrelated existing workspace changes preserved.
Existing R5 tests were not rerun as no production/native arithmetic changed.
