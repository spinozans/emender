# X40/X43 gap decomposition: measured redundancy; full-step stress ratio blocked

Direct reuse of the existing trained keys has **no cross-head or cross-layer
sharing win in this sample: 1.000x**. The requested B16/T1024, peak 257,
1.3B full-step ratio is **not measured**: forward and backward completed, but
AdamW moment allocation exhausted the permitted 48 GB card. This is a failed
registration-facing measurement, not a successful performance receipt.

All evidence is under `receipts/gap_decomposition/`. No training run, retained
optimizer update, checkpoint write, production kernel edit, or protocol change.

## 1. Cross-head / cross-layer redundancy

PB3 retains operation/occupancy telemetry, not live-key sets or read-weight
overlap. `redundancy_final.json` adds a short read-only instrumented pass over
the frozen B1-v2 checkpoint, SHA256
`9c87a6a306b708608d7f466dbd62c70c53263c5322f24963b91b94afc8f4def8`.
The sample is the first sixteen 128-byte windows of training document 0, with
fresh state per window and a deterministic CPU seed 3401 consultation keep mask.
This is not a replay of the original training dropout RNG trajectory.

GPU projections use the existing model. The independent CPU scalar store
records every pre-read live-key set and softmax weight. Its read outputs differ
from native outputs by at most 1.193e-6 across the four heads. Local raw bytes
are document-hash checked and matched to `documents.jsonl`; the consultation
shard and local offsets are hash bound. No original NVMe source is accessed.

| Pair: layer/head | Mean nearest-key cosine A to B | Exact key-set / top 8 Jaccard | Read mass on keys held by other head, either direction |
|---|---:|---:|---:|
|0/0 to 0/1|0.2505|0 / 0|0|
|0/0 to 1/0|0.3164|0 / 0|0|
|0/0 to 1/1|0.1756|0 / 0|0|
|0/1 to 1/0|0.1317|0 / 0|0|
|0/1 to 1/1|0.3426|0 / 0|0|
|1/0 to 1/1|0.1636|0 / 0|0|

The receipt retains min/p10/p50/p90/max distributions and sample counts.
The largest nearest-key cosine is 0.88291; read mass at cosine >=0.95 is also
zero in both directions for every pair. Pair statistics condition on both
stores being nonempty. Top 8 means the sets of exact keys selected by the eight
largest addressing weights; repeated keys can reduce set cardinality.

At each position, sum of per-head unique-key counts divided by the four-head
union is exactly 1 at all 2042 nonempty sampled positions. Thus exact cross-head
and cross-layer key sharing gives **1.000x** storage reduction, and no evidenced
addressing-cost reduction. The larger 1.76306x sum-live/union number includes
**within-head repeated keys**; it is not a GQA win. `redundancy.json` is the
superseded preliminary aggregation, preserved with that correction in
`attempt_audit.json` and covered by dedicated regression tests.

**Decision:** the proposed direct key-redundancy lever is dead on this sample;
it does not earn a 2–10x sharing claim or a sharing registration. Separate head
coordinate systems make raw cosine a limited test of semantic equivalence.
A newly tied or aligned geometry could behave differently, but its quality and
speed are unmeasured. Even identical keys do not eliminate distinct queries'
dot products. Values, strengths, ownership and lifecycle cannot automatically
be merged from this key-only evidence. Read masses are addressing softmax
weights before strength, not fractions of output value magnitude.

## 2. B16 full-step measurements

The disposable byte model has 1,301,052,902 parameters, width 2048, 22 layers,
two store heads per layer, B16/T1024, and masked document tails. Existing dense
BF16 autocast and FP32 store arithmetic are retained. This is a performance
fixture with random initial weights, not a trained 1.3B checkpoint or quality
experiment. Each iteration restores weights outside timing and discards all
updates after the run. Timing includes forward, checkpoint recomputation,
existing PyTorch replay backward, clipping and AdamW update; data assembly,
weight restoration and gradient reset are outside timing.

R5 native forward is graph captured, with reset initialization inside replay;
backward uses the existing captured replay adapter and its mandatory
native/replay tolerance check. Compilation/capture and the first two samples
are excluded from reported medians; all five samples are retained.

| Fixture | Physical S / BS | Observed peak | Full-step median, last 3 | Outcome |
|---|---|---:|---:|---|
|R5 controlled budgeted|48 / 64|48|7631.602 ms|measured, `stress48_fixed.json`|
|R5 controlled unbounded|2049 / 2048|257|unavailable|AdamW update 0 OOM, `stress2049_fixed.json`|
|Natural zero-consultation, budgeted capacity|48 / 64|38|7491.645 ms|supplement, `full48.json`|
|Natural zero-consultation, unbounded capacity|257 / 512|37|11552.679 ms|supplement, `full257.json`|

The controlled fixture is the existing R5 construction: constant first-token
value, pop=0, append=20, write enabled for 257 positions then disabled, decay=1,
zero routing coefficients and no array insertions. Backbone inputs are real
document-bounded bytes. The fixture is applied identically to checkpoint's
initial forward and recomputation. It is not natural trained occupancy.

Unbounded graph read/trace outputs were byte-exact against eager R5 before the
failure. Forward and backward finished at peak 257; optimizer allocation of a
112 MiB second-moment tensor failed with only 28.69 MiB free. CUDA reported
47.35 GiB process use, including 28.49 GiB in private pools. This was not a timeout
or a measured runtime ceiling. There is **no complete peak 257 full-step time,
no valid stress ratio, and no full-step BF16 forecast calibrated to it**.
Changing optimizer/state storage or graph allocation to make it fit would
extend this measurement task; no such implementation was introduced.

The supplemental natural low-occupancy full-step ratio is **1.542075x**.
It does not satisfy the requested peak 257 receipt and must not be presented as
achieving the unbounded target. Historical PB3 budgeted heads have 32/64 caps;
the uniform 48-slot performance control is not a remeasurement of that trained
arm's quality or end-to-end exposure rate.

An initial stress wrapper omitted the fixture from checkpoint's initial
no-grad forward; that run is rejected, regardless of its timing.
The first 2049 capture also failed on a host-to-device valid-length construction
inside the existing reach-bounded wrapper. Prestaging the identical immutable
length vector in the measurement adapter fixed capture without editing the
production wrapper. Failed attempts, exact corrected probe sources, lease
receipts and terminal exits are retained. See `attempt_audit.json`.

## BF16 analytic simulation and stacked-lever arithmetic

`bf16_analytic.json` uses the unchanged R6 logical payload model. At T1024,
peak 257 is 180.075776 MB/head/sequence and peak 48 is 40.269824 MB. Halving all
modeled floating payload gives 90.037888 MB and 20.134912 MB, respectively;
at B16 these are 1440.606208 MB and 322.158592 MB per head invocation.
The **traffic-only projected intra-family ratio is 4.471730x**, before and
after halving both arms. It is a conditional fast-tier payload calculation,
not measured compulsory HBM traffic or a full-step prediction.

For common unchanged cost C and payload-time coefficient A, the full-step BF16
scenario is `(C + A*M_U/2)/(C + A*M_B/2)`. Neither C nor A is calibrated by the
failed stress pair. Sensitivity examples are retained, explicitly unfitted.
Halving only the numerator of the FP32 ratio would be wrong when both siblings
use the same precision change. FP32 accumulation, metadata, backward, dense,
launch and optimizer work need not halve. No store BF16 code was implemented.

Keep the operator's goals separate: the receipted ~1.6x small-regime improvement
(phase 1 reports 1.639x for its particular wrapper/baseline), the ~2.3x
occupancy-proportional scenario, and the desired 1.6–2.5x unbounded/budgeted
full-step ratio are different quantities. The 2.3x scenario is not a universal
full-step lower bound; this R5 peak 257 stress traffic scenario is different.
The supplemental 1.542x low-occupancy result cannot replace the missing stress
ratio. The historical byte-versus-token 15–25x comparison cannot supply it.

For affected step fraction f and independent reductions s, total speedup is
`1 / ((1-f) + f/s)`. Insert **sharing s=1.000** from this receipt. Precision
has at most 2x for the modeled payload fraction, not the whole step, and affects
both siblings. Fusion remains unmeasured/compiler-rejected in round 8. Round 8's
B16 forward-store graph benefit 2.036x is already represented by graph replay
here; multiplying it in again double counts it. Neither a 2–10x sharing win nor
a certified stacked path to the operator target follows from these receipts.
Round 9 has no measured direct-key-sharing case; control-fusion remains an
unproven candidate, and the registration-facing full-step ratio remains open.

## Validation and residual risks

Seven focused accounting/boundary tests pass (`tests_final.txt`). No new kernel
or production promotion is claimed; standing broad kernel certification was
not rerun for this measurement-only adapter. All GPU execution was serial,
attended, exclusively leased physical 4, with the approved venv Python. Final
process/GPU evidence is retained. Card 1, sxgc, the original NVMe source and
existing checkpoints were untouched. No WorksGood service or subagents used.
No resilient DiLoCo behavior was changed or run; gap-matrix IDs do not apply.

Residual risks are the one-document redundancy sample, independent key
coordinates, untested tied-geometry quality, synthetic scale initialization and
routing, the failed full-step stress pair, uncalibrated BF16 cost fractions,
and NVIDIA-only execution. Runtime/compiler and shell/test setup failures are
listed rather than silently discarded. The tool catalog had no
`contact_supervisor` capability; no supervisor contact or approval is asserted.
