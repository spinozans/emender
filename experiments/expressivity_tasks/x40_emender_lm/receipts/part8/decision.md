# Part 8 execution scope

Searched available tool metadata first for contact_supervisor and Intercom;
no matching tool is available. Work proceeds in the attended session without
WG or delegation. The explicit Cartesian deliverable is 36 arms (12 shapes
by three architectures); the reference to 24 describes the prior two-arm
surface. Protected cards 0/2/3/4/7 and PID 634691 remain untouched.

Read REPORT_PART5.md, the 24 surface receipt filenames and implementation,
REPORT_PART7.md, part7_decision.md and fusion_comparison.json before changes.
The existing part7 report has stale missing fields despite later successful
raw artifacts; raw receipts take precedence over its summary prose.

GDN2 is the registered gdn2-mlp external GatedDeltaNet2 recipe, not the older
fla-gdn GatedDeltaNet wrapper. Use external native chunk Triton kernels plus
installed FLA helpers. Match total pure parameters within 0.1%, retaining the
historical scale names for pure trainable parameters. SwiGLU width is adjusted
to capacity, so these are parameter-matched recipe-family configurations,
not the paper's identical 2304/17 architecture at every scale.

All arms begin at physical batch with no checkpoint, packing or offload.
An actual CUDA OOM alone permits progression through the declared envelope
ordering. Other errors stop the arm. Successful attempts include 3 warmups
and 20 synchronized complete updates. OOM peak is a lower bound on required
memory, not a measured successful peak. Minimal means first fitting envelope
in the declared ordering; it does not establish a globally optimal combination.
FP32 master parameters and AdamW are shared, with bf16 autocast on dense
blocks; baselines additionally use fused bf16 linear cross entropy. Pure
retains the certified fp32 native store and bounded fp32 projection. This
asymmetry is explicit in every receipt. Identical independent synthetic-data
RNG streams avoid initialization-dependent token differences.

No Frontier, Slurm or resilient DiLoCo runs or changes. Their requirement IDs
are inapplicable. AGENTS.md and CLAUDE.md remain unchanged. No .pt artifacts.

UPDATE: fresh pressure-inclusive native production gates pass at T512 and
T2048 before surface timing. Meta construction verifies all three GDN2
capacity matches: 72,451,340 / 250,912,584 / 1,501,831,552 total parameters,
against 72,449,214 / 250,904,904 / 1,501,813,176 pure totals. The first
transformer bare B64/T512 arm completed 20 measured updates; GDN2 native
kernel compilation is in progress. No global envelope flags were enabled.

UPDATE: both bf16 baseline architectures pass bare 22M/T512 at B64/B128;
GDN2 receipts identify GatedDeltaNet2 and FLA 0.5.1. The first pure bare B64
arm also passes. All five new CPU acceptance regressions pass, including
rejection of 19-update timing and retention of all 36 missing arms. The
external recipe and kernel tree hashes are retained in configs.json.

UPDATE / correction: the first GDN2 construction forced the pure backbone
width/depth, producing a nonstandard 42.5x MLP ratio at 22M. These initial
GDN2 timings are rejected as best-practice evidence and archived under
rejected_forced_gdn_backbone/. The dispatcher was stopped while its active
pure checkpoint measurement completed independently. The accepted GDN2
family instead searches width/depth and 64-aligned SwiGLU hidden width for
the nearest registered paper ratio under the same 0.1% capacity constraint.
Only GDN2 rows are rerun; completed transformer/pure rows retain their source
receipts. This corrects the earlier fixed-backbone configuration note above.

The final GDN2 selection uses 16-aligned SwiGLU widths and scales head counts
from registered 2304/8 (128-dimensional key heads), expand_v=2. The actual
width/depth/hidden and parameter counts are in configs.json; the smaller
capacity points are tuned recipe-family variants, not identical paper
hyperparameters. Gate helper outputs are redirected to part8 so historical
part7 parity and curve artifacts are preserved.

UPDATE: accepted GDN2 family configurations are width/depth/MLP-hidden
448/14/1200, 1280/8/3632, 2112/24/6000. Parameter totals match the pure totals
within 0.1%; master parameters and optimizer remain fp32, dense and projection
compute use bf16. The first two accepted GDN2 T512 arms fit bare. At T2048,
both transformer and GDN2 first demonstrate actual bare OOMs before trying
checkpointing; these measured failures override any prior expectation that
all small configurations would fit bare. No fabricated required peak is
assigned to an OOM.

The prior envelope directory was also parsed as 24 raw JSON receipts: all
passed, all checkpointed/packed; the report's 22M/150M/1.3B rates and memory
peaks agree with those rows. The historical bf16 native-store gate is
receipts/part3_bf16/gate.json: failed, enabled=false, with explicit native
read/key/value/strength/routing mismatches. Final rows retain its hash and a
part8 copy to make the intentional precision asymmetry auditable.

UPDATE: 150M pure B128/T512 stopped before timing because the preflight
requested a nonexistent B1/T512 packed-expandable filename. This is a runner
failure, not an OOM or native parity failure. The strict B32/T512 and B32/T2048
packed-expandable gates exist, pass raw-byte equality and bind the unchanged
helpers. The selector now explicitly requires those B32 gates (plus matching
flags); the original failed preflight is retained under preflight_failures/.
A sequential post-surface retry repairs only this missing-receipt class of
failure, before the pilot gate. Genuine runtime/numerical failures are not
reclassified or automatically bypassed.

UPDATE: all 12 small-scale cells now have accepted sustained receipts. At
150M/T512, B64 fits bare for all three architectures; B128 requires checkpointing
for both best-practice baselines. The pure B128 cell awaits its documented
preflight repair, with no rate assigned in the interim. 150M pure B64/T2048
passes checkpoint-only at 7,479 tok/s and 19.92 GiB. Seven CPU regressions now
pass, including both context-specific strict checkpoint receipt lookups.

UPDATE: 150M B64/T2048 sustained points pass for all three arms, each after
an actual bare OOM and with checkpoint-only selected. Both B128/T2048 bf16
baselines also pass checkpoint-only. The pure B128/T2048 full physical-batch
measurement is progressing normally at roughly 25 s/update. No accumulation,
packing or offload has been enabled for any accepted small/mid-scale point.

UPDATE: both 1.3B/T512 B64 bf16 baseline arms require checkpointing after
bare OOM and complete sustained updates normally. Protected cards remain
unchanged: physical 0 still holds 39,106 MiB with the depth computation;
3/4 retain 20,164 MiB; 2/7 are untouched. All accepted v2 measurements use
the same two expandable allocator environment aliases. The v2 runner did
not serialize allocator_settings per attempt, so the final metadata labels
this as a common request rather than claiming an observed per-attempt
backend assertion. Peak allocated and reserved byte counters are observed.

UPDATE: all six 1.3B/T512 arms pass checkpoint-only after bare OOM. Sustained
B64/B128 rates are transformer 5,753/6,040, GDN2 5,936/5,810, pure 3,629/4,585
tok/s. Pure peaks are 27.57/38.32 GiB, so neither packing nor accumulation is
used for those cells. At 1.3B/B64/T2048 the transformer also demonstrates a
physical-batch checkpoint OOM; the next declared envelope is B32 accumulation.
All accepted GDN2 rows carry the final recipe-family config metadata.

UPDATE: 1.3B/B64/T2048 bf16 baselines complete: transformer 6,233 tok/s,
42.75 GiB with checkpoint+B32 accumulation; GDN2 6,364 tok/s, 34.65 GiB with
checkpoint+B16 accumulation. GDN2 B32 actually OOMs, so its smaller microbatch
is measured rather than assumed. Pure physical checkpoint and checkpoint+
packing both OOM; checkpoint+packing+B32 is now measuring normally at roughly
73 s/update. These are independent per-arm envelopes, not shared global flags.

UPDATE / policy correction: review found the initial fallback policy would
carry packing into every pure microbatch, even though checkpoint-only B32
might fit. The ongoing packed B32 long pure attempt was interrupted after
only a few updates (no accepted sustained rate) and preserved under
policy_corrections/. No protected process was signaled. The dispatcher and
its waiting postprocessing shell were stopped, then resumed with the policy
fixed: at each smaller physical microbatch test checkpoint-only before adding
packing. B128 also tests B64 before B32; it does not assume B64 OOM from the
separate B64 arm. Already completed points use unchanged envelope indices and
are retained. This prevents the exact unnecessary-packing problem requested
by the operator. The no-packing long pure B32 variant is now measured first.

UPDATE: the policy correction is confirmed by measurement: 1.3B pure
B64/T2048 completes 20 measured updates with checkpoint+B32 accumulation
and **no packing**. The omitted checkpoint-only variant fits, so packing
would have been unnecessary. Its four retained attempts show bare OOM,
physical checkpoint OOM, physical checkpoint+packing OOM, and checkpoint-only
B32 success. The B128 baseline arms now test B64 before B32 rather than
assuming the separately measured B64 arm's result.

UPDATE: all baseline arms now have sustained receipts. B128/T2048 selects
transformer checkpoint+B32 and GDN2 checkpoint+B16 only after recording
B128, B64 and (for GDN2) B32 OOMs. The final pure B128/T2048 cell has begun
checkpoint-only B32 at roughly 147 s/update. Packing/offload remain disabled.
The repair classifier now checks traceback filenames, since Python's
FileNotFoundError repr omits its filename; a regression prevents silently
skipping the known missing-gate preflight repair. The pilot gate CLI will
exit nonzero if the 100k acceptance predicate fails.

UPDATE: the final B128/T2048 pure arm continues normally through measured
update 15 (three warmups plus iterations through 17), about 146 s per complete
update, checkpoint-only B32. All 10 CPU regressions pass. Postprocessing is
queued sequentially after this process: repair the one missing-gate preflight,
run fresh fused/sparse bare pilot parity/performance/curve, then validate the
full 36-row surface and render ratio curves. No sustained result is inferred
from the still-running final arm.

UPDATE: final large pure B128/T2048 completes all 23 updates: 1,789.55 tok/s,
42.729 GiB, checkpoint-only B32. All large arms have sustained results. The
sequential preflight repair is now executing the previously missing 150M
pure B128/T512 cell, after retaining its original failed preflight. Protected
PID 634691 remains active and untouched. No accepted surface arm needs
packing or offload; mixed physical batches and checkpoint policies are fully
receipted rather than assumed from scale labels.

FINAL: all 36 sustained surface arms pass (9 bare, 27 checkpointed; no
accepted arm uses packing or offload). All bare/fallback OOMs, the failed
preflight, rejected GDN2 prototype and interrupted packed trial remain in
separate receipts. The external GDN2 checkout is exactly the registered
95709fc250357c2dd109361c353192f2aa5913f9 commit and its source tree hash is
unchanged after measurements. Both exported ratio figures were inspected.

The fresh bare fused/sparse pilot measures **54,836.10 tok/s**, failing 100k.
Fresh fusion/production parity passes. Its independent 2,031,616-token curve
passes at maximum loss difference 1.9073486328125e-6 versus 1e-5 tolerance.
No new arithmetic sparsification or full inclusive fusion closure is claimed.
The public report distinguishes this original v1 pilot from the parameter-
targeted 22M scale model; their bare rates are not substituted for each other.

Pure/GDN2 throughput ratios increase across all four scale curves. Pure/
transformer increases monotonically in three; T2048/B128 reverses at 1.3B
(0.2473 -> 0.3557 -> 0.2866). Uniform monotonic confirmation against both
baselines is therefore not established. The reversal is neither averaged
away nor relabeled as a successful universal confirmation. The validated
summary correctly retains overall status=failed and surface_status=passed.

Ten CPU regressions pass; Python compilation and 36-row receipt validation
pass. No native emender implementation or architecture guide changed. The
current main parent was externally amended to 45159576 during this long
experiment; fetch verifies it equals origin/main, and the task's native
source files and AGENTS.md/CLAUDE.md have no working-tree diff. Only part8
sources, its new test/report and its receipts are staged for the requested
single commit. Unrelated work and protected GPU processes are preserved.
