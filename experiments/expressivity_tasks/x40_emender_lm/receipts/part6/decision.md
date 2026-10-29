# Part 6 coordination and scope

Tool discovery searched contact_supervisor first; no such tool is available in
this runtime. No alternate messaging service, WG service or subagents is used.
GPU work retains run_leased.sh's physical 1/5/6 allowlist. Cards 0/2/3/4/7
are untouched. Existing unrelated working-tree changes are preserved.

The baseline scale runner already calls nonreentrant checkpoint for each block.
Replay stores native owners, trace and journal in ctx attributes, outside
save_for_backward, so that checkpoint cannot release these objects. Additionally,
Cell.gate_mean retains an auxiliary graph outside the checkpoint return value.
The rejected part-5 reentrant checkpoint entered the nonrecording native path.
The new isolated helper forces a recorded forward in both checkpoint passes,
returns the auxiliary gate through checkpoint, and clears the cell attribute.
It leaves pilot-family configs and production defaults unchanged.

X36 part 6 pools *adjoint rows*, not event journal reservations. Its mutation-count
journal allocator is already imported by X40. A measured-max event allocator
would be a further change; no existing sparse pool machinery is misrepresented
as already providing it. Allocation reductions require independent exactness
and overflow checks before performance acceptance.

Recorded checkpoint B1/T512 passes exact outputs, auxiliary gate, input gradients
and every parameter gradient. This isolates the prior nonrecording-path defect
without weakening exactness. T2048 validation is next; no throughput is accepted
from this block-only gate. Static memory arithmetic is explicitly marked partial:
part5 lacks snapshots for activation/replay peaks, and reserved-minus-allocated
is not silently labeled fragmentation.

Recorded checkpoint B1/T2048 also passes exact full-block gradients. The formerly
failing 150M B128/T512 arm now reaches multiple full accumulated updates with
192 native replays/update (12 blocks x 4 heads x 4 microbatches). This is smoke
validation only until a full 20-update receipt exists. Projection precision in
the new probe is explicitly fp32, bounded; both architectures use that protocol.

The optional completed-journal packing candidate retains actual event prefixes
at the measured maximum lane length, preserving the existing writer reservation
and its overflow assertions. It cannot be described as eliminating transient
reservation peaks. Its host maximum read must be counted in throughput.

The first instrumented historical OOM (150M B64/T512, baseline flags disabled)
reproduces the 4-GiB failed request. Allocator history's requested-byte high water
is 40,008,128,512 bytes; journals account for 30,797,772,800 bytes. This confirms
journal retention dominates that shape. The separate new smoke completes with
5,235,933,696 peak allocated bytes. This is a combined checkpoint/accumulation/
bounded-projection change, not an isolated checkpoint or journal improvement.
The full surface driver requires fresh production pressure-inclusive parity at
both contexts, bf16 checkpoint exactness gates, and 20 synchronized measured
updates after three warmups. No pilot budgets are changed.

Fresh pressure-inclusive production parity now passes for T512 and T2048 with
all four existing opt-in performance flags. Completed-journal packing and
recorded checkpointing pass bf16 full-block output, gate, input-gradient and
parameter-gradient equality at both contexts, in addition to their fp32 gates.
The first sustained 22M B64/T512 pure arm completes 3+20 updates with all 48
native replays per accumulated update. The matched transformer arm also completes.
No extrapolated surface or 100k result is claimed.

A more precise reanalysis of the retained part5 B128 trace separates autograd
worker launches from the main-thread backward annotation and excludes GPU
mirrored annotation ranges. Backward device busy is 855.651 ms: replay 596.760
ms (69.74%), core 258.891 ms (30.26%). The synchronous host backward range is
1047.524 ms, with replay host ranges totaling 893.902 ms. Required wall saving
remains 540.277 ms at the historical 54,812.6 tok/s point. These are historical
trace arithmetic, not a fresh part6 throughput or a floor. An earlier transient
analysis that restricted all launches to the main thread was corrected before
acceptance: it omitted core launches on autograd worker threads.

Next scoped pilot lever: overlap independent heads on CUDA streams while keeping
native warp count and every kernel's floating reduction tree unchanged. This
must pass full Cell output/gate/input/parameter bitwise gates before timing.
Shared-feature gradient accumulation order remains a possible exactness blocker;
there is no advance permission to relax it. Default Cell.forward is never edited.

The completed 22M B64/T2048 pure arm also fits and completes 20 measured
updates. Full scale runs keep the original heterogeneous per-head budgets:
64/256/736/1472 slots at horizons 128/512/2048/8192. Native payload arithmetic
uses the sum of these budgets, with actual 6/12/24 block counts. A scale-only
quarter-budget option is recorded as unvalidated and unapplied; no native-store
dominance has justified changing the pilot family.

A further scoped occupancy lever is prepared for parity-first evaluation after
surface completion: cap replay registers at 128/64 without changing num_warps=4
or the floating expression tree. Compiler spilling may still change behavior,
so production zero-tolerance parity and matched v1 curves remain mandatory.
If standalone and independent-head overlap gates pass, the repaired recorded
checkpoint can make a physical B256 pilot feasible. Such a timing remains
unqualified until a matched-token curve at that physical/effective batch passes.
No new compiler setting or stream policy is promoted to production defaults.

Surface progress: 12/24 arms and 6/12 matched ratios now complete, all measured arms passed. Both 150M T512 effective batches now fit, with identical weighted microbatch token counts and expected native replay counts. No long/larger missing result is extrapolated.

New T512 memory receipts: 150M pure B64/B128 both sustain about 7.67k tok/s,
with peaks 4,813,245,952 / 4,811,696,128 bytes. The historical B64 baseline
reproduced OOM with about 40.0 GB logical live allocation at its high water;
journals account for about 30.8 GB. These are combined-path before/after
measurements, not isolated per-fix attribution. The allocator analyzer now
labels alloc/free_requested history precisely and reconstructs pending physical
allocations and inactive gaps at each OOM; it does not conflate terminal cache
slack with fragmentation at failure.

The pilot overlap plan additionally includes deferring journal prefix packing
to a whole-Cell boundary. All four native head scans can finish concurrently,
then one combined maximum transfer sizes all four records. Only ctx.log and
its matching pitch are rebound; native owners, decision traces, boundaries and
all before-images remain unchanged. This needs its own full-Cell bitwise gate.
It is not active in the ongoing scale measurements or production defaults.

Separate memory attribution controls are prepared for all eight failing pure
150M/1.3B shapes: (0) legacy nonreentrant checkpoint with bounded projection,
(a) recorded checkpoint at the full physical batch, (b) B32 accumulation without
packing, (c) the same accumulation with measured-prefix packing. Each diagnostic
runs two complete updates so the second forward/backward includes initialized
Adam states. Individual stages may still OOM; those remain failures. These
controls isolate the a/b/c changes under one fixed bounded-projection and native
flag protocol, separately from the 3+20 sustained surface. Transient writer
reservation is still reported honestly, even when retained logs are packed.

Surface progress: both 1.3B T512 pure arms complete at full registered capacity;
all 20 T512/earlier-scale arms are now measured or in final control accounting,
and the long 1.3B cases are running sequentially. No failing arm is assigned a
rate. The confirmation gate remains closed until all 24 sustained arms exist.

Gate tightening: new checkpoint/overlap controls compare raw contiguous bytes
(including signed zero) and reject nonfinite values. Earlier numeric-equality
receipts are archived before strict byte revalidation. Six CPU regressions pass.

The overlap probe keeps dense transforms and their backward GEMMs on the
original CUDA stream; only independent native scans overlap. This avoids
changing dense cuBLAS workspace/accumulation order. It also retains query/value
input gradients to distinguish a native exactness break from a dense/shared
feature accumulation break. New projection chunk trials 4096/16384 must pass
the unchanged X36 loss/gradient tolerances before timing; native tolerances
remain zero. Imported native suites run again under each accepted register cap.
Any >=100k candidate receives a separate trace and a >=2M-token same-batch curve
against the accumulated reference. The driver stops further pilot trials only
when that complete qualification passes.

Auxiliary graph lifetime is part of the memory fix: Cell.gate_mean otherwise
retains upstream replay contexts between updates, and their ctx.log/owners are
not save_for_backward-managed tensors. New features returns auxiliary gates
explicitly and clears the attribute; recorded checkpoint first passes discard
these graphs after returning detached outputs. Production model defaults and
pilot-family policy configurations remain unchanged.

Unexpected late failures: the 1.3B pure T2048 arms still OOM at physical B32.
B64 reaches update 19; B128 reaches update 3. Both fail a 1.40-GiB allocation
inside checkpoint recompute, with 39.60 GiB allocated and about 6.3 GiB reserved
but unused. These are failures, not accepted partial throughput. The full traces
and partial console sequences will be archived under surface_before_offload/.

Additional exact memory relief: store only checkpoint inputs on pinned CPU
memory. For width2048/layers24/B32/T2048 this moves about 12.9 GB of saved inputs
per microbatch off device. Inputs return byte-for-byte for recorded recompute;
native stores/journals stay fp32 on device. An inner detached-tensor hook keeps
only ephemeral recorded-forward internals on device without introducing saved
output reference cycles. Transfers are timed. Strict offload byte gates cover
both B1 and noncontiguous-ID B2 before retrying the two failed pure arms.
Controls retain their already-successful memory strategy; this extra pure-only
transfer cost is explicit in the matched protocol and its throughput ratios.

Allocator discovery changes the retry plan: the actual allocator snapshot reports
expandable_segments=False and an empty PYTORCH_CUDA_ALLOC_CONF despite the newer
PYTORCH_ALLOC_CONF setting. This installed CUDA backend still needs the legacy
CUDA-specific variable. The CPU-offloaded B64 diagnostic passes four complete
updates and both byte gates, but its sustained run is intentionally interrupted
(SIGINT to our own Python PID 2149404 only) to test the direct fragmentation fix.
It is archived as an interrupted diagnostic, with no accepted throughput. PID
634691 and all protected cards remain untouched.

The next retry sets both allocator aliases identically, requires an actual
expandable_segments=True snapshot, and re-runs pressure-inclusive production
parity before timing. This can retain checkpoint inputs on GPU while reclaiming
inactive mapped pages. The earlier 1.3B failures and CPU-offload diagnostic stay
in separate archives. Nothing is called a hardware floor or a passing surface.

The CUDA-specific allocator alias is now empirically verified: the new byte
gate reports expandable_segments=True, and fresh pressure-inclusive production
gates pass at T512 and T2048. The two 1.3B long pure arms are retried without
CPU offload under this verified allocator. Their first repeated losses match
the earlier completed updates exactly. The CPU-offload implementation remains
a separately receipted, exact optional memory control, not a qualified rate.
The canonical driver now requests and verifies the real allocator setting and
returns failure if any surface arm fails; its initial source is archived.

Follow-up validation is queued behind the sole remaining surface process; it
cannot run GPU work concurrently. It repeats the 18 historical OOMs with
allocator history, isolates checkpoint/physical-batch/accumulation/packing
memory controls for all eight large pure shapes, and then runs the native
fixtures and the matched-token pilot curve. The exactness gates now compare
raw bytes (including signed zero), with additional B32 tests at both contexts.
Compiler register limits retain the four-warp arithmetic layout; head streams
are separately byte-gated. Physical-batch trials on the original stream are
kept independent of the stream probe so an overlap exactness failure does not
prevent an otherwise safe occupancy trial. All rejected variants remain
untimed. Extra decision-density statistics are collected only in the separate
profile update, outside the sustained timing samples.

An additional scoped fusion candidate now replaces the sparse replay pool's
clone/cat/sentinel/scatter/count/index setup with two integer-only Triton
kernels. It preserves stable logical row ordering and measured maximum sizing;
unsupported large/noncontiguous inputs fall back to the original builder. Its
component maps, pressure-inclusive production parity, native fixtures and
matched curve must pass before timing. A matched fusion-off/fusion-on B128
profile is planned to count actual removed launches. No native floating-point
recurrence is edited. Register variants additionally compare full Cell raw
bytes against an uncapped reference, rather than merely comparing two cells
under the same compiler limit.
