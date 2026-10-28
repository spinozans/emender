# Probe-only telemetry scope

Operator part-5 option (b) authorizes excluding pressure from the scale probe.
Production certify() defaults to certify_pressure=True; no tolerance changes.
Scale values, native payload/strength, routing traces and gradients retain
zero tolerance; evidence wires retain their production thresholds.
Part4 pressure_probe.json establishes nondeterministic unchanged-X31 pressure
and unchanged compared functional fields; diagnostic barrier fixes that probe.
The operator cites X30/X31 telemetry exclusion precedent. The directly located
X31 REPORT.md:854 exclusion is specifically usage EMA, not pressure; it is
not evidence of a historical blanket pressure exclusion. This new probe-only
choice rests on explicit operator authorization and part4 evidence.

contact_supervisor was searched first and is unavailable in this runtime.
Depth PID 634691 remains active on card0; wrapper 1/5/6 retained.

Scale naming: X40's historical "22M" pilot is 21,719,112 trainable /
72,162,960 total parameters (receipts/parameters.json). The new full-stack
surface targets 22M/150M/1.3B *trainable* pure parameters, with frozen token
features in every block. Transformer capacity matches the actual pure TOTAL
within 0.1%, following X40 RECEIPTS.md's registered matching convention.
Both counts are emitted at every point; transformer trainable counts are
larger, so this is not a matched-trainable comparison. Width/layers are
256/6, 768/12, 2048/24; per-block FF widths solve the trainable target.
This is a full memory stack, unlike the historical X42 one-lane adapter.
Identical synthetic token batches, bf16 dense layers, fp32 native store,
block checkpointing, clipping and fused AdamW apply to both arms. Each
successful point retains 3 warmups and 20 synchronized measured updates.
Failures, including actual OOMs, receive JSON receipts and do not imply a
throughput ratio. No inferred timing fills a missing surface cell.

Unexpected discovery: after the corrected T512 scale gate PASSED, T2048 failed
inside unchanged X31 backward.rebase_owned with CUDA invalid argument before
comparison. Its reservation-based grid.y exceeds CUDA's 65535 limit. The
probe-only probe_rebase.py transposes the store and log-block launch axes,
leaving all integral pointer-mapping expressions and masks identical. This
repairs launch geometry, not comparison thresholds or floating operations.
Original failure receipt is retained under scale_initial/. Production oracle
sources and default certification are unchanged. contact_supervisor remains
unavailable; discovery is recorded here instead of using another service.

Parity-first execution: all new optimizations are initially disabled via
X40_BATCHED_READS=0, X40_FRESH_POOL=0, X40_FUSED_INIT=0. The complete scale
surface runs this baseline after a fresh production certification. Each lever
is enabled only in its leased process and must pass the unweakened production
certify() before any timing. No source default is promoted on historical
receipts. The pool candidate extends the existing exact union representation
to an empty entry; event suppression remains the X36 implementation.
Fusion candidate combines only seven fresh native/history initialization
fills into one launch; it does not claim to fuse every inclusive activity.

Progress: all 24 scale arms executed; 6 passed, 18 OOM, only 1/12 matched
ratios exists. This is a failed scale-curve closure. Both rebased scale gates
pass. Read port production gate passes. Its measured forward range improves
242.371 -> 174.467 ms, while replay stays about 655 ms. All 24 replay scans
have 32768 value-consuming events and no pool; existing journal suppression
cannot eliminate that workload. Fresh wall is ~5 sec despite <1 sec native
GPU component time. A separate CPU profile is being collected before choosing
the backward change; repeated specialization of address-sorted mapping tables
is a hypothesis, not yet a measured attribution.

The separate post-read CPU probe (pilot_reads_cpu/cpu_profile.txt) attributes
0.930/1.091 seconds to 48 rebase_owned calls, including 48 cuda_utils.load_binary
calls totaling 0.793 seconds. Forty-eight mapping compile lookups still occur
after warmups. Mapping tables sort owners by allocator addresses; constexpr
sizes/dimensions and differently typed pointer tuples therefore specialize on
address-order permutations. The measured probe has 1.076 sec wall versus ~5 sec
in the earlier 20-sample run; this variability is retained, not silently erased.
The backward candidate uses uniform byte pointer views and runtime integer
sizes/dimensions to author the identical sorted table without address-order
specialization. It also enables the existing exact live/allocation-union pool
for fresh T<=512 entries. Both are opt-in and go through the original exact
production gate before timing. This targets measured replay setup while also
measuring the requested fresh-origin pooled path; it does not claim a new
floating reduction or absent-scatter optimization.

Backward+fresh-pool production parity passes; its 20-update pilot measures
36,126.6 tok/s. Pools activate with 64/256 rows across heads and native reverse
busy time falls to 569.352 ms. The new exact sizing path introduces 24 extra
DtoH transfers (26 total including whole-step telemetry) and 24 extra host
waits. Native event suppression still cannot skip any consuming history.
Fusion production parity passes; seven native/history fills become one launch,
removing exactly 144 kernels across 24 cells (2526 -> 2382 full-update kernels).
Measured wall stays about 36k tok/s. Full 59-launch closure is not claimed.
Both B128 and B256 actually OOM. Releasing the Python logits reference early
also OOMs at B128; that ineffective change was reverted and its patch saved.
The original 153-update v1 comparison passes for 5,013,504 tokens with maximum
loss difference zero; this does not establish 100k throughput.

Further occupancy attempts are justified by the actual allocation failure:
X36's existing ChunkedLoss can bound the vocabulary projection, and reentrant
block checkpointing can bound retained journals. These are explicit benchmark
modes, with actual 24-replay execution asserted on every update. The bounded
projection uses X36's established component tolerances; the native production
zero-tolerance gate remains unchanged. Any such mode needs its own >=2M v1
curve receipt. A scoped 8/16-warp reverse launch trial will go through the exact
production gate before timing, restoring the original launch even on failure.

Additional results: all 109 unchanged X31 fixtures and all 70 X36 fixtures pass
with the four candidate flags enabled (109 + 70 = 179). B128 with the existing
bounded vocabulary projection completes 20 sustained updates at 55,658.5 tok/s;
it remains below 60k/85k/100k. Global eight-warp replay and eight-warps only
for S>=736 both fail exact gradients (~72% differing elements, maximum errors
9.54e-7 and 5.96e-7). Their scoped launch overrides are restored and no timing
is run for either rejected layout. No gradient tolerance is changed.

The extra B256 bounded-projection + reentrant-checkpoint experiment FAILS with
an actual `invalid heap inverse position` device assertion after four updates.
No sustained wall receipt is accepted. Checkpoint code and its proposed
nonrecording-forward gate extension were reverted; the complete patch and
raw failure are retained in checkpoint_reverted.patch and the point's log.
The heap failure remains unexplained and is not relabeled as pressure telemetry.
The recorded production core and its existing certification remain unchanged.

Final qualification audit found that the initial surface runner gated B64/T512
but did not independently require its production-policy T2048 gate. All of its
T2048 arms had OOMed and no long-context throughput was accepted. An additional
B1/T2048 gate now PASSES with the production policies and exact pressure too,
using only the validated integer rebase-grid transpose for the oracle launch.
The long arms are repeated after that gate; initial diagnostic JSONs are
archived under surface_pre_long_gate/. The surface runner now requires both
context gates. This avoids treating retrospective certification as a timing
gate. No successful long-context number is inferred from any failed attempt.

Safety scope after the checkpoint failure: opt-in batched reads now fail closed
for nonrecording execution. Recorded training remains enabled and receives a
fresh post-guard production gate and matched-curve run. This does not assume
the heap failure is caused by checkpointing alone; its root cause is open.
Default inference is unchanged. A CPU regression test checks that the rejected
path stops before any device work. Fast-mode evaluation requires resolving
this failure before its nonrecording path can be enabled.
