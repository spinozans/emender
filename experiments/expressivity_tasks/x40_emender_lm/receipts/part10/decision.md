# Part 10 decision — hold BF16 adoption; 1x is not established

Task gate: **FAILED / incomplete**. No production arithmetic, store, replay,
checkpoint helper or architecture was changed. Experimental sources are opt-in
probe modules only; production never imports the BF16 candidate. No performance
optimization is accepted. No claim that 1x is impossible is made.

## Corrected 1.3B evidence

The shape is 24 blocks, width 2048, 1,300,012,920 trainable parameters,
1,501,813,176 total parameters. The transformer matches total capacity within
0.1%. Both use effective B64, physical B32, two weighted microbatches,
reentrant block checkpointing, GPU checkpoint inputs, BF16 dense compute,
FP32 parameters and fused AdamW. Pure alone packs its native journals and uses
bounded FP32 vocabulary projection; the control uses fused BF16 linear cross
entropy. Thus the reported protocol matches accumulation/checkpoint/storage,
while retaining the explicit production precision/loss asymmetry. It does not
reproduce part8's individually selected B64 checkpoint envelope at T512.

| Exclusive wall seconds | T512 | T2048 |
|---|---:|---:|
| sequential_forward_core | 2.247369 | 15.467662 |
| forward_read_elementwise_tiles | 0.237177 | 1.854635 |
| initial_checkpoint_dense_and_native_setup | 1.057989 | 4.058950 |
| checkpoint_recompute_dense_and_native_setup | 1.062686 | 3.959087 |
| forward_dense_loss_and_native_setup | 0.183863 | 0.866241 |
| backward_dense_loss_and_replay_setup | 1.937353 | 8.848908 |
| reverse_fused_read_math_and_adjoints | 4.926609 | 36.914377 |
| reverse_setup_and_teardown | 0.262357 | 0.322132 |
| clip | 0.023325 | 0.023960 |
| optimizer | 0.058934 | 0.058822 |
| Full diagnostic update | 12.000401 | 72.377731 |


These are exclusive synchronized wall regions from one post-warmup diagnostic
update. Child times are subtracted, and residual wall is under 2 ms. Core and
batched read rows combine initial and checkpoint-recompute passes; the two
dense/checkpoint rows exclude those child regions. The transformer diagnostic
wall is 5.019638 / 21.119440 s
at T512/T2048. Synchronization changes overlap; these ratios are not sustained
surface results. Each pure diagnostic observes 384 core/read launches and 192
reverse calls: four heads × 24 layers × two microbatches × two forward passes,
with only one reverse per recomputed graph. There is no extra native forward
inside reverse, consistent with part9.

The dominant fused reverse is 41.1% / 51.0% of diagnostic wall. Sequential core
is 18.7% / 21.4%, so it is not negligible in this envelope. Reverse Python/setup/
teardown is 2.2% / 0.45%; its elimination alone cannot explain a move from 1.6x
to 1x. Store-event adjoints, undo and read adjoints remain inside the fused
reverse; **their individual wall contributions are not measured**. No-write
sparsity does not prove a dead read gradient. No further event suppression was
landed without a zero-contribution proof and exactness gate.

Crucially, the current read is **not FP32 GEMM**: X35 causal_read._tiles uses
FP32 elementwise multiply/reductions, and reverse_program uses scalar-dimension
FP32 loops. The BF16 candidate changes both dtype and algorithm to MMA. A GPU
peak FP32/BF16 GEMM-rate ratio is not an attribution of the existing 1.6x.
Forward batched reads (both passes) are only 1.98% / 2.56% of this diagnostic;
even deleting them entirely gives at most 1.020x / 1.026x diagnostic speedup.
This candidate leaves the dominant reverse FP32. It cannot justify a 1x claim.

Supplemental clock-instrumented reverse diagnostics run fresh pressure-inclusive
parity before the same 1.3B B64/B32 profile. They collect per-CTA cycle counters
around undo, strength/store adjoints, score/softmax recomputation and read
adjoints. They do not split additive wall time:

| T | Undo/input % | Strength/store adjoints % | Score/softmax recompute % | Read adjoints % |
|---|---:|---:|---:|---:|
| 512 | 8.68 | 16.37 | 4.07 | 70.88 |
| 2048 | 4.87 | 18.80 | 11.79 | 64.53 |


T512: score/softmax recomputation is 4.07% of instrumented reverse CTA cycles (cycle-weighted device estimate 197.45 ms); read adjoints are 70.88%. T2048: score/softmax recomputation is 11.79% of instrumented reverse CTA cycles (cycle-weighted device estimate 4363.98 ms); read adjoints are 64.53%.

These diagnostics distinguish repeated forward score/softmax mathematics from
the read adjoints that consume the final output gradient. They do not support
treating the entire reverse as removable recomputation. The measured bottleneck
points toward read-adjoint execution/memory/parallelism; retaining read outputs
alone does not remove it.

Raw CTA cycles, CUDA-event kernel durations, cycle-weighted device-time estimates
and parity gates live under reverse_clock/. Clock instructions, added barriers
and register pressure perturb execution; cycle sums weight occupancy across
CTAs. These numbers are diagnostic evidence only, not a subtraction from the
original fused wall or an accepted caching speedup. Failures are preserved.
The fused reverse split and a true BF16 reverse remain open, so the evidence
cannot decide whether 1x is reachable without a new read definition.

## BF16 read-only experiment and empirical cost

The isolated candidate is part10_bf16_read.py. It gathers pre-token payload
versions, casts Q/K to BF16, computes tile-32 scores by BF16 MMA with FP32
accumulation, keeps score scaling/exponentials/denominators FP32, casts the
unnormalized probability × strength operand and V to BF16 for the value MMA,
then uses the original FP32 cross-tile reduction. Output storage stays FP32.
It replicates the query to the minimum 16 MMA rows; this is an experimental
implementation, not an optimized lower bound on BF16 read performance.

The native scan/store kernel, delta-rule update, routing and FP32 storage are
untouched. Read-probe input bytes are unchanged in all eight cases. The separate
store_probe.json checks native state and route bytes at identical Cell inputs;
its gate status is **passed**. This does not imply whole-model state
equality after changed reads feed subsequent layers or optimizer updates.

Both arms processed **5,013,504 tokens** in 153 updates, with identical token/target SHA256. Maximum absolute loss difference: **2.8014183e-05**; mean: **6.300172e-06**; final FP32 / candidate losses: **3.89899993 / 3.8989923** (difference 7.62939453e-06).

The six-block pilot uses TinyStories, B64/T512, FP32 dense training in both arms,
the same initialization/optimizer/data order, and the read variant only in the
candidate. Raw per-update losses and gradient norms are retained. **The candidate
retains the FP32 production replay as a surrogate adjoint.** The slice measures
this precise experimental training recipe, not a correctly recertified BF16
adjoint or 1.3B model-quality equivalence. It cannot authorize BF16 adoption.

| T | Head | FP32 native us | BF16 MMA us | Native / candidate | Max output difference |
|---|---:|---:|---:|---:|---:|
| 512 | 0 | 91.29 | 86.95 | 1.0499 | 0.0047784 |
| 512 | 1 | 95.43 | 86.18 | 1.1073 | 0.0052253 |
| 512 | 2 | 90.99 | 87.23 | 1.0430 | 0.0066975 |
| 512 | 3 | 111.78 | 109.86 | 1.0175 | 0.0057016 |
| 2048 | 0 | 119.93 | 93.28 | 1.2857 | 0.0053104 |
| 2048 | 1 | 131.25 | 123.51 | 1.0627 | 0.0071008 |
| 2048 | 2 | 148.03 | 184.66 | 0.8016 | 0.0059845 |
| 2048 | 3 | 197.54 | 247.60 | 0.7978 | 0.0056249 |


Measurements are B1, three component warmups and 20 calls on actual recorded
views. Native head widths (64 key, 128 value) and budgets are identical at 22M
and 1.3B; those scales repeat the same four heads in 6 and 24 blocks. Thus these
are their shared isolated read shapes, not independent full-model scale runs or
an extrapolated B32 speedup. The deep T2048 lanes slow down in this B1 candidate.
The separate batch_read_probe.json repeats actual physical B32 geometry shared
by 22M and 1.3B at the fixed measurement envelope, without a full-model claim:

| T | Head | B32 native ms | B32 BF16 MMA ms | Native / candidate |
|---|---:|---:|---:|---:|
| 512 | 0 | 0.2995 | 0.3137 | 0.9545 |
| 512 | 1 | 0.5711 | 0.6303 | 0.9061 |
| 512 | 2 | 0.6159 | 0.8338 | 0.7387 |
| 512 | 3 | 0.7792 | 0.9505 | 0.8197 |
| 2048 | 0 | 1.2050 | 1.4614 | 0.8245 |
| 2048 | 1 | 3.8695 | 5.0039 | 0.7733 |
| 2048 | 2 | 9.0436 | 10.7811 | 0.8388 |
| 2048 | 3 | 11.8933 | 13.3808 | 0.8888 |

The supplemental rate_probe.json holds the tile/dot algorithm fixed and compares
IEEE FP32 dot with BF16 MMA; it still does not attribute full-step time.

| T | Head | IEEE-dot / BF16-dot time | Native / BF16-dot time |
|---|---:|---:|---:|
| 512 | 0 | 1.0141 | 1.0466 |
| 512 | 1 | 1.0523 | 1.0739 |
| 512 | 2 | 0.9951 | 0.9845 |
| 512 | 3 | 0.9899 | 0.9761 |
| 2048 | 0 | 1.0857 | 1.0358 |
| 2048 | 1 | 2.7154 | 0.9384 |
| 2048 | 2 | 2.9682 | 0.8075 |
| 2048 | 3 | 2.5822 | 0.7838 |


The fixed-algorithm factor quantifies this dot implementation's datatype/backend
trade, including BF16 rounding. Dividing the native/BF16 factor by that factor
gives native/IEEE-dot time, which exposes the algorithm change. Neither factor
can be applied to the fused reverse or multiplied by a published peak GEMM rate.

## Certification re-scope required before any operator adoption

Decision: **HOLD**, retain the existing FP32 production definition. The operator
has not selected BF16. Before selection, explicitly freeze the casts, FP32 MMA
accumulation order, tile sizes, softmax/tile reductions, output rounding, and
surrogate versus quantized-operand adjoint definition. If adopted, the production
kernel would be redefined as BF16-read/FP32-store; the CPU oracle must implement
that exact definition and require bitwise parity for outputs and all declared
adjoints, while preserving the FP32 store contract and routing/state checks.

The experimental CPU reference models the new operand casts and tile semantics,
but is **not a bitwise MMA/reduction oracle**: all eight prefix comparisons fail
bitwise equality (maximum absolute error approximately 1e-7 to 2e-7). The existing
FP32 certification cannot certify this candidate. No tolerance was substituted
for the requested bitwise contract. Defining and certifying its adjoints is a
remaining blocker, independent of whether the training loss divergence is small.

## Hybrid read caching

Read outputs alone do not determine the probability/query/key/strength adjoints.
Retaining only two lanes' output vectors across checkpoint cannot remove reverse
score/softmax recomputation. A probability cache is a useful candidate; undo and
value/strength/read adjoints remain. Initial-pass retention must be attached to
checkpoint lifetime, with exact recompute inputs and invalidation, not global
state keyed only by shape.

| T | Head 2+3 probability cache GiB | Output-only GiB |
|---|---:|---:|
| 512 | 1.5000 | 0.3750 |
| 2048 | 12.9375 | 1.5000 |


Storage accounting is for the 24-block 1.3B stack at physical B32, one microbatch
live at a time; head budgets are **736 / 1472**, not doubled. T2048 adds 12.9375
GiB of FP32 probabilities before writer scratch/metadata. The unchanged pure
profile already peaks at 41.100 GiB on a 48-GiB-class card.
A conservative sum of the two exceeds capacity; actual lifetime overlap needs
measurement. This is accounting, not an observed
OOM or a measured new peak. T512 adds 1.5 GiB and merits a scoped prototype.

| T | Head | Recompute us | Cached us | Saved us | Probability MiB (B1) |
|---|---:|---:|---:|---:|---:|
| 512 | 0 | 72.77 | 57.92 | 14.86 | 0.125 |
| 512 | 1 | 91.36 | 70.43 | 20.93 | 0.500 |
| 512 | 2 | 96.20 | 75.36 | 20.84 | 1.000 |
| 512 | 3 | 95.86 | 75.81 | 20.06 | 1.000 |
| 2048 | 0 | 244.63 | 190.98 | 53.64 | 0.500 |
| 2048 | 1 | 895.36 | 664.23 | 231.12 | 2.000 |
| 2048 | 2 | 2098.75 | 1583.36 | 515.39 | 5.750 |
| 2048 | 3 | 1929.72 | 1858.64 | 71.07 | 11.500 |


The small-scale experiment caches probabilities on B1 recorded views and checks
bytewise equality of the isolated FP32 query adjoint at full T512/T2048. It
parallelizes tokens and excludes store undo and value/strength adjoints, unlike
the production fused reverse. These component savings cannot be multiplied by
batch/layer counts and called full-step savings. Head 2 at T2048 saves more than
head 3 in this measurement despite head 3's larger storage. A subset should be
chosen from measured saving per byte, not archival labeling alone. Full-step
B32 hybrid cache memory/time is **not measured**; no caching landing is accepted.

## Fresh unchanged production measurements

| T | Pure tok/s | BF16 transformer tok/s | Pure step / control step |
|---|---:|---:|---:|
| 512 | 2761.32 | 6110.27 | 2.2128 |
| 2048 | 1808.83 | 5801.37 | 3.2073 |


All successful sustained arms require three warmups and 20 synchronized full
forward/backward/clip/AdamW updates. These are the fixed matched envelope above;
part8's optimized per-arm envelopes remain historical results and are not mixed
into new ratios. Physical GPU IDs and the hardware inventory are retained.
There is no after-optimization arm because no production optimization landed.

Fresh unchanged B128 bare/fused-pool pilot: **55,429.54 tok/s**, 3 warmups and 20 full updates. 100k gate: **FAILED**. Qualification inherits the unchanged part8 B128 matched curve; this is not a new quality run.

## Validation and residual work

Three independent CPU reference checks pass (live strengths, all-dead reads,
and a live tile after an empty tile). GPU component/candidate failures, if any,
remain in commands.tsv and raw JSON/logs. The unchanged production sources bind
to part8's source-bound T512/T2048 gates by SHA256, with part9's fresh
pressure-inclusive gates also required; no new production
change is being certified. All experiments use the approved Python and leased
cards 1/5/6. Protected PID 634691 and cards 0/2/3/4/7 are not modified. No .pt
files, WG, subagents, Frontier, Slurm or resilient DiLoCo changes; its requirement
IDs do not apply. AGENTS.md and CLAUDE.md remain unchanged.

Residuals: fused reverse read-math/store-adjoint wall split; BF16 CPU/native
bitwise oracle and adjoint certification; full-scale hybrid-cache memory/time;
trained-density exact event suppression; and an accepted optimization reaching
1x / the pilot 100k gate. The task gate remains failed, not an accepted speedup.
Tool discovery found no contact_supervisor capability; no supervisor decision
was obtained. See coordination.md and gate.json for machine-readable limits.
