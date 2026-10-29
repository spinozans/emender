"""Render the decision receipt from measured artifacts, preserving failed gates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def main():
    gate=json.loads((OUT/'gate.json').read_text());read=json.loads((OUT/'read_probe.json').read_text());cache=json.loads((OUT/'cache_probe.json').read_text())
    phases=['sequential_forward_core','forward_read_elementwise_tiles','initial_checkpoint_dense_and_native_setup','checkpoint_recompute_dense_and_native_setup','forward_dense_loss_and_native_setup','backward_dense_loss_and_replay_setup','reverse_fused_read_math_and_adjoints','reverse_setup_and_teardown','clip','optimizer']
    table='| Exclusive wall seconds | T512 | T2048 |\n|---|---:|---:|\n'
    profiles=[json.loads((OUT/f'profile_pure_T{t}.json').read_text()) for t in (512,2048)]
    for phase in phases:table+=f'| {phase} | {profiles[0]["exclusive_seconds"][phase]:.6f} | {profiles[1]["exclusive_seconds"][phase]:.6f} |\n'
    table+=f'| Full diagnostic update | {profiles[0]["wall_seconds"]:.6f} | {profiles[1]["wall_seconds"]:.6f} |\n'
    rates='| T | Pure tok/s | BF16 transformer tok/s | Pure step / control step |\n|---|---:|---:|---:|\n'
    for row in gate['sustained_ratios']:rates+=f'| {row["context"]} | {row["pure_tok_s"]:.2f} | {row["transformer_tok_s"]:.2f} | {row["pure_time_over_transformer"]:.4f} |\n'
    curve=gate.get('curve_comparison')
    quality=(f'Both arms processed **{curve["tokens"]:,} tokens** in 153 updates, with identical token/target SHA256. Maximum absolute loss difference: **{curve["max_abs_loss_difference"]:.9g}**; mean: **{curve["mean_abs_loss_difference"]:.9g}**; final FP32 / candidate losses: **{curve["final_fp32_loss"]:.9g} / {curve["final_bf16_loss"]:.9g}** (difference {curve["final_absolute_loss_difference"]:.9g}).' if curve else f'Paired curve did not complete: {gate["curve_status"]}.')
    readtable='| T | Head | FP32 native us | BF16 MMA us | Native / candidate | Max output difference |\n|---|---:|---:|---:|---:|---:|\n'
    for r in read.get('rows',[]):readtable+=f'| {r["context"]} | {r["head"]} | {r["fp32_seconds"]*1e6:.2f} | {r["bf16_seconds"]*1e6:.2f} | {r["speedup"]:.4f} | {r["forward_max_abs_difference"]:.7f} |\n'
    batchpath=OUT/'batch_read_probe.json'
    batchread=json.loads(batchpath.read_text()) if batchpath.exists() else dict(rows=[])
    batchtable='| T | Head | B32 native ms | B32 BF16 MMA ms | Native / candidate |\n|---|---:|---:|---:|---:|\n'
    for r in batchread.get('rows',[]):batchtable+=f'| {r["context"]} | {r["head"]} | {r["fp32_seconds"]*1e3:.4f} | {r["bf16_seconds"]*1e3:.4f} | {r["speedup"]:.4f} |\n'
    ratepath=OUT/'rate_probe.json'
    rate=json.loads(ratepath.read_text()) if ratepath.exists() else dict(rows=[])
    ratetable='| T | Head | IEEE-dot / BF16-dot time | Native / BF16-dot time |\n|---|---:|---:|---:|\n'
    for r in rate.get('rows',[]):ratetable+=f'| {r["context"]} | {r["head"]} | {r["fixed_algorithm_speedup"]:.4f} | {r["production_to_bf16_speedup"]:.4f} |\n'
    cachetable='| T | Head | Recompute us | Cached us | Saved us | Probability MiB (B1) |\n|---|---:|---:|---:|---:|---:|\n'
    for r in cache.get('rows',[]):cachetable+=f'| {r["context"]} | {r["head"]} | {r["recompute_seconds"]*1e6:.2f} | {r["cached_seconds"]*1e6:.2f} | {r["saved_seconds"]*1e6:.2f} | {r["probability_cache_bytes"]/2**20:.3f} |\n'
    memory='| T | Head 2+3 probability cache GiB | Output-only GiB |\n|---|---:|---:|\n'
    for r in gate['cache_analysis']:memory+=f'| {r["context"]} | {r["probability_cache_GiB"]:.4f} | {r["output_only_cache_GiB"]:.4f} |\n'
    cycleparagraphs=[]
    cycletable='| T | Undo/input % | Strength/store adjoints % | Score/softmax recompute % | Read adjoints % |\n|---|---:|---:|---:|---:|\n'
    for context in (512,2048):
        p=OUT/f'reverse_clock/cycles_T{context}.json'
        if p.exists():
            r=json.loads(p.read_text())
            if r['status']=='passed':
                f=r['cycle_fractions'];cycleparagraphs.append(f'T{context}: score/softmax recomputation is {f["read_score_softmax_recompute"]*100:.2f}% of instrumented reverse CTA cycles (cycle-weighted device estimate {r["cycle_weighted_device_ms_estimates"]["read_score_softmax_recompute"]:.2f} ms); read adjoints are {f["read_adjoints"]*100:.2f}%.');cycletable+=f'| {context} | {f["undo_and_input_init"]*100:.2f} | {f["strength_restore_and_store_adjoints"]*100:.2f} | {f["read_score_softmax_recompute"]*100:.2f} | {f["read_adjoints"]*100:.2f} |\n'
    pilot=gate['pilot'];pilottext=(f'Fresh unchanged B128 bare/fused-pool pilot: **{pilot["tokens_per_second"]:,.2f} tok/s**, 3 warmups and 20 full updates. 100k gate: **{pilot["gate"].upper()}**. Qualification inherits the unchanged part8 B128 matched curve; this is not a new quality run.' if 'tokens_per_second' in pilot else f'Fresh pilot measurement: {pilot["measurement_status"]}.')
    memo=f'''# Part 10 decision — hold BF16 adoption; 1x is not established

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

{table}

These are exclusive synchronized wall regions from one post-warmup diagnostic
update. Child times are subtracted, and residual wall is under 2 ms. Core and
batched read rows combine initial and checkpoint-recompute passes; the two
dense/checkpoint rows exclude those child regions. The transformer diagnostic
wall is {gate['profiles'][0].get('transformer_seconds',float('nan')):.6f} / {gate['profiles'][1].get('transformer_seconds',float('nan')):.6f} s
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

{cycletable}

{" ".join(cycleparagraphs)}

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
its gate status is **{gate.get('store_probe','not recorded')}**. This does not imply whole-model state
equality after changed reads feed subsequent layers or optimizer updates.

{quality}

The six-block pilot uses TinyStories, B64/T512, FP32 dense training in both arms,
the same initialization/optimizer/data order, and the read variant only in the
candidate. Raw per-update losses and gradient norms are retained. **The candidate
retains the FP32 production replay as a surrogate adjoint.** The slice measures
this precise experimental training recipe, not a correctly recertified BF16
adjoint or 1.3B model-quality equivalence. It cannot authorize BF16 adoption.

{readtable}

Measurements are B1, three component warmups and 20 calls on actual recorded
views. Native head widths (64 key, 128 value) and budgets are identical at 22M
and 1.3B; those scales repeat the same four heads in 6 and 24 blocks. Thus these
are their shared isolated read shapes, not independent full-model scale runs or
an extrapolated B32 speedup. The deep T2048 lanes slow down in this B1 candidate.
The separate batch_read_probe.json repeats actual physical B32 geometry shared
by 22M and 1.3B at the fixed measurement envelope, without a full-model claim:

{batchtable}
The supplemental rate_probe.json holds the tile/dot algorithm fixed and compares
IEEE FP32 dot with BF16 MMA; it still does not attribute full-step time.

{ratetable}

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

{memory}

Storage accounting is for the 24-block 1.3B stack at physical B32, one microbatch
live at a time; head budgets are **736 / 1472**, not doubled. T2048 adds 12.9375
GiB of FP32 probabilities before writer scratch/metadata. The unchanged pure
profile already peaks at {profiles[1]['peak_allocated']/2**30:.3f} GiB on a 48-GiB-class card.
A conservative sum of the two exceeds capacity; actual lifetime overlap needs
measurement. This is accounting, not an observed
OOM or a measured new peak. T512 adds 1.5 GiB and merits a scoped prototype.

{cachetable}

The small-scale experiment caches probabilities on B1 recorded views and checks
bytewise equality of the isolated FP32 query adjoint at full T512/T2048. It
parallelizes tokens and excludes store undo and value/strength adjoints, unlike
the production fused reverse. These component savings cannot be multiplied by
batch/layer counts and called full-step savings. Head 2 at T2048 saves more than
head 3 in this measurement despite head 3's larger storage. A subset should be
chosen from measured saving per byte, not archival labeling alone. Full-step
B32 hybrid cache memory/time is **not measured**; no caching landing is accepted.

## Fresh unchanged production measurements

{rates}

All successful sustained arms require three warmups and 20 synchronized full
forward/backward/clip/AdamW updates. These are the fixed matched envelope above;
part8's optimized per-arm envelopes remain historical results and are not mixed
into new ratios. Physical GPU IDs and the hardware inventory are retained.
There is no after-optimization arm because no production optimization landed.

{pilottext}

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
'''
    (OUT/'decision.md').write_text(memo)
    report=f'''# X40 part 10 — corrected decomposition and unpromoted BF16 evidence

**Task acceptance: FAILED / incomplete. No production speedup or BF16 adoption
is accepted.** Detailed [decision receipt](receipts/part10/decision.md) and
[machine-readable gate](receipts/part10/gate.json) preserve the remaining gaps.

![Exclusive diagnostic phases](receipts/part10/diagnostic_phases.svg)

At the 1.3B shape, fixed B64/B32 checkpoint diagnostics measure 12.000 / 72.378 s
at T512/T2048. Fused reverse accounts for 4.927 / 36.914 s; sequential forward
core (initial plus recompute) 2.247 / 15.468 s; batched read 0.237 / 1.855 s;
reverse setup/teardown 0.262 / 0.322 s. Reverse read-math and sparse/event adjoints
remain fused; no invented wall split is reported.

{" ".join(cycleparagraphs)} These occupancy-weighted cycle diagnostics pass
fresh pressure-inclusive parity but are not additive wall phases. Read adjoints,
rather than repeated score/softmax alone, dominate the instrumented reverse.

The current native read uses
FP32 vector arithmetic, not GEMM. BF16 forward read alone cannot explain 1x.

{quality}

![Matched 5M-token loss slice](receipts/part10/loss_curves.svg)

The read-only BF16 MMA experiment leaves production store kernels untouched,
but retains FP32 surrogate replay and fails CPU/native bitwise read parity.
It stays isolated from production. B1 isolated throughput ranges from small
gains to slowdowns on deep T2048 lanes; shared head shapes are the same at 22M
and 1.3B. Read-only caching is insufficient for reverse; two deep-lane probability
caches cost 1.5 / 12.9375 GiB at T512/T2048 for B32 × 24 blocks. The query-adjoint
component cache passes bytewise checks; no full-step caching speedup is claimed.

{rates}

{pilottext}

Validation: three CPU tests pass; source hashes bind unchanged production to
part8 gates, with part9 pressure gates also required. Residuals: fused reverse phase split, BF16 adjoint/bitwise oracle,
full-step cache memory/time, exact trained-density suppression, 1x and 100k.
No .pt artifacts or production source changes; protected GPUs remain untouched.
'''
    (ROOT/'REPORT_PART10.md').write_text(report)

if __name__=='__main__':main()
