# X40/X43 scalability phase 2: register-pressure work

The scoped C14 kernel battery is green. **The overall scalability objective
remains incomplete:** no >=8 high-live full-model training rate, approximately
linear live-slot scaling, or 1.3B tractability is certified. No training-scale
promotion is authorized. See `receipts/scalability_phase2/performance_gate.json`.

The selected opt-in variant combines global value staging between tokens,
exact allocated key-row writes, and explicit Gluon reduction layouts matching
the reference TTGIR. The fresh standard trained-batch diagnostic improves from
**4.45781 to 7.98754 compute steps/s (1.792x)**. This includes disposable
optimizer updates restored outside timing; it excludes source, decode,
assembly and checkpointing. It is neither sustained training nor a measured
110-live full-model rate. This batch peaks at 21/53/14/44 by head.

## Baseline and decision

Before mutation, retained controlled baseline timing, compiler metadata and PTX
measure BS256 at live peaks48/110/128, and BS512 at257. BS256 cannot represent
257 live physical slots. The trained BS256 specialization has2246 spills;
the synthetic specialization has2276. BS512 synthetic has4276. These are
compiler spill counts, not dynamically measured spill transactions.

Synchronized real-batch component timings show77.10ms native forward and
73.55ms native checkpoint recomputation,41.71ms replay/autograd and16.07ms
telemetry. They are diagnostic component timings, not additive training-rate
measurements. The controlled128-live fused kernel accounts for99.85% of CUDA
operator time. Native READ/APPEND/CORRECT/DECAY remain fused; no isolated
internal-phase timing is claimed.

The grid has16 CTAs on142 SMs: an analytic global warp-occupancy upper bound
of1.878%, distinct from live-slot occupancy. A later search found Nsight
Compute at `/usr/local/cuda-12.8/bin/ncu` despite its absence from PATH. Its
card3 probe fails with **ERR_NVGPUCTRPERM**, exit1; the driver reports
`RmProfilingAdminOnly: 1`. Actual hardware achieved occupancy is unavailable.
The probe also conflicts with torch.profiler's CUPTI subscriber; none of its
profiler timings is used as certification evidence. No driver settings changed.

The first global-staging candidate reaches8.15117 diagnostic steps/s but fails
84 full-model fingerprint hash leaves. It is rejected. Zero-valued synthetic
fingerprints had passed; they cannot substitute for model/gradient identity.
Retained TTGIR shows changed inferred reduction layouts. Explicit Gluon layouts
restore the original reduction trees. The intermediate exact variant reaches
6.90254; exact row-key writes then reach7.98754. Designs and all failures are
retained. The final trained BS256 compiler metadata is255 registers/thread,
862 spills versus40/2246 baseline; BS512 synthetic still spills3204.
Small-grid underutilization and residual spilling remain.

## Occupancy versus cost

B16/cap257,20 synchronized wall samples per arm after5 warmups. Values are
native head invocation ms, not model steps/s. Inputs ramp from empty; decay1
and zero values deliberately control peak occupancy. First three rows useT128;
the257-live row usesT257 and must be treated separately.

| Peak live | T | Phase-1 bounded ms | Row-key ms |
|---:|---:|---:|---:|
|48|128|15.023|5.256|
|110|128|18.563|5.312|
|128|128|19.614|6.612|
|257|257|76.371|45.233|

The independent fixed-T257 curve removes the sequence-length confound:

| Peak live | Original BS512 ms | Row-key BS512 ms |
|---:|---:|---:|
|48|59.727|36.418|
|110|65.727|41.703|
|128|67.654|42.540|
|257|79.858|48.738|

All pairs are repeat-exact and reference-exact. These curves do not establish
approximately-linear live-slot scaling. See the standalone
`receipts/scalability_phase2/occupancy_cost.svg` and raw sample JSON.

## Fresh scoped certification

- **105 C14/operator tests pass in74.49s**, including standing graph/eager,
  checkpoint/eager exact model checks, document boundaries and masked padding.
- **109 X31 oracle tests pass in16.51s**; this is a fresh regression run.
- **8/8 exact** nonzero shape/state/read/raw-trace and repeat pairs at caps
  48/110/257, including T128, T257, T12 and T1.
- **4/4 exact** nonzero native/state/replay/gradient and repeat pairs at actual
  live peaks48/110/128/257, all at cap257. Gradient fixtures have T equal to
  the live peak; the fixed-T timing curve is separate.
- Full standard trained-batch fingerprint matches the reference and repeat:
  loss, reads, physical state, raw traces, replay outputs/gradients, all model
  gradients including None-versus-zero, and telemetry.
- **988 paired updates,2,000,483 real-document masked targets:** every loss,
  gradient, parameter, optimizer and telemetry fingerprint matches exactly.
  This uses the production verified document/cache assembly, BF16, mask, clip
  and AdamW path. Both arms test atomic checkpoint serialization/load of
  model, optimizer, cursor, targets, telemetry and all RNG states at step2;
  all subsequent updates remain exact. Disposable checkpoint files are deleted.

This is a fresh source-bound training-path gate and a disposable certification
slice, not production-runner promotion or historical-gate reuse. See
`training_runner_gate.json`, `validation_summary.json`, `curve_rowkeys/result.json`
and every update in `curve_rowkeys/steps.jsonl`.

## Registration, survival and exact resume

Original production sources, runners, registrations, trained checkpoints and
protected workloads are unchanged. Only explicit process-local installation
of `c14_phase2_staged.install_rowkeys()` selects the new kernel. Rejected
variants remain available solely as receipted diagnostics. The phase-2
registration binds the new sources and tested sources; no historical source
rebind is required. Three terminal blank lines were normalized in a source-
binding commit: the retained-source AST/EOF audit proves no code changes, and
the complete2M-target slice runs against the normalized committed sources.

The certification launcher used Popen(start_new_session=True), DEVNULL stdin,
retained stdout/stderr and an internal broker lease. Retained evidence verifies
PPID1, matching lease owner/starttime, heartbeat renewal and advancing updates.
It completed normally and released its lease; no training is left running.
The immutable original checkpoint is unchanged, SHA256
`056b876cec5f5aae0b9bcea24544e9e3efccc2791fe9bb91654ca192cfb3eb06`.
The disposable slice ends at next window15808; no resulting model is promoted.
A repeat must start from that original checkpoint with seed3401 and fresh
receipt paths; completed receipts refuse overwrite.

Before any further experiment run:
`/home/erikg/emender/.venv/bin/python scripts/registration_guard.py check experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase2/registration.json`.
Next work must address BS512/257-live spill pressure and the16-CTA grid, then
repeat the affected battery and establish an actual high-live full-model rate
and scale measurements. Do not infer1.3B safety or affordability from this C14
fixture. Current certification is RTX6000Ada/Triton3.5.1 only.

Manual execution notes are in `receipts/scalability_phase2/EXECUTION_NOTES.md`.
No resilient DiLoCo behavior changes; gap-matrix requirement IDs are inapplicable.
Raw logs, IR and source snapshots are retained losslessly as gzip with raw
SHA256 receipts. No .pt files are committed. Unrelated workspace changes remain.
