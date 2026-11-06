# X40/X43 scalability phase 2: register-pressure experiments

Certification in progress. No training-scale promotion is authorized.

The retained baseline re-measures BS256 rather than quoting BS512 spill counts:
trained-batch specialization 2246 spills, controlled synthetic specialization
2276. A 257-live physical state requires BS512 (4276 synthetic spills).
Native forward and checkpoint recomputation dominate the remaining baseline
compute time: 77.10ms and 73.55ms respectively, with replay/autograd41.71ms.
Native internal phases are fused; no separate READ/APPEND timing is claimed.
Analytic grid occupancy is at most1.878% (16 CTAs on142 SMs), distinct from
live-slot occupancy and from the per-SM resource ceiling. ncu is unavailable.

A direct global-staging experiment reaches8.15117 fixed-batch compute steps/s
but fails84 fingerprint hash leaves. It is rejected. Synthetic zero-value
fingerprints alone would have incorrectly accepted it. The retained TTGIR
shows changed compiler reduction layouts. Explicit Gluon reduction layouts
restore the original arithmetic ordering. Global staging plus direct allocated
key-row writes then reaches **7.98754 fixed-batch compute steps/s**, versus
fresh **4.45781** phase-1 baseline (1.792x). This includes disposable optimizer
updates restored outside timing. It excludes data/decode/assembly/checkpoints
and is neither sustained training nor a measured high-live full-model rate.
The first real batch still has historical head peaks21/53/14/44.

Compiled trained BS256:255 registers/thread,862 spills versus40/2246 baseline.
The design trades deliberate global accesses for reduced implicit spilling;
spill elimination and small-grid parallelism are not achieved. Original
production source bytes, historical runners, registrations and checkpoints are
unchanged. Installation is explicit and process-local. New variants are
isolated experiments, not implicit changes to an active training.

Native-only controlled ramp curve, cap257/B16; first three points useT128,
last usesT257 and must be considered separately:

| Peak live | Baseline ms | Row-key candidate ms | Candidate spills |
|---:|---:|---:|---:|
|48|15.023|5.256|862|
|110|18.563|5.312|862|
|128|19.614|6.612|862|
|257|76.371|45.233|3204|

These are per-head native invocations, not full-model steps. All curves ramp
from empty and deliberately set decay1 and zero values to control occupancy.
The fixed-T257 four-occupancy curve remains a separate required receipt.
No approximately-linear live-slot scaling or1.3B tractability is certified.

Completed intermediate explicit-layout variant:105 C14/operator tests,
109 X31 oracle tests, standard model fingerprint, and4/4 nonzero occupancy
state/replay/gradient fingerprint pairs, all green. Final row-key variant: **105 C14/operator tests in74.49s,109 X31 tests in16.51s,8/8 strict shape/state/read/trace pairs,4/4 nonzero occupancy state/replay/gradient pairs, standard model fingerprint and fixed-T257 curve all pass**. Neither variant has a passing2M-target curve
or fresh training-runner gate yet. No prose overrides a failed or missing gate.

Resume from source-bound `receipts/scalability_phase2/registration.json` after
committing its matching sources. The final validation command is
`bash experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase2/rowkeys_validate_v2_launcher.sh`;
its individual gates refuse to overwrite existing evidence, so resume only
unfinished commands into fresh paths. The disposable curve requires all final
gate terminals green and checks the source registration before running.
Original trained checkpoint SHA256 remains
056b876cec5f5aae0b9bcea24544e9e3efccc2791fe9bb91654ca192cfb3eb06.

Manual execution notes and failures are retained in
`receipts/scalability_phase2/EXECUTION_NOTES.md`. No resilient DiLoCo changes;
gap-matrix requirement IDs are inapplicable. No .pt artifacts are committed.

Fixed-T257 curve (native ms, reference/candidate):

- 48 live: 59.727/36.418; both repeat-exact and reference-exact.
- 110 live: 65.727/41.703; both repeat-exact and reference-exact.
- 128 live: 67.654/42.540; both repeat-exact and reference-exact.
- 257 live: 79.858/48.738; both repeat-exact and reference-exact.

The curve source was completed after the row-key kernel gates; its final
version is retained separately in `curve_source_final.py`. Kernel and all
other phase-2 module bytes still match `rowkeys_sources_v2`.
