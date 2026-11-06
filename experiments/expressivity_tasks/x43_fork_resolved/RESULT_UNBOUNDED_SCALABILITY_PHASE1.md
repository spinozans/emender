# X40/X43 unbounded-store scalability phase 1

Reach-bounded launch implemented as an **opt-in wrapper** in
`c14_launch_shape.py`; existing production sources, historical registrations,
training runners and frozen arms remain unchanged. The standard 16x128 trained
batch is bit-identical and improves fixed-weight compute from **2.74447 to
4.49928 steps/s (1.639x)**. This includes disposable optimizer updates restored
outside timing, excludes source/decode/assembly/checkpointing, and is **not a
sustained training rate or a measured 110-live full-model training rate**.
The first real batch still peaks at 21/53/14/44 by head. Historical training
2.644903 steps/s is retained, not overwritten by a diagnostic.

## Decision and implementation

Design commits **17fe3a88** and **7fd5ef0c** precede their respective experiments.
The first design enumerates launch variants, dynamic shape, tiling/shared
staging/kernel splitting, occupancy selection and already-batched scans, with
hypotheses and certification scope. Warp16/32 improved compute to
4.18228/4.24765 steps/s but **failed byte-exact fingerprints**, with 80 differing
leaves including loss and 45 model gradient tensors; telemetry matched. They
are rejected, despite the 99-test tolerance/checkpoint battery passing warp16.

The chosen wrapper retains S=257, all allocation/policy/ownership laws, the
original kernel body, original 8 warps and original replay. Every invocation
starts empty, and each token can allocate at most one array and one stream
slot. Therefore no physical slot >=2*T can be reached. Compute tile BS becomes
next_power_of_2(min(S, max(1, 2*T))); standard T128 uses **BS256 instead of
BS512**. S remains 257 in occupancy arithmetic; this is not a cap reduction.

Two wrapper restorations are necessary for full physical byte identity:
initialize unreachable physical tail K/V/strength to zero and bookkeeping to
its exact empty bits; restore the original padded-lane empty-read POP counter
from pre-read cardinality and the original POP input, masked by valid length.
No floating-point model arithmetic is added. Merely shrinking BS without these
restorations would fail the raw state/trace contract. The source wrapper is
cloned with its existing encoding/cache logic into a private launch namespace;
installation is explicit and process-local. Existing running trainings receive
no mutation. Do not install twice or use this diagnostic entrypoint to bypass
any historical runner's gates.

Compiled standard-shape metadata: **n_regs40, n_spills2246, shared16384 bytes**,
BS256, num_warps8, compared with original BS512/n_spills4078. This is reduced
spilling, not spill elimination. `shape_final/composition.json` contains the
actual observed launch metadata. Its older `physical_gpu_slots_power2`
intervention field is calculated from capacity and still says 512; it is not
an observation of the new launch. The preliminary shape run had no launch
metadata observation because its private proxy captured the kernel before the
observer. `shape_source_v1.py` retains that exact source; the final proxy uses
the observer and has a fresh complete fingerprint and timing run.

## Controlled occupancy and capacity curves

`shape_curve/result.json`: physical card3, batch16, 5 warmups and 20 synchronized
wall-time samples per arm, same inputs within each pair, original 8 warps.
Values are native milliseconds per head invocation, **not model steps/s**.
Decay=1 and zero values force a controlled append prefix followed by writes
disabled at the target peak. Both are explicit synthetic input/configuration
interventions, not trained loss or a claim of stationary occupancy. Peak
cardinalities and zero capacity merges are asserted. Live occupancy ramps from
empty; T257 is shown separately rather than fitted into the T128 curve.

| Cap | Peak live | T | Original ms | Bounded ms | Speedup |
|---:|---:|---:|---:|---:|---:|
|48|48|128|2.247|2.232|1.006x|
|110|110|128|4.414|4.124|1.070x|
|257|48|128|30.425|15.033|2.024x|
|257|110|128|36.409|18.460|1.972x|
|257|128|128|38.218|19.641|1.946x|
|257|257|257|76.453|76.218|1.003x|

The unchanged-cap 48/110/128 points distinguish live occupancy from allocated
capacity, while 48/110/257 allocated-capacity points expose padding costs.
The 257-live case correctly falls back to BS512 and shows no meaningful win.
The extra fixed-T257 curve in `fixed_curve/result.json` independently retains
three controlled live peaks at cap257, and rejected warp candidates. No power
law or approximately linear high-occupancy implementation is certified.
The first variable-T sweep was deliberately interrupted, exit130: natural
decay prevented its intended 257-live target and varying T confounded costs.
Its partial results/source and replacement protocol are retained, not a pass.

## Scoped certification

No kernel body or shared X31/X40 production source changed: the 23-file
`production_source_audit.json` matches working bytes to committed source.
This is a launch-configuration plus integer/tail-restoration wrapper change;
the applicable scope is the C14/configuration parity battery. The standing
109 X31 oracle, registered cost/append/storage/structural/checkpoint gates,
shared graph/eager recertification and bitwise >=2M-token curve remain mandatory
for subsequent kernel-internal changes; none is falsely claimed rerun here.

- Final complete C14 plus operator-round battery: **105 passed in 179.99s**.
  Includes the unchanged graph/replay-vs-native checks, checkpoint-vs-eager
  exact model checks and document boundary tests. New tests cover empty-read
  raw POP traces with masked positions in both control modes, exact physical
  tails, and two allocations/token reaching the 256-slot bound.
- Strict nonzero-read/state/trace repeat and reference fingerprints: **8/8**,
  including caps48/110/257, live peaks48/110/128/257, and short T12/T1.
- Fresh standard trained-batch fingerprints: exact loss, reads, full physical
  K/V/strength/bookkeeping, raw traces, replay outputs/gradients, model
  gradients including None-vs-zero, and telemetry. Repeat and reference pass.
- Controlled high-occupancy native/state/replay/gradient fingerprints: **4/4
  exact and repeat-exact**, covering 48/110/128/257 live peaks with nonzero
  values and fixed random upstream gradients.

The earlier 103-test shape run collected before the two maximum-allocation
fixtures were added; its loaded test source is retained. The complete final
105-test run removes that source-provenance ambiguity. No failed run is counted
as a passing gate. The new variant receives its own source-bound registration
in this implementation commit. Historical registrations need no rebind because
their source bytes are unchanged; matched-twin registration is untouched.

## Exact resume and remaining phases

Use `/home/erikg/emender/.venv/bin/python scripts/registration_guard.py check
experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase1/registration.json`
before any further experiment. The
wrapper is an opt-in implementation and diagnostic entrypoint, not a newly
registered training runner. A fresh runner/gate set is still required for new
training; do not quietly reuse old hash-bound control or matched-twin gates.
The matched twin on card2 was not inspected or operated on.

Original unbudgeted resume remains untouched:
`/home/erikg/c14_dense_bytes/emender_c14_b1_v2_unbudgeted_resume.pt`,
SHA256 `056b876cec5f5aae0b9bcea24544e9e3efccc2791fe9bb91654ca192cfb3eb06`,
7142 steps, 14,455,380 targets, next window114272, active seconds
2700.2882562745363. Each real profiler independently verified that immutable
checkpoint receipt before loading. No optimizer/RNG checkpoint was saved.

Next: preserve the original reduction ordering while tiling or staging the
persistent K/V vectors to remove the remaining spills. Keep raw POP/state
semantics and physical slot identity. Run the full kernel-internal standing
battery and rebind affected registrations in the same commit before promotion.
At T>=129/cap257 the conservative shape bound may provide no reduction; the
257-live fallback demonstrates the remaining problem directly. >=8 steps/s,
150M affordability and 1.3B tractability are **not achieved or measured**.
The 9.045x data-efficiency endpoint statistic motivates the work but remains
single-seed descriptive improvement per exposure, not matched-loss efficiency.

Manual notes: all GPU work attended and sequential on broker-leased card3;
only project .venv Python. No WG, union-service operation, protected card1/2/7
workload, sxgc, /mnt/nvme2n1 writes, training, or .pt commits. No resilient
DiLoCo behavior changed; gap-matrix IDs are inapplicable. Unrelated workspace
changes remain. Supervisor/intercom tools were searched first by name and
then description and are absent; no supervisor contact/approval is claimed.

Scoped diff checks pass. A whole-worktree diff check found pre-existing EOF
whitespace in unrelated stage14/stage16 growth reports; those files were left
alone. This is not a failing implementation test.

Raw text logs and profiler tables are retained losslessly as `.txt.gz`; JSON
receipts and fingerprints remain directly readable. Original text hashes are
retained separately in the manifest. Staged whitespace checks exclude no code.
