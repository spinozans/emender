# X43 — 150M mini-lattice preregistration

Registered 2026-11-09 before training or held-out scoring. Operator authorization:
“do the 150M point — the next natural point.” This seven-arm byte experiment is
the flagship's dress rehearsal at approximately 1/9 the parameters and the
paper's scale-trajectory receipt. Registration only here; no training launches.
No result is claimed. The 1.3B board retains its separate launch sign-off.

## Frozen provenance and stream

Follow [PREREGISTRATION_1P3B.md](PREREGISTRATION_1P3B.md), including its registered
flagship stream at commit **2e022242** (not the C14 stream):
[registration.json](receipts/flagship_stream/registration.json), SHA256
`c36f84cfd1a59530a7991a4be6c1f68b424a79c5b7de169eabfc90d3cd31cc31`.
The compressed documents manifest SHA256 is
`48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414`;
identity-stream SHA256 is
`886a50d5bee4a44a6d33364da6d3cec978d5d6097c498bad213d6f8b24d00b7b`.
All seven arms consume prefixes in the same source order, fully beyond byte
100,000,000,000; no wrap, reordering or cross-document packing. Training has
238,461 documents / 1,466,137,768 shifted targets. Calibration and final test
retain the flagship's document-disjoint splits (16,673,137 / 16,700,678 targets),
including content-hash duplicate exclusion across splits.

Use natural raw text bytes, vocabulary 256. Inputs, shifted targets, consultation
contexts and held-out sequences remain inside one document. Use the checked
[c14_documents.py](c14_documents.py) boundary discipline, hash-match source and
documents receipts, truncate tails and mask padding from loss. Reset all model
and consultation state per window; no state carries across windows/documents.
Primary geometry is flagship T2048, with the registered 1024-byte causal
consultation suffix cap. Other reset geometries cannot reuse this cache silently.

Features are the **same flagship extraction cache** used by the 1.3B arms under
`/home/erikg/flagship_consultation_features`, against immutable pile-head100.
FLAGSHIPSPARSE2 decodes the full 260 scalars: 256 byte probabilities, log1p
outgoing count, entropy, fork indicator and log1p context length. This lattice
consumes its prefix; **no new extraction**. Before launch bind the completion
receipt, exact shard hashes, reset/feature identities and full decode parity.
Require union_complete and all registered counts, not merely service exit.

## Seven arms and controls

| Arm | 150M configuration | Mechanism |
|---|---|---|
| 1 | Byte transformer, byte_velocity_transformer_v1 searched nomination | Dense byte baseline |
| 2 | Byte GDN2, byte_velocity_gdn2_v1 searched nomination | Dense recurrent baseline |
| 3 | Byte pure Emender, byte_velocity_pure_v1 searched nomination | Bounded native store |
| 4 | Same pure configuration as 3, cap released | Unbounded pure store |
| 5 | Arm 1 transformer + XSA | Passive 260-scalar consultation channel |
| 6 | Arm 4 unbounded Emender + XSA | Store door over array-derived memories |
| 7 | Arm 6's append-only twin | APPEND only; CORRECT, POP, DECAY, MERGE disabled |

Arms 3 versus 4 ARE the bounded/unbounded matched pair; no independent unbounded
search. Arms 6 versus 7 differ only in the operation intervention, including its
application to array-derived memories. Keep initialization, optimizer, input
features, projection, dropout, geometry, precision and physical allocation paired;
receipt the operation masks so no disabled op fires, including hidden capacity
merges or retirement. Arm 6 enables all five controller ops: APPEND, CORRECT,
POP, DECAY, MERGE. Passive consultation is not a substitute for the store door.
Follow [ARCHITECTURE.md](ARCHITECTURE.md) and the C14-B mechanism receipts.

Use seed **3401** throughout and freeze arm order 1–7 before execution. Preserve
searched configs verbatim at their nominated 150M class except the registered
feature channel/cap/op interventions. Bind all three committed nomination hashes,
exact configs, code, optimizer, precision, batch/accumulation and parameter counts
before launch. Retain FP32 certified stores and nomination baseline precision.
Match total capacity within 0.1%, disclose trainable/frozen/total counts and
projection overhead; preserve paired zero-input projections. Qualify memory,
finite gradients, native/replay parity and op masks before training. Record
physical allocation, reachable writes, occupancy and cap refusals; released caps
must be proven nonbinding, never treated as infinite physical memory.

## Governing budget and comparisons

**2700 ACTIVE SECONDS per arm** governs, following line 47 of
[PREREGISTRATION_C14_BYTE.md](PREREGISTRATION_C14_BYTE.md). Frozen prefixes are
consumption records, not alternative stopping rules. Synchronize timing and
include data assembly/transfer, forward/backward/replay, clipping and optimizer;
record complete-update overshoot. Equal active wall-clock is primary. Report
elapsed wall time separately. Seven arms allocate 18,900 active seconds (5.25
card-hours), excluding separately receipted qualification and evaluation.

Actual raw bytes, real shifted targets, windows, steps, byte rates, active and
elapsed seconds are first-class deconfound columns. Equal time need not mean
equal exposure. Retain prospective common-prefix checkpoints at complete-update
boundaries shared by paired arms, score the greatest exactly shared real-target
prefix available within both budgets, and report its exposure/time explicitly.
If none exists, mark matched exposure unavailable; do not interpolate or extend
training to manufacture it. Stream/cache exhaustion before the time budget is a
failure, requiring prospective amendment rather than wrapping or early success.
Report matched compute secondarily: measured FLOPs where available, otherwise
explicit card-time with matched-FLOP loss unavailable. Card-hours are not FLOPs.

## Registered claims and honest-null allowance

All deltas below are held-out **nats/byte**, summed NLL / real byte targets.
Negative deltas favor the first named arm. Nulls, reversals and failures are valid
outcomes; historical magnitudes are priors, not thresholds or promised gains.

- **C-A — causal value of controller ops:** arm 6 minus arm 7 at matched
  exposure is predicted negative. The 2.13M C14 lifecycle receipt was about
  0.66 nats/byte (0.661320101 at fixed time); that historical fixed-time result
  is not already matched-exposure causal evidence at 150M.
- **C-B — unbounded efficiency:** arm 4 minus arm 3 at matched exposure is
  predicted negative. The 2.13M precedents are −0.019429161 and −0.018504590
  across two seeds, plus about 2.3× endpoint-normalized data efficiency. That
  efficiency statistic is not a measured learning slope or matched-loss sample
  complexity; no universal fixed-time superiority is predicted.
- **C-C — spine marginal value versus capacity:** report (arm 5 − arm 1) and
  (arm 6 − arm 4), at equal active time and available common exposure, against
  the 2.13M C14 deltas. The trajectory hypothesis is declining marginal array
  value as in-weights memorization improves with scale; stable or increasing
  value refutes that direction. Preserve consultation-alone versus calibrated
  soft-fusion columns separately rather than mixing endpoints.
- **C-D — pilot role:** measure all six flagship mechanisms at 150M, plus the
  seventh ops-causality control, before the 1.3B commitment. Completion is a
  readiness/trajectory receipt, not evidence that any mechanism must win or
  automatic flagship launch authorization.

Use the [C14 byte registration](PREREGISTRATION_C14_BYTE.md),
[C14-B registration](PREREGISTRATION_C14_B.md),
[matched-exposure registration](PREREGISTRATION_C14_B_MATCHED.md),
[RESULT_C14_FIRST_PASS.md](RESULT_C14_FIRST_PASS.md),
[RESULT_C14_B_MATCHED.md](RESULT_C14_B_MATCHED.md) and
[RESULT_C14_B_SEED3402.md](RESULT_C14_B_SEED3402.md) as the 2.13M scale point.
The first-pass passive transformer delta is −0.138365000 nats/byte
(1.266063829 − 1.404428829). Preserve the appropriate Emender mechanism/control
and exposure labels from C14 rather than relabeling passive or append-only
controls as the pure unbounded arm.

## Evaluation, caveats and firing gates

Freeze weights and fit any C14-style soft-fusion gates only on calibration;
freeze gates before final test. Report consultation-alone as primary mechanism
rows, calibrated fusion separately, and on/off interventions at fixed weights.
Use keep=.5 all-input consultation dropout without rescaling, including metadata,
with all-zero device-off inputs and matched control projections. Include every
real target, retaining zero-support and impossible targets (including infinities).
Report support/zero-target/empty/fork strata, reset/interior NLL, NLL sums and
byte denominators, paired deltas, op traces/ownership/read weights, peak state
and memory, precision, extraction/cache costs and total card-hours. Extraction
is prepaid in training-time columns and charged in end-to-end columns.

Four explicit limitations govern interpretation:
1. **Hyperparameter transfer:** nominations were searched on byte_velocity
   registration spans and are applied to the flagship stream. This is the same
   transfer assumption stated by the flagship registration, not fresh tuning.
2. **Stream caveat:** the 2.13M C14 lattice used the C14 stream (and its own
   geometry/index). Cross-scale directions are descriptive with those caveats;
   within this lattice paired comparisons use the exact same stream/features.
3. **Single seed:** seed 3401 is a pilot, not a significance test. The two-seed
   C14 precedent is the follow-up pattern if a sign matters; any repeat requires
   a prospective registered budget and cannot replace an unfavorable seed.
4. **Pilot budget:** 2700 seconds makes the 2.13M comparison interpretable;
   it does NOT preempt the 1.3B board's own budget or imply compute equivalence.

The lattice fires only when BOTH all three committed byte nominations land AND
flagship extraction completes and verifies. Run on free cards only, with the
same qualified hardware envelope across arms, preserving card 1 and all live
search/extraction services. Wait on contention; no preemption. Complete the
lattice before the 1.3B launch with **zero contention**. Its full receipts,
including failures and missing columns, accompany the operator's flagship
sign-off. No training is launched by this registration task, and no changes to
live services, ~/sxgc or /mnt/nvme2n1 are authorized here. Future launches use
the standing detached broker-wrapper survival and checkpoint discipline.

Hash-bind this document in PROGRAM's C14 receipt registry after committing its
bytes; C-A through C-D are registered questions, not new measured outcomes.

---

**Dated annotation (2026-11-09):** consulted arms (5, 6) of this lattice switch
feature delivery from the pre-extracted cache to **live emender-xsa
consultation** per the same pivot recorded in PREREGISTRATION_1P3B.md and
receipts/flagship_stream/REMOVAL_RECEIPT.md. Dense arms (1–4) and the
append-only control (7) are unaffected.
