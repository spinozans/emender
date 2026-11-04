# Attended C14 paired-eval notes

Read PREREGISTRATION_C14_BYTE.md and both consultation training reports first.
The tool catalog was searched for contact_supervisor; it is absent. No supervisor
reply, approval, or decision is implied. Work remains single-threaded; no wg
service/dispatcher/worker or completion machinery was used. The GPU lease broker
is the explicitly requested standalone cards-2/4 broker.

The first cache audit failed closed because the extraction identity predates the
stopping-rule decision. The repaired audit verifies the original stream audit
at exact commit f29462c3abcdc23bc974efe2d3510cf55577d7ff and proves all frozen
stream fields/receipt hashes match the current amended audit. The complete
26,807-shard audit then passed. This preserves historical provenance rather than
rewriting the running service's identities.

The first test command without importlib collection failed on relative imports;
the corrected test imports the full package. The retained targeted suite passes.
An early evaluation launch was refused by the audit's exclusive lock. A first
9-shard exploratory run was stopped after checking the registration's agreement
definition: branch agreement is a positive target count, not argmax agreement.
Its initial_argmax_smoke.json is diagnostic only and never contributes to the
adopted table. The corrected table retains argmax agreement as supplemental.

An attempted on/off concatenated 256-row forward failed the finite-loss guard
before any accepted paired aggregate. Its failure log is retained. A separate
first-shard diagnostic at batch 16/128/256 was finite for all three checkpoints;
this does not erase or establish a root cause for the failed attempt. The primary
path uses separate forwards with at most 128 rows, fails closed on nonfinite
losses, and retains a specific nonfinite_failure.json if encountered again.

Unused Cell eval profiling does substantial per-row telemetry work. The primary
path suppresses only that telemetry branch by setting Cell.training=True inside
an otherwise eval-mode model under inference_mode. The Cell source's sole use of
this flag is telemetry. First-shard loss bits for all five modes must equal normal
eval-mode losses before any primary cumulative receipt is accepted. Model weights,
dropout policy and source windows are unchanged; no gradient or optimizer runs.

The source mount and all dense/consultation checkpoints are read-only inputs.
The union service receives no commands except read-only systemctl show; its journal
is snapshotted read-only. No .pt files belong to these changes. The transformer
projection receipt preserves both the requested ~45-hour scenario and the frozen
continuation's 35.24-hour projection at the historically receipted baseline rate;
these are scenarios, not a promised boundary.

Further first-shard repetition with all three models resident found occasional
finite loss-bit differences even across repeated normal eval calls without the
telemetry change (up to .0312 for v1). No root cause is certified. Enabling
PyTorch deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 produced
zero bit differences/nonfinite losses across eight alternating telemetry/normal
calls per checkpoint at batch 16 and separately batch 128. Both diagnostics are
retained. Primary inference requires these deterministic settings and still
fails closed on the normal-eval parity check and every nonfinite loss. This is
scoped diagnostic evidence, not an exhaustive native-kernel determinism proof.

The deterministic primary smoke completed 65 shards / 8,320 windows / 1,053,474
real targets in 5.383878 inference-loop seconds. Normal-eval/telemetry-suppressed
loss bits matched for all five modes at its first shard. The full run resumed
that exact aggregate under the same runner hash, batch shape, checkpoint hashes,
deterministic settings and broker card 2; the smoke contributes exactly once.
Final-test rows begin only when the frozen final-test documents are reached;
null interim final-test curves are expected during the calibration prefix.

The full deterministic run completed all 26,807 shards / 3,431,242 windows / 434,204,084 targets in each of five modes, with no nonfinite accepted losses. Independent frozen-window denominators, 75 partition-sum checks, journal coverage and checkpoint hashes pass. All 11 targeted tests pass. Both v1 and primary v2 PASS P1 and FAIL P2. Final-test device-off minus dense is +0.191308754 (v1) and +0.033791827 (v2), above .02. The final-test dense replay differs from its historical batch-16 average by -0.000001069 nats/byte. This is not byte-exact historical prediction parity.

The source repository contains unrelated pre-existing tracked/untracked changes; they are excluded from these commits. No checkpoint was modified or committed. Final service progress was captured read-only after evaluation.
