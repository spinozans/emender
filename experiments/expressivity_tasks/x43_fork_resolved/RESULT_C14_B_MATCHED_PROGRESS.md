# C14-B matched-exposure twin: registered and training, verdict pending

Registration commit **8ad08f70** preceded training. All 23 original HARVEST
source hashes bind in `/home/erikg/emender_c14_matched_bound` at c190956d,
including the required 58a8acb8 replay. The optimized current source was not
substituted. No frozen arm was changed. New independent checkpoint, receipts,
explicit exposure stopping rule and all original training hyperparameters are
registered in PREREGISTRATION_C14_B_MATCHED.md.

Training launched through the shared broker on physical card 2. Launcher PID
4099459, trainer PID 4099514, tool session 45901; these are observations, not
permanent liveness guarantees. No worker/service orchestration was used.
The live checkpoint saves every 120 active seconds and on SIGTERM/SIGINT.
Do not launch a concurrent resume while this trainer remains alive.

An independently retained exact checkpoint snapshot at **303 steps / 612,322
real targets**, next window **4,848**, active seconds **120.02114207204431**:
`/home/erikg/c14_dense_bytes/emender_c14_b_matched_attested_step303.pt`
SHA256 `8a98b2fabb0c69e0dd6801b03427e441bec268924519b6a404a9d6dd24b99426`.
Model, optimizer, all three RNG states, stream cursor, exposure, timing,
telemetry, curve and dropout counters are retained. RESUME_STATE.json records
all bindings and the exact resume command. The live checkpoint may be newer.
The snapshot is external to git and immutable by the training runner.

Partial PB3 peaks: 51/51/51/51, all capacity merges zero. These early peaks and
sampled occupancies are not final-run occupancy or a scaling law. No frozen
quality score is assigned to this partial state.

Validation: full source/data/cache preflight passed. Independent traversal of
the raw document-hash-checked stream confirms 35,419 steps correspond to
71,677,042 nonpadding targets / 566,704 windows. Effective trainer source
parses and has the exposure stop and exact terminal assertion, with the
2700-second stop removed. The evaluator rejects partial training before GPU
model construction. Both leased wrappers pass bash syntax checks. PROGRAM
validator is green: 17 claims / 369 receipt bindings. Scoped whitespace checks
pass. Whole-tree whitespace checking found existing unrelated blank lines in
stage14/stage16 GROWTH_REPORT files; those files were preserved.

Remaining work: complete training, verify final PB3 nonbinding telemetry, then
run `bash experiments/expressivity_tasks/x43_fork_resolved/run_c14_b_matched_eval_leased.sh`
(add `--resume` only for an existing evaluation state). The evaluator accepts
only the completed exact-exposure checkpoint, and otherwise retains the frozen
window identity, strata, on/off modes and sums/count reductions. The evaluation
runner is prepared and syntax/partial-state-guard checked; full evaluation has
not run. Prior full evaluation cost ~52,111 seconds, beyond the ~3.8h training
projection. Commit all 66 paired rows, denominator matches, PT1/PT2/PT3 verdicts,
final measured theory/PROGRAM updates and green validator when complete.

Do not claim a three-arm exactly matched-exposure table: retained append-only
has 75,394,452 targets, versus the budgeted/twin target 71,677,042. No exact
append-only checkpoint at 35,419 updates has been established. Report its
actual exposure or establish that checkpoint; never interpolate a score.
Single seed, capacity/implementation and trajectory confounding remain risks.
Budget-pressure causality and efficiency extrapolation remain unresolved.

Manual notes: contact_supervisor is absent from the enabled tool catalog, so
no supervisor contact or approval is claimed. No card-1 search, union service,
WG machinery, sxgc or /mnt/nvme2n1 write was performed. No .pt file is committed.
No resilient DiLoCo behavior was changed; gap-matrix IDs are inapplicable.

Additional identity evidence: the first 303 per-step scalar losses and target
counts exactly equal the original frozen unbudgeted arm's retained curve.
This supports identical early training behavior, without claiming complete
model/optimizer fingerprints or future trajectory identity. The immutable
snapshot SHA256 was rechecked after retention.

Validation failure retained: a provisional Part I progress edit caused PROGRAM
validation to turn red because THEORY_INTERVAL_OPERATORS.md is hash-pinned.
That provisional edit was removed; the measured theory remains frozen pending
the actual verdict and coordinated binding update. Final validation is green
(17 claims / 369 bindings). The failed check is not counted as a pass.
