# C14-B matched-exposure twin: detached relaunch, verdict pending

Registration commit **8ad08f70** preceded training. All 23 original HARVEST
source hashes bind in `/home/erikg/emender_c14_matched_bound` at c190956d,
including the required 58a8acb8 replay. The optimized current source was not
substituted. No frozen arm was changed. New independent checkpoint, receipts,
explicit exposure stopping rule and all original training hyperparameters are
registered in PREREGISTRATION_C14_B_MATCHED.md.

Historical first launch (subsequently killed with its session): training launched through the shared broker on physical card 2. Launcher PID
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


## Session-kill recovery, 2026-11-05

The original console stopped at step 876 / 1,771,107 targets. Fresh process
inspection found neither original PID nor any matched trainer; the broker
listed only the unrelated card-1 lease, with cards 2/3/4 idle. The prior
"training remains active" observation did not survive the session ceiling.
The last complete atomic checkpoint loads at **625 steps / 1,262,440 targets**,
next window 10,000, active seconds 240.9313050354831. It supersedes step303.
Immutable external copy: `/home/erikg/c14_dense_bytes/emender_c14_b_matched_attested_step625.pt`,
SHA256 `da285ab343b19240e6e01c8f5f49cea5af3312a3f26eb24943299beae7fd7a4e`.
Exactly **251 updates / 508,667 targets** seen in the old console were unsaved
and must be replayed; they are not credited toward the final exposure.
The old console is retained as `receipts/c14_b_matched/session_killed_console.txt`.

Relaunch uses the unchanged registered `--resume` machinery, original frozen
source and broker wrapper, through `c14_b_matched_detached.py` with Popen,
new session, DEVNULL stdin and retained file descriptors. No training source,
frozen arm, union service or card-1 workload was changed. All checkpoint files
remain outside git. This is the second session-child loss; the mandatory
session survival playbook is synchronized in AGENTS.md and CLAUDE.md.

## Harvest state and completion requirements

Do not wait in an attended session. Monitor `train_console.txt`, the live
broker lease, and `training.json` under `receipts/c14_b_matched/`.
`train_terminal.json` must report exit_code 0; training.json must say
`completed_matched_exposure`, exactly 35,419 steps / 71,677,042 targets.
Verify the external checkpoint hash and final PB3 peaks <257 / capacity merges
zero; a terminal marker alone is insufficient. RESUME_STATE.json holds the
recovery checkpoint; train_detached_launch.json and relaunch_verified.json
hold the new launch and liveness evidence. No final training/eval success is
claimed by this relaunch receipt.

Prepared detached evaluation command (NOT started):
`/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_matched_detached.py eval`
Use `--resume` only when eval_resume.json already exists. The launcher and
registered evaluator both reject incomplete training. Evaluation writes
`eval_resume.json`, `eval_table.json`, `verdicts.json`, `eval_console.txt`, and
`eval_terminal.json`; require completed state and exit_code 0, all 66 paired
rows and exact frozen denominators. Commit PT1 sign versus budgeted
**1.584700014**, PT2 strata, PT3 efficiency extrapolation/saturation with
measured evidence, final report and coordinated theory/PROGRAM.md bindings,
then run validate_program.py green. No interpolated score or relabeled
append-only exposure. Full evaluation previously cost ~52,111 seconds.
Residual risks: host failure or future process death, one seed, capacity and
implementation confounding, training/evaluation still incomplete. Session
survival is attested by process ancestry and progress, not future completion.
No resilient DiLoCo behavior changed; gap-matrix requirements inapplicable.

Verified 2026-11-05T21:24:00.781911+00:00: detached wrapper PID 2490350, PPID=1, SID=2490350; trainer 2490415; card-2 lease starttime matched and heartbeat renewed across 1793913611 → 1793913791. Console advanced from resumed step625 through step664 to **step922 / 1,865,315 targets**, past the old step876 ceiling. Full evidence: relaunch_verified.json. Validator green (17 claims / 369 bindings); wrapper syntax and launcher parse passed. Intentional early eval launch probe failed closed on incomplete training, before spawning any evaluator.
