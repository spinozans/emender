# F8 — persistent cross-cycle correction service (staged, OFF)

2026-10-13, lambda01. Implementation commit **`8a5c89e7`**, scoped to six F8
source/deployment files, pushed to origin/main. No live scripts/configuration,
GPU leases, training behavior or bank receipts were changed by this task.
The existing single collection lane (PID 3127801) remained running. Report and
machine evidence are also supplied under `emender-scratch/fix-fanout`.

## Result / approved architecture

The per-cycle correction fence is replaced, when the existing durable
`async_enabled` knob is ON, by a **persistent CPU correction executor plus
collect-owner publication**. Async remains default OFF. Existing F7 configuration
files remain valid; `pool_width` now means persistent workers per lane (default
1), and `backlog_capacity` bounds admitted-but-unpublished work (default 64).
Full backlog returns F7's explicit `pool exhausted` failure immediately.

The supervisor resolved two real contract seams during implementation:

1. The **10s correction-EPISODE-start gate moves to the persistent service**.
   Durable enqueue never sleeps. Retaining collect-side admission sleep would
   contradict GPU-never-blocks when attempts finish in under 10s. Start times
   are durable across cycles/restarts, with monotonic pacing in a live service.
2. **Conversation-async support is deferred to F9 deployment.** F8's pre-F9
   runtime explicitly retires scripted conversation tasks failed/ineligible
   before generation; worker execution also rejects them. Routing stays
   fresh-only, never spliced. F9's separately qualified staged driver/bridge
   are deliberately not merged into F8. No conversation tasks exist in the
   current live pool; the real F9 runtime seam needs separate qualification.

The publication choice preserves F7's logical single writer rather than
introducing a concurrent stream writer. The lane daemon waits synchronously
for its collect child and publishes only outside that child's lifetime; the
child polls ready corrections without a worker/result wait. New late receipts
use the **current validated stream tail**, but their original task, cycle,
served-checkpoint SHA, failed attempt/grade and splice provenance stay bound.
No checkpoint adoption, PG/SFT arithmetic or exposure rule is changed.

## Durability, summary and failure boundaries

* Admission copies the owned attempt/workspace bytes, fsyncs them and publishes
  a read-only spec by durable rename. Owned-file hashes augment F7's task,
  attempt/grade/checkpoint and fresh/spliced bindings. Crashed-cycle archival
  cannot rename that owned workspace. Unadmitted staging from a dead admission
  incarnation is swept; live staging is not removed.
* The durable job takes over the original task-claim lifetime. TTL sweeps and
  failed-collect requeue skip that exact delegated task incarnation, preventing
  duplicate/splice reuse while its cycle/daemon is gone. Publication or an
  explicit failure retires it under the original round-retry policy.
* A nonblocking singleton service lock prevents competing managers. Each
  correction retains its separate Python/Pi process group, F7's 750s hard wall,
  parent-death containment and deadline adapter. Worker crashes release slots
  with explicit failures. On service restart, interrupted execution claims
  fail closed (not rerun against mutated splice state); queued jobs resume.
  Completed API calls and partial-log-tail status survive service loss.
* A durable publication intent precedes append. Recovery handles an intent/
  immutable-copy-only crash and an append-before-retirement crash. It validates
  the exact existing receipt and retires once; it cannot duplicate an append
  or reuse a stale tail. Corrupt/partial stream tails stop loudly/fail closed,
  not by truncating evidence or guessing whether a line committed.
* Cycle summaries no longer claim all corrections finished: they list pending
  job IDs explicitly. Summary snapshots stay immutable; final outcomes, API
  metrics, receipt intents and acknowledgements live in the spool. Accounted
  publication IDs retain late receipt counters across owner restart.
* Async OFF stops new admissions, not already owned work. A stopped daemon does
  not destroy the service/backlog; resuming the owner publishes ready outcomes.
  Owned workspaces remain referenced evidence after completion, not orphaned
  worker processes. Read `F8-deployment.md` before any attended shutdown/relaunch.

## First-class backlog accounting

Cycle summaries AND cycle metrics include queue/backlog depth, in-flight and
ready-to-publish counts, per-correction wait time, oldest pending age, admission/
start/finalization/native-close totals, published receipts/targets, generation
attempts/h, correction admissions/outcomes/h and backlog growth/h. Cycle rates
use one common generation observation wall; lifetime rates expose their own
observation wall. A finalized failure is explicitly not a native completion or
eligible receipt. These measurements enable endpoint/GPU sizing; F8 does not
relabel F7's projected ~5–6-generator equilibrium as a live sustained result.

## GPU-never-blocks proof and validation

**Repo: 215/215 CPU bank checks. Staged bundle: 215/215. Pytest: 6/6.**
The actual `bank_collect` loop, real durable spool and an actual OS correction
worker were exercised with CPU policy/grader fixtures: six attempts, two
admissions, four explicit exhaustions at the bound, a correction sleeping 3s,
and the real 10s configured pacing value. Attempts 2–6 ran while the correction
process was still alive; the collect loop ended in **0.021913s repo /
0.022052s staged**, with **0.0s terminal correction drain**. No correction-result
or pacing sleep entered the generation path. Timing artifacts:
`F8-unit/F8-gpu-never-blocks.json` and
`F8-unit-staged/F8-gpu-never-blocks.json`.

These are **CPU integration fixtures, not live GPU/API throughput benchmarks**.
Admission copying/fsync, ready-receipt encoding/publication, policy grading,
ordinary lease/training boundaries and historical-spool metric scans still
have costs. The proved property is no wait for a correction's execution,
readiness, endpoint pacing or exhausted capacity—not zero filesystem work.

`rl_bank_unit_test.py` now also covers cross-cycle reverse-order publication,
actual digest-chain verification and full canonical token/mask re-encoding,
original checkpoint binding after cycle archival, both owner publication crash
boundaries, actual worker SIGKILL, actual service SIGKILL and Pi-like descendant
containment, singleton manager fencing, durable restart/pacing/backlog metrics,
owned-workspace non-aliasing/hash rejection, delegated claim sweep/requeue,
late failure/unclosed/malformed outcomes and explicit conversation fail-closed.
F7 pool tests and prior PG/SFT/adoption/exposure regression checks remain.

Initial development runs exposed fixture/runtime issues (mock grade lacked a
true grade gate; worker fixture passed an extra argv; a cold manager import
needed a larger test-only startup bound; conversation-only collect needed its
summary imports outside the attempt loop; a routing fixture and test-local
encoder import needed correction). All were fixed before both final suites.
No API requests, GPU model loads or scheduler submissions ran.

Commands (lambda01 `.venv/bin/python`; no Frontier activation/submission):

```bash
W=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1
export PYTHONPATH=/home/erikg/emender/scripts/rl-loop-v1:/home/erikg/emender:$W/F8-staged/scripts
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/rl-loop-v1/rl_bank_unit_test.py --scratch /home/erikg/emender-scratch/fix-fanout/F8-unit
export PYTHONPATH=/home/erikg/emender:$W/F8-staged/scripts
CUDA_VISIBLE_DEVICES= .venv/bin/python "$W/F8-staged/scripts/rl_bank_unit_test.py" --scratch /home/erikg/emender-scratch/fix-fanout/F8-unit-staged
.venv/bin/python -m pytest -q tests/test_rl_teacher_deadlines.py tests/test_rl_teacher_probe_report.py
.venv/bin/python -m py_compile scripts/rl-loop-v1/{rl_bank,rl_bank_lane,rl_bank_unit_test,rl_correction_config,rl_correction_service}.py
bash -n "$W/F8-staged/scripts/run_bank.sh"
git diff --check
```

Every overlapping repo/NVMe staged source is byte-identical; unchanged companion
scripts come from F7 staging, not live edits or F9. `F8-evidence.json` binds source,
complete staged bundle, final test logs and fixture timing artifacts by SHA-256.
No files remain staged after commits. Independent review is still required
before an operator-approved relaunch or OFF-to-ON trial.

## Live safety receipt / remaining qualification

Repeated read-only checks retained unchanged live SHA-256 values:

| Live artifact | SHA-256 |
|---|---|
| lane | `0c1257f1207a0fabd721571b566465597ee2c8ee7610be5d3900128cae4b9a0c` |
| driver | `da9c5f79d3584c69e933c04e8b4719c792c1ed2b4d6fae2e22ede4e19af00821` |
| bank helper | `3da594e8c9a27dabcab834d02e3d8d295ef2cbcffcdb26fe3e4296ebc4bd5e98` |
| existing correction config | `5519dbc52e1ea6f37539f36810f8041c3664ae514c8bc838a8b4df4835bc5876` |

The existing live knob is DeepSeek / async OFF / width 1; it was not created or
changed here. Test correction services/workers were shut down; the live single
collector stayed untouched. Do not substitute F7's older live-lane hash for this
operator-updated baseline.

Remaining rollout gates: independent review; complete lane/coordinator helper
relaunch together; all-caller endpoint budgeting; sustained mixed-workload/
provider-cancellation and large-history I/O/retention qualification; real F9
conversation-async qualification after deployment. Interrupted splices fail
rather than auto-retry, and storage-corrupt streams require attended repair.
No global semaphore, endpoint reservation, cognitive-quality, fully-busy GPU,
resilient-training overlap or scale gate is claimed.

Authority read: `RESILIENT_DILOCO_COMPUTE_POOL.md` and its gap matrix. Applicable
safety/evidence intent: **R14/NDP13** bounded local termination and nonblocking
background failure; **R15** unchanged eligibility/reference accounting; **R16**
exact-source evidence and honest scope. R07/R12 checkpoint mutation and retired
elastic/native/v2.1/ISP conformance are unclaimed. Frontier Partition/QOS evidence
is not applicable on lambda01 without a submission.
