# Evaluation: build-launch-wrapper

Task ID: `build-launch-wrapper`
Evaluator: `agent-1466`
Date: 2026-06-15

## Verdict

Score: 0.00 / 1.00
Confidence: 0.98
Rubric underspecified: No

The task had explicit validation criteria. The delivered work does not include
the required `scripts/launch_detached_run.sh` script, no implementation commit
is present on this branch, and no self-test evidence was recorded.

## Dimension Scores

- Required wrapper exists, executable, committed: 0.00
- Detached parent-exit self-test proof: 0.00
- GPU lease behavior implemented correctly: 0.00
- PID, manifest, stdout, and prompt-return behavior: 0.00
- Safety constraints: no training and no real GPU lease: 1.00
- Git/WG hygiene for implementation task: 0.00

Overall score is 0.00 because the central deliverable is absent. The only
satisfied safety property is negative evidence: no training run or GPU lease
appears to have been started, but that does not compensate for the missing
wrapper.

## Evidence

- `scripts/launch_detached_run.sh` is absent.
- `git log --oneline main..HEAD` showed no implementation commits before this
  evaluation artifact.
- `git status --short` only showed untracked `.wg` before this evaluation
  artifact was created.
- `find . -maxdepth 3 -type f \( -name '*launch_detached*' -o -name '*detached*run*' \) -print`
  returned no files.
- `wg show build-launch-wrapper` showed the task still in progress with zero
  commits ahead and no artifacts or validation logs from an implementing actor.

## Criteria Assessment

- `scripts/launch_detached_run.sh` exists, executable, committed: Not met.
- Self-test proves detached process outlives launching shell, with pid alive and
  log growing after parent exit: Not met.
- Lease acquired inside detached session for `--gpus > 0`, and `--gpus 0`
  skips leasing: Not met; no script exists to inspect or test.
- Writes `run.pid` and `launch_manifest.json`, prints PID, returns promptly:
  Not met; no script exists to inspect or test.
- No training started, no real GPU leased: Met by absence of implementation
  activity observed during evaluation.

## Recommendation

Mark the task incomplete and retry with an implementation agent. Downstream run
tasks should not consume this dependency until a future attempt creates,
self-tests, commits, and records the detached launch wrapper.
