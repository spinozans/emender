# Evaluation: run-32n-e97

Date: 2026-06-27
Evaluator: agent-369
Task: `run-32n-e97` (`Run 32n E97 merge topology A/B`)

## Verdict

Score: **0.00 / 1.00**

Confidence: **0.97**

Rubric underspecified: **No**. The task has concrete required work and a
specific `## Validation` checklist.

## Rationale

The task requested a 32-node A/B smoke comparing bucketed global DiLoCo merge
against hierarchical DiLoCo merge from the same step-383500 E97 1.3B checkpoint.
At evaluation time, there is no evidence that the A/B work was started:

- `wg show run-32n-e97` shows the task is still `in-progress`, assigned to
  this evaluator worktree, with no artifacts recorded.
- `git log main..HEAD` is empty for this worktree; no task commit exists.
- `reports/frontier/` contains prerequisite reports from `debug-e97-1-3b`,
  `integrate-and-test`, and `fix-failed-hierarchical`, but no report comparing
  bucketed global versus hierarchical at 32 nodes for `run-32n-e97`.
- `rg` over `reports`, `docs`, and `.wg` found no `run-32n-e97` evidence other
  than task creation/assignment/evaluation scaffolding.
- No Slurm job IDs, stdout/stderr paths, run roots, configs, monitoring results,
  throughput figures, merge counts, loss trends, checkpoint status, collective
  failures, or ladder recommendation were produced for this task.

Earlier reports prove that bucketed global had prior 32-node diagnostic evidence
and that a 1-node hierarchical smoke passed after the root-group fix. They do
not satisfy this task's requested same-checkpoint 32-node A/B comparison.

## Dimension Scores

| Dimension | Score | Notes |
|---|---:|---|
| Required A/B job submission | 0.00 | No global or hierarchical 32-node A/B job IDs are present for this task. |
| Configuration fidelity | 0.00 | No evidence of `DILOCO_MERGE_TOPOLOGY=global`, `DILOCO_MERGE_TOPOLOGY=hierarchical`, shared bucket size, group sizing, GPU-island/no-DDP, RCCL env, or alt rendezvous setup for this task. |
| Monitoring to terminal/blocking state | 0.00 | No monitoring evidence exists. |
| Report completeness | 0.00 | No task report under `reports/frontier/` compares the two 32-node arms. |
| Metrics and recommendation | 0.00 | No throughput, merge counts, averaged loss trend, checkpoint status, failure analysis, or 64/128/256 ladder recommendation. |
| Ladder safety constraint | 0.50 | I found no evidence in the inspected task record that larger ladder tasks were unpaused by this task, but the primary gate evidence was never produced. |

Overall, the actor output is absent for the requested experiment. The calibrated
grade is therefore zero despite the task being well specified and despite
successful prerequisite work in neighboring tasks.

## Validation Checklist Assessment

- Both job IDs, run roots, stdout/stderr paths, and configs are logged: **Not met**.
- Both jobs are monitored to terminal state, or queue/blocking state is explicitly reported: **Not met**.
- Report under `reports/frontier/` compares bucketed global vs hierarchical at 32 nodes: **Not met**.
- No larger scale ladder tasks are unpaused unless the 32n A/B supports doing so: **Partially met / no contrary evidence found**, but this does not compensate for the missing A/B.

## Recommendation

Return `run-32n-e97` to implementation rather than unblocking downstream ladder
work. A retried actor should submit the two short 32-node K20 jobs from the
verified step-383500 checkpoint, monitor both to completion or explicit queue
block, and write the required comparison report before any 64/128/256 ladder
tasks are resumed.
