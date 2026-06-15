# Evaluation: run-ref-gdn2

Task: `run-ref-gdn2`
Evaluator: `agent-1452`
Date: 2026-06-15

## Grade

Overall score: 0.00 / 1.00
Confidence: 0.99
Rubric underspecified: no

The task rubric is explicit and operational: it requires a completed single-GPU
GDN-2 MLP reference training run to at least 2B tokens, using the exact
schedule-free constant-LR recipe from `docs/SCALE_PLAN.md`, with fused kernels,
matched corpus/tokenizer/held-out data, durable checkpoints, and a measured
loss/BPB curve. The submission has no durable run artifact or measured result to
evaluate against those criteria.

## Dimension Scores

| Dimension | Weight | Score | Rationale |
| --- | ---: | ---: | --- |
| Correct training recipe and GPU discipline | 0.20 | 0.00 | No run logs or launcher artifact demonstrate exactly one leased GPU, schedule-free optimizer, constant LR `4.74e-4`, or absence of warmup/cosine/decay. |
| Required model/data parity | 0.15 | 0.00 | No evidence shows the GDN-2 MLP geometry `dim2176 nh30 dep12 mlp3.259 bs4`, or that the corpus/tokenizer/held-out tensor matched the emender reference setup. |
| Fused-kernel requirement | 0.15 | 0.00 | No logs confirm the FLA chunked GDN-2 fused path, and no evidence rules out eager or slow fallback. |
| Durable measured output | 0.25 | 0.00 | `/mnt/nvme1n1/erikg/ref_gdn2_mlp/` was absent/empty at evaluation time; no checkpoints or `(step, tokens, train_loss, heldout_bpb)` curve were available. |
| Completion to budget and measured conclusion | 0.20 | 0.00 | The run did not reach `>=2B` tokens and no measured held-out monotonicity or rollover report exists. |
| WG delivery hygiene | 0.05 | 0.00 | The task had no actor commit, no recorded artifact, and no progress log beyond spawn/evaluator messages. |

Weighted total:

`0.20*0.00 + 0.15*0.00 + 0.15*0.00 + 0.25*0.00 + 0.20*0.00 + 0.05*0.00 = 0.00`.

## Evidence

- `wg show run-ref-gdn2` reported the task in progress with `Commits ahead: 0`
  and no implementation or run artifacts recorded before this evaluation.
- `wg log run-ref-gdn2 --list` contained only pause/publish/spawn entries before
  evaluator inspection; it did not contain training setup, launch, validation, or
  measured output logs.
- `find /mnt/nvme1n1/erikg/ref_gdn2_mlp -maxdepth 3 -type f ...` returned no
  files, so the required durable checkpoint/curve directory was not populated.
- `wg context run-ref-gdn2` reported no dependency artifacts.

## Validation Run By Evaluator

- Inspected `wg show run-ref-gdn2`.
- Inspected `wg log run-ref-gdn2 --list`.
- Searched the required durable output path:
  `/mnt/nvme1n1/erikg/ref_gdn2_mlp/`.
- Checked graph context with `wg context run-ref-gdn2`.

No task-specific acceptance criterion was satisfied. This is not a partial
completion or a launch-confirmation case; the required measured reference run is
missing.

## Final Assessment

The calibrated grade is `0.00`. The task asked for a durable, completed
reference training run and measured curve, but no such run output exists and no
logs establish any of the recipe, kernel, data, or completion requirements.

## Re-evaluation Note: 2026-06-15T16:58:59Z

After the original evaluation, the task was reset from `Done` back to
`in-progress` with a mandatory launch protocol. That reset confirms the original
submission was not accepted as a completed reference run. I rechecked the durable
output location and task context after the reset:

- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/` still did not exist at re-evaluation time.
- `wg context run-ref-gdn2` still reported no dependency artifacts.
- The new task log instruction requires a detached healthy training process, PID
  record, live GPU verification, and curve-file verification before completion;
  none of those post-reset artifacts were present for the actor output being
  graded.

This does not change the calibrated grade. The score remains `0.00 / 1.00` with
confidence `0.99`, because the evaluated actor output contains no measured run,
no durable curve, no checkpoint, and no validation evidence for any required
training criterion.

## Retry-Pass Note: 2026-06-15T17:01:36Z

The task was retried after the incomplete mark. I rechecked the evidence surface
before closing this evaluator pass:

- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/` was still missing.
- Process inspection found an unrelated `ref_emender_mlp` training process, but
  no `ref_gdn2_mlp` or GDN-2 MLP process.
- `wg context run-ref-gdn2` still reported no dependency artifacts.

The calibrated grade remains `0.00 / 1.00` with confidence `0.99`. No new
evidence satisfies any validation item for the required GDN-2 MLP reference run.

## Retry-Pass Note: 2026-06-15T17:02:00Z

This evaluator retry continued from commit `d058606` rather than restarting the
review. I rechecked the same acceptance-critical evidence:

- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/` still did not exist.
- A search under `/mnt/nvme1n1/erikg` found `ref_emender_mlp` only, not the
  required `ref_gdn2_mlp` durable output directory.
- Process inspection found the unrelated `ref_emender_mlp` training process and
  no live `ref_gdn2_mlp`/GDN-2 MLP training process.
- `wg context run-ref-gdn2` still reported no dependency artifacts.

The calibrated grade remains `0.00 / 1.00` with confidence `0.99`. The task is
not merely missing final polish; it is missing the required detached GDN-2 MLP
reference run, durable checkpoint/curve output, and measured validation results.

## Continuation Note: 2026-06-15T17:04:21Z

This continuation picked up after commit `59db68b` and checked only for new
evidence that would change the retry-pass evaluation:

- `wg msg read run-ref-gdn2 --agent $WG_AGENT_ID` reported no unread messages.
- `/mnt/nvme1n1/erikg/ref_gdn2_mlp/` still did not exist.
- Process inspection showed unrelated `ref_emender_mlp` and DiLoCo training
  commands, but no live `ref_gdn2_mlp`/GDN-2 MLP reference process.
- `wg context run-ref-gdn2` still reported no dependency artifacts.

The calibrated grade remains `0.00 / 1.00` with confidence `0.99`; the explicit
validation checklist remains unsatisfied because there is still no durable
GDN-2 MLP run output, checkpoint, curve, or measured >=2B-token result.
