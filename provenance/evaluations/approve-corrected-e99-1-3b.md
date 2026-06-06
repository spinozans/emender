# Evaluation: approve-corrected-e99-1-3b

Task: `approve-corrected-e99-1-3b`
Evaluator: `agent-1154`
Date: 2026-06-06

## Grade

Overall score: 0.55 / 1.00
Confidence: 0.89
Rubric underspecified: no

The task rubric is explicit enough to grade: the approval gate had to record a
human approval note for `corrected-e99-1-3b`, include approved launch
scope/constraints, avoid launching any LM/CMA/capability GPU run itself, and
ensure the corrected task is retried with the approval note passed through the
launch flags. There is a small tension between "record the approval message
only" and the validation item requiring a downstream retry, but the checklist
still gives clear observable criteria.

This evaluation grades the actor output before this evaluation artifact was
added.

## Dimension Scores

| Dimension | Weight | Score | Rationale |
| --- | ---: | ---: | --- |
| Human approval note naming the gated launch | 0.30 | 0.80 | The task log records `HUMAN APPROVAL (Erik Garrison, 2026-06-06)` and says there is an explicit go to launch the corrected E99 1.3B fair comparison. It includes the quoted human approval text. Credit is not full because the note does not literally name the task id `corrected-e99-1-3b` in the quoted approval and relies on surrounding paraphrase for identification. |
| Approved budget, GPU scope, and wallclock/token constraints | 0.25 | 0.25 | The approval note records only a rough "1.3 billion run" scope plus prerequisite/fairness context. It does not specify GPU set/count, parallelism, wallclock cap, token cap, idle-only/no-preempt policy, or similar budget constraints. A later human scope note appears on `corrected-e99-1-3b`, but it was not part of this approval task's recorded approval note. |
| No approval-task GPU launch | 0.25 | 1.00 | The approval task produced no actor commits or artifacts and only recorded a WG log entry before marking pending evaluation. The downstream task remained failed, and there is no evidence this approval gate launched an LM, CMA, capability, or GPU job. |
| Downstream retry with approval flags | 0.20 | 0.00 | `wg show corrected-e99-1-3b` still reports the task as failed, with no retry of the corrected run after approval and no evidence of `--approved-human-go --approval-note ...` being passed to the launch drivers. |

Weighted total:

`0.30*0.80 + 0.25*0.25 + 0.25*1.00 + 0.20*0.00 = 0.5525`, reported as `0.55`.

## Evidence Reviewed

- `wg show approve-corrected-e99-1-3b` showed the actor log:
  `HUMAN APPROVAL (Erik Garrison, 2026-06-06): explicit go to launch the corrected E99 1.3B fair comparison. Verbatim: 'Yes, launch detected. 1.3 billion run.' Both technical prerequisites (fix-e99-mixture, implement-triton-fused) are done; size-fair + throughput-fair conditions met.`
- `wg show approve-corrected-e99-1-3b` reported `Commits ahead: 0`,
  `Uncommitted files: 1`, and no artifacts before this evaluation report.
- `wg artifact approve-corrected-e99-1-3b` reported no produced artifacts.
- `git log --oneline main..HEAD` returned no actor commits before this
  evaluation artifact.
- `git diff --stat main...HEAD` returned no actor delta before this evaluation
  artifact.
- `wg show corrected-e99-1-3b` showed the downstream task remained `failed`.
  Its failure reason was still the missing human approval gate from the earlier
  attempt, and there was no evidence of a retry carrying `--approved-human-go`
  and `--approval-note`.

## Validation Checklist Assessment

- A human posts an explicit approval note naming `corrected-e99-1-3b` and
  authorizing launch: partially met. The WG log attributes approval to Erik and
  authorizes the corrected E99 1.3B fair comparison, but does not quote the
  exact task id.
- The note records approved budget/GPU scope and any wallclock/token
  constraints: mostly not met. It records a rough 1.3B scope, but omits the
  concrete resource and runtime constraints the checklist asks for.
- No LM/CMA/capability GPU run is launched by this approval task: met.
- After approval, retry `corrected-e99-1-3b` with the approval note passed to
  the launch drivers via `--approved-human-go --approval-note ...`: not met.

## Final Assessment

The actor handled the narrowest approval-gate action safely: it recorded a
human approval note and did not start an expensive run. The output is materially
incomplete for the full checklist because the approval note lacks concrete
budget/GPU/wallclock/token constraints, and the downstream corrected task was
not retried with the required approval flags. The calibrated score is therefore
slightly above half credit, driven by satisfying the human-go/no-launch guardrail
but missing the launch-scope and handoff requirements.
