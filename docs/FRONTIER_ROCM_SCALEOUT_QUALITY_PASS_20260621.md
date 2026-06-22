# Frontier ROCm/Scaleout Quality Pass - 2026-06-21

Task: `quality-pass-frontier-rocm-scaleout`

## Scope Reviewed

Inspected the newly created Frontier ROCm/scaleout batch:

- `frontier-orient-handoff-paper`
- `frontier-allocation-ledger`
- `frontier-env-debug-recipe`
- `rocm-kernel-port-audit`
- `rocm-e97-mlp-port`
- `rocm-gdn2-mlp-port`
- `rocm-kernel-integration-smoke`
- `frontier-debug-benchmark-matrix`
- `frontier-stage-commapile-mainmix`
- `frontier-extended-e97-readiness`
- `frontier-diloco-scaleout-readiness`
- `frontier-results-synthesis`

## Edits Made

- Added or tightened `## Deliverables` sections so each task has a concrete
  expected artifact, command set, table, memo, or WG log/artifact output.
- Expanded `## Validation` sections to include exact evidence workers must
  capture: job ids, queue, node count, walltime, requested and actual
  node-hours, environment, git commit, logs, throughput, memory observations,
  loss sanity, and artifact links where applicable.
- Strengthened neutral framing across orientation, audit, porting, benchmark,
  readiness, and synthesis tasks. Workers are now asked to separate observed
  Frontier evidence, paper claims, prior-agent interpretations, hypotheses,
  risks, and recommendations.
- Made the expensive-run gate explicit:
  `frontier-debug-benchmark-matrix`, `frontier-extended-e97-readiness`,
  `frontier-diloco-scaleout-readiness`, and `frontier-results-synthesis` now
  require successful debug evidence and allocation accounting before any
  extended-queue recommendation.
- Added explicit human-approval language to extended e97 and DiLoCo readiness:
  those tasks prepare launch packages and designs but do not authorize
  extended-scale submissions without recorded WG approval.
- Added tags to mark the gated/neutral tasks:
  `debug-gated`, `budget-gated`, `human-approval`, and
  `neutral-hypothesis` where relevant.

## Gate Summary

The intended execution path is:

1. Build neutral context and accounting:
   `frontier-orient-handoff-paper`, `frontier-allocation-ledger`,
   `frontier-env-debug-recipe`, and `frontier-stage-commapile-mainmix`.
2. Audit and implement ROCm/HIP kernel paths:
   `rocm-kernel-port-audit`, then `rocm-e97-mlp-port` and
   `rocm-gdn2-mlp-port`.
3. Run debug-queue evidence collection:
   `rocm-kernel-integration-smoke`, then `frontier-debug-benchmark-matrix`.
4. Only after debug evidence and ledger accounting, prepare expensive-run
   packages:
   `frontier-extended-e97-readiness` and
   `frontier-diloco-scaleout-readiness`.
5. Synthesize the evidence and next launch decision:
   `frontier-results-synthesis`.

## Validation Result

- Every visible task in the Frontier ROCm/scaleout batch was inspected.
- Every visible worker task now has concrete deliverables and validation
  criteria.
- Extended Frontier work is gated behind debug evidence, budget accounting,
  and explicit human approval language.
- Prior no-go/go statements are framed as hypotheses or prior-agent
  interpretations unless reproduced by new evidence.
