# E97 Checkpoint and DiLoCo Quality Pass

Date: 2026-06-23
Task: `quality-pass-e97-checkpoint-diloco`

## Scope

This pass tightened the open E97-first checkpointing and DiLoCo task batch before execution. The research priority is Emender E97-MLP. GDN2-MLP remains a high-quality control arm, not the target that should absorb the batch because it is currently cleaner.

## Task Spec Updates

- `fix-e97-mlp-checkpoint-finalization`: now explicitly scoped to single-node E97-MLP checkpoint finalization and live Frontier resume validation. GDN2 is limited to control/reference use.
- `implement-walltime-final-checkpoint`: now requires live Frontier short-walltime E97-MLP validation, live resume from the final checkpoint, explicit rank/head behavior, and metadata/latest.pt consistency.
- `validate-e97-two-node-checkpoint-canary`: now explicitly depends on the two checkpoint tasks producing live E97 evidence, is limited to a cheap 2-node canary, and forbids proceeding to 4/8 nodes from that task.
- `plan-e97-first-diloco-scaleout`: now separates implementation prerequisites, live validation/canary evidence, and future scaleout execution. All 4/8-node launches are gated on clean E97 2-node checkpoint/resume evidence.
- `design-frontier-cmaes-tuning-track`: now explicitly stays non-blocking for the immediate E97 checkpoint/DiLoCo path and keeps GDN2 as the control.

## Validation Against Requested Criteria

- Every relevant task states E97-MLP is primary and GDN2-MLP is the control.
- Checkpoint/finalization tasks require live Frontier validation, not only syntax or unit checks.
- Multi-node DiLoCo work is gated behind reliable checkpoint/resume behavior, with the two-node task limited to a cheap launch canary.
- Task descriptions now distinguish implementation, validation/canary execution, scaleout planning, and the separate CMAES design track.

## Execution Boundary

No Frontier jobs were launched in this quality pass. The purpose was coordination hardening before dispatching implementation and validation workers.
