# Larger native SFT exposure: 50M agent targets

## Decision and frozen data policy

The completed six-rate screen shortlists **1e-5** provisionally for a substantive
run, not as a proven optimum. See `e97-native-lr-screen-v1.md` for the full
learning/retention/generation trade-off. No checkpoint has been promoted.

`configs/pi/e97-native-training-50m-agent-mix-v1.json` requests:

- 50,000,000 native assistant targets, complete trajectories, **no repeated
  native source record** in this derivative. Shuffle problems per round;
  remove exhausted problems instead of repeating their sole attempt.
- 100,000,000 complete no-think conversation targets, without replacement.
- 16,666,667 Pi/compositional retention targets. The admitted source has only
  6,339,569 eligible targets, so explicitly replay up to **three shuffled
  epochs**, without replacement within each epoch. This is retention replay,
  not novel data. Record unique/repeated source-record counts.

Nominal fractions remain 30/60/10. Whole-record overshoot is counted; no text,
arguments, observations, private/public routing or loss masks are rewritten.
Use source-bound reconstruction/executor evidence and the same explicit
internal-use authorization. Protected repositories and development problems
remain excluded. No raw authority mutation, first-party registry admission,
licensing-clearance claim or independent-final-holdout claim.

The sampler extension preserves the original smoke's default behavior. New
options are explicit: `unique_native:true` only for native records and
`epochs:2/3` only for retention. Reject insufficient quotas within that bound.

## Execution direction

Build and validate boundary-aware 65,536-context packs, then materialize the
complete fixed-world schedule on CPU. Plan enough eight-rank batches for one
complete assembled-mixture traversal, rounded up to a K4 boundary; report the
small alignment replay separately. Runtime IDs/counts must match this schedule.
The resulting agent exposure must be between 50M and 65M before any GPU launch.

Use the established BF16 SR, FP32 checkpointed CE128, MLP4096, group3,
eight-rank full-world DDP/no-merge numerical recipe, fresh optimizer from the
trusted parent y, provisionally LR1e-5. Planned successful segments are up to
128 updates, with separate checkpoint evaluations before continuation. Preserve
all checkpoints. Failure stops the program; no failed-child retry, requeue or
communicator reuse. Freeze actual data hashes, update endpoints, launchers and
non-catastrophic retention criteria before running the first segment.

This document records preparation, **not a launched or completed GPU run**.
The 50M-token program and autonomous execution validation remain to be run.
The earlier exact-restoration evidence remains valid; numerical fresh
continuation has not been measured and is not relabeled as passing.

ADR-003 safety intent: R07/R12 committed atomic checkpoint/restoration,
R14/NDP13 bounded failed-child termination, R16 evidence identity discipline,
and NDP15 synchronous atomic publication. No elastic/native data-plane,
async/overlap, Frontier/ROCm, additional-node scale or communicator-shrink claim.
