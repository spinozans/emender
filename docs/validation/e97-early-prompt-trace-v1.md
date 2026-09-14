# E97 early-prompt first-divergence trace

## Target and frozen references

The [packed-stage prerequisite](e97-packed-forward-v1.md) failed on its early
prompt probes before any64K forward: record0 prediction row5 had actor/trainer
log-probability gap.056125640869140625. The96 assistant probes agreed within
.00003610987914726138, but the frozen120-position gate remains failed.

Trace logical **input rows0–5**, the causal inputs for prediction row5/token6.
Keep the original complete native executions: actor scores all eight boundary
predictions using the original fresh-cache/tokenwise path; teacher evaluates the
entire2,657-token record with2,656 input steps, original masks and final-state
observation. Do not truncate the teacher to six tokens or alter the actor prefill.
Weights, architecture/chunks, `fp32-linear-v2` and all thresholds stay fixed.

Reference root `packed-forward-v1` under
`/mnt/nvme2n1/erikg/e97_systematic_posttraining`:

| Authority | SHA256 |
|---|---|
|Base plan|`af9339afe3819cd495ec1a052b77c5d46761bb060ac0d910ac0f73b2240d1fbf`|
|Inputs|`7340c9d573acce2b1f9e79c77cc25a3130f4759c80726cab72f13ca00e9da7b9`|
|Each original actor boundary receipt|`aa16f806993d852e1e0a8fd5b4478e0ea852965d119abf5a7fb65a959e1db99b`|
|Each original standalone teacher receipt|`e5eae759930f2bd1bae3b1368297ef0f1526c0b39acb3a37802f9465f769c491`|

## Trace and interpretation gates

Reuse the original packed assay's `actor` and `teacher` functions without changing
their execution logic. `Rows` captures only the first six logical rows across
prefill chunks and one-token calls. Its hooks return `None`; any scoped norm
observer returns original objects and restores the original symbol.

1. Coarse37 sites: embedding and18 block inputs/outputs, actor/teacher twice each.
2. Fine28 sites (or the existing fused-norm equivalent) in the first measured
   differing block: outer norm, raw projections/gates, mixer readout/projection,
   post-mixer norm and MLP, again twice per path.

Before interpretation, every execution must reproduce the original eight actor
scores or the **entire original teacher receipt**:40 selected scores, hidden/logit
hashes, native CE/count/delta and all18 final-state hashes. Trace tensors must
repeat byte-exactly. Coarse/fine selected-block input/output hashes must also bind.
On binding failure, stop and mark the trace uninterpretable.

Report both first differing site in declared topological order and earliest
logical token. Neither implies instruction-level causality or observation of all
internal recurrent registers. A later-token discrepancy cannot explain an earlier
one merely because its module comes first in the source.

After successful binding only, inspect at most three same-input/different-output
Linear coordinates, chosen by earliest logical row then declared projection order.
Use the actual BF16 input, weight row and optional bias for a **CPU FP64 selected-dot
reference**, with explicit nearest-even BF16 rounding (no intermediate float32
rounding). Save these small operands privately. This is not a full FP64 model,
replacement output, numerical-policy repair or GPU operation replay; FP64 arithmetic
itself remains finite precision. If no eligible Linear coordinate exists, report
zero cases rather than substitute different inputs.

## Safety and budget

A terminal `try/finally` guard retains parameter/buffer fingerprints, gradient/tuner
checks and allocated HBM on ordinary success or failure. It preserves the primary
exception if safety passes and fails closed if the safety audit fails. SIGTERM
attempts orderly unwinding; forced SIGKILL or an unusable CUDA context cannot
provide guaranteed in-memory evidence. This addresses the missing terminal
measurements in the prior early-stop path without rewriting that old failure.

**105 CPU tests passed,14 CUDA skips.** Coverage includes primary-exception
preservation, parameter mutation and audit
failure reporting, temporal-versus-site ordering, BF16 rounding near a midpoint,
and actual BF16 selected-dot/bias operands, alongside the existing trace, packed
layout and policy tests.

One leased worker; maximum eight native evaluations, three selected CPU dots,
16MiB per trace bank,64MiB live trace payload,16GiB allocated HBM. Worker900s,
outer1,500s, teardown30s. Immutable export/inventory, before/after source audits
composed with owned lease release, NUMA/local CUDA routing, isolated cache and both
expandable allocator variables. No retries, precision sweeps or overlapping jobs.

Frozen plan SHA:
`fb23283d38c95992e1e3f85bb4808694ee05045401a87a4b1a21223dce4273e9`.
CPU authority preflight: `early-prompt-trace-v1-preflight`.
Run root: `early-prompt-trace-v1`.
Driver `scripts/run_e97_early_prompt.sh`; observer/worker
`scripts/diagnose_e97_early_prompt.py`.

No64K forwards, backward, optimizer updates, rollout-probability replacement,
training admission or reopened880+32 budget. The numerical prerequisite stays
failed regardless of diagnostic success. Private token/activation/probability/
selected-weight artifacts remain unpublished. Results pending.
