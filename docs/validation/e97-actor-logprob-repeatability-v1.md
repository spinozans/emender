# Actor log-probability repeatability diagnostic v1

The [no-update probability assay](e97-native-rl-logprob-qualification-v1.md)
failed its original actor-replay and maximum training-layout tolerances. This
is an investigation of that failure, not a replacement acceptance gate.

## Frozen diagnostic

Source recipe SHA:
`ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f`.
Use recorded sample-0 turns: task 000 turn 0 (stable control), task 004 turn 1,
task 010 turn 1, task 014 turn 4 (observed large discrepancies). This is
explicitly post-result diagnostic selection, not independent validation.

On two fixed leased GPU workers, load the same immutable correction-y checkpoint
and hash all BF16 parameters before/after. Each worker performs two repetitions
of each selected turn under three execution shapes:

1. Compute log probability **before** consuming each forced recorded token.
2. Retain the logits reference, consume that token, then compute probability
   **after** the step, matching the actor's capture placement.
3. As (2), but also perform/discard an untruncated sampling draw before consuming
   the recorded token, probing the sampler's allocation/RNG execution shape.

No variant generates a new rollout: all consumed token IDs are forced from the
original trace. Mode order is reversed on the second repetition. Seed is fixed
per trial; this probes sampling execution shape, not the entire historical RNG
stream. Record selected logits, log normalizers, selected log probabilities and
full-logit byte hashes, without publicly decoding any private generation.

Compare repeat calls, capture/sampling shapes and workers. Check worker parameter
identity, finite values and exact coverage. Record GPU/runtime/math metadata.
No tolerances are changed and this diagnostic cannot mark the original
qualification passed. Measurements may themselves affect scheduling/allocation;
interpret observed differences as localization evidence, not proof of a cause.

Two checked leased GPUs, explicit local device and NUMA binding, isolated Triton
caches, both expandable allocator settings, BF16 persistent weights, no optimizer
or external tools. Worker deadline 1,800 seconds; outer bound 2,100 seconds;
30-second kill grace; no retry or communicator reuse. No resilient/distributed
training qualification, first-party admission, promotion or automatic expansion.

Scripts:
- `scripts/diagnose_e97_actor_logprob_repeatability.py`
- `scripts/run_e97_actor_logprob_repeatability.sh`

Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-rl-replay-diagnostic-v1`.
CPU validation: **72 passed**, including comparison/coverage regression checks
and the existing policy-loss, mask and runtime tests. GPU results pending.
`rl_optimizer_ready:false` remains unchanged.
