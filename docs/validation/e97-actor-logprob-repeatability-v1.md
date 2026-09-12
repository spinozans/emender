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
and the existing policy-loss, mask and runtime tests.

## Completed controlled diagnostic

`proc_e5fd` completed in **280 seconds** from immutable `ad1433f0`. Both source
inventories passed. All tested repeat calls, before/after capture shapes,
sampling-shaped calls and worker comparisons were **bit-identical**, including
full-logit byte hashes: maximum log-probability delta **0** in all three groups.
Both RTX 6000 Ada workers had identical unchanged BF16 parameter hash
`f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd`, also matching
the original probability assay. Metadata agreed: Torch 2.9.1+cu128, CUDA 12.8,
TF32 disabled, BF16 reduced-precision reduction disabled, highest float32
matmul precision, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, visible GPUs 0/1.

A CPU cross-run audit shows that these four controlled replay results exactly
match the earlier probability assay's **forced actor replay** values, not the
original on-policy actor traces. The original actor discrepancy remains as large
as .1335446835 in this selected subset. This localizes the investigation but
neither proves a root cause nor clears the failed qualification.

- Recipe SHA: `702d9a18f1cfbd41d99242a27edcb1ff281d56a979bcc5d0dd80f19d79570790`.
- Summary SHA: `2912e97cb24beaf666a7402833e417c02e169e27c1a1f3bdbabec0e64bc58509`.
- Cross-run audit SHA: `0f8935e319b5eac2cfec2d55ec1ede8ef09e55560789e84782bd528107dfa50a`.

## Next inspected diagnostic: complete generation routine

Use a new immutable export/root `native-rl-replay-diagnostic-v2`, retaining the
same four selected inputs, two repetitions, two workers and original modes.
Add `full-generator-forced`: call the actual native `generate_turn` routine,
including its cache-generation wrapper, decode/validation loop and probability
capture. A scoped diagnostic sampler hook consumes/discards normal draws but
forces the original token sequence. It does not modify deployed sampling,
produce new environment rollouts, or change rewards/weights. Require exact
prompt and generated-token coverage. This mode records the actual generation
routine's probability trace, not fabricated full-logit hashes; absent raw-logit
measurements are explicitly null.

This is a separate inspected experiment, not a failed-worker retry. Original
thresholds and failed artifacts remain unchanged. The new recipe binds all four
modes; incompatible recipe/source combinations fail closed. The same 1,800-second
worker and 2,100-second outer limits apply. `rl_optimizer_ready:false` remains
unchanged. Results for v2 are pending.
