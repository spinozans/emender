# Native RL probability/masking candidate qualification v1

## Why this next step

The [on-policy canary](e97-native-onpolicy-canary-v1.md) completed 32 real rollouts
and 16 verified repairs, but had no within-task reward contrast. That batch is
not suitable for a group-centered outcome update. Separately, an on-policy
optimizer must not interpret numerical differences between actor and trainer
as policy improvement. This is a **no-update numerical assay**, not a relabeled
RL training run or an expansion of the previous 32-update SFT budget.

## Frozen assay

Source: every generated turn from sample 0 of all 16 training-only tasks in
`native-onpolicy-canary-v2`. No selection by success, loss or probability.
Input summary SHA:
`8278969f8ff65afd6f49251abb8a05c758ef6586c748824d3cdbc8345e187e6d`.
The same u32 correction-y checkpoint is used:
`48dffaf72900419a6481bae05bfd149710627f4bf684a152d7d17804cf55b3e6`.

Before GPU execution, freeze prompt/sample IDs and recorded actor probabilities.
Reject rather than filter turns exceeding a 16,384-token prefix or 512-token
sample; at most 128 turns. For each:

1. Replay the exact actor prefix plus forced sampled tokens with the actor's
   original segmented-prefill/tokenwise-continuation path.
2. Compute its generated-token probabilities under a training-mode, padded,
   masked, single-turn layout: BF16 parameters, group3/MLP4096, checkpointed
   FP32 CE128, 128-token alignment, explicit valid/reset/assistant masks.
3. Capture actual LM-head chunk outputs and verify selected log probabilities
   agree with the scalar masked CE. Prompt/observation predictions and padding
   receive no direct loss.

Predeclared limits, with **no fallback or retrospective relaxation**:

| Comparison | Maximum |
|---|---:|
| Recorded actor vs exact actor replay, absolute log-probability delta | .0001 |
| Recorded actor vs training-layout, maximum absolute delta | .05 |
| Recorded actor vs training-layout, p99 absolute delta | .02 |
| Mean CE vs captured head-token NLL, absolute delta | .0001 |

The .05 maximum corresponds to a probability ratio within approximately
.951–1.051, materially narrower than a candidate .8–1.2 clipping interval. All
measured values must be finite and model parameter hashes must be unchanged.
This bounds this assay only: it does not qualify packed 64K RL training,
distributed normalization, optimizer updates or numerical fresh continuation.

## Candidate loss arithmetic

`ndm/e97_outcome_rl_candidate.py` implements complete-task-group normalized
advantages and a clipped token surrogate with k3 reference KL. Constant-reward
groups have zero outcome advantage. Behavior/reference/reward tensors are
explicitly detached. Only masked model-generated predictions enter the direct
loss; conditioning through the model context is not prohibited.

Normalization uses complete-episode generated-token counts and a supplied global
number of distinct episodes. Turn/chunk contributions must sum; they must not
be independently averaged. This helper does **not** yet implement DDP scaling.
CPU gradient tests cover mask exclusion, frozen inputs, clipping direction,
constant-reward zero gradients, whole-episode chunk normalization, overflow
rejection and exact shifted/padded target-mask alignment.

## Execution and remaining gates

One checked leased GPU, explicit local-rank device, NUMA placement, isolated
Triton cache, both expandable allocator variables, 2,400-second GPU deadline
and no child restart. All parameters remain BF16 and are hashed before/after;
FP32 intermediates are bounded logits/probability arithmetic, not master weights.
No Slurm, external services, registry admission, model promotion, optimizer or
resilient/distributed training qualification is involved.

Scripts:
- `scripts/qualify_e97_native_rl_logprobs.py`
- `scripts/run_e97_native_rl_logprob_qualification.sh`

Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-rl-logprob-qualification-v1`.
CPU validation: **71 passed**, including 11 candidate loss/alignment tests and
60 existing on-policy/native runtime tests. GPU result pending.
Even a probability-path pass leaves `rl_optimizer_ready:false`
until end-to-end RL gradients, distributed normalization, safe optimizer
integration and a task batch with usable reward contrast are qualified.
