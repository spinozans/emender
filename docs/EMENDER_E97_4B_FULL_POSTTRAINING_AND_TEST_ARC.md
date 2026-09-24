# Emender E97 4B full post-training and test arc

**Status:** governing end-to-end execution map

**Date:** 2026-09-08

This document consolidates the complete path from the trusted pretrained/action
parent to an okay conversational and action agent. Detailed protocol and task-lake
documents remain normative for their respective subsystems; this document defines
the order, gates, decision branches, and intended evidence across the whole program.

## Objective and current hypothesis

The 4B E97 parent is hypothesized to contain enough pretrained knowledge to become
an okay general conversational assistant and useful typed-tool agent. Its current
behavior shows strong protocol retention but insufficient generalization:

- trusted parent: train-`y` checkpoint `aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39`;
- core: 120/120;
- compositional: 236/240;
- fresh correction/recovery V4: 0/128.

The working explanation is that the model knows the action grammar but has not
received a sufficiently large and diverse post-training curriculum of complete
conversation, reasoning, action, observation, recovery, and final-answer
trajectories. Eight-update correction or balanced branches are numerical and
retention probes, not a path to general capability.

The proposed capability arc is:

> broad sustained SFT -> behavioral representation/checkpoint selection ->
> verified on-policy acquisition -> rejection-sampling and DAgger-style
> consolidation -> carefully gated RLVR/GRPO -> portable recurrent-state and
> release qualification.

No stage advances solely because loss decreases.

## Phase 0: close data and runtime foundations

### Work

1. Finish the bounded-private-analysis Open-SWE authority.
2. Build its matched action-only counterpart from identical trajectory identities.
3. Validate immutable inputs and outputs, licenses, repository holdouts, source
   filters, trajectory receipts, complete 64K units, masks, target counts, reset
   boundaries, gradient isolation, and padding invariance.
4. Complete fused-CUDA token-delta versus full-transcript replay qualification.
5. Preserve legacy Pi-v2 bytes and runtime identity unchanged when private analysis
   is disabled.

### Current evidence

- Installed Pi completed eight streamed analysis/action/tool-observation turns and
  one final turn with exact `reasoning_content` replay.
- The provisional analysis envelope is 2,048 `p50k_base` tokens and 65,536 UTF-8
  bytes per emitted analysis.
- Oversize analysis excludes the complete trajectory; no truncation is permitted.
- The raw individual-message cap affects 60 of 14,314 legacy-action-normalizable
  trajectories; strict emitted-turn conversion excludes 70 after folding internal
  `think` reasoning into the following executable action.

### Gate

No GPU behavioral training until both representations validate, the Pi/server/
controller path is exact, and fused-CUDA recurrence is numerically qualified.

## Phase 1: select the trajectory representation

Run controlled, predeclared experimental branches from the same trusted parent:

- A: action-only;
- B: bounded private analysis plus actions;
- C, optional: concise verified rationale plus actions.

All branches must use:

- the same unique trajectory identities;
- matched consumed assistant-target scale;
- the same optimizer family, LR schedule, checkpoint cadence, and evaluation points;
- frozen development holdouts;
- explicit train-`y` and saved-`x` evaluation.

Representation selection uses unseen behavior:

- verified completion rate;
- action validity;
- use of observations;
- recovery after authentic errors;
- repeated-action cycle rate;
- long-horizon consistency;
- ordinary conversation and instruction quality;
- retention;
- inference token cost.

Select the simplest representation with a meaningful behavioral advantage. If
private analysis does not help, use action-only. If it helps but costs too many
tokens, evaluate the concise-rationale arm.

## Phase 2: freeze the serious SFT corpus and recipe

Build one sealed 50--100M assistant-target-token stage containing at least 10,000
diverse complete trajectories.

Provisional mixture envelope:

| Component | Share |
|---|---:|
| Broad conversation and ordinary instruction | 30--40% |
| Successful repository/action trajectories | 25--35% |
| Document reading and grounded retrieval | 10--20% |
| Reasoning and compositional work | 10--20% |
| Authentic failed-state corrections | at most 5--10% |

Before launch, freeze:

- authority and source hashes;
- selected representation and exact system prompt;
- tool schema;
- source, repository, issue, language, and family caps;
- without-replacement sampling identity and order;
- target-token horizon;
- LR schedule and optimizer configuration;
- evaluation milestones;
- regression floors and stop rules;
- independent V5 holdout identities.

Consumed V3/V4 data, traces, paths, values, task derivatives, and teacher context
remain prohibited.

## Phase 3: numerical canary

Run one eight-update canary only to verify:

- eight-GPU DDP;
- training on Schedule-Free `y`;
- saving averaged `x`;
- exact resume and optimizer state;
- boundary-aware complete-record packing;
- target-token accounting;
- both expandable-segment allocator variables;
- no redundant outer DiLoCo merge under full-world DDP;
- atomic checkpoint publication and mmap reload.

The canary is systems evidence and cannot be selected behaviorally.

## Phase 4: sustained broad SFT

Start from trusted train-`y` parent `aae654aa...` and execute the frozen 50--100M
target stage continuously. Save resumable checkpoints every eight updates, but do
not reinterpret each save as a new behavioral experiment.

Initial behavioral milestones:

- 1M consumed assistant targets;
- 5M;
- 10M;
- 25M;
- 50M;
- terminal or the declared upper horizon.

Every evaluation must bind `WEIGHT_MODE=train` or `WEIGHT_MODE=saved` explicitly.
The former evaluates live Schedule-Free `y`; the latter evaluates averaged saved
`x`. No mixture, objective, LR, horizon, or stop-rule changes are allowed mid-run.

## Phase 5: behavioral checkpoint selection

### Conversation and instruction

- ordinary multi-turn conversation;
- instruction and constraint following;
- formatting control;
- clarification when information is missing;
- concise versus detailed response control;
- honest uncertainty and blockers.

### Documents and knowledge

- long-document comprehension;
- grounded extraction, comparison, and synthesis;
- multi-file reading;
- non-overlapping language-model loss;
- citation and evidence fidelity.

### Agent behavior

- unseen typed-tool tasks;
- real-repository tasks;
- long action sequences;
- observation-conditioned replanning;
- stale-path and missing-file recovery;
- test-driven repair;
- one-shot trusted-controller `no_progress` recovery;
- invalid-tool and malformed-argument rejection.

### Retention and safety

- mandatory parent 120/120 core preflight;
- 240-task compositional panel;
- independent V5;
- broad conversation/instruction retention;
- destructive-action restraint and grounded blockers;
- repeated canonical-action cycle rate;
- paired `x/y` divergence.

A multi-objective behavioral rule selects checkpoints. Loss, perplexity, schema
validity, terminal position, or core score alone cannot select a model.

## Phase 6: verified on-policy acquisition

Only after sustained SFT produces regular executable success:

1. Run eight independent rollout actors, one per GPU.
2. Draw fresh content-addressed tasks under atomic leases.
3. Preserve complete model turns and authentic observations/errors.
4. Replay every claimed success against fresh fixtures and deterministic validators.
5. Separate verified successes, useful failed student states, and invalid traces.
6. Stop acquisition cleanly before switching all eight GPUs to DDP training.

Rollout receipts bind the model/checkpoint, prompt, tools, runtime image, limits,
completion usage, observations, actions, validators, and final outcome.

## Phase 7: rejection sampling and DAgger-style correction

Use the on-policy lake to:

- train on verified successful trajectories;
- retain exact failed student prefixes as zero-loss context;
- target only replay-verified corrective suffixes;
- aggregate corrections across diverse identities;
- prevent small task or correction families from dominating.

Then run a frozen consolidation stage mixing:

- new verified successes;
- authentic student-state corrections;
- broad conversation replay;
- document and reasoning replay;
- core/compositional agent retention.

This stage is the primary hypothesis for moving from an okay imitator to a more
robust action agent.

## Phase 8: verifier and reward readiness

Build calibrated outcome/process verifiers only from executable evidence:

- deterministic tests and validators;
- workspace diffs and content hashes;
- grounded final claims;
- action/progress receipts;
- exact replay;
- no-progress classification.

Test them against false success, stale fixtures, reward hacking, forged receipts,
and superficially plausible but ungrounded finals.

If the SFT model does not already achieve meaningful on-policy success, return to
broader/diverser SFT. RL must not be used to install absent basic capability.

## Phase 9: RLVR/GRPO

Only after success-frequency and verifier gates pass:

- use tasks with robust deterministic rewards;
- retain KL/reference anchoring;
- include broad replay or equivalent retention control;
- bound rollout length, completion tokens, and tools;
- monitor reward hacking, cycles, and conversational regressions;
- evaluate explicit `x/y` behavior at declared milestones.

RL is intended to improve reliability, recovery, and efficient action selection,
not replace foundational supervised learning.

## Phase 10: final systems and release qualification

For the selected model:

- repeat the full broad behavioral battery;
- run the frozen real-repository holdout;
- complete safety and destructive-action review;
- prove exact checkpoint resume;
- save, restore, clone, transport, and fork recurrent state;
- implement and test the token-delta session API;
- run the 100-file kill/restore/fork demonstration;
- compare fixed recurrent state with Transformer KV-cache growth;
- validate state privacy, permissions, checksums, and revision compatibility;
- package immutable checkpoint and runtime identities for release.

## Decision branches

- **Conversation remains weak:** increase broad conversation/instruction diversity,
  not toy corrections.
- **Tool syntax is correct but tasks fail:** increase complete repository,
  document, planning, and recovery trajectories.
- **Private analysis gives no gain:** select action-only.
- **Private analysis helps but is inefficient:** test concise verified rationale.
- **Core or compositional retention falls:** stop and revise the frozen mixture/LR
  in a new stage identity.
- **Saved `x` and train `y` diverge materially:** reject promotion until understood.
- **On-policy success remains rare:** return to SFT; do not begin RL.
- **Verifiers are exploitable:** strengthen deterministic validation before RL.
- **A client drops or changes reasoning:** analysis protocol is unsupported for
  that client and fails closed.

## Non-paths

The following are not accepted capability strategies:

- 8/16/32/64-update behavioral checkpoint pinball;
- repeatedly oversampling the 128 toy corrections;
- correction-only training without broad replay;
- selecting by training loss;
- supervising isolated actions without observations and finals;
- silently truncating trajectories or reasoning;
- using consumed panels as training or teacher context;
- starting RL before regular verified success.

## Related authorities

- `docs/EMENDER_E97_4B_SYSTEMATIC_AGENT_POSTTRAINING_REGIME.md`
- `docs/E97_BOUNDED_PRIVATE_ANALYSIS_PROTOCOL.md`
- `docs/EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md`
- `docs/EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md`
- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`
- `docs/validation/e97-private-analysis-runtime-and-authority-progress.md`
- `docs/validation/e97-open-swe-private-analysis-cap-audit.md`
