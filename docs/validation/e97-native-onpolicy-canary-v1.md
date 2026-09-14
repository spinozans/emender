# Native on-policy collection and failed-state correction canary v1

**Subsequent programme decision:** the numerical exploration is documented in
[its index](e97-numerical-exploration-index.md). Current priority is verified data
and SFT/DAgger tuning; probability-path/RL qualification is not a blanket prerequisite
for supervised correction. See [current learning status](../E97_AGENT_LEARNING_STATUS.md).
All original canary verdicts and candidate eligibility below remain unchanged.

## Decision and readiness

Operator requested a continuous task/reward/correction loop after the 32-update
[grounding correction](e97-grounding-correction-v1.md) produced 7/16 autonomous
completions and 16/16 exact first calls, while missing its 4/8 fresh gate.
We are ready to test sampled rollouts and same-environment teacher repair, **not
to assert that a qualified RL optimizer already exists**. No public benchmark
is being repurposed as training data and no external environment is downloaded.

Recipe: `configs/pi/e97-native-onpolicy-canary-v1.json`.
Controller: `scripts/e97_native_onpolicy_canary.py`.
Launcher: `scripts/run_e97_native_onpolicy_canary.sh`.

## Frozen bounded experiment

- 16 newly generated **training-only** tasks, four each lookup, sum, edit,
  recovery; two independently seeded stochastic rollouts/task = 32 rollouts.
- New path namespace and answer values, rotated selector keys/order, edit
  increments 1/3/7/11. These remain same-family exercises, not generalization to
  new repository tasks. Current native minimal system/tool profile only; no Pi
  translation or supplied-read assistance.
- Fixed checkpoint u32 correction y, SHA
  `48dffaf72900419a6481bae05bfd149710627f4bf684a152d7d17804cf55b3e6`.
- Temperature 1, no top-k/nucleus truncation. Record original prompt token IDs,
  sampled token IDs and selected FP32 log probabilities from actual actor
  logits. Greedy diagnostic behavior is unchanged outside this canary.
- Existing limits: eight turns, 4,096 generated tokens/turn, 8,192/episode,
  65,536 context tokens, 600-second generation deadline, bounded native RPCs.
- Teacher deadline 300 seconds; 16 authored failed-state preflight tasks with
  a 900-second overall deadline; actor command 5,400 seconds; whole sequence
  7,200 seconds. Deadline wrappers have a 30-second kill grace. No retries.
- Eight fixed inference actors, checked GPU lease, explicit local device,
  NUMA affinity, isolated Triton caches and both expandable allocator settings.
  No optimizer/collective training or resilient-compute qualification is claimed.
- **Zero optimizer updates.** Candidates remain `training_eligible:false`;
  no registry admission, model promotion, automatic expansion or RL success claim.

## Real-state corrections, not rewritten rollouts

The autonomous attempt is graded against a paused filesystem snapshot before
any teacher action. That reward is immutable: a teacher success never becomes
an autonomous success. Model-generated histories and failed final answers are
retained privately.

The teacher resumes the **same owned container**, without setup/reset, path
normalization or replacement of observations. An incorrect finish is replaced
only at its pre-finish decision boundary: finish has no external tool effect;
the rejected attempt remains in original episode evidence. Accepted earlier
model actions and authentic errors stay in the corrective context, with **zero
loss mask** on all model-prefix assistant turns. Only the verified teacher
suffix is supervised. Successful autonomous trajectories are separate candidate
records. Whole overlong/invalid records are not clipped into admissibility.

Inputs changed by the model and unresolved dispatch/protocol states are excluded
from repair. Teachers read actual inputs, verify sums using executed Python,
write the explicit destination with no-follow file opening, read back edits,
and traverse the actual recovery catalog. Outcome grading uses the original
answer and source-preservation requirements, plus the declared destination and
missing path. Source files and outputs are checked again with a separately
recorded paused-PID snapshot. No previous reader receipt is overwritten.

The host-side outcome oracle and immutable snapshot reader remain outside the
agent's editable environment. Sandboxes retain nonroot/no-network/no-GPU/no-host-
workspace restrictions and ownership-checked cleanup. Private model reasoning,
sampled histories and candidate payloads are not public report content.

## Gates and interpretation

Before sampling, all 16 authored failed-state teacher continuations must pass
in the real executor. This preflight is explicitly authored, not autonomous.

After collection, require complete 32-rollout coverage, exact policy/panel
identity, finite per-token log-probability coverage, preserved original rewards,
same-container repair identity, candidate hashes and cleanup receipts. Report
all failures. A useful first rollout signal requires at least one task group
with both reward 0 and reward 1, plus one verified repair of an actual model
failure. That small gate does not establish scaling or generalization.

`rl_optimizer_ready:false` remains mandatory: actor-versus-trainer likelihood
parity, assistant-only RL masks, probability-ratio/KL arithmetic, and optimizer
behavior need separate numerical qualification before policy-gradient updates.
Any subsequent training budget and data derivative must be frozen separately;
retain the existing conversation/native/tool-retention gates and never train
on the protected evaluation tasks. The near-term loop can use verified successes
and corrections while that RL path is qualified.

Initial CPU tests: **58 passed**, including exact suffix-only masks, immutable
autonomous failure despite teacher success, untruncated sampled-logprob capture,
generalized outcome paths and existing runtime/sandbox regressions.

## Inspected authored-preflight failure and correction

`proc_4959`, source `5d2852a7`, failed after 49 seconds during the third authored
teacher task, **before any model rollout or optimizer update**. Lookup and sum
preflights passed. The edit itself was correct in the recorded real observation,
but the read-back verifier compared JSON as text: publication had sorted the
oracle dictionary's keys while the executed program preserved input key order.
This was a teacher-verification bug, not a failed model attempt.

The failed root remains
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-onpolicy-canary-v1`,
including `failure-analysis.json`, all three agent cleanup receipts, the real
calls/observations and original source inventory (verified again after failure).
Panel SHA: `795ef4e090d6606ddc099c0f630b6e876c39833dc9ae8494263429ac1fea7be8`.

The inspected fix parses JSON from a complete, contiguous numbered editor view
and compares the resulting content with the original oracle. It rejects wrong
headers, missing/noncontiguous lines and truncation. It does not alter recorded
observations or relax task/outcome requirements. Regression coverage includes
key-order and multiline-whitespace differences and genuinely wrong counts.
**60 CPU tests passed**, and the corrected parser was checked against the actual
failed preflight observation.

A new immutable export/run is used, not an automatic child retry. Tasks, seed,
policy, budgets, reward and readiness criteria remain unchanged. Replacement
artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-onpolicy-canary-v2`.
## Completed corrected canary

`proc_4fce` completed in **521 seconds** from immutable source `75d72601`.
All 16 authored real-executor preflights passed, followed by all 32 stochastic
model rollouts. Source inventories verified before/after and 60 clean-export
CPU tests passed. Panel SHA is unchanged from v1: no tasks, rewards, limits or
criteria were changed to obtain a pass.

| Family | Autonomous successes | Verified failed-state repairs |
|---|---:|---:|
| Lookup | 0/8 | 8/8 |
| Sum | 8/8 | — |
| Edit | 0/8 | 8/8 |
| Recovery | 8/8 | — |
| **Total** | **16/32** | **16/16 failures** |

These are training-only tasks with stochastic sampling, not a new evaluation
score or a model improvement: **zero optimizer updates** occurred. Recovery
success here does not overturn earlier failure on differently phrased/path-
structured evaluation fixtures. All lookup finals matched none of the three
observed candidate values; that does not isolate selection from copying failure.
A representative edit used the correct destination, but copied the unchanged
count rather than applying the requested increment, then declared `done`.

Every failed rollout was repaired in the same container after actual accepted
model actions. The original autonomous reward stayed zero. There are 16 verified
autonomous-success candidates (2,718 assistant targets) and 16 teacher-only
suffix candidates (3,012 targets), still **not training eligible**. A separate
CPU audit reconstructed every candidate's full native serialization, BPE tokens
and masks exactly, including unsupervised failed model prefixes. All 48 agent
and 80 reader containers across authored preflight and sampled collection have
cleanup receipts.

**The rollout signal gate failed:** every task's two attempts had identical
binary reward, so there were **zero mixed-reward task groups**. With this batch,
group-centered outcome advantages would all be zero; aggregate 50% success does
not imply a useful GRPO task-group gradient. No threshold is relaxed.
`rollout_signal_ready:false`, `rl_optimizer_ready:false`, `automatic_expansion:false`
and `checkpoint_promotion:false` are preserved.

Next qualification is actor-versus-trainer log-probability consistency and
assistant-only policy-loss masking, without parameter updates. A later training
curriculum needs intermediate difficulty and/or better initialization from these
verified repairs to obtain within-task reward variation; repeating this exact
all-or-nothing batch is not an RL solution.

Evidence under the v2 root:
- Preflight summary: `cefc7d55c5aa2615a0ba4dbb13910f4f49811cde20b048133a92d4caf362f11c`.
- Rollout summary: `8278969f8ff65afd6f49251abb8a05c758ef6586c748824d3cdbc8345e187e6d`.
- Full candidate-mask audit: `5b6deacaf92f9ac7604048d808f338bf08fdee723479b4b731b90c8666418f79`.
