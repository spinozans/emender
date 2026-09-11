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

## Published data and initial segment

The clean-export build (`proc_d5e2`) passed in 88 seconds, following 23 CPU tests.
Authority: `/mnt/nvme2n1/erikg/sft/e97-4b-native-training-50m-agent-mix-v1`,
SHA `44aa42389a3b9a2674b06e3a1586ada56c959c0d01fddf5ff8b62b03627619a5`.
It contains 4,216 unique native trajectories / 50,007,466 targets;
106,637 unique conversation records / 100,000,696 targets; and 104,117
retention records / 16,666,680 targets (39,604 unique, 64,513 replayed).

The exact 880-update schedule consumes **50,069,457 native / 100,128,589
conversation / 16,687,692 retention targets**, or **166,885,738 assistant
targets and 379,323,286 input tokens**. Its 215,246 record occurrences cover
214,970 distinct emitted records; 276 extra occurrences are K-alignment replay.
Schedule SHA: `92cf52f06d8b6b50da81f1c5e47481293d822db8f5204482e3f9f9a7341d99e8`.
Pack SHA: `dc49350c6f8c752f08409953ae0b1de4a8c7d5dbd0e07cf2dd1f01b0f0f16b04`.

`configs/pi/e97-native-training-50m-agent-program-v1.json` freezes endpoints
128, 256, 384, 512, 640, 768, 880. Each segment has a four-hour deadline,
30-second kill grace, separate logs/caches and a fresh fixed-world process set.
All segments share one checkpoint directory/atomic latest pointer and retain
up to 32 checkpoints (more than the seven planned saves). The numerical
trainer remains the immutable b8ee034f export; the new controller renders only
data, LR, budget, isolated paths, checked lease acquisition and the collector.
A failed segment is never automatically retried.

Eight new fitting examples are frozen from **planned first-128 consumption**
before training, then checked against actual runtime IDs. The 24 development/
retention example payloads remain unchanged from the LR screen. Separate x/y
evaluations precede any continuation: both need tool-retention token accuracy
at least 98%, conversation NLL no more than parent +0.15, and native development
NLL no worse than parent. These predeclared non-catastrophic guards are not
capability acceptance. Report generation separately; no tiny two-example task
success gate or forced source-argument imitation. An accepted evaluation receipt
is required before preparing a later successful segment from committed latest.

### Update 128 completed

`proc_8b96` completed successfully in 9,742 seconds (2h42m), under controller
commit `fd39b5b4f961e5c721632b05b5ff3ef5e9afa435`; its clean-export CPU suite
passed 25 tests. Evidence is retained under
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-training-50m-agent-lr1e5-v1/segment-000128`.

- 54,944,287 input tokens and 24,220,503 assistant targets.
- Native 7,239,263; conversation 14,543,784; retention 2,437,456 targets.
- All 128 runtime sample-ID sets, per-step counts and cumulative clocks matched
  the frozen schedule. All losses/gradient norms were finite.
- Rank-0 peak allocated HBM: 36,128,739,328 bytes (33.65 GiB), not an all-rank
  peak measurement.
- Complete finite BF16 model/optimizer/live-y state, optimizer and sampler
  clocks, precision/LR/data identities and atomic latest pointer verified.
- Shared checkpoint: `checkpoints/checkpoint_agent_sft_u000128_loss_1.2650.pt`;
  SHA `e7ec5cd9dd05976006d9f421697f21ecb000fdf8f15448d6c40cc05510cf15f9`.
  The filename loss is the last-100-update mean, not final-step or evaluation loss.
- No checkpoint promotion. The full 880 updates are not completed.

### Evaluation runner

`scripts/run_e97_native_training_segment_eval.sh` prepares a four-model panel:
parent y, the fixed 1e-5/u8 pilot y, current y, and current x. It uses the
unchanged numerical evaluator export, eight GPU lanes, NUMA placement, isolated
Triton caches, both allocator variables, checked GPU lease acquisition and a
2,400-second TERM/30-second KILL deadline. It does not run concurrently with the
training segment. `eval_e97_native_training_segment.py` checks the fixed policy
and publishes a gate bound to the training summary, program, panel and evaluation
summary. A rejected guard publishes `continue_training:false` and exits nonzero;
there is no fallback. The clean-export evaluation/controller tests passed 13
cases. Update-128 evaluation (`proc_8984`) completed in 718 seconds, from
controller commit `084ce6ccf2dcaafc233986bfc7e28f268b24f6de`.

| Metric | Parent y | 1e-5/u8 pilot y | u128 y | u128 x |
|---|---:|---:|---:|---:|
| Native fitting NLL (new, actually consumed panel) | 1.6603 | 1.1513 | .9025 | .9072 |
| Native development NLL (unchanged panel) | 1.5551 | 1.1177 | .9614 | .9626 |
| Conversation NLL | 1.6811 | 1.6773 | 1.6529 | 1.6505 |
| Tool-retention token accuracy | 100% | 100% | 100% | 100% |
| Valid fitting first turns | 0/2 | 1/2 | 2/2 | 2/2 |
| Valid development first turns | 0/2 | 0/2 | 2/2 | 2/2 |
| Matching development argument payloads | 0/2 | 0/2 | 2/2 | 2/2 |

Both u128 representations also matched fitting source tool names and full
argument payloads on 2/2 prompts. These are four distinct generation prompts,
measured in both modes—not eight independent tasks. No generated tools were
dispatched. This is stronger first-turn acquisition evidence, not task completion
or observation-dependent execution. All six predeclared continuation checks
passed; `continue_training:true`, `checkpoint_promotion:false`.

Evaluation summary SHA:
`7b825c11e9aaaf2db38874b8b2aa04af80e2944112b866c661eeeebf3597cfe1`.
Panel SHA: `116376b79d6db27621d0ac3640c49bed0954c3f44bf85305550f0a0ea5776fca`.
### Update 256 completed

The next planned segment (`proc_2381`) completed in 10,403 seconds, using the
unchanged frozen controller and committed atomic latest checkpoint. Its 128
additional updates passed exact runtime sample/count/cumulative-clock checks.
Cumulative exposure: 110,402,587 input tokens; 48,577,590 assistant targets,
comprising native 14,421,967, conversation 29,264,439 and retention 4,891,184.
Rank-0 peak allocated HBM remained 36,128,739,328 bytes.

Complete finite BF16 checkpoint:
`checkpoints/checkpoint_agent_sft_u000256_loss_1.2355.pt`, SHA
`66fc0ee107c20d6869d0827ce40a9eb1ee381c778c4da75929560a65c3291c27`.
The filename reports the last-100-update mean, not evaluation loss. This proves
successful continued training, not the separately unmeasured numerical
fresh-continuation comparison. No checkpoint is promoted.

Update-256 evaluation (`proc_f593`) completed successfully in 657 seconds,
using the unchanged frozen evaluation source and example payloads. All six
continuation checks passed; no checkpoint promotion.

| Metric | u128 y | u256 y | u256 x |
|---|---:|---:|---:|
| Native fitting NLL | .9025 | .8559 | .8584 |
| Native development NLL | .9614 | .9281 | .9307 |
| Conversation NLL | 1.6529 | 1.6096 | 1.6151 |
| Tool-retention token accuracy | 100% | 100% | 100% |
| Valid / matching full first tool calls, fitting | 2/2 | 2/2 | 2/2 |
| Valid / matching full first tool calls, development | 2/2 | 2/2 | 2/2 |

These are the same four distinct generation prompts, not additional independent
tasks. No tools were dispatched, and no observation-dependent task-completion
claim follows. Evaluation summary SHA:
`cc4d779f615231d583b97e6f8dbb1041ff9df10125b8fe7d85ef897066102c90`;
panel SHA `0e12850f6e316e6dcde787cfaa47e9b26b94a87c9e0834fa062f14ab8e460fc2`.
### Update 384 completed

The next planned segment (`proc_2b1d`) completed in 9,983 seconds, from the
accepted atomic u256 checkpoint with the unchanged frozen training recipe.
All 128 additional updates passed runtime sample/count/cumulative-clock checks.
Cumulative exposure: 165,177,221 input tokens and 72,793,317 assistant targets:
native 21,606,255, conversation 43,835,840 and retention 7,351,222.
Rank-0 peak allocated HBM remained 36,128,739,328 bytes.

Complete finite BF16 checkpoint:
`checkpoints/checkpoint_agent_sft_u000384_loss_1.2184.pt`, SHA
`f50814da17fc0480abc4fa231008f6f56d5c543b1aa537bfdac8d55018c5ef26`.
The filename loss is the last-100-update mean, not an evaluation metric.
No checkpoint promotion or numerical fresh-continuation qualification is claimed.

Update-384 evaluation (`proc_1fab`) completed in 671 seconds, on the same
frozen examples and numerical evaluator. All six continuation checks passed;
no checkpoint promotion.

| Metric | u256 y | u384 y | u384 x |
|---|---:|---:|---:|
| Native fitting NLL | .8559 | .8354 | .8361 |
| Native development NLL | .9281 | .9099 | .9117 |
| Conversation NLL | 1.6096 | 1.5890 | 1.5932 |
| Tool-retention token accuracy | 100% | 100% | 100% |
| Valid / matching full first tool calls, fitting | 2/2 | 2/2 | 2/2 |
| Valid / matching full first tool calls, development | 2/2 | 2/2 | 2/2 |

Generation still uses the same four distinct prompts, with no tool dispatch
or task-completion claim. Evaluation summary SHA:
`f7dff0c7af86a7d911909333533a49f822ea43e1b680c46e3beadcf3aad150af`;
panel SHA `4a3947ed3a9f445b0be35dd5785ee014890c282d9e681840e103a7e22fec1444`.
### Update 512 completed

The next planned segment (`proc_47cd`) completed in 10,086 seconds, from the
accepted atomic u384 checkpoint under the unchanged frozen recipe. All 128
additional updates passed runtime sample/count/cumulative-clock checks.
Cumulative exposure: 221,068,760 input tokens and 97,589,057 assistant targets:
native 28,818,594, conversation 58,879,995 and retention 9,890,468.
Rank-0 peak allocated HBM remained 36,128,739,328 bytes.

Complete finite BF16 checkpoint:
`checkpoints/checkpoint_agent_sft_u000512_loss_1.2005.pt`, SHA
`18725f77bf23a43f41d362edbbd949658d3ca27ad0b1aad7b7b1da3c963777d6`.
The filename loss is the last-100-update mean, not an evaluation metric.
No checkpoint promotion or numerical fresh-continuation qualification is claimed.

Update-512 evaluation (`proc_5cf8`) completed in 655 seconds, on the unchanged
frozen examples and numerical evaluator. All six continuation checks passed;
no checkpoint promotion.

| Metric | u384 y | u512 y | u512 x |
|---|---:|---:|---:|
| Native fitting NLL | .8354 | .8185 | .8214 |
| Native development NLL | .9099 | .8989 | .9004 |
| Conversation NLL | 1.5890 | 1.5648 | 1.5713 |
| Tool-retention token accuracy | 100% | 100% | 100% |
| Valid / matching full first tool calls, fitting | 2/2 | 2/2 | 2/2 |
| Valid / matching full first tool calls, development | 2/2 | 2/2 | 2/2 |

The same four distinct generation prompts remain correct at the first-call
level; no tool dispatch or task-completion claim is made. Development NLL
continues improving with smaller successive gains on this repeatedly measured
panel. Evaluation summary SHA:
`3c6ec5bead6366ede80b31c5eb0991fc02b1d7e1018a9446ca9953194d3917a5`;
panel SHA `9a162e1282da6437a932cfd1d7c8b6bec9ddcc2a3586cbe2c17bfd62ac2a3653`.
### Update 640 completed

The next planned segment (`proc_21fe`) completed in 10,059 seconds, from the
accepted atomic u512 checkpoint under the unchanged frozen recipe. All 128
additional updates passed runtime sample/count/cumulative-clock checks.
Cumulative exposure: 275,791,351 input tokens and 121,788,130 assistant targets:
native 36,034,395, conversation 73,445,989 and retention 12,307,746.
Rank-0 peak allocated HBM remained 36,128,739,328 bytes.

Complete finite BF16 checkpoint:
`checkpoints/checkpoint_agent_sft_u000640_loss_1.1906.pt`, SHA
`9f914dfa643c9a675c5d340a89de63417824b5359b8be040c9a52ef84e559662`.
The filename loss is the last-100-update mean, not an evaluation metric.
No checkpoint promotion or numerical fresh-continuation qualification is claimed.

Update-640 evaluation was launched as `proc_b8ca`, on the unchanged frozen
examples and numerical evaluator. Its results and continuation to update 768
remain pending.

The earlier exact-restoration evidence remains valid; numerical fresh
continuation has not been measured and is not relabeled as passing.

ADR-003 safety intent: R07/R12 committed atomic checkpoint/restoration,
R14/NDP13 bounded failed-child termination, R16 evidence identity discipline,
and NDP15 synchronous atomic publication. No elastic/native data-plane,
async/overlap, Frontier/ROCm, additional-node scale or communicator-shrink claim.
