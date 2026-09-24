# E97 4B systematic agent post-training regime

**Status:** governing successor to micro-update behavioral steering

**Decision date:** 2026-09-08

**Reference parent:** `aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39`

The consolidated phase order, evaluation battery, gates, and decision branches
are defined in
[`EMENDER_E97_4B_FULL_POSTTRAINING_AND_TEST_ARC.md`](EMENDER_E97_4B_FULL_POSTTRAINING_AND_TEST_ARC.md).
This regime remains the detailed authority for the sustained SFT stage.

## Decision

Stop treating eight-update checkpoints as independent behavioral interventions.
Eight-update checkpoints exist for numerical qualification, exact resume, and
recovery only. Mixture weights, objectives, and learning-rate schedules are
frozen before a serious stage and do not change in response to individual
checkpoint scores.

The correction-only u64 branch and balanced q8 branch remain experimental,
permanently non-promotable diagnostics. Neither is a new parent. The interrupted
balanced u16 attempt performed no optimizer update and produced no checkpoint.

## Training target

Build one generally useful chatbot and terminal/document agent rather than a
collection of benchmark-specific policies. The deployment distribution includes:

- ordinary conversation, instruction following, writing, and clarification;
- code and document comprehension;
- grounded navigation and retrieval through typed tools;
- multi-step repository inspection, editing, testing, and diagnosis;
- authentic failures, stale information, empty results, timeouts, and recovery;
- long-horizon completion, evidence-based final answers, and safe abstention.

Runtime enforcement remains the security and protocol boundary. Training teaches
the policy to operate within that boundary; it never replaces typed schemas,
path confinement, deadlines, output limits, validator execution, or no-progress
termination.

## Corpus gate before training

No serious stage begins until its complete immutable corpus, mixture, exclusions,
and held-outs are sealed. The corpus must contain complete trajectories and be
reported by unique task/trajectory identities as well as assistant-target tokens.
Repeated copies of a tiny correction set do not count as diversity.

Initial scale target:

- 50--100 million consumed assistant-target tokens;
- at least 10,000 diverse complete agent trajectories;
- thousands of independently generated executable tasks;
- multiple repositories, languages, document formats, directory layouts,
  failure conditions, action orders, and user phrasings;
- source/task caps that prevent easy templates or repositories from dominating.

Current useful inventory includes:

- broad Pi/Tulu instruction authority: about 46.0M assistant targets;
- compositional retention authority: about 7.4M assistant targets;
- live-aligned authority: about 2.4M assistant targets;
- verified Open-SWE Pi-v2 authority: 14,314 successful trajectories, 747,117
  assistant actions, and about 110.3M assistant targets.

The Open-SWE conversion dropped about 307.7M source reasoning characters,
corresponding to 80,894,626 p50k tokens. Before using it as the main agent
curriculum, compare the action-only representation against the versioned
bounded private-analysis representation specified in
[`E97_BOUNDED_PRIVATE_ANALYSIS_PROTOCOL.md`](E97_BOUNDED_PRIVATE_ANALYSIS_PROTOCOL.md).
Never place hidden reasoning inside tool JSON or expose it as a user-visible
final answer by accident.

The private-analysis transport gate now passes an installed-Pi CPU probe with
eight streamed tool turns plus a final turn. The cap audit selects a provisional
2,048-token/65,536-byte envelope and excludes complete trajectories rather than
truncating. The individual-message audit affects 60 of 14,314 legacy-action-normalizable
trajectories; the strict full converter excludes 70 after folding `think` reasoning
into the next executable turn. Candidate matched action-only and private-analysis
64K converters share that admission gate
and remain `training_eligible: false` until their full outputs validate. See
`docs/validation/e97-open-swe-private-analysis-cap-audit.md`.

The 128-record failed-state correction authority has only 5,953 unique assistant
targets. It may be a capped diagnostic component, but it cannot be the foundation
of the serious corpus. New corrections must come from fresh task identities and
verified student states.

## Assistant and tool boundary

Each training trajectory preserves the exact runtime serialization:

1. system and user messages are context;
2. assistant planning, when supported by a dedicated private channel, is an
   assistant target with an explicit bounded representation;
3. each assistant action is exactly one schema-valid typed tool call;
4. authentic tool observations and errors are zero-loss context;
5. subsequent assistant actions condition on those observations;
6. the final assistant answer is grounded in observed evidence or states an
   honest blocker.

Independent trajectories reset every recurrent layer. Cross-record targets are
masked, gradients cannot cross document boundaries, and padding is recurrently
inert. Complete logical units are never silently truncated.

## Fixed initial SFT stage

The final percentages are selected once from source audit and small numeric/LR
qualification, then frozen. A starting design envelope is:

- 30--40% broad conversation and instruction;
- 25--35% successful terminal/repository trajectories;
- 10--20% document and grounded retrieval trajectories;
- 10--20% reasoning and compositional tasks;
- at most 5--10% authentic failure/recovery corrections, capped by unique source
  identity and reduced when diversity is inadequate.

Use the trusted parent train-`y` representation with a fresh Schedule-Free stage.
Select a conservative learning-rate schedule through a predeclared numerical
scan, not behavioral hill climbing. Run the selected schedule continuously for
its declared token horizon. Save exact resumable checkpoints every eight updates,
but do not reinterpret every save as a new experiment.

## Evaluation cadence and selection

Before launch, freeze independent development and regression suites by user,
repository, issue, generator family, template family, seed, path, value, URL, and
content hash. Consumed V3/V4 remain diagnostic-only and are never training or
selection targets.

Evaluate explicit saved `x` and train `y` at meaningful consumed-target
milestones, initially approximately 1M, 5M, 10M, 25M, 50M, and terminal targets.
The eight-update save cadence is not the behavioral-evaluation cadence.

Selection requires a multi-objective score covering:

- broad conversation and instruction following;
- document comprehension and grounded retrieval;
- compositional core behavior;
- executable terminal/repository success;
- unseen-family and unseen-repository generalization;
- recovery, cycle rate, efficiency, and repeated-trial reliability;
- safety, clarification, and appropriate abstention;
- non-overlapping language/code loss.

Predeclare regression floors and stop conditions. Do not change the training
mixture to chase one panel while a run is active.

## On-policy improvement after SFT

After the fixed SFT stage reaches a nontrivial executable success rate:

1. sample multiple trajectories per fresh task;
2. retain only deterministically replayed successes and useful verified
   corrections;
3. train a trajectory verifier on disjoint data;
4. aggregate a large new dataset rather than immediately updating on each
   failure;
5. perform a named consolidation SFT stage with broad replay;
6. use GRPO/RLVR only where rewards are executable and resistant to hacking.

RL uses outcome success as the dominant reward, with protocol, grounding,
safety, and bounded-efficiency terms. It retains a frozen/reference policy and
KL or equivalent replay protection. Sparse zero-success task families first need
teacher/rejection-sampling bootstrapping; they are not useful RL curricula.

## Stage governance

A release cycle is:

```text
freeze interface and data recipe
  -> build and audit corpus
  -> numerical/LR qualification
  -> one sustained SFT stage
  -> milestone evaluation
  -> large-batch on-policy acquisition
  -> verified consolidation
  -> executable-feedback RL where justified
  -> broad final alignment and independent holdout
```

Changing the interface, objective, mixture, or target distribution starts a new
named stage. It does not produce an ad hoc eight-update continuation of the
current checkpoint.
