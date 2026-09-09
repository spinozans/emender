# E97 4B: sustained learning and three-axis validation

**Decision date:** 2026-09-09

**Status:** operator-approved program direction; data audit and experiment freeze next

**Planning horizon:** one to two weeks to seek useful results, not a capability guarantee

## Executive overview

Use a substantial, audited corpus to teach general instruction following,
reasoning-associated behavior, and observation-dependent tool use. Judge learning
from three complementary directions:

1. **Fitting:** can the model generate coherent decisions on consumed training
   examples, not merely score a supplied continuation more highly?
2. **Transfer:** do decision likelihood and unassisted generation improve on
   development examples excluded from the new training stage?
3. **Execution:** can the model act in a real environment, use authentic
   observations, recover, and finish a verified task?

Select an initial learning rate by a controlled sweep using decision-balanced
validation loss, subject to numerical and retention constraints. Then run a
sustained SFT stage, followed by verified on-policy corrections and, when rewards
become informative, outcome-based reinforcement learning. Private analysis is an
experimental representation, not the definition of agent competence.

An hour-long pilot is not a fair test of the complete program. Conversely, lower
average language-model loss is not enough to justify scaling an unchanged recipe
when its important decisions regress. Fix demonstrated engineering problems once,
then measure sustained learning rather than repeatedly steering tiny checkpoints.

This is a current operational addendum to the
[agent post-training research plan](EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md)
and [task-lake execution plan](EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md).
It supersedes earlier micro-update behavioral steering and provisional token
budgets where they conflict. It does not change runtime security, data-admission,
protected-holdout, or [fixed-world resilience](RESILIENT_DILOCO_COMPUTE_POOL.md)
authority. This documentation publication does not publish or qualify the local
implementation candidates.

## What data do we actually have?

**A billion usable agent-training tokens is not an established inventory.**
Large document/general-continuation authorities are not interchangeable with
verified agent demonstrations. A billion targets was a possible compute budget,
not a selected experiment.

The existing converted private-analysis Open-SWE candidate reports:

| Quantity | Count |
|---|---:|
| Included trajectory identities | 10,905 |
| Emitted records | 13,112 |
| Complete targeted assistant units | 558,858 |
| Input tokens | 564,969,738 |
| Assistant-target tokens | 151,602,474 |
| Raw private-analysis tokens before serialization | 59,507,270 |
| Train / validation records | 12,985 / 127 |

These are whole-candidate counts, including its validation split; 151.6M is not
an audited new-stage training-only budget, nor a count of unique decision tokens.
Its manifest explicitly says **`training_eligible: false`**. The new stage needs
an admitted derivative after semantic audit, split verification, and exclusions.
Any repetition must be disclosed separately from unique coverage.

Inventory authority:
`/mnt/nvme2n1/erikg/sft/e97-4b-open-swe-private-analysis-64k-v3/manifest.json`,
SHA-256 `248a02e6d977b83474eba6e48589a691e4fc36115602c931cfac4960b73dc196`.
The matched action-only candidate contains the same 10,905 trajectory identities
and approximately 87.3M assistant targets. Neither candidate is admitted merely
by inclusion in this plan.

The current high-LR u64 screen consumed 17,984,082 total targets but only
1,920,701 agent targets: 169 agent records from 169 trajectory identities and
6,949 analysis-response starts. Its agent-target share was 10.68%, not the
nominal full-stage 30%. The selected order began with pack IDs 849, 851, 853,
855: without replacement did not mean a well-shuffled, source-balanced prefix.
Earlier larger stages also exist; the entire project has not had only an hour
of post-training, and those heterogeneous stages are not a clean scaling study.

## First gate: semantic consistency, not just valid serialization

Use the 151.6M candidate as a proposed backbone, not as an unquestioned training
input. Code inspection identifies these mechanisms requiring measurement:

1. **Read bounds:** `action_from_call` defaults converted editor views to
   `limit=200`. This is a plausible source of the generated default, not causal
   proof. Explicit user bounds must override learned defaults.
2. **Action/observation equivalence:** actions are translated while observations
   retain normalized source-tool output. Verify that the destination tool, given
   those exact arguments and the actual environment, could produce the supplied
   evidence. Check ranges, truncation, directory views, ordering, errors, shell
   state, and path roots. A bounded read must not magically return a whole file.
3. **Causal context across windows:** `segment_complete_units` may restart with
   system/user plus at most the previous complete action/observation unit.
   Every emitted record resets recurrence. Preserving all action units does not
   establish that each later decision retains its necessary earlier evidence.
4. **Instruction consistency:** check replaced system prompts, retained user
   messages, tool schemas, and preserved reasoning against the translated
   actions. Byte/token parity alone cannot prove semantic agreement.
5. **Source quality and exposure:** upstream resolved status is not our own
   execution verification. Audit representative trajectories, exclusions,
   duplication, repository concentration, and prior training-lineage exposure.

Freeze the audit inputs and sampling rule before inspecting model outcomes. Run
mechanical checks over all records where possible, plus a bounded stratified
semantic/replay sample covering tools, repositories, lengths, and window starts.
Publish counts and examples for each risk, with denominators and an explicit
unverified category. Do not present a sampled audit as full replay qualification.

Do not silently repair the immutable candidate. Corrections or exclusions produce
a new version with receipts. Preserve complete logical units and authentic
observations. Context-dependent windows need adequate source-backed context,
separate valid treatment, or exclusion; invented summaries are not a repair.

Learning general agent behavior before final harness specialization is reasonable,
but the general stage must already describe a consistent tool world. Later
harness tuning cannot be relied on to repair contradictory demonstrations.

## Three-axis evaluation contract

| Axis | Inputs and intervention | Measurements | Claim allowed |
|---|---|---|---|
| Fitting | Fixed consumed training trajectories; teacher-forced scoring **and** separate unassisted generation | Decision losses, first-token ranks, exact argument fields, coherent turns | Whether the training setup fits its examples |
| Transfer | Frozen development examples excluded from the new stage; audit earlier lineage | Same scores, free-generation validity, repetition and termination | Transfer within the declared development scope |
| Execution | Fresh executable fixtures/repositories with real observations | Valid dispatches, argument fidelity, observation-dependent choices, verified task success and cost | Actual behavior in the tested environment |

The final independent holdout is separate from all sweep/CMA selection data.
Prior-lineage overlap must be reported; merely rehashing a split does not make it
independent. Consumed V3/V4 and the current diagnostic panels remain excluded
from training and teacher context. Exact-marker novelty is not family novelty.

For execution, include paired observation interventions: hold the instruction
fixed, change a relevant value or error condition in the environment, and require
the correct downstream action or answer to change. Include irrelevant changes
that should not change the decision. Increase delays, distractors, value updates,
and multi-step dependencies to exercise recurrent memory rather than memorized
scripts. Admit any new task sources through the existing authorization process.

Keep autonomous, supplied-header, and teacher-forced scores separate. Reasoning
that sounds correct is not itself evidence of correct reasoning or tool use.
Exact imitation scores are also not complete correctness judgments: alternative
valid plans must ultimately be evaluated by execution, not rejected solely for
being unlike the demonstration.

## The LR-sweep metric

Ordinary token-average loss can hide a catastrophic decision behind thousands of
easy continuation tokens. Use four separately reported decision components:

- **Opening:** entry into the requested response representation.
- **Choice:** tool versus final answer, and the selected tool.
- **Arguments:** paths, bounds, commands, values, and other typed fields.
- **Termination:** completion of the action/final unit and its required boundary.

Proposed primary metric:

```text
D(theta) = (L_opening + L_choice + L_arguments + L_termination) / 4
fitness  = D(parent) - D(candidate at the fixed training budget)
```

For each component, first normalize token NLL within its annotated span; for
arguments, average semantic fields rather than letting a long command dominate.
Then average eligible turns within each trajectory and average trajectories.
Report coverage for components that do not apply to a particular turn. The
component inventory, disjoint token-span rules, BPE boundary treatment, and
missing-component policy must be implemented, tested, and frozen before launch.
Score actual serialized target tokens, not isolated strings with artificial
trailing-space tokenization. This formula is a proposed selection contract, not
an already implemented or validated metric.

Score reasoning continuation separately, alongside conventional token-average
loss. It must not dominate the LR ranking. Every teacher-forced score must state
what preceding reasoning/actions/observations were supplied. In particular,
action likelihood after gold reasoning does not prove the model can generate
that reasoning or reach the action unaided.

Candidates are eligible for ranking only if they satisfy the frozen numerical
and retention rules. Report improvement per GPU-hour as an efficiency measure;
primary comparisons use equal consumed targets and the same data order. Also
report training-versus-development gaps, per-source results, first-token top-1
and rank, full-field exact matches, and free-generation failure rates. Do not
select on the scalar alone when behavior contradicts it; extend the comparison
or declare that no candidate qualifies rather than changing weights after seeing
results. Confirm shortlisted candidates with replicated runs before selection.

## Learning rates: measure rather than assume

The operator reports successful language-model training near **1e-3**, selected
through CMA-ES learning-speed measurements. Verify the exact run, optimizer,
batch/target normalization, warmup, clipping, and weight representation before
making a numerical comparison. Do not present that recalled rate as a verified
post-training optimum. Do not dismiss it merely because conventional SFT often
uses lower rates.

Begin with an LR-only, logarithmically spaced sweep. An illustrative range is
`2e-6, 1e-5, 5e-5, 2e-4, 1e-3`, including bounded higher-rate probes. These are
candidate rates, not an approved executable launch configuration. Finalize the
range and budget after numerical qualification and data audit.

The reference start is the trusted parent
`aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39`,
loaded explicitly in train/y mode, not the stopped u64 branch.
Each candidate must share that parent, qualified optimizer/precision
schema, fresh-stage state policy, examples/order, global batch, warmup convention,
clipping, target budget, evaluation panels, and primary weight representation.
Warmup must not consume the entire comparison. Evaluate explicit saved/x and
train/y separately; do not cherry-pick the better representation per metric.
Predeclare the selection representation and retention requirements for both.

Resolve BF16 effective-update behavior and production fused-gradient routing
before interpreting the LR sweep. Tiny rounds-to-nearest updates can be suppressed;
higher LR and changed precision are not interchangeable treatments. The existing
SR candidate's CPU distributed results do not establish CUDA/ROCm or production
loader compatibility. No FP32 master weights or CPU Adam arithmetic are introduced
by this plan. Checkpoint schema and Schedule-Free x/z/y semantics stay explicit.

Only after an informative LR sweep should CMA explore additional parameters.
Use the same frozen fitness and held-out policy; do not simultaneously optimize
LR, mixture, precision, and protocol and then claim an LR result. Numerical
qualification can be short, but the learning comparison must contain enough
matched agent exposure to distinguish signal from seed/order noise.

## Sustained learning program and budget

1. **Audit and qualify:** semantic corpus report; source/split admission;
   precision, production-gradient, and checkpoint checks; metric implementation.
2. **Fit and select:** bounded fitting checks on admitted training examples;
   matched LR sweep with both likelihood and free-generation measurements.
3. **Sustained SFT:** provisional envelope of **50--150M agent targets** and
   **200--600M total supervised targets**, adjusted to the audited inventory and
   frozen mixture. This is not a promise that all those agent targets remain
   usable after audit. Report unique coverage separately from repetition.
4. **On-policy correction:** once useful interactions occur, collect the
   student's actual failed states and verified successes. Use rejection sampling
   and DAgger-style correction with broad replay, in named consolidation stages.
5. **Outcome RL:** use RLVR/GRPO only when executable rewards have useful success
   variation and validators resist exploitation. Sparse all-zero rewards are not
   the default method for installing missing basic tool behavior.

Use a short-to-long curriculum that preserves complete units and causal evidence.
General conversation, document grounding, instruction following, and retention
remain part of the mixture. Increase coverage of observation-dependent branches
and memory demands, not just explanation length. Compare action-only/private
analysis with identical trajectory coverage; equal target counts alone can hide
unequal repetition or compute.

The local eight-GPU box supports a substantial 4B post-training investigation.
Rough training-only planning estimates from the earlier runtime are 3--5 hours
for 50M total targets, 12--20 hours for 200M, and 3--5 days for 1B. These are
extrapolations, not benchmarks for the new optimizer or audited agent-heavy
mixture. Re-measure throughput; collection, replay, qualification, and evaluation
are additional costs. A one-to-two-week program is an investigation horizon,
not a promise of a generally capable agent.

## Expected learning curves and decisions

- **Decision loss and free generation improve on training and development:**
  continue the frozen stage and look for increasing executable success.
- **Reasoning/average loss falls, decision loss worsens:** do not call it a
  winning candidate; investigate the objective, source competition, or updates.
- **Training improves, development does not:** examine coverage and overfitting.
- **Likelihood improves, free generation does not:** examine rollout drift,
  response transitions, and correction training rather than claiming capability.
- **Decisions improve, execution fails:** inspect tool semantics, environment
  mismatch, observation handling, and longer-horizon composition.
- **Nothing fits even consumed examples:** return to implementation/optimization
  diagnosis rather than expanding the corpus blindly.

Milestones should require measurable progress, not a finished agent at the first
checkpoint. Freeze target-token milestones, quantitative regression floors,
uncertainty/replication policy, and stop rules before each stage. No mid-run
mixture/LR/objective steering. A failed stage can motivate a new named experiment,
not retrospective revision of its gate.

## Existing evidence and boundaries

The prior diagnostics found improved teacher-forced agent continuation likelihood,
while the u64 train/y opening decision worsened. CPU label/gradient routing and
short forward/serving comparisons passed; full production backward/update behavior
remains a separate question. In the 24-task supplied-`Analysis:` diagnostic:

| Representation | Valid prefilled first turns | Task successes | Admitted reads |
|---|---:|---:|---:|
| Parent train/y | 0/24 | 0/24 | 0 |
| u64 train/y | 1/24 | 0/24 | 0 |
| u64 saved/x | 12/24 | 0/24 | 0 |

The supplied opening alone was insufficient. Saved/x's completed turns all used
limit 200; two selected the correct first task path, but violated the requested
bounds. This is evidence of partial conditional learning, not successful tool
execution, nor evidence that the architecture cannot learn it.

Local evidence roots under `/mnt/nvme2n1/erikg/e97_systematic_posttraining/`:
`learning-diagnostic-v1`, `analysis-prefill-diagnostic-v1`, and
`private-analysis-lr5e5-screen-v1`. Prefill aggregate SHA-256 identities:

- parent train/y: `4bd4fa9ed4f0d6a8bc13b442e4a26d1912f7b7c29e6e97dd4a37271b7321efc3`;
- u64 train/y: `098405b006e94d6557a082e748bbed52cbb64af254a8cffc4c2ccfdce419bc6c`;
- u64 saved/x: `bb35bd3f59b734ae155183847b383fca648d9db2260871e32b12934ac31e866a`.

The existing u64 stop gate remains immutable and false: no u128 continuation or
promotion is authorized by this addendum. The approved new direction is a new
program, not a resumption of that stopped branch. No automatic failed-run retries,
source admission, external spending, or Frontier launch is implied. Work remains
attended and single-threaded; numerical/distributed qualification may use the
leased local GPUs without delegated workers.

## Literature calibration

- [AgentTuning](https://arxiv.org/html/2310.12823v2) used 1,866 trajectories and
  started from Llama-2-chat, mixed with general conversations. That small agent
  count does not measure the instruction-learning effort inherited upstream.
- [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
  contains approximately 939,000 examples; its
  [post-training program](https://allenai.org/blog/tulu-3-technical) includes SFT,
  preference optimization, and verifiable-reward RL. This supports staged
  post-training, not a universal token requirement or LR for E97.

Dataset counts, nominal LRs, optimizer families, base-model capabilities, and
supervision density are not interchangeable across papers. Our choice must rest
on measured learning speed and retained capability in this implementation.

## Immediate next step

Produce a **bounded, source-bound semantic audit of the 151.6M candidate**, with
particular attention to translated reads/observations, path normalization, and
causal context at window starts. Deliver measured risk counts, replay limits,
and a keep/exclude/rebuild recommendation. That report determines the usable
backbone for the decision-metric implementation and LR-sweep freeze. Do not
launch the sustained run merely because the candidate's token count is large.
