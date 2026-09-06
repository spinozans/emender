# E97 4B grounded-agent post-training research plan

**Status:** normative research and decision plan; implementation stages require
separate immutable data and run receipts

**Created:** 2026-09-04

**Scope:** broad conversation, grounded document use, typed `list_files`/`read`
tools, observation-conditioned recovery, and recurrent-state save/restore/fork

**Related plans:**

- [`EMENDER_E97_4B_BROAD_POSTTRAINING_PLAN.md`](EMENDER_E97_4B_BROAD_POSTTRAINING_PLAN.md)
- [`EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md`](EMENDER_E97_4B_ONPOLICY_TASK_LAKE_EXECUTION_PLAN.md)
- [`operations/e97-4b-pi-runtime.md`](operations/e97-4b-pi-runtime.md)
- [`validation/e97-4b-pi-v4-post-broad-preflight.md`](validation/e97-4b-pi-v4-post-broad-preflight.md)

## Decision summary

The immediate failure mode is an incompletely post-trained agent policy, not yet
evidence that E97 requires Transformer layers. The model can emit valid tool
schemas and has previously acquired narrow tool behavior, but it does not
reliably bind literal paths, condition its next action on tool observations,
recover from errors, or terminate with grounded evidence.

Recent open research supports a staged remedy based on a model-oriented
agent-computer interface, diverse complete trajectories, verified outcomes,
student-state corrections, and only then preference optimization or RL with
verifiable rewards. The first serious behavioral stage should be measured in
tens of millions of assistant-target tokens, not another billion-token broad
campaign.

Pure fixed-state recurrence nevertheless has a credible copying and retrieval
risk. A controlled random-binding experiment at increasing context lengths is
therefore a required architecture gate. Short-context failure after clean
supervision is evidence against the current architecture or runtime. Failure
only at long context motivates typed external memory or sparse/selective
attention rather than abandoning recurrence wholesale.

## Current evidence

### What is already working

- The mature foundation has 99,723,771,904 accepted pretraining tokens.
- Boundary-aware 64K training resets every independent document and blocks
  cross-document information and gradient flow.
- Fixed-world eight-rank DDP, Schedule-Free state, atomic checkpoints, exact
  resume, and mmap reload validation are qualified.
- Pi tool calls are schema-valid in the diagnostic panels.
- A narrow earlier checkpoint reached 119/120 on its trained smoke distribution,
  proving that E97 can acquire the protocol and small tool policies.
- The active `5e-5` broad repeat is numerically stable and is a candidate source
  of a stronger raw language/code parent. Its loss is not promotion evidence.

### What is not working

The completed low-LR broad checkpoint scored 0/240 on diagnostic V4. Across the
240 tasks it produced schema-valid calls, but no correct tool sequence, grounded
final, or completed task. It commonly substituted memorized paths, changed file
extensions, ignored explicit directories, or repeated the same failed call.

V3 and V4 are now diagnostic-only. Their records, exact values, paths, and
expected trajectories must not be copied into training. They may be used to
classify failure modes. A new independent V5-style holdout is frozen only after
the final runtime and training recipe are fixed.

### Interpretation boundary

A low masked-token loss can coexist with poor agent behavior. Most loss comes
from predictable prose, syntax, and formatting, while the few tokens selecting
a path, value, or recovery action determine task success. Teacher forcing also
does not expose the compounding consequences of an early wrong action.

The project therefore distinguishes:

1. language/code distribution fit;
2. one-step instruction and value binding;
3. observation-conditioned policy execution;
4. repeated end-to-end task reliability;
5. long-context retrieval and persistent recurrent memory.

No metric from one level substitutes for another.

## Research basis

The plan is informed by the following open work.

### Behavioral post-training and trajectory data

- **CodeAct** collected about 7,000 multi-turn interactions and deliberately
  retained examples where an initial error was corrected after execution
  feedback. It showed that a modest, selected trajectory set can improve agent
  behavior without sacrificing broad capability:
  <https://arxiv.org/abs/2402.01030>.
- **SWE-Gym** used 491 successful trajectories, averaging about 19 turns and
  19,000 tokens, to improve Qwen2.5-Coder-32B from 7.0% to 20.6% on SWE-bench
  Verified. It reports reduced stuck-in-loop behavior. A naïve later mixture of
  successful on-policy trajectories reduced performance, showing that success
  filtering alone is insufficient:
  <https://arxiv.org/abs/2412.21139>.
- **SWE-smith** scaled executable tasks across 128 repositories and trained on
  5,016 expert trajectories. Its ablations favor task/repository diversity and
  cap repeated easy instances because oversampling easy successes can hurt:
  <https://arxiv.org/abs/2504.21798>.
- **Tülu 3** uses a modern open sequence of curated SFT, preference optimization,
  and RL with verifiable rewards, with development and unseen evaluation sets:
  <https://arxiv.org/abs/2411.15124>.
- **SWE-RL** provides evidence that verified RL can improve software reasoning
  after a capable starting policy, while its SFT control was less generalizable.
  Its patch-similarity reward is not adopted here as a substitute for executable
  correctness: <https://arxiv.org/abs/2502.18449>.

### Interface and evaluation

- **SWE-agent** shows that a compact LM-oriented agent-computer interface can
  materially improve behavior relative to generic computer interaction:
  <https://arxiv.org/abs/2405.15793>.
- **ToolSandbox** evaluates state dependencies, exceptions, missing information,
  clarification, and dynamic trajectories using milestones and minefields:
  <https://arxiv.org/abs/2408.04682>.
- **τ-bench** evaluates final world state and repeated reliability with
  `pass^k`; even strong proprietary models were inconsistent across repeated
  trials: <https://arxiv.org/abs/2406.12045>.
- **xLAM** and **ToolACE** show that small open models can acquire strong atomic
  function-calling behavior from diverse synthesized and verified tool data.
  Atomic calling is necessary but does not establish long-horizon recovery:
  <https://arxiv.org/abs/2409.03215> and
  <https://arxiv.org/abs/2409.00920>.

### Recurrent architecture risk

- **Repeat After Me** gives theoretical and empirical evidence that fixed-state
  sequence models are disadvantaged relative to attention on exact copying and
  contextual retrieval: <https://proceedings.mlr.press/v235/jelassi24a.html>.
- **RULER** finds substantial degradation with context length and reports RWKV
  and Mamba behind Transformer models on retrieval, variable tracing, and
  aggregation: <https://arxiv.org/abs/2404.06654>.
- **Jamba 1.5** reports that a hybrid with approximately one attention layer per
  seven Mamba layers outperformed pure Mamba-2 in its experiments, supporting a
  sparse-attention contingency rather than an all-or-nothing architecture
  choice: <https://arxiv.org/abs/2408.12570>.

These results are directional rather than directly comparable. Most published
agent models start from mature Transformer instruction checkpoints, are larger
than 4B, and often use proprietary teachers. E97 is a pure recurrent 4B model
with different state capacity and training history. Claims must come from the
controlled experiments below.

## Research questions and falsifiable hypotheses

### RQ1 — Does the live runtime present usable observations?

**Hypothesis:** live transcript serialization and recurrent continuation expose
exactly the role boundaries, tool output, errors, and suffix tokens represented
in training.

**Evidence required:**

- byte/token comparison between a training record and its live equivalent;
- changed tool observations produce changed next-action logits;
- full transcript replay and token-delta recurrent continuation agree within the
  declared fused-CUDA numerical tolerance;
- save/restore produces the same next-token distribution as uninterrupted
  continuation;
- cloned state branches remain isolated after different suffixes.

A failure here stops behavioral training until the runtime is repaired.

### RQ2 — Can E97 bind arbitrary recent values?

**Hypothesis:** after a small clean tune, E97 can copy and use unseen random
paths, keys, and values at short context.

Build deterministic tasks using opaque random strings, with train/development
separation by generator seed and whole template family. Measure:

- literal path copying from the current user turn;
- selector-to-value binding;
- selection among distractor key/value pairs;
- use of a successful observation;
- correction after a structured `not_found` observation.

Evaluate at approximately 128, 1K, 8K, 32K, and 64K input tokens. Report exact
match by length and operation, not an aggregate alone.

Short-context failure after verified runtime parity and targeted acquisition is
an architecture/base-capability warning. A length-dependent decline supports
external memory or hybrid attention for long contexts while retaining the
recurrent core.

### RQ3 — Is trajectory diversity more important than raw token volume?

**Hypothesis:** structurally varied complete trajectories generalize better than
larger templated synthetic mixtures.

Compare equal-target-token branches:

1. many surface variations of a small number of templates;
2. independent task structures, directory layouts, error conditions, and valid
   solution paths;
3. the second branch plus authentic student-error corrections.

Development tasks must use unseen values, seeds, paths, templates, and whole
families. V3/V4 are not development targets.

### RQ4 — Does correction from student states repair cycles?

**Hypothesis:** training a verified next action after the student's actual bad
prefix reduces repeated-call cycles and improves completion more efficiently
than adding another clean expert trajectory.

For each rollout, classify the first meaningful divergence. Preserve the exact
student history and authentic observation, mask harmful student actions as
training targets, and supervise a verified correction. Compare against
clean-history SFT at matched target-token count.

### RQ5 — When does RL add value?

**Hypothesis:** rejection sampling and RLVR help only after SFT produces a
nontrivial distribution of successful trajectories and the reward measures the
intended outcome.

Do not launch GRPO merely to discover basic path grounding. RL eligibility
requires stable runtime semantics, executable rewards, a behaviorally competent
SFT parent, and frozen development gates.

## Target agent-computer interface

The initial product is a recurrent conversational read-observe assistant, not a
full autonomous coding agent. Its initial typed tools are:

```text
list_files(path, depth, limit)
read(path, offset, limit)
```

The runtime provides a compact current working directory and bounded top-level
listing. Tool results are structured, bounded, and explicit about success,
empty output, partial output, and failure. A missing file should resemble:

```json
{"ok":false,"error":"not_found","path":"metrics/a.json"}
```

The model must learn to ask for clarification when required information is
absent. The controller enforces tool schemas, prevents unbounded identical-call
cycles, records every observation, and requires an explicit grounded final.

`edit`, focused tests, and broader shell access are later capabilities. They are
not prerequisites for qualifying conversation, document comprehension,
recurrent memory, or read/list grounding.

## Data program

### Initial scale hypothesis

Build 5,000–20,000 complete trajectories, targeting roughly 10M–30M consumed
assistant tokens. This is a research range, not a success claim or mandatory
clock. Stop or expand based on frozen development curves.

### Required strata

- ordinary multi-turn conversation without tools;
- direct grounded document questions;
- list-then-read discovery;
- exact explicit-path reads;
- multiple-file comparison and synthesis;
- absent, partial, stale, and misleading context;
- missing-file and wrong-extension recovery;
- clarification when the requested fact is unavailable;
- successful empty observations;
- final answers with citations to observed paths or content;
- long sessions used for state save/restore/fork tests.

Vary paths, extensions, directory depth, identifiers, prose style, document
format, distractor count, observation length, and valid action order
independently. Cap records per latent template and per source identity. Preserve
complete records and authentic tool observations; never silently truncate.

### Correction records

Run the current student on development generators. Label the first failure as
one of:

- ignored literal value;
- invented path or directory;
- wrong extension;
- wrong tool or arguments;
- ignored successful observation;
- repeated failed action;
- unsupported assertion;
- premature or absent final.

A correction record contains the exact student prefix and environment result,
followed by a verified teacher continuation. Failed student actions remain
context, not supervised assistant targets. Corrections and clean expert
trajectories are separately counted.

### Retention data

Every behavioral stage includes broad conversation and document replay selected
without replacement where possible. Pi/tool examples must no longer be a tiny
incidental fraction of a large general mixture during acquisition. Mixture
weights change only at named stage boundaries and are reported by consumed
assistant-target tokens.

## Training sequence

### Phase 0 — broad-parent selection

Complete the active aggressive broad run unless a systems stop condition fires.
Evaluate saved `x` checkpoints on fresh text loss, broad conversation,
instruction following, document comprehension, and short grounding. Select the
parent behaviorally. Do not select solely by train/`y` loss.

### Phase 1 — runtime and architecture preflight

Complete RQ1 and run the untrained RQ2 baseline. No behavioral training begins
until serialization, observation conditioning, and recurrent-state transitions
are auditable.

### Phase 2 — clean acquisition canary

Train a small, diverse, clean-trajectory stage. With the current global target
rate, illustrative clocks are:

- 8 updates: acquisition/systems canary;
- 32 updates: about 6M targets;
- 64 updates: about 12M targets;
- 128 updates: about 25M targets.

Actual target counts, not nominal update estimates, are authoritative. Use a
learning rate large enough to move BF16 Schedule-Free parameters, save
frequently, and evaluate every saved `x` checkpoint.

If the canary does not improve one-step binding, observation conditioning, and
cycle rate, stop. Do not scale a broken recipe.

### Phase 3 — student-state correction

Collect fresh rollouts from the best Phase 2 checkpoint, add verified
corrections at actual failure states, and compare against a matched clean-SFT
control. Promote only if unseen-family completion and recovery improve without
broad conversational collapse.

### Phase 4 — authentic trajectories and rejection sampling

Add license-approved authentic read/repository trajectories with executable
postconditions. Sample multiple continuations and retain successful trajectories
with caps per task. Do not treat every successful rollout as high-quality;
reject needless cycles, unsupported finals, accidental test passes, and unsafe
mutations.

### Phase 5 — preference optimization and RLVR

Once the student succeeds often enough to supply useful rollouts, train
preferences favoring grounded progress, recovery, concise completion, and safe
abstention over fabricated paths and loops. GRPO/RLVR may then use rewards for:

- executable postcondition success;
- required milestone completion;
- absence of minefield actions;
- grounded final response;
- valid protocol;
- no identical-call cycle;
- bounded efficiency.

Outcome success dominates formatting rewards. Retain a reference policy and
broad replay/KL protection.

### Phase 6 — recurrent-memory demonstration

Qualify static system-prompt prefill, true suffix-only sessions, fixed-size state
serialization, exact restore within the declared numerical contract, cloning,
forking, and branch isolation. Demonstrate that independent continuations use
saved recurrent state without replaying the full transcript. Report state size
and distinguish task-relevant retained context from perfect arbitrary recall.

## Evaluation program

Use separate acquisition, development, regression, and final-holdout roles.

### Acquisition metrics

- teacher-forced loss by assistant-turn type;
- exact tool name and argument accuracy;
- exact opaque-string binding;
- next-action accuracy after success and failure observations;
- final-versus-action stopping accuracy.

### Interactive development metrics

- executable final-state/postcondition success;
- milestone and minefield results;
- grounded final answers;
- repeated-call cycle rate;
- recovery success after injected failures;
- unnecessary action count;
- repeated-trial `pass^k`;
- performance by unseen template and whole family.

Alternative valid trajectories may pass. Exact expected sequences remain useful
for diagnosis but are not the sole outcome metric.

### Regression metrics

- broad conversation quality;
- instruction following;
- document comprehension;
- safety and appropriate abstention;
- existing Pi smoke and compositional telemetry;
- long-session state retention and branch isolation.

### Independent evidence

The four real-repository holdouts remain untouched until repository-level claims
are contemplated. V3 and V4 remain disclosed diagnostics. Freeze V5 only after
the final runtime, tool schema, data recipe, training phases, and promotion
thresholds are fixed.

## Architecture decision gate

Do not add Transformer layers merely because the current checkpoint resembles a
raw completion model. Do not dismiss the fixed-state limitation either.

Retain pure E97 when:

- short-context opaque binding becomes reliable after targeted training;
- observation-conditioned actions improve materially;
- failure is primarily length-dependent or task-policy-dependent;
- typed external memory satisfies the intended product.

Prototype sparse/selective attention when:

- runtime parity is proven;
- clean targeted training succeeds on a matched Transformer control but not E97;
- E97 persistently fails short-context arbitrary binding, or long-context exact
  retrieval is a required product capability that typed memory cannot satisfy;
- the comparison controls parameter count, tokenizer, training data, objective,
  and evaluation.

A hybrid experiment is a named architecture branch, not an undocumented change
to the active 4B lineage.

## Stop conditions

Stop a stage on corruption, nonfinite values, OOM without a declared recovery,
checkpoint reload failure, source/holdout overlap, runtime serialization drift,
or loss of exact Schedule-Free checkpoint semantics.

Stop and redesign a behavioral recipe when additional consumed targets improve
training loss but not unseen binding, recovery, completion, or repeated
reliability. Ordinary regression on previously consumed synthetic templates is
telemetry; broad catastrophic degradation is a stop condition.

## Expected outcome

The near-term success criterion is an honest recurrent chatbot/read-observe
system that converses coherently, discovers and reads files, grounds answers in
observations, recovers from common errors, and saves/restores/forks fixed-size
hidden state. It does not imply parity with frontier autonomous coding agents.

The central research claim, if supported, will be that a substantially recurrent
4B model can acquire reliable grounded interactive behavior with a compact
agent-computer interface and bounded high-quality post-training. The plan also
permits the opposite finding: controlled copying and retrieval evidence may show
that sparse attention is necessary for the desired reliability or context
length.
