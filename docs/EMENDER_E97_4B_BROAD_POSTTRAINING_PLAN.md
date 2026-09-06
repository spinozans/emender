# EMENDER E97 4B broad post-training plan

**Status:** normative next-phase plan; source audit and holdout freeze precede training

**Created:** 2026-09-01

**Foundation parent:** E97 4B step 24,448 / 99,723,771,904 accepted tokens

**Foundation SHA-256:** `3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568`

**Agent baseline:** cumulative-recovery u8

**Agent baseline SHA-256:** `eed48419d657b0be03953f6c99dc0ba14e61579ee748174ef0e6794f895157ed`

**Current narrow development parent:** complete-64K u256, SHA-256
`2f24db49be7bafb0e155bf3698f20193def8e90efc1bd69bdfbfe2b9661b541b`

## 2026-09-04 grounded-agent research amendment

The dedicated
[`EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md`](EMENDER_E97_4B_AGENT_POSTTRAINING_RESEARCH_PLAN.md)
now governs grounded conversation, typed read/list tools, observation-conditioned
recovery, recurrent-state qualification, behavioral data scale, and the
architecture decision gate. This broad plan remains authoritative for source
admission, boundary-aware packing, Schedule-Free training controls, and the
multi-stage post-training lineage. Where its earlier agent-stage sequencing or
token targets conflict with the dedicated research plan, the newer research
plan controls.

## 2026-09-02 document-aware recipe amendment

V4 has now been consumed by its first independent evaluation and failed 0/240.
It is diagnostic-only and cannot support another blind claim. A future V5 must
be frozen after this recipe is fixed and before the resulting model is scored.

Pure masked-SFT continuation is superseded as the next training recipe. The new
declared stage jointly trains causal document replay and masked SFT in explicit
64K packs. Every pack provides:

- `tokens`: token-aligned `uint32[context_size+1]`;
- `loss_mask`: prediction-aligned `bool[context_size]`;
- `valid_mask`: token-aligned `bool[context_size+1]`;
- `reset_before`: token-aligned `bool[context_size+1]`.

Every independent record sets `reset_before` on its first token in every E97
layer. The causal target entering a new record is masked. Invalid padding is a
recurrent identity transition and contributes no loss. The fused forward and
backward kernels must block both information and gradient across resets.
Authentic trajectory records do not reset internally.

The initial amended mixture is 20% filtered CommaPile causal replay, 35%
authentic OpenHands actions, 25% broad instruction, and 20% Pi retention. This
is a named fresh-optimizer stage from the behaviorally selected complete-64K
u256 parent, not an exact-resume mutation of the stopped pure-SFT stream.

## Decision and motivation

The first Pi curriculum established an exact runtime contract and reliable
retention within trained task families. It did not establish broad instruction
following or family-level transfer. The selected cumulative-recovery u8
checkpoint scored:

- 120/120 on the original Pi core smoke panel;
- 226/240 on template-held-out compositional v2;
- 0/240 on the first family-held-out v3 panel.

V3 traces replaced explicit user paths with memorized defaults such as
`config.json` and repeated the resulting failed call. Schema conformance was
240/240, but exact arguments were 0/240 and no trajectory completed. V3 is now
diagnostic and must not be used again as blind promotion evidence.

The active agent lineage consumed only about 2.17 million assistant-target SFT
tokens. Synthetic record counts and large unconsumed authorities must not be
reported as training exposure. The next phase therefore stops one-family repair
stages and adopts broad public instruction, reasoning, and repository-agent data
with cumulative Pi replay.

## Objectives

1. Improve instruction grounding, path/value copying, planning, comparison,
   recovery, and long-horizon repository behavior.
2. Distill strong reasoning into a 4B recurrent model efficiently rather than
   expecting it to discover reasoning through reinforcement learning from
   scratch.
3. Preserve exact Pi serialization, tool schemas, stopping, sandbox invariants,
   and all previously acquired tool skills.
4. Select checkpoints by frozen behavioral evaluations rather than loss.
5. Keep every source, filter, sample, pack, optimizer clock, checkpoint, and
   evaluation reproducible by immutable revision and SHA-256.

## Evaluation firewall

Before downloading training payloads, freeze two new evaluation authorities:

### V4 synthetic structural holdout

V4 used generators, identities, paths, values, action structures, and failure
modes absent from both the training sources and v3. Its first independent result
was 0/240, after which it became diagnostic-only. It must not be reused for an
independent claim; freeze a new V5-style authority before evaluating the amended
recipe.

### Real-repository holdout

Freeze repository URL, immutable commit, task description, allowed commands,
expected patch/postconditions, and focused/full tests. Split by whole repository,
not issue or file. No repository or issue in this holdout may enter SFT,
rejection-sampling, or RL prompts. Record content hashes before training-source
processing.

Also retain smoke and v2 as regression suites. V3 remains a disclosed diagnostic
baseline but is no longer blind.

## Candidate public sources

Every source requires an immutable Hugging Face revision, dataset-card snapshot,
license/provenance review, schema receipt, content statistics, and contamination
report before admission.

| Track | Preferred sources | Role |
|---|---|---|
| Broad instruction | AllenAI Tulu 3; SmolTalk2; reviewed TuluTalk components | General instruction, chat, code-language, safety, and tool retention |
| Reasoning | OpenThoughts3; OpenR1 Mixture-of-Thoughts; NVIDIA OpenCodeReasoning-2 | Verified math, code, science, critique, and solution decomposition |
| Repository agent | NVIDIA Open-SWE-Traces; SWE-Next SFT trajectories; reviewed SWE-agent/OpenHands traces | Multi-turn repository inspection, execution, failure, repair, and verification |
| Pi alignment | Existing cumulative Pi authorities plus on-policy corrected traces | Exact Action/Arguments/Final grammar, real observations, stopping, and recovery |

A source name is not a quality verdict. Failed trajectories, unverifiable
outputs, benchmark overlap, unsupported licenses, malformed tools, or excessive
truncation fail closed.

## Target mixture and consumed-token clock

Twenty-five million **consumed** targets is only the systems and recipe pilot,
not the expected final exposure. The already admitted authorities contain
775,217,258 targets and 1,491,924,900 input tokens even though the first
CommaPile extraction contains only 5.19M causal targets. The scaled Stage A
therefore evaluates 25M/100M/250M/500M consumed-target milestones and may
continue toward unique coverage of the admitted pool when frozen behavioral
curves remain positive. Total processed context is reported separately.

The scaled authority must sample source records without replacement and the
fixed-world pack sampler must traverse a deterministic epoch permutation.
Small retention sources are not replayed at a fixed 20% indefinitely; they are
consumed once or augmented with new authentic/on-policy trajectories. The
original proposed mixture was superseded by the document-aware amendment above.
The first immutable 25M pilot authority uses:

| Track | Initial target share |
|---|---:|
| Source-filtered CommaPile causal documents | 20% |
| Successful OpenHands repository trajectories | 35% |
| Broad instruction/reasoning | 25% |
| Pi protocol and retention replay | 20% |

The first predeclared scaled tranche requests approximately 519.5M unique
source targets: 300M broad, all 110,268,050 eligible OpenHands targets, all
9,250,214 Pi-retention targets, and 100M newly extracted filtered CommaPile
causal targets. The remaining admitted broad records are reserved for a named
continuation rather than silently folded into the first tranche.

Sampling is stratified by track and subdomain. A flat union is prohibited because
large sources would dilute rare recovery and protocol skills. Checkpoint sampler
cursor plus immutable authority/pack manifests must permit exact reconstruction
of aggregate and per-source consumed targets.

Mixture weights may change only at named stage boundaries. Every boundary starts
a new immutable authority and records the behavioral reason for the change.

## Serialization tracks

### General instruction and reasoning

Preserve the source conversation roles, mask every correct assistant turn, and
reset recurrent state between records. Keep solution reasoning only when it is
source-provided or teacher-generated under a documented prompt and passes the
track's verifier. Do not invent reasoning during conversion.

### Repository trajectories

Normalize foreign tools into a bounded intermediate schema before mapping to Pi.
Preserve authentic observations, stderr, exit status, and empty output. Retain
only successful trajectories or prefixes with an independently verified teacher
correction. Split long trajectories at semantically valid state boundaries; do
not silently truncate assistant targets.

### Pi protocol

Retain the canonical prompt in
`configs/pi/e97-pi-core-system-prompt.txt`, the closed tool contract in
`configs/pi/e97-core-tools.ts`, and the transcript serializer in
`ndm/e97_agent_protocol.py`. Successful empty tools remain the literal
`(no tool output)` observation.

## DeepSeek-style training sequence

The practical technique is reasoning distillation followed by rejection
sampling and, only after a strong SFT baseline, GRPO/RL with verifiable rewards.

### Stage A — broad cold-start SFT

Use 25M only to qualify the objective, then train scaled no-replacement
authorities through 100M, 250M, and 500M consumed-target milestones. Continue
toward broader unique-source coverage only while frozen behavioral curves
improve. This is full-parameter Schedule-Free SFT and retains the fixed-world
eight-rank local execution contract. Select by broad instruction, code, Pi
smoke, v2, and frozen repository evaluations. Stop early on regression or
saturation; do not spend the nominal clock merely because data remain.

### Stage B — verified reasoning distillation

Use strong public reasoning traces and strong-teacher solutions. Compare two
branches:

1. action/final-only distillation, where teacher reasoning selects verified
   actions but is not emitted by the student;
2. bounded private-analysis distillation, where concise state, hypothesis, and
   verification reasoning precedes an action and is stripped from the external
   Pi response.

Private-analysis support requires a separately reviewed parser/server change,
length bound, logging policy, stopping tests, and an ablation. Raw verbose
chain-of-thought is not assumed beneficial.

### Stage C — repository-agent distillation

Train execution-grounded multi-turn trajectories including exploration,
misdiagnosis, correction, minimal mutation, focused tests, broader verification,
and evidence-grounded completion. Repository and issue identities are disjoint
from all evaluation sets.

### Stage D — on-policy data aggregation

Run the current student in the sandbox. At the first divergence or failed
postcondition, retain the exact model-generated prefix and attach a verified
teacher correction. This directly addresses exposure bias that static
gold-history SFT cannot.

### Stage E — rejection sampling

Sample multiple student or teacher continuations, execute them, and retain only
schema-valid trajectories whose postconditions and tests pass. Prefer the
shortest fully successful trajectory when several are equivalent.

### Stage F — GRPO/RLVR

Apply group-relative policy optimization only to prompts with reliable automated
rewards. Candidate reward components are test/postcondition success, valid tool
schema, grounded final, no repeated-call cycle, sandbox safety, and a bounded
efficiency penalty. Outcome success dominates formatting rewards. Keep a frozen
reference policy and cumulative replay/KL protection. RL cannot authorize a
checkpoint that regresses the SFT gates.

## Training and systems controls

- Use the existing eight-GPU lease, NUMA locality, isolated Triton caches,
  pinned-CPU Schedule-Free state, atomic K8 checkpoints, and mmap reload gate.
- Keep gradient clipping configurable and recorded; the controlled `1.0` versus
  disabled ablation showed equivalent dynamics for the recover-read case.
- Qualify boundary-aware 64K packs before launching the amended stage. Whole
  records are packed without truncation; oversized logical units are excluded
  with explicit record, token, and target counts. No reset is inserted inside
  an authentic trajectory.
- Preserve train/`y` versus saved/`x` semantics and start fresh optimizer state
  only at declared distribution boundaries.
- Retain intermediate checkpoints and choose behaviorally. Lower loss is never a
  promotion argument by itself.

## Long-context execution evidence

The legacy local 65,536-token single-record SFT path is systems-qualified with
group-three activation checkpointing, cache release before records at least
32K, and 4,096-token checkpointed SwiGLU chunks; see
[`validation/e97-4b-pi-sft-64k-local-qualification.md`](validation/e97-4b-pi-sft-64k-local-qualification.md).
That evidence establishes fit, backward, update, checkpoint, and exact resume,
not packed-document correctness or long-context behavior. The amended path adds
explicit per-token recurrent resets and padding identity semantics and therefore
requires its own qualification and behavioral selection.

## Promotion gates

A replacement checkpoint must satisfy all of the following:

1. original smoke: at least 118/120, with no collapsed family;
2. compositional v2: at least 220/240 and at least 32/40 per family;
3. V3 and consumed V4 are diagnostic-only; a newly frozen V5 supplies the next
   independent unfamiliar-workflow gate;
4. real-repository holdout: predeclared patch/test gates and zero sandbox
   violations;
5. broad instruction/code/reasoning benchmarks: no material regression from the
   broad-SFT checkpoint;
6. long-horizon and adversarial recovery: no immediate identical-call loops and
   bounded completion;
7. checkpoint SHA-256, immutable data/source identities, and mmap reload pass.

Temperature and alternate-prompt robustness are reported separately. The exact
canonical Pi prompt remains the primary supported integration contract.

## Publication

Do not replace the current Hugging Face promotion pointer during development.
An accepted checkpoint receives a new immutable revision and tag containing:

- raw checkpoint hash and security warning;
- foundation and behavioral lineage;
- all source revisions, licenses, filters, manifests, and consumed-token counts;
- canonical prompt/tool contract;
- smoke, v2, diagnostic V3/V4, independent V5, real-repository, reasoning, and
  long-horizon receipts;
- explicit residual risks and unsupported uses.

A weights-only portable release remains a separate deliverable.

## Execution checklist

1. Record the v3 terminal diagnosis and mark it non-blind.
2. Freeze V4 and real-repository holdouts before training-payload download.
3. Audit candidate public dataset cards, revisions, licenses, schemas, and sizes.
4. Download only admitted immutable revisions and hash every payload.
5. Implement source-specific converters plus a common normalized trajectory
   schema and validation suite.
6. Build stratified 4K authorities; qualify 8K/16K separately.
7. Run a small source-mixture qualification and inspect decoded masked records.
8. Qualify at 25M, then train the 100M/250M/500M scaled Stage A ladder with
   deterministic no-replacement source selection and epoch-permuted packs;
   evaluate every retained milestone.
9. Build and compare reasoning-distillation branches.
10. Aggregate on-policy failures and perform verified rejection sampling.
11. Add GRPO/RLVR only after the strongest distilled checkpoint clears all SFT
    regression gates.
12. Publish only after complete independent rehash, reload, and behavioral
    review.
