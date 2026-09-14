# E97 numerical exploration: findings, limits and return to learning

This index preserves the long exploration without requiring a reader to infer a
programme-level verdict from individual passing/failing jobs. See
[the current learning status and decisions](../E97_AGENT_LEARNING_STATUS.md).
Original reports, immutable source exports, private measurements, failed attempts
and thresholds remain authoritative for their own experiments.

## What started the investigation

We were qualifying sampled actor probabilities against trainer scoring for an
outcome-RL path. The original full57 assay failed actor/historical and trainer
agreement checks. Subsequent investigations asked increasingly specific questions
about replay semantics, observation effects, precision, recurrence and projection
geometry. This was not evidence that ordinary mixed-precision SFT had never worked:
the completed correction already demonstrated0/16→7/16 autonomous completions.

## Chronological evidence map

| Stage | Established finding / intervention | Limits and primary report |
|---|---|---|
|BF16 update/storage and CE work|Small ordinary BF16 parameter stores can discard updates; stochastic stores and exact live-y handling address measured cases. Connected checkpointed gradient and memory tests were exercised.|Real optimizer/storage issues, distinct from later forward noise. [BF16 update diagnostics](e97-bf16-update-diagnostics-v1.md), [SR CUDA](e97-sr-cuda-qualification-v2.md), [precision integration](e97-sft-precision-integration-v1.md)|
|Initial full57 probability assay|Measured disagreement with retained historical probabilities; comparison and CE gates were separate.|Original failures retained. [RL probability qualification](e97-native-rl-logprob-qualification-v1.md)|
|Replay/capture controls|Actual generator-path replay could reproduce historical values; repeatability alone was not the same as historical binding. Hash-seed hypothesis was unsupported in tested cases.|No replacement of recorded behavior scores. [Actor repeatability](e97-actor-logprob-repeatability-v1.md), [live capture audit](e97-live-actor-capture-audit-v1.md)|
|Head and activation observations|Head rounding could amplify or hide differences; a head-only FP32 shadow did not eliminate the discrepancy. Early instrumented traces required exact native binding before interpretation.|Some observation attempts failed binding or alignment guards; those remain failures. [Head probe](e97-head-precision-probe-v1.md), [activation alignment](e97-activation-alignment-v1.md), [observer effects](e97-observer-effect-v1.md), [readback effects](e97-readback-effect-v1.md)|
|First mixer / FP32 recurrent state|Projection layout and recurrent state/checkpoint choices mattered. Merely using FP32 live state did not settle full-model agreement.|Older compatibility paths and failed candidate versions preserved. [First mixer](e97-first-mixer-probe-v1.md), [FP32 state versions](e97-fp32-recurrent-state-v1.md)|
|Recurrence launch selection|Production H60 entered a Python wall-clock tuner that small H2/H4 fixtures bypassed. FP32 recurrence was pinned to BLOCK_H1 / warps4 and production-geometry cases tested.|Historical tuner winners unavailable; pinning did not alone pass the full57 current-policy gate. [Fixed recurrence](e97-fixed-recurrent-kernel-v1.md)|
|Pinned head audit|Bound FP32/FP64 selected-dot checks separated head re-storage effects from already-different hidden states.|Not all outliers were head-only. [Pinned head audit](e97-pinned-head-numeric-audit-v1.md)|
|Composite `fp32-linear-v1`|FP32 arithmetic in163 Linears, BF16 persistent parameters/body/gradients and actual FP32 readout; repeated full57 current-policy max gap.051846 exceeded.05.|No FP32 master model. Still a numerical failure. [Composite v1](e97-fp32-linear-candidate-v1.md)|
|First-divergence trace|Layer0 readout first differed at token1, although raw projections were identical throughout the2,519-token prefill. A later projection difference was a separate interval.|Site order is not token chronology. [First divergence](e97-first-divergence-v1.md)|
|Actual recurrence capsule/control matrix|Fresh model-free replay reproduced actual outputs/final states exactly. At fixed numeric allocations, masks-only changes had no effect; checkpoint-workspace selection switched actor/teacher results exactly.|Live/final states were FP32 in both paths. Allocation and generated storage code remained coupled. [Actual recurrence operands](e97-recurrent-operands-v1.md)|
|Composite `fp32-linear-v2`|Uniform FP32 workspace repaired that measured dependence. Full57 current-policy max.0282168 passed. Teacher scores/CE stayed exact to v1 and both fresh measurement files repeated byte-exactly.|Historical-probability binding still failed; not4B backward,64K or training qualification. Initial observer-fixture failure also retained. [Uniform workspace](e97-uniform-workspace-v2.md)|
|Packed-stage prerequisite|Before any64K forward, standalone-versus-actor120 probes failed: max.0561256,p99.0300760. The96 assistant probes agreed within.00003611; the early prompt probes exposed another case.|Zero packed forwards. Do not turn the agreeing subset into a retrospective pass. [Packed forward attempt](e97-packed-forward-v1.md)|
|Early-prompt localization|Exact native score/trace binding located layer0 QKV output at token0/feature36, with identical preceding inputs. Selected dots showed strong cancellation sensitivity.|Not evidence of semantic memory erasure or sole attribution of the entire score gap to one coordinate. [Early prompt](e97-early-prompt-trace-v1.md)|
|Actual Linear capsule/control replay|All full-model references and eight model-free FP32/BF16 results bound exactly. At common numeric input/weight addresses and strides, matrix height controlled the pre-store FP32 result. First-row max gap1.90735e-6; BF16 conversions were faithful.|Backend/output-allocation choices remain coupled; no instruction-level explanation claimed. [Actual Linear operands](e97-linear-operands-v1.md)|
|One fixed-row kernel candidate|A fixed IEEE FP32 schedule removed measured height and row-placement dependence. Accuracy/tail/bias checks passed; isolated QKV cost1.92×/3.77× at heights1/512.|Non-dispatched prototype only. No model-policy integration, backward or capability qualification. [Fixed Linear micro](e97-fixed-linear-micro-v1.md)|

## Conclusions we should carry forward

- **The observed mechanisms are understood at a useful operational level.** We
  have original-result-bound reproductions, not just guesses about BF16 or a set
  of precision knobs that happened to improve one score.
- **Ordinary floating-point shape dependence is not a blanket veto on supervised
  learning or inference.** The numerical investigation became too broad a
  dependency of the agent-learning programme. That priority is now corrected.
- **BF16 is lossy, but semantic/context loss was not demonstrated here.** Tiny
  arithmetic errors and near cancellation explain measured numerical sensitivity;
  they do not diagnose all task failures, vanishing gradients or catastrophic
  long-context memory decay. Current FP32 carry/workspace protects one arithmetic
  location, not all possible sources of error.
- **Some discovered issues were genuinely important:** stochastic update storage,
  exact Schedule-Free representation, faithful scoring paths, precision/workspace
  metadata, production-shape coverage and observer binding. Do not discard these
  safeguards simply because the latest discrepancy is normal numerical variation.
- **A pass is scoped.** Full57 is not universal agreement; a micro kernel is not
  a qualified model; a teacher correction is not autonomous success; successful
  training/low loss is not reliable task completion.
- **No failed evidence is rewritten.** Historical probability binding, the
 120-position gate, unrun64K controls and exact-continuation failure retain their
  original status. New SFT experiments may use the previously exercised baseline
  rather than inherit every experimental policy or diagnostic as a prerequisite.

## What is and is not deployed

The exercised training baseline produced the closed880-update run and separate
32-update behavioral correction. Experimental numerical policies v1/v2 exist as
opt-in code paths with their documented qualifications. The fixed-row Linear
prototype is **not selected by any model policy** and is not a new default.
No full FP32/master model was introduced. No numerical experiment here added
optimizer updates, admitted new candidates, promoted a checkpoint, or established
an additional behavioral gain.

The fixed kernel is retained for possible reference/repair use. Its throughput
cost must be justified by end-to-end needs, not by treating bitwise agreement as
an end in itself. New architecture/performance work is not the immediate priority.

## Return to data and tuning

The operator has directed the programme back toward training/tuning. Next work is
verified, observation-dependent curriculum and same-state corrections, with fresh
autonomous evaluation and conversation/tool retention. Data fidelity, useful
coverage and the closed feedback loop are the actionable gaps—not just adding
raw tokens, replaying the same templates or eliminating floating-point noise.

Keep the880+32 budgets closed. Freeze the next tranche's data identity, weight/
optimizer representation, actual exposure, budget and evaluation gates separately.
Use SFT/DAgger before RL on the present correction material: the existing sampled
batch has zero within-task mixed-reward groups, regardless of its numerical path.
No production admission or historical-probability substitution is implied.

## Artifact navigation and privacy

Private artifacts are under `/mnt/nvme2n1/erikg/e97_systematic_posttraining/`, with
run roots and hashes specified by each linked report. Sibling `*-control/worktree`
exports and source inventories preserve executable code. Captured prompts,
probabilities, activations and selected/full operand matrices remain private.
Read the corresponding bound receipts and terminal audits before reinterpreting
any result. Preserve failed preflights and manually reviewed replacement attempts;
never overwrite an older authority to make the chronology look cleaner.
