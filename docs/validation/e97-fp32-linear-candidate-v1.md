# E97 composite FP32-linear repair candidate

## Purpose

The bound [head diagnostic](e97-pinned-head-numeric-audit-v1.md) isolated BF16
readout rounding amplification in two large outliers and a third upstream error
that survived FP32 readout. Earlier bound first-mixer measurements also showed
layout-dependent BF16 projection outputs from equal real inputs. This motivates
an **actual arithmetic intervention**, not another output-replacing observer.
The exact upstream operation responsible for task008 is not yet established;
this candidate is not presented as proof of that root cause.

One opt-in setting: **`--numerical-policy fp32-linear-v1`**. Existing defaults and
the earlier state-only `--recurrent-state-precision fp32` control remain intact.
The composite policy selects:

- FP32 recurrent carry, retained checkpoints, replay scratch and state gradients;
  fixed `(BLOCK_H=1,num_warps=4)` forward/backward geometry.
- FP32 arithmetic in every Linear projection, including the MLP, followed by
  **BF16 storage of ordinary hidden/projection activations**.
- **FP32 final vocabulary logits**, in both actor and training paths.
- BF16 persistent weights, BF16 parameter gradients, unchanged optimizer storage.

SiLU, normalizations, gate formulas, recurrent chunk sizes, sparse checkpoint
intervals, CE128, MLP4096 and checkpoint group3 are not changed in the full assay.
The v3 discarded inference-workspace exception is unchanged. No head count,
model width, depth, tokenizer, weights or training data is changed.

## Bounded implementation and persistence

`ndm/numerical_policy.py` uses a recomputing autograd Linear. Forward casts one
weight matrix temporarily; backward recreates casts as needed. The context saves
only the **original BF16 input, weight and optional bias**, never an FP32 weight
copy. Parameter/module identities, aliases (including tied embeddings), hooks
and state-dict keys are preserved. This is a real forward implementation, not a
forward hook that replaces outputs.

Each FP32 weight-sized temporary is capped at1GiB. The real head is
`50281 x 3840`, or772,316,160 bytes in FP32. This is not a persistent FP32 master
model. Activations and gradient-result scratch have additional transient cost;
64K memory/performance remains unqualified. Higher-order derivatives explicitly
raise rather than silently truncate. CUDA requires highest FP32 matmul precision
without TF32; the policy sets that and disables BF16 reduced-precision reduction.

SFT precision metadata records the versioned composite policy. The E97 loader
restores it even with separate architecture arguments, and normal exact resume
comparison rejects a changed policy. Conflicting state declarations and unknown
policies fail closed. No checkpoint has been trained under this policy.

## Frozen validation

CPU: **95 passed, 13 CUDA skips**. Covers output/gradient comparison to ordinary
FP32 PyTorch autograd, saved-tensor dtype/identity, BF16 parameters and hidden
stores, FP32 head output, no parameter mutation, inheritance, checkpoint loading,
state/config conflicts, scratch bounds and higher-order rejection. A meta-device
architecture check independently confirmed **163 Linear modules**, vocabulary
size50,281 and4,045,972,080 parameters for the frozen4B architecture.

Thirteen mandatory CUDA cases:

1. The ten prior storage/gradient/production-H60 cases, unchanged.
2. Real projection `3840 -> 11520` and real readout `3840 -> 50281`, 33 rows each:
   exact forward comparison, finite BF16 gradients, gradient relative L2<=.005
   versus FP32 PyTorch autograd; saved tensors must be original BF16 tensors.
3. A two-layer E97 model with input1041/prediction1040, projection512, sparse
   checkpoint16, MLP4096 and checkpointed CE128. Include reset at512 and padding;
   compare loss<=1e-4 and gradients relative L2<=.02 against identical arithmetic
   using ordinary autograd, assert real FP32 head/BF16 body outputs and unchanged
   parameters. This is **not a full4B backward qualification**.

Then two fresh workers run the actual4B candidate on the unchanged full57-turn,
2,711-token authority. No observer is active. Require actual FP32 logits on actor
and teacher, all163 Linear modules under the policy, zero recurrence autotune
entries, finite results, unchanged weights and peak allocated HBM<=16GiB.
Current-policy max log-probability gap<=.05, p99<=.02, CE/head delta<=.0001.
Require exact actor/teacher scores and CE means across fresh processes. Historical
recorded behavior probabilities and their original failed gates remain separate;
no replacement probabilities or tolerance fallback are permitted.

Recipe SHA:
`916b1e87207a161d17ecf61881e5be3952a4fcd23ac3e6e6a6fe81d282045d15`.
The CPU freezer verified that inputs and thresholds equal the prior state-only
recipe after removing only the explicit candidate policy/scope. Its preflight is
retained in `fp32-linear-candidate-v1-preflight` under the shared artifact root.

Run root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/fp32-linear-candidate-v1`.
Driver `scripts/run_e97_fp32_linear.sh`; freezer/auditor
`scripts/qualify_e97_fp32_linear.py`.

One checked GPU lease, local-device selection, NUMA binding, isolated caches for
kernel tests and each worker, both expandable allocator variables, immutable
source with before/after audits including failure, and composed lease cleanup.
Kernel900s, each worker1,800s, outer5,100s, teardown30s. No retries, no concurrent
GPU jobs, no delegated workers and no automatic additional candidate.

## Audited result — partial improvement, failed qualification

Source **`2c17fb02`**, managed process `proc_821c`,934s. Exit1 was the declared
`pinned-policy numerical gate failed`, **not a runtime crash**. All13 CUDA cases
passed in35.60s. The two production-sized Linear tests had gradient relative
L2=0 against their autograd references. The checkpointed two-layer model had
loss delta=0 and maximum gradient relative L2=0. This is bounded numerical
agreement, not a claim of universally bitwise-identical gradients.

Both fresh4B workers produced **byte-identical measurement files and summaries**.
All2,711 actor/teacher scores and CE means repeated exactly. Both verified163
FP32-arithmetic Linear modules, actual FP32 actor/teacher logits, zero recurrence
autotune entries, unchanged parameter SHA
`f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd`,
and peak allocated HBM9,697,057,792 bytes.

- Current-policy max: **.05184602737426758 > .05 — failed**.
- Current-policy p99: .0010260593146085753 < .02 — passed, but larger than the
  state-only baseline's .0007408213801682023. Not every error improved.
- CE/head mean delta:5.2447273901623775e-8 — passed.
- Historical actor replay max:.09257817268371582 — failed.
- Teacher versus historical recorded max:.09374070167541504 — failed.

The original three current-policy outliers are now below .05. However, another
position worsened and crossed the threshold; this is **not a qualified fix**:

| Task / turn / generated position | State-only actual gap | Composite actual gap |
|---|---:|---:|
|004 /1 /23|.027909040451049805|**.05184602737426758**|
|004 /1 /29|.11964964866638184|.003026247024536133|
|008 /1 /23|.07160699367523193|.013171017169952393|
|014 /4 /15|.12353801727294922|.01941204071044922|

The sole over-limit position is now **`onpolicy-task-004-sample-0`,turn1,
zero-based generated position23**, not the earlier task008 target. Any further
localization must bind to this composite reference, not silently reuse earlier
state-only activation traces. No additional candidate was launched.

An independent CPU audit checked all57 identities, frozen historical values,
2,711-token coverage, finiteness, exact fresh equality, maximum error, CE means,
and outlier count. All17,675 source files passed before/after checks in both
runner and controller; inventory bytes also matched. `cleanup.log` records all
eight GPUs idle, no compute processes and no active lease files, without reaping.

Hashes:

| Artifact | SHA256 |
|---|---|
|Source inventory|`113e54ca29acd090e922060fdb72d2e8d783c17c10ab8a816af94454f9180927`|
|Kernel tests|`529a0c70f10a0ecbbf15844097e7b35e7101929bd356f32a1fa938fa2a15733b`|
|Each worker measurements|`64ba00df08cf48bc87e87ca1afc9c5257bfb88bb25addb63b03cedd88812e34e`|
|Each worker summary|`490092c0e3aa4f9154aee5437350aec0a9b548889205c4ab9abeccd56fb4c7c8`|
|Audit|`ce5b7b4c6dbe2e5d988a3eb434ddb08f21a23e2277898a1721f3fc7345953a6e`|

Private probability records remain unpublished. `result-audit.json` retains the
independent CPU checks. Thresholds and failed historical evidence are unchanged.
Zero optimizer updates. The candidate remains opt-in and unqualified; no data
admission, behavioral gain, full4B backward,64K, multirank, restart or RL optimizer
qualification is implied. The880+32 training budgets remain closed.
