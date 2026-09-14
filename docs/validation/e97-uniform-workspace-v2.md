# E97 composite v2: uniform FP32 recurrence workspace

## Evidence-driven repair

The [captured recurrence experiment](e97-recurrent-operands-v1.md), source
`43646850`, results `cbef0e6e`, proved that on the actual first512-token call,
identical numeric operands and fixed input addresses produced the exact actor or
trainer result according to checkpoint-workspace selection. Mask-only changes
had no effect in that positive-zero-state case. Both standalone original calls
reproduced full-model output and final state bitwise. This is the repair's causal
basis, not another precision sweep.

Introduce **`--numerical-policy fp32-linear-v2`**: the same composite policy as v1,
but FP32 checkpoint workspace is now used during inference as well as training.
There is no independent operator workspace switch. `uniform_workspace` is an
internal derived control threaded through both chunked-projection and ordinary
E97/E88 adapter calls. It cannot be combined with legacy live state. Composite
v1, state-only FP32, and legacy defaults retain their previous workspace rules.

The compatibility workspace was previously BF16 in eval/no-grad despite live
FP32 state. This repair changes that workspace selection only; it does not claim
to distinguish allocation effects from compiled-storage-code effects.

Unchanged:

- Original BF16 weights, gradients, optimizer storage; no FP32 master model.
- FP32 Linear arithmetic with bounded recomputed casts; BF16 body activations;
  actual FP32 vocabulary readout; same163 Linear modules.
- FP32 live state and `(BLOCK_H=1,num_warps=4)` pinned recurrence.
- Highest FP32 matmul/no TF32 and disabled reduced BF16 reduction.
- Width3840,18 layers,H60/N64/V64; projection512,checkpoint16,MLP4096,CE128,
  checkpoint group3 and original token shapes, masks and full57-turn panel.

Policy versions are persisted/restored through checkpoint metadata; conflicting
explicit/saved versions fail closed. CPU tests verify both versions round-trip,
v2 derives uniform workspace, and configuring v1 restores its compatibility
selection. No trained checkpoint under v2 exists yet.

## Predeclared validation

**92 CPU tests passed,14 CUDA cases skipped on CPU.** Includes metadata/inheritance,
state/workspace guards, facade forwarding, original capsule/trace tests, and an
exact teacher-control comparator that rejects signed-zero, CE, layout-length and
coverage changes while intentionally allowing new actor measurements.

The bounded GPU run executes14 cases before full-model work: the previous13
state/production-H60/Linear/autograd/checkpointed-model cases, plus an eval/no-grad
real E97 layer case observing actual FP32 checkpoint and final-state storage across
512+17 projection chunks and a subsequent single token. Expected padded kernel
lengths are512,32,16 and valid lengths512,17,1. This case also checks finite outputs,
FP32 carry, BF16 parameters and absent gradients. The original v1 runner explicitly
excludes this new case to retain its frozen13-case scope.

Then two sequential fresh full57 workers must retain unchanged gates:

-2,711 finite actual scores; current actor/trainer max≤.05,p99≤.02.
-CE/head mean difference≤.0001; exact fresh-worker actor/teacher/CE repeats.
-163 FP32 Linears, FP32 actual heads, uniform workspace derived on all18 mixers.
-Unchanged effective parameters, zero autotuner entries, allocated HBM≤16GiB.
-**Teacher scores, CE and padded lengths must remain exact versus both saved v1
 measurement files**, SHA
 `64ba00df08cf48bc87e87ca1afc9c5257bfb88bb25addb63b03cedd88812e34e`.
 This is an additional causal control: the teacher already used FP32 workspace.

Historical behavior binding remains separate and is not repaired by substituting
new scores. Existing failures stay intact. The later token2519 same-input QKV
layout discrepancy is not declared fixed by this workspace repair.

Frozen recipe:
`3ab8a9237cd2102b34c83f125ed529758982f74a39aa6beef52c069c5811e364`.
Frozen plan:
`5c78873af0fedece63b86677503796674de151f156ee369d9a4dfd8d7f379205`.
The freezer proves unchanged inputs/limits versus the base state-only recipe after
removing only the composite identifier/scope. Preflight root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/uniform-workspace-v2-preflight`.
Driver `scripts/run_e97_uniform_workspace.sh`; freezer/auditor
`scripts/qualify_e97_uniform_workspace.py`.

One checked lease; immutable source export/inventory; source audit and lease cleanup
composed on success/failure. NUMA/local CUDA routing, isolated kernel/worker caches,
both expandable allocator variables. Kernel900s, two workers≤1,800s each,
outer5,100s, teardown30s. No retries, sweeps, overlapping GPU jobs or workers.

No optimizer updates, new rollouts, admission, promotion, full4B backward,
64K multi-document packing, eight-rank or restart qualification. The880+32 training
budgets remain closed. A numerical pass is not a behavioral gain or literal
cross-runner identity.

## First attempt: test observer failed before full-model execution

Source **`c99fed50`**, process `proc_8ce4`,68s,exit1. The13 previous CUDA cases
passed. The new workspace case failed before constructing its layer:
`AttributeError: module 'ndm.triton.e88_triton_backward' has no attribute
'e88_triton_forward'`. The backward function imports that symbol locally;
the observer must patch its defining `ndm.triton.e88_triton_forward` module.
Neither full-model worker started. This is a test-binding failure, not evidence
that the new workspace path passed or failed numerically.

Original artifacts remain in `uniform-workspace-v2`, including
`failure-audit.json`, JUnit SHA
`577b95cc0e891cae95706ef64e856362de7f0f8668a647423780cf1fe15ea79e`
and inventory SHA
`759bdefe276239360fc4da0195c01b3e511a20eb43baa320007613abf3c92d35`.
All17,689 source files verified before/after, and cleanup showed all eight GPUs
idle, no compute processes and no lease files without reaping.

The repair changes **only the test observer**, not model/kernel/policy code. A
new CPU regression patches the defining module and directly invokes the
actual autograd forward, proving it reaches that patched symbol. Repaired CPU
suite: **93 passed,14 CUDA skips**.

A manually reviewed fresh attempt is frozen under `uniform-workspace-v2-r2`
with the same recipe, plan, thresholds,14-case CUDA scope and two-worker budget;
no automated retry or expansion. Original failed artifacts are not overwritten.
## Audited second attempt — current-policy forward gate passed

Source **`020d6047`**, process `proc_7a2a`,1,022s,exit0. All14 CUDA cases
passed in41.75s before either full-model worker. The new workspace observer
confirmed FP32 checkpoint/final-state storage with valid/padded lengths
`512/512`, `17/32`, `1/16`.

Both fresh full57 workers completed. Their complete measurement files **and
summaries are byte-identical**. All2,711 trainer scores, CE means and padded lengths
also match the saved v1 trainer exactly. Actor and trainer are **not** bitwise
identical to one another; the bounded agreement gate passes:

| Gate | Composite v1 | Uniform workspace v2 | Unchanged limit |
|---|---:|---:|---:|
|Current actor/trainer maximum gap|.05184602737426758|**.028216838836669922**|≤.05|
|Current pair p99|.0010260593146085753|**.0007740734145045376**|≤.02|
|CE/head mean difference|5.2447273901623775e-8|**5.2447273901623775e-8**|≤.0001|
|Targets above maximum limit|1|**0**|0|

The former sole failing target, task004/turn1/position23, improved from.0518460
to.0163653. The largest residual is task004/turn1/position27 at.0282168 (previously
.0178742). Changes are therefore not uniformly improvements at every target.
The later same-input/different-layout projection discrepancy remains unresolved;
this pass does not establish instruction-level identity or eliminate all error.

Both workers report163 FP32-arithmetic Linears, actual FP32 heads, uniform workspace
on all18 mixers, pinned recurrence with zero tuner entries, highest FP32/no TF32,
and unchanged parameter fingerprint
`f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd`.
Peak allocated HBM was9,697,057,792 bytes in each worker, the same reported peak as
v1 and below16GiB. No optimizer updates occurred.

**Historical probability binding still fails**: actor versus recorded maximum
.0926365852355957 and teacher versus recorded maximum.09374070167541504.
The original behavior probabilities remain unchanged. `probability_path_passed`
and `training_eligible` remain false; only `current_policy_numerics_passed` is true.
This is not a new rollout dataset, admitted training policy, behavioral gain,
packed64K/full4B backward/eight-rank/cache/restart qualification, or reopening of
the880+32 budgets.

Independent CPU audit rechecked all57 identities, original2,711 historical values,
finiteness, score coverage, exact v1 teacher binding including CE bytes, both fresh
files, thresholds/quantiles, all14 JUnit cases, and all17,689 source files before
and after. `result-audit.json` retains the recomputation. Cleanup confirms eight
idle GPUs, no compute processes and no lease files without reaping.

| Artifact | SHA256 |
|---|---|
|Source inventory|`50b4df5fb83557a87c52745a63ea5c15df6b855db96322cf2139469642f174aa`|
|CUDA JUnit|`009e946d7a2997a6e6c6e05744c392322eac2562f64238b720db09796d40b274`|
|Each measurement file|`d4cc255e9e3f9dcaf19b8970a336ffabe6c4c1ff52bc8f16a9f3276fb257267d`|
|Each summary|`ac08e521df60a2df76f87fda5e76aebe868d64d8100b3bfc9371549809ba9ec7`|
|Audit|`b4438166109c6ec041c1fdac0bfdb80938b466c0810464fd9bbf093ddb003ab8`|

No further GPU experiment was launched. Next qualification must address actual
multi-document64K packs/reset isolation and full4B gradients, then the intended
rank/cache/restart contracts. Private score/token/activation artifacts remain
unpublished; failed first-attempt evidence is retained separately.
