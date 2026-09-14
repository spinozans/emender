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

Results pending; current full-panel numerical gate remains failed.
