# E97 fixed recurrent launch candidate

## Concrete mechanism found

`ndm/triton/e88_triton_forward.py::_autotune_kernel` is a separate Python,
wall-clock-based tuner, not the `triton.autotune` machinery covered by the
repository's pin registry. E97 has **60 heads per layer**, which enters this tuner:

- H<16: fixed `(BLOCK_H=1, num_warps=4)`.
- 16<=H<64: time candidate launches and cache the fastest in the process.
- H>=64: fixed `(1,1)`.

For N=V=64, the timed choices are `(1,2), (1,4), (2,2), (2,4), (4,2), (4,4)`.
The cache key omits initial/checkpoint-state dtype, tensor strides, valid length,
and output-gate presence. Its winner can therefore depend on timing and the first
call that populates a key. Forward can choose a different arithmetic geometry
from backward replay, whose H60 default was already `(1,4)`.

The earlier two-/four-head GPU fixtures bypassed this production branch. The new
CPU regression demonstrates all48 timed/warmup launches for H60, cached reuse
across different valid lengths, and the no-tuning H2 branch. This establishes the
code mechanism, not yet a complete causal explanation of every historical error.

## Fix under test: same user switch

The existing `recurrent_state_precision="fp32"` policy now fixes forward and
backward replay to **one head per program, four warps**, independently of shape,
process timing and allocator history. No new operator tuning switch is added.
The choice passes through the optimized adapter and custom autograd context; it
also applies to inference even when its discarded workspace uses legacy dtype.
Low-level explicit launch overrides remain available for kernel experiments.

Legacy mode retains the historical tuner for comparison. Head count, model
width/depth, weights, optimizer storage, projection chunks, sparse checkpoint
intervals, masks and loss thresholds are unchanged. This is not a claim that
`(1,4)` is performance-optimal, or that every BF16 projection/layout difference
has been removed.

Kernel identity: `e88-sequential-fp32-bh1-nw4-v1`. New SFT precision metadata
records this identity; resume's exact policy comparison rejects an older policy
without it. Checkpoint reconstruction rejects an explicitly unsupported kernel
identity. No real FP32 SFT checkpoint or weight update has been produced.

## Frozen qualification

CPU: **71 passed, 10 CUDA skips**. An initial new test imported the package's
function alias instead of the module; that harness-only import failure was fixed
with `importlib.import_module` before any GPU execution.

Ten mandatory CUDA cases:

- The prior eight FP32 storage/gradient/workspace cases.
- Production H60/N64/V64 at T32 and T512, with fused SiLU/normalization, split edit,
  reset/padding controls and nonzero final-state gradients. Patch the timer to
  raise if entered. Assert backward receives `(1,4)`. Compare to the independent
  FP32 PyTorch recurrence with the existing bounds: state relative L2<=1e-4,
  dS0<=.005, projection/gate gradients<=.02, all finite.
- Within the T32 case, replay identical tensors with each of the six explicit
  configurations twice. Require exact within-configuration repeats; report
  across-configuration state/output differences rather than assume they exist.
  These are synthetic production-geometry operands, not captured failing tokens.

Then **two fresh workers**, each replaying the unchanged full57-turn/2,711-token
selection with the pinned FP32 policy. Separate Triton caches; no inherited tuner
cache. Require zero entries in this recurrence's legacy autotune cache. Compare
actual current-policy actor and trainer log probabilities: max<=.05, p99<=.02,
CE/head mean delta<=.0001, finite, unchanged parameters. Require **exact actor and
teacher probabilities and CE means across fresh workers**.

### Historical probabilities are not replaced

Fixing launch geometry changes the numerical policy and may change historical
actor probabilities. The original `probability_path_passed` checks and recorded
behavior values remain intact and are reported separately. Current-policy paired
scores are diagnostic forced-token scores, **not replacement recorded behavior
probabilities** and not a license to train on the old rollout log probabilities.
A current-policy pass cannot reclassify any earlier failed assay. No RL optimizer,
new rollout, data admission or training budget is included.

Input recipe SHA (same selection and limits):
`82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9`.
The additional `plan.json`, frozen source and reported kernel identity bind the
new execution policy. Artifact root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/fixed-recurrent-kernel-v1`.

Driver: `scripts/run_e97_fixed_recurrent_kernel.sh`.
Audit: `scripts/audit_e97_fixed_recurrent_kernel.py`.
Tests: `tests/test_e97_recurrent_precision.py`,
`tests/test_e97_fp32_state_layer_cuda.py`,
`tests/test_e97_fixed_recurrent_kernel_audit.py`.

One checked GPU lease; explicit local device and NUMA binding; isolated caches;
both expandable allocator flags. Immutable source audited before/after, including
child failure, with composed lease cleanup. Kernel900s, each worker1,200s,
outer3,900s, teardown30s. No retries, requeues, concurrent GPU jobs or communicator
reuse. Work is single-threaded in the attended session, with no WG service.

Results pending. Full4B backward, 64K memory/performance, eight-rank integration,
restart qualification and behavioral capability remain separate requirements.
The original unqualified FP32-state results are preserved in
[e97-fp32-recurrent-state-v1.md](e97-fp32-recurrent-state-v1.md).
