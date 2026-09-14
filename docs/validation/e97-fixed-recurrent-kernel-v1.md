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

## Completed result: repeatable, still inconsistent

Numerical source **`f149c77c`**, managed `proc_624c`, PID1917494. The bounded
program completed in648s and exited1 at the declared numerical gate, not from a
runtime crash. Both full57 measurements were completed; no additional GPU run
was launched afterward.

All **ten CUDA cases passed**. Production H60 oracle comparisons:

| Tokens | State relative L2 | dS0 relative L2 | Worst projection/gate gradient relative L2 |
| --- | ---: | ---: | ---: |
| 32 | 1.95754e-5 | .000263373 | .000422972 |
| 512 | 2.06559e-5 | .0000979905 | .000417409 |

Every explicit launch configuration repeated exactly in the synthetic T32 test.
Against `(1,4)`, the other five configurations produced BF16 output differences
up to `1.9073486328125e-6` and FP32 final-state differences up to
`4.76837158203125e-7`. Thus identical operands can produce different numerical
results across these launch configurations. This does not establish that every
historical whole-model discrepancy arose from the tuner or that memory-access
correctness has been exhaustively checked.

Both fresh full-model workers produced **byte-identical measurement files and
summaries**, not merely similar maxima. Each reported zero legacy recurrence
autotune-cache entries. All2,711 actor scores, teacher scores and per-turn CE means
repeated exactly; weights remained unchanged.

| Measure, both workers | Result | Verdict |
| --- | ---: | --- |
| Current actor versus current trainer, max | .12353801727294922 | **FAIL**, limit .05 |
| Current actor versus current trainer, p99 | .0007408213801682023 | pass, limit .02 |
| CE/head mean delta | 4.135482429063115e-8 | pass, limit .0001 |
| Actor versus historical recorded behavior, max | .13056039810180664 | **FAIL**, limit .0001 |
| Trainer versus historical recorded behavior, max | .04821127653121948 | pass; not current-policy agreement |
| Peak allocated HBM | 8,566,558,208 bytes | measured, not64K qualification |

Three current-policy outliers (zero-based generated positions):

- `onpolicy-task-004-sample-0`, turn1, position29: `.11964964866638184`.
- `onpolicy-task-008-sample-0`, turn1, position23: `.07160699367523193`.
- `onpolicy-task-014-sample-0`, turn4, position15: `.12353801727294922`.

An independent CPU audit rechecked coverage, historical probability identity,
file equality, artifact hashes and all before/after source-check logs, including
the failed controller exit. Each source check covered17,666 files. The new teacher
vectors also equal v2's teacher vectors exactly. That equality does **not** identify
v2's launch choices, which were not recorded.

The lease directory contained no active lease files, without running a reaper;
all eight GPUs were idle with no compute processes. Optimizer updates: **zero**.

### Artifact identities

- Source inventory: `2b759c4e39261064659a1a0c440b3bec72c41160fe88e1ef290988199a43c2dc`.
- CUDA JUnit: `8396ee2c3561fed58fecf2d233a0ed17e6c177203a6965687bc1d0c410627fb7`.
- Each worker's measurements: `6745d29adda5c7b537e9daa615822aa8abdf87cc42938075e6c20c8ffd8698fc`.
- Each worker's summary: `0ef8d58e4cf00f3b28e2c17451ba0f91544d5cb2eaadc4f016f398ca16da8220`.
- `audit.json`: `19566370e5d354a41fbef94a4ff493648cb0b1882de4b137509eea44890dd95a`.
- Independent retained CPU evidence: `result-audit.json` in the artifact root.

### Interpretation and stop

The pin removes timing-based recurrent launch selection and supplies a
reproducible remaining failure in this bounded check. **It is not a complete
actor/trainer consistency fix**, a universal determinism guarantee, or proof of
long-context numerical stability. The old and current-policy gates both remain
failed. No historical probabilities were replaced.

The next causal target is the remaining layout-dependent discrepancy with this
launch policy held fixed: bind captured projection/recurrence operands to these
current-policy references before interpreting batching, discarded-checkpoint
store code or allocation controls. No additional repair or sweep was started.
Full4B backward, 64K memory/performance, eight-rank integration, restart
qualification and behavioral capability remain separate requirements. The
original unqualified FP32-state results are preserved in
[e97-fp32-recurrent-state-v1.md](e97-fp32-recurrent-state-v1.md).
