# E97 unified FP32 recurrent-state candidate

## One setting, not a full-model conversion

`recurrent_state_precision = "fp32"` selects FP32 for:

1. initial/final state and every internal projection-chunk handoff;
2. sparse recurrent checkpoints saved for backward;
3. backward segment replay scratch;
4. incoming/outgoing recurrent-state gradients.

Persistent parameters, ordinary projections/activations and their gradients,
and the existing BF16 Schedule-Free optimizer storage remain unchanged. There
are no FP32 master weights or CPU optimizer arithmetic. Internal recurrent
arithmetic was already FP32; the correction removes narrowing at state-storage
boundaries. Chunk sizes and the 16-step sparse checkpoint interval are unchanged.
The separate linear/chunked-E97 backend and CPU model execution reject this new
policy rather than silently claiming it. This qualification targets sequential
CUDA/Triton E97, not Frontier/ROCm or resilient-distributed training.

SFT entry point: `scripts/train_e97_4b_pi_sft.py --recurrent-state-precision fp32`.
The other choice is `legacy`. Omitting the option inherits the loaded policy;
old checkpoints remain legacy. Model construction uses the same setting in
`layer_kwargs`. No additional precision sub-switches exist for individual state
buffers. Low-level Triton calls take state precision from `S0.dtype`; sparse
checkpoints, replay scratch and dS0 follow it, not the projection dtype.

The SFT checkpoint records the nonlegacy policy in `sft_precision`. The loader
restores it even with a separate foundation architecture args file and rejects
an explicit conflicting config. Resume metadata comparison rejects changing
this policy mid-resume. Legacy metadata remains byte-structure compatible.
Changing policy for a new training stage still needs its own frozen budget.

Legacy training retains its BF16 state storage. Inference already carries FP32
state; its previously BF16, unused sparse checkpoint buffer now also follows S0.
Thus legacy inference arithmetic is intended to remain unchanged, but allocation
and compiled kernel details are not claimed identical to the immutable old source.
Historical actor agreement must still be measured, not assumed.

## Implementation and gradient path

- `ndm/recurrent_precision.py`: validation, uniform model policy, saved-policy restoration.
- `ndm/models/e88_fla_hybrid.py`: the single policy selects the initial state dtype;
  both ordinary and checkpointed projection paths pass that state onward.
- `ndm/triton/e88_triton_forward.py`: checkpoints use S0 dtype, as does final state.
- `ndm/triton/e88_triton_backward.py`: checkpoint dtype controls replay scratch,
  zero dS-final defaults and returned dS0. Every custom-autograd branch returns
  dS0 directly. Parameter/projection gradients retain their existing dtypes.
- `ndm/e97.py` and the SFT entry point: reconstruction and persistence.

This is not TBPTT. The foundation args say `tbptt:false`; 2,048-token foundation
sequences, 512-token projection chunks and 16-step sparse checkpoints are distinct.
Neither inspecting the graph nor the old tolerance checks prove exact replay;
new numerical tests are necessary. Historical weights and training remain
preserved; no evidence establishes that the previous ~1B-token continuation
must be repeated.

## Frozen first qualification (no optimizer updates)

`tests/test_e97_recurrent_precision.py` checks configuration, inheritance,
checkpoint reconstruction, state/activation dtype separation, and CUDA recurrence.
`tests/test_e97_fp32_state_layer_cuda.py` exercises a real E97 layer through
1,040 tokens and 512-token checkpointed projection chunks.

Three mandatory executed CUDA tests (a skipped test fails the runner):

- BF16 projections, FP32 state, split edit, fused SiLU and normalization,
  nonzero final-state gradient, reset at32 and three padding steps. Full64 versus
  four connected16-step calls, plus an independent PyTorch FP32 arithmetic oracle.
  State relative L2 <=1e-4, initial-state gradient <=.005, projection/gate gradients
  <=.02. Full/chunked state <=1e-4, outputs <=.001, gradients <=.005. Require finite
  gradients, FP32 saved checkpoints, actual FP32 replay allocation and returned dS0.
- Legacy BF16 state/checkpoint/dS0 storage control.
- Actual checkpointed projection wrapper: FP32 carries, finite backward through
  the512-token boundaries, BF16 weights/parameter gradients and unchanged weights.

Then run the **entire original57-turn/2,711-token probability selection** from
`qualify_e97_native_rl_logprobs.py` in fresh legacy and FP32 workers. The legacy
recipe is byte-identical to the original (`ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f`);
its actor must bind to history, and every teacher probability must bind within
.0001 to the original measurements before interpreting the intervention. The
FP32 recipe differs only by the explicit state policy and scope description.
Require historical replay max <=.0001, trainer max <=.05, trainer p99 <=.02,
CE/head mean delta <=.0001, finiteness and unchanged weights. No threshold
relaxation or substitution of counterfactual behavior probabilities. Original
failed runs remain failures. Recipe SHA:
`82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9`.

`scripts/run_e97_fp32_recurrent_state.sh` checks an immutable export before/after,
including child failure, and composes lease release with that audit. One checked
GPU lease, explicit local device/NUMA binding, separate Triton caches and fresh
workers for kernel/legacy/FP32 stages; both expandable allocator flags. No retries,
requeues, overlapping GPU jobs, or communicator reuse. Kernel stage900s, each
assay1,200s, outer3,900s, teardown30s. No optimizer steps or production training.

## Status

CPU regressions: **124 passed, 3 skipped** in155s (before the final additional
policy-inheritance test). Final focused suite: **6 passed, 3 CUDA skips**.
The earlier synchronous CPU invocation exceeded its
120s tool timeout; that is not a numerical verdict. It was replaced by a managed,
900s-bounded CPU invocation with explicit thread limits.

GPU results pending. Full4B backward, 64K peak memory/performance, eight-rank
integration and restart qualification are still required before training under
this policy. Observed BF16 projection-layout differences may remain even with
consistent FP32 state. This candidate is not a proven probability fix, a model
improvement, an RL qualification, or authorization to reopen closed budgets.
