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
buffers. The same named policy is threaded through the model, E97 facade,
optimized wrapper and custom autograd forward. FP32 requires FP32 S0 and retained
checkpoints; replay scratch and dS0 follow those checkpoints. Eval/no-grad's
**discarded** checkpoint workspace retains the historical projection dtype (v3);
it is never fed back as live state or retained for backward. Training, including
no-grad training previews, and every gradient-enabled eval use FP32 checkpoints.
Legacy explicitly
retains projection-dtype checkpoints, including when inference carries FP32 S0.

The SFT checkpoint records the nonlegacy policy in `sft_precision`. The loader
restores it even with a separate foundation architecture args file and rejects
an explicit conflicting config. Resume metadata comparison rejects changing
this policy mid-resume. Legacy metadata remains byte-structure compatible.
Changing policy for a new training stage still needs its own frozen budget.

Legacy training retains its BF16 state storage. Legacy inference retains both
FP32 carry and the original BF16 sparse checkpoint buffer. The first implementation
incorrectly changed that unused inference buffer to FP32; v2 restores the actual
legacy storage contract. Historical actor agreement must still be measured,
not assumed. This correction does not prove the buffer change caused v1's mismatch.

## Implementation and gradient path

- `ndm/recurrent_precision.py`: validation, uniform model policy, saved-policy restoration.
- `ndm/models/e88_fla_hybrid.py`: the single policy selects the initial state dtype;
  both ordinary and checkpointed projection paths pass that state onward.
- `ndm/triton/e88_triton_forward.py`: FP32-policy checkpoints use FP32; legacy
  checkpoints retain projection dtype. Final state still uses S0 dtype.
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

Eight mandatory executed CUDA cases in v3 (a skipped test fails the runner):

- BF16 projections, FP32 state, split edit, fused SiLU and normalization,
  nonzero final-state gradient, reset at32 and three padding steps. Full64 versus
  four connected16-step calls, plus an independent PyTorch FP32 arithmetic oracle.
  State relative L2 <=1e-4, initial-state gradient <=.005, projection/gate gradients
  <=.02. Full/chunked state <=1e-4, outputs <=.001, gradients <=.005. Require finite
  gradients, FP32 saved checkpoints, actual FP32 replay allocation and returned dS0.
- Legacy storage controls with both BF16 and FP32 initial carry; checkpoints
  must remain BF16 in both cases (the missing compatibility regression in v1).
- Actual checkpointed projection wrapper: FP32 carries, finite backward through
  the512-token boundaries, BF16 weights/parameter gradients and unchanged weights.

- Four additional workspace cases cover training/eval crossed with gradient
  enabled/disabled. Assert actual checkpoint allocation dtype, always-FP32 live
  final state, and FP32 state gradients whenever gradients are enabled. Only
  eval/no-grad uses the discarded legacy workspace.

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

### v1 partial results and preserved stop

`proc_10d1` stopped after500s, source `0fd96dc9`, in
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/fp32-recurrent-state-v1`.
All three original CUDA tests passed. Relative L2 against the FP32 oracle:
state `.000020747227608808316`, state gradient `.000393029855331406`,
projection/gate gradients `.0006805506418459117`. The512-token projection-wrapper
backward test also passed. These are bounded kernel/layer checks, not4B/64K proof.

The legacy full57 teacher probabilities matched the historical teacher **exactly**,
but actor replay differed from recorded behavior by up to `.11957478523254395`.
The baseline binding guard stopped the run; **the corrected full4B assay never
executed**. Unchanged parameters/finiteness passed, no optimizer updates occurred,
and both inner and controller source audits completed on failure. The lease was
released and all eight GPUs were idle afterward. This is not a numerical failure
of the unexecuted FP32 full-model assay.

Failure audit SHA: `b8a0a32bfef0004c802cb9a983405446164007ac2b65ba71935888a29456f2a1`.
Kernel JUnit SHA: `a8a0b224902174bde4feffaf2341b8404078dbc31683484086439a1f454e5753`.
Legacy summary SHA: `b7305c90d440221eefbc041c2ab9708d918400fd55b597d9d1f6267a19642784`.
Legacy measurements SHA: `9f86c223b46220be948db4fe18168011e0b6fef92c031d120884e5bdb97f3832`.

### v2 compatibility repair

The v1 policy inferred all checkpoint dtypes from S0 and therefore failed to keep
legacy inference allocation unchanged. v2 threads the same `recurrent_state_precision`
setting explicitly through both projection paths and all facades, preserving BF16
legacy checkpoints even with FP32 carry. Added CPU policy/plumbing tests and the
missing FP32-carry/BF16-checkpoint CUDA control. CPU suite: **59 passed, 4 skipped**.
Repeat the same frozen full57 recipes and thresholds in a new immutable export
and new output root; no automatic retry or changes to old evidence.

### v2 measured outcome

`proc_9553` completed all four CUDA tests and both full57 assays, then failed the
numerical gate after1,192s. Source `672fa1e8`; artifacts in
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/fp32-recurrent-state-v2`.
The legacy summary and measurements are byte-identical to the earlier bound
full57 reproduction, including exact historical actor agreement.

FP32 training-to-recorded-actor maximum improved from `.11980986595153809` to
`.04821127653121948` (passes .05); p99 `.0007128755562007432` and CE/head delta
`4.135482429063115e-08` pass. However FP32 actor replay changed by
`.11957478523254395`, exactly reproducing v1's actor probabilities. Actual FP32
teacher versus actual FP32 actor maximum is `.1196904182434082`, with two tokens
above .05. **The full gate remains failed**; training-reference improvement alone
is not runtime consistency or RL qualification. Parameters were unchanged and
all inner/controller source audits completed; zero updates, lease released.

Audit SHA: `22455a4413a4258a17b88491d1770a96e8defb8efcbd3ae835202f85ccd1e423`.
FP32 summary SHA: `7601c1e9309c98b965042503e22db64691cddcaf78d45961646bfde2a61ce5e5`.
FP32 measurements SHA: `3f388a0c1673ef5ddb740611220646a75e14b100bdbb982ab507cd4a699be11e`.
Legacy summary SHA: `4ca426dc7dda5efcd0665d36d207bcd47d86475365dbc83f48f61d308e87043b`.
Legacy measurements SHA: `a2e3399ba8974310db27df9094596237e16a2459050e3f8f8169bd197d6dc9f9`.

### v3 discarded inference workspace compatibility

The controlled comparison reproduces an actor difference when the unused
checkpoint workspace changes dtype, while live inference state was already
FP32. This does not distinguish allocator effects from compiled-kernel effects,
or explain every older historical replay variation.

v3 preserves the historical workspace only for **eval with gradients disabled**.
No state/cache tensor is narrowed and no outputs are replaced. Every checkpoint
that can be used in backward remains FP32, including gradient-enabled eval;
no-grad training previews also keep FP32 workspace to match actual training.
Initial-state validation remains based on the requested policy, not the workspace
exception. This is automatic workspace handling, not an additional user switch.

CPU regressions: **63 passed, 8 CUDA skips**. The same two full57 recipes, bounds
and thresholds were frozen again, with the eight CUDA cases above.

### v3 completed: actor restored, full gate still failed

`proc_3bf4` completed after971s from `e1a8e4eb`, in
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/fp32-recurrent-state-v3`.
All **eight CUDA cases passed**, including actual workspace allocations and
FP32 state gradients in gradient-enabled eval. Legacy results again match the
historical full57 reproduction byte-for-byte.

FP32 actor replay now matches all recorded probabilities **exactly**. However,
FP32 trainer-to-actor maximum is **`.12353801727294922`**, failing .05. Two of2,711
tokens exceed .05; p99 `.0006216555833816547`, CE/head delta
`4.85139758489659e-08`, finiteness and unchanged parameters pass. The two outliers
are task004/turn1/generated-position24 (`.07596421241760254`) and
task014/turn4/generated-position15 (`.12353801727294922`).

Teacher probabilities changed from v2 by up to `.12353801727294922` despite
retaining the requested FP32 training checkpoint policy. Thus v2's passing
training-reference maximum **did not survive** the inference-workspace change.
The state-policy implementation and bounded gradient tests do not establish
whole-model actor/trainer numerical consistency. There is no sufficient causal
explanation yet; neither the earlier .04821 result nor a successful process phase
may be promoted to a full qualification claim.

Independent audit verified all eight executed cases, all57/2,711 paired records,
both inner/controller source audits on failure, and zero updates. The GPU lease
was released and all GPUs were idle. No additional run was launched.

Audit SHA: `fb316fbf40f113a3c84feaf93eae5e9f4a16b01274588c3f029622e6787f5821`.
FP32 summary SHA: `f18e924db2cf9064468671f286fde55a4c39f7db17216523f6aba52ecf245605`.
FP32 measurements SHA: `c6c0d421125ff08d86a06a915d7909d6ffc567328294d3f1367c00f689457f95`.
Kernel JUnit SHA: `577f5f2760b9663ec7ba4d42f65c645396053d66070b78d863a16a4e0d0626f3`.

A post-result CPU comparison to the four existing reference-bound first-mixer
profiles did not establish equality to either full-eval or masked-padded-eval
across all turns. This does not supply a root cause or new qualification;
`prior-profile-comparison.json` SHA:
`34e6ffc8b8fbaaa470b166baf09b5798f8af0593cc2db69a9cb682cfc553427e`.

All earlier failures remain unchanged. The implementation remains opt-in and
unqualified for production training/RL. Known BF16 projection/layout differences
are a remaining candidate, not a proven complete explanation. Keep the existing
weights; these results do not justify repeating the prior long-context training.

Full4B backward, 64K peak memory/performance, eight-rank integration and restart
qualification are still required before training under this policy. Observed BF16 projection-layout differences may remain even with
consistent FP32 state. This candidate is not a proven probability fix, a model
improvement, an RL qualification, or authorization to reopen closed budgets.
