# E97 actual recurrence operands and bounded control isolation

## Authority and question

The [first-divergence trace](e97-first-divergence-v1.md) found an early first-layer
mixer-readout discrepancy at prefix token1 despite bitwise-equal raw projections
throughout all2,519 prefill rows. The later QKV mismatch at2519 cannot explain it.

Capture the first actual recurrence call after the adapter has selected its
operands, masks, flags and workspace policy. Both paths' first projection calls
were verified from the prior receipts to contain512 rows. Geometry remains
T512/B1/H60/N64/V64. Use the same task004/turn1, unchanged composite numerical
policy, checkpoint, full recorded turn, padding and chunk settings.

Reference trace summary SHA:
`34c7b1dd33ef245b3c5c63b99071a0b6912d38cbed2088537ebc5631dcc88c4a`.
The underlying actual-probability reference remains the two identical composite
workers, not historical recorded behavior probabilities substituted with new ones.

## Capture and binding

`e97_recurrent_call_capture.Capture` temporarily observes the shared
`e88_triton_optimized.e88_triton` call. It binds the real function signature,
snapshots actual tensor inputs and resolved scalar controls, calls the original
function without changing arguments, and returns the **same original result
objects**. Only its first call is captured; the original function is restored.
Inputs must remain unchanged after execution.

Two full-model executions per path (four total) must reproduce:

- Every original current-reference score, plus teacher CE mean/padded length.
- All28 existing fine-trace tensor hashes.
- The captured kernel output, flattened over its first512 rows, versus the
  actual tensor entering the output projection.
- Captured input/output values, strides, offsets, alias relationships and controls
  exactly across repeats (original allocation addresses are recorded, not required
  to repeat).

`scripts/e97_tensor_capsule.py` preserves full underlying storage bytes once per
storage, and reconstructs shared views with original dtype/shape/stride/offset.
It does **not** restore GPU addresses. Unreferenced storage bytes are retained
but are not used as a repeatability requirement; logical view hashes and layouts
are checked. Each capsule is bounded to256MiB. Shape/dtype/storage bounds are
validated before destination allocation. Capsules and activations remain private.

## Minimal replay and causal controls

A second, fresh process loads only the captured operands—no4B model. Both
original actor and teacher calls must reproduce captured **output and final-state
bytes exactly**, twice each. Failure stops before interpretation.

Only if numeric input values, dtypes, strides, offsets and alias relationships
match, all other controls match, and the observed masks are equivalent on a
positive-zero initial state, execute this fixed two-by-two matrix:

| Masks | Workspace selection |
|---|---|
|captured actor|captured actor|
|captured teacher|captured actor|
|captured actor|captured teacher|
|captured teacher|captured teacher|

Each cell repeats twice. This is at most12 small calls including the four original
replays. The matrix holds the same actor numeric input allocations throughout and
borrows only the captured mask tensors. It uses the two existing workspace
selections (`legacy` discarded-checkpoint storage versus `fp32`), not a new model
precision setting. Both live initial/final states remain FP32. It requires all
valid-mask entries true, no reset after position0, and positive-zero initial state;
these mask equivalences must not be generalized to nonzero state or arbitrary
packed records. If numeric inputs already differ, retain that result and skip the
control matrix rather than manufacture equal operands.

**Limit:** workspace changes affect allocation and generated storage code together.
Mask controls also change kernel specialization and host checks. This experiment
can isolate dependence on those controls, but does not by itself distinguish an
allocation/address effect from code generation. Output/workspace addresses are
not held fixed. No tolerance fallback or unseen combinations are allowed.

## Validation and budget

CPU: **18 passed,3 CUDA skips**. Tests cover storage aliases/strides/offsets,
scalars/empty/None tensors, budget rejection before destination copies, capture
return identity/restoration, mask-equivalence guards, changed numeric inputs,
and ND/signed-zero output differences, plus the prior trace and policy tests.

An initial CPU test exposed scalar byte-view hashing in the earlier trace helper;
the failure is retained in `recurrent-operands-cpu-v1/failure.json`. Hashing now
flattens contiguous data before byte viewing while preserving the original
shape/dtype header. Existing matrix hashes are unchanged.

Plan SHA:
`d29f851e2b3beabc2b767617a8ce278ddfa143a573f03d7022a431d3f56eb4f8`.
CPU authority preflight retained in `recurrent-operands-v1-preflight`.
Run root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/recurrent-operands-v1`.
Driver `scripts/run_e97_recurrent_operands.sh`; executor
`scripts/diagnose_e97_recurrent_operands.py`.

One lease, two **sequential** fresh processes, no overlapping GPU work or delegated
workers. Immutable export/inventory and composed before/after source audits plus
lease release including failures. Local CUDA selection, NUMA binding, isolated
capture/replay caches, both expandable allocator variables. Capture900s, replay300s,
outer1,500s, teardown30s. Allocated HBM limits: capture16GiB, micro2GiB. No retries,
additional candidate, optimizer updates, new rollout probabilities, admission,
64K/gradient/restart qualification or reopening of the880+32 training budgets.

## Audited result — workspace selection isolates the early difference

Source **`43646850`**, process `proc_3911`,243s,exit0. Four full-model captures
preserved all native scores, teacher CE, fine-trace hashes and kernel-to-readout
binding exactly. Numeric kernel operands, their layouts and aliases matched
between actor and teacher. Both initial states were FP32 positive zero; both
launches used `(1,4)`, SiLU-QKV and L2 normalization. Only masks and workspace
selection differed.

A fresh process reproduced both captured outputs **and final states bitwise**
twice each without the4B model. All twelve minimal calls completed. The fixed
control matrix, repeated exactly, gave:

| Masks | Workspace | Exact output and final-state match |
|---|---|---|
|actor|actor/BF16 checkpoints|actor|
|teacher|actor/BF16 checkpoints|actor|
|actor|teacher/FP32 checkpoints|teacher|
|teacher|teacher/FP32 checkpoints|teacher|

**On these real captured operands, mask changes alone had no effect. Workspace
selection alone switched both results exactly between actor and teacher.**
Numeric input addresses were identical throughout the matrix. This is causal
evidence for the workspace selection, not a guessed explanation from a final
score difference. It still does not separate changed allocation from generated
storage code, nor prove mask invariance for nonzero state or different streams.

Between original modes, first output difference was at `[1,0,26,11]` (flattened
feature1675), matching the first-divergence trace. There were358 differing BF16
output elements, maximum.000244140625. The FP32 final states differed in930
elements, maximum3.5762786865234375e-7. The compatibility workspace is therefore
not numerically inert even though live state is FP32 and its checkpoints are
unused during inference.

This justifies separately testing a composite policy with **uniform FP32
workspace in actor and trainer**. It does not yet qualify that repair across the
full57 panel. The later token2519 QKV projection discrepancy remains distinct.

Independent CPU audit verified capsule file hashes, all capture/replay binding
receipts, loaded all four matrix outputs and checked their exact reference
matches, fixed numeric input addresses, and all17,686 source files before/after.
Peak allocated HBM: capture9,538,251,264 bytes; standalone replay86,636,544 bytes.
Parameter fingerprint remained
`f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd`;
buffers unchanged, no gradients or recurrence autotune entries. Cleanup recorded
all GPUs idle, no compute processes and no lease files without reaping.

| Artifact | SHA256 |
|---|---|
|Source inventory|`b1dbe0b9f5ddd0639f7a29cfd72526d1536e0a387d15507ed7fd1513c96c664e`|
|Capture summary|`ca7cf4e638fd22c95d3e11993f78dc38865a0fc742211cfbd26a97e5168a1a6f`|
|Matrix preflight|`bfbf8ac3746edf1f3ef1cfa7507d4cb4e6f3cd547569af7f4dae20051f469851`|
|Replay summary|`61a2aff628a23cee516bef28ffba88bcf3527f4f030966eee8d5a1412da724c4`|

`result-audit.json` retains independent checks. Private operands/output tensors
remain unpublished. Existing numerical and behavioral gates remain failed.
Zero optimizer updates; no admission, promotion or expanded training budget.
