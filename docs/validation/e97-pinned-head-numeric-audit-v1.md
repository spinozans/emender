# Pinned E97 head-arithmetic diagnostic

## Question and fixed authority

After pinning recurrence to `(BLOCK_H=1, num_warps=4)`, two fresh full57 workers
produced byte-identical scores, but the actual actor/trainer maximum remained
`.12353801727294922`, with three tokens above `.05`. See
[e97-fixed-recurrent-kernel-v1.md](e97-fixed-recurrent-kernel-v1.md).

This experiment asks whether the remaining large gap is amplified by the final
BF16 vocabulary projection/store, or persists in the upstream representation even
when the readout uses FP32 arithmetic. It does not change model outputs.

Binding reference: both pinned workers' measurements SHA
`6745d29adda5c7b537e9daa615822aa8abdf87cc42938075e6c20c8ffd8698fc`;
each summary SHA
`0ef8d58e4cf00f3b28e2c17451ba0f91544d5cb2eaadc4f016f398ca16da8220`.
The underlying input recipe is
`82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9`.
The freezer verifies that **only observer configuration/scope** differs from that
recipe. Weights, pinned FP32 recurrence, discarded inference-workspace policy,
projection/checkpoint chunks, masks, all57 turns and2,711 targets remain fixed.

## Read-only measurements

Extend the existing head observer with a diagnostic-only `numeric_audit` option:

1. Preserve actual BF16 logits and native log probabilities.
2. Compute a shadow FP32 vocabulary projection, disabling autocast and TF32;
   at most128 rows and4,096 vocabulary rows per temporary weight block.
3. Re-quantize those shadow logits to BF16 before log-softmax. Measure whether
   the large discrepancy reappears under this controlled final store.
4. Compute independent **CPU FP64 selected dot products** from the actual BF16
   hidden vectors and selected weight/bias rows. These check selected logits,
   not an FP64 whole-vocabulary normalizer. They do not copy the complete model
   into FP32/FP64 or change persistent weights/optimizer storage.
5. Compare actor/teacher hidden-vector differences, selected-logit differences,
   FP32-versus-FP64 selected-logit errors, and actual versus re-quantized logits.

Hooks return `None`. The shadow/rounded tensors are never substituted for actual
logits. Native scores and CE means must bind the saved pinned-policy reference
within1e-4, and the observer's own native scores must bind the actual actor/teacher
scores. Otherwise the diagnostic is not ready for interpretation.

A CPU control demonstrates that a small hidden difference can cross a BF16 output
rounding boundary and produce a log-probability gap>.05 while the FP32 gap is<.001.
That is a mechanism demonstration, **not evidence it explains these model errors**.

## Frozen bounds and interpretation

One fresh full57 GPU worker, no sweep or automatic follow-on. Current-policy
pair bounds remain max<=.05/p99<=.02 when assessing shadow scores. A shadow pass
is only a counterfactual finding, not a changed-model qualification. Actual
historical and current-policy failures remain intact; no probabilities are
replaced, no training data is admitted and optimizer updates remain zero.

Require finite diagnostics, unchanged parameter fingerprint, zero recurrence
autotune-cache entries, complete coverage and peak allocated HBM<=12GiB. One
checked lease, local CUDA device and NUMA binding, isolated Triton cache, both
expandable allocator variables, CUBLAS workspace `:4096:8`, highest FP32 matmul
precision and no BF16 reduced-precision reduction. Worker3,600s, outer3,900s,
teardown30s. Source audited before/after and on failed child exit; lease cleanup
composed with those audits. No WG service or delegated work.

Frozen probe recipe SHA:
`eb99ad750df1722d7cd58c36ab8844d4c9e9655d76ae9774eed88c069186cf96`.
The CPU authority preflight passed and is retained in
`pinned-head-numeric-audit-v1-preflight` beside the run root.

CPU validation: **88 passed, 10 CUDA skips**. Includes read-only hook checks with
the numeric extension enabled, chunked FP32 arithmetic under autocast, selected
FP64 dots checked independently with Python `math.fsum`, final-store sensitivity,
and fail-closed coverage/reference/nonfinite/dot-error tests.

Artifacts:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/pinned-head-numeric-audit-v1`.
Driver `scripts/run_e97_pinned_head_audit.sh`; freezer/auditor
`scripts/audit_e97_pinned_head.py`; diagnostic
`scripts/e97_head_precision_probe.py`.

Results pending. No numerical-model repair, new rollout, learning/capability
claim, full4B backward, 64K, multirank or restart qualification is included.
