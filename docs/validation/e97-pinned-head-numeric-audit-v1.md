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

## Completed: head rounding explains amplification, but not every failure

Source **`d568d84c`**, managed `proc_a742`, PID2010863; completed in322s with
exit0. This means the **diagnostic's binding/safety checks passed**, not that the
model's numerical gate passed.

All57 turns/2,711 targets were measured. Actual actor probabilities, teacher
probabilities and CE means matched the pinned uninstrumented reference exactly.
The observer's own native probabilities also matched exactly: maximum binding
difference **0**. Parameters stayed unchanged, recurrence autotune-cache entries
remained zero, and peak allocated HBM was8,629,706,240 bytes. No model output was
replaced.

| Readout used for diagnostic scores | Max actor/trainer log-probability gap | p99 | Tokens above .05 |
| --- | ---: | ---: | ---: |
| Actual BF16 | .12353801727294922 | .0007408213801682023 | 3 |
| Shadow FP32 | .07699280977249146 | .0011633686721325034 | 1 |
| Shadow FP32, then stored in BF16 | .12353801727294922 | .000743687152862549 | 3 |

The FP32 shadow therefore **still fails** the unchanged max<=.05 criterion.
Rounding can amplify or conceal discrepancies; the p99 does not improve here.

### Three actual outliers

Generated positions below are zero-based. All are from sample0.

| Task / turn / position | Actual gap | FP32 shadow gap | Re-quantized gap | BF16 selected-logit delta | FP64 selected-dot delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 004 / 1 / 29 | .119649649 | .009301186 | .119650364 | .125 | .010394634 |
| 008 / 1 / 23 | .071606994 | .076992810 | .071606755 | .0625 | .072576367 |
| 014 / 4 / 15 | .123538017 | .001582146 | .123538017 | .125 | .014000899 |

For004 and014, higher-precision readout substantially reduces the discrepancy;
reintroducing a BF16 final store recreates the large gaps while holding the
captured hidden vectors fixed. This isolates substantial readout rounding
amplification. Native BF16 GEMM logits and re-quantized FP32 logits are not
identical across the whole vocabulary (maximum difference .03125 at these
positions), so the experiment does not assert that their accumulation paths are
identical.

For008, the discrepancy persists in FP32. Its selected-dot delta is
`.07257636671420187` in CPU FP64, while the FP32-versus-FP64 selected-dot error at
that token is only `5.370238795876503e-7`. Its incoming hidden vectors already
differ (relative L2 `.011663807556033134`, absolute max `.046875`). Thus this is
not explained by final BF16 logit storage alone; a substantial difference exists
upstream of the head.

Across all measured selected dots, maximum FP32-versus-FP64 error was
`1.2320757377892733e-5`. Maximum actor/teacher hidden relative L2 was
`.019574584439396858`. These are local measurements, not long-context error-growth
bounds or a full FP64 probability reference.

### Audits and artifacts

An independent CPU audit rechecked actual-score/CE equality, auxiliary native
binding, finite values, counts, maxima and threshold-crossing counts. Before/after
source checks each covered17,670 files and matched; controller inventory checks
also passed. Lease files were absent without running a reaper, and all eight
GPUs were idle with no compute processes. Cleanup is retained in `cleanup.log`.

- Source inventory: `34c3e3ca9e12005d20366b56e245f51bb6748872d0ee1ef887eee46cf0ed58b9`.
- Measurements: `eb08b3b445cabebf328270a6d08689770abebed80f7e4d9cf57431d3fc03864b`.
- Assay summary: `9292862d00afa004db1871777267c2b44907c746f16e706bd06a5a857231eba4`.
- Diagnostic audit: `558594fd66fe52ec783e1cd505eef0d557933786c1606da1d35f427c535c28a9`.
- Independent evidence: `result-audit.json`; per-token diagnostics remain private
  in `details-private.json` and the assay measurements.

### Next target and qualification boundary

A head-only FP32 change would not close the measured gate. The next causal target
is **task008, turn1, generated position23**, tracing its upstream discrepancy with
the current pinned recurrence, weights and chunk sizes held fixed. Any captured
operator replay must first bind to the current reference; do not assume that
previous legacy-layout localization automatically applies.

No additional GPU run or model repair was launched after this diagnostic. Actual
historical/current-policy failures remain unchanged, no behavior probabilities
were replaced, and optimizer updates remain zero. No new rollout, admission,
learning/capability claim, full4B backward, 64K, multirank or restart qualification
is included.
