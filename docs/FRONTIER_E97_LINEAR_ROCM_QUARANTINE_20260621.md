# Frontier e97-linear ROCm Quarantine - 2026-06-21

Task: `fix-or-quarantine`

## Decision

`e97-linear-MLP` is quarantined from Frontier readiness decisions until the
chunked-E97 ROCm parity/finiteness failure is fixed and a new one-rank Frontier
smoke passes.

This task did not submit any new Frontier jobs. It records the existing debug
failure and makes the readiness boundary explicit. Extended e97 or DiLoCo
scaleout recommendations may consider `e97-MLP` or `gdn2-MLP` only after their
own gates pass, but must not use `e97-linear-MLP` as a launch candidate, fallback
candidate, tie-breaker, or evidence source while this quarantine is active.

## Failure Cited

Observed failure: Frontier debug job `4880730` for `e97-linear-MLP`.

Job/accounting facts copied from the committed retry report and run summary:

| Field | Value |
| --- | --- |
| Job id | `4880730` |
| Queue / QOS | `batch` partition, `debug` QOS |
| Nodes | 1 |
| Requested walltime | `00:30:00` |
| Requested node-hours | `0.500000` |
| Elapsed walltime | `00:01:41` |
| Elapsed node-hours | `0.028056` |
| Variant | `e97-linear-MLP` |
| Kernel smoke status | failed before training |
| Training status | skipped because kernel smoke failed |

Primary artifact links:

- `docs/FRONTIER_DEBUG_BENCHMARK_RETRY_20260621.md`
- `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/summaries/summary.md`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/artifacts/manifest.json`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/artifacts/env.txt`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/logs/kernel_smoke.log`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/logs/train.log`
- `logs/frontier/debug/emender-smoke-4880730.out`
- `logs/frontier/debug/emender-smoke-4880730.err`

The kernel smoke ran seven selected checks from `tests/test_e97_chunked.py` on a
single visible MI250X rank. All seven failed.

Observed failure signatures from the captured `kernel_smoke.log`:

- `test_fused_triton_forward_parity_fp32[64]`: max absolute error
  `1.53930640e8` versus tolerance `2e-4`.
- `test_fused_triton_forward_parity_fp32[128]`: max absolute error
  `5.1271134347264e13` versus tolerance `2e-4`.
- `test_fused_triton_forward_parity_fp32[256]`: max absolute error
  `2.6356457056796805e26` versus tolerance `2e-4`.
- `test_fused_triton_forward_parity_fp32[96]`: max absolute error
  `2.18813120512e11` versus tolerance `2e-4`.
- `test_fused_triton_backward_parity_bf16`: bf16 forward relative error
  `2.4178516123642406e24` versus tolerance `0.05`.
- `test_strong_decay_cluster_backward_finite`: forward output contained
  non-finite values.
- `test_glog_floor_drift_overflow_finite`: forward output contained non-finite
  values.

Because the kernel smoke failed, the smoke script skipped training by design.
There is therefore no `e97-linear-MLP` loss, throughput, memory peak, checkpoint,
or resume evidence from Frontier.

## Quarantine Rule

Until a successor task removes this quarantine, any Frontier launch-readiness
document or launch package must satisfy all of the following:

1. Do not recommend `e97-linear-MLP` for extended, scaleout, canary, or
   production-shaped submissions.
2. Do not count `e97-linear-MLP` as support for e97-family launch readiness.
3. Do not spend Frontier node-hours on `e97-linear-MLP` except for a scoped
   debug correctness rerun whose purpose is to fix or verify the chunked-E97
   ROCm kernel.
4. If a future launch script still contains an `e97-linear-MLP` branch for
   debugging, callers must treat it as a quarantined debug-only path, not a
   readiness candidate.

## Exit Criteria

The quarantine can be removed only after a new committed report records:

1. The code change or configuration change that addresses the ROCm
   parity/finiteness failure.
2. A passing Frontier one-rank smoke for the relevant chunked-E97 tests,
   including job id, queue/QOS, nodes, requested walltime, requested and elapsed
   node-hours, git commit, stdout/stderr paths, and run artifact directory.
3. At least one finite training startup check if the arm is being considered for
   launch readiness rather than kernel correctness only.
4. Updates to extended and scaleout readiness docs that explicitly lift this
   quarantine and cite the passing artifacts.

## Validation Notes

No new Frontier jobs were submitted for this task, so
`docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` does not need a new spend row.
The existing retry row already records job `4880730`, its accounting, and its
artifacts.
