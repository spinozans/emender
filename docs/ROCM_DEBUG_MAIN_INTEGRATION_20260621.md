# ROCm Debug Main Integration - 2026-06-21

Task: `frontier-merge-rocm-debug-main`

This note records the controlled integration state consumed by Frontier debug,
benchmark, extended, and DiLoCo planning tasks. No expensive Frontier training
jobs were launched for this integration task.

## Integrated Main

Integrated tree validated from:

- branch: `wg/agent-34/frontier-merge-rocm-debug-main`
- base/current `origin/main`: `190404e` (`feat: rocm-kernel-integration-smoke (agent-33)`)
- integration-report commit: filled by `frontier-merge-rocm-debug-main` logs

The predecessor WG branch tips were already represented on `main` as mainline
integration commits when this task started. For the first four tasks, `git
cherry -v 2dee6ce <branch>` reported patch-equivalent commits with `-`, meaning
their exact branch tips had no remaining patch delta against the integrated
ROCm tree. The gdn2 branch contains original/split commits plus a merge commit,
while `main` contains the same file content as squashed commit `2dee6ce`; `git
diff 2dee6ce wg/agent-28/rocm-gdn2-mlp-port` was empty.

## Predecessor Accounting

| Task | WG branch tip | Main integration commit | Status |
| --- | --- | --- | --- |
| `frontier-allocation-ledger` | `795ba3f` | `09a5c2c` | Included. Adds `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md`. |
| `frontier-env-debug-recipe` | `050fd95` | `52ef628` | Included. Adds `.gitignore`, `docs/FRONTIER_DEBUG_RECIPE_20260621.md`, and `scripts/frontier/debug_smoke_one_node.slurm`. |
| `rocm-kernel-port-audit` | `bbaeb5a` | `d455115` | Included. Adds `docs/ROCM_TRITON_KERNEL_PORT_AUDIT_20260621.md`. |
| `rocm-e97-mlp-port` | `36531c6` | `4d87cdd` | Included. Adds e97/e97-linear ROCm runtime controls, smoke hook updates, and `tests/test_rocm_e97_runtime_config.py`. |
| `rocm-gdn2-mlp-port` | `6bdc650` merge of `3c856b1` and `25939a4` | `2dee6ce` | Included as a squashed mainline commit. `git diff HEAD wg/agent-28/rocm-gdn2-mlp-port` was empty before this integration-report patch. |

No predecessor work was intentionally excluded.

## Smoke Coordination

The `rocm-kernel-integration-smoke` task prepared useful smoke runner/report
material while this task was validating main integration. That work landed on
`main` as `190404e` before this report commit was rebased. Its report explicitly
states that exact predecessor branch tips were not ancestors of its
then-current `HEAD` at smoke-preparation time and that its generated blocker
artifacts are pre-merge evidence only.

Useful generic pieces from that work are preserved on integrated `main`:

- `scripts/frontier/submit_kernel_integration_smokes.sh` submits `e97-MLP`,
  `e97-linear-MLP`, and `gdn2-MLP` one at a time with the shared debug smoke
  template.
- `scripts/frontier/debug_smoke_one_node.slurm` now records a one-rank GPU
  visibility probe and automatically extracts final metric, loss, throughput,
  runtime-path, memory, and first actionable error lines into `summary.md`.

The smoke task or its downstream consumer should rerun or update evidence from
integrated main after this task completes. Its pre-merge logs should not admit
any variant into the benchmark matrix.

## Validation Commands

Run from `wg/agent-34/frontier-merge-rocm-debug-main`:

```bash
git cherry -v 2dee6ce wg/agent-14/frontier-allocation-ledger
git cherry -v 2dee6ce wg/agent-16/frontier-env-debug-recipe
git cherry -v 2dee6ce wg/agent-23/rocm-kernel-port-audit
git cherry -v 2dee6ce wg/agent-27/rocm-e97-mlp-port
git diff 2dee6ce wg/agent-28/rocm-gdn2-mlp-port
bash -n scripts/frontier/debug_smoke_one_node.slurm
bash -n scripts/frontier/submit_kernel_integration_smokes.sh
python3.11 -m py_compile train.py ndm/models/e88_fla_hybrid.py ndm/triton/e97_chunked_autograd.py ndm/models/external_gdn2.py scripts/frontier/gdn2_rocm_preflight.py tests/test_rocm_e97_runtime_config.py
```

No `sbatch`, training, benchmark, or extended allocation job was launched by
this task.
