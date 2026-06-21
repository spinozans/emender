# Frontier Debug Smoke Summary

- task: retry-frontier-debug
- variant: gdn2-MLP
- job_id: 4880747
- account: bif148
- partition: batch
- qos: debug
- nodes: 1
- requested_walltime: 00:30:00
- elapsed: fill from sacct after completion
- requested_node_hours: 0.500000
- actual_node_hours: fill from sacct after completion
- git_commit: 17dcc782a997b2c34106d6e59600338210eb54b1
- stdout: logs/frontier/debug/emender-smoke-4880747.out
- stderr: logs/frontier/debug/emender-smoke-4880747.err
- artifacts: /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-44/frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z
- kernel_smoke_status: 0
- train_status: 1
- exit_status: 1

## Observation
final_metric_line: not observed
final_loss_line: not observed
throughput_line: not observed
runtime_path_line: not observed
peak_memory_line: not observed
first_actionable_error: Current Triton version 3.2.0 is below the recommended 3.3.0 version. Errors may occur and these issues will not be fixed. Please consider upgrading Triton.

## Failure Mode
Use first_actionable_error above plus /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-44/frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z/logs/kernel_smoke.log and
/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-44/frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z/logs/train.log. Kernel failure skips training by design so the blocker is
not hidden behind a later DDP launch.

## Interpretation Boundary
This is one short debug-QOS run. Treat it as launch/import/kernel/memory/loss
sanity evidence only, not as extended-allocation readiness.
