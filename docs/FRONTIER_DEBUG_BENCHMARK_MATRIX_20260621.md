# Frontier Debug Benchmark Matrix - 2026-06-21

Task: `frontier-debug-benchmark-matrix`

## Decision

Next gate: **retry debug**.

No benchmark job was submitted in this pass. The integrated code tree is present
and Frontier SLURM commands are available, but the runtime prerequisites needed
for a meaningful debug benchmark are not observable from `login04`:

- Expected ROCm Python environment is absent:
  `/lustre/orion/scratch/erikgarrison/emender/conda/emender-rocm711`.
- System `python3.11` lacks `torch`, `triton`, `schedulefree`, and `tiktoken`.
- The only visible micromamba base does not expose a runnable `python`.
- GDN2 external checkout is absent at the expected path:
  `/lustre/orion/scratch/erikgarrison/emender/src/GatedDeltaNet-2`.
- The canonical commapile mainmix source is present, but the smoke-sized
  `DATA` and `VAL_DATA` paths referenced by the predecessor smoke recipe are
  absent.

Submitting the three debug jobs in this state would mainly measure missing
runtime dependencies, not e97/gdn2 correctness, throughput, memory behavior, or
GPU-island viability. The observed result is therefore a launch-environment
blocker, not negative kernel or model evidence.

## Preflight Evidence

Observed on 2026-06-21 from `login04` in worktree
`/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-41`.

```text
hostname: login04
git commit: d39bac8a061b8016aaa16d6eee6e5403acade9c6
sbatch: /usr/bin/sbatch
squeue: /usr/bin/squeue
sacct: /usr/bin/sacct
current user debug jobs: none shown by squeue -u "$USER"
canonical data: /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt, 932G
expected smoke train data: missing /lustre/orion/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt
expected smoke val data: missing /lustre/orion/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
expected ROCm env: missing /lustre/orion/scratch/erikgarrison/emender/conda/emender-rocm711
expected GDN2 checkout: missing /lustre/orion/scratch/erikgarrison/emender/src/GatedDeltaNet-2
```

System Python module check:

```text
torch: missing
triton: missing
schedulefree: missing
numpy: present, 1.24.2
tiktoken: missing
```

The predecessor integration report remains relevant only as static/code
integration evidence. Its own Frontier smoke artifacts explicitly reported that
no variant job was submitted before the integrated main tree was available, and
those artifacts must not be reinterpreted as benchmark results.

## Benchmark Matrix

Requested node-hours are shown for the intended one-node, 30-minute debug
submissions from `scripts/frontier/debug_smoke_one_node.slurm`. Actual
node-hours are zero because no `sbatch` submission was made.

| Variant | Command | Queue | Nodes | Walltime | Job ID | Requested node-hours | Actual node-hours | Artifacts | Outcome |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| `e97-MLP` | `DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt TRAIN_MINUTES=20 SMOKE_VARIANT=e97-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm` | debug | 1 | 00:30:00 intended | not submitted | 0.500000 planned | 0.000000 | `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/env.txt`, `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/manifest.json` | Blocked before launch by missing ROCm Python runtime; no throughput, memory, or loss data. |
| `e97-linear-MLP` | `DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt TRAIN_MINUTES=20 SMOKE_VARIANT=e97-linear-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm` | debug | 1 | 00:30:00 intended | not submitted | 0.500000 planned | 0.000000 | `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/env.txt`, `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/manifest.json` | Blocked before launch by missing ROCm Python runtime; no chunked-E97 runtime, throughput, memory, or loss data. |
| `gdn2-MLP` | `DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt TRAIN_MINUTES=20 SMOKE_VARIANT=gdn2-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm` | debug | 1 | 00:30:00 intended | not submitted | 0.500000 planned | 0.000000 | `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/env.txt`, `frontier_runs/debug/20260621/frontier-debug-benchmark-matrix/preflight/manifest.json` | Blocked before launch by missing ROCm Python runtime and missing `GDN2_PATH`; no GDN2 import/preflight, throughput, memory, or loss data. |

## Evidence Summary

### e97-MLP

Observed: the integrated tree contains the Frontier debug smoke script and the
variant command path. The environment needed to import PyTorch/Triton and run
training is not present. No scheduler job was submitted for this variant.

Throughput: not measured.

Memory: not measured.

Loss sanity: not measured.

Neutral hypothesis for retry: e97-MLP may still be viable after the ROCm Python
environment is staged; this pass does not test the kernel or training path.

### e97-linear-MLP

Observed: the integrated tree contains the `e97-linear-MLP` smoke command path,
including the chunked-E97 settings in the smoke script. The runtime prerequisite
is missing before launch, so no job was submitted.

Throughput: not measured.

Memory: not measured.

Loss sanity: not measured.

Neutral hypothesis for retry: the chunked E97 path still needs a real one-node
debug run to establish loss, throughput, memory behavior, and whether the
runtime selects the intended fused path.

### gdn2-MLP

Observed: the integrated tree contains the `gdn2-MLP` smoke command path, but
the external GatedDeltaNet-2 checkout is absent and the ROCm Python runtime is
missing. No job was submitted.

Throughput: not measured.

Memory: not measured.

Loss sanity: not measured.

Neutral hypothesis for retry: gdn2-MLP remains untested on Frontier in this
matrix. A retry must first stage `GDN2_PATH` and confirm the one-rank GDN2
preflight before interpreting any training result.

## GPU-Island Pattern

The GPU-island pattern was **deferred**. It was feasible from a scheduler
capacity perspective because no current user debug job was shown, but it was
not feasible as a correctness test because the training runtime prerequisites
were missing. A launch that fails before importing PyTorch/Triton would not
distinguish isolated single-GPU processes from DDP-like modes.

Retry-debug should first run the existing one-node, eight-rank template once a
valid ROCm environment and data paths exist. If that run reaches model training,
the next debug comparison should explicitly launch:

1. eight isolated one-GPU processes on one node with rank-strided data and no
   gradient synchronization, and
2. the current DDP-like eight-rank mode,

using the same variant, data slice, walltime, and logging fields.

## Ledger Accounting

This pass submits no jobs and consumes no Frontier node-hours.

```text
debug planned for matrix retry: 3 jobs * 1 node * 0.5h = 1.500000 requested node-hours
debug submitted in this pass: 0 jobs
actual node-hours consumed in this pass: 0.000000
allocation remaining before: 20,000
allocation remaining after: 20,000
reserve held: 4,928
```

The allocation ledger was updated with a `DEBUG-MATRIX-20260621-PREFLIGHT` row
that records the zero-spend blocker and points at this report plus the preflight
artifacts.

## Recommendation

Retry debug after staging the runtime prerequisites:

- create or point `EMENDER_CONDA_ENV` to a Frontier-accessible ROCm/PyTorch
  environment with `torch`, `triton`, `schedulefree`, and `tiktoken`;
- stage `GDN2_PATH` for the external GatedDeltaNet-2 checkout;
- create smoke-sized `DATA` and `VAL_DATA` files or intentionally use the
  canonical commapile path with an I/O plan appropriate for short debug runs;
- rerun the one-node matrix for `e97-MLP`, `e97-linear-MLP`, and `gdn2-MLP`;
- only after a variant records finite loss, throughput, memory behavior, and
  launch stability should downstream canary or extended-readiness tasks prepare
  larger submissions.

Do not prepare an extended launch from this evidence alone.
