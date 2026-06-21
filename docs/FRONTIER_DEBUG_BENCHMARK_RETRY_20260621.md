# Frontier Debug Benchmark Retry - 2026-06-21

Task: `retry-frontier-debug`

## Decision

Next gate: **fix tokenizer cache and chunked-E97 ROCm parity before extended
readiness**.

This retry staged the previously missing runtime prerequisites far enough to
submit the one-node debug matrix on Frontier. The matrix did not produce
training throughput or loss curves because the runs stopped before a training
step completed:

- `e97-MLP` passed the one-rank fused split-edit Triton kernel smoke, then all
  eight training ranks failed while `tiktoken` tried to download
  `p50k_base.tiktoken` from `openaipublic.blob.core.windows.net` on compute
  nodes.
- `e97-linear-MLP` failed the one-rank chunked-E97 Triton parity/finiteness
  smoke on MI250X, before training was launched.
- `gdn2-MLP` passed the GDN2 external checkout import and one-rank bf16
  forward/backward preflight with finite loss, then all eight training ranks
  hit the same compute-node `tiktoken` download timeout as `e97-MLP`.

The debug matrix therefore establishes scheduler/runtime reachability, GPU
visibility, and several kernel-specific outcomes, but it is not sufficient for
extended-queue launch readiness.

## Runtime Staging

Observed from the Frontier module-loaded runtime used by the jobs:

```text
module load miniforge3/23.11.0-0 rocm/7.1.1
EMENDER_CONDA_ENV=base
python=/autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/python
torch=2.8.0.dev20250422+rocm6.4
torch.version.hip=6.4.43482-0f2d60242
triton=3.2.0
schedulefree=1.4.1
tiktoken=0.13.0
numpy=2.0.0
pytest=9.1.1
flash-linear-attention=0.5.1
fla-core=0.5.1
```

The user-site ROCm torch installation was repaired from pip's temporary
`~orch*` directory names, and missing Python packages were installed into the
Frontier miniforge user site. This makes `EMENDER_CONDA_ENV=base` usable for
debug jobs, but it is still a staged user-site runtime rather than a clean,
named production conda environment.

## Data and GDN2

Smoke-sized data files were staged from the canonical commapile source:

```text
DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
source=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
DATA size=1.0M
VAL_DATA size=256K
```

The external GDN2 checkout was staged at:

```text
GDN2_PATH=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2
GDN2 commit=95709fc
```

The `gdn2-MLP` job recorded a successful one-rank preflight:

```text
device_name=AMD Instinct MI250X
dtype=torch.bfloat16
finite_output=true
finite_loss=true
finite_input_grad=true
finite_param_grads=true
loss=0.009861934930086136
chunk_ops=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py
```

## Scheduler Submissions

Debug QOS accepted one submitted/running job per user. Parallel submissions for
`e97-linear-MLP` and `gdn2-MLP` immediately after the first `e97-MLP` submit
were rejected with `QOSMaxSubmitJobPerUserLimit`; the matrix was then run
sequentially.

| Variant | Job ID | State | Queue | Nodes | Requested walltime | Elapsed | Requested node-hours | Actual node-hours | Kernel/preflight | Training outcome |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| `e97-MLP` | `4880875` | `FAILED` | debug | 1 | 00:30:00 | 00:05:08 | 0.500000 | 0.085556 | Passed `tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference` in 33.07s. | Failed before first training metric: all ranks timed out downloading `p50k_base.tiktoken` from `openaipublic.blob.core.windows.net`. |
| `e97-linear-MLP` | `4880730` | `FAILED` | debug | 1 | 00:30:00 | 00:01:41 | 0.500000 | 0.028056 | Failed chunked-E97 smoke: 7/7 parity/finiteness assertions failed on MI250X. | Not launched because kernel smoke failed. |
| `gdn2-MLP` | `4880747` | `FAILED` | debug | 1 | 00:30:00 | 00:04:14 | 0.500000 | 0.070556 | Passed external GDN2 import plus bf16 fwd/bwd preflight; finite loss 0.009861934930086136. | Failed before first training metric: all ranks timed out downloading `p50k_base.tiktoken` from `openaipublic.blob.core.windows.net`. |

An earlier `e97-MLP` attempt, `4880725`, failed in 00:00:47 before kernel
execution because `pytest` was not yet staged. It consumed 0.013056
node-hours and is treated as setup overhead, not the matrix result for
`e97-MLP`.

Total requested node-hours for accepted submissions:

```text
4 jobs * 1 node * 0.5h = 2.000000 requested node-hours
```

Total actual elapsed node-hours:

```text
4880725 00:00:47 = 0.013056
4880730 00:01:41 = 0.028056
4880747 00:04:14 = 0.070556
4880875 00:05:08 = 0.085556
total actual elapsed = 0.197224 node-hours
```

## Results by Variant

### e97-MLP

Observed:

- Job `4880875` had eight visible GPUs in the Python environment.
- One-rank GPU visibility reported `ROCR_VISIBLE_DEVICES=0`,
  `torch.cuda.device_count=1`, and device name `AMD Instinct MI250X`.
- The fused split-edit Triton kernel smoke passed.
- The eight-rank training launch selected the fused E97 path with no eager
  fallback.
- Training failed before any loss or throughput line because tiktoken attempted
  a network download from compute nodes.

Throughput: not measured.

Memory: `rocm-smi` baseline showed VRAM 0% before training; no peak memory line
was emitted before the tokenizer failure.

Loss sanity: not measured.

### e97-linear-MLP

Observed:

- Job `4880730` had eight visible GPUs in the Python environment.
- One-rank GPU visibility reported `ROCR_VISIBLE_DEVICES=0`,
  `torch.cuda.device_count=1`, and device name `AMD Instinct MI250X`.
- The chunked-E97 kernel smoke failed before training.
- Forward parity max error reached `1.53930640e8`, `5.1271134347264e13`,
  `2.6356457056796805e26`, and `2.18813120512e11` across tested sequence
  lengths.
- The bf16 forward relative error was `2.4178516123642406e24`.
- Strong-decay and g-drift finiteness tests produced non-finite outputs.

Throughput: not measured.

Memory: `rocm-smi` baseline showed VRAM 0% before the kernel smoke; no peak
memory line was emitted.

Loss sanity: not measured.

### gdn2-MLP

Observed:

- Job `4880747` had eight visible GPUs in the Python environment.
- One-rank GPU visibility reported `ROCR_VISIBLE_DEVICES=0`,
  `torch.cuda.device_count=1`, and device name `AMD Instinct MI250X`.
- External GDN2 import and bf16 forward/backward preflight passed.
- The eight-rank training launch selected the FLA chunked GDN-2 fused kernel
  import path with no eager fallback.
- Training failed before any loss or throughput line because tiktoken attempted
  a network download from compute nodes.

Throughput: not measured.

Memory: `rocm-smi` baseline showed VRAM 0% before training; no peak memory line
was emitted before the tokenizer failure.

Loss sanity: one-rank preflight loss was finite (`0.009861934930086136`);
training loss was not measured.

## GPU-Island Observation

The template exercised one-node, eight-rank GPU binding for `e97-MLP` and
`gdn2-MLP`. The environment capture also included a one-rank visibility probe
showing rank-local `ROCR_VISIBLE_DEVICES=0` and one visible MI250X under
`srun -N1 -n1 --gpus-per-task=1 --gpu-bind=closest`, while the full Python
environment sees eight GPUs before `srun`.

This is partial GPU-island evidence for SLURM GPU binding on one node, but it is
not a clean isolated-island training test because the current training command
uses one eight-rank `srun` and all ranks fail before tokenization completes. A
proper island-vs-DDP comparison remains deferred until tokenizer files are
pre-cached in a shared path and at least one variant reaches a training step.

## Artifact Links

- `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z/artifacts/env.txt`
- `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z/artifacts/manifest.json`
- `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z/logs/kernel_smoke.log`
- `frontier_runs/debug/20260621/e97-MLP/4880875-20260621T140101Z/logs/train.log`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/artifacts/env.txt`
- `frontier_runs/debug/20260621/e97-linear-MLP/4880730-20260621T134833Z/logs/kernel_smoke.log`
- `frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z/artifacts/env.txt`
- `frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z/logs/kernel_smoke.log`
- `frontier_runs/debug/20260621/gdn2-MLP/4880747-20260621T135335Z/logs/train.log`
- `logs/frontier/debug/emender-smoke-4880875.out`
- `logs/frontier/debug/emender-smoke-4880730.out`
- `logs/frontier/debug/emender-smoke-4880747.out`

## Recommendation

Do not prepare an extended e97 or DiLoCo launch from this evidence.

Next gate should be a narrow debug fix pass:

1. Add an explicit shared `TIKTOKEN_CACHE_DIR` under Frontier scratch or
   project-shared storage and pre-populate `p50k_base.tiktoken` before every
   compute-node launch.
2. Rerun only `e97-MLP` and `gdn2-MLP` to confirm they reach first loss and
   throughput metrics after tokenizer cache staging.
3. Treat `e97-linear-MLP` as a kernel correctness failure on Frontier ROCm until
   the chunked-E97 parity/finiteness failures are fixed or the variant is
   disabled for extended readiness.
4. After `e97-MLP` or `gdn2-MLP` records finite training loss, add an explicit
   one-node GPU-island comparison against the current eight-rank mode.
