# ROCm GDN2-MLP Port Handoff - 2026-06-21

Task: `rocm-gdn2-mlp-port`

Commit: `3c856b1` (`feat: add gdn2 ROCm preflight path (rocm-gdn2-mlp-port)`)

## Code Changes

- `ndm/models/external_gdn2.py` now exposes
  `probe_gdn2_external_dependencies()`, which uses the production external
  GDN-2 loader and FLA compatibility shims to verify:
  - resolved `GDN2_PATH` and `lit_gpt/gdn2.py`;
  - `torch.version.hip`, CUDA/HIP availability as reported by PyTorch, and
    device count;
  - importability of `fla`;
  - loadability of external `GatedDeltaNet2`;
  - importability of `_external_gdn2_lit_gpt.gdn2_ops.chunk_gdn2`;
  - presence of the required `chunk_gla_fwd_o_gk` chunk-op symbol used by the
    existing compatibility shim.
- `train.py` uses the probe for `--level gdn2` and `--level gdn2-mlp` before
  training starts. The fused guard now fails with the exact missing module or
  symbol and logs backend, `torch.version.hip`, `GDN2_PATH`, and chunk-op module
  path when the import path is valid.
- `scripts/frontier/gdn2_rocm_preflight.py` provides a Frontier-friendly smoke
  command. Default mode is import-only JSON preflight. `--run-fwdbwd` launches a
  small gdn2-MLP forward/backward step and checks finite output, loss, input
  gradients, and parameter gradients.
- `scripts/frontier/debug_smoke_one_node.slurm` now runs the one-rank
  `gdn2_rocm_preflight.py --run-fwdbwd --bf16` command before the eight-rank
  `gdn2-MLP` training smoke. If the preflight fails, training is skipped and the
  manifest records `gdn2_rocm_preflight_status` and
  `gdn2_rocm_preflight_log`.
- `docs/FRONTIER_DEBUG_RECIPE_20260621.md` documents the import-only and
  forward/backward preflight commands.

## Validation Commands Run Locally

```bash
python3.11 -m py_compile \
  ndm/models/external_gdn2.py \
  train.py \
  scripts/frontier/gdn2_rocm_preflight.py
```

Result: PASS.

```bash
python3.11 -m pytest tests/test_cmaes_accounting.py -q
```

Result: NOT RUN. `/usr/bin/python3.11` in this worktree environment does not
have `pytest` installed.

```bash
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  git rev-parse HEAD
  python3.11 --version
  python3.11 scripts/frontier/gdn2_rocm_preflight.py
} > logs/rocm-gdn2-mlp-port/gdn2_rocm_preflight_local.log 2>&1
```

Result: FAIL as expected in the local non-ML Python environment. The structured
JSON failure is:

```text
could not import ndm.models.external_gdn2: ModuleNotFoundError("No module named 'torch'")
```

This confirms the script reports an actionable environment blocker instead of a
raw traceback when the PyTorch stack is absent.

## Frontier Commands For Downstream Smoke

Import-only dependency check:

```bash
GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2" \
python -u scripts/frontier/gdn2_rocm_preflight.py
```

One-GCD compile/runtime smoke:

```bash
GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2" \
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
  python -u scripts/frontier/gdn2_rocm_preflight.py \
    --run-fwdbwd --bf16 --dim 128 --n-heads 4 --head-dim 16 \
    --seq-len 16 --batch-size 1 --expansion 1.0 --use-conv 1 \
    --d-conv 4 --mlp-ratio 2.0
```

Full debug template:

```bash
mkdir -p logs/frontier/debug
SMOKE_VARIANT=gdn2-MLP \
DATA=/path/to/smoke_train.txt \
VAL_DATA=/path/to/smoke_val.txt \
EMENDER_CONDA_ENV="$MEMBERWORK/emender/conda/emender-rocm711" \
GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2" \
sbatch scripts/frontier/debug_smoke_one_node.slurm
```

## Observation Boundary

No Frontier allocation was used in this task, so this handoff does not claim
that the external GDN2 Triton kernels compile on HIP. The local evidence only
supports syntax validity and structured failure reporting in an environment
without `torch`. The Frontier preflight and SLURM template are the intended
commands for confirming or narrowing HIP compile/runtime behavior.
