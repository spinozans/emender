# Frontier ROCm Readiness Inventory for E97 and GDN2

Date: 2026-06-21

Scope: source inspection for fused Triton recurrence paths and training entrypoints that matter before running Emender E97/E97-MLP, E97-linear-MLP controls, and GDN2/GDN2-MLP on Oak Ridge Frontier ROCm/HIP debug allocations.

Non-negotiable validation rule: every smoke below must run recurrence/state dynamics through fused Triton kernels and must emit the per-rank `[fused-guard] ... NO eager fallback` line. Eager or pure-Python recurrence validation is rejected for prototypes, sanity checks, parity probes, fallback paths, and "quick signal" runs. If a new probe needs extra state summaries, readouts, or metrics, implement those summaries in the fused forward kernel and the matching backward/VJP first.

## Inspected Source Paths

- `train.py`
  - CLI surface for `--level`, `--use_triton`, `--linear_state`, `--e88_raw_write`, `--bf16`, `--gdn2_mlp_ratio`, `--projection_chunk_size`, `--gradient_checkpointing`, DDP/DiLoCo knobs, checkpointing, dataset paths, and fused guards.
  - Auto-enables `--use_triton 1` for `level in {E97, 97, E97-M2}` or `--e88_raw_write 1` under `--bf16`.
  - Distributed setup currently calls `dist.init_process_group(backend='nccl')` and binds `torch.cuda.set_device(local_rank)`.
- `ndm/models/ladder_lm.py`
  - Maps `E97`/`97` to `E88FLAHybrid(use_split_edit=True)`.
  - Maps `E97-M2` to `E88FLAHybrid(use_split_edit=True, use_triton=True, use_chunked_e97=True, linear_state=True)`.
  - Maps `gdn2` and `gdn2-mlp` to external checkout wrappers.
  - Adds `_LadderProtocolAdapter` for some full-sequence mixers, but plain E97 and external GDN2 use the normal `LadderLM` layer protocol.
- `ndm/models/e88_fla_hybrid.py`
  - Implements E88/E97 projections, split-edit gates, fused-path dispatch, fallback recurrence, output gate/projection, and linear-state/chunked E97 dispatch.
  - `use_split_edit=True` adds `erase_gate_proj` and `value_write_gate_proj`; E97 recurrence uses `read_key = erase * k_norm` and `write_value = write * v`.
  - Sequential fused E97 uses `ndm.triton.e88_triton_optimized.e88_triton_optimized_apply`.
  - Chunked E97 linear-state path uses `ndm.triton.e97_chunked_autograd.e97_delta_chunked_triton`.
  - E97-M2 multi-query readout uses `ndm.triton.e97_multiquery_autograd.e97_multiquery_chunked_triton`.
  - The pure PyTorch recurrence remains in the file but is not a valid Frontier smoke path for this project.
- `ndm/triton/e88_triton_forward.py`
  - `@triton.jit` `_e88_forward_kernel`.
  - Supports E88/E97 forward with `RAW_WRITE`, `LINEAR_STATE`, and `SPLIT_EDIT` constexprs.
  - Stores sparse checkpoints every `CKPT_INTERVAL` steps; current wrapper requires `T % ckpt_interval == 0`.
  - Supports `N,V <= 64`.
  - Uses grid `(B, ceil(H / BLOCK_H))`, power-of-two padded `BLOCK_N/BLOCK_V`, and launch heuristics/autotune for `BLOCK_H`/`num_warps`.
- `ndm/triton/e88_triton_backward.py`
  - `@triton.jit` `_e88_backward_kernel`.
  - Reverse-segment replay BPTT: reloads sparse forward checkpoints, replays each segment forward into scratch, then walks the segment backward while threading `dS`.
  - `E88TritonFunction` glues the forward and backward kernels and returns gradients for split-edit gates when present.
  - Requires the same `T % ckpt_interval == 0` and `N,V <= 64` constraints.
- `ndm/triton/e88_triton_optimized.py`
  - Drop-in wrapper that pads sequence length up to the checkpoint interval before calling the E88 Triton autograd function, then crops output back to the caller length.
- `ndm/triton/e97_chunked_autograd.py`
  - `@triton.jit` `_e97_fwd_save_kernel` and `_e97_bwd_kernel`.
  - Linear-state E97 split-edit chunked recurrence. Forward saves per-chunk entry state; backward reverse-replays chunks and recomputes chunk intermediates for the matching VJP.
  - Scope: `N,V <= 64`, `S0=0`, linear state only, default backward chunk `C=32`.
  - Has log-decay mode so gradients are returned w.r.t. log decay directly, avoiding `dg/decay` blow-up as decay approaches zero.
  - Uses `tl.dot(..., allow_tf32=ALLOW_TF32)` for bf16/fp16 inputs.
- `ndm/triton/e97_multiquery_autograd.py`
  - `@triton.jit` `_e97_fwd_mq_kernel` and `_e97_bwd_mq_kernel`.
  - Same chunked linear-state E97 recurrence as `e97_chunked_autograd.py`, but emits `R` readouts per head and accumulates query VJP terms across readouts.
- `ndm/models/external_gdn2.py`
  - Loads external `GatedDeltaNet-2` from `GDN2_PATH` or `/home/erikg/GatedDeltaNet-2`.
  - Installs FLA import shims, wraps `lit_gpt/gdn2.py::GatedDeltaNet2`, forces `mode="chunk"`, `use_short_conv=True` by default, and rejects `head_dim > 256`.
  - `GDN2ExternalMLPLayer` composes the external GDN2 mixer with a second RMSNorm and bias-free SwiGLU MLP.
- `ndm/triton/pin_autotune.py` and `ndm/triton/pinned_autotune_configs.json`
  - Monkeypatch Triton autotuner selection by default to avoid first-call autotune storms. Registry includes measured CUDA/NVIDIA shapes for FLA layernorm, causal conv, and GDN2 chunk kernels.
- `pyproject.toml` / `requirements.txt`
  - `pyproject.toml` pins `torch==2.9.1` and `triton==3.5.1`. `requirements.txt` is looser.
- `scripts/cmaes_search_v2.py`
  - Confirms `e97-linear` exists as a command-generating control: it uses `--level E97 --linear_state 1` and conditionally `--use_triton 1`.
  - Also sets CUDA-specific subprocess environment in its local worker helper, including `CUDA_VISIBLE_DEVICES` and optional `/usr/local/cuda-12.8`.
- OLCF Frontier user guide checked on 2026-06-21
  - Official source: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
  - Slurm is the command surface.
  - Debug QOS is `-q debug`, short non-production work only, one job per user in any state, and maximum walltime is 2 hours.
  - NVMe burst buffer requires `-C nvme` and appears at `/mnt/bb/<userid>`.

## Presence and ROCm Status by Model

| Model surface | Present? | Current fused recurrence | ROCm/HIP readiness |
|---|---:|---|---|
| E97 | Yes | Sequential E97 split-edit through `e88_triton_optimized_apply` -> E88 Triton fwd/bwd with `SPLIT_EDIT=True`; tanh state by default | Needs ROCm compile/run evidence for both forward and reverse-replay backward. Source is Triton, but launch heuristics and pinned/autotune choices are CUDA-shaped. |
| E97-MLP | Yes as `--level E97 --mlp_ratio ...` in `LadderLM` | Same E97 recurrence as above; SwiGLU MLP is dense PyTorch/cuBLAS/hipBLAS outside recurrence | Same recurrence risks as E97. Dense MLP should be ROCm-supported through PyTorch, but memory/optimizer/checkpoint throughput needs Frontier evidence. |
| E97-linear-MLP | Usable control exists | `--level E97 --linear_state 1 --mlp_ratio ... --use_triton 1`; source routes to E88 Triton sequential path unless `use_chunked_e97=True` is also set | Usable as a linear-state control, but not automatically the chunked GDN2-class E97 path. If the desired Frontier control is the chunked E97-linear MLP path, command surface must pass `use_chunked_e97=True` via level `E97-M2` or a new explicit CLI/layer_kwargs route before smokes. Do not treat `e97-linear` CMA naming as proof of chunked throughput. |
| E97-M2 | Yes | Chunked linear-state split-edit E97 with multi-query readout in `e97_multiquery_autograd.py` | Useful as a chunked-kernel ROCm probe, but it changes readout rank and parameters unless `multiquery_r=1`. For a pure E97-linear control, add an explicit chunked single-query surface rather than overloading M2. |
| GDN2 | Yes, external | External NVIDIA GatedDeltaNet-2 FLA chunked kernel via `GDN2ExternalLayer` | Requires the external checkout and an FLA/Triton stack that imports and compiles on ROCm. In-repo source only wraps/shims; the real recurrence kernels live outside this repo. |
| GDN2-MLP | Yes | Same external GDN2 mixer plus in-repo RMSNorm/SwiGLU MLP | Same external-kernel risks as GDN2 plus dense MLP/second RMSNorm memory checks. This is the main GDN2 Frontier smoke target. |

## Confirmed Source Facts

- Fused guard enforcement exists in `train.py`:
  - E97/raw-write under bf16 asserts `args.use_triton == 1` and prints `use_triton=1 -> fused split-edit Triton kernel, NO eager fallback`.
  - GDN2/GDN2-MLP asserts `GDN2_PATH` exists and prints `FLA chunked GDN-2 fused kernel, NO eager fallback`.
- `--use_triton` defaults to `1` only for E97/raw-write/E97-M2 under `--bf16`. Any command that passes `--use_triton 0`, omits `--bf16`, or fails before the fused guard is not a valid smoke.
- E97 sequential fused path is not CUDA extension code when `use_triton=True`; it imports `ndm.triton.e88_triton_optimized`.
- E97 split-edit with `linear_state=True` still routes through the sequential E88 Triton path unless `use_chunked_e97=True` is set.
- Chunked E97 fwd/bwd is fused Triton and reverse-replay based, but only for linear state and `S0=0`.
- Non-saturating state activations `relu`/`softplus` deliberately raise if `use_triton`/bf16 kernels would be used; they are not valid Frontier smoke paths.
- The `e88_recurrence_mode='scan'` associative path is a PyTorch scan/eval path for linear-state analysis, not the required fused Triton recurrence path for Frontier validation.
- GDN2 import success proves only that the external checkout was found and the wrapper loaded. The actual fused GDN2 recurrence kernels must be verified by a live ROCm run and by inspecting external GDN2/FLA logs/imports on Frontier.
- Local GPU work must use `scripts/gpu_lease.sh`; Frontier work must use Slurm allocation instead of local leasing.

## CUDA/NVIDIA-Specific Assumptions

- Package pins are CUDA-oriented unless a ROCm wheel/container is selected explicitly:
  - `pyproject.toml` pins `torch==2.9.1` and `triton==3.5.1`.
  - `requirements.txt` leaves `torch` and `triton` unconstrained.
- `train.py` uses `backend='nccl'`. On ROCm PyTorch this must map to RCCL correctly; the code does not expose a `--dist_backend` or environment override today.
- Code uses `torch.cuda.*`, `cuda` device strings, `x.is_cuda`, and autocast `device_type='cuda'`. In PyTorch ROCm these APIs usually remain named `cuda`, but this must be verified inside the Frontier environment.
- `scripts/cmaes_search_v2.py` hard-codes local CUDA process assumptions (`CUDA_VISIBLE_DEVICES`, optional `CUDA_HOME=/usr/local/cuda-12.8`). It is not a Frontier launcher as-is.
- Triton kernels rely on NVIDIA-measured launch choices:
  - `num_warps` decisions in E88 forward/backward are based on Ada/Ampere observations.
  - E97 chunked default `C=32` cites Ada/Ampere shared-memory limits and Hopper capacity.
  - `allow_tf32` is enabled for bf16/fp16 in E97 chunked kernels. TF32 semantics are NVIDIA-specific; on AMD this flag may be ignored, lowered differently, or affect codegen assumptions.
- `pin_autotune.py` registry was measured on CUDA/NVIDIA shapes. Reusing these configs on MI250X may be correct but throughput-suboptimal, and strict pinning can hide the need for ROCm-specific configs.
- Optional/native CUDA extension paths through `hasty_pytorch_lib` and CUDA register-owned kernels are irrelevant for Frontier ROCm smokes and must not be relied on.
- FLA fused norm and external GDN2 kernels are third-party Triton surfaces; their ROCm compatibility is not established by in-repo tests.

## AMD/Triton Compatibility Risks

- Wavefront/register pressure: E88 and E97 kernels keep `[BLOCK_N, BLOCK_V]` or several `[C,C]` fp32 tiles live. MI250X wavefront size, VGPR allocation, LDS pressure, and occupancy may make CUDA-optimal `BLOCK_H`, `C`, `num_warps`, and `num_stages` choices poor or uncompilable.
- `tl.dot` lowering: E97 chunked kernels depend on many small `tl.dot` operations and `allow_tf32` behavior. Need ROCm evidence for numerical parity, performance, and no compiler fallback/slowness.
- Scalar exponential/logistic hot paths: E88/E97 kernels use `tl.exp`, sigmoid/silu, tanh/identity branches, and log-decay floors. Confirm ROCm codegen does not introduce NaNs or excessive latency.
- Sparse checkpoint reverse replay: E88 backward requires exact agreement between sparse checkpoints and replay. ROCm drift should be checked against short fused-kernel correctness criteria, not eager recurrence acceptance.
- Sequence length padding/constraints: E88 forward/backward requires `T % ckpt_interval == 0`, though `e88_triton_optimized.py` pads around this. Frontier smoke commands should choose `--chunk_size` multiples of 16 and check logs/artifacts for completed backward steps.
- External GDN2/FLA import shims may fail against a Frontier-installed FLA version. GDN2 wrapper expects `lit_gpt/gdn2.py` and specific chunk helper signatures.
- Autotune pinning:
  - Default pinning prevents massive first-call storms, which is desirable for Frontier.
  - First ROCm debug jobs should set `NDM_PIN_TRITON_VERBOSE=1` to record pin/fallback decisions.
  - A later ROCm tuning pass should record a Frontier-specific registry, but only after correctness smokes pass.
- Container/environment: ROCm PyTorch plus Triton 3.5.1 must be installed in a Frontier-compatible container or module environment. The repo currently has no Frontier-specific environment file.

## Frontier Unknowns Requiring Debug-Queue Evidence

- Whether in-repo E88/E97 Triton kernels compile successfully on MI250X for production-like E97 shapes (`N=V=32`, high `H`, bf16).
- Whether E97 sequential split-edit backward returns finite gradients and completes optimizer steps under ROCm.
- Whether chunked E97 linear-state kernels compile and run on MI250X at `C=32` without register/LDS spills that destroy throughput.
- Whether `allow_tf32=True` in Triton is harmless on AMD and whether bf16 dot precision differs enough to change the smoke tolerance.
- Whether the external GDN2 checkout and installed FLA version compile all required chunked GDN2, short-conv, and RMSNorm kernels on Frontier.
- Whether `backend='nccl'` in `train.py` works through RCCL on Frontier without code changes, including `torchrun` rank/device binding.
- Whether pinned CUDA autotune configs are acceptable on MI250X or need ROCm-specific registry capture.
- Whether checkpoint/write paths should use Orion directly or stage hot writes through `/mnt/bb/$USER` for longer runs; debug smokes should at least confirm output directory creation and checkpoint roundtrip on the intended filesystem.
- Whether Frontier login/batch environment exposes `GDN2_PATH`, data paths, and tokenizer cache locations consistently across ranks.

## Prioritized First-Smoke Checklist

All commands below are skeletons. Replace `<ACCOUNT>`, `<REPO>`, `<DATA>`, `<OUT>`, and `<GDN2_PATH>` with Frontier paths. Use Slurm `#SBATCH -p batch` plus `#SBATCH -q debug`, maximum `#SBATCH -t 02:00:00`, and one debug job at a time per user. Use `-C nvme` only when staging data/checkpoints to `/mnt/bb/$USER`; otherwise write to Orion/member/project work paths.

### 1. Environment/import compile smoke, no training claim

Purpose: verify ROCm Python environment imports local kernels and external GDN2 without running eager recurrence.

Command skeleton:

```bash
#SBATCH -A <ACCOUNT>
#SBATCH -J emender-rocm-import
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:20:00

cd <REPO>
export GDN2_PATH=<GDN2_PATH>
export NDM_PIN_TRITON_AUTOTUNE=1
export NDM_PIN_TRITON_VERBOSE=1
srun -N1 -n1 python - <<'PY'
import torch, triton
from ndm.triton.e88_triton_optimized import e88_triton_optimized_apply
from ndm.triton.e97_chunked_autograd import e97_delta_chunked_triton
from ndm.triton.e97_multiquery_autograd import e97_multiquery_chunked_triton
from ndm.models.external_gdn2 import GDN2ExternalLayer, GDN2ExternalMLPLayer
_ = GDN2ExternalLayer(dim=256, num_heads=2)
_ = GDN2ExternalMLPLayer(dim=256, num_heads=2, gdn2_mlp_ratio=2.0)
print("torch", torch.__version__, "triton", triton.__version__, "cuda_api", torch.cuda.is_available())
print("imports_ok")
PY
```

Expected artifacts: Slurm stdout/stderr with package versions and `imports_ok`.

Pass/fail:

- Pass if imports succeed in the Frontier job environment.
- Fail if `GDN2_PATH` or FLA/GDN2 imports are missing, or if PyTorch/Triton ROCm packages are incompatible.
- This is not correctness validation and must not be reported as a model smoke.

### 2. E97-MLP one-rank fused train/backward smoke

Purpose: prove sequential E97 split-edit Triton forward and matching reverse-replay backward run through `train.py` on one MI250X GCD, with the guard visible.

Command skeleton:

```bash
#SBATCH -A <ACCOUNT>
#SBATCH -J e97-mlp-rocm-smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:40:00

cd <REPO>
export NDM_PIN_TRITON_AUTOTUNE=1
export NDM_PIN_TRITON_VERBOSE=1
srun -N1 -n1 python train.py \
  --data <DATA> --tokenizer p50k_base \
  --level E97 --dim 256 --depth 2 --n_heads 8 --n_state 32 \
  --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.0 --mlp_multiple 64 \
  --use_triton 1 --bf16 --batch_size 1 --chunk_size 256 \
  --optimizer adamw --lr 1e-3 --steps 3 --log_every 1 \
  --save_every 3 --output <OUT>/e97_mlp_rocm_smoke
```

Expected artifacts: run directory, `run.log` or Slurm output, `config.json`/args, at least one checkpoint at step 3 if checkpointing is enabled by the current train script.

Pass/fail:

- Pass if output contains `[fused-guard] rank 0/1: level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback`, completes at least one optimizer step and one backward pass, reports finite loss, and writes expected artifacts.
- Fail if any eager/PyTorch recurrence path is used or suggested, if `--use_triton` is not `1`, if the guard is absent, if gradients/loss are non-finite, or if Triton compilation fails.

### 3. E97-linear-MLP control smoke, plus chunked-control decision

Purpose: establish whether the existing `--linear_state 1` control is sufficient, and separately decide whether a chunked single-query control must be implemented before Frontier comparisons.

Current usable control skeleton:

```bash
srun -N1 -n1 python train.py \
  --data <DATA> --tokenizer p50k_base \
  --level E97 --linear_state 1 \
  --dim 256 --depth 2 --n_heads 8 --n_state 32 --expansion 1.0 \
  --use_gate 1 --gate_activation silu --mlp_ratio 2.0 --mlp_multiple 64 \
  --use_triton 1 --bf16 --batch_size 1 --chunk_size 256 \
  --optimizer adamw --lr 1e-3 --steps 3 --log_every 1 \
  --save_every 3 --output <OUT>/e97_linear_mlp_rocm_smoke
```

Expected artifacts: same as E97-MLP smoke.

Pass/fail:

- Pass as an existing linear-state control only if the fused guard is present, loss is finite, and at least one backward step completes.
- Do not call this a chunked E97 throughput control unless logs/source confirm `e97_delta_chunked_triton` was called. Source inspection says plain `--linear_state 1` does not set `use_chunked_e97`.
- If downstream wants a GDN2-class chunked single-query E97-linear-MLP smoke, first implement a tiny command-surface route to `E88FLAHybrid(use_split_edit=True, use_triton=True, use_chunked_e97=True, linear_state=True, multiquery_r=1)` and its fused guard/logging. That implementation is out of scope for this inventory.

### 4. GDN2-MLP one-rank fused external-kernel smoke

Purpose: prove external GDN2/FLA chunked kernels compile and run on ROCm through the current wrapper.

Command skeleton:

```bash
#SBATCH -A <ACCOUNT>
#SBATCH -J gdn2-mlp-rocm-smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:50:00

cd <REPO>
export GDN2_PATH=<GDN2_PATH>
export NDM_PIN_TRITON_AUTOTUNE=1
export NDM_PIN_TRITON_VERBOSE=1
srun -N1 -n1 python train.py \
  --data <DATA> --tokenizer p50k_base \
  --level gdn2-mlp --dim 256 --depth 2 --n_heads 2 \
  --gdn2_mlp_ratio 2.0 --use_conv 1 --d_conv 4 \
  --bf16 --batch_size 1 --chunk_size 256 \
  --optimizer adamw --lr 1e-3 --steps 3 --log_every 1 \
  --save_every 3 --output <OUT>/gdn2_mlp_rocm_smoke
```

Expected artifacts: run directory, Slurm/stdout log, config/args, checkpoint.

Pass/fail:

- Pass if output contains `[fused-guard] rank 0/1: level=gdn2-mlp ... FLA chunked GDN-2 fused kernel, NO eager fallback`, completes optimizer steps with finite loss, and writes artifacts.
- Fail if external checkout is missing, FLA/Triton kernels fail on ROCm, the guard is absent, or any recurrence fallback is observed.

### 5. Minimal two-rank distributed smoke

Purpose: verify `train.py` distributed binding, `backend='nccl'`/RCCL behavior, per-rank fused guards, and non-main output handling.

Command skeleton:

```bash
#SBATCH -A <ACCOUNT>
#SBATCH -J e97-ddp-rocm-smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:40:00

cd <REPO>
export NDM_PIN_TRITON_AUTOTUNE=1
export NDM_PIN_TRITON_VERBOSE=1
srun -N1 -n1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --data <DATA> --tokenizer p50k_base \
  --level E97 --dim 256 --depth 2 --n_heads 8 --n_state 32 \
  --expansion 1.0 --use_gate 1 --gate_activation silu \
  --mlp_ratio 2.0 --mlp_multiple 64 \
  --use_triton 1 --bf16 --batch_size 1 --chunk_size 256 \
  --optimizer adamw --lr 1e-3 --steps 2 --log_every 1 \
  --save_every 2 --output <OUT>/e97_ddp_rocm_smoke
```

Expected artifacts: one rank-0 run directory/checkpoint, Slurm logs from both ranks.

Pass/fail:

- Pass if every rank prints the E97 fused guard with `NO eager fallback`, DDP initializes through ROCm/RCCL, loss is finite, and only rank 0 writes checkpoint artifacts.
- Fail if only rank 0 prints the guard, if DDP backend setup fails, or if there is any fallback recurrence.

### 6. Throughput sanity, not a benchmark

Purpose: catch pathological ROCm codegen or autotune choices after correctness passes.

Command skeleton: rerun the E97-MLP and GDN2-MLP one-rank smokes for 20 to 50 steps with `--log_every 5`, same fused requirements, and no eager fallback.

Expected artifacts: logs with tokens/sec or step timing, plus pin-autotune verbose decisions.

Pass/fail:

- Pass if steps proceed steadily after first compile, no long autotune hang occurs, and throughput is stable enough to justify longer debug/batch jobs.
- Fail if the job spends most of the debug allocation compiling/autotuning, if steps wedge after the fused guard, or if throughput is orders of magnitude below local fused expectations. Do not rescue by switching recurrence to eager.

## Recommended First Actions

1. Run the one-node import/compile smoke on Frontier with the intended ROCm PyTorch/Triton/FLA/GDN2 environment and `NDM_PIN_TRITON_VERBOSE=1`.
2. Run the E97-MLP one-rank train/backward smoke and require the fused guard plus finite loss/checkpoint artifacts before any GDN2 comparison.
3. Decide whether the existing `--level E97 --linear_state 1 --mlp_ratio ...` sequential fused control is enough, or add a tiny explicit chunked single-query E97-linear-MLP command surface before using it as the GDN2-class throughput control.

## Out of Scope for This Inventory

- Editing `docs/FRONTIER_KICKOFF_SYNTHESIS.md` or `docs/FRONTIER_EXECUTION_GRAPH_DRAFT.md`.
- Implementing ROCm environment files, Slurm launch wrappers, distributed backend flags, or a new E97 chunked command surface.
- Running local or Frontier GPU jobs.
- Eager/Python recurrence parity as a validation step.
