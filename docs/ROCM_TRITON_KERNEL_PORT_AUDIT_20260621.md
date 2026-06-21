# ROCm Triton Kernel Port Audit - 2026-06-21

Task: `rocm-kernel-port-audit`

Scope: `e97-MLP`, `e97-linear-MLP`, and `gdn2-MLP` on Frontier AMD MI250X
GCDs. This is an audit and implementation plan, not a code port. Earlier
portability and DDP/no-go statements are treated as hypotheses unless reproduced
or directly observed in this pass.

## Concrete Implementation Plan

1. Freeze ownership before editing.
   - E97/e97-linear owner: `ndm/triton/e97_chunked_autograd.py`,
     `ndm/triton/e97_chunked.py`, `ndm/triton/e88_triton_forward.py`,
     `ndm/triton/e88_triton_backward.py`,
     `ndm/triton/e88_triton_optimized.py`,
     `ndm/models/e88_fla_hybrid.py`, and E97-specific tests.
   - GDN2 owner: `ndm/models/external_gdn2.py`, `ndm/models/ladder_lm.py`,
     GDN2 dependency/environment checks, and GDN2 smoke tests.
   - Shared runtime files requiring serialization: `train.py`,
     `scripts/frontier/debug_smoke_one_node.slurm`,
     `docs/FRONTIER_DEBUG_RECIPE_20260621.md`, `pyproject.toml`, any smoke
     manifest or broad benchmark script that invokes both arms.

2. First E97/e97-linear change set.
   - Add a backend probe/helper in the E97 runtime, not scattered conditionals:
     identify `torch.version.hip` and Triton HIP execution; log backend and
     selected kernel path once per run.
   - Confirm the intended semantics of the `e97-MLP` Frontier variant before
     changing kernels. The current template labels it `e97-MLP` but passes
     `--e88_raw_write 1`, which is an E97 raw-write ablation rather than the
     delta-correcting split-edit cell. Treat the label as unresolved until the
     downstream port task either renames the variant or changes the flag.
   - In `e97_chunked_autograd.py`, make HIP-sensitive launch constants explicit:
     `num_warps`, `num_stages`, `ALLOW_TF32`, `BC`, and supported `(N,V,C)`
     combinations. Default to a conservative Frontier profile first
     (`chunk_size=32`, `num_warps=4`, `num_stages=1`), then benchmark.
   - Add an AMD skip/xfail boundary only for unsupported combinations. Do not
     add eager fallback for E97 bf16 training; `train.py` currently treats fused
     Triton as mandatory for E97/raw-write bf16.
   - Verify e97-MLP and e97-linear-MLP separately. The nonlinear E97 smoke
     (`--linear_state 0`) exercises the sequential E88 Triton path; the
     linear-state smoke (`--linear_state 1`) is the chunked E97 path candidate.

3. First GDN2 change set.
   - Add a preflight/import diagnostic for `GDN2_PATH`, `fla`, and the external
     `lit_gpt/gdn2.py` module before training starts. Keep it outside the E97
     files.
   - Confirm the external GDN2 checkout and installed FLA version expose the
     HIP-capable chunk kernel APIs expected by the shims in
     `ndm/models/external_gdn2.py`.
   - If the external package has CUDA-only assumptions, patch only the wrapper or
     document the exact external patch needed; do not vendor GDN2 source here
     unless licensing and project direction change.

4. Shared launcher change set, serialized after both arms have local evidence.
   - Update `scripts/frontier/debug_smoke_one_node.slurm` only after E97 and GDN2
     owners agree on environment variables and variant arguments.
   - Add short kernel-only pytest commands to the Frontier debug recipe before
     the 20-minute training smoke so failures separate compile/runtime issues
     from optimizer/data issues.
   - Preserve the existing one-job-at-a-time debug queue rule.

5. Frontier validation order.
   - Local syntax/import checks on login/head node where possible.
   - One-GCD interactive or debug allocation: E97 kernel-only tests.
   - One-GCD interactive or debug allocation: GDN2 import/fused path test.
   - One-node `srun -n8` smoke for `e97-MLP`, then `e97-linear-MLP`, then
     `gdn2-MLP`.
   - Only after all three finish or fail with classified errors, run benchmark
     matrix and extended readiness tasks.

## Observed Kernel And Invocation Paths

### e97-MLP

Observed model construction:

- CLI flags are defined in `train.py`: `--linear_state`, `--e88_raw_write`,
  `--mlp_ratio`, `--use_conv`, and `--gdn2_mlp_ratio`
  (`train.py:166`, `train.py:175`, `train.py:177`, `train.py:185`,
  `train.py:189`).
- `train.py` auto-enables `--use_triton 1` for E97/raw-write families under
  bf16 and prints a fused no-eager guard (`train.py:651`, `train.py:657`,
  `train.py:724`).
- `train.py` passes `linear_state`, `e88_raw_write`, `use_triton`, and
  `mlp_ratio` into `LadderLM` (`train.py:1094`, `train.py:1098`,
  `train.py:1109`, `train.py:1110`).
- `LadderLM` maps level `gdn2-mlp` separately but imports `E88FLAHybrid` for
  E97/E88-family levels (`ndm/models/ladder_lm.py:95`,
  `ndm/models/ladder_lm.py:107`, `ndm/models/ladder_lm.py:263`).

Observed kernel source:

- The sequential split-edit Triton path is the E88 Triton wrapper and kernels:
  `ndm/triton/e88_triton_forward.py:75`,
  `ndm/triton/e88_triton_backward.py:72`, and
  `ndm/triton/e88_triton_optimized.py:25`.
- The forward kernel has a sequential `for t in range(T)` loop
  (`ndm/triton/e88_triton_forward.py:161`), supports split-edit gates
  (`ndm/triton/e88_triton_forward.py:208`), and uses stable sigmoid-based tanh
  instead of raw exp (`ndm/triton/e88_triton_forward.py:237`).
- The wrapper pads unaligned eval sequence lengths to the checkpoint interval
  and slices outputs back (`ndm/triton/e88_triton_optimized.py:57`,
  `ndm/triton/e88_triton_optimized.py:68`,
  `ndm/triton/e88_triton_optimized.py:134`).
- `E88FLAHybrid` exposes `use_triton`, `linear_state`,
  `state_activation`, `use_split_edit`, `use_chunked_e97`, and
  `e97_chunk_size` knobs (`ndm/models/e88_fla_hybrid.py:848`,
  `ndm/models/e88_fla_hybrid.py:865`,
  `ndm/models/e88_fla_hybrid.py:876`,
  `ndm/models/e88_fla_hybrid.py:877`).
- The non-chunked `self.use_triton` path calls
  `e88_triton_optimized_apply` (`ndm/models/e88_fla_hybrid.py:1823`).

Expected code changes for ROCm:

- Add HIP backend logging and kernel-path assertions around the E97 fused path.
- Audit and tune Triton launch metadata in `e88_triton_forward.py` and
  `e88_triton_backward.py` for AMD wavefront/occupancy behavior.
- Keep the no-eager guard. If E97 cannot compile on HIP, fail loudly in debug
  and record the compiler error.

### e97-linear-MLP

Observed model construction:

- Frontier smoke maps `e97-linear-MLP` to `--level E97 --e88_raw_write 1
  --linear_state 1 --use_triton 1` plus MLP args
  (`scripts/frontier/debug_smoke_one_node.slurm:121`).
- The debug recipe repeats the same command mapping
  (`docs/FRONTIER_DEBUG_RECIPE_20260621.md:213`).
- `E88FLAHybrid` resolves `linear_state=True` to `state_activation='identity'`
  (`ndm/models/e88_fla_hybrid.py:915`, `ndm/models/e88_fla_hybrid.py:928`).

Observed kernel source:

- The chunked E97 derivation and PyTorch chunked reference live in
  `ndm/triton/e97_chunked.py:1`. That file explicitly scopes the chunked form
  to linear state (`ndm/triton/e97_chunked.py:60`) and says the per-step tanh
  cell is not associative (`ndm/triton/e97_chunked.py:62`).
- The fused fwd/bwd Triton implementation is
  `ndm/triton/e97_chunked_autograd.py:54` and
  `ndm/triton/e97_chunked_autograd.py:149`.
- The fused kernel uses `tl.dot` with `allow_tf32=ALLOW_TF32`
  (`ndm/triton/e97_chunked_autograd.py:104`,
  `ndm/triton/e97_chunked_autograd.py:121`) and fixed launch metadata
  (`ndm/triton/e97_chunked_autograd.py:377`,
  `ndm/triton/e97_chunked_autograd.py:405`).
- The fused kernel floors log-decay through `E97_GLOG_FLOOR`
  (`ndm/triton/e97_chunked_autograd.py:39`) and pads `T` to chunk size
  (`ndm/triton/e97_chunked_autograd.py:347`).
- `E88FLAHybrid` routes to the chunked kernel only when
  `use_chunked_e97`, `use_triton`, `use_split_edit`, `not raw_write`, and
  `linear_state` are all true (`ndm/models/e88_fla_hybrid.py:1751`).

Expected code changes for ROCm:

- Resolve the `--e88_raw_write 1` label issue for both `e97-MLP` and
  `e97-linear-MLP`: if the intended target is delta-correcting E97, remove
  `--e88_raw_write 1`; if the intended target is raw-write, rename the variants
  so benchmark summaries do not conflate E97 raw-write with E97 delta.
- Confirm whether current `e97-linear-MLP` Frontier smoke actually enables
  `use_chunked_e97`. The current smoke passes `--linear_state 1` and
  `--use_triton 1`, but this audit did not find a direct smoke flag for
  `use_chunked_e97`; if absent, the run uses the sequential E88 Triton path,
  not the chunked E97 throughput path.
- Add a debug-only assert or log line that reports
  `use_chunked_e97=<bool> e97_chunk_size=<int> log_decay=<bool>`.
- On HIP, first test `chunk_size=32`, `N=32`, `V<=64`. Do not start with
  `chunk_size=64` because the file comments say `C=64` is a higher shared-memory
  pressure configuration.

### gdn2-MLP

Observed model construction:

- `LadderLM` maps `gdn2-mlp` to `GDN2ExternalMLPLayer`
  (`ndm/models/ladder_lm.py:263`).
- `GDN2ExternalMLPLayer` composes `GDN2ExternalLayer`, `nn.RMSNorm`, and a
  bias-free SwiGLU MLP (`ndm/models/external_gdn2.py:249`,
  `ndm/models/external_gdn2.py:281`, `ndm/models/external_gdn2.py:291`).
- `GDN2ExternalLayer` loads an external checkout from `GDN2_PATH`, defaulting
  to `/home/erikg/GatedDeltaNet-2` (`ndm/models/external_gdn2.py:23`,
  `ndm/models/external_gdn2.py:28`, `ndm/models/external_gdn2.py:121`).
- The external class is loaded from `lit_gpt/gdn2.py` and its chunk op
  compatibility shim wraps `gdn2_ops.chunk_gdn2`
  (`ndm/models/external_gdn2.py:121`, `ndm/models/external_gdn2.py:147`).
- `train.py` asserts the external path exists and prints a fused no-eager guard
  for `gdn2` and `gdn2-mlp` (`train.py:731`).
- Frontier smoke maps `gdn2-MLP` to `--level gdn2-mlp --dim 2176 --depth 12
  --n_heads 30 --expansion 1.0 --use_conv 1 --d_conv 4`
  (`scripts/frontier/debug_smoke_one_node.slurm:138`).

Expected code changes for ROCm:

- Add a small import probe that verifies `GDN2_PATH`, `fla`, external
  `lit_gpt.gdn2`, and `gdn2_ops.chunk_gdn2` import before model construction.
- Record whether the imported external chunk kernel compiles for HIP. The
  repository does not vendor that source, so this audit cannot directly inspect
  its Triton code.
- Preserve the path-exists guard in `train.py`; improve it only with actionable
  module/function names.

## Existing Tests, Benchmarks, And Gaps

E97/e97-linear tests:

- `tests/test_e88_triton.py` covers E88 forward/backward, raw-write,
  linear-state, split-edit, and split-edit linear parity
  (`tests/test_e88_triton.py:36`, `tests/test_e88_triton.py:58`,
  `tests/test_e88_triton.py:75`, `tests/test_e88_triton.py:95`,
  `tests/test_e88_triton.py:115`, `tests/test_e88_triton.py:172`).
- `tests/test_e97_chunked.py` covers PyTorch chunked parity, fused Triton
  fwd/bwd parity, bf16 parity, log-decay parity, and strong-decay finite-gradient
  regressions (`tests/test_e97_chunked.py:44`,
  `tests/test_e97_chunked.py:101`, `tests/test_e97_chunked.py:137`,
  `tests/test_e97_chunked.py:184`, `tests/test_e97_chunked.py:261`,
  `tests/test_e97_chunked.py:302`, `tests/test_e97_chunked.py:341`).
- `tests/test_typed_head_mixture.py` has no-eager and unaligned-T fused-head
  checks for typed mixture E97 paths, but this is adjacent rather than the
  primary `e97-MLP`/`e97-linear-MLP` smoke.

GDN2 tests:

- `tests/test_cmaes_accounting.py` imports `GDN2ExternalLayer` for parameter and
  alias accounting, but it is not a HIP fused-kernel test.
- `scripts/verify_pinned_autotune.py` launches real `emender` and `gdn2-mlp`
  short training runs and checks pinned autotune behavior, fast init, loss
  parity, and fused guard (`scripts/verify_pinned_autotune.py:40`,
  `scripts/verify_pinned_autotune.py:57`, `scripts/verify_pinned_autotune.py:121`).
- Gap: no in-repo unit test directly imports and compiles the external GDN2
  Triton chunk kernel on HIP without starting a training run.

Frontier/debug benchmarks:

- `scripts/frontier/debug_smoke_one_node.slurm` is the committed one-node
  Frontier debug template. It loads ROCm 7.1.1, runs one process per GCD, records
  `torch.version.hip`, package versions, git status, and command line, then
  launches `train.py` (`scripts/frontier/debug_smoke_one_node.slurm:54`,
  `scripts/frontier/debug_smoke_one_node.slurm:66`,
  `scripts/frontier/debug_smoke_one_node.slurm:207`,
  `scripts/frontier/debug_smoke_one_node.slurm:238`).
- `docs/FRONTIER_DEBUG_RECIPE_20260621.md` documents the same variants and
  environment (`docs/FRONTIER_DEBUG_RECIPE_20260621.md:42`,
  `docs/FRONTIER_DEBUG_RECIPE_20260621.md:129`,
  `docs/FRONTIER_DEBUG_RECIPE_20260621.md:204`).
- Gap: the template jumps straight to a training smoke. Add kernel-only pytest
  commands first so compile errors are smaller and cheaper.

Baseline dependency pins:

- The package pins `torch==2.9.1` and `triton==3.5.1`
  (`pyproject.toml:27`, `pyproject.toml:31`). The Frontier debug recipe proposes
  a ROCm wheel/module environment that may differ; every smoke artifact must
  record exact versions.

## ROCm/HIP Compatibility Risk List And Smallest Debug Tests

| Risk | Smallest confirming/refuting test | Expected fix if confirmed |
| --- | --- | --- |
| E97 fused chunk kernel fails HIP compilation due to `tl.dot(..., allow_tf32=...)` or unsupported dot lowering | One GCD: `python -m pytest tests/test_e97_chunked.py::test_fused_triton_forward_parity_fp32 -q -s` with `HIP_VISIBLE_DEVICES=0` | Gate `ALLOW_TF32=False` on HIP if needed; add HIP-specific launch profile; keep parity thresholds explicit |
| E97 fused backward exceeds MI250X register/shared-memory limits at `C=32` or selected `(N,V)` | One GCD: `python -m pytest tests/test_e97_chunked.py::test_fused_triton_backward_parity_fp32 -q -s` | Reduce chunk size or launch metadata; split backward only if unavoidable; document unsupported shapes |
| bf16 E97 gradients diverge numerically on HIP | One GCD: `python -m pytest tests/test_e97_chunked.py::test_fused_triton_backward_parity_bf16 -q -s` | Disable TF32 assumptions on HIP; tighten fp32 accumulation paths; update tolerances only with measured evidence |
| Strong-decay/log-decay NaN fixes regress on HIP exp/cumsum lowering | One GCD: `python -m pytest tests/test_e97_chunked.py::test_strong_decay_cluster_backward_finite tests/test_e97_chunked.py::test_glog_floor_drift_overflow_finite -q -s` | Preserve exponent clamp and floor; inspect HIP codegen for `tl.exp`, `tl.cumsum`, masked upper-triangle behavior |
| `e97-MLP` label does not match flags because smoke uses `--e88_raw_write 1` | CPU/log-only: run `SMOKE_VARIANT=e97-MLP TRAIN_MINUTES=0 ...` or inspect the captured command in the smoke artifact | Rename to raw-write or remove `--e88_raw_write 1`; do this before comparing against GDN2 |
| Sequential E97/e97-MLP split-edit path compiles but is too slow or silently takes wrong path | One GCD: `python -m pytest tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference -q -s`, then `SMOKE_VARIANT=e97-MLP TRAIN_MINUTES=2 sbatch scripts/frontier/debug_smoke_one_node.slurm` | Add path logging in `E88FLAHybrid`; tune `BLOCK_H`, `num_warps`, and checkpoint interval; fail if no fused guard |
| e97-linear-MLP smoke does not exercise chunked E97 at all | One GCD or CPU log-only: run `SMOKE_VARIANT=e97-linear-MLP TRAIN_MINUTES=2 ...` and inspect for explicit `use_chunked_e97` log | Add CLI/layer kwarg for `use_chunked_e97` in the Frontier variant or rename smoke if it only tests sequential linear state |
| GDN2 external checkout missing on Frontier | Login node or one GCD: `GDN2_PATH=$MEMBERWORK/emender/src/GatedDeltaNet-2 python - <<'PY'\nfrom ndm.models.external_gdn2 import GDN2ExternalMLPLayer\nm=GDN2ExternalMLPLayer(dim=128, expansion=1.0, head_dim=16, num_heads=4)\nprint(type(m.gdn2.gdn2).__name__)\nPY` | Improve `GDN2_PATH` setup and error text; do not start training until import succeeds |
| External FLA/GDN2 imports expect CUDA-only modules or missing APIs | Same import probe plus `python - <<'PY'\nimport importlib\nprint(importlib.import_module('fla').__file__)\nPY` | Extend compatibility shims in `external_gdn2.py` or patch external checkout; record external commit |
| External GDN2 chunk kernel imports but fails HIP JIT | One GCD: `SMOKE_VARIANT=gdn2-MLP TRAIN_MINUTES=2 COMPILE_WARMUP_STEPS=1 sbatch scripts/frontier/debug_smoke_one_node.slurm` | Patch external FLA/GDN2 kernel launch metadata or select a HIP-supported FLA version |
| Autotune storm or slow first step consumes debug allocation | One GCD local/debug: `DATA=<smoke> STEPS=3 python scripts/verify_pinned_autotune.py` | Keep `NDM_PIN_TRITON_AUTOTUNE=1`; add pinned HIP configs after first successful compile |
| DDP/RCCL masks kernel result with distributed failure | Run kernel-only pytest and single-rank train command before one-node `srun -n8` | Classify distributed failures separately; do not call them kernel no-go evidence |
| Frontier environment differs from repo pins | The existing smoke env capture prints `torch.version.hip`, package versions, and `pip freeze` | Pin a known-good Frontier environment file only after successful debug evidence |

## Independence Recommendation

Implement E97/e97-linear and GDN2 in parallel only for disjoint source files:

- E97/e97-linear can proceed independently in `ndm/triton/e97_*`,
  `ndm/triton/e88_triton_*`, `ndm/models/e88_fla_hybrid.py`, and E97 tests.
- GDN2 can proceed independently in `ndm/models/external_gdn2.py`, GDN2 import
  probes, and GDN2 tests.

Serialize changes to shared files:

- `train.py`
- `scripts/frontier/debug_smoke_one_node.slurm`
- `docs/FRONTIER_DEBUG_RECIPE_20260621.md`
- `pyproject.toml`
- any shared smoke/benchmark harness, manifest, or result schema

Recommended order: run independent E97 and GDN2 implementation tasks first, then
a small integration task that edits the shared launcher and debug recipe once
both owners know their exact flags and environment needs.

## Prior Hypotheses Versus Observations In This Pass

Direct observations from this audit:

- The current E97 nonlinear smoke uses `--level E97 --e88_raw_write 1
  --linear_state 0 --use_triton 1` in the Frontier template.
- The current e97-linear smoke uses `--level E97 --e88_raw_write 1
  --linear_state 1 --use_triton 1` in the Frontier template.
- Because both smoke variants pass `--e88_raw_write 1`, their current labels do
  not by themselves prove they are testing the delta-correcting E97 cell.
- The fused chunked E97 implementation exists in
  `ndm/triton/e97_chunked_autograd.py`, but this pass did not find a direct
  Frontier smoke flag that enables `use_chunked_e97`.
- GDN2 source is not vendored here; this repo loads it from `GDN2_PATH`.
- Existing tests are CUDA-gated through `torch.cuda.is_available()` rather than
  Frontier/HIP-specific markers.

Prior statements treated as hypotheses until debug evidence exists:

- "GDN2 is mature ROCm via FLA" is plausible because GDN2 uses FLA external
  chunk kernels, but this pass did not compile those kernels on Frontier.
- "E97 is a no-go on Frontier" is not established by this audit. The code has
  explicit Triton kernels and guards; the actual status depends on HIP compile,
  parity, finite-gradient, and throughput tests above.
- "DDP is a no-go" is separate from kernel portability. `train.py` has DDP and
  DiLoCo branches, but kernel-only tests and single-rank debug smokes should run
  before attributing any one-node failure to Triton/HIP.
- Throughput claims such as E97 being faster/slower than GDN2 are historical
  local hypotheses until measured under the Frontier debug environment with the
  exact kernel path logged.

## Validation Commands For This Audit

Documentation-only local validation:

```bash
python -m py_compile train.py ndm/triton/e97_chunked.py ndm/triton/e97_chunked_autograd.py ndm/triton/e88_triton_forward.py ndm/triton/e88_triton_backward.py ndm/triton/e88_triton_optimized.py ndm/models/e88_fla_hybrid.py ndm/models/external_gdn2.py ndm/models/ladder_lm.py scripts/verify_pinned_autotune.py
python -m pytest tests/test_imports.py tests/test_standalone_minimal.py tests/test_e88_state_activation.py tests/test_cmaes_accounting.py -q
```

Frontier/debug validation to run after port changes:

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference -q -s
HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_e88_triton.py::test_e97_split_edit_linear_triton_matches_reference -q -s
HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_e97_chunked.py::test_fused_triton_forward_parity_fp32 -q -s
HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_e97_chunked.py::test_fused_triton_backward_parity_bf16 -q -s
SMOKE_VARIANT=e97-MLP TRAIN_MINUTES=2 sbatch scripts/frontier/debug_smoke_one_node.slurm
SMOKE_VARIANT=e97-linear-MLP TRAIN_MINUTES=2 sbatch scripts/frontier/debug_smoke_one_node.slurm
SMOKE_VARIANT=gdn2-MLP TRAIN_MINUTES=2 sbatch scripts/frontier/debug_smoke_one_node.slurm
```
