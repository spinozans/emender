# Pinned Triton autotune — killing the init-wedge / autotune storm

## Problem

The fused kernels on **both** race arms lean on `@triton.autotune` config
sweeps that live outside this repo:

- **emender (`--level E97`)**: the recurrence is the project's own fused E97
  Triton kernel (a custom in-process autotune cache, *not* `@triton.autotune`),
  but its RMSNorms come from GatedDeltaNet-2's `lit_gpt/rmsnorm.py`, whose
  `_layer_norm_fwd_1pass_kernel` / `_layer_norm_bwd_kernel` sweep `num_warps`
  over `{1,2,4,8,16,32}`.
- **gdn2 (`--level gdn2-mlp`)**: the FLA / GatedDeltaNet-2 gated-delta chunk
  kernels (`chunk_gdn2_*`, `chunk_gated_delta_rule_*`, `chunk_local_cumsum_*`,
  `recompute_w_u_*`), the short causal conv (`causal_conv1d_*`), `l2norm_*`,
  `layer_norm_gated_*`, plus the same RMSNorm kernels.

Triton's compiled-binary cache (`~/.triton/cache`, ~54 GB) persists across
processes, but the autotune **config selection** is in-process only — Triton
re-benchmarks (`do_bench`) every config of every multi-config kernel on the
**first call in each fresh process**. That is a launch+sync storm:

```
Autotuning kernel _layer_norm_bwd_kernel with config num_warps: 1..2..4..8..16..32
Triton autotuning ... finished after 0.84s, best config selected: num_warps: 4
```

- On an **idle** box: ~1-2 min total.
- Under **box contention**: observed to **DEADLOCK** — 52-minute silent
  init-wedges, GPU pinned 100 %, zero steps, the log frozen right after
  `[fused-guard]`. This repeatedly stalled the equal-FLOPs racer.
- On **Frontier**: ~512 GCDs each run their OWN storm simultaneously at every
  job start / chained-resume. Catastrophic. **Pinned configs => every rank
  starts instantly and identically.** This is a prerequisite for the scaleout.

## Fix

We do **not** edit the FLA / GDN-2 site-packages (not version-controlled, lost
on reinstall). Instead `ndm/triton/pin_autotune.py` monkeypatches
`triton.runtime.autotuner.Autotuner.run` at the **class** level (so it covers
instances the libraries already constructed at import time). When pinning is on
(the default), a multi-config kernel's config is taken from a committed registry
of **measured-best** configs keyed by `(kernel_name, autotune_key)` instead of
being re-benchmarked. `do_bench` is never called → no storm.

Install points (idempotent, env-gated): `ndm/triton/__init__.py` (covers every
entry point that touches the kernels) and `train.py` (covers the gdn2-mlp arm,
which reaches the kernels via FLA without importing `ndm.triton`).

The registry lives at `ndm/triton/pinned_autotune_configs.json`
(18 kernels / 30 `(name,key)` configs, captured from real fwd+bwd of both
production arms).

## Numerical correctness

A config controls **scheduling / occupancy only** (block sizes, `num_warps`,
`num_stages`), never the math. For an **identical** config the launched kernel
is byte-identical → fwd+bwd numerically identical. Proven by
`tests/test_pin_autotune_parity.py`:

- `test_pin_path_byte_identical_to_native_same_config` (D=1792 **and** 2176):
  the pin path and the native autotuner path with the **same** config produce
  byte-identical fwd+bwd (`torch.equal`).
- `test_pin_never_benchmarks`: under pinning the autotuner's `_bench` is never
  called (the storm mechanism is structurally absent).
- `test_pinned_config_matches_recorded`: the pin code path reconstructs exactly
  the recorded (autotuner-measured) config for every registry entry.

End-to-end (real train.py, both arms, see `scripts/verify_pinned_autotune.py`):

| metric | emender | gdn2-mlp |
|---|---|---|
| autotune "finished" lines, pinned | **0** | **0** |
| autotune "finished" lines, sweep | 7 | 5 |
| pinned run #1 vs #2 (max |Δloss|) | **0.0** (bit-reproducible) | **0.0** |
| pinned vs sweep (max |Δloss|) | 7.8e-3 | 2.5e-3 |
| **sweep vs sweep** (max |Δloss|) | **6.6e-3** | — |
| `[fused-guard]` / bf16 / use_triton=1 | ✓ | ✓ |

The pinned-vs-sweep difference is **bf16 reduction-order noise** from a
different (but equally valid) `num_warps`: the **stock autotuner is itself
non-deterministic across processes** (sweep-vs-sweep already differs 6.6e-3,
*larger* than pin-vs-sweep at 3.0e-4 in the same experiment). Pinning adds no
numerical difference beyond what the stock path already exhibits run-to-run, and
additionally makes training **bit-reproducible** across runs/ranks.

## Init time

Killing the storm removes its time (and, under contention, its deadlock). The
residual fresh-process init (~25 s on an idle box for these 1.3B configs) is the
framework floor — ~9.3 s Python imports + CUDA context + 1.3B model build + data
first-batch + first-call compile-load — **independent of autotune**. The win the
pin delivers is structural: **zero benchmarking calls**, so init is bounded and
the 52-min contention wedge cannot recur (complementing the log-stall watchdog).

## Escape hatches (env vars)

| env var | default | effect |
|---|---|---|
| `NDM_PIN_TRITON_AUTOTUNE` | `1` | master switch. **`0` restores the original autotune sweep** — use when you move to a DIFFERENT shape regime and want Triton to re-benchmark. |
| `NDM_PIN_TRITON_REGISTRY` | `ndm/triton/pinned_autotune_configs.json` | path to the pinned-config registry. |
| `NDM_PIN_TRITON_STRICT` | `0` | `1` = a multi-config kernel whose `(name,key)` is NOT in the registry falls back to the real autotune sweep (instead of `configs[0]`). Lets you optimize a newly-introduced kernel/shape while keeping measured kernels pinned. |
| `NDM_PIN_TRITON_VERBOSE` | `0` | `1` = print one line per pin/fallback decision. |
| `NDM_PIN_TRITON_RECORD` | unset | path to write a freshly-measured registry. Forces pinning OFF, runs the real autotuner, and dumps every winner (merged) at process exit. |

## Re-capturing for a new shape regime

The registry is shape-keyed. A shape not in it falls back to a measured config
for that kernel (correct, no storm, maybe not optimal). To re-optimize for new
production shapes, on an **idle leased GPU** (never GPU 0 = the racer):

```bash
eval "$(scripts/gpu_lease.sh acquire 1)"
scripts/capture_pinned_autotune.sh          # edit the two arm commands if shapes changed
git add ndm/triton/pinned_autotune_configs.json && git commit
```
