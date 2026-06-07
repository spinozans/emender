# E97 Fused LM Kernel — Wiring, Parity, Speedup (task `wire-fused-e97`)

**TL;DR.** The fused E97 split-edit Triton fwd/bwd kernel (the `--use_triton_e88`
path) is now actually engaged in the hybrid / convergent / audit LM training path.
It was previously **inert** in that path — wired at the flag level but silently
falling back to the eager PyTorch T-scan. The fix makes E97 / e97-raw / e97-delta
arms train **43–266× faster** (the multiplier grows with sequence length) with
**numerical parity** to the eager reference at the actual bf16 precision, verified
forward + backward at T = 128 / 512 / 1024. The expressivity/audit runner and the
real-Pile `train.py` LM path now **default to fused**.

---

## 1. The bug: `--use_triton_e88` was inert in the hybrid path

The model-level wiring (`train_hybrid.py --use_triton_e88` → `HybridLadderLM` →
`E88FLAHybrid(use_split_edit=True, use_triton=…)`) already existed (added with the
expressivity harness, commit `dd632ab`). But it never fired:

- The fused kernel's dispatch gate requires `x.dtype == torch.bfloat16` (the Triton
  kernel is bf16-only). See `ndm/models/e88_fla_hybrid.py` `use_optimized`
  (≈L1557) and the split-edit dispatch (≈L1653).
- In the hybrid path the model keeps **fp32 parameters** and relies on
  `torch.amp.autocast`. `RMSNorm` emits **fp32** under autocast, so the tensor
  entering each E97 layer is fp32 → the bf16 gate fails → `use_optimized` is
  `False` → the recurrence runs the **eager T-scan** regardless of
  `--use_triton_e88`.
- Production `train.py` / `LadderLM` does NOT hit this because `--bf16` casts the
  whole model to bf16, so its residual stream is already bf16 and the kernel
  engages — which is why the 1.3B leaderboard ran fused while the hybrid
  expressivity battery did not.

**Symptom that exposed it:** an initial end-to-end parity run reported *bit-exact*
(0.0) forward/backward AND *identical* throughput (1.0×) between `use_triton=True`
and `False` — impossible if one side were the Triton kernel (the kernel differs
from eager at the bf16 ~1e-3 level). Bit-exact + same speed = both sides eager.

### Fix (`ndm/models/hybrid_ladder.py`)

`HybridLadderLM` now casts the **E88/E97-family layer inputs to bf16** under
autocast when the fused path is requested (`cast_recurrent_bf16`, defaulting on
with `use_triton_e88`). This mirrors what `--bf16` gives the production path for
free. Gated on `torch.is_autocast_enabled()` and on the per-layer E88/E97 flag, so:

- `--use_triton_e88` (now the runner default) → bf16 input → **fused engages**.
- no flag / `--disable_autocast` → **unchanged** (fp32 eager). Fully backward
  compatible — no existing run changes behaviour unless it opts into the flag.

### Real-Pile `train.py` default (`train.py`)

`--use_triton` default changed `0 → None = AUTO`: for E97 / split-edit / raw-write
under `--bf16`, Triton is auto-enabled (it is their **only** fused path — the CUDA
register-owned kernel rejects split-edit and raw-write). Everything else keeps the
historical default. `--use_triton 0` still forces eager.

---

## 2. Parity — fused bf16 Triton vs eager bf16 reference (fwd + bwd, end-to-end)

Harness: `experiments/expressivity_tasks/verify_e97_fused_parity.py`. Two
bit-identically-initialised `HybridLadderLM`s (`layer_pattern=['E97']`, depth 4,
dim 256, 32 heads, N = V = 32, ~7.4 M params) differing ONLY in `use_triton_e88`;
**both fed bf16** so the comparison isolates kernel-vs-reference, not a bf16-vs-fp32
precision gap. Metric is relative-L2 (Frobenius) error — per-element relative diff
is meaningless here because logits that are ≈0 in one path blow the ratio up while
contributing nothing to the loss. REAL model, REAL recurrence, REAL random token
batches (no mocks).

| T | arm | fwd logits rel-L2 | loss relΔ | grad rel-L2 | loss-curve max relΔ | verdict |
|---|-----|------------------:|----------:|------------:|--------------------:|:------:|
| 128 | e97-raw    | 8.6e-4 | 9.3e-6 | 3.8e-3 | 4.0e-3 | PASS |
| 128 | e97 (delta)| 8.8e-4 | 1.9e-6 | 2.6e-3 | 3.3e-3 | PASS |
| 128 | e97-linear | 7.2e-4 | 6.7e-6 | 2.5e-3 | 9.6e-3 | PASS |
| 512 | e97-raw    | 9.6e-4 | 3.1e-5 | 6.7e-3 | 1.8e-3 | PASS |
| 512 | e97 (delta)| 7.8e-4 | 4.3e-6 | 4.9e-3 | 7.7e-4 | PASS |
| 512 | e97-linear | 7.7e-4 | 1.2e-5 | 5.3e-3 | 6.1e-3 | PASS |
| 1024| e97-raw    | 9.4e-4 | 2.7e-5 | 7.5e-3 | 4.7e-4 | PASS |
| 1024| e97 (delta)| 7.7e-4 | 2.0e-6 | 5.6e-3 | 1.3e-4 | PASS |
| 1024| e97-linear | 8.1e-4 | 2.1e-5 | 6.2e-3 | 1.1e-3 | PASS |

- Forward rel-L2 stays ~8e-4 across **all T** — bf16 drift does **not** grow with
  sequence length. Loss matches to <3e-5 relative; gradients to ≤7.5e-3 rel-L2;
  short shared-data training-loss curves track to <1% per step.
- Tolerances: fwd 3e-2, loss 1e-2, grad 3e-2, curve 5e-2 — all passed with 1–2
  orders of margin.
- Kernel-level parity (the underlying Triton kernel vs the torch reference) is also
  covered by `tests/test_e88_triton.py` (`test_e97_split_edit_triton_matches_reference`,
  `test_e88_triton_raw_write_matches_reference`, …) — 7/7 pass.

### Benign finding: the erase gate is dead under raw-write

For `e97-raw` (`raw_write=True`) the split-edit **erase gate** carries no learning
signal in either path — raw-write drops the delta/read term that consumes it. The
eager path leaves `erase_gate_proj` out of the autograd graph (`grad = None`); the
Triton path keeps it as a Function input but its gradient is exactly `0.0`. Both
mean "no update", so parity holds. The harness reports these as
`dead/disconnected` rather than failing on a spurious presence mismatch.

---

## 3. Speedup (bf16 fused vs the eager path the runner used before)

Same shape; full train step (fwd + bwd + opt). The eager column is what the runner
ran before this change (the fp32 / autocast T-scan).

| T (batch) | arm | fused tok/s | eager tok/s | speedup |
|-----------|-----|------------:|------------:|--------:|
| 128 (B32) | e97-raw    | 292,241 | 6,785 | **43.1×** |
| 128 (B32) | e97 (delta)| 284,301 | 5,056 | **56.2×** |
| 128 (B32) | e97-linear | 312,290 | 5,357 | **58.3×** |
| 512 (B8)  | e97-raw    | 303,240 | 1,736 | **174.7×** |
| 512 (B8)  | e97 (delta)| 272,655 | 1,347 | **202.4×** |
| 512 (B8)  | e97-linear | 273,845 | 1,386 | **197.6×** |
| 1024 (B4) | e97-raw    | 203,153 |   825 | **246.2×** |
| 1024 (B4) | e97 (delta)| 160,171 |   630 | **254.1×** |
| 1024 (B4) | e97-linear | 172,646 |   648 | **266.3×** |

The eager T-scan is sequential in T, so its cost scales with T while the fused
kernel parallelises — hence the multiplier climbs from ~43× at T=128 to ~266× at
T=1024. (Absolute eager tok/s here is higher than the 733 tok/s the task cited
because that figure was at 1.3B production scale; the *ratio* is the portable
result.) Measured on an idle RTX 6000 Ada; the running C / convergent studies on
the other GPUs were **not** disrupted (parity/benchmark ran <2 GB co-located, then
on freed GPUs).

---

## 4. Which runners default to fused now

- **Expressivity / audit battery** — `run_e97_raw_expressivity.py`: the three E97
  arms (`e97-raw`, `e97`, `e97-linear`; tanh/identity state, kernel-compatible) now
  run **bf16 + `--use_triton_e88`** by default. `--eager-fp32` reproduces the
  original fp32 eager arms. (`gdn` arm already ran bf16 via its FLA kernel.)
- **Hybrid / single-task** — `train_hybrid.py --use_triton_e88` now actually
  engages the fused kernel (the wiring fix). Any convergent/hybrid runner that
  builds the E97 arm flags + `--use_triton_e88` gets fused.
- **Real-Pile LM** — `train.py` AUTO-enables Triton for E97 / split-edit /
  raw-write under `--bf16` (default `--use_triton None`). The convergent and
  generalization-audit real-Pile arms therefore train fused without any extra flag.

### Guardrail — non-saturating state stays eager

Only **tanh / identity** state is kernel-compatible. The non-saturating variants
(`--state_activation relu|softplus`) are implemented **only** in the fp32 reference
recurrence; the kernel **raises** rather than silently run tanh
(`e88_fla_hybrid.py` ≈L1334). Those arms must keep `--disable_autocast` (fp32
eager). No E97 arm in the current battery uses them.

---

## 5. Files

- `ndm/models/hybrid_ladder.py` — bf16 cast for E88/E97 layers so the fused kernel
  engages under autocast (`cast_recurrent_bf16`, `_is_e88_layer`).
- `train.py` — `--use_triton` AUTO default → fused for E97/raw-write under bf16.
- `experiments/expressivity_tasks/run_e97_raw_expressivity.py` — E97 arms default
  to fused bf16; `--eager-fp32` opt-out.
- `experiments/expressivity_tasks/verify_e97_fused_parity.py` — the parity +
  throughput harness (reproduces every number above).

### Reproduce

```bash
# T=128 (fast); also --seq_len 512 / 1024 with smaller --batch
CUDA_VISIBLE_DEVICES=0 python experiments/expressivity_tasks/verify_e97_fused_parity.py
```
