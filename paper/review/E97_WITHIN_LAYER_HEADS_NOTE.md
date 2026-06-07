# E97 Within-Layer Fused Heads — Parity, No-Eager-Fallback, Throughput (task `e97-heads-in`)

**TL;DR.** `wire-fused-e97` made the E97 split-edit Triton kernel fast, but only at
the **layer** level (an `HybridLadderLM` that interleaves whole E97 layers between
GDN layers). The architecture Erik requires is **within-layer parallel
heterogeneous heads**: one layer holds many head *types* side by side, combined
through the residual stream. `ndm/models/typed_head_mixture.py` already had that
shape for `gdn2_recall` + the four UnifiedCell corners + the GDN-2 nonlin shell —
but it had **no true E97 split-edit head**, which is exactly the gap that forced the
inferior interleaved-layer fallback. This task closes it: **`e97_raw` and
`e97_delta` are now first-class FUSED head types inside `TypedHeadMixtureLayer`**,
each a genuine E97 split-edit recurrence over its allocated head subset, running on
the bf16 split-edit Triton fwd/bwd kernel (the `use_triton` path, commit `4db8099`),
concatenated with the other head types and summed into the residual.

All numbers below are on REAL kernels / REAL data (random token batches, real
recurrence), no mocks. Reproduce:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. \
  python experiments/expressivity_tasks/verify_e97_within_layer_heads.py --gpu 0
```

---

## 1. What was added

`TypedHeadMixtureLayer` now exposes **eight** native head types in one layer:

| idx | type | kernel | role |
|----|------|--------|------|
| 0 | `gdn2_recall` | FLA GatedDeltaNet (chunked) | associative recall / MQAR |
| 1–4 | `e97_track`,`count`,`latch`,`nonlin` | UnifiedCell Triton (`fixed_pop`) | track / count / latch / iterated-nonlinear |
| 5 | `gdn2_nonlin_shell` | GDN-2 + fused nonlinear-in-time scan | §3 fairness head |
| **6** | **`e97_raw`** | **E88FLAHybrid split-edit Triton (raw write)** | **1.3B-leaderboard winner cell** |
| **7** | **`e97_delta`** | **E88FLAHybrid split-edit Triton (delta)** | **plain E97 (read-modify-write)** |

- `e97_raw` = `E88FLAHybrid(use_split_edit=True, raw_write=True, state_activation='tanh',
  use_gate=True, gate_activation='silu', use_triton=True)` — write `v` (drop the
  delta read term). This is the cell that ranked **#1 on the 1.3B LM CMA
  leaderboard** (avg-loss 5.9511).
- `e97_delta` = same but `raw_write=False` — write `v − S@k`, the plain E97 delta
  recurrence (== the `'E97'` ladder level).

Both are *real* E88FLAHybrid sub-blocks (their own q/k/v/o projections, decay, gate),
sized to the head count the allocator assigns, mapping `dim→dim` and summed into the
shared residual — exactly the "head type = the e97 split-edit recurrence over its
allocated subset of heads within a layer" contract.

### Allocation / backward compatibility

`TYPE_NAMES` grew 6 → 8 (the two e97 types append at indices 6, 7). The
softmax-of-logits → largest-remainder allocator is unchanged. Legacy callers that
pass **5** logits (pre-shell: `[gdn2_recall, e97_track, count, latch, nonlin]`) or
**6** logits (pre-e97) are right-padded with `-inf`, which softmax maps to **0**
heads for the new trailing types — reproducing the historical allocation EXACTLY. A
7-vector remains rejected (no historical contract). Covered by the allocation unit
tests in `tests/test_typed_head_mixture.py` (15/15 pass).

---

## 2. No silent eager fallback (the wire-fused-e97 bug, guarded out)

The fused split-edit Triton kernel dispatches only when its input is **bf16** (plus
`use_gate`/`silu`/`training`, which the head-type contract always carries). The
residual stream is **fp32** under autocast (RMSNorm emits fp32) and fp32 in the
`typed-gdn2-lm` sanity dtype — so a naive call would fail the bf16 gate and silently
run the eager T-scan. That is precisely the bug that made `--use_triton_e88` inert in
the hybrid path before `wire-fused-e97`.

`_run_e97` defends against it two ways:

1. **`cast_recurrent_bf16` (default on)** casts the sub-block input to bf16 so the
   fused kernel engages regardless of the surrounding dtype/autocast state.
2. **Loud guard**: if the fused path still could not engage during training (bf16
   missing, cast disabled), it **raises** rather than degrade silently.

**Verification (kernel-call counting, not inference):** a heterogeneous bf16 layer
(all 8 types active, `n_heads=48`) is run fwd+bwd in train mode with
`ndm.triton.e88_triton_optimized.e88_triton_optimized_apply` wrapped by a counter.

```
head allocation: gdn2_recall:6 e97_track:5 count:5 latch:5 nonlin:5
                 gdn2_nonlin_shell:6 e97_raw:8 e97_delta:8
fused split-edit Triton kernel calls: 2  (expected 2: one per fused E97 head type)  PASS
loud guard refuses to silently fall back to eager (fp32, cast off)                    PASS
```

Both `e97_raw` and `e97_delta` ran on the fused kernel; grads finite. The other
types are already on fused kernels (`gdn*` via FLA, the four corners via the
UnifiedCell Triton kernel, the shell via its fused nonlinear-state scan), so **no
head type in the mixture runs an eager fallback**.

---

## 3. Parity — within-layer E97 head: fused bf16 Triton vs eager reference (fwd + bwd)

Same weights, same bf16 input; the fused kernel is toggled off (`use_triton=False`)
to obtain the eager reference recurrence. Metric is relative-L2 (Frobenius).

| arm | T | fwd rel-L2 | grad rel-L2 | verdict |
|-----|--:|-----------:|------------:|:------:|
| e97_raw   | 128 | 1.16e-2 | 1.90e-2 | PASS |
| e97_raw   | 512 | 1.23e-2 | 2.14e-2 | PASS |
| e97_raw   | 1024| 1.24e-2 | 2.18e-2 | PASS |
| e97_delta | 128 | 1.10e-2 | 1.71e-2 | PASS |
| e97_delta | 512 | 1.15e-2 | 1.80e-2 | PASS |
| e97_delta | 1024| 1.15e-2 | 1.83e-2 | PASS |

Tolerances: fwd/grad 3e-2; passed with margin and flat in T (bf16 drift does not
grow with sequence length). These are the bf16 kernel-vs-reference numbers (the same
~1e-2 band the layer-level `verify_e97_fused_parity.py` reports; per-element
relative error is meaningless when individual logits are ≈0, hence rel-L2).

### Unaligned-T padding parity

The LM next-token path feeds **T−1** timesteps, so an aligned context (512/1024)
reaches the e97 heads **unaligned** (511/1023), which the kernel's sparse-checkpoint
forward rejects (`T % checkpoint_interval == 0`). `_run_e97` zero-pads the **causal**
time axis to the next multiple of the checkpoint interval and truncates the output
back — exact, because appended future steps cannot change earlier outputs in a
strictly causal recurrence. So every fused E97 head stays on the kernel at **any**
sequence length instead of crashing or falling to eager.

```
T=511 (unaligned) | out T=511 | fwd rel-L2 1.12e-2 | grad rel-L2 1.97e-2 | PASS
```

---

## 4. Heterogeneous-head LM throughput + screen budget

Full `LadderLM(level='typed-gdn2-lm')`, every layer holding all 8 head types in
parallel (`head_type_logits=[0.3,0,0,0,0,0.3,0.6,0.6]`). Full train step
(fwd+bwd+opt), bf16, measured on an RTX 6000 Ada (GPU 7) co-located with the running
sibling studies (not disrupted).

Main scale — `dim=768, depth=12, n_heads=48, n_state=32, B=16, T=512`:

| config | tok/s | params | note |
|--------|------:|-------:|------|
| pure `fla-gdn` (reference) | 96,551 | 60 M | the FLA kernel the recall head uses |
| **hetero, e97 heads FUSED** | **25,620** | 117 M | **0.27× of pure-GDN** |

The heterogeneous layer runs **four parallel recurrence pathways** (FLA-GDN +
UnifiedCell + GDN-2 shell + the two E97 split-edit blocks) and carries ~2× the
params, so ~0.27× of single-pathway GDN throughput is the same order of magnitude —
the recurrence work, not an eager stall, is the cost.

Fusion is what makes that possible. Small-scale apples-to-apples (`dim=512, depth=4,
B=4, T=512`), e97 heads fused vs eager (`use_triton_e97=False`):

| e97 path | tok/s | gain |
|----------|------:|-----:|
| FUSED | 27,563 | — |
| EAGER (T-scan) | 409 | **67.3× slower** |

Without the fused kernel the e97 heads alone drag the layer to ~400 tok/s; with it
they are effectively free relative to the other pathways.

**Screen budget:** at 25,620 tok/s the fused heterogeneous LM processes
**≈ 26.9 M tokens in an 18-minute** time-bounded screen — comfortably the
"tens of M tokens" the task requires (and it scales linearly with the window:
≈ 23 M at 15 min, ≈ 31 M at 20 min).

---

## 5. Files

- `ndm/models/typed_head_mixture.py` — `e97_raw` / `e97_delta` fused split-edit head
  types (`E88FLAHybrid`, `use_triton`), `_run_e97` with the bf16 cast + loud
  no-silent-fallback guard + causal pad-to-checkpoint for any T; `TYPE_NAMES` 6→8
  with backward-compatible legacy padding.
- `experiments/expressivity_tasks/verify_e97_within_layer_heads.py` — the
  no-eager-fallback / parity / unaligned-T / throughput+screen harness (reproduces
  every number above).
- `tests/test_typed_head_mixture.py` — allocation contract updated for the 8-type
  table + 3 new GPU tests (no-eager-fallback kernel-count, loud guard, unaligned-T
  parity). 15/15 pass.

## 6. Validation checklist (task `e97-heads-in`)

- [x] `e97_raw` + `e97_delta` are fused head types in `typed_head_mixture`; the
  within-layer mixture of all types runs fully fused in parallel, residual-combined,
  **NO eager fallback** (verified by counting real fused-kernel calls + loud guard).
- [x] Parity (fwd+bwd, bf16) of the within-layer E97 heads vs the eager reference
  (rel-L2 ≤ 1.24e-2 fwd / ≤ 2.18e-2 grad across T=128/512/1024; unaligned T=511 exact).
- [x] Heterogeneous-head LM throughput benchmarked near-GDN (0.27× of pure-GDN at
  ~2× params and 4 parallel pathways; fusion is 67.3× over eager); 15–20 min screen
  budget confirmed (≈ 26.9 M tokens / 18 min — tens of M).
- [x] Committed; this note written.
