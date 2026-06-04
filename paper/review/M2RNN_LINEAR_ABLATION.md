# M2RNN raw-write state-nonlinearity ablation (tanh vs linear)

Task `m2rnn-linear-ablation`. Symmetric counterpart of the E88 `linear_state`
knob (s5sym-eval), restricted to the M2RNN-CMA winner. Completes the
2×2-by-family: *does the tanh-vs-linear state knob matter for raw-write the way
it (barely) did for delta-correction?*

**Answer: yes, and far more — and in the opposite direction.** For raw-write,
removing the state tanh **helps** substantially (linear ≫ nonlinear on both S5
and S3). For delta-correction (E88) the same knob barely moved the needle.

## What "linear M2RNN" means in the code

M2RNN raw-write writes a fresh candidate matrix state each step and blends it in
with the forget gate. The as-built update (`ndm/models/m2rnn_baseline.py`,
PyTorch loop) is:

```python
outer = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)   # k vᵀ  (rank-1 write)
pre = torch.matmul(h, W) + outer                # h W + k vᵀ
candidate = pre if self.linear_state else torch.tanh(pre)
h = f_t * h + (1.0 - f_t) * candidate           # forget-gated blend
y_t = q_tᵀ h  (+ D·v residual)
```

- **nonlinear (as built):** `Z = tanh(h W + k vᵀ)` — the released M2RNN family update.
- **linear (this ablation):** `Z = h W + k vᵀ` — the candidate state-nonlinearity removed.

`linear_state` is the **exact raw-write analogue of E88's `linear_state`**
(`ndm/models/e88_fla_hybrid.py:1709`: `S = decay·S + outer` vs
`tanh(decay·S + outer)`). Only the candidate's tanh is touched; the forget gate
`f_t`, the read-out `qᵀh`, the `D·v` residual, the silu output gate, and all
shapes/params are identical between arms.

**Code-semantics caveat (documented, not faked).** The upstream XMA Triton
kernel (`xma.layers.m2rnn`) hardcodes the tanh state-nonlinearity and exposes no
linear toggle, so it *cannot* express the linear variant. The toggle therefore
forces the PyTorch reference loop whenever `linear_state=True`. `XMA_PATH` is
unset in this environment, so `XMA_M2RNN_AVAILABLE=False` and the **nonlinear arm
also runs the PyTorch loop** — both arms share the identical code path, so the
comparison is matched. (If one wanted a linear variant inside the Triton kernel,
the kernel's `tanh` would have to be made conditional — a kernel edit, not a
config flag. We did not fake that; we documented it and used the loop.)

## Protocol

Identical to the E88 winner-eval (s5sym-eval). M2RNN-CMA winner shape/lr verbatim
from `winners/m2rnn.args.json`: dim=512, depth=5, n_heads=19, n_state=32,
lr=1.6058e-3 (~8.0M params). 3 seeds {42,123,456}; S5 `s5_permutation` train
T=128 / 20000 steps; S3 `s3_permutation` train T=128 / 10000 steps (solvable
control); eval grid {128,256,512,1024}; schedule-free AdamW, batch 32, seq_len
128; fp32 (autocast disabled for M2RNN). **GPUs 6,7 only.** Random baselines:
S5 = 1/120 = 0.83%, S3 = 1/6 = 16.67%.

Raw JSONs:
`experiments/expressivity_tasks/results/m2rnn_linear_ablation_20260604/eval/{arm}_{S5,S3}_seed{42,123,456}.json`.
The `m2rnn-nonlinear_*` runs are the s5sym-eval M2RNN winner runs reused verbatim
(same winner config, `linear_state=None`→default `False`→tanh, byte-identical to
`--linear_state 0` under the new code); provenance in the directory `README.md`.
The 6 `m2rnn-linear_*` runs were trained fresh for this task.

## Results — accuracy % (mean ± std over 3 seeds)

| Family | Arm | Task | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|---|---|
| raw-write | m2rnn-nonlinear | S5 | 16.6±3.5 | 8.6±1.7 | 4.8±0.9 | 2.8±0.5 |
| raw-write | **m2rnn-linear** | S5 | **31.1±4.9** | **15.9±2.0** | **8.3±1.0** | **4.5±0.7** |
| raw-write | m2rnn-nonlinear | S3 | 19.0±1.0 | 17.8±0.6 | 17.4±0.2 | 17.0±0.2 |
| raw-write | **m2rnn-linear** | S3 | **39.7±27.6** | **31.9±19.7** | **25.0±11.2** | **20.7±5.5** |
| delta-corr (ref) | e88-tanh | S5 | 98.9±1.1 | 63.0±3.7 | 32.2±1.8 | 16.8±1.1 |
| delta-corr (ref) | e88-linear | S5 | 100.0±0.0 | 75.2±12.5 | 39.1±7.8 | 20.0±3.3 |
| delta-corr (ref) | e88-tanh | S3 | 100.0±0.0 | 99.8±0.1 | 89.3±3.8 | 61.5±3.4 |
| delta-corr (ref) | e88-linear | S3 | 100.0±0.0 | 99.2±0.7 | 86.5±9.6 | 64.8±15.0 |

Per-seed at T=128 (to characterize the S3 variance honestly):

| Arm | S5 seeds {42,123,456} | S3 seeds {42,123,456} |
|---|---|---|
| m2rnn-nonlinear | 20.2 / 16.0 / 13.4 | 20.0 / 19.1 / 18.0 |
| m2rnn-linear | 36.8 / 28.0 / 28.5 | 32.3 / 16.7 / 70.3 |

## Interpretation

**The state-nonlinearity knob matters a lot for raw-write — removing the tanh
helps.** On S5, linear beats nonlinear in *every seed and at every eval length*:
T=128 jumps 16.6→31.1% (≈ ×1.9; +14.5pp, ~3σ separation), and the gain persists
out to T=1024 (2.8→4.5%). On S3 (the solvable control) the contrast is even more
telling: nonlinear M2RNN sits **at chance** in all three seeds (18–20% ≈ 16.67%),
i.e. it never learns S3 at all, whereas the linear variant escapes chance in 2 of
3 seeds (32.3% and 70.3%) — hence the large 39.7±27.6 spread. The tanh is not a
harmless bound here; it actively suppresses the raw-write model's ability to
track the permutation state.

**Why the asymmetry vs E88.** This is the *opposite* of the delta-correction
result. For E88 the tanh-vs-linear knob barely moved S5/S3 (S5@128 98.9 vs 100.0;
S3 essentially tied; small linear edge that grows only at the longest lengths,
within or near noise). E88 already solves the task either way, and its
delta-rule write (`S += outer(write − read, k)`) keeps the state well-scaled, so
the saturating tanh neither helps nor hurts much. M2RNN's raw rank-1 write
(`Z = tanh(h W + k vᵀ)`) is different: the tanh is applied to the *entire*
recomputed candidate every step, saturating the freshly-written key→value
association and compressing the dynamic range the read-out needs to separate
120 (S5) / 6 (S3) states. Linearizing the candidate removes that per-step
squashing and lets the write survive, which is why it nearly doubles S5 and
unlocks S3 learning.

**Caveat.** Even linearized, M2RNN remains a weak state-tracker on this probe
(S5@128 31% vs E88 ≈99–100%; S3 still seed-fragile). The knob changes M2RNN's
behavior substantially but does not close the gap to the delta-correction family.

## Bottom line for the 2×2-by-family

| | tanh (nonlinear) | linear | knob effect |
|---|---|---|---|
| **delta-correction (E88)** | solves S5/S3 | solves S5/S3 | **barely matters** (linear ≈ tanh, tiny edge) |
| **raw-write (M2RNN)** | weak S5, S3 at chance | ~2× S5, S3 off chance | **matters a lot — tanh hurts** |

The state-nonlinearity is load-bearing for the *raw-write* family and inert for
the *delta-correction* family: where the write rule keeps the state bounded (E88),
the tanh is redundant; where the write rule recomputes and could over-write
(M2RNN), the tanh saturates the signal and removing it helps.
