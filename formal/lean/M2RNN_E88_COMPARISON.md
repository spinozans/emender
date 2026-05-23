# M2RNN vs E88: Prior Art and Formal Comparison Plan

Date: 2026-05-08

## Why This Paper Matters

Mishra, Tan, Stoica, Gonzalez, and Dao posted **M2RNN: Non-Linear RNNs with Matrix-Valued States for Scalable Language Modeling** in March 2026:

- Paper: https://arxiv.org/abs/2603.14360
- Training code: https://github.com/open-lm-engine/lm-engine
- Kernels: https://github.com/open-lm-engine/accelerated-model-architectures
- Models: https://huggingface.co/collections/open-lm-engine/m2rnn

This is immediate prior art for any broad claim that E88 is the first scalable nonlinear recurrent language model with matrix-valued state. The correct framing is narrower:

> E88 is a nonlinear **delta-rule** matrix-state recurrent language model. M2RNN is a nonlinear **matrix-to-matrix transition** RNN with raw outer-product writes and an external forget carry.

Both are in the same high-level research family: nonlinear temporal recurrence plus matrix state. They are not the same architecture.

## Reported M2RNN Performance

M2RNN trains dense 410M models and 7B total / 1.1B active MoE models on 100B tokens from a high-quality Nemotron-CC-v2 subset. The strongest reported large-scale results are often hybrid models with attention interleaved every 8 layers.

At 7B MoE scale:

| Model | Wiki PPL | LAMBADA PPL | Avg 0-shot acc |
| --- | ---: | ---: | ---: |
| Mamba-2 | 13.73 | 11.03 | 65.22 |
| Gated DeltaNet | 13.89 | 11.10 | 64.28 |
| M2RNN | 13.80 | 11.48 | 65.17 |
| Hybrid Mamba-2 | 13.10 | 11.05 | 65.03 |
| Hybrid Gated DeltaNet | 13.51 | 11.06 | 64.64 |
| Hybrid M2RNN | 13.00 | 10.58 | 65.56 |
| Hybrid Gated DeltaNet + M2RNN-1 | 13.07 | 10.33 | 65.80 |
| Hybrid Gated DeltaNet + M2RNN-5 | 12.85 | 10.29 | 65.66 |

Interpretation:

- Homogeneous M2RNN is roughly at parity with Mamba-2 and ahead of Gated DeltaNet on average accuracy, but not cleanly ahead on every LM perplexity.
- Hybrid M2RNN is stronger than hybrid Mamba-2 and hybrid Gated DeltaNet in the reported 7B MoE table.
- Replacing a small number of Gated DeltaNet layers with M2RNN layers is especially strong.

For E88, this changes the novelty claim. The interesting claim is no longer "first nonlinear matrix-state RNN at scale." The interesting claim is whether a **pure dense nonlinear delta-rule matrix-state RNN** can approach Gated DeltaNet/Mamba2 on Pile-scale LM while retaining E88's synthetic state-tracking advantages.

## Core Update Rules

Use matrix state `H_t in R^{K x V}`.

### M2RNN

The M2RNN recurrent core is:

```text
Z_t = tanh(H_{t-1} W + k_t v_t^T)
H_t = f_t H_{t-1} + (1 - f_t) Z_t
y_t = H_t^T q_t + w_r * v_t
```

Important structural features:

- `W` is a learned right transition on the value/state-column dimension.
- `k_t v_t^T` is a raw outer-product write.
- `tanh` is applied to the candidate state `Z_t`.
- `f_t H_{t-1}` is an external linear carry path outside the tanh.
- The transition is nonlinear in the previous state because `H_{t-1}` passes through `tanh(H_{t-1} W + ...)`.

### E88

Ignoring orientation conventions, the E88 delta-rule core can be written:

```text
delta_t = v_t - H_{t-1}^T k_t
H_t = tanh(lambda_t H_{t-1} + k_t delta_t^T)
y_t = H_t^T q_t
```

Expanding the delta write:

```text
H_t = tanh((lambda_t I - k_t k_t^T) H_{t-1} + k_t v_t^T)
```

This is the key comparison point:

- M2RNN: `tanh(H W + k v^T)`
- E88: `tanh(A_t(k) H + k v^T)` where `A_t(k) = lambda_t I - k_t k_t^T`

So E88 is not just M2RNN with renamed variables. It is closer to a nonlinear, tanh-bounded Gated DeltaNet/DeltaNet: the write is error-correcting and creates an input-dependent projection-like transition.

## Formal Comparison Axes

The Lean framework should classify recurrent architectures by:

1. **State geometry**
   - Vector state: `h_t in R^d`
   - Matrix state: `H_t in R^{K x V}`

2. **Linearity in previous state**
   - Affine/linear-in-state: `H_t = A_t(x_t) H_{t-1} + B_t(x_t)`
   - Nonlinear-in-state: `H_t = phi(A_t H_{t-1} + B_t)` with nonlinear `phi`

3. **Transition side**
   - Right transition: `H W`
   - Left transition: `A H`
   - Bilateral transition: `A H W`
   - Elementwise/diagonal transition

4. **Transition dependence**
   - Fixed learned transition, e.g. M2RNN's `W`
   - Input-dependent transition, e.g. E88's `lambda I - k k^T`

5. **Write rule**
   - Raw write: `k v^T`
   - Delta/error-correcting write: `k (v - H^T k)^T`

6. **Carry placement**
   - Inside nonlinearity: `tanh(lambda H + write)`
   - Outside nonlinearity: `f H + (1-f)tanh(...)`

7. **Readout**
   - Query readout: `H^T q`
   - Residual value path: `H^T q + w_r * v`
   - Output gating/RMSNorm after readout

8. **Parallelism**
   - Linear scan-compatible
   - Chunkwise/recompute BPTT
   - Fully sequential nonlinear recurrence

## Theorem Roadmap

### 1. Recurrence Classification

Formalize:

```text
LinearInState(F) := exists A(x), B(x), F(H, x) = A(x) H + B(x)
NonlinearMatrixState(F) := matrix state and not LinearInState(F)
```

Expected results:

- Mamba2, RetNet, Gated DeltaNet are affine/linear in state.
- M2RNN is nonlinear matrix-state.
- E88 is nonlinear delta-rule matrix-state.

### 2. M2RNN/E88 Structural Separation

Show their feature signatures differ:

- M2RNN has a learned fixed right transition `W`.
- E88 has an input-dependent left transition `lambda I - k k^T`.
- M2RNN uses raw write `k v^T`.
- E88 uses delta write `k(v - H^T k)^T`.
- M2RNN has an external forget carry.
- E88 saturates the full next state directly.

### 3. Reduction Cases

Useful equivalences:

- If M2RNN has `W = I` and `f = 0`, its core reduces to `tanh(H + k v^T)`.
- If E88 removes the delta correction and uses `lambda = 1`, its raw-write variant also reduces to `tanh(H + k v^T)`.
- Therefore the shared ancestor is a nonlinear raw-write matrix RNN; the architectures diverge by transition and write rule.

### 4. Delta Transition Spectral Analysis

For normalized `k`, E88's expanded transition is:

```text
A_t = lambda I - k k^T
```

Eigenstructure:

- Along `k`: eigenvalue `lambda - ||k||^2`
- Orthogonal to `k`: eigenvalue `lambda`

This connects E88 to DeltaNet/Householder-style transition analysis. For `lambda ~= 1` and `||k||=1`, the key direction is erased or flipped depending on scaling, while orthogonal components are retained.

### 5. Expressivity

M2RNN claims it can represent all vector nonlinear RNN computations by embedding vector state into matrix state. We should prove the analogous embedding for E88 variants, with care:

- Raw-write E88 variant should be straightforward.
- Delta E88 requires handling the retrieval correction term.
- The delta term may help memory stability but complicates simulation proofs.

### 6. Experimental Follow-Ups

To position E88 cleanly against M2RNN:

- Add M2RNN baseline implementation or import their kernel where possible.
- Run the same synthetic expressivity suite: parity, FSM tracking, modular counter, Dyck/copy/assoc recall.
- Compare pure dense recurrent models at matched params/tokens/wallclock.
- Compare hybrid versions separately; M2RNN's strongest results are hybrid/MoE, not directly the same claim as pure E88.

## Revised Novelty Language

Avoid:

> first nonlinear matrix-state RNN at GPT-2 scale

Use:

> E88 is a nonlinear delta-rule matrix-state RNN. Unlike M2RNN's learned right-transition raw-write recurrence, E88 expands the delta correction into an input-dependent projection-like transition. Our evidence tests whether this delta-rule nonlinear matrix state can approach modern linear recurrent baselines in a pure dense Pile-scale setting while retaining stronger state-tracking behavior.

