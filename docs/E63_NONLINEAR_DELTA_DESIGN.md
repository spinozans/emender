# E63: Nonlinear Delta - UTM-Class Gated Recurrence

## The Problem with E61/E62

E61/E62 are just "vector DeltaNet" - same linear dynamics, different state shape:

```
E61: h_t = α_t * h_{t-1} + (1-α_t) * v_t      # Linear in h
E62: h_t = (1-k_t) * h_{t-1} + k_t * v_t       # Linear in h
```

**These are NOT Turing complete.** Linear recurrence can only compute:
- Linear combinations of past inputs
- Weighted by products of gate values

No matter how sophisticated the gating, if state evolution is linear in h, we're limited to regular languages in the limit.

## What We Need for UTM Expressivity

Siegelmann-Sontag (1995) proved RNNs are Turing complete with:
1. Nonlinear activation (sigmoid/tanh)
2. Recurrent state dependence: h_t = f(h_{t-1}, x_t) where f is nonlinear in h

The minimal UTM-class RNN:
```
h_t = tanh(W @ h_{t-1} + U @ x_t)
```

But this has vanishing gradients. The question: **Can we have nonlinear temporal dynamics AND gradient control?**

## E63: The Answer - Nonlinear Value, Gated Mixing

### Core Insight

LSTM solved this! The cell state update is:
```
c_t = f_t * c_{t-1} + i_t * g_t
     ↑ linear mixing     ↑ but g_t = tanh(W @ h + U @ x) is NONLINEAR in h!
```

E63 applies this insight to gated delta:

```python
# Gates (control gradient flow)
α_t = sigmoid(W_α @ x_t)           # Retain gate (0-1)
β_t = sigmoid(W_β @ x_t)           # Write gate (0-1)

# NONLINEAR value (h-dependent for UTM expressivity!)
v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)

# Gated update (linear mixing of h and nonlinear v)
h_t = α_t * h_{t-1} + β_t * v_t
```

### Why This Works

**Jacobian:**
```
∂h_t/∂h_{t-1} = diag(α_t) + diag(β_t) * diag(tanh'(...)) * W_h
              = diag(α_t) + diag(β_t * (1 - v_t²)) * W_h
```

**Gradient path analysis:**
1. **Identity path**: α_t → when α ≈ 1, gradient flows through identity
2. **Nonlinear path**: β_t * tanh' * W_h → provides h-dependent computation

When the model needs memory: α → 1, gradient preserved
When the model needs to compute: β → 1, nonlinear transformation applied

**This is exactly LSTM's trick but cleaner!**

---

## E63 Variants

### E63a: Basic Nonlinear Delta
```python
α_t = sigmoid(W_α @ x_t)
β_t = 1 - α_t                              # Complementary (like GRU)
v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
h_t = α_t * h_{t-1} + β_t * v_t
```

### E63b: Independent Gates (like LSTM)
```python
α_t = sigmoid(W_α @ x_t)                   # Retain (forget gate)
β_t = sigmoid(W_β @ x_t)                   # Write (input gate)
v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)  # Candidate
h_t = α_t * h_{t-1} + β_t * v_t

# Note: α and β independent → can both be high (grow) or both low (shrink)
# May need normalization or careful init
```

### E63c: H-Dependent Gates (Self-Referential)
```python
# Gates ALSO depend on h (maximum expressivity!)
α_t = sigmoid(W_α @ x_t + U_α @ h_{t-1})
β_t = sigmoid(W_β @ x_t + U_β @ h_{t-1})
v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
h_t = α_t * h_{t-1} + β_t * v_t
```

This is closest to full LSTM and the "self-referential" models from Irie 2022.

### E63d: Residual Nonlinear (E60-style but gated)
```python
α_t = sigmoid(W_α @ x_t)                   # How much to trust the update
v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
h_t = h_{t-1} + α_t * v_t                  # Residual: always keep h, add scaled nonlinear update
```

Jacobian = I + diag(α_t * tanh'(...)) * W_h

This is E60 but with input-dependent gating on the nonlinear term!

---

## Matrix State Variants

### E63m: Nonlinear Matrix Delta

For maximum expressivity, use matrix state with nonlinear operations:

```python
# Matrix state S ∈ ℝ^(d×d)
k_t = W_k @ x_t                            # Key
q_t = W_q @ x_t                            # Query

# NONLINEAR retrieval (the key difference from linear DeltaNet!)
retrieved = tanh(S_{t-1} @ k_t)            # Nonlinear read!

# Value depends on retrieved content + input
v_t = W_v @ x_t + W_r @ retrieved          # Or: v_t = tanh(W_r @ retrieved + W_v @ x_t)

# Gated update
α_t = sigmoid(W_α @ x_t)
S_t = α_t * S_{t-1} + (1 - α_t) * v_t @ k_t^T

# Output
y_t = S_t @ q_t                            # Or: y_t = tanh(S_t @ q_t)
```

The nonlinear retrieval `tanh(S @ k)` is what makes this UTM-class.
DeltaNet's `S @ k` is just linear - no computation happening.

### E63m-RNN: Recurrent in the Fast Net (Irie 2021)

From "Going Beyond Linear Transformers":
```python
# Delta RNN variant
S_t = S_{t-1} + v_t @ k_t^T                # Standard delta update
y_t = S_t @ q_t + R_t @ tanh(y_{t-1})      # RECURRENT fast net!

# R_t could also be a fast weight matrix updated via delta rule
```

This makes the OUTPUT recurrent, not just the state.

---

## Comparison Table

| Model | State | Update | h-Dependence | UTM? | Parallelizable? |
|-------|-------|--------|--------------|------|-----------------|
| E42 | Vector | W @ (h+x) | Linear (W @ h) | No | No |
| E61 | Vector | α*h + (1-α)*v | None (v from x only) | No | Yes |
| E62 | Vector | (1-k)*h + k*v | None | No | Yes |
| DeltaNet | Matrix | α*S + β*v@k^T | None (linear S@k) | No | Yes |
| **E63** | Vector | α*h + β*tanh(Wh+Ux) | **Nonlinear!** | **Yes** | No |
| **E63m** | Matrix | α*S + β*v@k^T, v=f(S@k) | **Nonlinear!** | **Yes** | No |
| LSTM | Vector | f*c + i*tanh(Wh+Ux) | Nonlinear | Yes | No |

---

## The Expressivity-Parallelism Tradeoff

This is fundamental and unavoidable:

**Linear in h** (E61, E62, Mamba, DeltaNet):
- Associative: (A @ B) @ C = A @ (B @ C)
- Enables parallel scan
- But limited expressivity (can't do state-dependent computation)

**Nonlinear in h** (E63, LSTM, standard RNN):
- Not associative: f(g(h)) ≠ composable
- Must be sequential
- But UTM-class expressivity

**The Irie papers showed**: Delta LSTM (nonlinear) beats Delta Net (linear) on algorithmic tasks, but at 4-5x slower training.

**Our bet with E63**: The expressivity gain is worth the parallelism loss, especially at scale where:
1. Longer sequences need more computation (not just more memory)
2. Complex reasoning requires state-dependent branching
3. The "workarounds" (chunk-wise TTT, Titans) add complexity without full benefit

---

## Implementation Priority

1. **E63a (complementary gates)** - Simplest, closest to GRU
2. **E63c (h-dependent gates)** - Maximum expressivity, closest to LSTM
3. **E63d (residual nonlinear)** - E60 fixed with proper gating
4. **E63m (matrix state)** - If vector state isn't expressive enough

## Key Questions to Answer

1. Does E63 beat E42 at 100M scale on loss?
2. Does E63 beat E61/E62 on algorithmic/reasoning tasks?
3. Is the nonlinearity actually being used? (Ablation: random vs learned W_h)
4. What's the speed penalty vs linear variants?

---

## Why This Matters

The field has been chasing parallelization (Mamba, DeltaNet, Linear Attention) at the cost of expressivity. But:

1. **Transformers aren't parallel in the temporal dimension either** - they're just parallel in the sequence dimension
2. **Inference is sequential anyway** - you generate one token at a time
3. **Training can use teacher forcing** - which is parallel even for RNNs!

Maybe the right answer isn't "make RNNs linear so they parallelize" but "make RNNs nonlinear and accept the sequential training cost."

E63 tests this hypothesis: **Is nonlinear temporal expressivity worth the parallelism loss?**

---

*Created: 2026-01-14*
