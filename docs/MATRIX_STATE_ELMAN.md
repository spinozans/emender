# Matrix State Elman: Trading Weight Capacity for State Capacity

## Summary

This document describes a modification to the Gated Elman (E1) architecture where the hidden state is a **matrix** rather than a vector. The key insight is that we can have **d² dynamic state parameters** for the same computational cost as E1's **d dynamic state parameters**, by using element-wise and outer-product operations instead of matrix-vector multiplies.

## Motivation

### Current E1 Architecture

```
State:  h ∈ ℝ^d           # d dynamic parameters
Weight: W_h ∈ ℝ^(d×d)     # d² learned parameters
Update: h' = tanh(W_h @ h + W_x @ x)
Cost:   O(d²) for W_h @ h
```

The d² capacity is in the **learned weights** (W_h), not the **dynamic state** (h).

### Proposed Matrix State Architecture

```
State:  H ∈ ℝ^(d×k)       # dk dynamic parameters
Update: H' = decay ⊙ H + key ⊗ value   # element-wise + outer product
Cost:   O(dk) for element-wise ops + O(dk) for outer product
```

If k = d, we get **d² dynamic state** for **O(d²) cost** - same as E1!

The key difference: the d² capacity is now in the **dynamic state** (H), not fixed weights.

## Mathematical Formulation

### State Representation

- **Hidden state**: `H ∈ ℝ^(d×k)` where d is model dimension, k is state expansion factor
- **State capacity**: dk parameters (vs d for vector state)
- When k = d: d² dynamic state parameters

### Update Rule

Given input `x ∈ ℝ^d` and current state `H ∈ ℝ^(d×k)`:

```
# Step 1: Compute key and value from input
key = tanh(W_key @ x + b_key)     # key ∈ ℝ^d, provides nonlinearity
value = W_val @ x + b_val          # value ∈ ℝ^k

# Step 2: Compute decay (can be scalar, per-row, or input-dependent)
decay = sigmoid(W_decay @ x + b_decay)  # decay ∈ ℝ^d or scalar

# Step 3: Update state via outer product
# This is O(dk) element-wise operations, NOT a GEMM
H_new = decay.unsqueeze(-1) * H + key.unsqueeze(-1) * value.unsqueeze(0)
# Equivalently: H_new[i,j] = decay[i] * H[i,j] + key[i] * value[j]

# Step 4: Output via matrix-vector multiply
query = W_query @ x + b_query      # query ∈ ℝ^k
output = H_new @ query             # output ∈ ℝ^d, cost O(dk)
```

### Computational Cost Analysis

| Operation | Cost |
|-----------|------|
| Key projection (W_key @ x) | O(d²) |
| Value projection (W_val @ x) | O(dk) |
| Decay computation | O(d) or O(d²) |
| State update (element-wise + outer) | O(dk) |
| Query projection | O(dk) |
| Output (H @ query) | O(dk) |
| **Total** | **O(d² + dk)** |

When k = d: Total is O(d²), same as E1.

But now we have **d² dynamic state** vs E1's **d dynamic state**.

## Key Insights

### 1. Outer Product as Associative Memory

The update `H' = decay * H + key ⊗ value` accumulates key-value pairs:

```
H = Σ (decay^t) * key_t ⊗ value_t
```

This is essentially **linear attention's memory state**. The matrix H stores associations between keys and values.

### 2. Nonlinearity Location

In E1: `h' = tanh(W_h @ h + ...)` - nonlinearity applied to h directly.

In Matrix State: `key = tanh(W_key @ x)` - nonlinearity in the key, not in H.

The nonlinearity is "one step removed" from the state. This may affect expressivity.

### 3. Rank Accumulation

Each outer product `key ⊗ value` is rank-1. After T steps with decay λ:

```
H_T = Σ_{t=1}^{T} λ^{T-t} key_t ⊗ value_t
```

Maximum rank of H_T is min(T, d, k). After d steps, H can be full rank.

### 4. Connection to Existing Work

- **Linear Attention**: Uses S = Σ kᵢ ⊗ vᵢ as state, queries via S @ q
- **Fast Weights**: Schmidhuber's idea of using outer products as "fast weights"
- **Mamba2**: Uses expanded state (d_state > d_output) with diagonal transitions

## Implementation Details

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixStateElmanLayer(nn.Module):
    """
    Elman RNN with matrix hidden state.

    State: H ∈ ℝ^(d×k) where d is model dim, k is state expansion
    Update: H' = decay * H + key ⊗ value
    Output: H @ query
    """

    def __init__(self, d_model: int, d_state: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or d_model  # k = d by default

        # Key projection: provides nonlinearity via tanh
        self.W_key = nn.Linear(d_model, d_model)

        # Value projection
        self.W_val = nn.Linear(d_model, self.d_state)

        # Query projection for output
        self.W_query = nn.Linear(d_model, self.d_state)

        # Decay: can be learned scalar, per-dimension, or input-dependent
        # Option 1: Learned scalar
        # self.log_decay = nn.Parameter(torch.zeros(1))

        # Option 2: Per-row decay (recommended)
        self.W_decay = nn.Linear(d_model, d_model)

        # Output projection (if d_state != d_model)
        if self.d_state != d_model:
            self.W_out = nn.Linear(d_model, d_model)
        else:
            self.W_out = None

    def forward(self, x: torch.Tensor, H: torch.Tensor = None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            H: Initial state of shape (batch, d_model, d_state) or None

        Returns:
            output: Shape (batch, seq_len, d_model)
            H_final: Final state shape (batch, d_model, d_state)
        """
        batch, seq_len, d = x.shape

        # Initialize state if needed
        if H is None:
            H = torch.zeros(batch, self.d_model, self.d_state,
                          device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d_model)

            # Compute key with nonlinearity
            key = torch.tanh(self.W_key(x_t))  # (batch, d_model)

            # Compute value
            value = self.W_val(x_t)  # (batch, d_state)

            # Compute decay (input-dependent)
            decay = torch.sigmoid(self.W_decay(x_t))  # (batch, d_model)

            # Update state: H' = decay * H + key ⊗ value
            # decay: (batch, d_model) -> (batch, d_model, 1)
            # key: (batch, d_model) -> (batch, d_model, 1)
            # value: (batch, d_state) -> (batch, 1, d_state)
            H = decay.unsqueeze(-1) * H + key.unsqueeze(-1) * value.unsqueeze(1)
            # H shape: (batch, d_model, d_state)

            # Compute output: H @ query
            query = self.W_query(x_t)  # (batch, d_state)
            out = torch.bmm(H, query.unsqueeze(-1)).squeeze(-1)  # (batch, d_model)

            # Optional output projection
            if self.W_out is not None:
                out = self.W_out(out)

            outputs.append(out)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return output, H

    def init_state(self, batch_size: int, device, dtype):
        """Initialize zero state."""
        return torch.zeros(batch_size, self.d_model, self.d_state,
                          device=device, dtype=dtype)
```

### Full Model Architecture

```python
class MatrixStateElman(nn.Module):
    """
    Full model with matrix state Elman layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int = None,  # State expansion factor
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            MatrixStateElmanLayer(d_model, self.d_state)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, states=None):
        """
        Args:
            input_ids: (batch, seq_len)
            states: List of H matrices, one per layer

        Returns:
            logits: (batch, seq_len, vocab_size)
            new_states: List of updated H matrices
        """
        x = self.embedding(input_ids)

        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            new_states.append(new_state)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_states
```

## Hyperparameter Recommendations

### State Expansion (k)

- `k = d`: Maximum state capacity for given cost. d² state parameters.
- `k = d // 2`: Half the state, but faster.
- `k = 2 * d`: More state, higher cost. Useful if memory is cheap.

### Decay Initialization

Initialize decay to produce values around 0.9-0.99 (slow forgetting):

```python
# Initialize W_decay bias so sigmoid output ≈ 0.95
nn.init.zeros_(self.W_decay.weight)
nn.init.constant_(self.W_decay.bias, 3.0)  # sigmoid(3) ≈ 0.95
```

### Key Projection

The nonlinearity in the key is critical. Options:
- `tanh`: Bounded, provides saturation
- `silu/swish`: Smooth, unbounded positive side
- `gelu`: Similar to silu

## Expected Properties

### Advantages over E1

1. **More state capacity**: d² vs d dynamic parameters (when k=d)
2. **Associative memory**: Naturally stores key-value associations
3. **Same computational cost**: O(d²) when k=d

### Potential Disadvantages

1. **No direct state nonlinearity**: H is updated linearly; nonlinearity only in key
2. **Different optimization landscape**: May need different learning rates
3. **Rank limitations**: Need multiple steps to build full-rank state

### Questions to Investigate

1. Does the extra state capacity translate to better loss?
2. Does the "delayed" nonlinearity (in key vs in state) hurt expressivity?
3. What's the optimal k for different model sizes?
4. How does this compare to Mamba2's state expansion?

## Experimental Plan

### Baseline Comparison

Compare at same parameter count and FLOP count:
- E1 (d=1760, depth=26): ~400M params
- Matrix State Elman (d=?, k=?, depth=?): ~400M params

### Ablations

1. **State expansion k**: k ∈ {d/4, d/2, d, 2d}
2. **Decay type**: scalar vs per-row vs input-dependent
3. **Nonlinearity**: tanh vs silu vs gelu in key
4. **With/without output projection**

### Metrics

- Training loss (nats)
- Throughput (tokens/sec)
- Memory usage
- Gradient flow (check for vanishing/exploding)

## References

- Fast Weight Programmers (Schmidhuber, 1992)
- Linear Attention / Performers (Choromanski et al., 2020)
- Mamba2 (Dao & Gu, 2024)
- Based: Simple Linear Attention (Arora et al., 2024)
