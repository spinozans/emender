# Matrix Elman Triton Kernel Specification

## Overview

Implement Triton kernels for E70-E73 matrix state Elman models. These have O(n²) state where n = n_state (typically 64-256).

**Priority order:** E70 > E73 > E71 > E72

## Common Structure

All models share:
- State: `S` is `[B, n_state, n_state]` matrix
- Input projections: `k, v, q` are `[B, n_state]` vectors from `x`
- Output: `out = (S @ q) * silu(S @ q)` (self-gating)
- Sequential: NOT parallelizable (S-dependent updates)

## E70: Matrix Linear (Priority 1)

**Simplest - start here.**

```python
# Forward
decay = clamp(self.decay, 0, 0.999)  # scalar
for t in range(T):
    k = W_k @ x[t]                    # [B, n]
    v = W_v @ x[t]                    # [B, n]
    q = W_q @ x[t]                    # [B, n]

    S = decay * S + outer(v, k)       # [B, n, n]
    S = tanh(S)

    out = S @ q                       # [B, n]
    out = out * silu(out)             # self-gate
```

**Triton strategy:**
1. Fuse `outer(v, k)` + `decay * S` + `tanh` into single kernel
2. Fuse `S @ q` + `silu` self-gating
3. Batch k, v, q projections upfront (standard GEMM)

**Kernel signature:**
```c
e70_matrix_linear_forward(
    float* S,           // [B, n, n] state (in/out)
    float* out,         // [T, B, n] output
    float* x,           // [T, B, d] input
    float* W_k,         // [n, d]
    float* W_v,         // [n, d]
    float* W_q,         // [n, d]
    float decay,        // scalar
    int T, int B, int d, int n
)
```

## E73: Matrix Nonlinear (Priority 2)

**E1-style with S inside tanh.**

```python
for t in range(T):
    k = W_k @ x[t]
    v = W_v @ x[t]
    q = W_q @ x[t]
    z = sigmoid(W_z @ x[t] + b_z)     # modulation

    # Column modulation (default)
    S = tanh(S * z.unsqueeze(1) + outer(v, k))

    out = S @ q
    out = out * silu(out)
```

**Triton strategy:**
- The `S * z.unsqueeze(1)` is column-wise broadcast multiply
- Fuse: column_scale + outer_add + tanh

**Variants:**
- `column`: `S * z.unsqueeze(1)` - scale columns
- `row`: `S * z.unsqueeze(2)` - scale rows
- `full`: `S * outer(z, z)` - element-wise

## E71: Matrix Gated (Priority 3)

**S affects the gate.**

```python
for t in range(T):
    k = W_k @ x[t]
    v = W_v @ x[t]
    q = W_q @ x[t]
    alpha_x = W_alpha @ x[t]

    # S-dependent gate
    retrieved = S @ k                           # [B, n]
    alpha = sigmoid(alpha_x + d_alpha * retrieved + b_alpha)

    S = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * outer(v, k)

    out = S @ q
    out = out * silu(out)
```

**Triton strategy:**
- `S @ k` is matrix-vector multiply (can use existing matmul)
- Gate computation is element-wise
- Update has broadcast multiply

## E72: Matrix Self-Gate (Priority 4)

**S gates the value.**

```python
for t in range(T):
    k = W_k @ x[t]
    v = W_v @ x[t]
    q = W_q @ x[t]
    alpha = sigmoid(W_alpha @ x[t] + b_alpha)

    # S gates value
    retrieved = S @ k                           # [B, n]
    g = sigmoid(d_g * retrieved + b_g)          # gate from memory
    v_gated = v * g                             # memory controls writing

    S = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * outer(v_gated, k)

    out = S @ q
    out = out * silu(out)
```

## Key Operations to Optimize

### 1. Outer Product + Accumulate
```python
S = decay * S + outer(v, k)  # [B,n,n] = scalar * [B,n,n] + [B,n] x [B,n]
```
This is the core operation. Fuse into single kernel.

### 2. Matrix-Vector Multiply
```python
out = S @ q  # [B,n] = [B,n,n] @ [B,n]
```
Batched matmul, can use existing Triton matmul.

### 3. Column/Row Scaling
```python
S_scaled = S * z.unsqueeze(1)  # column scaling
S_scaled = S * z.unsqueeze(2)  # row scaling
```
Broadcast multiply, straightforward.

### 4. Self-Gating
```python
out = out * silu(out)  # [B,n]
```
Element-wise, can fuse with matmul output.

## Memory Layout

Use **row-major** for S: `S[b, i, j]` at offset `b*n*n + i*n + j`

This makes:
- `S @ q` (sum over j) efficient (contiguous read)
- `outer(v, k)` write efficient

## Backward Pass

For E70 backward:
```python
# Forward saved: S_all (all states), k_all, v_all, q_all, out_all

# Backward (reverse time)
dS = zeros_like(S)
for t in reversed(range(T)):
    # d(out * silu(out)) / d_out
    d_out_raw = d_out[t] * (silu(out[t]) + out[t] * sigmoid(out[t]) * (1 - sigmoid(out[t])))

    # d(S @ q) / dS, dq
    dS += outer(d_out_raw, q[t])
    dq = S[t].T @ d_out_raw

    # d(tanh(S_pre)) / dS_pre
    dS_pre = dS * (1 - S[t]**2)

    # d(decay * S_prev + outer(v, k))
    dS_prev = decay * dS_pre
    dv = dS_pre @ k[t]
    dk = dS_pre.T @ v[t]

    # Accumulate to weight gradients
    dW_q += outer(dq, x[t])
    dW_v += outer(dv, x[t])
    dW_k += outer(dk, x[t])
    dx[t] = W_q.T @ dq + W_v.T @ dv + W_k.T @ dk

    dS = dS_prev
```

## Testing

```bash
# Test Python implementations
python -m ndm.models.e70_matrix_linear
python -m ndm.models.e71_matrix_gated
python -m ndm.models.e72_matrix_selfgate
python -m ndm.models.e73_matrix_nonlinear

# After Triton implementation
python -c "
from ndm.models.e70_matrix_linear import E70MatrixLinear
import torch
model = E70MatrixLinear(dim=512, n_state=64).cuda().bfloat16()
x = torch.randn(2, 32, 512, device='cuda', dtype=torch.bfloat16)
out, S = model(x)
out.sum().backward()
print('E70 Triton test passed!')
"
```

## File Structure

```
elman/
  models/
    e70_matrix_linear.py      ✓ (implemented)
    e71_matrix_gated.py       ✓ (implemented)
    e72_matrix_selfgate.py    ✓ (implemented)
    e73_matrix_nonlinear.py   ✓ (implemented)
  triton_kernels/
    e70_matrix_linear_triton.py    (to implement)
    e71_matrix_gated_triton.py     (to implement)
    e72_matrix_selfgate_triton.py  (to implement)
    e73_matrix_nonlinear_triton.py (to implement)
```

## Performance Target

- n_state = 64: Should be ~50k tok/s at 100M params
- n_state = 128: Should be ~30k tok/s at 100M params
- Compare against FLA-GDN (75k tok/s) and E68 (78k tok/s)

The O(n²) state makes these inherently slower than O(n) vector state models, but they may have better token efficiency (loss per token).
