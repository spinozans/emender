# Stability and Parameterization: Lessons from Mamba2 and FLA-GDN

## Executive Summary

Our 8-model comparison benchmark (100M params, 10 min training) revealed a significant performance gap:
- **Mamba2**: 1.23 loss (best)
- **FLA-GDN**: 1.61 loss
- **E68**: 1.69 loss (best E-series)
- **E75 variants**: 1.91-2.32 loss (underperforming despite richer state)

This document analyzes why E-series models, particularly E75 with its theoretically more expressive matrix state, underperform compared to Mamba2/FLA-GDN. We identify specific initialization and normalization techniques that likely account for this gap.

## The E75 Paradox

E75 maintains a matrix state S ∈ ℝ^{n×n} updated via the gated delta rule:
```
S_t = tanh(β * S_{t-1} + outer(δ_t, k_t))
```

**Theoretical advantages:**
- O(n²) state capacity vs O(n) for vector-state models
- Can encode richer temporal dependencies
- Sparse outer-product updates allow selective memory writes

**Observed reality:**
- E75n32: 1.93 loss, 57K tok/s
- E75n48: 1.91 loss, 51K tok/s
- E75n64: 2.32 loss, 42K tok/s (worst performance, slowest)
- Larger n_state → worse performance (opposite of expectation)

**Key observation**: Gradient norms during E75 training show extreme variance:
- E75n32 step 260: grad=344 (vs typical 2-5 for stable models)
- E75n32 step 350: grad=225
- These spikes suggest numerical instability in the matrix update

## What Mamba2 and FLA-GDN Do Differently

### 1. Log-Space Parameterization for Decay Factors

**Mamba2/FLA-GDN:**
```python
# A stored in log-space, initialized Uniform[1, 16]
A = torch.empty(nheads).uniform_(1, 16)
self.A_log = nn.Parameter(torch.log(A))

# Forward pass: A = -exp(A_log)
decay = -self.A_log.float().exp()
```

**Why this matters:**
- Decay factors must be negative (for stability) and bounded
- Direct parameterization allows A to drift into unstable regions
- Log-space ensures `exp(A_log)` is always positive
- The negative sign ensures decay, not growth
- Gradients are scaled by the parameter value, preventing explosion

**E-series current approach:**
- Direct weight matrices or sigmoid-bounded gates
- No log-space reparameterization
- Susceptible to gradient explosion when decay approaches instability

### 2. Inverse Softplus for Timestep Bias

**Mamba2/FLA-GDN:**
```python
# Initialize dt in log-uniform [0.001, 0.1]
dt = torch.exp(
    torch.rand(nheads) * (log(0.1) - log(0.001)) + log(0.001)
)
dt = torch.clamp(dt, min=1e-4)

# Store in inverse-softplus space
inv_dt = dt + torch.log(-torch.expm1(-dt))
self.dt_bias = nn.Parameter(inv_dt)

# Forward: dt = softplus(raw + dt_bias)
```

**Why this matters:**
- Ensures timestep scaling is always positive
- softplus(x) = log(1 + exp(x)) is smooth and bounded below by 0
- Initial values in [0.001, 0.1] provide reasonable learning dynamics
- Prevents dt from becoming zero (division issues) or exploding

**E-series current approach:**
- No explicit timestep parameterization
- Gates often initialized near 0.5 (sigmoid) or with small bias
- Missing the fine-grained control over update magnitude

### 3. Weight Decay Exemptions

**Mamba2/FLA-GDN:**
```python
self.A_log._no_weight_decay = True
self.dt_bias._no_weight_decay = True
self.D._no_weight_decay = True  # Skip connection
```

**Why this matters:**
- These parameters control fundamental recurrence dynamics
- Weight decay pushes them toward zero, destabilizing the SSM
- A_log → 0 means A → 1, which is neutral (no decay)
- dt_bias → 0 disrupts the learned timestep scaling

**E-series current approach:**
- All parameters subject to weight decay
- Structural parameters (gates, decay factors) are regularized alongside content parameters
- This may prevent learning optimal recurrence dynamics

### 4. Per-Head Normalization Inside Recurrence

**Mamba2:**
```python
# RMSNormGated with per-head grouping
self.norm = RMSNormGated(
    d_ssm,
    eps=1e-5,
    group_size=d_ssm // ngroups  # Per-head normalization
)
```

**FLA-GDN:**
```python
# L2 normalization on keys (critical for delta rule)
k_norm = k / (k.norm(dim=-1, keepdim=True) + eps)

# Fused RMSNorm with gating on output
o = self.o_norm(o, g)  # FusedRMSNormGated
```

**Why this matters for E75:**
- Matrix state S can have elements growing unboundedly
- Without normalization, outer(δ, k) accumulates without bound
- L2-normalized keys ensure bounded contribution to S
- Per-head normalization prevents cross-head interference

**E-series current approach:**
- Layer-level RMSNorm only (between layers)
- No normalization inside the recurrence
- E75's matrix state has no internal normalization

### 5. Residual Pathway Scaling

**Mamba2/FLA-GDN:**
```python
# Scale output projections by depth
with torch.no_grad():
    out_proj.weight /= math.sqrt(2 * num_hidden_layers)
```

**Why this matters:**
- Deep networks accumulate activations through residual connections
- Without scaling, gradient magnitude grows with depth
- 1/√(2L) scaling maintains constant variance through the network

**E-series current approach:**
- No depth-aware initialization
- 20-layer networks may have gradient flow issues

## Specific Issues with E75

### Issue 1: Unbounded Matrix State
```python
# Current E75 update
S = tanh(beta * S + outer(delta, k))
```

The tanh bounds the final state, but `outer(delta, k)` can be arbitrarily large before tanh. With high-variance inputs:
- Large outer products → tanh saturation → vanishing gradients
- Small outer products → tanh linear regime → potential instability

**Proposed fix:** L2-normalize k before the outer product:
```python
k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
S = tanh(beta * S + outer(delta, k_norm))
```

### Issue 2: Beta Gate Dynamics
```python
# Current: beta from sigmoid
beta = sigmoid(beta_proj(x) + beta_bias)  # Range [0, 1]
```

This is reasonable but lacks the log-space benefits:
- Gradients through sigmoid can vanish at extremes
- No fine-grained control over decay rate

**Proposed fix:** Log-space decay with softplus:
```python
# Store beta_log initialized in stable range
beta_log = nn.Parameter(torch.zeros(n_state).uniform_(0.5, 2.0).log())

# Forward: bounded positive decay
beta = (-beta_log.exp() * softplus(beta_proj(x) + dt_bias)).exp()
```

### Issue 3: No Weight Decay Exemptions
E75's gate parameters are regularized like content weights, potentially disrupting learned dynamics.

### Issue 4: Gradient Variance
The observed gradient spikes (up to 344) suggest:
- Occasional large outer products
- Gradient accumulation through matrix state
- Lack of gradient clipping or normalization

## Research Direction

### Phase 1: Log-Space Parameterization for E75

1. **Add A_log parameter** for state decay:
   ```python
   self.A_log = nn.Parameter(torch.zeros(n_state).uniform_(1, 16).log())
   self.A_log._no_weight_decay = True
   ```

2. **Add dt_bias parameter** for update magnitude:
   ```python
   dt = torch.exp(torch.rand(n_state) * (log(0.1) - log(0.001)) + log(0.001))
   self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
   self.dt_bias._no_weight_decay = True
   ```

3. **Modify forward pass**:
   ```python
   # Decay factor in log-space
   decay = (-self.A_log.exp() * F.softplus(gate_input + self.dt_bias)).exp()

   # Key normalization
   k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

   # Update with bounded decay
   S = decay.unsqueeze(-1) * S + outer(delta, k_norm)
   ```

### Phase 2: Apply to E61, E68, E42

These models already perform reasonably well. Adding log-space parameterization may provide incremental gains:

1. **E68** (self-gating, h-dependence): Add log-space for recurrence weight
2. **E61** (decay-gated): Natural fit for log-space decay factor
3. **E42** (linear tied): Add A_log for spectral radius control

### Phase 3: Ablation Studies

Test each component independently:
1. Log-space A only
2. dt_bias only
3. Key L2 normalization only
4. Weight decay exemptions only
5. All combined

### Phase 4: CUDA Kernel Updates

Once Python implementations are validated, update CUDA kernels:
- `e75_gated_delta_gpu.cu.cc`: Add log-space computation
- Ensure numerical stability in exp/log operations
- Consider mixed precision (compute in fp32, store in bf16)

## Expected Outcomes

If our hypothesis is correct:
1. **E75 with log-space**: Should see 0.2-0.4 nat improvement (from ~1.9 to ~1.5-1.7)
2. **Gradient stability**: Gradient norms should stay below 10 consistently
3. **Scaling with n_state**: Larger n_state should finally show benefits
4. **Throughput**: Minimal impact (log/exp are fast on GPU)

## Success Criteria

- E75n64 matching or exceeding FLA-GDN (1.61 loss)
- Stable training without gradient spikes
- Positive scaling with n_state (larger = better)
- Competitive throughput (>50K tok/s)

## References

- Mamba2: `mamba_ssm/modules/mamba2.py` - A_log, dt_bias initialization
- FLA-GDN: `fla/layers/gated_deltanet.py` - Delta rule with L2 normalization
- E75 current: `elman/models/e75_gated_delta.py`, `elman/cuda/lib/e75_gated_delta_gpu.cu.cc`
