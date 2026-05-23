# Stability Fix Guidance for Level 6 and log_0

## Problem Diagnosis

After analyzing the CUDA kernels, the root cause is clear:

**The selective output (compete × silu) only bounds OUTPUT gradients, NOT recurrent gradients.**

### Level 6 Architecture (UNSTABLE)
```
Forward:
  1. h_new = (1-delta) * h_prev + delta * polynomial(v)   ← UNBOUNDED
  2. output = compete(h_new) * silu(W_out @ h_new)        ← Bounded output

Backward:
  dh_prev = (1-delta) * grad_h + delta * α * |v|^(α-1) * r_h * grad_h
            ↑ This path has NO normalization, gradients compound through time
```

### log_0 Architecture (UNSTABLE)
```
Forward:
  1. log_h = alpha * log_v (polynomial in log-space)
  2. log_h_bounded = soft_bound(log_h)                    ← Partially bounded
  3. output = RMSNorm(exp(log_h_bounded))                 ← No compete × silu!

Missing: No selective output mechanism at all
```

### log_1 Architecture (MARGINAL - has selectivity but still struggles)
```
Forward:
  1. Same polynomial as log_0
  2. h_linear = exp(log_h_bounded)
  3. h_normed = RMSNorm(h_linear)
  4. output = compete(h_normed) * silu(W_out @ h_normed)  ← Bounded output

But recurrent path still goes through unbounded log_h!
```

## Key Insight

The stable architectures (log_3, log_4, log_5) likely apply normalization or selectivity **to the recurrent state itself**, not just the output.

## Recommended Fixes

### Fix 1: RMSNorm the Hidden State Before Recurrence (Simplest)

Modify `LinearPolynomialGatedUpdate` (Level 6) and `LogPolyGatedUpdateKernel` (log_0):

```cuda
// BEFORE (unstable):
float h_new = (1.0f - delta) * h_p + delta * candidate;
h_out[idx] = static_cast<T>(h_new);

// AFTER (stable):
float h_new = (1.0f - delta) * h_p + delta * candidate;
// Apply per-batch RMSNorm to bound hidden state magnitude
// This requires a two-pass approach or approximate single-pass
float h_normed = h_new / (rms_estimate + 1e-6f);  // See implementation below
h_out[idx] = static_cast<T>(h_normed);
```

**Implementation**: This needs a new kernel that:
1. Computes RMS across the hidden dimension for each batch element
2. Normalizes h_new before storing

### Fix 2: Selective Gating in the Recurrent Path (Recommended)

Replace delta-based gating with compete-based gating:

```cuda
// BEFORE (delta gating):
float h_new = (1.0f - delta) * h_prev + delta * candidate;

// AFTER (compete gating):
// Compute compete weights from candidate magnitudes
float compete = softmax_weight(candidate, group);  // Within-group softmax
float h_new = compete * candidate + (1.0f - compete) * h_prev;
// Or equivalently: weighted blend where weights sum to 1
```

This ensures the recurrent update is always a convex combination bounded by softmax weights.

### Fix 3: Pre-Activation Normalization (Alternative)

Bound `v` before the polynomial to prevent |v|^α explosion:

```cuda
// BEFORE:
float abs_v = fabsf(v);
abs_v = fminf(abs_v, 10.0f);  // Hard clamp
float candidate = sign_v * powf(abs_v + 1e-6f, alpha);

// AFTER:
float v_squashed = v / (1.0f + fabsf(v));  // Maps to (-1, 1)
float candidate = sign_v * powf(fabsf(v_squashed) + 1e-6f, alpha);
// Now |candidate| ≤ 1 always
```

## Specific CUDA Modifications

### For Level 6 (`linear_polynomial_gpu.cu.cc`)

**Option A: Add Hidden State Normalization**

Create a new kernel `LinearPolynomialRMSNorm`:
```cuda
template<typename T>
__global__ void LinearPolynomialRMSNorm(
    const int batch_size,
    const int dim,
    T* __restrict__ h,           // In-place normalization
    T* __restrict__ rms_cache) { // Optional cache for backward

    extern __shared__ float smem[];
    const int b = blockIdx.x;
    const int base = b * dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float val = static_cast<float>(h[base + d]);
        sum_sq += val * val;
    }

    // Reduce
    smem[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(smem[0] / dim + 1e-6f);
    if (rms_cache && threadIdx.x == 0) rms_cache[b] = static_cast<T>(rms);

    // Normalize
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        h[base + d] = static_cast<T>(static_cast<float>(h[base + d]) / rms);
    }
}
```

Call this AFTER `LinearPolynomialGatedUpdate` and BEFORE the output computation:
```cpp
// In LinearPolynomialForward::Run, after the update kernel:
LinearPolynomialGatedUpdate<T><<<...>>>(... h_t ...);

// ADD: Normalize hidden state
LinearPolynomialRMSNorm<T><<<batch_size_, block_size, smem_size, stream_>>>(
    batch_size_, dim_, h_t, rms_cache_t);

// Then continue with output computation
blas<T>::gemm(...);  // w_out_h = h_t @ W_out.T
LinearPolynomialOutput<T><<<...>>>(... h_t ...);
```

**Option B: Modify the Update Kernel Directly**

Change `LinearPolynomialGatedUpdate` to squash the candidate:
```cuda
// Line 106-108 in LinearPolynomialGatedUpdate:
// BEFORE:
float candidate = sign_v * powf(abs_v + 1e-6f, alpha);
candidate = fmaxf(fminf(candidate, 10.0f), -10.0f);

// AFTER:
float v_squashed = abs_v / (1.0f + abs_v);  // Squash to (0, 1)
float candidate = sign_v * powf(v_squashed + 1e-6f, alpha);
// No additional clamp needed since v_squashed ∈ (0, 1) and alpha ≥ 1
// means candidate ∈ (-1, 1)
```

### For log_0 (`logspace_polynomial_gpu.cu.cc`)

**Add Selective Output (like log_1)**

The simplest fix: copy the `LogSelectiveOutput` kernel from `logspace_selective_gpu.cu.cc` and integrate it.

In `LogPolyElmanForward::Run`:
```cpp
// AFTER LogPolyGatedUpdateKernel:
LogPolyGatedUpdateKernel<T><<<...>>>(...);

// CHANGE: Instead of just LogSpaceRMSNormKernel -> h_linear
// 1. First apply RMSNorm
LogSpaceRMSNormKernel<T><<<...>>>(log_h_t, sign_h_t, log_gamma, h_linear + t*BD, ...);

// 2. Compute W_out @ h_linear
blas<T>::gemm(... W_out, h_linear + t*BD, w_out_h ...);

// 3. ADD: Apply selective output (compete × silu)
dim3 grid(batch_size_, n_groups_);
LogSelectiveOutput<T><<<grid, block_size, smem_size, stream_>>>(
    batch_size_, dim_, n_groups_, group_size,
    h_linear + t*BD, w_out_h, output + t*BD, compete_cache + t*BD);
```

This requires:
1. Adding `n_groups` parameter to the class
2. Adding `compete_cache` buffer
3. Copying `LogSelectiveOutput` and `LogSelectiveOutputBackward` kernels

## Testing Strategy

1. **Unit test**: Run single forward/backward pass, check gradient magnitudes
2. **Stability test**: Train for 100 steps, monitor grad norm (should stay < 10)
3. **Convergence test**: Train for 1000 steps, loss should decrease

## Expected Outcomes

| Architecture | Before Fix | After Fix |
|-------------|------------|-----------|
| Level 6 | NaN, grad→∞ | Grad norm ~2-3, stable |
| log_0 | Grad 10⁹→10¹² | Grad norm ~100-200 (marginal) |
| log_0 + selectivity | N/A | Grad norm ~2-3 (stable like log_3) |

## Priority Order

1. **Highest priority**: Add selective output to log_0 (should make it like log_1+)
2. **Second priority**: Add RMSNorm to Level 6 hidden state
3. **Third priority**: Squash pre-activation v for Level 6

The selective output fix for log_0 is likely the simplest since the kernels already exist in log_1.
