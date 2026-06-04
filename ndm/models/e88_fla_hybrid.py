"""
E88: FLA-GDN Hybrid with Nonlinear Matrix State

Combines FLA-GatedDeltaNet's proven design elements with E75's nonlinear
matrix state mechanism. Key adaptations from FLA-GDN:

1. **Mamba2-style exponential decay** (replaces sigmoid beta):
   - A_log: learned log eigenvalues (like Mamba2)
   - dt_bias: learned time-step bias
   - a_proj: input-dependent alpha gate
   - Decay: g = -exp(A_log) * softplus(a_proj(x) + dt_bias)

2. **Output gating** (FLA-GDN style):
   - g_proj: output gate projection
   - Gated output: o * sigmoid(g)

3. **Short convolutions** (FLA-GDN style):
   - Depthwise conv on k, v, q after projection
   - bias=False (matching FLA default)
   - SiLU activation fused

4. **L2-normalized Q and K** in retrieval

Architecture per head h:
    # Projections with conv+silu
    k_h = silu(conv(W_k @ x))   # [n_state]
    v_h = silu(conv(W_v @ x))   # [n_state]
    q_h = silu(conv(W_q @ x))   # [n_state]

    # Mamba2-style exponential decay
    g_h = -exp(A_log) * softplus(a_proj(x) + dt_bias)  # scalar per head

    # L2 normalize k and q
    k_norm = k_h / ||k_h||
    q_norm = q_h / ||q_h||

    # Matrix state update (NONLINEAR - key differentiator from FLA-GDN)
    r = S_h @ k_norm            # retrieve
    delta = v_h - r             # delta
    S_h = tanh(exp(g_h) * S_h + outer(delta, k_norm))  # exp decay + nonlinear

    # Output with gating
    Sq_h = S_h @ q_norm
    out_h = Sq_h * sigmoid(g_out_h)  # gated output

Output: out_proj(concat(out_0, ..., out_{H-1}))
"""

import math
import torch

# Try to import Triton ops for fused element-wise operations
try:
    from .triton_ops import mamba2_decay as triton_mamba2_decay, l2_norm as triton_l2_norm
    TRITON_OPS_AVAILABLE = True
except ImportError:
    TRITON_OPS_AVAILABLE = False
    triton_mamba2_decay = None
    triton_l2_norm = None

# Global flag to enable Triton ops (can be disabled for debugging)
USE_TRITON_OPS = True
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Try to import FLA's per-head normalization modules
try:
    from fla.modules import FusedRMSNormGated, RMSNorm as FLARMSNorm
    FLA_FUSED_NORM_AVAILABLE = True
except ImportError:
    FLA_FUSED_NORM_AVAILABLE = False
    FusedRMSNormGated = None
    FLARMSNorm = None

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    # E88 native CUDA kernel
    E88_NATIVE_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_forward')
    # E88 cuBLAS tensor core backward kernel
    E88_CUBLAS_BACKWARD_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_fla_hybrid_backward_cublas')
    # E88 fused projection kernel (GEMM + conv + silu + L2 norm + decay)
    E88_FUSED_PROJECTION_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_fused_projection')
    # E88 fused gate forward kernel (fuses Sq * silu(g) into forward pass)
    E88_FUSED_GATE_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_fused_gate_forward')
    # Legacy E75 kernels for backwards compatibility
    E88_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e75_multihead_forward')
    E88_PRECOMPUTED_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e75_multihead_precomputed_forward')
    # E88 optimized kernels with [B, T, H, dim] layout (no transpose overhead)
    E88_OPTIMIZED_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_fused_forward')
    E88_WARP_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_warp_optimized_forward')
    E88_COALESCED_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_coalesced_forward')
    # E88 register-owned backward kernel (5-6x faster for n_state <= 32, head_v_dim <= 32)
    E88_REGISTER_OWNED_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_register_owned_backward')
except ImportError:
    E88_NATIVE_CUDA_AVAILABLE = False
    E88_CUBLAS_BACKWARD_AVAILABLE = False
    E88_FUSED_PROJECTION_AVAILABLE = False
    E88_FUSED_GATE_AVAILABLE = False
    E88_CUDA_AVAILABLE = False
    E88_PRECOMPUTED_CUDA_AVAILABLE = False
    E88_OPTIMIZED_AVAILABLE = False
    E88_WARP_AVAILABLE = False
    E88_COALESCED_AVAILABLE = False
    E88_REGISTER_OWNED_AVAILABLE = False

# Global flag to enable cuBLAS backward kernel (can be toggled at runtime)
USE_CUBLAS_BACKWARD = False

# Global flag to enable reduced-sync backward kernel (fewer __syncthreads())
# Default True: ~15% faster backward (21ms vs 24.7ms at B=32, T=512, dim=1792)
USE_REDUCED_SYNC_BACKWARD = True

# Global flag to enable fused projection kernel (can be toggled at runtime)
USE_FUSED_PROJECTION = True

# Global flag to enable fused gate forward kernel (fuses output gating into forward)
# Saves ~4.5% forward time by eliminating separate gating kernel
USE_FUSED_GATE = True

# Global flag to enable optimized [B, T, H, dim] layout kernels (no transpose overhead)
# Auto-selects best kernel: warp for n_state<=32, coalesced for n_state>32 or long sequences
# Enable optimized kernels with [B, T, H, dim] layout (no transpose overhead)
USE_OPTIMIZED_KERNELS = True

# Global flag to enable in-kernel L2 normalization of k and q
# When True, the optimized forward/backward kernels normalize k/q in shared memory,
# eliminating separate L2 norm kernel launches (saves ~3-5% end-to-end)
USE_FUSED_L2_NORM = True

# Backwards compat
E75MH_CUDA_AVAILABLE = E88_CUDA_AVAILABLE
E75MH_PRECOMPUTED_CUDA_AVAILABLE = E88_PRECOMPUTED_CUDA_AVAILABLE


class E75MultiHeadCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E75 Multi-Head autograd function with gradient checkpointing."""

    @staticmethod
    def forward(ctx, training, x, S0, W_k, W_v, W_q, W_beta, b_beta, n_heads):
        results = hasty_pytorch_lib.e75_multihead_forward(
            training, x, S0, W_k, W_v, W_q, W_beta, b_beta, n_heads
        )
        # results = [output, S, k_cache, v_cache, q_cache, beta_cache, S_cache]
        # S_cache contains both S_checkpoints and Sq_cache concatenated
        output = results[0]
        S = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        beta_cache = results[5]
        S_cache = results[6]  # Combined checkpoints + Sq_cache

        ctx.save_for_backward(
            x, S_cache,
            k_cache, v_cache, q_cache, beta_cache,
            W_k, W_v, W_q, W_beta
        )
        ctx.n_heads = n_heads
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_cache,
         k_cache, v_cache, q_cache, beta_cache,
         W_k, W_v, W_q, W_beta) = ctx.saved_tensors
        n_heads = ctx.n_heads

        # Split S_cache into S_checkpoints and Sq_cache
        # S_cache layout: [checkpoints_flat || sq_cache_flat]
        T, B, _ = x.shape
        n_state = k_cache.size(3)
        checkpoint_interval = 16  # Must match E88_CHECKPOINT_INTERVAL in CUDA
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        checkpoints_size = num_checkpoints * B * n_heads * n_state * n_state
        sq_cache_size = T * B * n_heads * n_state

        S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, n_heads, n_state, n_state)
        Sq_cache = S_cache[checkpoints_size:].view(T, B, n_heads, n_state)

        grads = hasty_pytorch_lib.e75_multihead_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache, beta_cache,
            d_output.contiguous(),
            W_k, W_v, W_q, W_beta,
            n_heads
        )
        # grads = [dx, dW_k, dW_v, dW_q, dW_beta, db_beta]
        dx = grads[0]
        dW_k = grads[1]
        dW_v = grads[2]
        dW_q = grads[3]
        dW_beta = grads[4]
        db_beta = grads[5]

        # Return gradients for: training, x, S0, W_k, W_v, W_q, W_beta, b_beta, n_heads
        return None, dx, None, dW_k, dW_v, dW_q, dW_beta, db_beta, None


class E88FLAHybridCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E88 FLA Hybrid autograd function.

    E88 uses:
    - Rectangular state [n_state x head_v_dim]
    - Mamba2-style scalar exponential decay per head
    - L2-normalized k and q (done externally before calling this)
    - Output: Sq directly (FLA-GDN style gating applied in Python layer)
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads):
        """
        Args:
            training: bool
            k: [T, B, H, n_state] L2 normalized keys
            v: [T, B, H, head_v_dim] values
            q: [T, B, H, n_state] L2 normalized queries
            decay: [T, B, H] exponential decay factors
            S0: [B, H, n_state, head_v_dim] initial state
            n_heads: int
        """
        results = hasty_pytorch_lib.e88_fla_hybrid_forward(
            training, k, v, q, decay, S0, n_heads
        )
        # results = [S_final, output, S_cache]
        S_final = results[0]  # [B, H, n_state, head_v_dim]
        output = results[1]   # [T, B, H, head_v_dim]
        S_cache = results[2]  # Combined checkpoints + Sq_cache

        ctx.save_for_backward(k, v, q, decay, S_cache)
        ctx.n_heads = n_heads
        ctx.n_state = k.size(-1)
        ctx.head_v_dim = v.size(-1)
        return S_final, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, decay, S_cache = ctx.saved_tensors
        n_heads = ctx.n_heads
        n_state = ctx.n_state
        head_v_dim = ctx.head_v_dim

        T, B, H, _ = k.shape
        checkpoint_interval = 16  # Must match E88_CHECKPOINT_INTERVAL in CUDA
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1

        # Split S_cache into S_checkpoints and Sq_cache
        checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim
        sq_cache_size = T * B * H * head_v_dim

        S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, head_v_dim)
        Sq_cache = S_cache[checkpoints_size:checkpoints_size + sq_cache_size].view(T, B, H, head_v_dim)

        # Choose backward kernel
        if USE_CUBLAS_BACKWARD and E88_CUBLAS_BACKWARD_AVAILABLE:
            grads = hasty_pytorch_lib.e88_fla_hybrid_backward_cublas(
                k, v, q, decay,
                S_checkpoints.view(-1),  # Flatten for cuBLAS kernel
                d_output.contiguous(),
                n_heads,
                checkpoint_interval
            )
        elif USE_REDUCED_SYNC_BACKWARD:
            # Allocate segment cache for reduced sync kernel
            cache_entry_size = n_state * head_v_dim + n_state + head_v_dim + 1
            segment_cache = torch.empty(
                B * H * checkpoint_interval * cache_entry_size,
                dtype=k.dtype, device=k.device
            )
            grads = hasty_pytorch_lib.e88_fla_hybrid_backward_reduced_sync(
                k, v, q, decay,
                S_checkpoints, Sq_cache,
                segment_cache,
                d_output.contiguous(),
                n_heads,
                checkpoint_interval
            )
        else:
            grads = hasty_pytorch_lib.e88_fla_hybrid_backward(
                k, v, q, decay,
                S_checkpoints, Sq_cache,
                d_output.contiguous(),
                n_heads
            )
        # grads = [d_k, d_v, d_q, d_decay]
        d_k = grads[0]
        d_v = grads[1]
        d_q = grads[2]
        d_decay = grads[3]

        # Return gradients for: training, k, v, q, decay, S0, n_heads
        return None, d_k, d_v, d_q, d_decay, None, None


class E88WithBetaCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E88 with write gate (beta) support.

    Beta gates how much delta gets written to memory:
    S = tanh(decay * S + beta * outer(delta, k))
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, beta, S0, n_heads):
        """
        Args:
            training: bool
            k: [T, B, H, n_state] L2 normalized keys
            v: [T, B, H, head_v_dim] values
            q: [T, B, H, n_state] L2 normalized queries
            decay: [T, B, H] exponential decay factors
            beta: [T, B, H] write gate (0-1)
            S0: [B, H, n_state, head_v_dim] initial state
            n_heads: int
        """
        results = hasty_pytorch_lib.e88_fla_hybrid_forward_with_beta(
            training, k, v, q, decay, beta, S0, n_heads
        )
        # results = [S_final, output, S_cache]
        S_final = results[0]  # [B, H, n_state, head_v_dim]
        output = results[1]   # [T, B, H, head_v_dim]
        S_cache = results[2]  # Combined checkpoints + Sq_cache

        ctx.save_for_backward(k, v, q, decay, beta, S_cache)
        ctx.n_heads = n_heads
        ctx.n_state = k.size(-1)
        ctx.head_v_dim = v.size(-1)
        return S_final, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, decay, beta, S_cache = ctx.saved_tensors
        n_heads = ctx.n_heads
        n_state = ctx.n_state
        head_v_dim = ctx.head_v_dim

        T, B, H, _ = k.shape
        checkpoint_interval = 16  # Must match E88_CHECKPOINT_INTERVAL in CUDA
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1

        # Split S_cache into S_checkpoints and Sq_cache
        checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim
        sq_cache_size = T * B * H * head_v_dim

        S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, head_v_dim)
        Sq_cache = S_cache[checkpoints_size:checkpoints_size + sq_cache_size].view(T, B, H, head_v_dim)

        # Use backward with beta
        grads = hasty_pytorch_lib.e88_fla_hybrid_backward_with_beta(
            k, v, q, decay, beta,
            S_checkpoints, Sq_cache,
            d_output.contiguous(),
            n_heads
        )
        # grads = [d_k, d_v, d_q, d_decay, d_beta]
        d_k = grads[0]
        d_v = grads[1]
        d_q = grads[2]
        d_decay = grads[3]
        d_beta = grads[4]

        # Return gradients for: training, k, v, q, decay, beta, S0, n_heads
        return None, d_k, d_v, d_q, d_decay, d_beta, None, None


# Helper for fused gate backward
def _compute_gate_gradients(d_output, Sq, g):
    """Compute d_g and d_Sq from fused gate backward.

    output = Sq * silu(g) = Sq * g * sigmoid(g)
    d_g = d_output * Sq * silu'(g) = d_output * Sq * sigmoid(g) * (1 + g * (1 - sigmoid(g)))
    d_Sq = d_output * silu(g) = d_output * g * sigmoid(g)
    """
    sig_g = torch.sigmoid(g)
    silu_grad = sig_g * (1.0 + g * (1.0 - sig_g))
    d_g = d_output * Sq * silu_grad
    d_Sq = d_output * g * sig_g
    return d_g, d_Sq


class E88FusedGateCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E88 with fused output gating.

    Fuses Sq * silu(g) directly into the forward kernel, saving memory bandwidth.
    Backward computes gradients for both the recurrence (k, v, q, decay) and gate (g).
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, g, S0, n_heads):
        """
        Args:
            training: bool
            k: [T, B, H, n_state] L2 normalized keys
            v: [T, B, H, head_v_dim] values
            q: [T, B, H, n_state] L2 normalized queries
            decay: [T, B, H] exponential decay factors
            g: [T, B, H, head_v_dim] gate projections (pre-activation)
            S0: [B, H, n_state, head_v_dim] initial state
            n_heads: int
        """
        results = hasty_pytorch_lib.e88_fused_gate_forward(
            training, k, v, q, decay, g, S0, n_heads
        )
        # results = [S_final, output, S_checkpoints, Sq_cache]
        S_final = results[0]      # [B, H, n_state, head_v_dim]
        output = results[1]       # [T, B, H, head_v_dim] - already gated (Sq * silu(g))
        S_checkpoints = results[2]  # Checkpoints for backward
        Sq_cache = results[3]     # Un-gated Sq for backward

        ctx.save_for_backward(k, v, q, decay, g, S_checkpoints, Sq_cache)
        ctx.n_heads = n_heads
        ctx.n_state = k.size(-1)
        ctx.head_v_dim = v.size(-1)
        return S_final, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, decay, g, S_checkpoints, Sq_cache = ctx.saved_tensors
        n_heads = ctx.n_heads
        n_state = ctx.n_state
        head_v_dim = ctx.head_v_dim

        T, B, H, _ = k.shape
        checkpoint_interval = 16

        # Compute d_g and d_Sq using compiled fused kernel (~80% faster)
        Sq = Sq_cache.view(T, B, H, head_v_dim)
        d_g, d_Sq = _compute_gate_gradients(d_output, Sq, g)

        # Reshape S_checkpoints for backward kernel
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        S_checkpoints_view = S_checkpoints.view(num_checkpoints, B, H, n_state, head_v_dim)

        # Call backward kernel with d_Sq (gradients w.r.t. un-gated output)
        if USE_REDUCED_SYNC_BACKWARD:
            cache_entry_size = n_state * head_v_dim + n_state + head_v_dim + 1
            segment_cache = torch.empty(
                B * H * checkpoint_interval * cache_entry_size,
                dtype=k.dtype, device=k.device
            )
            grads = hasty_pytorch_lib.e88_fla_hybrid_backward_reduced_sync(
                k, v, q, decay,
                S_checkpoints_view, Sq,
                segment_cache,
                d_Sq.contiguous(),
                n_heads,
                checkpoint_interval
            )
        else:
            grads = hasty_pytorch_lib.e88_fla_hybrid_backward(
                k, v, q, decay,
                S_checkpoints_view, Sq,
                d_Sq.contiguous(),
                n_heads
            )

        d_k, d_v, d_q, d_decay = grads[0], grads[1], grads[2], grads[3]

        # Return gradients for: training, k, v, q, decay, g, S0, n_heads
        return None, d_k, d_v, d_q, d_decay, d_g, None, None


class E88OptimizedCUDAFunction(torch.autograd.Function):
    """Optimized CUDA kernel with [B, T, H, dim] layout (no transpose overhead).

    Auto-selects the best kernel based on configuration:
    - Warp-optimized kernel: Best for n_state <= 32 (up to 2.8x faster)
    - Coalesced kernel: Best for n_state > 32 or long sequences (up to 1.4x faster)

    Performance vs original fused kernel (with transpose):
    - Standard config (n=32, T=512): 1.17x faster
    - Small n_state (n=16): 2.84x faster
    - Large n_state (n=48): 1.44x faster
    - Long sequences (T=2048): 1.17x faster
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, g, S0, n_heads, apply_gate=True, normalize_kq=False, checkpoint_interval=16):
        """
        Args:
            training: bool
            k: [B, T, H, n_state] keys (L2 normalized, or raw if normalize_kq=True)
            v: [B, T, H, head_v_dim] values
            q: [B, T, H, n_state] queries (L2 normalized, or raw if normalize_kq=True)
            decay: [B, T, H] exponential decay factors
            g: [B, T, H, head_v_dim] gate values (can be None if apply_gate=False)
            S0: [B, H, n_state, head_v_dim] initial state
            n_heads: int
            apply_gate: bool
            normalize_kq: bool - if True, kernel normalizes k/q in shared memory (avoids separate L2 norm launches)
            checkpoint_interval: int - steps between state checkpoints (16=default, larger=less memory)
        """
        B, T, H, n_state = k.shape
        head_v_dim = v.size(-1)

        # Allocate output
        output = torch.empty(B, T, H, head_v_dim, device=k.device, dtype=k.dtype)

        # Allocate cache for checkpoints + Sq
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        cache_size = num_checkpoints * B * H * n_state * head_v_dim + B * T * H * head_v_dim
        S_cache = torch.empty(cache_size, device=k.device, dtype=k.dtype)

        # Handle empty gate tensor
        g_tensor = g if g is not None and apply_gate else torch.empty(0, device=k.device, dtype=k.dtype)

        # Select best kernel based on configuration
        # Warp: best for n_state <= 32
        # Coalesced: best for n_state > 32 or T > 1024
        use_coalesced = n_state > 32 or T > 1024

        if use_coalesced and E88_COALESCED_AVAILABLE:
            result = hasty_pytorch_lib.e88_coalesced_forward(
                training, k.contiguous(), v.contiguous(), q.contiguous(),
                decay.contiguous(), g_tensor.contiguous(),
                S0.contiguous(), output, S_cache, H, apply_gate, normalize_kq,
                checkpoint_interval
            )
        elif E88_WARP_AVAILABLE:
            result = hasty_pytorch_lib.e88_warp_optimized_forward(
                training, k.contiguous(), v.contiguous(), q.contiguous(),
                decay.contiguous(), g_tensor.contiguous(),
                S0.contiguous(), output, S_cache, H, apply_gate, normalize_kq,
                checkpoint_interval
            )
        elif E88_OPTIMIZED_AVAILABLE:
            result = hasty_pytorch_lib.e88_fused_forward(
                training, k.contiguous(), v.contiguous(), q.contiguous(),
                decay.contiguous(), g_tensor.contiguous(),
                S0.contiguous(), output, S_cache, H, apply_gate,
                checkpoint_interval
            )
        else:
            raise RuntimeError("No optimized E88 kernel available")

        # Extract final state from kernel return value
        S_final = result[0]  # C++ kernel returns {S_updated, output}

        ctx.save_for_backward(k, v, q, decay, g_tensor if apply_gate else torch.empty(0, device=k.device, dtype=k.dtype), S_cache)
        ctx.n_heads = n_heads
        ctx.n_state = n_state
        ctx.head_v_dim = head_v_dim
        ctx.apply_gate = apply_gate
        ctx.normalize_kq = normalize_kq
        ctx.checkpoint_interval = checkpoint_interval

        return S_final, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, decay, g, S_cache = ctx.saved_tensors
        n_heads = ctx.n_heads
        n_state = ctx.n_state
        head_v_dim = ctx.head_v_dim
        apply_gate = ctx.apply_gate
        normalize_kq = ctx.normalize_kq
        checkpoint_interval = ctx.checkpoint_interval

        B, T, H, _ = k.shape

        # Allocate output gradients
        d_k = torch.empty_like(k)
        d_v = torch.empty_like(v)
        d_q = torch.empty_like(q)
        d_decay = torch.empty_like(decay)
        d_g = torch.empty_like(g) if apply_gate and g.numel() > 0 else torch.empty(0, device=k.device, dtype=k.dtype)

        # Allocate segment cache for backward
        cache_entry_size = n_state * head_v_dim + n_state + head_v_dim + 1
        segment_cache = torch.empty(
            B * H * checkpoint_interval * cache_entry_size,
            dtype=k.dtype, device=k.device
        )

        # Use register_owned_backward for supported sizes (n_state <= 32, head_v_dim <= 32)
        # Register-owned is 5-6x faster: 1.5ms vs 10ms for 32x32
        g_tensor = g if apply_gate and g.numel() > 0 else torch.empty(0, device=k.device, dtype=k.dtype)
        d_g_tensor = d_g if apply_gate and g.numel() > 0 else torch.empty(0, device=k.device, dtype=k.dtype)
        has_gate = apply_gate and g.numel() > 0

        # register_owned_backward is 1.5-1.6x faster than fused_backward for n_state<=32
        if E88_REGISTER_OWNED_AVAILABLE and n_state <= 32 and head_v_dim <= 32:
            hasty_pytorch_lib.e88_register_owned_backward(
                k, v, q, decay, g_tensor,
                S_cache, d_output.contiguous(),
                d_k, d_v, d_q, d_decay, d_g_tensor,
                segment_cache, n_heads, has_gate, normalize_kq,
                checkpoint_interval
            )
        else:
            # Fall back to fused_backward for larger sizes (no in-kernel norm yet)
            hasty_pytorch_lib.e88_fused_backward(
                k, v, q, decay, g_tensor,
                S_cache, d_output.contiguous(),
                d_k, d_v, d_q, d_decay, d_g_tensor,
                segment_cache, n_heads, has_gate,
                checkpoint_interval
            )

        # Return gradients for: training, k, v, q, decay, g, S0, n_heads, apply_gate, normalize_kq, checkpoint_interval
        d_g_out = d_g if apply_gate and g.numel() > 0 else None
        return None, d_k, d_v, d_q, d_decay, d_g_out, None, None, None, None, None


class E75MultiHeadPrecomputedCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E75 Multi-Head with pre-computed k, v, q, beta.

    Used for post-projection convolution mode (FLA-GDN style).
    k, v, q have already had conv+silu applied.
    beta has already had sigmoid applied.
    """

    @staticmethod
    def forward(ctx, training, k, v, q, beta, S0, n_heads):
        """
        Args:
            training: bool
            k: [T, B, H, n_state] pre-computed (with conv+silu)
            v: [T, B, H, n_state] pre-computed (with conv+silu)
            q: [T, B, H, n_state] pre-computed (with conv+silu)
            beta: [T, B, H, n_state] pre-computed (with sigmoid)
            S0: [B, H, n_state, n_state] initial state
            n_heads: int
        """
        results = hasty_pytorch_lib.e75_multihead_precomputed_forward(
            training, k, v, q, beta, S0, n_heads
        )
        # results = [output, S, S_cache]
        output = results[0]  # [T, B, H, n_state]
        S = results[1]       # [B, H, n_state, n_state]
        S_cache = results[2] # Combined checkpoints + Sq_cache

        ctx.save_for_backward(k, v, q, beta, S_cache)
        ctx.n_heads = n_heads
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, beta, S_cache = ctx.saved_tensors
        n_heads = ctx.n_heads

        # Split S_cache into S_checkpoints and Sq_cache
        T, B, H, n_state = k.shape
        checkpoint_interval = 16  # Must match E88_CHECKPOINT_INTERVAL in CUDA
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        checkpoints_size = num_checkpoints * B * H * n_state * n_state
        sq_cache_size = T * B * H * n_state

        S_checkpoints = S_cache[:checkpoints_size].view(num_checkpoints, B, H, n_state, n_state)
        Sq_cache = S_cache[checkpoints_size:].view(T, B, H, n_state)

        grads = hasty_pytorch_lib.e75_multihead_precomputed_backward(
            k, v, q, beta,
            S_checkpoints, Sq_cache,
            d_output.contiguous(),
            n_heads
        )
        # grads = [d_k, d_v, d_q, d_beta]
        d_k = grads[0]
        d_v = grads[1]
        d_q = grads[2]
        d_beta = grads[3]

        # Return gradients for: training, k, v, q, beta, S0, n_heads
        return None, d_k, d_v, d_q, d_beta, None, None


class E75MultiHeadCell(nn.Module):
    """
    E75 Multi-Head Gated Delta Matrix cell.

    H independent heads, each with its own n_state x n_state matrix state.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 4,
        init_beta_bias: float = 2.0,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        self.use_cuda = use_cuda and E75MH_CUDA_AVAILABLE

        # Fused projections: [H * n_state, dim] for efficiency
        # Each head gets its own slice of the projection
        self.W_k = nn.Parameter(torch.empty(n_heads * n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_heads * n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_heads * n_state, dim))
        self.W_beta = nn.Parameter(torch.empty(n_heads * n_state, dim))

        # Per-head beta biases: [H, n_state]
        self.b_beta = nn.Parameter(torch.full((n_heads, n_state), init_beta_bias))

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        H = self.n_heads

        # Initialize each head's projections with xavier
        for h in range(H):
            start = h * n
            end = (h + 1) * n
            nn.init.xavier_uniform_(self.W_k[start:end])
            nn.init.xavier_uniform_(self.W_v[start:end])
            nn.init.xavier_uniform_(self.W_q[start:end])
            nn.init.xavier_uniform_(self.W_beta[start:end])

    def forward(
        self,
        x: torch.Tensor,
        S_list: Optional[List[torch.Tensor]] = None,
        use_cuda: Optional[bool] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [T, B, dim] input sequence
            S_list: list of H tensors, each [B, n_state, n_state] initial matrix states
            use_cuda: Override instance setting for CUDA usage

        Returns:
            output: [T, B, H * n_state] concatenated outputs from all heads
            S_list: list of H final matrix states [B, n_state, n_state]
        """
        T, B, D = x.shape
        n = self.n_state
        H = self.n_heads

        # Initialize states if not provided
        if S_list is None:
            S_list = [torch.zeros(B, n, n, device=x.device, dtype=x.dtype) for _ in range(H)]

        _use_cuda = use_cuda if use_cuda is not None else self.use_cuda

        # Use CUDA kernel if available
        if _use_cuda and E75MH_CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
            # Stack S_list into single tensor: [B, H, n, n]
            S0 = torch.stack(S_list, dim=1)

            S_final, output = E75MultiHeadCUDAFunction.apply(
                self.training, x, S0,
                self.W_k, self.W_v, self.W_q, self.W_beta, self.b_beta,
                H
            )

            # Convert S_final [B, H, n, n] back to list
            S_list_out = [S_final[:, h] for h in range(H)]
            return output, S_list_out

        # PyTorch fallback
        # Project all inputs at once: [T*B, dim] @ [dim, H*n] -> [T*B, H*n]
        x_flat = x.reshape(T * B, D)

        # Compute all projections: [T, B, H, n]
        k_all = (x_flat @ self.W_k.T).reshape(T, B, H, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, B, H, n)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, H, n)

        # Beta with bias: [T, B, H, n]
        beta_proj = (x_flat @ self.W_beta.T).reshape(T, B, H, n)
        beta_all = torch.sigmoid(beta_proj + self.b_beta)  # Broadcasting [H, n] over [T, B, H, n]

        # Clone S_list for in-place updates
        S_list = [S.clone() for S in S_list]

        outputs = []
        for t in range(T):
            head_outputs = []

            for h in range(H):
                # Get projections for this head at this timestep: [B, n]
                k = k_all[t, :, h]
                v = v_all[t, :, h]
                q = q_all[t, :, h]
                beta = beta_all[t, :, h]  # [B, n]

                # Normalize k
                k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n]

                # Retrieve from memory: S @ k_norm -> [B, n]
                retrieved = torch.einsum('bij,bj->bi', S_list[h], k_norm)

                # Delta update with forget gate
                delta = v - retrieved  # [B, n]
                outer = torch.einsum('bi,bj->bij', delta, k_norm)  # [B, n, n]

                # Gated update: S = tanh(beta * S + outer)
                # beta: [B, n] -> [B, n, 1] for row-wise gating
                S_list[h] = torch.tanh(beta.unsqueeze(-1) * S_list[h] + outer)

                # Self-gating output: Sq * silu(Sq)
                Sq = torch.einsum('bij,bj->bi', S_list[h], q)  # [B, n]
                out_h = Sq * F.silu(Sq)  # [B, n]
                head_outputs.append(out_h)

            # Concatenate all head outputs: [B, H * n]
            out_t = torch.cat(head_outputs, dim=-1)
            outputs.append(out_t)

        # Stack outputs: [T, B, H * n]
        output = torch.stack(outputs, dim=0)
        return output, S_list


class E88FLAHybrid(nn.Module):
    """
    E88: FLA-GDN Hybrid with Nonlinear Matrix State.

    Combines FLA-GatedDeltaNet's proven design with E75's nonlinear matrix state.

    Key FLA-GDN elements:
    1. Mamba2-style exponential decay (A_log, dt_bias, a_proj)
    2. Output gating with g_proj
    3. Short convolutions on k, v, q (bias=False)
    4. L2-normalized Q and K

    Kept from E75:
    - Nonlinear matrix state: S = tanh(decay * S + outer(delta, k_norm))
    - Multi-head structure with independent states
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,  # E88 default (1.0 = square state, better for nonlinear recurrence)
        n_state: int = 32,
        n_heads: int = 8,  # More heads like FLA-GDN
        dropout: float = 0.0,
        d_conv: int = 4,
        use_cuda: bool = True,
        tie_kv: bool = False,  # When True and expansion=1.0, skip v_proj (v=k)
        # Ablation options
        use_conv: bool = False,  # E88 optimal: no short convolutions (unlike FLA-GDN)
        linear_state: bool = False,  # Set True to use linear state update (no tanh)
        use_gate: bool = False,  # E88 optimal: no output gating (gating hurts E88)
        use_write_gate: bool = False,  # Set True to gate the delta write (like FLA-GDN beta)
        raw_write: bool = False,  # Set True to ablate delta correction: write v instead of v - S@k
        use_split_edit: bool = False,  # E97: separate key-axis erase/read and value-axis write gates
        simple_decay: bool = False,  # Set True to use simple sigmoid decay instead of Mamba2
        decay_mode: str = 'mamba',  # mamba, simple, none, or constant
        use_value_residual: bool = False,  # Add direct D*v residual before output gating
        use_silu: bool = True,  # Set False to skip SiLU on projections
        use_l2_norm: bool = True,  # Set False to skip L2 normalization on k/q
        use_output_norm: bool = False,  # E88 optimal: no output RMSNorm
        head_mix: str = 'concat',  # Head mixing: 'concat', 'weighted_sum', 'per_head', 'input_weighted', 'sum'
        gate_activation: str = 'silu',  # Gate activation: 'silu' (FLA-GDN style, enables optimized kernels) or 'sigmoid'
        checkpoint_interval: int = 16,  # Steps between state checkpoints (larger = less memory, more recompute)
        projection_chunk_size: int = 0,  # Chunk size for projection recomputation (0=disabled). Saves ~5GB/layer at T=32K.
        use_triton: bool = False,  # Use Triton fwd+bwd kernels instead of CUDA register-owned (portable across NVIDIA/AMD ROCm)
        **kwargs
    ):
        super().__init__()

        # Validate n_state is multiple of 4 (CUDA kernel supports 4, 8, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 80, 96, 128)
        if n_state % 4 != 0:
            raise ValueError(f"n_state must be multiple of 4, got {n_state}")

        # Validate gate activation
        if gate_activation not in ['sigmoid', 'silu', 'swish']:
            raise ValueError(f"gate_activation must be 'sigmoid', 'silu', or 'swish', got {gate_activation}")
        if simple_decay:
            decay_mode = 'simple'
        if decay_mode not in ['mamba', 'simple', 'none', 'constant']:
            raise ValueError(f"decay_mode must be 'mamba', 'simple', 'none', or 'constant', got {decay_mode}")

        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        self.d_conv = d_conv
        self.expansion = expansion
        self.checkpoint_interval = checkpoint_interval
        self.projection_chunk_size = projection_chunk_size

        # Ablation flags
        self.use_conv = use_conv
        self.linear_state = linear_state
        self.use_gate = use_gate
        self.use_write_gate = use_write_gate
        self.raw_write = raw_write
        self.use_split_edit = use_split_edit
        self.use_value_residual = use_value_residual
        self.gate_activation = gate_activation
        self.decay_mode = decay_mode
        self.simple_decay = decay_mode == 'simple'
        self.use_silu = use_silu
        self.use_l2_norm = use_l2_norm
        self.use_output_norm = use_output_norm
        self.head_mix = head_mix
        self.use_triton = use_triton

        # Key and query dimensions (like FLA-GDN: key_dim = num_heads * head_k_dim)
        self.key_dim = n_heads * n_state
        # Value dimension with expansion (like FLA-GDN: value_dim = expand_v * key_dim)
        self.value_dim = int(n_heads * n_state * expansion)
        self.head_v_dim = self.value_dim // n_heads

        # === Skip v_proj when expansion=1.0 and tie_kv=True ===
        # When expansion=1.0, k and v have same dimension, so we can tie them
        self.tie_kv = tie_kv and (expansion == 1.0)
        if self.tie_kv:
            assert self.head_v_dim == n_state, "tie_kv requires expansion=1.0"

        # === Projections (FLA-GDN style) ===
        # Fused QKV projection for better GEMM efficiency
        # Output layout: [q (key_dim), k (key_dim), v (value_dim)]
        self.fuse_qkv = True  # Enable fused projection
        if self.fuse_qkv and not self.tie_kv:
            self.qkv_proj = nn.Linear(dim, 2 * self.key_dim + self.value_dim, bias=False)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
            self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
            # Skip v_proj if tied (v will use k_proj output)
            if not self.tie_kv:
                self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
            else:
                self.v_proj = None  # v = k when tied

        # === Decay parameters ===
        self.a_proj = None
        self.A_log = None
        self.dt_bias = None
        self.beta_proj = None
        self.decay_logit = None
        if self.decay_mode == 'mamba':
            # Mamba2-style exponential decay
            # a_proj: input-dependent alpha (maps to num_heads scalars)
            self.a_proj = nn.Linear(dim, n_heads, bias=False)

            # A_log: learned log eigenvalues (Mamba2 style)
            A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True

            # dt_bias: learned time-step bias (Mamba2 style initialization)
            dt_min, dt_max = 0.001, 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True
        elif self.decay_mode == 'simple':
            # Simple sigmoid decay (ablation)
            self.beta_proj = nn.Linear(dim, n_heads, bias=False)
        elif self.decay_mode == 'constant':
            # Learned per-head constant decay. Init near 0.98 so it starts in
            # the same high-retention regime as the input-dependent decay.
            self.decay_logit = nn.Parameter(torch.full((n_heads,), 4.0, dtype=torch.float32))
            self.decay_logit._no_weight_decay = True

        # === Short convolutions (FLA-GDN style, bias=False) ===
        if use_conv and d_conv > 1:
            self.q_conv = nn.Conv1d(
                self.key_dim, self.key_dim, d_conv,
                padding=d_conv - 1, groups=self.key_dim, bias=False
            )
            self.k_conv = nn.Conv1d(
                self.key_dim, self.key_dim, d_conv,
                padding=d_conv - 1, groups=self.key_dim, bias=False
            )
            # Skip v_conv if tied (v uses k after conv+silu)
            if not self.tie_kv:
                self.v_conv = nn.Conv1d(
                    self.value_dim, self.value_dim, d_conv,
                    padding=d_conv - 1, groups=self.value_dim, bias=False
                )
            else:
                self.v_conv = None
        else:
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None

        # === Output gating (FLA-GDN style) ===
        if use_gate:
            self.g_proj = nn.Linear(dim, self.value_dim, bias=False)
        else:
            self.g_proj = None

        if use_value_residual:
            self.value_residual = nn.Parameter(torch.ones(n_heads, self.head_v_dim))
        else:
            self.value_residual = None

        # === Write gating (FLA-GDN beta style) ===
        # Gates the delta before writing to memory: S += beta * outer(delta, k)
        if use_write_gate:
            self.write_gate_proj = nn.Linear(dim, n_heads, bias=False)
        else:
            self.write_gate_proj = None

        # === Split edit gates (E97) ===
        # GDN-2-inspired decoupling: read/erase uses b_t * k_t while write
        # target uses w_t * v_t. The outer-product key remains k_t.
        if use_split_edit:
            self.erase_gate_proj = nn.Linear(dim, self.key_dim, bias=False)
            self.value_write_gate_proj = nn.Linear(dim, self.value_dim, bias=False)
        else:
            self.erase_gate_proj = None
            self.value_write_gate_proj = None

        # === Output projection (depends on head_mix strategy) ===
        if head_mix == 'concat':
            # Standard: concat all heads, project to dim
            self.o_proj = nn.Linear(self.value_dim, dim, bias=False)
        elif head_mix == 'weighted_sum':
            # Learnable scalar per head, then project from head_v_dim to dim
            self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
            self.o_proj = nn.Linear(self.head_v_dim, dim, bias=False)
        elif head_mix == 'per_head':
            # Each head has its own projection to dim, then sum
            self.head_projs = nn.ModuleList([
                nn.Linear(self.head_v_dim, dim, bias=False) for _ in range(n_heads)
            ])
            self.o_proj = None  # Not used
        elif head_mix == 'input_weighted':
            # Input-dependent head weighting
            self.head_attn = nn.Linear(dim, n_heads, bias=False)
            self.o_proj = nn.Linear(self.head_v_dim, dim, bias=False)
        elif head_mix == 'sum':
            # Direct sum of heads (requires head_v_dim projectable to dim)
            self.o_proj = nn.Linear(self.head_v_dim, dim, bias=False)
        else:
            raise ValueError(f"Unknown head_mix: {head_mix}")

        # === Output normalization (per-head RMSNorm like FLA-GDN) ===
        # NOTE: Per-head norm HURTS E88 in practice (loss 1.53 -> 3.0 at 480M params)
        # FLA-GDN benefits from per-head norm because it has linear dynamics that can grow unboundedly
        # E88's tanh-bounded state already prevents magnitude explosion, so norm interferes
        # Only enable when use_output_norm=True (default False)
        self.norm_eps = 1e-5
        self._use_fused_norm_gate = False
        if use_output_norm:
            if FLA_FUSED_NORM_AVAILABLE:
                if use_gate:
                    # FusedRMSNormGated: rms_norm(x) * weight * g * sigmoid(g) in one Triton kernel
                    self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=self.norm_eps)
                    self._use_fused_norm_gate = True
                else:
                    # FLA RMSNorm without gating (still efficient Triton kernel)
                    self.o_norm = FLARMSNorm(self.head_v_dim, eps=self.norm_eps)
            else:
                # Fallback: manual per-head RMSNorm weight
                self.o_norm_weight = nn.Parameter(torch.ones(self.head_v_dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # === Fused projection support ===
        # Check if we can use the fused projection kernel
        # Requires: qkv_proj (not separate), conv enabled, not simple_decay
        self._use_fused_projection = (
            E88_FUSED_PROJECTION_AVAILABLE and
            self.qkv_proj is not None and
            use_conv and
            self.decay_mode == 'mamba' and
            not tie_kv
        )

        self._init_weights()

    def _init_weights(self):
        if self.qkv_proj is not None:
            nn.init.xavier_uniform_(self.qkv_proj.weight)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            if self.v_proj is not None:
                nn.init.xavier_uniform_(self.v_proj.weight)
        if self.g_proj is not None:
            nn.init.xavier_uniform_(self.g_proj.weight)
        if self.o_proj is not None:
            nn.init.xavier_uniform_(self.o_proj.weight)
        if hasattr(self, 'head_projs'):
            for proj in self.head_projs:
                nn.init.xavier_uniform_(proj.weight)
        if hasattr(self, 'head_attn'):
            nn.init.xavier_uniform_(self.head_attn.weight)
        if self.a_proj is not None:
            nn.init.xavier_uniform_(self.a_proj.weight)
        if self.beta_proj is not None:
            nn.init.xavier_uniform_(self.beta_proj.weight)
        if self.value_residual is not None:
            self.value_residual.data.fill_(1)

    def _get_fused_projection_weights(self):
        """Construct combined W_qkva weight and conv weights for fused kernel.

        Returns:
            W_qkva: [2*key_dim + value_dim + n_heads, dim] combined projection weight
            conv_q: [key_dim, d_conv] query conv weights
            conv_k: [key_dim, d_conv] key conv weights
            conv_v: [value_dim, d_conv] value conv weights
        """
        # Combine qkv_proj with a_proj to get W_qkva
        # Layout: [q (key_dim), k (key_dim), v (value_dim), alpha (n_heads)]
        W_qkva = torch.cat([
            self.qkv_proj.weight,  # [2*key_dim + value_dim, dim]
            self.a_proj.weight,    # [n_heads, dim]
        ], dim=0)

        # Extract conv weights from Conv1d layers
        # Conv1d with groups=channels has weight shape [channels, 1, d_conv]
        # Kernel expects [channels, d_conv]
        conv_q = self.q_conv.weight.squeeze(1)  # [key_dim, d_conv]
        conv_k = self.k_conv.weight.squeeze(1)  # [key_dim, d_conv]
        conv_v = self.v_conv.weight.squeeze(1)  # [value_dim, d_conv]

        return W_qkva, conv_q, conv_k, conv_v

    def _compute_projections(self, x_chunk, input_dtype, use_fused_l2):
        """Compute k, v, q, decay, g projections for a time chunk.

        Args:
            x_chunk: [B, C, dim] input chunk
            input_dtype: dtype for output tensors
            use_fused_l2: if True, skip separate L2 norm (kernel handles it)

        Returns:
            k: [B, C, H, n_state]
            v: [B, C, H, head_v_dim]
            q: [B, C, H, n_state]
            decay: [B, C, H]
            g: [B, C, H, head_v_dim]
        """
        B, C, D = x_chunk.shape
        H = self.n_heads
        n = self.n_state

        # QKV projection
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(x_chunk)
            qkv_silu_in_kernel = self.use_triton and self.use_silu and not self.use_conv
            q = qkv[..., :self.key_dim]
            k = qkv[..., self.key_dim:2*self.key_dim]
            v = qkv[..., 2*self.key_dim:]
        else:
            qkv_silu_in_kernel = False
            q = self.q_proj(x_chunk)
            k = self.k_proj(x_chunk)
            v = None

        # Short convolutions (not supported with chunking, but handle for completeness)
        if self.use_conv and self.q_conv is not None:
            q = self.q_conv(q.transpose(1, 2))[:, :, :C].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :C].transpose(1, 2)

        # SiLU activation
        if self.use_silu and not qkv_silu_in_kernel:
            q = F.silu(q)
            k = F.silu(k)

        # Value projection
        if self.tie_kv:
            v = k
        elif v is None:
            v = self.v_proj(x_chunk)

        # Apply conv and SiLU to v (skip if tied — k already has SiLU)
        if not self.tie_kv:
            if self.use_conv and self.v_conv is not None:
                v = self.v_conv(v.transpose(1, 2))[:, :, :C].transpose(1, 2)
            if self.use_silu and not qkv_silu_in_kernel:
                v = F.silu(v)

        # Reshape for per-head processing
        q = q.view(B, C, H, n)
        k = k.view(B, C, H, n)
        v = v.view(B, C, H, self.head_v_dim)

        # L2 normalize k and q
        if use_fused_l2:
            k = k.to(input_dtype)
            q = q.to(input_dtype)
        elif self.use_l2_norm:
            if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and k.dtype == torch.bfloat16:
                k = triton_l2_norm(k.contiguous())
                q = triton_l2_norm(q.contiguous())
            else:
                k = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)
                q = (q / (q.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)
        else:
            k = k.to(input_dtype)
            q = q.to(input_dtype)

        # Compute decay
        if self.decay_mode == 'none':
            decay = torch.ones(B, C, H, device=x_chunk.device, dtype=x_chunk.dtype)
        elif self.decay_mode == 'constant':
            decay = torch.sigmoid(self.decay_logit).view(1, 1, H).expand(B, C, H).to(x_chunk.dtype).contiguous()
        elif self.decay_mode == 'simple':
            decay = torch.sigmoid(self.beta_proj(x_chunk))
        else:
            alpha = self.a_proj(x_chunk)
            if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and x_chunk.is_cuda and x_chunk.dtype == torch.bfloat16:
                decay = triton_mamba2_decay(alpha, self.A_log.float(), self.dt_bias.float())
            else:
                g_decay = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
                decay = g_decay.exp().to(x_chunk.dtype)

        # Gate projection
        g = self.g_proj(x_chunk).view(B, C, H, self.head_v_dim).to(input_dtype)

        return k, v.to(input_dtype), q, decay.to(input_dtype), g, qkv_silu_in_kernel

    def _process_chunk(self, x_chunk, S_prev, input_dtype, use_fused_l2):
        """Compute projections and run CUDA kernel for one time chunk.

        Args:
            x_chunk: [B, C, dim] input chunk
            S_prev: [B, H, n_state, head_v_dim] state from previous chunk
            input_dtype: dtype for computation
            use_fused_l2: if True, kernel normalizes k/q

        Returns:
            S_new: [B, H, n_state, head_v_dim] updated state
            output: [B, C, H, head_v_dim] output for this chunk
        """
        k, v, q, decay, g, qkv_silu_in_kernel = self._compute_projections(x_chunk, input_dtype, use_fused_l2)

        if self.use_triton:
            from ndm.triton.e88_triton_optimized import e88_triton_optimized_apply
            S_new, output = e88_triton_optimized_apply(
                self.training, k, v, q, decay, g, S_prev,
                self.n_heads, True, use_fused_l2, self.checkpoint_interval,
                apply_silu_qkv=qkv_silu_in_kernel,
                raw_write=self.raw_write,
                linear_state=self.linear_state,
            )
        else:
            S_new, output = E88OptimizedCUDAFunction.apply(
                self.training, k, v, q, decay, g, S_prev,
                self.n_heads, True, use_fused_l2, self.checkpoint_interval
            )
        return S_new, output

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input
            hidden: Optional list of H matrices, each [B, n_state, n_state]

        Returns:
            output: [B, T, dim]
            hidden: list of H final matrix states
        """
        B, T, D = x.shape
        n = self.n_state
        H = self.n_heads
        erase_gate = None
        value_write_gate = None

        # === Check if we can use fused projection CUDA kernel ===
        # Note: Fused projection is forward-only, use only for inference
        use_fused_proj = (
            USE_FUSED_PROJECTION and
            self._use_fused_projection and
            x.is_cuda and
            x.dtype == torch.bfloat16 and
            not self.training and  # Only use for inference - no gradients
            not self.use_split_edit and
            # E1 parallel-scan experiment: when an explicit recurrence mode is
            # requested (serial/scan), force the eager projection path so both
            # eval modes share IDENTICAL projections and differ ONLY in the
            # state recurrence execution order.
            getattr(self, 'e88_recurrence_mode', None) is None
        )

        if use_fused_proj:
            # === Use fused projection kernel ===
            # Combines GEMM + conv + silu + L2 norm + decay in one kernel
            W_qkva, conv_q, conv_k, conv_v = self._get_fused_projection_weights()

            # Transpose input for CUDA: [B, T, dim] -> [T, B, dim]
            x_cuda = x.transpose(0, 1).contiguous()

            # Call fused projection kernel
            # Returns: q [T, B, H, n_state], k [T, B, H, n_state], v [T, B, H, head_v_dim], decay [T, B, H]
            # Note: A_log and dt_bias must be float32 for numerical stability
            q, k, v, decay = hasty_pytorch_lib.e88_fused_projection(
                x_cuda, W_qkva,
                conv_q, conv_k, conv_v,
                self.A_log.float(), self.dt_bias.float(),
                H, self.key_dim, self.value_dim, self.d_conv,
                self.use_conv, self.use_silu, self.use_l2_norm
            )

            # Transpose back: [T, B, H, dim] -> [B, T, H, dim]
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            decay = decay.transpose(0, 1)

            # Note: q, k are already L2 normalized by the fused kernel if use_l2_norm=True

        else:
            # === Check early if chunked projection path will be used ===
            # When chunking, we skip full-T projections entirely (computed per-chunk instead)
            _use_optimized = (
                USE_OPTIMIZED_KERNELS and
                (E88_OPTIMIZED_AVAILABLE or self.use_triton) and
                x.is_cuda and
                x.dtype == torch.bfloat16 and
                self.training and
                self.use_gate and
                self.g_proj is not None and
                self.gate_activation == 'silu' and
                not self.use_output_norm and
                not self._use_fused_norm_gate and
                not self.use_write_gate and
                (not self.use_split_edit or self.use_triton) and
                (not self.raw_write or self.use_triton)
            )
            _will_chunk = (
                _use_optimized and
                self.projection_chunk_size > 0 and
                T > self.projection_chunk_size and
                not self.use_conv and
                not self.use_split_edit
            )

            if not _will_chunk:
                # === Original code path: separate projections for full T ===
                # Projections (fused QKV for GEMM efficiency)
                if self.qkv_proj is not None:
                    # Single fused GEMM: [B, T, dim] @ [dim, 2*key_dim + value_dim]
                    qkv = self.qkv_proj(x)  # [B, T, 2*key_dim + value_dim]
                    qkv_silu_in_kernel = _use_optimized and self.use_triton and self.use_silu and not self.use_conv
                    q = qkv[..., :self.key_dim]
                    k = qkv[..., self.key_dim:2*self.key_dim]
                    v = qkv[..., 2*self.key_dim:]
                else:
                    qkv_silu_in_kernel = False
                    q = self.q_proj(x)  # [B, T, key_dim]
                    k = self.k_proj(x)  # [B, T, key_dim]
                    v = None  # Set below

                # Short convolutions with optional SiLU (FLA-GDN style)
                # Can be disabled for ablation
                if self.use_conv and self.q_conv is not None:
                    # Conv expects [B, C, T]
                    q = self.q_conv(q.transpose(1, 2))[:, :, :T].transpose(1, 2)
                    k = self.k_conv(k.transpose(1, 2))[:, :, :T].transpose(1, 2)
                # Apply SiLU if enabled
                if self.use_silu and not qkv_silu_in_kernel:
                    q = F.silu(q)
                    k = F.silu(k)

                # v projection (or tied to k when expansion=1.0 and tie_kv=True)
                if self.tie_kv:
                    # Skip v_proj - v shares k after conv+silu (square state)
                    v = k  # [B, T, key_dim] = [B, T, value_dim] when expansion=1.0
                elif v is None:
                    # Non-fused path: separate v projection
                    v = self.v_proj(x)  # [B, T, value_dim]

                # Apply conv and SiLU to v if not using fused projection (fused already did q,k,v together)
                if not self.tie_kv and self.qkv_proj is None:
                    if self.use_conv and self.v_conv is not None:
                        v = self.v_conv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
                    if self.use_silu and not qkv_silu_in_kernel:
                        v = F.silu(v)
                elif not self.tie_kv:
                    # Fused path: apply conv and SiLU to v
                    if self.use_conv and self.v_conv is not None:
                        v = self.v_conv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
                    if self.use_silu and not qkv_silu_in_kernel:
                        v = F.silu(v)

                # Compute decay
                if self.decay_mode == 'none':
                    decay = torch.ones(B, T, H, device=x.device, dtype=x.dtype)
                elif self.decay_mode == 'constant':
                    decay = torch.sigmoid(self.decay_logit).view(1, 1, H).expand(B, T, H).to(x.dtype).contiguous()
                elif self.decay_mode == 'simple':
                    # Simple sigmoid decay (ablation: replaces Mamba2-style)
                    decay = torch.sigmoid(self.beta_proj(x))  # [B, T, H]
                else:
                    # Mamba2-style exponential decay
                    alpha = self.a_proj(x)  # [B, T, H]
                    if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
                        # Use fused Triton kernel (19x faster)
                        decay = triton_mamba2_decay(alpha, self.A_log.float(), self.dt_bias.float())
                    else:
                        # Fallback: PyTorch ops
                        # g = -exp(A_log) * softplus(a_proj(x) + dt_bias)
                        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
                        decay = g.exp().to(x.dtype)  # [B, T, H]

                # Compute write gate (beta) if enabled - gates how much delta writes to memory
                if self.use_write_gate and self.write_gate_proj is not None:
                    write_beta = torch.sigmoid(self.write_gate_proj(x))  # [B, T, H]
                else:
                    write_beta = None

                if self.use_split_edit:
                    erase_gate = torch.sigmoid(self.erase_gate_proj(x)).view(B, T, H, n)
                    value_write_gate = torch.sigmoid(self.value_write_gate_proj(x)).view(
                        B, T, H, self.head_v_dim
                    )

                # Reshape for per-head processing
                # q, k: [B, T, H, n_state]
                q = q.view(B, T, H, n)
                k = k.view(B, T, H, n)
                # v: [B, T, H, head_v_dim]
                v = v.view(B, T, H, self.head_v_dim)
            # else: chunked path — projections computed per-chunk in _process_chunk

        # === Initialize states ===
        # State shape: [B, H, n_state, head_v_dim] stacked (for CUDA kernel)
        if hidden is None:
            S0 = torch.zeros(B, H, n, self.head_v_dim, device=x.device, dtype=x.dtype)
        else:
            # Convert list to stacked tensor
            S0 = torch.stack(hidden, dim=1)  # [B, H, n, head_v_dim]

        # === Fast single-token step kernel (Triton) ===
        # WIP: kernel is correct in isolation (0.006 diff from ref) but end-to-end
        # end-to-end logit diff grows to ~7 after 10 layers — 20× precision
        # drift per layer vs the vectorized fallback despite fp32 internals.
        # Suspect: bf16 rounding order differs vs PyTorch einsum's tensor-core
        # accumulator. Leaving gated off until root-caused. Set env var
        # E88_USE_STEP_KERNEL=1 to experiment.
        import os as _os
        use_step_kernel = (
            _os.environ.get('E88_USE_STEP_KERNEL') == '1' and
            getattr(self, 'e88_recurrence_mode', None) is None and
            T == 1 and
            not self.training and
            x.is_cuda and
            x.dtype == torch.bfloat16 and
            n == self.head_v_dim and
            not self.use_write_gate and
            not self.use_value_residual
            and not self.raw_write
            and not self.use_split_edit
        )
        if use_step_kernel:
            from .e88_step_kernel import e88_step
            # Pre-gate value: gate is applied inside the kernel
            g_step = None
            if self.use_gate and self.g_proj is not None:
                g_step = self.g_proj(x).view(B, T, H, self.head_v_dim)[:, 0]  # [B, H, V]
            out_t, S_new = e88_step(
                k[:, 0], q[:, 0], v[:, 0], decay[:, 0],
                S_in=S0,
                g=g_step,
                use_l2_norm=self.use_l2_norm,
                linear_state=self.linear_state,
                gate_activation=self.gate_activation,
            )
            output = out_t.unsqueeze(1)  # [B, 1, H, V]
            S_list = [S_new[:, h] for h in range(H)]
            fused_gate_used = g_step is not None
            # Skip the remaining recurrence branches; jump straight to head mixing.
            # Handled by setting markers that the code below checks.
            _step_kernel_used = True
        else:
            _step_kernel_used = False

        # === Use CUDA kernel if available ===
        use_cuda = (E88_NATIVE_CUDA_AVAILABLE and x.is_cuda and
                    x.dtype == torch.bfloat16 and self.training and
                    not self.raw_write and
                    not self.use_split_edit)

        # Check if we can use optimized kernels with [B, T, H, dim] layout (no transpose)
        use_optimized = (
            USE_OPTIMIZED_KERNELS and
            (E88_OPTIMIZED_AVAILABLE or self.use_triton) and
            x.is_cuda and
            x.dtype == torch.bfloat16 and
            self.training and
            self.use_gate and
            self.g_proj is not None and
            self.gate_activation == 'silu' and
            not self.use_output_norm and
            not self._use_fused_norm_gate and
            not self.use_write_gate and  # Optimized kernels don't support write gate yet
            not self.use_value_residual and  # Fused path gates before we can add D*v
            (not self.use_split_edit or self.use_triton) and
            (not self.raw_write or self.use_triton)  # raw-write is implemented in Triton/PyTorch fallback
        )

        # Track whether fused gating was used (to skip separate gating later)
        fused_gate_used = False

        if _step_kernel_used:
            pass  # already produced output and S_list above
        elif use_optimized:
            # === Optimized path: No transpose, 1.17-2.84x faster ===
            input_dtype = x.dtype

            # Determine if we can use in-kernel L2 normalization
            # Requires warp or coalesced forward kernel (not fused) + register-owned backward
            use_fused_l2 = (
                USE_FUSED_L2_NORM and
                self.use_l2_norm and
                not use_fused_proj and
                (
                    (self.use_triton and n <= 64 and self.head_v_dim <= 64) or
                    (
                        (E88_WARP_AVAILABLE or E88_COALESCED_AVAILABLE) and
                        E88_REGISTER_OWNED_AVAILABLE and
                        n <= 32 and self.head_v_dim <= 32
                    )
                )
            )

            # Check if we should use chunked projection recomputation
            # Saves ~5GB/layer at T=32K by only materializing one chunk's projections at a time
            can_chunk = (
                self.projection_chunk_size > 0 and
                T > self.projection_chunk_size and
                not self.use_conv and  # Conv1d has cross-chunk dependencies
                not self.use_split_edit
            )

            if can_chunk:
                # === Chunked path: recompute projections per-chunk during backward ===
                # Full-T projections were skipped above — computed per-chunk here instead
                C = self.projection_chunk_size
                S = S0.to(input_dtype)
                output_chunks = []

                for t_start in range(0, T, C):
                    t_end = min(t_start + C, T)
                    x_chunk = x[:, t_start:t_end]

                    S, out_chunk = torch.utils.checkpoint.checkpoint(
                        self._process_chunk, x_chunk, S, input_dtype, use_fused_l2,
                        use_reentrant=False
                    )
                    output_chunks.append(out_chunk)

                output = torch.cat(output_chunks, dim=1)
                S_final = S
            else:
                # === Non-chunked path (original) ===
                if use_fused_l2:
                    # Skip separate L2 norm - kernel will normalize in shared memory
                    k_norm = k.to(input_dtype)
                    q_norm = q.to(input_dtype)
                elif self.use_l2_norm and not use_fused_proj:
                    # Fallback: separate L2 norm kernel launches
                    if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and k.dtype == torch.bfloat16:
                        k_norm = triton_l2_norm(k.contiguous())
                        q_norm = triton_l2_norm(q.contiguous())
                    else:
                        k_norm = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)
                        q_norm = (q / (q.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)
                else:
                    k_norm = k.to(input_dtype)
                    q_norm = q.to(input_dtype)

                # Compute gate projection (kept in [B, T, H, dim] layout)
                g = self.g_proj(x).view(B, T, H, self.head_v_dim).to(input_dtype)
                erase_for_kernel = erase_gate.to(input_dtype) if self.use_split_edit else None
                value_write_for_kernel = (
                    value_write_gate.to(input_dtype) if self.use_split_edit else None
                )

                if self.use_triton:
                    from ndm.triton.e88_triton_optimized import e88_triton_optimized_apply
                    S_final, output = e88_triton_optimized_apply(
                        self.training,
                        k_norm, v.to(input_dtype), q_norm, decay.to(input_dtype),
                        g, S0.to(input_dtype), H, True, use_fused_l2,
                        self.checkpoint_interval,
                        apply_silu_qkv=qkv_silu_in_kernel,
                        raw_write=self.raw_write,
                        linear_state=self.linear_state,
                        erase_gate=erase_for_kernel,
                        value_write_gate=value_write_for_kernel,
                    )
                else:
                    # Call optimized kernel (auto-selects warp vs coalesced based on n_state)
                    S_final, output = E88OptimizedCUDAFunction.apply(
                        self.training,
                        k_norm, v.to(input_dtype), q_norm, decay.to(input_dtype),
                        g, S0.to(input_dtype), H, True, use_fused_l2,  # apply_gate=True, normalize_kq
                        self.checkpoint_interval
                    )

            fused_gate_used = True

            # Convert S_final back to list for hidden state
            S_list = [S_final[:, h] for h in range(H)]

        elif use_cuda:
            input_dtype = x.dtype  # Use x.dtype as authoritative (should be bf16)

            # L2 normalize k and q if enabled and not already done by fused projection
            # Fused projection already does L2 norm, so skip if used
            if self.use_l2_norm and not use_fused_proj:
                if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and k.dtype == torch.bfloat16:
                    # Use fused Triton kernel (36% faster)
                    k_norm = triton_l2_norm(k.contiguous())
                    q_norm = triton_l2_norm(q.contiguous())
                else:
                    # Fallback: PyTorch ops
                    k_norm = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)
                    q_norm = (q / (q.norm(dim=-1, keepdim=True) + 1e-6)).to(input_dtype)
            else:
                # Either L2 norm disabled, or already done by fused projection
                k_norm = k.to(input_dtype)
                q_norm = q.to(input_dtype)

            # Transpose for CUDA: [B, T, H, dim] -> [T, B, H, dim]
            # Cast ALL inputs to input_dtype to handle any autocast promotions
            k_cuda = k_norm.transpose(0, 1).contiguous()
            v_cuda = v.to(input_dtype).transpose(0, 1).contiguous()
            q_cuda = q_norm.transpose(0, 1).contiguous()
            decay_cuda = decay.to(input_dtype).transpose(0, 1).contiguous()
            S0 = S0.to(input_dtype)

            # Check if we can use fused gate kernel
            # Conditions: use_gate + silu + no output norm + fused gate available
            use_fused_gate = (
                USE_FUSED_GATE and
                E88_FUSED_GATE_AVAILABLE and
                self.use_gate and
                self.g_proj is not None and
                self.gate_activation == 'silu' and
                not self.use_output_norm and
                not self._use_fused_norm_gate and
                not self.use_value_residual
            )

            if use_fused_gate:
                # Compute gate projection and reshape for CUDA kernel
                g = self.g_proj(x).view(B, T, H, self.head_v_dim)
                g_cuda = g.to(input_dtype).transpose(0, 1).contiguous()

                # Call fused gate CUDA kernel
                S_final, output_cuda = E88FusedGateCUDAFunction.apply(
                    self.training, k_cuda, v_cuda, q_cuda, decay_cuda, g_cuda, S0, H
                )
                fused_gate_used = True
            elif self.use_write_gate and write_beta is not None:
                # Use CUDA kernel with write gate (beta)
                beta_cuda = write_beta.to(input_dtype).transpose(0, 1).contiguous()
                S_final, output_cuda = E88WithBetaCUDAFunction.apply(
                    self.training, k_cuda, v_cuda, q_cuda, decay_cuda, beta_cuda, S0, H
                )
            else:
                # Call standard CUDA kernel via autograd.Function
                S_final, output_cuda = E88FLAHybridCUDAFunction.apply(
                    self.training, k_cuda, v_cuda, q_cuda, decay_cuda, S0, H
                )

            # Transpose output back: [T, B, H, head_v_dim] -> [B, T, H, head_v_dim]
            output = output_cuda.transpose(0, 1)

            # Convert S_final back to list for hidden state
            S_list = [S_final[:, h] for h in range(H)]
        elif getattr(self, 'e88_recurrence_mode', None) == 'scan':
            # === E1 parallel-scan path (associative / Blelloch-style) ===
            # linear_state=True makes the matrix-state update an input-dependent
            # AFFINE map of the state: S_t = A_t S_{t-1} + B_t. Affine maps form
            # a (non-commutative) monoid under composition, so the per-step maps
            # can be combined with an associative scan instead of a serial time
            # loop. Result is exact up to floating-point reassociation. This is
            # the faithful chunk/scan eval path the experiment compares against
            # the serial loop on IDENTICAL weights and dtype.
            assert self.linear_state, (
                "associative-scan eval requires linear_state=True; the tanh "
                "nonlinearity is NOT a linear scan and cannot be composed.")
            output, S_list = self._scan_recurrence(
                k, v, q, decay, S0, B, T, H, n,
                write_beta=write_beta if self.use_write_gate else None,
                erase_gate=erase_gate, value_write_gate=value_write_gate,
            )
        else:
            # === PyTorch fallback: Recurrence with nonlinear matrix state ===
            # Vectorized over heads (no per-head loop). Time loop remains for
            # the recurrence. S has shape [B, H, n, head_v_dim].
            S = S0.clone()  # [B, H, n, head_v_dim]

            outputs = []
            for t in range(T):
                k_t = k[:, t]       # [B, H, n]
                q_t = q[:, t]       # [B, H, n]
                v_t = v[:, t]       # [B, H, head_v_dim]
                decay_t = decay[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

                if self.use_l2_norm:
                    k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)
                    q_norm = q_t / (q_t.norm(dim=-1, keepdim=True) + 1e-6)
                else:
                    k_norm = k_t
                    q_norm = q_t

                if self.use_split_edit:
                    read_key = k_norm * erase_gate[:, t]
                    write_value = v_t * value_write_gate[:, t]
                else:
                    read_key = k_norm
                    write_value = v_t

                # Retrieve from memory: einsum over n_state
                # S: [B, H, n, head_v_dim], read_key: [B, H, n] -> retrieved: [B, H, head_v_dim]
                if self.raw_write:
                    delta = write_value  # [B, H, head_v_dim]
                else:
                    retrieved = torch.einsum('bhiv,bhi->bhv', S, read_key)
                    delta = write_value - retrieved  # [B, H, head_v_dim]

                # Outer product: [B, H, n, head_v_dim]
                outer = torch.einsum('bhv,bhi->bhiv', delta, k_norm)

                if self.use_write_gate and write_beta is not None:
                    beta_t = write_beta[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
                    outer = beta_t * outer

                if self.linear_state:
                    S = decay_t * S + outer
                else:
                    S = torch.tanh(decay_t * S + outer)

                # Query the state: [B, H, head_v_dim]
                Sq = torch.einsum('bhiv,bhi->bhv', S, q_norm)
                outputs.append(Sq)

            # Stack time: [B, T, H, head_v_dim]
            output = torch.stack(outputs, dim=1)
            # Materialize S_list for downstream code paths that expect it
            S_list = [S[:, h] for h in range(H)]

        if self.use_value_residual and self.value_residual is not None:
            output = output + v.to(output.dtype) * self.value_residual.to(output.dtype).view(1, 1, H, self.head_v_dim)

        # === Output normalization and gating ===
        # NOTE: Per-head norm hurts E88 (tanh-bounded state doesn't need it)
        # Only applied when use_output_norm=True (default False)
        # Skip gating if already done by fused gate kernel
        if self._use_fused_norm_gate and self.g_proj is not None:
            # FusedRMSNormGated: rms_norm(x) * weight * g * sigmoid(g) in one Triton kernel
            g = self.g_proj(x).view(B, T, H, self.head_v_dim)  # [B, T, H, head_v_dim]
            # Reshape to 2D for Triton kernel: [B*T*H, head_v_dim]
            output_2d = output.reshape(-1, self.head_v_dim)
            g_2d = g.reshape(-1, self.head_v_dim)
            output_2d = self.o_norm(output_2d, g_2d)
            output = output_2d.view(B, T, H, self.head_v_dim)
        elif FLA_FUSED_NORM_AVAILABLE and hasattr(self, 'o_norm'):
            # FLA RMSNorm without gating (still efficient Triton kernel)
            output_2d = output.reshape(-1, self.head_v_dim)
            output_2d = self.o_norm(output_2d)
            output = output_2d.view(B, T, H, self.head_v_dim)
            # Apply gating separately if enabled (skip if fused gate was used)
            if self.use_gate and self.g_proj is not None and not fused_gate_used:
                g = self.g_proj(x).view(B, T, H, self.head_v_dim)
                if self.gate_activation == 'sigmoid':
                    output = output * torch.sigmoid(g)
                else:  # silu/swish
                    output = output * F.silu(g)
        else:
            # Fallback: manual per-head norm and gating
            if self.use_output_norm and hasattr(self, 'o_norm_weight'):
                # Per-head RMSNorm: x / sqrt(mean(x^2) + eps) * weight
                rms = output.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
                output = output * rms * self.o_norm_weight

            # Apply gating (skip if fused gate was used)
            if self.use_gate and self.g_proj is not None and not fused_gate_used:
                g = self.g_proj(x).view(B, T, H, self.head_v_dim)  # [B, T, H, head_v_dim]
                if self.gate_activation == 'sigmoid':
                    output = output * torch.sigmoid(g)
                else:  # silu/swish - FLA-GDN style
                    output = output * F.silu(g)

        # Head mixing (output is [B, T, H, head_v_dim])
        if self.head_mix == 'concat':
            # Standard: concat all heads, project to dim
            output = output.reshape(B, T, self.value_dim)
            output = self.o_proj(output)
        elif self.head_mix == 'weighted_sum':
            # Learnable scalar per head, then project
            weights = self.head_weights.softmax(dim=0).view(1, 1, self.n_heads, 1)
            output = (output * weights).sum(dim=2)  # [B, T, head_v_dim]
            output = self.o_proj(output)
        elif self.head_mix == 'per_head':
            # Each head projects independently, then sum
            output = sum(self.head_projs[h](output[:, :, h]) for h in range(self.n_heads))
        elif self.head_mix == 'input_weighted':
            # Input-dependent head weighting
            weights = self.head_attn(x).softmax(dim=-1).unsqueeze(-1)  # [B, T, H, 1]
            output = (output * weights).sum(dim=2)  # [B, T, head_v_dim]
            output = self.o_proj(output)
        elif self.head_mix == 'sum':
            # Direct sum of heads
            output = output.sum(dim=2)  # [B, T, head_v_dim]
            output = self.o_proj(output)

        output = self.dropout(output)

        return output, S_list

    # ------------------------------------------------------------------
    # E1 experiment: associative / parallel-scan recurrence
    # ------------------------------------------------------------------
    @staticmethod
    def _affine_scan(A, Bm):
        """Inclusive associative scan of affine maps along the time axis (dim=1).

        Each (A[t], Bm[t]) represents the map  S -> A[t] S + Bm[t].  The monoid
        product (apply map `a` BEFORE map `b` in time) is

            combine(a, b) = (A_b @ A_a,  A_b @ B_a + B_b).

        We use the Hillis-Steele doubling scan: at stride d, every position i>=d
        is combined with position i-d (earlier prefix on the right). After
        ceil(log2 T) passes, position t holds the composition of maps 0..t. The
        reduction ORDER (a balanced doubling tree) differs from the serial left
        fold, which is exactly the floating-point reassociation under test.

        Shapes: A [B,T,H,n,n], Bm [B,T,H,n,v].
        """
        T = A.shape[1]
        d = 1
        while d < T:
            A_prev = A[:, : T - d]   # earlier maps (positions i-d) for i in [d:T)
            B_prev = Bm[:, : T - d]
            A_cur = A[:, d:]         # later maps (positions i)
            B_cur = Bm[:, d:]
            # combine(earlier=prev, later=cur)
            A_comb = torch.matmul(A_cur, A_prev)
            B_comb = torch.matmul(A_cur, B_prev) + B_cur
            A = torch.cat([A[:, :d], A_comb], dim=1)
            Bm = torch.cat([Bm[:, :d], B_comb], dim=1)
            d *= 2
        return A, Bm

    def _scan_recurrence(self, k, v, q, decay, S0, B, T, H, n,
                         write_beta=None, erase_gate=None, value_write_gate=None):
        """Compute the matrix-state sequence via an associative scan over the
        per-step affine maps, instead of the serial time loop. Algebra mirrors
        the serial fallback EXACTLY (same L2 norm, same delta-correction, same
        write/erase gates) so the only difference is execution order.

        Returns (output [B,T,H,head_v_dim], S_list[H] of [B,n,head_v_dim]).
        """
        # --- projections shared with the serial path ---
        if self.use_l2_norm:
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        else:
            k_norm = k
            q_norm = q
        dtype = k_norm.dtype

        if self.use_split_edit:
            read_key = k_norm * erase_gate          # [B,T,H,n]
            write_value = v * value_write_gate      # [B,T,H,head_v_dim]
        else:
            read_key = k_norm
            write_value = v

        beta = write_beta if (self.use_write_gate and write_beta is not None) else None

        # --- build the elementary affine maps ---
        # A_t[i,j] = decay_t * delta_ij - beta_t * k_norm[i] * read_key[j]
        # B_t[i,vd] = beta_t * k_norm[i] * write_value[vd]
        eye = torch.eye(n, device=k.device, dtype=dtype)
        A = decay.to(dtype)[..., None, None] * eye          # [B,T,H,n,n]
        if not self.raw_write:
            kk = torch.einsum('bthi,bthj->bthij', k_norm, read_key)  # [B,T,H,n,n]
            if beta is not None:
                kk = beta[..., None, None] * kk
            A = A - kk
        Bm = torch.einsum('bthi,bthv->bthiv', k_norm, write_value)   # [B,T,H,n,v]
        if beta is not None:
            Bm = beta[..., None, None] * Bm

        # --- associative scan: position t -> composition of maps 0..t ---
        A_scan, B_scan = self._affine_scan(A, Bm)

        # S_t = A_scan_t @ S0 + B_scan_t   (S0 broadcast over time)
        S0e = S0.to(dtype).unsqueeze(1)                     # [B,1,H,n,v]
        S = torch.matmul(A_scan, S0e) + B_scan             # [B,T,H,n,v]

        # output_t = q_norm_t . S_t   (contract over state index n)
        output = torch.einsum('bthi,bthiv->bthv', q_norm, S)  # [B,T,H,head_v_dim]
        S_list = [S[:, -1, h] for h in range(H)]
        return output, S_list

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        ablation_str = []
        if not self.use_conv:
            ablation_str.append('no_conv')
        if self.linear_state:
            ablation_str.append('linear_state')
        if self.raw_write:
            ablation_str.append('raw_write')
        if self.use_split_edit:
            ablation_str.append('split_edit')
        if not self.use_gate:
            ablation_str.append('no_gate')
        if self.decay_mode != 'mamba':
            ablation_str.append(f'decay={self.decay_mode}')
        if self.use_value_residual:
            ablation_str.append('value_residual')
        if not self.use_silu:
            ablation_str.append('no_silu')
        if not self.use_l2_norm:
            ablation_str.append('no_l2_norm')
        if not self.use_output_norm:
            ablation_str.append('no_output_norm')
        if self.head_mix != 'concat':
            ablation_str.append(f'head_mix={self.head_mix}')
        if self.gate_activation != 'sigmoid':
            ablation_str.append(f'gate={self.gate_activation}')
        ablation_info = f', ablations=[{",".join(ablation_str)}]' if ablation_str else ''
        return (f'dim={self.dim}, key_dim={self.key_dim}, value_dim={self.value_dim}, '
                f'n_state={self.n_state}, n_heads={self.n_heads}, expansion={self.expansion}, '
                f'tie_kv={self.tie_kv}{ablation_info}, LEVEL=88_FLA_HYBRID')


# Alias for backwards compatibility
E75MultiHead = E88FLAHybrid


if __name__ == "__main__":
    print("Testing E88 FLA Hybrid (Nonlinear Matrix State + FLA-GDN Design)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")

    # Test dimensions
    B, T, dim = 4, 32, 512
    n_state = 32
    n_heads = 8

    print(f"\nConfig: B={B}, T={T}, dim={dim}, n_state={n_state}, n_heads={n_heads}")

    # Test E88FLAHybrid
    print("\n--- E88 FLA Hybrid ---")
    model = E88FLAHybrid(
        dim=dim,
        expansion=2.0,
        n_state=n_state,
        n_heads=n_heads,
    ).to(device).to(dtype)

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Key dim: {model.key_dim}, Value dim: {model.value_dim}")

    # Test forward pass
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    out, S_list = model(x)
    print(f"Forward: Input {x.shape} -> Output {out.shape}")
    print(f"  Number of state matrices: {len(S_list)}, each {S_list[0].shape}")

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print("Backward: OK")

    # Check gradients exist
    grad_k = model.k_proj.weight.grad
    grad_a = model.a_proj.weight.grad
    print(f"  k_proj grad norm: {grad_k.norm().item():.4f}")
    print(f"  a_proj grad norm: {grad_a.norm().item():.4f}")

    # Test multiple head configurations
    print("\n--- Testing different head configurations ---")
    configs = [
        (4, 32, 2.0),   # 4 heads, 32 state, 2x expansion
        (8, 24, 2.0),   # 8 heads, 24 state, 2x expansion
        (8, 32, 1.5),   # 8 heads, 32 state, 1.5x expansion
        (16, 16, 2.0),  # 16 heads, 16 state, 2x expansion
    ]

    for H, n, exp in configs:
        model_test = E88FLAHybrid(
            dim=dim,
            expansion=exp,
            n_state=n,
            n_heads=H,
        ).to(device).to(dtype)

        x_test = torch.randn(B, T, dim, device=device, dtype=dtype)
        out_test, S_test = model_test(x_test)

        params = model_test.get_num_params()
        state_size = H * n * model_test.head_v_dim  # Rectangular state
        print(f"  H={H}, n={n}, exp={exp}: params={params:,}, state_size={state_size}")

        # Quick backward test
        out_test.sum().backward()

    print("\n" + "=" * 60)
    print("All tests passed!")
