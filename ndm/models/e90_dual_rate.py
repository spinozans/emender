"""
E90: Dual-Rate Factorized State (extends E88)

Extends E88 with two memory systems per head:
- **Fast state**: Small (k_fast × k_fast), updated every timestep
- **Slow state**: Larger (k_slow × k_slow), updated via learned soft gate

This allows more total state capacity while keeping per-step compute manageable:
- Per-step compute: O(H × k_fast²) + O(H × k_slow² × avg_gate)
- Total state: O(H × (k_fast² + k_slow²))

All parameters are learned and input-dependent:
- Fast/slow decay rates (Mamba2-style)
- Slow update gate (how much to update slow state each step)
- Output mixing (how to combine fast and slow retrievals)

Architecture per head h:
    # Fast state (updated every step)
    k_fast, v_fast, q_fast = project_fast(x)
    S_fast = tanh(decay_fast * S_fast + outer(delta_fast, k_fast))
    out_fast = S_fast @ q_fast

    # Slow state (gated update)
    k_slow, v_slow, q_slow = project_slow(x)
    slow_gate = sigmoid(slow_gate_proj(x))  # [0, 1] per head
    S_slow = tanh(decay_slow * S_slow + slow_gate * outer(delta_slow, k_slow))
    out_slow = S_slow @ q_slow

    # Input-dependent mixing
    mix_weights = softmax(mix_proj(x))  # [fast_weight, slow_weight] per head
    out_h = mix_fast * out_fast + mix_slow * out_slow

Inherits from E88:
- Mamba2-style exponential decay
- L2-normalized Q and K
- Output gating option
- CUDA kernel support (uses E88 kernel for fast state, Python for slow)
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
    # E90 dual-rate CUDA kernel
    E90_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e90_dual_rate_forward')
except ImportError:
    E88_NATIVE_CUDA_AVAILABLE = False
    E88_CUBLAS_BACKWARD_AVAILABLE = False
    E88_FUSED_PROJECTION_AVAILABLE = False
    E88_FUSED_GATE_AVAILABLE = False
    E88_CUDA_AVAILABLE = False
    E88_PRECOMPUTED_CUDA_AVAILABLE = False
    E90_CUDA_AVAILABLE = False

# Global flag to enable E90 CUDA kernel (can be toggled at runtime)
# Now uses O(T) checkpointed backward like E88 for full CUDA acceleration
USE_E90_CUDA = True

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


def _e90_cuda_forward_only(
    k_fast, v_fast, q_fast, decay_fast,
    k_slow, v_slow, q_slow, decay_slow, slow_gate,
    mix_fast, mix_slow,
    S_fast0, S_slow0, n_heads
):
    """CUDA forward pass for E90. Returns tensors that retain grad for autograd."""
    results = hasty_pytorch_lib.e90_dual_rate_forward(
        k_fast, v_fast, q_fast, decay_fast,
        k_slow, v_slow, q_slow, decay_slow, slow_gate,
        mix_fast, mix_slow,
        S_fast0, S_slow0, n_heads
    )
    return results[0], results[1], results[2]


class E90DualRateCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E90 Dual-Rate autograd function with checkpointing.

    E90 uses two states:
    - Fast state: [k_fast x v_fast], updated every timestep
    - Slow state: [k_slow x v_slow], updated via learned soft gate

    Both forward and backward use CUDA kernels with gradient checkpointing.
    Backward is O(T) instead of O(T²) by using segment-based checkpointing.
    """

    @staticmethod
    def forward(ctx, k_fast, v_fast, q_fast, decay_fast,
                k_slow, v_slow, q_slow, decay_slow, slow_gate,
                mix_fast, mix_slow,
                S_fast0, S_slow0, n_heads):
        """
        Args:
            k_fast: [T, B, H, k_fast] L2 normalized keys for fast state
            v_fast: [T, B, H, v_fast] values for fast state
            q_fast: [T, B, H, k_fast] L2 normalized queries for fast state
            decay_fast: [T, B, H] decay factors for fast state
            k_slow: [T, B, H, k_slow] L2 normalized keys for slow state
            v_slow: [T, B, H, v_slow] values for slow state
            q_slow: [T, B, H, k_slow] L2 normalized queries for slow state
            decay_slow: [T, B, H] decay factors for slow state
            slow_gate: [T, B, H] gate for slow state update
            mix_fast: [T, B, H] mixing weight for fast output
            mix_slow: [T, B, H] mixing weight for slow output
            S_fast0: [B, H, k_fast, v_fast] initial fast state
            S_slow0: [B, H, k_slow, v_slow] initial slow state
            n_heads: int
        """
        # Use CUDA for forward pass with checkpointing
        results = hasty_pytorch_lib.e90_dual_rate_forward(
            k_fast, v_fast, q_fast, decay_fast,
            k_slow, v_slow, q_slow, decay_slow, slow_gate,
            mix_fast, mix_slow,
            S_fast0, S_slow0, n_heads
        )
        S_fast_final = results[0]       # [B, H, k_fast, v_fast]
        S_slow_final = results[1]       # [B, H, k_slow, v_slow]
        output = results[2]             # [T, B, H, out_v_dim]
        S_fast_checkpoints = results[3] # [num_cp, B, H, k_fast, v_fast]
        S_slow_checkpoints = results[4] # [num_cp, B, H, k_slow, v_slow]

        # Save tensors for CUDA backward with checkpointing
        ctx.save_for_backward(
            k_fast, v_fast, q_fast, decay_fast,
            k_slow, v_slow, q_slow, decay_slow, slow_gate,
            mix_fast, mix_slow,
            S_fast_checkpoints, S_slow_checkpoints
        )
        ctx.n_heads = n_heads
        ctx.k_fast_dim = k_fast.size(-1)
        ctx.v_fast_dim = v_fast.size(-1)
        ctx.k_slow_dim = k_slow.size(-1)
        ctx.v_slow_dim = v_slow.size(-1)

        return S_fast_final, S_slow_final, output

    @staticmethod
    def backward(ctx, dS_fast, dS_slow, d_output):
        """CUDA backward pass with O(T) complexity using gradient checkpointing.

        For each segment:
        1. Replay forward once from checkpoint, caching S_{t-1}
        2. Backward through segment using cached states
        """
        (k_fast, v_fast, q_fast, decay_fast,
         k_slow, v_slow, q_slow, decay_slow, slow_gate,
         mix_fast, mix_slow,
         S_fast_checkpoints, S_slow_checkpoints) = ctx.saved_tensors

        n_heads = ctx.n_heads

        # Use CUDA backward kernel with checkpointing
        grads = hasty_pytorch_lib.e90_dual_rate_backward(
            k_fast, v_fast, q_fast, decay_fast,
            k_slow, v_slow, q_slow, decay_slow, slow_gate,
            mix_fast, mix_slow,
            S_fast_checkpoints, S_slow_checkpoints,
            d_output.contiguous(),
            n_heads
        )
        # grads = [d_k_fast, d_v_fast, d_q_fast, d_decay_fast,
        #          d_k_slow, d_v_slow, d_q_slow, d_decay_slow,
        #          d_slow_gate, d_mix_fast, d_mix_slow]
        d_k_fast = grads[0]
        d_v_fast = grads[1]
        d_q_fast = grads[2]
        d_decay_fast = grads[3]
        d_k_slow = grads[4]
        d_v_slow = grads[5]
        d_q_slow = grads[6]
        d_decay_slow = grads[7]
        d_slow_gate = grads[8]
        d_mix_fast = grads[9]
        d_mix_slow = grads[10]

        return (d_k_fast, d_v_fast, d_q_fast, d_decay_fast,
                d_k_slow, d_v_slow, d_q_slow, d_decay_slow, d_slow_gate,
                d_mix_fast, d_mix_slow,
                None, None, None)


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


class E90DualRate(nn.Module):
    """
    E90: Dual-Rate Factorized State (extends E88).

    Two memory systems per head:
    - Fast state: Small (k_fast × k_fast), updated every timestep
    - Slow state: Larger (k_slow × k_slow), updated via learned soft gate

    Total state per head: k_fast² + k_slow²
    Effective compute: O(k_fast²) + O(k_slow² × avg_gate)
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 32,  # Kept for compatibility, but k_fast/k_slow override
        n_heads: int = 8,
        # === DUAL-RATE PARAMETERS ===
        k_fast: int = None,   # Fast state dimension (small, cheap, updated every step), default=16
        k_slow: int = None,   # Slow state dimension (larger, gated update), default=48
        # ============================
        dropout: float = 0.0,
        d_conv: int = 4,
        use_cuda: bool = True,
        tie_kv: bool = False,
        # Ablation options
        use_conv: bool = False,
        linear_state: bool = False,
        use_gate: bool = True,  # E90 default: use gate (for output gating)
        simple_decay: bool = False,
        use_silu: bool = True,
        use_l2_norm: bool = True,
        use_output_norm: bool = False,
        head_mix: str = 'concat',
        gate_activation: str = 'silu',  # E90 default: silu gating
        **kwargs
    ):
        super().__init__()

        # Apply defaults for None values
        if k_fast is None:
            k_fast = 16
        if k_slow is None:
            k_slow = 48

        # Validate state dimensions
        if k_fast % 4 != 0:
            raise ValueError(f"k_fast must be multiple of 4, got {k_fast}")
        if k_slow % 4 != 0:
            raise ValueError(f"k_slow must be multiple of 4, got {k_slow}")

        # Validate gate activation
        if gate_activation not in ['sigmoid', 'silu', 'swish']:
            raise ValueError(f"gate_activation must be 'sigmoid', 'silu', or 'swish', got {gate_activation}")

        self.dim = dim
        self.n_state = n_state  # Kept for compatibility
        self.n_heads = n_heads
        self.d_conv = d_conv
        self.expansion = expansion

        # === DUAL-RATE STATE DIMENSIONS ===
        self.k_fast = k_fast
        self.k_slow = k_slow
        self.v_fast = int(k_fast * expansion)  # Value dim for fast state
        self.v_slow = int(k_slow * expansion)  # Value dim for slow state

        # Ablation flags
        self.use_conv = use_conv
        self.linear_state = linear_state
        self.use_gate = use_gate
        self.gate_activation = gate_activation
        self.simple_decay = simple_decay
        self.use_silu = use_silu
        self.use_l2_norm = use_l2_norm
        self.use_output_norm = use_output_norm
        self.head_mix = head_mix

        # === FAST STATE PROJECTIONS ===
        self.key_dim_fast = n_heads * k_fast
        self.value_dim_fast = n_heads * self.v_fast
        self.head_v_dim_fast = self.v_fast

        # For E88 CUDA kernel compatibility
        self.key_dim = self.key_dim_fast
        self.value_dim = self.value_dim_fast
        self.head_v_dim = self.head_v_dim_fast

        # Fused QKV for fast state
        self.tie_kv = False  # Disable tie_kv for dual-rate
        self.fuse_qkv = True
        self.qkv_proj = nn.Linear(dim, 2 * self.key_dim_fast + self.value_dim_fast, bias=False)
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

        # === SLOW STATE PROJECTIONS ===
        self.key_dim_slow = n_heads * k_slow
        self.value_dim_slow = n_heads * self.v_slow
        self.head_v_dim_slow = self.v_slow

        # Separate QKV for slow state
        self.qkv_slow_proj = nn.Linear(dim, 2 * self.key_dim_slow + self.value_dim_slow, bias=False)

        # === SLOW UPDATE GATE (input-dependent) ===
        # Controls how much the slow state updates each step
        self.slow_gate_proj = nn.Linear(dim, n_heads, bias=True)
        # Initialize bias negative so slow state updates less frequently by default
        nn.init.constant_(self.slow_gate_proj.bias, -2.0)

        # === OUTPUT MIXING (input-dependent) ===
        # Learns how to combine fast and slow retrievals
        self.mix_proj = nn.Linear(dim, n_heads * 2, bias=True)  # [fast_weight, slow_weight] per head

        # === Decay parameters (separate for fast and slow) ===
        if not simple_decay:
            # FAST STATE DECAY (Mamba2-style)
            self.a_proj = nn.Linear(dim, n_heads, bias=False)  # For E88 CUDA kernel
            self.a_fast_proj = self.a_proj  # Alias
            A_fast = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A_fast))
            self.A_log._no_weight_decay = True
            self.A_fast_log = self.A_log  # Alias
            dt_min, dt_max = 0.001, 0.1
            dt_init_floor = 1e-4
            dt_fast = torch.exp(
                torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            dt_fast = torch.clamp(dt_fast, min=dt_init_floor)
            inv_dt_fast = dt_fast + torch.log(-torch.expm1(-dt_fast))
            self.dt_bias = nn.Parameter(inv_dt_fast)
            self.dt_bias._no_weight_decay = True
            self.dt_fast_bias = self.dt_bias  # Alias

            # SLOW STATE DECAY (typically slower/larger retention)
            self.a_slow_proj = nn.Linear(dim, n_heads, bias=False)
            A_slow = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 4)  # Slower decay
            self.A_slow_log = nn.Parameter(torch.log(A_slow))
            self.A_slow_log._no_weight_decay = True
            dt_slow = torch.exp(
                torch.rand(n_heads) * (math.log(0.01) - math.log(0.001)) + math.log(0.001)
            )
            dt_slow = torch.clamp(dt_slow, min=dt_init_floor)
            inv_dt_slow = dt_slow + torch.log(-torch.expm1(-dt_slow))
            self.dt_slow_bias = nn.Parameter(inv_dt_slow)
            self.dt_slow_bias._no_weight_decay = True
        else:
            # Simple sigmoid decay (ablation)
            self.a_proj = None
            self.A_log = None
            self.dt_bias = None
            self.a_fast_proj = None
            self.A_fast_log = None
            self.dt_fast_bias = None
            self.a_slow_proj = None
            self.A_slow_log = None
            self.dt_slow_bias = None

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
        # E90: gate the combined (mixed) output
        # Use max of fast/slow value dims for output dimension
        self.out_v_dim = max(self.v_fast, self.v_slow)
        if use_gate:
            self.g_proj = nn.Linear(dim, n_heads * self.out_v_dim, bias=False)
        else:
            self.g_proj = None

        # === Simple decay option (replaces Mamba2-style) ===
        if simple_decay:
            self.beta_proj = nn.Linear(dim, n_heads, bias=False)
            self.beta_fast_proj = self.beta_proj
            self.beta_slow_proj = nn.Linear(dim, n_heads, bias=False)
        else:
            self.beta_proj = None
            self.beta_fast_proj = None
            self.beta_slow_proj = None

        # === Output projection ===
        # E90: output is [B, T, H, out_v_dim] after mixing fast and slow
        # Only concat mode supported for now
        if head_mix != 'concat':
            raise ValueError(f"E90 only supports head_mix='concat', got {head_mix}")
        self.o_proj = nn.Linear(n_heads * self.out_v_dim, dim, bias=False)

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
            not simple_decay and
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

    def _compute_decay(self, x: torch.Tensor, a_proj: nn.Linear,
                       A_log: nn.Parameter, dt_bias: nn.Parameter) -> torch.Tensor:
        """Compute Mamba2-style decay factor."""
        alpha = a_proj(x)  # [B, T, H]
        if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
            decay = triton_mamba2_decay(alpha, A_log.float(), dt_bias.float())
        else:
            g = -A_log.float().exp() * F.softplus(alpha.float() + dt_bias)
            decay = g.exp().to(x.dtype)
        return decay

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """L2 normalize along specified dimension."""
        if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
            return triton_l2_norm(x)
        return x / (x.norm(dim=dim, keepdim=True) + 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input
            hidden: Optional tuple of (S_fast, S_slow) states
                    S_fast: [B, H, k_fast, v_fast]
                    S_slow: [B, H, k_slow, v_slow]

        Returns:
            output: [B, T, dim]
            hidden: tuple of (S_fast, S_slow) final states
        """
        B, T, D = x.shape
        H = self.n_heads
        kf, vf = self.k_fast, self.v_fast
        ks, vs = self.k_slow, self.v_slow

        # === FAST STATE PROJECTIONS ===
        qkv_fast = self.qkv_proj(x)  # [B, T, 2*key_dim_fast + value_dim_fast]
        q_fast = qkv_fast[..., :self.key_dim_fast]
        k_fast = qkv_fast[..., self.key_dim_fast:2*self.key_dim_fast]
        v_fast = qkv_fast[..., 2*self.key_dim_fast:]

        if self.use_silu:
            q_fast = F.silu(q_fast)
            k_fast = F.silu(k_fast)
            v_fast = F.silu(v_fast)

        # Reshape: [B, T, H, k_fast] and [B, T, H, v_fast]
        q_fast = q_fast.view(B, T, H, kf)
        k_fast = k_fast.view(B, T, H, kf)
        v_fast = v_fast.view(B, T, H, vf)

        # L2 normalize
        if self.use_l2_norm:
            k_fast = self._l2_normalize(k_fast)
            q_fast = self._l2_normalize(q_fast)

        # === SLOW STATE PROJECTIONS ===
        qkv_slow = self.qkv_slow_proj(x)  # [B, T, 2*key_dim_slow + value_dim_slow]
        q_slow = qkv_slow[..., :self.key_dim_slow]
        k_slow = qkv_slow[..., self.key_dim_slow:2*self.key_dim_slow]
        v_slow = qkv_slow[..., 2*self.key_dim_slow:]

        if self.use_silu:
            q_slow = F.silu(q_slow)
            k_slow = F.silu(k_slow)
            v_slow = F.silu(v_slow)

        # Reshape: [B, T, H, k_slow] and [B, T, H, v_slow]
        q_slow = q_slow.view(B, T, H, ks)
        k_slow = k_slow.view(B, T, H, ks)
        v_slow = v_slow.view(B, T, H, vs)

        # L2 normalize
        if self.use_l2_norm:
            k_slow = self._l2_normalize(k_slow)
            q_slow = self._l2_normalize(q_slow)

        # === COMPUTE DECAYS ===
        if self.simple_decay:
            decay_fast = torch.sigmoid(self.beta_fast_proj(x))  # [B, T, H]
            decay_slow = torch.sigmoid(self.beta_slow_proj(x))  # [B, T, H]
        else:
            decay_fast = self._compute_decay(x, self.a_fast_proj, self.A_fast_log, self.dt_fast_bias)
            decay_slow = self._compute_decay(x, self.a_slow_proj, self.A_slow_log, self.dt_slow_bias)

        # === SLOW UPDATE GATE ===
        slow_gate = torch.sigmoid(self.slow_gate_proj(x))  # [B, T, H]

        # === OUTPUT MIXING WEIGHTS ===
        mix_raw = self.mix_proj(x).view(B, T, H, 2)  # [B, T, H, 2]
        mix_weights = F.softmax(mix_raw, dim=-1)
        mix_fast = mix_weights[..., 0]  # [B, T, H]
        mix_slow = mix_weights[..., 1]  # [B, T, H]

        # === INITIALIZE STATES ===
        if hidden is None:
            S_fast = torch.zeros(B, H, kf, vf, device=x.device, dtype=x.dtype)
            S_slow = torch.zeros(B, H, ks, vs, device=x.device, dtype=x.dtype)
        else:
            S_fast, S_slow = hidden
            S_fast = S_fast.to(dtype=x.dtype)
            S_slow = S_slow.to(dtype=x.dtype)

        # === DUAL-RATE RECURRENCE ===
        # Check if we can use CUDA kernel
        use_cuda = (
            USE_E90_CUDA and
            E90_CUDA_AVAILABLE and
            x.is_cuda and
            x.dtype == torch.bfloat16 and
            not self.linear_state  # CUDA kernel uses tanh
        )

        if use_cuda:
            # === CUDA PATH ===
            # Cast tensors back to bf16 if autocast promoted them to fp32
            # (F.normalize and F.softmax promote to fp32 for numerical stability)
            target_dtype = torch.bfloat16
            k_fast = k_fast.to(target_dtype)
            v_fast = v_fast.to(target_dtype)
            q_fast = q_fast.to(target_dtype)
            decay_fast = decay_fast.to(target_dtype)
            k_slow = k_slow.to(target_dtype)
            v_slow = v_slow.to(target_dtype)
            q_slow = q_slow.to(target_dtype)
            decay_slow = decay_slow.to(target_dtype)
            slow_gate = slow_gate.to(target_dtype)
            mix_fast = mix_fast.to(target_dtype)
            mix_slow = mix_slow.to(target_dtype)
            S_fast = S_fast.to(target_dtype)
            S_slow = S_slow.to(target_dtype)

            # Transpose: [B, T, H, dim] -> [T, B, H, dim] for CUDA kernel
            k_fast_cuda = k_fast.transpose(0, 1).contiguous()
            v_fast_cuda = v_fast.transpose(0, 1).contiguous()
            q_fast_cuda = q_fast.transpose(0, 1).contiguous()
            decay_fast_cuda = decay_fast.transpose(0, 1).contiguous()

            k_slow_cuda = k_slow.transpose(0, 1).contiguous()
            v_slow_cuda = v_slow.transpose(0, 1).contiguous()
            q_slow_cuda = q_slow.transpose(0, 1).contiguous()
            decay_slow_cuda = decay_slow.transpose(0, 1).contiguous()

            slow_gate_cuda = slow_gate.transpose(0, 1).contiguous()
            mix_fast_cuda = mix_fast.transpose(0, 1).contiguous()
            mix_slow_cuda = mix_slow.transpose(0, 1).contiguous()

            # Call CUDA kernel via autograd function
            S_fast, S_slow, output_cuda = E90DualRateCUDAFunction.apply(
                k_fast_cuda, v_fast_cuda, q_fast_cuda, decay_fast_cuda,
                k_slow_cuda, v_slow_cuda, q_slow_cuda, decay_slow_cuda, slow_gate_cuda,
                mix_fast_cuda, mix_slow_cuda,
                S_fast, S_slow, H
            )

            # Transpose output back: [T, B, H, out_v_dim] -> [B, T, H, out_v_dim]
            output = output_cuda.transpose(0, 1)

        else:
            # === PYTHON FALLBACK ===
            outputs = []

            for t in range(T):
                # Get values for this timestep
                kf_t = k_fast[:, t]  # [B, H, k_fast]
                vf_t = v_fast[:, t]  # [B, H, v_fast]
                qf_t = q_fast[:, t]  # [B, H, k_fast]
                decay_f_t = decay_fast[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

                ks_t = k_slow[:, t]  # [B, H, k_slow]
                vs_t = v_slow[:, t]  # [B, H, v_slow]
                qs_t = q_slow[:, t]  # [B, H, k_slow]
                decay_s_t = decay_slow[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

                slow_gate_t = slow_gate[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
                mix_f_t = mix_fast[:, t].unsqueeze(-1)  # [B, H, 1]
                mix_s_t = mix_slow[:, t].unsqueeze(-1)  # [B, H, 1]

                # === FAST STATE UPDATE (every step) ===
                retrieved_fast = torch.einsum('bhkv,bhk->bhv', S_fast, kf_t)
                delta_fast = vf_t - retrieved_fast
                outer_fast = torch.einsum('bhk,bhv->bhkv', kf_t, delta_fast)
                S_fast_new = decay_f_t * S_fast + outer_fast
                if not self.linear_state:
                    S_fast_new = torch.tanh(S_fast_new)
                S_fast = S_fast_new
                out_fast = torch.einsum('bhkv,bhk->bhv', S_fast, qf_t)

                # === SLOW STATE UPDATE (gated) ===
                retrieved_slow = torch.einsum('bhkv,bhk->bhv', S_slow, ks_t)
                delta_slow = vs_t - retrieved_slow
                outer_slow = torch.einsum('bhk,bhv->bhkv', ks_t, delta_slow)
                # Gated update: only update proportionally to slow_gate
                S_slow_new = decay_s_t * S_slow + slow_gate_t * outer_slow
                if not self.linear_state:
                    S_slow_new = torch.tanh(S_slow_new)
                S_slow = S_slow_new
                out_slow = torch.einsum('bhkv,bhk->bhv', S_slow, qs_t)

                # === MIX FAST AND SLOW OUTPUTS ===
                # Pad to match dimensions if needed
                if vf != vs:
                    if vf < vs:
                        out_fast_padded = F.pad(out_fast, (0, vs - vf))
                        out_combined = mix_f_t * out_fast_padded + mix_s_t * out_slow
                    else:
                        out_slow_padded = F.pad(out_slow, (0, vf - vs))
                        out_combined = mix_f_t * out_fast + mix_s_t * out_slow_padded
                else:
                    out_combined = mix_f_t * out_fast + mix_s_t * out_slow

                outputs.append(out_combined)  # [B, H, out_v_dim]

            # Stack: [B, T, H, out_v_dim]
            output = torch.stack(outputs, dim=1)

        # === OUTPUT GATING ===
        if self.use_gate and self.g_proj is not None:
            g = self.g_proj(x).view(B, T, H, self.out_v_dim)
            if self.gate_activation == 'sigmoid':
                output = output * torch.sigmoid(g)
            else:
                output = output * F.silu(g)

        # === OUTPUT PROJECTION ===
        output = output.view(B, T, H * self.out_v_dim)
        output = self.o_proj(output)
        output = self.dropout(output)

        return output, (S_fast, S_slow)

    def _forward_e88_compat(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Original E88 forward for reference - NOT USED in E90.

        Keeping this for reference only. The actual forward() uses dual-rate.
        """
        B, T, D = x.shape
        n = self.n_state
        H = self.n_heads

        # === Check if we can use fused projection CUDA kernel ===
        use_fused_proj = (
            USE_FUSED_PROJECTION and
            self._use_fused_projection and
            x.is_cuda and
            x.dtype == torch.bfloat16 and
            not self.training
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
            # === Original code path: separate projections ===
            # Projections (fused QKV for GEMM efficiency)
            if self.qkv_proj is not None:
                # Single fused GEMM: [B, T, dim] @ [dim, 2*key_dim + value_dim]
                qkv = self.qkv_proj(x)  # [B, T, 2*key_dim + value_dim]
                q = qkv[..., :self.key_dim]
                k = qkv[..., self.key_dim:2*self.key_dim]
                v = qkv[..., 2*self.key_dim:]
            else:
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
            if self.use_silu:
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
                if self.use_silu:
                    v = F.silu(v)
            elif not self.tie_kv:
                # Fused path: apply conv and SiLU to v
                if self.use_conv and self.v_conv is not None:
                    v = self.v_conv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
                if self.use_silu:
                    v = F.silu(v)

            # Compute decay
            if self.simple_decay:
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

            # Reshape for per-head processing
            # q, k: [B, T, H, n_state]
            q = q.view(B, T, H, n)
            k = k.view(B, T, H, n)
            # v: [B, T, H, head_v_dim]
            v = v.view(B, T, H, self.head_v_dim)

        # === Initialize states ===
        # State shape: [B, H, n_state, head_v_dim] stacked (for CUDA kernel)
        if hidden is None:
            S0 = torch.zeros(B, H, n, self.head_v_dim, device=x.device, dtype=x.dtype)
        else:
            # Convert list to stacked tensor
            S0 = torch.stack(hidden, dim=1)  # [B, H, n, head_v_dim]

        # === Use CUDA kernel if available ===
        use_cuda = (E88_NATIVE_CUDA_AVAILABLE and x.is_cuda and
                    x.dtype == torch.bfloat16 and self.training)

        # Track whether fused gating was used (to skip separate gating later)
        fused_gate_used = False

        if use_cuda:
            input_dtype = x.dtype  # Use x.dtype as authoritative (should be bf16)

            # L2 normalize k and q if enabled and not already done by fused projection
            # Fused projection already does L2 norm, so skip if used
            if self.use_l2_norm and not use_fused_proj:
                if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and k.dtype == torch.bfloat16:
                    # Use fused Triton kernel (36% faster)
                    k_norm = triton_l2_norm(k)
                    q_norm = triton_l2_norm(q)
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
                not self._use_fused_norm_gate
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
            else:
                # Call standard CUDA kernel via autograd.Function
                S_final, output_cuda = E88FLAHybridCUDAFunction.apply(
                    self.training, k_cuda, v_cuda, q_cuda, decay_cuda, S0, H
                )

            # Transpose output back: [T, B, H, head_v_dim] -> [B, T, H, head_v_dim]
            output = output_cuda.transpose(0, 1)

            # Convert S_final back to list for hidden state
            S_list = [S_final[:, h] for h in range(H)]
        else:
            # === PyTorch fallback: Recurrence with nonlinear matrix state ===
            S_list = [S0[:, h].clone() for h in range(H)]  # Convert to list

            outputs = []
            for t in range(T):
                head_outputs = []
                for h in range(H):
                    k_t = k[:, t, h]  # [B, n_state]
                    q_t = q[:, t, h]  # [B, n_state]
                    v_t = v[:, t, h]  # [B, head_v_dim]
                    decay_t = decay[:, t, h:h+1]  # [B, 1]

                    # L2 normalize k and q if enabled (FLA-GDN style)
                    if self.use_l2_norm:
                        k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n_state]
                        q_norm = q_t / (q_t.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n_state]
                    else:
                        k_norm = k_t
                        q_norm = q_t

                    # Retrieve from memory: S @ k_norm -> [B, head_v_dim]
                    retrieved = torch.einsum('biv,bi->bv', S_list[h], k_norm)

                    # Delta update
                    delta = v_t - retrieved  # [B, head_v_dim]

                    # Outer product: [B, n_state, head_v_dim]
                    outer = torch.einsum('bv,bi->biv', delta, k_norm)

                    # Gated update (NONLINEAR tanh vs linear for ablation)
                    # S = tanh(decay * S + outer) or S = decay * S + outer
                    if self.linear_state:
                        S_list[h] = decay_t.unsqueeze(-1) * S_list[h] + outer
                    else:
                        S_list[h] = torch.tanh(decay_t.unsqueeze(-1) * S_list[h] + outer)

                    # Query the state: S @ q_norm -> [B, head_v_dim]
                    Sq = torch.einsum('biv,bi->bv', S_list[h], q_norm)

                    # Output directly (FLA-GDN style - gating applied below)
                    head_outputs.append(Sq)

                # Stack head outputs: [B, H, head_v_dim]
                out_t = torch.stack(head_outputs, dim=1)
                outputs.append(out_t)

            # Stack time: [B, T, H, head_v_dim]
            output = torch.stack(outputs, dim=1)

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
            output = output.view(B, T, self.value_dim)
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

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_state_size(self):
        """Return total state elements per batch element."""
        fast_size = self.n_heads * self.k_fast * self.v_fast
        slow_size = self.n_heads * self.k_slow * self.v_slow
        return fast_size + slow_size

    def extra_repr(self):
        return (f'dim={self.dim}, n_heads={self.n_heads}, '
                f'k_fast={self.k_fast}, k_slow={self.k_slow}, '
                f'v_fast={self.v_fast}, v_slow={self.v_slow}, '
                f'state_size={self.get_state_size()}, '
                f'gate={self.gate_activation}, LEVEL=E90_DUAL_RATE')


if __name__ == "__main__":
    print("Testing E90 Dual-Rate State...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")

    # Test dimensions
    B, T, dim = 4, 32, 512
    n_heads = 8
    k_fast = 16
    k_slow = 48

    print(f"\nConfig: B={B}, T={T}, dim={dim}, n_heads={n_heads}")
    print(f"        k_fast={k_fast}, k_slow={k_slow}")

    # Test E90DualRate
    print("\n--- E90 Dual-Rate ---")
    model = E90DualRate(
        dim=dim,
        expansion=1.0,
        n_heads=n_heads,
        k_fast=k_fast,
        k_slow=k_slow,
        use_gate=True,
        gate_activation='silu',
    ).to(device).to(dtype)

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"State size: {model.get_state_size():,} elements")
    print(f"  Fast state: {n_heads} x {k_fast} x {model.v_fast} = {n_heads * k_fast * model.v_fast:,}")
    print(f"  Slow state: {n_heads} x {k_slow} x {model.v_slow} = {n_heads * k_slow * model.v_slow:,}")

    # Test forward pass
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    out, (S_fast, S_slow) = model(x)
    print(f"\nForward: Input {x.shape} -> Output {out.shape}")
    print(f"  S_fast shape: {S_fast.shape}")
    print(f"  S_slow shape: {S_slow.shape}")

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print("Backward: OK")

    # Check gradients exist
    grad_qkv = model.qkv_proj.weight.grad
    grad_slow = model.qkv_slow_proj.weight.grad
    grad_slow_gate = model.slow_gate_proj.weight.grad
    grad_mix = model.mix_proj.weight.grad
    print(f"\nGradient norms:")
    print(f"  qkv_proj (fast): {grad_qkv.norm().item():.4f}")
    print(f"  qkv_slow_proj:   {grad_slow.norm().item():.4f}")
    print(f"  slow_gate_proj:  {grad_slow_gate.norm().item():.4f}")
    print(f"  mix_proj:        {grad_mix.norm().item():.4f}")

    # Test with initial state
    print("\n--- Testing with initial state ---")
    model.zero_grad()
    x2 = torch.randn(B, T, dim, device=device, dtype=dtype)
    out2, (S_fast2, S_slow2) = model(x2, hidden=(S_fast.detach(), S_slow.detach()))
    print(f"Forward with state: OK")
    out2.sum().backward()
    print("Backward with state: OK")

    # Test different configurations
    print("\n--- Testing different configurations ---")
    configs = [
        (8, 16, 32),   # Default: k_fast=16, k_slow=32
        (8, 16, 48),   # Larger slow state
        (16, 8, 24),   # More heads, smaller states
        (4, 24, 64),   # Fewer heads, larger states
    ]

    for H, kf, ks in configs:
        model_test = E90DualRate(
            dim=dim,
            expansion=1.0,
            n_heads=H,
            k_fast=kf,
            k_slow=ks,
        ).to(device).to(dtype)

        x_test = torch.randn(B, T, dim, device=device, dtype=dtype)
        out_test, states_test = model_test(x_test)

        params = model_test.get_num_params()
        state_size = model_test.get_state_size()
        print(f"  H={H}, k_fast={kf}, k_slow={ks}: params={params:,}, state_size={state_size:,}")

        # Quick backward test
        out_test.sum().backward()

    print("\n" + "=" * 60)
    print("All tests passed!")
