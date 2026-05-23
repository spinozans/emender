"""
E89: TRUE Residual State - Nonlinear Matrix State with Skip Connection

** EXPERIMENTAL - E88 significantly outperforms this approach **

Key difference from E88:
- E88: S = tanh(decay * S + outer(delta, k))  -- tanh on entire update (RECOMMENDED)
- E89: S = S + tanh(decay * S + outer(delta, k))  -- residual/skip connection

CRITICAL ISSUE: This approach causes unbounded state growth!
- After T=256 steps, state max reaches ~255 (grows ~1 per step)
- E88 keeps state bounded in [-1, 1] range
- Result: E88 achieves 1.81 loss, E89 plateaus at 2.65 loss

Why it fails:
- Unlike ResNets (finite depth), RNNs have unbounded sequence length
- Each timestep adds ~tanh(...)~O(1) to state, causing linear growth
- Unbounded state causes retrieval S@k to explode
- Delta v-retrieved becomes dominated by huge retrieval values

This adds a residual/skip connection similar to ResNet, allowing gradients to
flow directly through the identity path. The E88 update is computed as before
but added as a residual rather than replacing the state.

Gradient flow advantages (in theory, but doesn't help in practice):
- Identity gradient flows directly: dS_{t-1} += dS_t
- Tanh gradient: dS_{t-1} += dS_t * dtanh * decay
- Better gradient flow for longer sequences

Architecture is identical to E88 otherwise - same projections, decay, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E89_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e89_residual_state_forward')
except ImportError:
    E89_CUDA_AVAILABLE = False

# Import E88 base classes for shared functionality
from .e88_fla_hybrid import (
    USE_FUSED_PROJECTION,
    USE_REDUCED_SYNC_BACKWARD,
    E88_FUSED_PROJECTION_AVAILABLE,
)


class E89ResidualStateCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E89 Residual State autograd function.

    Same interface as E88, but calls E89 kernels which use:
    S = decay * S + tanh(outer(delta, k))
    instead of:
    S = tanh(decay * S + outer(delta, k))
    """

    @staticmethod
    def forward(ctx, training, k, v, q, decay, S0, n_heads):
        """Forward pass using E89 residual state CUDA kernel."""
        results = hasty_pytorch_lib.e89_residual_state_forward(
            training, k, v, q, decay, S0, n_heads
        )
        S = results[0]
        output = results[1]
        S_cache = results[2]

        if training:
            # Split S_cache into S_checkpoints and Sq_cache
            T, B, H = k.shape[:3]
            n_state = k.shape[3]
            head_v_dim = v.shape[3]
            checkpoint_interval = 32  # Must match E89_CHECKPOINT_INTERVAL in CUDA
            num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
            s_checkpoints_size = num_checkpoints * B * H * n_state * head_v_dim

            S_checkpoints = S_cache[:s_checkpoints_size]
            Sq_cache = S_cache[s_checkpoints_size:]

            ctx.save_for_backward(k, v, q, decay, S_checkpoints, Sq_cache)
            ctx.n_heads = n_heads

        return S, output

    @staticmethod
    def backward(ctx, d_S, d_output):
        """Backward pass using E89 residual state CUDA kernel."""
        k, v, q, decay, S_checkpoints, Sq_cache = ctx.saved_tensors
        n_heads = ctx.n_heads

        d_k, d_v, d_q, d_decay = hasty_pytorch_lib.e89_residual_state_backward(
            k, v, q, decay, S_checkpoints, Sq_cache, d_output, n_heads
        )

        return None, d_k, d_v, d_q, d_decay, None, None


class E89ResidualStateCell(nn.Module):
    """E89 Residual State Cell - Same as E88 but with residual state update.

    Key difference: S = decay * S + tanh(outer(delta, k)) instead of
                    S = tanh(decay * S + outer(delta, k))

    This provides better gradient flow through the decay path.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 56,
        n_state: int = 32,
        expansion: float = 1.0,
        use_conv: bool = False,
        d_conv: int = 4,
        use_gate: bool = False,
        use_silu: bool = True,
        use_output_norm: bool = False,
        **kwargs,  # Accept extra kwargs from LadderLM
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_state = n_state
        self.expansion = expansion

        # Match E88 dimension scheme: head_v_dim = n_state
        # This gives a square state matrix S[n_state, n_state] per head
        self.head_v_dim = n_state
        self.d_inner = n_heads * self.head_v_dim  # value_dim = n_heads * head_v_dim

        # Total key dim = n_state * n_heads
        key_dim = n_state * n_heads

        self.use_conv = use_conv
        self.d_conv = d_conv
        self.use_gate = use_gate
        self.use_silu = use_silu
        self.use_output_norm = use_output_norm

        # QKV projections - project to per-head dimensions
        self.q_proj = nn.Linear(dim, key_dim, bias=False)
        self.k_proj = nn.Linear(dim, key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Mamba2-style decay parameters
        self.A_log = nn.Parameter(torch.zeros(n_heads))  # log eigenvalues
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))  # time-step bias
        self.a_proj = nn.Linear(dim, n_heads, bias=False)  # input-dependent decay

        # Optional conv for k, v, q
        if use_conv:
            self.conv_q = nn.Conv1d(key_dim, key_dim, d_conv, groups=key_dim, padding=d_conv-1, bias=False)
            self.conv_k = nn.Conv1d(key_dim, key_dim, d_conv, groups=key_dim, padding=d_conv-1, bias=False)
            self.conv_v = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv-1, bias=False)

        # Optional output gating
        if use_gate:
            self.g_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Output projection back to dim
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional output normalization
        if use_output_norm:
            self.out_norm = nn.LayerNorm(self.d_inner)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following Mamba2 conventions."""
        # A_log init: small negative values for stable decay
        nn.init.uniform_(self.A_log, -5, -1)
        # dt_bias: slightly positive for reasonable time steps
        nn.init.uniform_(self.dt_bias, 0.5, 1.5)
        # Standard init for projections
        for name, p in self.named_parameters():
            if 'proj' in name and 'weight' in name:
                nn.init.xavier_uniform_(p)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create zero initial state: [B, H, n_state, head_v_dim]"""
        return torch.zeros(
            batch_size, self.n_heads, self.n_state, self.head_v_dim,
            device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [T, B, dim] input
            S: [B, H, n_state, head_v_dim] state (optional, zeros if None)

        Returns:
            output: [T, B, dim]
            S_new: [B, H, n_state, head_v_dim]
        """
        T, B, _ = x.shape

        # E89 debug: print input dtype
        # print(f'E89 forward: x.dtype={x.dtype}, x.is_cuda={x.is_cuda}')

        if S is None:
            S = self.initial_state(B, x.device, x.dtype)

        # Project to q, k, v
        q = self.q_proj(x)  # [T, B, key_dim]
        k = self.k_proj(x)  # [T, B, key_dim]
        v = self.v_proj(x)  # [T, B, d_inner]

        # Optional conv + silu
        if self.use_conv:
            # Conv expects [B, C, T]
            q = q.permute(1, 2, 0)  # [B, key_dim, T]
            k = k.permute(1, 2, 0)
            v = v.permute(1, 2, 0)  # [B, d_inner, T]

            q = self.conv_q(q)[..., :T]  # Remove padding
            k = self.conv_k(k)[..., :T]
            v = self.conv_v(v)[..., :T]

            if self.use_silu:
                q = F.silu(q)
                k = F.silu(k)
                v = F.silu(v)

            q = q.permute(2, 0, 1)  # [T, B, key_dim]
            k = k.permute(2, 0, 1)
            v = v.permute(2, 0, 1)  # [T, B, d_inner]
        elif self.use_silu:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # Reshape for multi-head: [T, B, H, dim_per_head]
        q = q.view(T, B, self.n_heads, self.n_state)
        k = k.view(T, B, self.n_heads, self.n_state)
        v = v.view(T, B, self.n_heads, self.head_v_dim)

        # L2 normalize q and k
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Compute Mamba2-style decay: exp(-exp(A_log) * softplus(a_proj(x) + dt_bias))
        a = self.a_proj(x)  # [T, B, H]
        dt = F.softplus(a + self.dt_bias)  # [T, B, H]
        decay = torch.exp(-torch.exp(self.A_log) * dt)  # [T, B, H]

        # Use CUDA kernel if available - cast to bf16 if needed
        if E89_CUDA_AVAILABLE and x.is_cuda:
            # CUDA kernel requires bf16 - cast all tensors
            k_cuda = k.to(torch.bfloat16)
            v_cuda = v.to(torch.bfloat16)
            q_cuda = q.to(torch.bfloat16)
            decay_cuda = decay.to(torch.bfloat16)
            S_cuda = S.to(torch.bfloat16)

            S_new, output = E89ResidualStateCUDAFunction.apply(
                self.training, k_cuda, v_cuda, q_cuda, decay_cuda, S_cuda, self.n_heads
            )

            # Cast output back to input dtype if needed
            if output.dtype != x.dtype:
                output = output.to(x.dtype)
            if S_new.dtype != S.dtype:
                S_new = S_new.to(S.dtype)
        else:
            # Fallback to Python implementation (handles other dtypes)
            S_new, output = self._forward_python(k, v, q, decay, S)

        # Reshape output: [T, B, H, head_v_dim] -> [T, B, d_inner]
        output = output.view(T, B, self.d_inner)

        # Optional output normalization
        if self.use_output_norm:
            output = self.out_norm(output)

        # Optional output gating
        if self.use_gate:
            g = torch.sigmoid(self.g_proj(x))
            output = output * g

        # Output projection
        output = self.out_proj(output)

        return output, S_new

    def _forward_python(self, k, v, q, decay, S):
        """Python fallback for E89 residual state forward pass."""
        T, B, H, n_state = k.shape
        head_v_dim = v.shape[-1]

        outputs = []
        for t in range(T):
            k_t = k[t]  # [B, H, n_state]
            v_t = v[t]  # [B, H, head_v_dim]
            q_t = q[t]  # [B, H, n_state]
            decay_t = decay[t]  # [B, H]

            # Retrieve: S @ k (S is [B, H, n_state, head_v_dim], k is [B, H, n_state])
            # retrieved[b,h,j] = sum_i S[b,h,i,j] * k[b,h,i]
            retrieved = torch.einsum('bhij,bhi->bhj', S, k_t)  # [B, H, head_v_dim]

            # Delta
            delta = v_t - retrieved  # [B, H, head_v_dim]

            # E89 TRUE RESIDUAL state update: S = S + tanh(decay * S + outer(delta, k))
            # This is like E88's update but with a residual/skip connection for better gradient flow
            outer = torch.einsum('bhi,bhj->bhij', delta, k_t)  # [B, H, head_v_dim, n_state]
            outer = outer.permute(0, 1, 3, 2)  # [B, H, n_state, head_v_dim]
            pre = decay_t[:, :, None, None] * S + outer
            S = S + torch.tanh(pre)  # True residual: S + tanh(pre)

            # Output: S^T @ q
            Sq = torch.einsum('bhij,bhi->bhj', S, q_t)  # [B, H, head_v_dim]
            outputs.append(Sq)

        output = torch.stack(outputs, dim=0)  # [T, B, H, head_v_dim]
        return S, output
