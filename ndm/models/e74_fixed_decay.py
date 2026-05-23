"""
E74: Fixed Decay Delta Rule - The simplest architecture in the autopoietic ladder.

Mathematical Definition:
    S' = alpha * S + outer(v - S @ k_norm, k_norm)
    output = (S' @ q) * silu(S' @ q)

Where alpha in (0,1) is a FIXED scalar decay (not learned, or optionally learned as a single parameter).

This is the baseline for the autopoietic architecture sequence:
    E74 (Fixed Decay) -> E75 (Gated Delta) -> E76 (Self-Modulation) -> E79 (Coupled)

Key properties:
- Single state matrix S [n_state x n_state]
- Delta rule update with fixed exponential decay
- No input-dependent gating (simplest possible dynamics)
- Self-gated output: Sq * silu(Sq)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel
E74_FIXED_DECAY_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e74_fixed_decay_forward') and hasattr(hasty_pytorch_lib, 'e74_fixed_decay_backward'):
        E74_FIXED_DECAY_CUDA_AVAILABLE = True
except ImportError:
    pass


class E74FixedDecayCUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E74 Fixed Decay CUDA kernel."""

    @staticmethod
    def forward(ctx, x, S0, W_kvq, alpha, training):
        """
        Args:
            x: [T, B, dim] input
            S0: [B, n_state, n_state] initial state matrix
            W_kvq: [3*n_state, dim] fused projection weights for k, v, q
            alpha: scalar decay factor (0 < alpha < 1)
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final state matrix
        """
        S, output, kvq_cache, S_checkpoints, Sq_cache = \
            hasty_pytorch_lib.e74_fixed_decay_forward(training, x, S0, W_kvq, alpha)

        if training:
            ctx.save_for_backward(x, S_checkpoints, Sq_cache, kvq_cache, W_kvq)
            ctx.alpha = alpha

        return output, S

    @staticmethod
    def backward(ctx, d_output, d_S):
        x, S_checkpoints, Sq_cache, kvq_cache, W_kvq = ctx.saved_tensors
        alpha = ctx.alpha

        d_output = d_output.contiguous()

        dx, dW_kvq, d_alpha = hasty_pytorch_lib.e74_fixed_decay_backward(
            x, S_checkpoints, Sq_cache, kvq_cache, d_output, W_kvq, alpha
        )

        return dx, None, dW_kvq, d_alpha, None


class E74FixedDecayCell(nn.Module):
    """
    E74 Fixed Decay Delta Rule cell.

    State update: S' = alpha * S + outer(v - S @ k_norm, k_norm)
    Output: (S' @ q) * silu(S' @ q)
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        alpha: float = 0.9,
        learnable_alpha: bool = False,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.use_cuda = use_cuda and E74_FIXED_DECAY_CUDA_AVAILABLE
        self.learnable_alpha = learnable_alpha

        # FUSED projection: single GEMM for k, v, q
        # Layout: [k | v | q] = [3 * n_state, dim]
        self.W_kvq = nn.Parameter(torch.empty(3 * n_state, dim))

        # Alpha parameter (decay factor)
        if learnable_alpha:
            # Use log-space parameterization for stability
            # alpha = sigmoid(alpha_logit), initialized to give alpha ~ 0.9
            self.alpha_logit = nn.Parameter(torch.tensor(2.2))  # sigmoid(2.2) ~ 0.9
        else:
            self.register_buffer('alpha', torch.tensor(alpha))

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        nn.init.xavier_uniform_(self.W_kvq[:n])      # W_k
        nn.init.xavier_uniform_(self.W_kvq[n:2*n])   # W_v
        nn.init.xavier_uniform_(self.W_kvq[2*n:])    # W_q

    def get_alpha(self):
        """Get the current alpha value."""
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit)
        else:
            return self.alpha

    def forward(
        self,
        x: torch.Tensor,
        S: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial state matrix

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final state matrix
        """
        T, B, D = x.shape
        n = self.n_state
        alpha = self.get_alpha()

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32):
            x = x.contiguous()
            S = S.contiguous()
            alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            return E74FixedDecayCUDAFunction.apply(x, S, self.W_kvq, alpha_val, self.training)

        # Python fallback
        x_flat = x.reshape(T * B, D)
        all_proj = (x_flat @ self.W_kvq.T).reshape(T, B, 3 * n)
        k_all = all_proj[:, :, :n]
        v_all = all_proj[:, :, n:2*n]
        q_all = all_proj[:, :, 2*n:]

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]

            # Normalize k
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

            # Delta rule update with fixed decay
            # S' = alpha * S + outer(v - S @ k_norm, k_norm)
            retrieved = torch.einsum('bij,bj->bi', S, k_norm)
            delta = v - retrieved
            S = alpha * S + torch.einsum('bi,bj->bij', delta, k_norm)

            # Output with self-gating
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E74FixedDecay(nn.Module):
    """
    E74: Fixed Decay Delta Rule - Full layer.

    The simplest architecture in the autopoietic ladder.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        alpha: float = 0.9,
        learnable_alpha: bool = False,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        use_cuda: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.use_conv = True
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )
        else:
            self.use_conv = False

        self.cell = E74FixedDecayCell(
            self.d_inner,
            n_state=n_state,
            alpha=alpha,
            learnable_alpha=learnable_alpha,
            use_cuda=use_cuda,
        )

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, dim] input sequence
            hidden: Optional [B, n_state, n_state] initial state matrix

        Returns:
            output: [B, T, dim] output
            hidden: [B, n_state, n_state] final state
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Optional conv
        if self.use_conv:
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.conv1d(x_proj)[:, :, :T]
            x_proj = x_proj.transpose(1, 2)

        # Apply SiLU activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_proj = x_proj.transpose(0, 1)

        # Run cell
        cell_out, S = self.cell(x_proj, hidden)

        # Transpose back: [B, T, n_state]
        cell_out = cell_out.transpose(0, 1)

        # Output projection
        output = self.out_proj(cell_out)
        output = self.dropout(output)

        return output, S

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        alpha = self.cell.get_alpha()
        alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, alpha={alpha_val:.3f}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32

    model = E74FixedDecay(dim=D, n_state=n_state, expansion=1.0).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E74 Fixed Decay params: {model.get_num_params():,}")
    print(f"E74 Fixed Decay CUDA available: {E74_FIXED_DECAY_CUDA_AVAILABLE}")
    print(model)

    # Forward
    out, S = model(x)
    print(f"Output shape: {out.shape}")
    print(f"S shape: {S.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")
