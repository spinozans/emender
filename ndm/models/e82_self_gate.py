"""
E82: Self-Gating Matrix - Pure Autopoiesis

A single matrix S gates itself - maximum autopoiesis where the memory
controls its own forgetting without any external modulation matrix.

Mathematical definition:
    # Single matrix S gates itself
    # Use different key projections for gating vs content

    G = sigmoid(outer(S @ m_norm, k_norm) + alpha * S)  # Self-computed gate
    S' = G * S + outer(v - S @ k_norm, k_norm)

    output = (S' @ q) * silu(S' @ q)

Key insight: Minimal architecture - only ONE matrix that determines its own
forgetting. This is maximum autopoiesis.

Stabilization to avoid degenerate fixed points (all-forget or all-remember):
- Initialize carefully
- alpha should be small (0.1) to blend local info with global self-reference
- Optional epsilon skip connection: G = sigmoid(...) + epsilon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel
E82_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e82_self_gate_forward') and hasattr(hasty_pytorch_lib, 'e82_self_gate_backward'):
        E82_CUDA_AVAILABLE = True
except ImportError:
    pass


class E82CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E82 CUDA kernel."""

    @staticmethod
    def forward(ctx, x, S0, W_kvqm, alpha, epsilon, training):
        """
        Args:
            x: [T, B, dim] input
            S0: [B, n_state, n_state] initial memory
            W_kvqm: [4*n_state, dim] fused projection weights
            alpha: scalar, self-reference blending factor
            epsilon: scalar, skip connection for gate stability
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final memory
        """
        S, output, kvqm_cache, S_checkpoints, Sq_cache, gate_cache = \
            hasty_pytorch_lib.e82_self_gate_forward(training, x, S0, W_kvqm, alpha, epsilon)

        if training:
            ctx.save_for_backward(x, S_checkpoints, Sq_cache, kvqm_cache, gate_cache,
                                  W_kvqm, alpha, epsilon)

        return output, S

    @staticmethod
    def backward(ctx, d_output, d_S):
        x, S_checkpoints, Sq_cache, kvqm_cache, gate_cache, W_kvqm, alpha, epsilon = ctx.saved_tensors

        d_output = d_output.contiguous()

        dx, dW_kvqm, d_alpha = hasty_pytorch_lib.e82_self_gate_backward(
            x, S_checkpoints, Sq_cache, kvqm_cache, gate_cache,
            d_output, W_kvqm, alpha, epsilon
        )

        return dx, None, dW_kvqm, d_alpha, None, None


class E82SelfGateCell(nn.Module):
    """
    E82 Self-Gating Matrix cell.

    A single matrix state that gates itself - pure autopoiesis.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        alpha_init: float = 0.1,
        epsilon: float = 0.0,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.epsilon = epsilon
        self.use_cuda = use_cuda and E82_CUDA_AVAILABLE

        # FUSED projection: single GEMM for k, v, q, m
        # Layout: [k | v | q | m] = [4 * n_state, dim]
        self.W_kvqm = nn.Parameter(torch.empty(4 * n_state, dim))

        # Learnable alpha for self-reference blending
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        nn.init.xavier_uniform_(self.W_kvqm[:n])      # W_k
        nn.init.xavier_uniform_(self.W_kvqm[n:2*n])   # W_v
        nn.init.xavier_uniform_(self.W_kvqm[2*n:3*n]) # W_q
        nn.init.xavier_uniform_(self.W_kvqm[3*n:])    # W_m

    def forward(
        self,
        x: torch.Tensor,
        S: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial memory

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final memory
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32):
            x = x.contiguous()
            S = S.contiguous()
            alpha_tensor = self.alpha.to(x.dtype)
            epsilon_tensor = torch.tensor(self.epsilon, device=x.device, dtype=x.dtype)
            return E82CUDAFunction.apply(x, S, self.W_kvqm, alpha_tensor, epsilon_tensor, self.training)

        # Python fallback
        x_flat = x.reshape(T * B, D)
        all_proj = (x_flat @ self.W_kvqm.T).reshape(T, B, 4 * n)
        k_all = all_proj[:, :, :n]
        v_all = all_proj[:, :, n:2*n]
        q_all = all_proj[:, :, 2*n:3*n]
        m_all = all_proj[:, :, 3*n:]

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            m_vec = m_all[t]

            # Normalize k and m
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

            # --- Self-gating ---
            # G = sigmoid(outer(S @ m_norm, k_norm) + alpha * S) + epsilon
            Sm = torch.einsum('bij,bj->bi', S, m_norm)  # [B, n]
            gate_logits = torch.einsum('bi,bj->bij', Sm, k_norm) + self.alpha * S  # [B, n, n]
            G = torch.sigmoid(gate_logits)
            if self.epsilon > 0:
                G = G + self.epsilon

            # Delta rule for S
            s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
            s_delta = v - s_retrieved

            # Update S: S' = G * S + outer(s_delta, k_norm)
            S = G * S + torch.einsum('bi,bj->bij', s_delta, k_norm)

            # --- Output ---
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E82SelfGate(nn.Module):
    """
    E82: Self-Gating Matrix - Pure Autopoiesis - Full layer.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        alpha_init: float = 0.1,
        epsilon: float = 0.0,
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
        self.use_conv = use_conv

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        self.cell = E82SelfGateCell(
            self.d_inner,
            n_state=n_state,
            alpha_init=alpha_init,
            epsilon=epsilon,
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
            hidden: Optional [B, n_state, n_state] initial memory

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
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, alpha={self.cell.alpha.item():.3f}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32

    model = E82SelfGate(dim=D, n_state=n_state, expansion=1.0).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E82 params: {model.get_num_params():,}")
    print(f"E82 CUDA available: {E82_CUDA_AVAILABLE}")

    # Forward
    out, S = model(x)
    print(f"Output shape: {out.shape}")
    print(f"S shape: {S.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")
