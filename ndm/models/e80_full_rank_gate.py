"""
E80: Full-Rank Mutual Gating

Extends E79 with full n×n gate matrices instead of rank-1 outer products.

Two coupled matrix states that mutually control each other's evolution:
- S [n_state x n_state]: Content memory - stores key-value associations
- M [n_state x n_state]: Modulation memory - controls how S updates

Key difference from E79: The gate is a full n×n matrix, not a rank-1 outer product.

Architecture:
    # Input projections
    k, v, q, m = W_kvqm @ x

    # M provides full-rank gate for S
    G_S = sigmoid(M + outer(M @ k_norm, k_norm) + B_S)  # Full n×n gate
    S' = G_S ⊙ S + outer(v - S @ k_norm, k_norm)

    # S provides full-rank gate for M
    G_M = sigmoid(S + outer(S @ m_norm, m_norm) + B_M)  # Full n×n gate
    M' = G_M ⊙ M + outer(delta_S - M @ m_norm, m_norm)

    output = (S' @ q) * silu(S' @ q)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel
E80_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e80_full_rank_gate_forward') and hasattr(hasty_pytorch_lib, 'e80_full_rank_gate_backward'):
        E80_CUDA_AVAILABLE = True
except ImportError:
    pass


class E80CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E80 CUDA kernel."""

    @staticmethod
    def forward(ctx, x, S0, M0, W_kvqm, B_S, B_M, training):
        """
        Args:
            x: [T, B, dim] input
            S0: [B, n_state, n_state] initial content memory
            M0: [B, n_state, n_state] initial modulation memory
            W_kvqm: [4*n_state, dim] fused projection weights
            B_S: [n_state, n_state] S gate bias matrix
            B_M: [n_state, n_state] M gate bias matrix
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final content memory
            M: [B, n_state, n_state] final modulation memory
        """
        S, M, output, kvqm_cache, S_checkpoints, M_checkpoints, Sq_cache, \
            G_S_cache, G_M_cache = \
            hasty_pytorch_lib.e80_full_rank_gate_forward(training, x, S0, M0, W_kvqm, B_S, B_M)

        if training:
            ctx.save_for_backward(x, S_checkpoints, M_checkpoints, Sq_cache,
                                  kvqm_cache, G_S_cache, G_M_cache,
                                  W_kvqm, B_S, B_M)

        return output, S, M

    @staticmethod
    def backward(ctx, d_output, d_S, d_M):
        x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache, \
            G_S_cache, G_M_cache, \
            W_kvqm, B_S, B_M = ctx.saved_tensors

        d_output = d_output.contiguous()

        dx, dW_kvqm, dB_S, dB_M = hasty_pytorch_lib.e80_full_rank_gate_backward(
            x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache,
            G_S_cache, G_M_cache,
            d_output, W_kvqm, B_S, B_M
        )

        return dx, None, None, dW_kvqm, dB_S, dB_M, None


class E80FullRankGateCell(nn.Module):
    """
    E80 Full-Rank Mutual Gating cell.

    Two coupled matrix states with full n×n gate matrices.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.use_cuda = use_cuda and E80_CUDA_AVAILABLE

        # FUSED projection: single GEMM for k, v, q, m
        # Layout: [k | v | q | m] = [4 * n_state, dim]
        self.W_kvqm = nn.Parameter(torch.empty(4 * n_state, dim))

        # Full n×n gate bias matrices for S and M
        self.B_S = nn.Parameter(torch.zeros(n_state, n_state))
        self.B_M = nn.Parameter(torch.zeros(n_state, n_state))

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        nn.init.xavier_uniform_(self.W_kvqm[:n])      # W_k
        nn.init.xavier_uniform_(self.W_kvqm[n:2*n])   # W_v
        nn.init.xavier_uniform_(self.W_kvqm[2*n:3*n]) # W_q
        nn.init.xavier_uniform_(self.W_kvqm[3*n:])    # W_m
        # Initialize gate biases for moderate decay (bias toward keeping state)
        nn.init.constant_(self.B_S, 2.0)  # sigmoid(2) ≈ 0.88
        nn.init.constant_(self.B_M, 2.5)  # M slightly slower decay

    def forward(
        self,
        x: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial content memory
            M: [B, n_state, n_state] initial modulation memory

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final content memory
            M: [B, n_state, n_state] final modulation memory
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)
        if M is None:
            # Initialize M with small values so it provides initial gating
            M = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32):
            x = x.contiguous()
            S = S.contiguous()
            M = M.contiguous()
            return E80CUDAFunction.apply(x, S, M, self.W_kvqm, self.B_S, self.B_M, self.training)

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

            # --- S update (M-controlled full-rank gating) ---
            # G_S = sigmoid(M + outer(M @ k_norm, k_norm) + B_S)
            M_k = torch.einsum('bij,bj->bi', M, k_norm)  # [B, n]
            gate_outer_S = torch.einsum('bi,bj->bij', M_k, k_norm)  # [B, n, n]
            G_S = torch.sigmoid(M + gate_outer_S + self.B_S)  # [B, n, n]

            # Delta rule for S
            s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
            s_delta = v - s_retrieved

            # Update S with full-rank gate
            S = G_S * S + torch.einsum('bi,bj->bij', s_delta, k_norm)

            # --- M update (S-controlled full-rank gating) ---
            # G_M = sigmoid(S + outer(S @ m_norm, m_norm) + B_M)
            S_m = torch.einsum('bij,bj->bi', S, m_norm)  # [B, n]
            gate_outer_M = torch.einsum('bi,bj->bij', S_m, m_norm)  # [B, n, n]
            G_M = torch.sigmoid(S + gate_outer_M + self.B_M)  # [B, n, n]

            # M tries to predict S's delta (learns meta-patterns)
            m_retrieved = torch.einsum('bij,bj->bi', M, m_norm)
            m_delta = s_delta - m_retrieved

            # Update M with full-rank gate
            M = G_M * M + torch.einsum('bi,bj->bij', m_delta, m_norm)

            # --- Output ---
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S, M


class E80FullRankGate(nn.Module):
    """
    E80: Full-Rank Mutual Gating - Full layer.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
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

        self.cell = E80FullRankGateCell(
            self.d_inner,
            n_state=n_state,
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
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input sequence
            hidden: Optional tuple of (S, M) where:
                S: [B, n_state, n_state] initial content memory
                M: [B, n_state, n_state] initial modulation memory

        Returns:
            output: [B, T, dim] output
            hidden: Tuple of (S, M) final states
        """
        B, T, D = x.shape

        # Unpack hidden state
        S, M = hidden if hidden is not None else (None, None)

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
        cell_out, S, M = self.cell(x_proj, S, M)

        # Transpose back: [B, T, n_state]
        cell_out = cell_out.transpose(0, 1)

        # Output projection
        output = self.out_proj(cell_out)
        output = self.dropout(output)

        return output, (S, M)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32

    model = E80FullRankGate(dim=D, n_state=n_state, expansion=1.0).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E80 params: {model.get_num_params():,}")
    print(f"E80 CUDA available: {E80_CUDA_AVAILABLE}")

    # Forward
    out, (S, M) = model(x)
    print(f"Output shape: {out.shape}")
    print(f"S shape: {S.shape}")
    print(f"M shape: {M.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")
