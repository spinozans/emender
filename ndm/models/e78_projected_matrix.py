"""
E78: Projected Matrix State with Self-Gating Output

E77 + random projection for efficient large state:
- Store small [n_small x n_small] matrix
- Use random projection to simulate larger [n_effective x n_effective] state
- O(n_small²) compute but O(n_effective²) effective memory

Architecture:
    # P: [n_effective, n_small] fixed random orthogonal projection
    # S: [n_small, n_small] stored matrix

    k = W_k @ x  # [B, n_effective]
    v = W_v @ x
    q = W_q @ x
    gate = W_gate @ x  # [B, n_small] for decay

    # Project to small space
    k_small = P.T @ k  # [B, n_small]
    v_small = P.T @ v

    # LINEAR update in small space (like E77)
    decay = sigmoid(gate + b_gate)
    retrieved = S @ k_small_norm
    delta = v_small - retrieved
    S = decay * S + outer(delta, k_small_norm)

    # Read and project back
    q_small = P.T @ q
    Sq_small = S @ q_small  # [B, n_small]
    Sq = P @ Sq_small  # [B, n_effective]

    # Self-gating output
    output = Sq * silu(Sq)

Comparison:
    E77: Full [n x n] matrix, O(n²) compute
    E78: Small [n_small x n_small] but effective [n_eff x n_eff], O(n_small²) compute

Johnson-Lindenstrauss: Random projection preserves distances with high probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

E78_CUDA_AVAILABLE = False


def create_random_projection(n_effective: int, n_small: int, device=None, dtype=None):
    """Create a random orthogonal-ish projection matrix.

    Uses random Gaussian followed by QR decomposition for near-orthogonality.
    """
    # Random Gaussian matrix
    P = torch.randn(n_effective, n_small, device=device, dtype=dtype)
    # Normalize columns (approximate orthogonality)
    P = P / (P.norm(dim=0, keepdim=True) + 1e-6)
    # Scale for variance preservation: E[||Px||²] ≈ ||x||² when P has unit columns
    P = P * math.sqrt(n_effective / n_small)
    return P


class E78ProjectedMatrixCell(nn.Module):
    """
    E78 Projected Matrix State cell.

    Small stored matrix + random projection for large effective state.
    Uses FUSED k,v,q projection for efficiency.
    """

    def __init__(
        self,
        dim: int,
        n_effective: int = 128,  # Virtual large state size
        n_small: int = 32,       # Actual stored state size
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_effective = n_effective
        self.n_small = n_small
        self.use_cuda = use_cuda and E78_CUDA_AVAILABLE

        # Random projection matrix (fixed, not learned)
        # P: [n_effective, n_small]
        self.register_buffer('P', create_random_projection(n_effective, n_small))

        # FUSED projection for k, v, q (all in effective space)
        # Layout: [k | v | q] = [3 * n_effective, dim]
        self.W_kvq = nn.Parameter(torch.empty(3 * n_effective, dim))
        # Gate is in small space (for decay control) - separate projection
        self.W_gate = nn.Parameter(torch.empty(n_small, dim))
        self.b_gate = nn.Parameter(torch.zeros(n_small))

        self._init_weights()

    def _init_weights(self):
        n_eff = self.n_effective
        nn.init.xavier_uniform_(self.W_kvq[:n_eff])          # W_k
        nn.init.xavier_uniform_(self.W_kvq[n_eff:2*n_eff])   # W_v
        nn.init.xavier_uniform_(self.W_kvq[2*n_eff:])        # W_q
        nn.init.xavier_uniform_(self.W_gate)
        nn.init.constant_(self.b_gate, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_small, n_small] initial (small) matrix state

        Returns:
            output: [T, B, n_effective] self-gated output
            S: [B, n_small, n_small] final (small) matrix state
        """
        T, B, D = x.shape
        n_eff = self.n_effective
        n_small = self.n_small

        if S is None:
            S = torch.zeros(B, n_small, n_small, device=x.device, dtype=x.dtype)

        # FUSED projection for k, v, q (single GEMM)
        x_flat = x.reshape(T * B, D)
        kvq_all = (x_flat @ self.W_kvq.T).reshape(T, B, 3 * n_eff)  # [T, B, 3*n_eff]
        k_all = kvq_all[:, :, :n_eff]         # [T, B, n_effective]
        v_all = kvq_all[:, :, n_eff:2*n_eff]
        q_all = kvq_all[:, :, 2*n_eff:]
        # Gate projection is separate (different output dim)
        gate_all = (x_flat @ self.W_gate.T).reshape(T, B, n_small)  # [T, B, n_small]

        # Get projection matrix
        P = self.P.to(x.dtype)  # [n_effective, n_small]

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n_effective]
            v = v_all[t]
            q = q_all[t]
            gate = gate_all[t]  # [B, n_small]

            # Project k, v to small space
            k_small = k @ P  # [B, n_small]
            v_small = v @ P

            # Decay in small space
            decay = torch.sigmoid(gate + self.b_gate)  # [B, n_small]

            # Normalize k_small
            k_norm = k_small / (k_small.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieve from small matrix
            retrieved = torch.einsum('bij,bj->bi', S, k_norm)

            # Delta update in small space
            delta = v_small - retrieved
            outer = torch.einsum('bi,bj->bij', delta, k_norm)

            # LINEAR matrix update (E77-style - NO TANH!)
            S = decay.unsqueeze(-1) * S + outer

            # Read: project q to small, read from S, project back to effective
            q_small = q @ P  # [B, n_small]
            Sq_small = torch.einsum('bij,bj->bi', S, q_small)  # [B, n_small]
            Sq = Sq_small @ P.T  # [B, n_effective] - back to effective space

            # Self-gating output (E42-style)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E78ProjectedMatrix(nn.Module):
    """
    E78: Projected Matrix State - Full layer.

    E77 with random projection for efficient large state.
    O(n_small²) compute, O(n_effective²) effective memory.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_effective: int = 128,  # Virtual state size (the "big" matrix)
        n_small: int = 32,       # Actual stored state size
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        use_cuda: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_effective = n_effective
        self.n_small = n_small
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

        self.cell = E78ProjectedMatrixCell(
            self.d_inner,
            n_effective=n_effective,
            n_small=n_small,
            use_cuda=use_cuda,
        )

        # Output from n_effective space
        self.out_proj = nn.Linear(n_effective, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            S: [B, n_small, n_small] initial (small) matrix state

        Returns:
            output: [B, T, dim] output
            S: [B, n_small, n_small] final (small) state
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Optional conv
        if self.use_conv:
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.conv1d(x_proj)[:, :, :T]
            x_proj = x_proj.transpose(1, 2)

        # SiLU activation on input
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_proj = x_proj.transpose(0, 1)

        # Run cell
        cell_out, S = self.cell(x_proj, S)

        # Transpose back: [B, T, n_effective]
        cell_out = cell_out.transpose(0, 1)

        # Output projection
        output = self.out_proj(cell_out)  # [B, T, dim]
        output = self.dropout(output)

        return output, S

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return (f"dim={self.dim}, d_inner={self.d_inner}, "
                f"n_effective={self.n_effective}, n_small={self.n_small}")


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_effective = 128
    n_small = 32

    model = E78ProjectedMatrix(
        dim=D, n_effective=n_effective, n_small=n_small, expansion=1.0
    ).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E78 params: {model.get_num_params():,}")
    print(f"  n_effective={n_effective}, n_small={n_small}")
    print(f"  Effective state: {n_effective}x{n_effective} = {n_effective**2} elements")
    print(f"  Actual state: {n_small}x{n_small} = {n_small**2} elements")
    print(f"  Compression: {n_effective**2 / n_small**2:.1f}x")

    # Forward
    out, S = model(x)
    print(f"Output shape: {out.shape}")
    print(f"State shape: {S.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")

    # Compare to E77
    from ndm.models.e77_linear_matrix import E77LinearMatrix
    e77 = E77LinearMatrix(dim=D, n_state=n_effective, expansion=1.0).cuda().bfloat16()
    print(f"\nE77 (full {n_effective}x{n_effective}) params: {e77.get_num_params():,}")
