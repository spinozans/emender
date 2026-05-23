"""
E86: Input-as-Matrix Delta Rule (Multi-Head)

Combines E85's input-as-matrix insight with E75's delta rule.
Supports multiple heads for capacity scaling.

Key insight: derive k, v, q, beta from input matrix A without learned projections.
All matrix operations are n_state x n_state -> fits in shared memory.

Architecture (per head h):
    A_h = x_h.view(n_state, n_state)    # Each head's input matrix

    # Derive k, v, q from input matrix (no learned weights!)
    k_h = A_h.mean(dim=1)               # [n_state] row means as key
    v_h = A_h.mean(dim=0)               # [n_state] col means as value
    q_h = k_h                           # [n_state] query = key
    beta_h = sigmoid(scale * A_h.mean() + bias)  # scalar forget gate

    # E75's delta rule (proven to work well)
    k_norm_h = k_h / ||k_h||
    retrieved_h = S_h @ k_norm_h
    delta_h = v_h - retrieved_h
    S_h = tanh(beta_h * S_h + outer(delta_h, k_norm_h))

    # Self-gating output
    Sq_h = S_h @ q_h
    out_h = Sq_h * silu(Sq_h)           # [n_state]

Combined output: concat(out_0, ..., out_{H-1}) -> [n_heads * n_state]

Properties:
- Input-as-matrix: cell_dim = n_heads * n_state^2
- No learned projections in recurrence (only scale, bias shared across heads)
- Delta rule provides associative memory semantics
- Self-gating output like E68/E75
- All per-head ops fit in shared memory
- Multi-head for capacity scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel
E86_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e86_input_matrix_delta_forward'):
        E86_CUDA_AVAILABLE = True
except ImportError:
    pass


class E86CUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E86 autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, scale, bias, n_heads):
        results = hasty_pytorch_lib.e86_input_matrix_delta_forward(
            training, x, S0, scale, bias, n_heads
        )
        S = results[0]
        output = results[1]
        k_cache = results[2]
        v_cache = results[3]
        beta_cache = results[4]
        S_checkpoints = results[5]
        Sq_cache = results[6]

        ctx.save_for_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, beta_cache,
            scale, bias
        )
        ctx.n_heads = n_heads
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_checkpoints, Sq_cache,
         k_cache, v_cache, beta_cache,
         scale, bias) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e86_input_matrix_delta_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, beta_cache,
            d_output.contiguous(),
            scale, bias, ctx.n_heads
        )
        dx = grads[0]
        d_scale = grads[1]
        d_bias = grads[2]

        return None, dx, None, d_scale, d_bias, None


class E86InputMatrixDeltaCell(nn.Module):
    """
    E86 Input-as-Matrix Delta Rule cell (Multi-Head).

    Combines E85's input-as-matrix with E75's delta rule.
    cell_dim = n_heads * n_state^2, derives k/v/q/beta from input matrices.

    Args:
        n_state: Per-head state matrix dimension (n_state x n_state matrix)
        n_heads: Number of parallel heads (default 1)
        init_scale: Initial scale for beta computation (default 1.0)
        init_bias: Initial bias for beta computation (default 2.0, preserving)
        use_cuda: Whether to use CUDA kernel if available
    """

    def __init__(
        self,
        n_state: int = 32,
        n_heads: int = 1,
        init_scale: float = 1.0,
        init_bias: float = 2.0,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_heads = n_heads
        self.dim = n_heads * n_state * n_state  # cell input dim
        self.out_dim = n_heads * n_state        # cell output dim
        self.use_cuda = use_cuda and E86_CUDA_AVAILABLE

        # Learnable parameters for beta = sigmoid(scale * A.mean() + bias)
        # Shared across all heads
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(
        self,
        x: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        use_cuda: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [T, B, n_heads * n_state^2] input
            S: [B, n_heads, n_state, n_state] initial state matrices

        Returns:
            output: [T, B, n_heads * n_state] self-gated output
            S: [B, n_heads, n_state, n_state] final state matrices
        """
        T, B, D = x.shape
        n = self.n_state
        H = self.n_heads

        assert D == self.dim, f"Input dim {D} != expected {self.dim} (n_heads * n_state^2 = {H} * {n}^2)"

        if S is None:
            S = torch.zeros(B, H, n, n, device=x.device, dtype=x.dtype)

        _use_cuda = use_cuda if use_cuda is not None else self.use_cuda

        if _use_cuda and E86_CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
            S_flat = S.view(B, H * n * n)
            S_final, output = E86CUDAFunction.apply(
                self.training, x, S_flat, self.scale, self.bias, H
            )
            S_final = S_final.view(B, H, n, n)
            return output, S_final

        # Python fallback
        return self._forward_python(x, S)

    def _forward_python(
        self,
        x: torch.Tensor,
        S: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure Python implementation - reference for CUDA kernel."""
        T, B, _ = x.shape
        n = self.n_state
        H = self.n_heads

        # Split S into per-head states to avoid in-place modifications
        # S: [B, H, n, n] -> list of H tensors each [B, n, n]
        S_heads = [S[:, h].clone() for h in range(H)]

        outputs = []

        for t in range(T):
            # Reshape input to per-head matrices: [B, H, n, n]
            A = x[t].view(B, H, n, n)

            head_outputs = []
            for h in range(H):
                A_h = A[:, h]  # [B, n, n]
                S_h = S_heads[h]  # [B, n, n]

                # Derive k, v from input matrix (no learned weights)
                k = A_h.mean(dim=2)  # [B, n] row means as key
                v = A_h.mean(dim=1)  # [B, n] col means as value

                # Compute beta from matrix mean
                A_mean = A_h.mean(dim=(1, 2))  # [B]
                beta = torch.sigmoid(self.scale * A_mean + self.bias)  # [B]

                # Normalize k
                k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)  # [B, n]

                # Retrieved from memory
                retrieved = torch.einsum('bij,bj->bi', S_h, k_norm)  # [B, n]

                # Delta update with forget gate
                delta = v - retrieved  # [B, n]
                outer = torch.einsum('bi,bj->bij', delta, k_norm)  # [B, n, n]

                # Gated update: S = tanh(beta * S + outer)
                S_h = torch.tanh(beta.unsqueeze(-1).unsqueeze(-1) * S_h + outer)
                S_heads[h] = S_h  # Update list (not in-place on tensor)

                # Self-gating output (q = k_norm)
                Sq = torch.einsum('bij,bj->bi', S_h, k_norm)  # [B, n]
                out_h = Sq * F.silu(Sq)  # [B, n]
                head_outputs.append(out_h)

            # Concatenate head outputs
            out = torch.cat(head_outputs, dim=-1)  # [B, H * n]
            outputs.append(out)

        output = torch.stack(outputs, dim=0)  # [T, B, H * n]
        # Stack head states back into [B, H, n, n]
        S_final = torch.stack(S_heads, dim=1)
        return output, S_final


class E86InputMatrixDelta(nn.Module):
    """
    E86: Input-as-Matrix Delta Rule - Full layer with in/out projections.

    Combines E85's input-as-matrix insight with E75's delta rule.
    Supports multiple heads for capacity scaling.

    Input dim can be any size, projected to n_heads * n_state^2.
    Output is n_heads * n_state, projected back to model dim.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 32,
        n_heads: int = 1,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        init_scale: float = 1.0,
        init_bias: float = 2.0,
        use_cuda: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        self.cell_dim = n_heads * n_state * n_state  # Input to cell
        self.cell_out_dim = n_heads * n_state        # Output from cell
        self.use_conv = use_conv

        # Project to cell dimension (n_heads * n_state^2)
        self.in_proj = nn.Linear(dim, self.cell_dim, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.cell_dim,
                out_channels=self.cell_dim,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.cell_dim,
                bias=True,
            )

        self.cell = E86InputMatrixDeltaCell(
            n_state=n_state,
            n_heads=n_heads,
            init_scale=init_scale,
            init_bias=init_bias,
            use_cuda=use_cuda,
        )

        # Project from n_heads * n_state to output dim
        self.out_proj = nn.Linear(self.cell_out_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            S: [B, n_heads, n_state, n_state] initial matrix states

        Returns:
            output: [B, T, dim] output sequence
            S: [B, n_heads, n_state, n_state] final states
        """
        B, T, D = x.shape

        # Project to cell dimension
        x_proj = self.in_proj(x)  # [B, T, n_heads * n_state^2]

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()  # [T, B, n_heads * n_state^2]

        cell_out, S_final = self.cell(x_rnn, S)  # [T, B, n_heads * n_state]

        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, n_heads * n_state]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)  # [B, T, dim]

        return output, S_final

    def extra_repr(self):
        return (f'dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, '
                f'cell_dim={self.cell_dim}, cell_out={self.cell_out_dim}, '
                f'LEVEL=86_INPUT_MATRIX_DELTA')


# Alias for ladder registration
E86InputMatrixDeltaLayer = E86InputMatrixDelta


if __name__ == "__main__":
    print("Testing E86 (Input-as-Matrix Delta Rule, Multi-Head)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E86_CUDA_AVAILABLE}")

    # Test single head
    print(f"\n--- Single Head Test (n_heads=1, n_state=32) ---")
    model = E86InputMatrixDelta(
        dim=512, n_state=32, n_heads=1, use_cuda=False
    ).to(device).to(dtype)

    x = torch.randn(2, 32, 512, device=device, dtype=dtype)
    out, S = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}, State: {S.shape}")

    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test multi-head
    print(f"\n--- Multi-Head Test (n_heads=4, n_state=32) ---")
    model_mh = E86InputMatrixDelta(
        dim=512, n_state=32, n_heads=4, use_cuda=False
    ).to(device).to(dtype)

    x = torch.randn(2, 32, 512, device=device, dtype=dtype)
    out, S = model_mh(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}, State: {S.shape}")
    print(f"  cell_dim={model_mh.cell_dim}, cell_out={model_mh.cell_out_dim}")

    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model_mh.parameters())
    print(f"Parameters: {params:,}")

    # Test smaller state with more heads
    print(f"\n--- More Heads Test (n_heads=8, n_state=16) ---")
    model_8h = E86InputMatrixDelta(
        dim=512, n_state=16, n_heads=8, use_cuda=False
    ).to(device).to(dtype)

    x = torch.randn(2, 32, 512, device=device, dtype=dtype)
    out, S = model_8h(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}, State: {S.shape}")
    print(f"  cell_dim={model_8h.cell_dim}, cell_out={model_8h.cell_out_dim}")

    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model_8h.parameters())
    print(f"Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("All tests completed!")
