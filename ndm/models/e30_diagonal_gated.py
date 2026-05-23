"""
E30: E1 + SSM-style diagonal gating

Like E1 but with learned element-wise scales on the gate:
    gate = silu(z * g_z + h * g_h + b_gate)
    output = h * gate

This adds input-dependent selectivity (like Mamba2's dt projection)
without the overhead of tape memory (E29c).

Extra params: 3*d_inner (g_z, g_h, b_gate) - negligible vs E1's params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import hasty_pytorch_lib
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


class E30DiagonalGatedCUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E30 CUDA kernel."""

    @staticmethod
    def forward(ctx, x, z, h0, W_x, W_h, b, g_z, g_h, b_gate):
        """
        Args:
            x: [T, B, dim] pre-activated input
            z: [T, B, dim] gate input
            h0: [B, dim] initial hidden state
            W_x, W_h: [dim, dim] weight matrices
            b: [dim] bias
            g_z, g_h, b_gate: [dim] diagonal gate params
        """
        h, output, v, gate_input_cache = hasty_pytorch_lib.e30_diagonal_gated_forward(
            True,  # training
            x.contiguous(),
            z.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            b.contiguous(),
            g_z.contiguous(),
            g_h.contiguous(),
            b_gate.contiguous()
        )
        ctx.save_for_backward(W_x, W_h, g_z, g_h, x, z, h, v, gate_input_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, d_h):
        W_x, W_h, g_z, g_h, x, z, h, v, gate_input_cache = ctx.saved_tensors

        dx, dz, dW_x, dW_h, db, dg_z, dg_h, db_gate = hasty_pytorch_lib.e30_diagonal_gated_backward(
            W_x, W_h, g_z, g_h, x, z, h, v, gate_input_cache, d_output.contiguous()
        )

        return dx, dz, None, dW_x, dW_h, db, dg_z, dg_h, db_gate


class E30DiagonalGatedCell(nn.Module):
    """
    E1 cell with diagonal gating.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    gate = silu(z * g_z + h * g_h + b_gate)
    output = h_t * gate
    """

    def __init__(self, dim, mamba2_init=False):
        super().__init__()
        self.dim = dim

        # RNN weights (same as E1)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Diagonal gate weights (NEW)
        # Initialize g_h to 0.0 so gate starts like E1 (gate = silu(z * g_z + b))
        # Then learns to use h if beneficial
        self.g_z = nn.Parameter(torch.full((dim,), 1.0))  # z gate scale (1.0 = same as E1)
        self.g_h = nn.Parameter(torch.zeros(dim))          # h gate scale (0.0 = disabled at init)
        self.b_gate = nn.Parameter(torch.zeros(dim))       # gate bias

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.W_x, std=0.02)
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
        else:
            nn.init.xavier_uniform_(self.W_x)
            nn.init.orthogonal_(self.W_h)
            self.W_h.data.mul_(0.9)

    def forward(self, x, z, h0=None, use_cuda=True):
        """
        Args:
            x: [T, B, dim] pre-activated input
            z: [T, B, dim] gate input
            h0: [B, dim] initial hidden state
            use_cuda: Use CUDA kernel (default True)

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available and requested
        if use_cuda and HAS_CUDA and x.is_cuda and x.dtype == torch.bfloat16:
            return E30DiagonalGatedCUDAFunction.apply(
                x, z, h0, self.W_x, self.W_h, self.b, self.g_z, self.g_h, self.b_gate
            )

        # Python fallback
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]
            z_t = z[t]

            # Elman recurrence
            raw = x_t @ self.W_x.T + h_prev @ self.W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Diagonal gating: gate = silu(z * g_z + h * g_h + b_gate)
            gate_input = z_t * self.g_z + h_new * self.g_h + self.b_gate
            gate = F.silu(gate_input)

            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E30DiagonalGated(nn.Module):
    """
    E30: E1 with diagonal gating.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h = elman_cell(x)           # RNN
        gate = silu(z * g_z + h * g_h + b_gate)  # Diagonal gating
        output = out_proj(h * gate)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Elman cell with diagonal gating
        self.cell = E30DiagonalGatedCell(self.d_inner, mamba2_init=mamba2_init)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, use_cuda=True, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state
            use_cuda: Use CUDA kernel (default True)

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_inner] final hidden state
        """
        B, T, D = x.shape

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run cell with diagonal gating
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0, use_cuda=use_cuda)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final


if __name__ == '__main__':
    # Quick test
    B, T, D = 2, 16, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    layer = E30DiagonalGated(dim=D, expansion=1.0).to(device).to(dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    output, h_final = layer(x)
    print(f"Output shape: {output.shape}")
    print(f"h_final shape: {h_final.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test backward
    loss = output.sum()
    loss.backward()
    print(f"g_z grad norm: {layer.cell.g_z.grad.norm():.4f}")
    print(f"g_h grad norm: {layer.cell.g_h.grad.norm():.4f}")
    print("PASS")
