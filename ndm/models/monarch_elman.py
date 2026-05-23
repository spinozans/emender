"""
E7: Monarch Elman - O(n*sqrt(n)) hidden state updates via Monarch matrices

h_t = tanh(monarch(B1_h, B2_h) @ h_{t-1} + monarch(B1_x, B2_x) @ x_t + b)
output_t = h_t * silu(W_gate @ x_t + b_gate)

Monarch matrix multiplication (n = m^2, so sqrt(n) = m):
B1, B2 are block-diagonal: [m, m, m] (m blocks of m x m)
monarch(B1, B2) @ v:
  1. Reshape v: [B, n] -> [B, m, m]
  2. Block matmul: z = einsum('kij,bkj->bki', B1, v_reshaped)  # [B, m, m]
  3. Transpose: z = z.transpose(-1, -2)  # This is the "permutation"
  4. Block matmul: out = einsum('kij,bkj->bki', B2, z)  # [B, m, m]
  5. Flatten: [B, m, m] -> [B, n]

Key: O(n*sqrt(n)) = O(m^3) complexity instead of O(n^2) = O(m^4) for dense matmul
For n=4096, m=64: O(262144) vs O(16777216) = 64x faster theoretically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    MONARCH_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'monarch_elman_forward')
except ImportError:
    MONARCH_CUDA_AVAILABLE = False


def monarch_matmul(B1, B2, v):
    """
    Monarch matrix multiplication: monarch(B1, B2) @ v

    Args:
        B1: [m, m, m] first set of block-diagonal matrices (m blocks of m x m)
        B2: [m, m, m] second set of block-diagonal matrices
        v: [B, n] input vectors where n = m * m

    Returns:
        out: [B, n] = monarch(B1, B2) @ v
    """
    m = B1.shape[0]
    B = v.shape[0]
    n = m * m

    # Step 1: Reshape v to [B, m, m]
    v_reshaped = v.view(B, m, m)

    # Step 2: Block matmul: z = einsum('kij,bkj->bki', B1, v_reshaped)
    # For each block k: z[:, k, :] = v[:, k, :] @ B1[k].T
    z = torch.einsum('kij,bkj->bki', B1, v_reshaped)

    # Step 3: Transpose (this is the permutation)
    z = z.transpose(-1, -2)  # [B, m, m] with swapped last two dims

    # Step 4: Block matmul: out = einsum('kij,bkj->bki', B2, z)
    out = torch.einsum('kij,bkj->bki', B2, z)

    # Step 5: Flatten back to [B, n]
    return out.reshape(B, n)


def monarch_matmul_backward(B1, B2, v, d_out):
    """
    Backward pass for monarch matmul.

    Returns:
        d_v: [B, n] gradient w.r.t. input
        d_B1: [m, m, m] gradient w.r.t. B1
        d_B2: [m, m, m] gradient w.r.t. B2
    """
    m = B1.shape[0]
    B = v.shape[0]
    n = m * m

    # Recompute forward intermediates
    v_reshaped = v.view(B, m, m)
    z1 = torch.einsum('kij,bkj->bki', B1, v_reshaped)  # After first block matmul
    z2 = z1.transpose(-1, -2)  # After transpose (permutation)

    # Reshape d_out
    d_out_reshaped = d_out.view(B, m, m)

    # Backward through second block matmul
    # out = einsum('kij,bkj->bki', B2, z2)
    # d_z2 = einsum('kij,bki->bkj', B2, d_out)  (B2 transposed)
    # d_B2 = einsum('bki,bkj->kij', d_out, z2)
    d_z2 = torch.einsum('kji,bki->bkj', B2, d_out_reshaped)
    d_B2 = torch.einsum('bki,bkj->kij', d_out_reshaped, z2)

    # Backward through transpose
    d_z1 = d_z2.transpose(-1, -2)

    # Backward through first block matmul
    # z1 = einsum('kij,bkj->bki', B1, v_reshaped)
    # d_v_reshaped = einsum('kij,bki->bkj', B1, d_z1)  (B1 transposed)
    # d_B1 = einsum('bki,bkj->kij', d_z1, v_reshaped)
    d_v_reshaped = torch.einsum('kji,bki->bkj', B1, d_z1)
    d_B1 = torch.einsum('bki,bkj->kij', d_z1, v_reshaped)

    d_v = d_v_reshaped.reshape(B, n)

    return d_v, d_B1, d_B2


class MonarchElmanFunction(torch.autograd.Function):
    """Autograd function for Monarch Elman with CUDA acceleration."""

    @staticmethod
    def forward(ctx, training, x, h0, B1_h, B2_h, B1_x, B2_x, W_gate, b, b_gate):
        h, output, v, gate_cache = hasty_pytorch_lib.monarch_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            B1_h.contiguous(),
            B2_h.contiguous(),
            B1_x.contiguous(),
            B2_x.contiguous(),
            W_gate.contiguous(),
            b.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, B1_h, B2_h, B1_x, B2_x, W_gate, b_gate, h, v, gate_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, B1_h, B2_h, B1_x, B2_x, W_gate, b_gate, h, v, gate_cache = ctx.saved_tensors
        dx, dB1_h, dB2_h, dB1_x, dB2_x, dW_gate, db, d_b_gate = hasty_pytorch_lib.monarch_elman_backward(
            B1_h, B2_h, B1_x, B2_x, W_gate, x, h, v, gate_cache, d_output.contiguous()
        )
        return None, dx, None, dB1_h, dB2_h, dB1_x, dB2_x, dW_gate, db, d_b_gate


class MonarchElmanCell(nn.Module):
    """
    E7 Monarch Elman cell - O(n*sqrt(n)) hidden state updates.

    h_t = tanh(monarch(B1_h, B2_h) @ h_{t-1} + monarch(B1_x, B2_x) @ x_t + b)
    output_t = h_t * silu(W_gate @ x_t + b_gate)

    Args:
        dim: Hidden dimension (must be a perfect square, e.g., 4096 = 64^2)
        m: Block size (sqrt(dim)). If None, computed automatically.
    """

    def __init__(self, dim, m=None):
        super().__init__()
        self.dim = dim

        # Compute m = sqrt(dim), must be integer
        if m is None:
            m = int(math.sqrt(dim))
        if m * m != dim:
            raise ValueError(f"dim must be a perfect square, got {dim} (sqrt = {math.sqrt(dim)})")
        self.m = m

        # Monarch matrices for hidden state: [m, m, m] each (m blocks of m x m)
        self.B1_h = nn.Parameter(torch.empty(m, m, m))
        self.B2_h = nn.Parameter(torch.empty(m, m, m))

        # Monarch matrices for input: [m, m, m] each
        self.B1_x = nn.Parameter(torch.empty(m, m, m))
        self.B2_x = nn.Parameter(torch.empty(m, m, m))

        # Gate projection (full dense for flexibility)
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        # Bias
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Initialize monarch matrices to approximate identity transform
        # Each block initialized as scaled identity + small noise
        for B in [self.B1_h, self.B2_h, self.B1_x, self.B2_x]:
            # Initialize each m x m block
            with torch.no_grad():
                nn.init.orthogonal_(B.view(self.m * self.m, self.m))
                B.data = B.data.view(self.m, self.m, self.m)
                # Scale down to prevent explosion
                B.data *= 0.5

        nn.init.xavier_uniform_(self.W_gate)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if MONARCH_CUDA_AVAILABLE and x.is_cuda:
            return MonarchElmanFunction.apply(
                self.training, x, h0,
                self.B1_h, self.B2_h, self.B1_x, self.B2_x,
                self.W_gate, self.b, self.b_gate
            )

        # PyTorch fallback
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation."""
        T, B, D = x.shape
        h_list = [h0]
        output_list = []

        # Pre-compute gate projections for all timesteps
        # gate_proj: [T, B, dim]
        gate_proj = torch.einsum('tbd,od->tbo', x, self.W_gate)

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Monarch matmuls
            Mh = monarch_matmul(self.B1_h, self.B2_h, h_prev)  # [B, dim]
            Mx = monarch_matmul(self.B1_x, self.B2_x, x_t)     # [B, dim]

            # h_t = tanh(Mh + Mx + b)
            raw = Mh + Mx + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # output = h * silu(gate_proj + b_gate)
            gate = F.silu(gate_proj[t] + self.b_gate)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class MonarchElman(nn.Module):
    """
    E7: Monarch Elman layer - O(n*sqrt(n)) hidden state updates.

    Wraps MonarchElmanCell with input/output projections for use in LM.

    For dim=4096 (m=64):
    - Dense matmul: O(4096^2) = 16.7M ops per timestep
    - Monarch matmul: O(64^3) = 262K ops per timestep (64x faster)

    Args:
        dim: Model dimension
        d_inner: Hidden dimension (must be perfect square). Defaults to dim if dim is perfect square.
        dropout: Dropout probability
        use_conv: Whether to use conv1d for local context
        d_conv: Conv kernel size
    """

    def __init__(
        self,
        dim,
        d_inner=None,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        # Default d_inner to a suitable perfect square
        if d_inner is None:
            # Try to use dim if it's a perfect square
            m = int(math.sqrt(dim))
            if m * m == dim:
                d_inner = dim
            else:
                # Find nearest perfect square >= dim
                d_inner = (m + 1) ** 2
        self.d_inner = d_inner

        # Verify d_inner is a perfect square
        m = int(math.sqrt(d_inner))
        if m * m != d_inner:
            raise ValueError(f"d_inner must be a perfect square, got {d_inner}")
        self.m = m

        # Input projection
        self.in_proj = nn.Linear(dim, d_inner, bias=False)

        # Optional conv1d for local context
        self.use_conv = use_conv
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=d_inner,
                out_channels=d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=d_inner,
                bias=True,
            )

        # Monarch Elman cell
        self.cell = MonarchElmanCell(d_inner, m=m)

        # Output projection
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_inner] final hidden state
        """
        B, T, D = x.shape

        # Project input
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal
            x_proj = F.silu(x_conv.transpose(1, 2))

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, d_inner]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, m={self.m}, LEVEL=7_MONARCH'


if __name__ == "__main__":
    print("Testing MonarchElman (E7)...")
    print("=" * 60)
    print(f"CUDA kernel available: {MONARCH_CUDA_AVAILABLE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    # Test with default settings (dim=4096 = 64^2)
    dim = 512
    d_inner = 4096  # 64^2

    print(f"\nTesting with dim={dim}, d_inner={d_inner} (m={int(math.sqrt(d_inner))})")

    model = MonarchElman(dim=dim, d_inner=d_inner).to(device).to(dtype)
    x = torch.randn(2, 32, dim, device=device, dtype=dtype)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    cell_params = sum(p.numel() for p in model.cell.parameters())
    print(f"\nTotal parameters: {params:,}")
    print(f"Cell parameters: {cell_params:,}")

    # Compare with theoretical dense parameters
    dense_hidden_params = d_inner * d_inner  # W_h for dense
    monarch_hidden_params = 4 * (int(math.sqrt(d_inner)) ** 3)  # B1_h, B2_h, B1_x, B2_x
    print(f"\nDense W_h would be: {dense_hidden_params:,} params")
    print(f"Monarch B1/B2 total: {monarch_hidden_params:,} params ({monarch_hidden_params/dense_hidden_params*100:.1f}%)")

    # Test monarch_matmul directly
    print("\n" + "=" * 60)
    print("Testing monarch_matmul directly...")
    m = 64
    n = m * m
    B1 = torch.randn(m, m, m, device=device, dtype=dtype)
    B2 = torch.randn(m, m, m, device=device, dtype=dtype)
    v = torch.randn(32, n, device=device, dtype=dtype)

    out = monarch_matmul(B1, B2, v)
    print(f"monarch_matmul: v {v.shape} -> out {out.shape}")

    print("\nE7 (Monarch Elman) test passed!")
