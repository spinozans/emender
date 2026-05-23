"""
E28: E1 + Mamba2's Exact Conv System

E1 (Gated Elman) with Mamba2's depthwise causal conv1d before the recurrence.

Architecture:
    x, z = split(in_proj(x))           # Split input
    x = causal_conv1d(x, conv_weight)  # Mamba2's depthwise conv (k=4)
    x = silu(x)                        # Pre-activation
    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  # Elman recurrence
    output = h * silu(z)               # Gate with z branch
    output = out_proj(output)

The conv is:
- Depthwise separable (groups=d_inner)
- Kernel size = 4 (Mamba2 default)
- Causal (left-padded, no future info)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E28_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e28_conv_forward')
except ImportError:
    E28_CUDA_AVAILABLE = False


class E28ConvFunction(torch.autograd.Function):
    """CUDA-accelerated E28 autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h_init, W_x, W_h, b, conv_weight, conv_bias):
        h_all, output = hasty_pytorch_lib.e28_conv_forward(
            training,
            x.contiguous(),
            z.contiguous(),
            h_init.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            b.contiguous(),
            conv_weight.contiguous(),
            conv_bias.contiguous()
        )
        ctx.save_for_backward(x, z, h_init, h_all, W_x, W_h, conv_weight, conv_bias)
        return h_all, output

    @staticmethod
    def backward(ctx, d_h_all, d_output):
        x, z, h_init, h_all, W_x, W_h, conv_weight, conv_bias = ctx.saved_tensors
        dx, dz, dW_x, dW_h, db, d_conv_weight, d_conv_bias = hasty_pytorch_lib.e28_conv_backward(
            x.contiguous(),
            z.contiguous(),
            h_init.contiguous(),
            h_all.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            conv_weight.contiguous(),
            conv_bias.contiguous(),
            d_output.contiguous()
        )
        return None, dx, dz, None, dW_x, dW_h, db, d_conv_weight, d_conv_bias


def causal_conv1d_python(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Causal depthwise conv1d - Python reference.

    Args:
        x: [B, T, D] input
        weight: [D, 1, K] or [D, K] conv weights (depthwise)
        bias: [D] conv bias

    Returns:
        [B, T, D] output (same shape, causal)
    """
    B, T, D = x.shape

    # Handle both [D, 1, K] and [D, K] shapes
    if weight.dim() == 2:
        weight = weight.unsqueeze(1)  # [D, K] -> [D, 1, K]
    K = weight.shape[2]  # kernel size

    # Transpose for conv1d: [B, D, T]
    x = x.transpose(1, 2)

    # Left-pad for causal: pad (K-1) on left, 0 on right
    x_padded = F.pad(x, (K - 1, 0))  # [B, D, T + K - 1]

    # Depthwise conv
    out = F.conv1d(x_padded, weight, bias, groups=D)  # [B, D, T]

    # Transpose back: [B, T, D]
    return out.transpose(1, 2)


def e28_forward_step_python(
    x_t: torch.Tensor,      # [B, D] - conv'd and silu'd input
    h_prev: torch.Tensor,   # [B, D] - previous hidden state
    W_x: torch.Tensor,      # [D, D]
    W_h: torch.Tensor,      # [D, D]
    b: torch.Tensor,        # [D]
) -> torch.Tensor:
    """Single E28 recurrence step (same as E1)."""
    pre_act = x_t @ W_x.T + h_prev @ W_h.T + b
    h_new = torch.tanh(pre_act)
    return h_new


def e28_forward_python(
    x: torch.Tensor,            # [B, T, D] input (after in_proj split)
    z: torch.Tensor,            # [B, T, D] gate input
    h_init: torch.Tensor,       # [B, D] initial hidden state
    W_x: torch.Tensor,          # [D, D]
    W_h: torch.Tensor,          # [D, D]
    b: torch.Tensor,            # [D]
    conv_weight: torch.Tensor,  # [D, 1, K] depthwise conv weights
    conv_bias: torch.Tensor,    # [D] conv bias
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E28 forward pass - Python reference.

    Returns:
        h_all: [B, T, D] - hidden states at each timestep
        output: [B, T, D] - gated output (h * silu(z))
        h_final: [B, D] - final hidden state
    """
    B, T, D = x.shape

    # Step 1: Causal conv1d
    x_conv = causal_conv1d_python(x, conv_weight, conv_bias)

    # Step 2: SiLU activation
    x_act = F.silu(x_conv)

    # Step 3: Elman recurrence
    h = h_init
    h_list = []

    for t in range(T):
        h = e28_forward_step_python(x_act[:, t], h, W_x, W_h, b)
        h_list.append(h)

    h_all = torch.stack(h_list, dim=1)  # [B, T, D]

    # Step 4: Gating with silu(z)
    output = h_all * F.silu(z)  # [B, T, D]

    return h_all, output, h_all[:, -1]


class E28ConvElmanCell(nn.Module):
    """E28 cell: E1 + Mamba2 conv."""

    def __init__(self, dim: int, d_conv: int = 4, w_h_init_scale: float = 0.9):
        super().__init__()
        self.dim = dim
        self.d_conv = d_conv

        # Conv weights (depthwise)
        self.conv_weight = nn.Parameter(torch.empty(dim, 1, d_conv))
        self.conv_bias = nn.Parameter(torch.zeros(dim))

        # Elman weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_scale)

    def _init_weights(self, w_h_init_scale: float):
        # Conv init: Match Mamba2 exactly (uniform[-0.5, 0.5], std~0.29)
        nn.init.uniform_(self.conv_weight, -0.5, 0.5)

        # Elman weights
        nn.init.xavier_uniform_(self.W_x)

        # W_h: orthogonal scaled for stable gradients
        W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_fp32)
        W_h_fp32.mul_(w_h_init_scale)
        with torch.no_grad():
            self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))

    def forward(
        self,
        x: torch.Tensor,        # [B, T, D] - after split from in_proj
        z: torch.Tensor,        # [B, T, D] - gate branch
        h_init: torch.Tensor = None,
        use_cuda: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_all: [B, T, D]
            output: [B, T, D]
            h_final: [B, D]
        """
        B, T, D = x.shape

        if h_init is None:
            h_init = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if use_cuda and E28_CUDA_AVAILABLE and x.is_cuda:
            h_all, output = E28ConvFunction.apply(
                self.training, x, z, h_init,
                self.W_x, self.W_h, self.b,
                self.conv_weight, self.conv_bias
            )
            return h_all, output, h_all[:, -1]

        # Python fallback (used for training)
        h_all, output, h_final = e28_forward_python(
            x, z, h_init, self.W_x, self.W_h, self.b,
            self.conv_weight, self.conv_bias
        )
        return h_all, output, h_final


class E28ConvElman(nn.Module):
    """
    E28: E1 + Mamba2's Conv System

    Full layer with input/output projections.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        d_conv: int = 4,
        dropout: float = 0.0,
        w_h_init_scale: float = 0.9,
        mamba2_init: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.d_conv = d_conv

        # Input projection: dim -> 2*d_inner (split into x and z)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # E28 cell
        self.cell = E28ConvElmanCell(self.d_inner, d_conv, w_h_init_scale)

        # Output projection: d_inner -> dim
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init: bool):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None, **kwargs):
        """
        Args:
            x: [B, T, dim] input
            h0: [B, d_inner] initial hidden state

        Returns:
            output: [B, T, dim]
            h_final: [B, d_inner]
        """
        B, T, D = x.shape

        # Project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # E28 cell (conv + silu + elman + gate)
        h_all, cell_out, h_final = self.cell(x_proj, z, h0)

        # Output projection
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, d_conv={self.d_conv}, LEVEL=28_CONV_ELMAN'


if __name__ == '__main__':
    print("Testing E28: E1 + Mamba2 Conv")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    B, T, D = 2, 16, 256
    d_conv = 4

    # Test causal conv
    print("\n1. Testing causal_conv1d_python...")
    x = torch.randn(B, T, D, device=device, dtype=dtype)
    weight = torch.randn(D, 1, d_conv, device=device, dtype=dtype) * 0.1
    bias = torch.zeros(D, device=device, dtype=dtype)

    out = causal_conv1d_python(x, weight, bias)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"

    # Verify causality: output at t should only depend on input at t-3, t-2, t-1, t
    print("   Verifying causality...")
    x_test = torch.zeros(1, 8, D, device=device, dtype=dtype)
    x_test[0, 4, 0] = 1.0  # Impulse at t=4
    out_test = causal_conv1d_python(x_test, weight, bias)
    # Response should be at t=4,5,6,7 (not before t=4)
    assert out_test[0, :4, 0].abs().max() < 1e-5, "Conv is not causal!"
    print("   Causality verified!")

    # Test E28 cell
    print("\n2. Testing E28ConvElmanCell...")
    cell = E28ConvElmanCell(D, d_conv=d_conv).to(device).to(dtype)
    z = torch.randn(B, T, D, device=device, dtype=dtype)
    h_init = torch.zeros(B, D, device=device, dtype=dtype)

    h_all, output, h_final = cell(x, z, h_init, use_cuda=False)
    print(f"   h_all: {h_all.shape}, output: {output.shape}, h_final: {h_final.shape}")

    # Test backward
    print("\n3. Testing backward...")
    loss = output.sum()
    loss.backward()
    print(f"   W_x grad norm: {cell.W_x.grad.norm().item():.4f}")
    print(f"   conv_weight grad norm: {cell.conv_weight.grad.norm().item():.4f}")

    # Test full layer
    print("\n4. Testing E28ConvElman layer...")
    layer = E28ConvElman(dim=D, expansion=1.0, d_conv=d_conv).to(device).to(dtype)
    x_in = torch.randn(B, T, D, device=device, dtype=dtype)

    output, h_final = layer(x_in)
    print(f"   Output: {output.shape}, h_final: {h_final.shape}")

    # Param count
    params = sum(p.numel() for p in layer.parameters())
    print(f"\n   Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("E28 Python reference test passed!")
