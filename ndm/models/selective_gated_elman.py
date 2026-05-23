"""
E12: Selective Gated Elman - Hidden-state-dependent gating

This is a minimal modification of E1 to make the gate "selective":
- E1:  output = h * silu(z)              # gate depends only on input
- E12: output = h * sigmoid(z + W_g @ h) # gate depends on hidden state

This makes the gate "selective" - it depends on what the model has computed
(hidden state h), similar to Mamba2's input-dependent gating mechanism.

Architecture per timestep:
    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)    # Same as E1
    g_t = W_g @ h_t                              # NEW: project h for gating
    gate_t = sigmoid(z_t + g_t)                  # NEW: selective gate
    output_t = h_t * gate_t                      # Gated output

Key differences from E1:
    - Adds W_g [dim, dim] parameter (one extra GEMM per timestep)
    - Uses sigmoid(z + W_g@h) instead of silu(z)
    - Gate now depends on hidden state, not just input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    SELECTIVE_GATED_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'selective_gated_elman_forward')
except ImportError:
    SELECTIVE_GATED_CUDA_AVAILABLE = False


class SelectiveGatedElmanFunction(torch.autograd.Function):
    """CUDA-accelerated selective gated elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, W_x, W_h, W_g, b):
        h, output, v, gate_cache = hasty_pytorch_lib.selective_gated_elman_forward(
            training, x, z, h0, W_x, W_h, W_g, b
        )
        ctx.save_for_backward(W_x, W_h, W_g, x, z, h, v, gate_cache)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, W_h, W_g, x, z, h, v, gate_cache = ctx.saved_tensors
        dx, dz, dW_x, dW_h, dW_g, db = hasty_pytorch_lib.selective_gated_elman_backward(
            W_x, W_h, W_g, x, z, h, v, gate_cache, d_output.contiguous()
        )
        return None, dx, dz, None, dW_x, dW_h, dW_g, db


class SelectiveGatedElmanCell(nn.Module):
    """
    E12 Elman cell with selective (hidden-state-dependent) gating.

    The input is already split and pre-activated before reaching this cell.
    The cell receives x (for RNN) and z (for gating) separately.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    g_t = W_g @ h_t
    output = h_t * sigmoid(z_t + g_t)
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode

        # RNN weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_g = nn.Parameter(torch.empty(dim, dim))  # NEW: gate projection
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h, gain=w_h_init_gain)
        # Initialize W_g small so initial gate ≈ sigmoid(z) ≈ silu(z) behavior
        nn.init.xavier_uniform_(self.W_g, gain=0.1)

    def get_W_h(self):
        """Get W_h with spectral normalization applied."""
        if self.w_h_mode == 'spectral_norm':
            target_radius = 0.99
            u = getattr(self, '_spectral_u', None)
            if u is None or u.shape[0] != self.dim:
                u = torch.randn(self.dim, device=self.W_h.device, dtype=self.W_h.dtype)
                u = u / u.norm()
            with torch.no_grad():
                for _ in range(3):
                    v = self.W_h.T @ u
                    v = v / (v.norm() + 1e-8)
                    u = self.W_h @ v
                    u = u / (u.norm() + 1e-8)
                self._spectral_u = u
            sigma = (u @ self.W_h @ v).abs()
            return self.W_h * (target_radius / (sigma + 1e-8))
        return self.W_h

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated with silu)
            z: [T, B, dim] input for gating
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        W_h = self.get_W_h()

        # Use CUDA kernel if available
        if SELECTIVE_GATED_CUDA_AVAILABLE and x.is_cuda:
            h, output = SelectiveGatedElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                W_h.contiguous(),
                self.W_g.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]
            z_t = z[t]

            # Elman recurrence
            raw = x_t @ self.W_x.T + h_prev @ W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Selective gating: output = h * sigmoid(z + W_g @ h)
            g = h_new @ self.W_g.T
            gate = torch.sigmoid(z_t + g)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class SelectiveGatedElman(nn.Module):
    """
    E12: Selective Gated Elman layer with hidden-state-dependent gating.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = conv1d(x) if use_conv   # Optional local context
        x = silu(x)                 # Pre-activation
        h = elman_cell(x, z)        # RNN with selective gated output
        output = out_proj(h)        # Project back to dim
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=1.0,
        use_conv=False,
        d_conv=4,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Optional conv1d for local context (like Mamba2)
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,  # Causal padding
                groups=self.d_inner,  # Depthwise
                bias=True,
            )

        # Elman cell with selective gating
        self.cell = SelectiveGatedElmanCell(
            self.d_inner,
            w_h_mode=r_h_mode,
            w_h_init_gain=r_h_init_gain
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

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

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Optional conv1d for local context
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Pre-activation (like Mamba2 applies silu after conv)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run Elman cell with selective gating
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, LEVEL=12_SELECTIVE_GATED'


if __name__ == "__main__":
    print("Testing SelectiveGatedElman (E12)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test without conv
    model = SelectiveGatedElman(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward (no conv)...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Test with conv
    model_conv = SelectiveGatedElman(dim=512, expansion=2.0, use_conv=True, d_conv=4).to(device).bfloat16()
    out_conv, h_conv = model_conv(x)
    print(f"\nWith conv1d: Output: {out_conv.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters (no conv): {params:,}")

    params_conv = sum(p.numel() for p in model_conv.parameters())
    print(f"Parameters (with conv): {params_conv:,}")

    # Compare with E1 parameter count
    try:
        from .mamba_gated_elman import MambaGatedElman
        e1_model = MambaGatedElman(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        e1_params = sum(p.numel() for p in e1_model.parameters())
        extra_params = params - e1_params
        print(f"\nE1 parameters: {e1_params:,}")
        print(f"Extra params from W_g: {extra_params:,} ({100*extra_params/e1_params:.1f}% increase)")
    except ImportError:
        pass

    print("\nE12 (Selective Gated Elman) test passed!")
