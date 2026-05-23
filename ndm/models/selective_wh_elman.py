"""
E17: Selective W_h Elman - Input-dependent gating on recurrence

Architecture:
    x, z = split(in_proj(x))           # Split input into two branches
    x = silu(x)                        # Pre-activation
    G = W_gate @ x                     # Gate projection
    gate = sigmoid(G)                  # Per-dim gate on recurrence
    Rh = W_h @ h_{t-1}                 # Dense recurrence
    h_t = tanh(W_x @ x_t + Rh * gate + b)  # Gated recurrence (like Mamba2's selective A)
    output = h * silu(z)               # Gate with other branch

Key insight: Mamba2's advantage may come from input-dependent A matrix.
This adds diagonal selectivity to E1's dense W_h @ h recurrence.

Differences from E1:
    - E1:  h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    - E17: h_t = tanh(W_x @ x_t + (W_h @ h_{t-1}) * sigmoid(W_gate @ x_t) + b)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    SELECTIVE_WH_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'selective_wh_elman_forward')
except ImportError:
    SELECTIVE_WH_CUDA_AVAILABLE = False


class SelectiveWhElmanFunction(torch.autograd.Function):
    """CUDA-accelerated selective W_h elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, W_x, W_h, W_gate, b):
        h, output, v, gate_cache, Rh_cache = hasty_pytorch_lib.selective_wh_elman_forward(
            training, x, z, h0, W_x, W_h, W_gate, b
        )
        ctx.save_for_backward(W_x, W_h, W_gate, x, z, h, v, gate_cache, Rh_cache)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, W_h, W_gate, x, z, h, v, gate_cache, Rh_cache = ctx.saved_tensors
        dx, dz, dW_x, dW_h, dW_gate, db = hasty_pytorch_lib.selective_wh_elman_backward(
            W_x, W_h, W_gate, x, z, h, v, gate_cache, Rh_cache, d_output.contiguous()
        )
        return None, dx, dz, None, dW_x, dW_h, dW_gate, db


class SelectiveWhElmanCell(nn.Module):
    """
    E17 Elman cell with input-dependent gating on W_h @ h.

    h_t = tanh(W_x @ x_t + (W_h @ h_{t-1}) * sigmoid(W_gate @ x_t) + b)
    output = h_t * silu(z_t)
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.mamba2_init = mamba2_init

        # RNN weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_gate = nn.Parameter(torch.empty(dim, dim))  # NEW: gate projection
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        if self.mamba2_init:
            # Mamba2-style initialization
            nn.init.normal_(self.W_x, std=0.02)
            nn.init.normal_(self.W_gate, std=0.02)  # Gate projection
            # W_h: orthogonal init scaled to have spectral radius ~0.999
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W_x)
            nn.init.xavier_uniform_(self.W_h, gain=w_h_init_gain)
            nn.init.xavier_uniform_(self.W_gate)

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
        if SELECTIVE_WH_CUDA_AVAILABLE and x.is_cuda:
            h, output = SelectiveWhElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                W_h.contiguous(),
                self.W_gate.contiguous(),
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

            # Compute W_h @ h_prev
            Rh = h_prev @ W_h.T

            # Input-dependent gate on recurrence (key E17 innovation)
            G = x_t @ self.W_gate.T
            gate = torch.sigmoid(G)

            # Gated Elman recurrence
            raw = x_t @ self.W_x.T + Rh * gate + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Mamba2-style gating: output = h * silu(z)
            output = h_new * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class SelectiveWhElman(nn.Module):
    """
    E17: Selective W_h Elman layer.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h = selective_cell(x, z)    # RNN with selective gating on W_h @ h
        output = out_proj(h)        # Project back to dim
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=1.0,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.mamba2_init = mamba2_init

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Selective W_h Elman cell
        self.cell = SelectiveWhElmanCell(
            self.d_inner,
            w_h_mode=r_h_mode,
            w_h_init_gain=r_h_init_gain,
            mamba2_init=mamba2_init
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
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

        # Pre-activation (like Mamba2)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run selective W_h Elman cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=17_SELECTIVE_WH'


if __name__ == "__main__":
    print("Testing SelectiveWhElman (E17)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test model
    model = SelectiveWhElman(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print(f"CUDA kernel available: {SELECTIVE_WH_CUDA_AVAILABLE}")

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare with E1
    from mamba_gated_elman import MambaGatedElman
    e1 = MambaGatedElman(dim=512, expansion=2.0).to(device).bfloat16()
    e1_params = sum(p.numel() for p in e1.parameters())
    print(f"E1 parameters: {e1_params:,}")
    print(f"E17 overhead: +{(params - e1_params):,} params (+{100*(params - e1_params)/e1_params:.1f}%)")

    print("\nE17 (Selective W_h Elman) test passed!")
