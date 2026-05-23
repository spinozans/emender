"""
E18: h-Aware Gate Elman - Three variants for output gating

E18-A: output = h * silu(z + h)     -- add h to gate (FREE)
E18-B: output = h * silu(z + Rh)    -- add Rh to gate (FREE, cache Rh)
E18-E: output = h                   -- no gate (faster, fewer params)

Key insight: E1's gate only sees input z, not hidden state h.
Adding h-awareness might help the model "query" attractor states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    HAWARE_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'haware_gate_elman_forward')
except ImportError:
    HAWARE_CUDA_AVAILABLE = False


class HAwareGateElmanFunction(torch.autograd.Function):
    """CUDA-accelerated h-aware gate elman autograd function."""

    @staticmethod
    def forward(ctx, training, gate_mode, x, z, h0, W_x, W_h, b):
        h, output, v, Rh_cache = hasty_pytorch_lib.haware_gate_elman_forward(
            training, gate_mode, x, z, h0, W_x, W_h, b
        )
        ctx.gate_mode = gate_mode
        ctx.save_for_backward(W_x, W_h, x, z, h, v, Rh_cache)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, W_h, x, z, h, v, Rh_cache = ctx.saved_tensors
        dx, dz, dW_x, dW_h, db = hasty_pytorch_lib.haware_gate_elman_backward(
            ctx.gate_mode, W_x, W_h, x, z, h, v, Rh_cache, d_output.contiguous()
        )
        return None, None, dx, dz, None, dW_x, dW_h, db


class HAwareGateElmanCell(nn.Module):
    """
    E18 Elman cell with h-aware output gating.

    gate_mode:
        0 = E18-A: output = h * silu(z + h)
        1 = E18-B: output = h * silu(z + Rh)
        2 = E18-E: output = h (no gate)
    """

    def __init__(self, dim, gate_mode=0, w_h_mode='spectral_norm', w_h_init_gain=1.0, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.gate_mode = gate_mode
        self.w_h_mode = w_h_mode
        self.mamba2_init = mamba2_init

        # RNN weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        if self.mamba2_init:
            nn.init.normal_(self.W_x, std=0.02)
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W_x)
            nn.init.xavier_uniform_(self.W_h, gain=w_h_init_gain)

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
            z: [T, B, dim] input for gating (ignored for gate_mode=2)
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
        if HAWARE_CUDA_AVAILABLE and x.is_cuda:
            h, output = HAwareGateElmanFunction.apply(
                self.training,
                self.gate_mode,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                W_h.contiguous(),
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
            Rh = h_prev @ W_h.T
            raw = x_t @ self.W_x.T + Rh + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Output gating based on mode
            if self.gate_mode == 0:
                # E18-A: gate = silu(z + h)
                output = h_new * F.silu(z_t + h_new)
            elif self.gate_mode == 1:
                # E18-B: gate = silu(z + Rh)
                output = h_new * F.silu(z_t + Rh)
            else:
                # E18-E: no gate
                output = h_new
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class HAwareGateElman(nn.Module):
    """
    E18: h-Aware Gate Elman layer with Mamba2-style split projection.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h = haware_cell(x, z)       # RNN with h-aware gated output
        output = out_proj(h)        # Project back to dim
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        gate_mode=0,  # 0=A, 1=B, 2=E
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=1.0,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.gate_mode = gate_mode
        self.mamba2_init = mamba2_init

        # For E18-E, we don't need z at all
        if gate_mode == 2:
            # Just project to d_inner (no split)
            self.in_proj = nn.Linear(dim, self.d_inner, bias=False)
        else:
            # Mamba2-style: project to 2*d_inner, then split
            self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Elman cell
        self.cell = HAwareGateElmanCell(
            self.d_inner,
            gate_mode=gate_mode,
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

        if self.gate_mode == 2:
            # E18-E: no gate, just project x
            x_proj = self.in_proj(x)  # [B, T, d_inner]
            x_proj = F.silu(x_proj)
            # Create dummy z tensor for CUDA kernel
            z = torch.zeros_like(x_proj)
        else:
            # Mamba2-style: project and split
            xz = self.in_proj(x)  # [B, T, 2*d_inner]
            x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]
            x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run Elman cell with h-aware gating
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        mode_names = {0: 'A (z+h)', 1: 'B (z+Rh)', 2: 'E (no gate)'}
        return f'dim={self.dim}, d_inner={self.d_inner}, gate_mode={self.gate_mode} ({mode_names[self.gate_mode]}), LEVEL=18_HAWARE_GATE'


if __name__ == "__main__":
    print("Testing HAwareGateElman (E18)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for mode, name in [(0, 'E18-A (z+h)'), (1, 'E18-B (z+Rh)'), (2, 'E18-E (no gate)')]:
        print(f"\n{name}:")
        model = HAwareGateElman(dim=512, expansion=2.0, gate_mode=mode).to(device).bfloat16()
        x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

        out, h = model(x)
        loss = out.sum()
        loss.backward()

        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}, Output: {out.shape}")

    print("\nE18 (h-Aware Gate Elman) test passed!")
