"""
Level 0: Stock Elman - Basic tanh recurrence with learned gate projection

h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
output_t = h_t * silu(W_gate @ x_t + b_gate)  -- x-only with learned projection

Serves as baseline for ablation ladder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Haste CUDA kernel
import os
REQUIRE_CUDA = os.environ.get('ELMAN_REQUIRE_CUDA', '0') == '1'

try:
    import hasty_pytorch_lib
    HASTE_AVAILABLE = hasattr(hasty_pytorch_lib, 'stock_elman_forward')
except ImportError as e:
    if REQUIRE_CUDA:
        raise ImportError(f"CUDA kernels required but not available: {e}")
    HASTE_AVAILABLE = False

LEVEL_0_AVAILABLE = True  # PyTorch fallback always available


class StockElmanFunction(torch.autograd.Function):
    """Autograd function for Stock Elman with learned gate projection."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, W_h, W_gate, b, b_gate):
        h, output, v, gate_cache = hasty_pytorch_lib.stock_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_x.contiguous(),
            W_h.contiguous(),
            W_gate.contiguous(),
            b.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, W_x, W_h, W_gate, b_gate, h, v, gate_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, W_x, W_h, W_gate, b_gate, h, v, gate_cache = ctx.saved_tensors
        dx, dW_x, dW_h, dW_gate, db, d_b_gate = hasty_pytorch_lib.stock_elman_backward(
            W_x, W_h, W_gate, x, h, v, gate_cache, d_output.contiguous()
        )
        return None, dx, None, dW_x, dW_h, dW_gate, db, d_b_gate


class StockElmanCell(nn.Module):
    """
    Stock Elman cell - Level 0 of ablation ladder with learned gate projection.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    output_t = h_t * silu(W_gate @ x_t + b_gate)  -- x-only with learned projection

    Args:
        dim: Hidden dimension
        w_h_mode: Constraint mode for W_h matrix:
            - 'free': No constraint (original behavior)
            - 'spectral_norm': Spectral normalization (constrains spectral radius to 0.99)
            - 'scaled_orthogonal': W_h = sigmoid(scale) * orthogonal_base
        w_h_init_gain: Initial gain for W_h initialization
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.w_h_init_gain = w_h_init_gain

        # Input projection
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Gate projection: output = h * silu(W_gate @ x + b_gate)
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        # Recurrence matrix with optional constraints
        if w_h_mode == 'scaled_orthogonal':
            self.register_buffer('W_h_base', torch.empty(dim, dim))
            self.w_h_log_scale = nn.Parameter(torch.tensor(-0.01))  # sigmoid(-0.01) ≈ 0.497
        else:
            self.W_h = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_gate)
        if self.w_h_mode == 'scaled_orthogonal':
            nn.init.orthogonal_(self.W_h_base)
        else:
            nn.init.xavier_uniform_(self.W_h, gain=self.w_h_init_gain)

    def get_W_h(self):
        """Get the effective W_h matrix with constraints applied."""
        if self.w_h_mode == 'scaled_orthogonal':
            scale = torch.sigmoid(self.w_h_log_scale)
            return scale * self.W_h_base
        elif self.w_h_mode == 'spectral_norm':
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
        else:
            return self.W_h

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state (optional)

        Returns:
            output: [T, B, dim] h+x selective output
            h: [T+1, B, dim] all hidden states including h0
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)

        # Get constrained W_h
        W_h = self.get_W_h()

        # Use Haste kernel if available
        if HASTE_AVAILABLE and x.is_cuda:
            return StockElmanFunction.apply(
                self.training, x, h0,
                self.W_x, W_h, self.W_gate, self.b, self.b_gate
            )

        # PyTorch fallback
        if REQUIRE_CUDA:
            raise RuntimeError("CUDA kernels required (ELMAN_REQUIRE_CUDA=1) but not available")
        return self._forward_pytorch(x, h0, W_h)

    def _forward_pytorch(self, x, h0, W_h):
        """Pure PyTorch implementation with learned gate projection."""
        T, B, D = x.shape
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # h_t = tanh(W_x @ x + W_h @ h + b)
            raw = x_t @ self.W_x.T + h_prev @ W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Learned gate projection: output = h * silu(W_gate @ x + b_gate)
            gate_proj = x_t @ self.W_gate.T + self.b_gate
            gate = F.silu(gate_proj)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class StockElman(nn.Module):
    """
    Stock Elman layer - Level 0 with projections and learned gate projection.

    Wraps StockElmanCell with input/output projections for use in LM.
    Uses learned gate projection: output = h * silu(W_gate @ x + b_gate)

    Optional conv1d for local context (like Mamba2).
    """

    def __init__(self, dim, expansion=1.0, dropout=0.0,
                 r_h_mode='spectral_norm', r_h_init_gain=1.0,
                 use_conv=False, d_conv=4, **kwargs):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.r_h_mode = r_h_mode
        self.use_conv = use_conv

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

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
        else:
            self.conv1d = None

        # Elman cell (use r_h_mode as w_h_mode for consistency)
        self.cell = StockElmanCell(self.d_inner, w_h_mode=r_h_mode, w_h_init_gain=r_h_init_gain)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Optional dropout
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

        # Optional conv1d for local context
        if self.use_conv and self.conv1d is not None:
            # Conv1d expects [B, C, T]
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)  # [B, d_inner, T + d_conv - 1]
            x_conv = x_conv[:, :, :T]  # Causal: keep only first T
            x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
            x_proj = F.silu(x_conv)  # SiLU activation after conv

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell - returns (output, h_all) with learned gate projection
        cell_out, h_all = self.cell(x_rnn, h0)  # [T, B, d_inner], [T+1, B, d_inner]
        h_final = h_all[-1]  # [B, d_inner]

        # Transpose back: [B, T, d_inner]
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Apply dropout and project
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=0_STOCK'


if __name__ == "__main__":
    print("Testing StockElman (Level 0)...")
    print("=" * 60)
    print(f"Haste CUDA kernel available: {HASTE_AVAILABLE}")

    # Test layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StockElman(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print("Level 0 (Stock Elman) test passed!")
