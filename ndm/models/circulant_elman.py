"""
E6: Circulant FFT Elman - O(n log n) hidden state updates via FFT

h_t = tanh(circ(c_h) @ h_{t-1} + circ(c_x) @ x_t + b)
output_t = h_t * silu(W_gate @ x_t + b_gate)

Circulant matrix-vector multiply via FFT:
circ(c) @ v = IFFT(FFT(c) * FFT(v))

This gives an effective n×n matrix using only n parameters per circulant.
Complexity: O(n log n) vs O(n²) for dense matmul

Key advantages:
- n parameters instead of n² for recurrence matrix
- FFT is highly optimized on GPU
- Can use much larger hidden dimensions (e.g., 4096 vs 1280)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    CIRCULANT_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'circulant_elman_forward')
except ImportError:
    CIRCULANT_CUDA_AVAILABLE = False


def circulant_matmul(c: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute circ(c) @ v using FFT.

    Args:
        c: [dim] circulant generating vector
        v: [B, dim] input vectors

    Returns:
        [B, dim] result of circulant matrix-vector multiply
    """
    orig_dtype = v.dtype

    # FFT doesn't support bfloat16, cast to float32
    c_f = c.float()
    v_f = v.float()

    # FFT of circulant vector (broadcast across batch)
    fft_c = torch.fft.fft(c_f)  # [dim] complex

    # FFT of input vectors
    fft_v = torch.fft.fft(v_f, dim=-1)  # [B, dim] complex

    # Pointwise multiply in frequency domain
    fft_result = fft_c.unsqueeze(0) * fft_v  # [B, dim] complex

    # IFFT to get result
    result = torch.fft.ifft(fft_result, dim=-1).real  # [B, dim]

    return result.to(orig_dtype)


class CirculantElmanFunction(torch.autograd.Function):
    """CUDA-accelerated circulant elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, c_h, c_x, W_gate, b, b_gate):
        h, output, v, gate_cache = hasty_pytorch_lib.circulant_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            c_h.contiguous(),
            c_x.contiguous(),
            W_gate.contiguous(),
            b.contiguous(),
            b_gate.contiguous()
        )
        if training:
            ctx.save_for_backward(x, c_h, c_x, W_gate, b_gate, h, v, gate_cache)
        return output, h

    @staticmethod
    def backward(ctx, d_output, dh_unused):
        x, c_h, c_x, W_gate, b_gate, h, v, gate_cache = ctx.saved_tensors
        dx, d_c_h, d_c_x, dW_gate, db, d_b_gate = hasty_pytorch_lib.circulant_elman_backward(
            c_h, c_x, W_gate, x, h, v, gate_cache, d_output.contiguous()
        )
        return None, dx, None, d_c_h, d_c_x, dW_gate, db, d_b_gate


class CirculantElmanCell(nn.Module):
    """
    E6 Circulant FFT Elman cell.

    Uses circulant matrices via FFT for O(n log n) hidden state updates.

    h_t = tanh(circ(c_h) @ h_{t-1} + circ(c_x) @ x_t + b)
    output_t = h_t * silu(W_gate @ x_t + b_gate)

    Args:
        dim: Hidden dimension (can be much larger than dense Elman, e.g., 4096)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Circulant vectors (n parameters each, not n²!)
        self.c_h = nn.Parameter(torch.empty(dim))
        self.c_x = nn.Parameter(torch.empty(dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Gate projection (still dense for now - could also be circulant)
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Initialize circulant vectors to have spectral radius < 1
        # For circulant matrices, eigenvalues are FFT(c), so we want |FFT(c)| < 1
        # Initialize with small random values
        nn.init.uniform_(self.c_h, -0.1, 0.1)
        nn.init.uniform_(self.c_x, -0.1, 0.1)

        # Scale to ensure spectral radius < 0.99
        with torch.no_grad():
            fft_c_h = torch.fft.fft(self.c_h)
            max_mag = fft_c_h.abs().max()
            if max_mag > 0.99:
                self.c_h.data *= 0.99 / max_mag

            fft_c_x = torch.fft.fft(self.c_x)
            max_mag = fft_c_x.abs().max()
            if max_mag > 0.99:
                self.c_x.data *= 0.99 / max_mag

        nn.init.xavier_uniform_(self.W_gate)

    def get_spectral_radius(self):
        """Return the spectral radius of the circulant recurrence matrix."""
        with torch.no_grad():
            # FFT doesn't support bfloat16, cast to float32
            fft_c_h = torch.fft.fft(self.c_h.float())
            return fft_c_h.abs().max().item()

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if CIRCULANT_CUDA_AVAILABLE and x.is_cuda:
            return CirculantElmanFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                self.c_h.contiguous(),
                self.c_x.contiguous(),
                self.W_gate.contiguous(),
                self.b.contiguous(),
                self.b_gate.contiguous()
            )

        # PyTorch fallback
        return self._forward_pytorch(x, h0)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch implementation using FFT for circulant matmul."""
        T, B, D = x.shape

        # Pre-compute gate projections for all timesteps
        # x is [T, B, D], reshape to [T*B, D] for matmul
        x_flat = x.reshape(T * B, D)
        gate_proj_all = x_flat @ self.W_gate.T  # [T*B, D]
        gate_proj_all = gate_proj_all.reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]
            gate_proj_t = gate_proj_all[t]

            # Circulant matmuls via FFT
            circ_h = circulant_matmul(self.c_h, h_prev)  # [B, D]
            circ_x = circulant_matmul(self.c_x, x_t)     # [B, D]

            # h_t = tanh(circ_h + circ_x + b)
            raw = circ_h + circ_x + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # output = h * silu(gate_proj + b_gate)
            gate_raw = gate_proj_t + self.b_gate
            gate = F.silu(gate_raw)
            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class CirculantElman(nn.Module):
    """
    E6: Circulant FFT Elman layer with input/output projections.

    Uses circulant matrices via FFT for O(n log n) hidden state updates.
    Can use much larger hidden dimensions than dense Elman (e.g., 4096).

    Architecture:
        x_proj = in_proj(x)
        h_t = tanh(circ(c_h) @ h_{t-1} + circ(c_x) @ x_proj + b)
        output = h * silu(W_gate @ x_proj + b_gate)
        y = out_proj(output)
    """

    def __init__(
        self,
        dim,
        d_inner=None,
        expansion=None,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        # Allow specifying d_inner directly or via expansion
        if d_inner is not None:
            self.d_inner = d_inner
        elif expansion is not None:
            self.d_inner = int(dim * expansion)
        else:
            # Default: much larger hidden dim for circulant (e.g., 4096)
            self.d_inner = 4096

        self.use_conv = use_conv

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Optional conv1d for local context
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # Circulant Elman cell
        self.cell = CirculantElmanCell(self.d_inner)

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

        # Project input
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Optional conv1d for local context
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]
            x_proj = F.silu(x_proj)  # SiLU after conv

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run circulant Elman cell
        cell_out, h_all = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def get_spectral_radius(self):
        """Return the spectral radius of the circulant recurrence matrix."""
        return self.cell.get_spectral_radius()

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, LEVEL=E6_CIRCULANT_FFT'


if __name__ == "__main__":
    print("Testing CirculantElman (E6)...")
    print("=" * 60)
    print(f"CUDA kernel available: {CIRCULANT_CUDA_AVAILABLE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test with target dimensions: dim=4096
    print("\nTesting with large hidden dim (d_inner=4096)...")
    model = CirculantElman(dim=512, d_inner=4096).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")
    print(f"Spectral radius of c_h: {model.get_spectral_radius():.4f}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Test with conv
    print("\nTesting with conv1d...")
    model_conv = CirculantElman(dim=512, d_inner=4096, use_conv=True, d_conv=4).to(device).bfloat16()
    out_conv, h_conv = model_conv(x)
    print(f"With conv1d: Output: {out_conv.shape}")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters (no conv): {params:,}")

    # Compare to dense Elman parameter count
    # Dense W_h: 4096×4096 = 16.7M params
    # Circulant c_h: 4096 params
    # Savings: 16.7M -> 4K = 4000× reduction in recurrence params!
    dense_params = 4096 * 4096  # Just W_h
    circulant_params = 4096  # c_h
    print(f"\nDense W_h params: {dense_params:,}")
    print(f"Circulant c_h params: {circulant_params:,}")
    print(f"Reduction factor: {dense_params / circulant_params:.0f}×")

    params_conv = sum(p.numel() for p in model_conv.parameters())
    print(f"Parameters (with conv): {params_conv:,}")

    print("\nE6 (Circulant FFT Elman) test passed!")
