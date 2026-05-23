"""
E16: Diagonal State-Expanded Elman

Combines Mamba2's efficiency with E1's expressivity:
- State expansion (d_state > d_model like Mamba2)
- Diagonal recurrence O(n) instead of O(n²)
- tanh nonlinearity (for composition depth like E1)
- Mamba2-style gating

h' = tanh(A ⊙ h + B @ x)
y = C @ h * silu(z)

Where:
- A is diagonal (d_state params, not d_state²)
- B: d_model -> d_state projection
- C: d_state -> d_model projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    DIAG_STATE_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'diagonal_state_elman_forward')
except ImportError:
    DIAG_STATE_CUDA_AVAILABLE = False


class DiagonalStateElmanFunction(torch.autograd.Function):
    """CUDA-accelerated diagonal state elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, B, C, A):
        h, output, v = hasty_pytorch_lib.diagonal_state_elman_forward(
            training, x, z, h0, B, C, A
        )
        ctx.save_for_backward(B, C, A, x, z, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        B, C, A, x, z, h, v = ctx.saved_tensors
        dx, dz, dB, dC, dA = hasty_pytorch_lib.diagonal_state_elman_backward(
            B, C, A, x, z, h, v, d_output.contiguous()
        )
        return None, dx, dz, None, dB, dC, dA


class DiagonalStateElmanCell(nn.Module):
    """
    Diagonal State-Expanded Elman cell.

    h' = tanh(A ⊙ h + B @ x)
    y = C @ h * silu(z)

    If selective=True (HybridSSM mode):
    A = sigmoid(log_A + A_proj @ x)  # Input-dependent decay
    """

    def __init__(self, d_model, d_state=None, mamba2_init=False, selective=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or d_model * 2
        self.mamba2_init = mamba2_init
        self.selective = selective

        # Projections
        self.B = nn.Parameter(torch.empty(d_model, self.d_state))
        self.C = nn.Parameter(torch.empty(self.d_state, d_model))

        # Diagonal decay (like Mamba2's A)
        self.log_A = nn.Parameter(torch.zeros(self.d_state))

        # Selective projection: A(x) = sigmoid(log_A + A_proj @ x)
        if selective:
            self.A_proj = nn.Linear(d_model, self.d_state, bias=False)
        else:
            self.A_proj = None

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
            # Mamba2-style initialization
            nn.init.normal_(self.B, std=0.02)
            nn.init.normal_(self.C, std=0.02)
            # Initialize A to have values around 0.9 (like Mamba2's dt_bias)
            nn.init.constant_(self.log_A, 0.0)  # sigmoid(0) = 0.5
            if self.A_proj is not None:
                nn.init.normal_(self.A_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.B)
            nn.init.xavier_uniform_(self.C)
            nn.init.constant_(self.log_A, 0.0)
            if self.A_proj is not None:
                nn.init.xavier_uniform_(self.A_proj.weight)

    def get_A(self):
        """Get diagonal A as sigmoid(log_A) for stability."""
        return torch.sigmoid(self.log_A)

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, d_model] input (pre-activated with silu)
            z: [T, B, d_model] gate input
            h0: [B, d_state] initial hidden state

        Returns:
            output: [T, B, d_model] gated output
            h: [T+1, B, d_state] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)

        A = self.get_A()

        # Use CUDA kernel if available
        if DIAG_STATE_CUDA_AVAILABLE and x.is_cuda:
            h, output = DiagonalStateElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.B.contiguous(),
                self.C.contiguous(),
                A.contiguous()
            )
            return output, h

        # PyTorch fallback
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]
            z_t = z[t]

            # Compute A - either fixed or input-dependent (selective)
            if self.selective:
                # HybridSSM mode: A(x) = sigmoid(log_A + A_proj(x))
                A_t = torch.sigmoid(self.log_A + self.A_proj(x_t))
            else:
                A_t = A

            # Diagonal recurrence: h' = tanh(A ⊙ h + B @ x)
            Bx = x_t @ self.B  # [B, d_state]
            pre = A_t * h_prev + Bx
            h_new = torch.tanh(pre)
            h_list.append(h_new)

            # Output: y = C @ h * silu(z)
            y = h_new @ self.C  # [B, d_model]
            output = y * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class DiagonalStateElman(nn.Module):
    """
    E16: Diagonal State-Expanded Elman layer.

    Architecture:
        x, z = split(in_proj(x))
        x = conv1d(x) if use_conv
        x = silu(x)
        h = diag_cell(x, z)      # Diagonal state recurrence
        output = out_proj(h)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        state_expansion=2,  # d_state = dim * state_expansion
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        selective=False,  # HybridSSM mode: input-dependent A
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.d_state = self.d_inner * state_expansion
        self.use_conv = use_conv
        self.mamba2_init = mamba2_init
        self.selective = selective

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

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

        # Diagonal state cell
        self.cell = DiagonalStateElmanCell(
            self.d_inner,
            d_state=self.d_state,
            mamba2_init=mamba2_init,
            selective=selective
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
            h0: [B, d_state] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_state] final hidden state
        """
        B, T, D = x.shape

        # Project and split
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run diagonal state cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        mode = 'SELECTIVE' if self.selective else 'FIXED'
        return f'dim={self.dim}, d_inner={self.d_inner}, d_state={self.d_state}, A={mode}, LEVEL=16_DIAG_STATE'


if __name__ == "__main__":
    print("Testing DiagonalStateElman (E16)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA available: {DIAG_STATE_CUDA_AVAILABLE}")

    # Test
    model = DiagonalStateElman(dim=512, expansion=2.0, state_expansion=2).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")
    print(f"d_state: {model.d_state}")

    print("\nE16 (Diagonal State Elman) test passed!")
