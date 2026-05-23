"""
E2: Slot-Based Elman - Multi-slot memory with cuBLAS GEMMs

Architecture (same GEMMs as e0, but with n_slots independent hidden states):
    h_t[s] = tanh(W_x @ x + W_h @ h_prev[s] + b)    for each slot s
    output = sum(C[s] * h_t[s]) * silu(z)

Key optimization: Batch slots into GEMM by treating [B, n_slots, d] as [B*n_slots, d]
This gives same GEMM speed as e0 but with n_slots more memory capacity.

Key differences from e1:
- e1: h ∈ R^d with one W_h matmul per step
- e2: h ∈ R^(n_slots, d) with batched W_h matmul (all slots in one GEMM)
- e2 has n_slots × more memory with ~same compute
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    SLOT_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'slot_elman_forward')
except ImportError:
    SLOT_CUDA_AVAILABLE = False


class SlotElmanFunction(torch.autograd.Function):
    """CUDA-accelerated slot elman autograd function with cuBLAS GEMMs."""

    @staticmethod
    def forward(ctx, training, x, z, h0, W_x, W_h, b, C):
        h, output, v = hasty_pytorch_lib.slot_elman_forward(
            training, x, z, h0, W_x, W_h, b, C
        )
        ctx.save_for_backward(W_x, W_h, C, x, z, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, W_h, C, x, z, h, v = ctx.saved_tensors
        dx, dz, dW_x, dW_h, db, dC = hasty_pytorch_lib.slot_elman_backward(
            W_x, W_h, C, x, z, h, v, d_output.contiguous()
        )
        return None, dx, dz, None, dW_x, dW_h, db, dC


class SlotElmanCell(nn.Module):
    """
    E2 Elman cell with slot-based memory.

    Recurrence options:
    - diag=False: h_t[s] = tanh(W_x @ x + W_h @ h_prev[s] + b)  (full matmul)
    - diag=True:  h_t[s] = tanh(W_x @ x + A * h_prev[s] + b)    (diagonal, fast!)

    Output: sum(C[s] * h_t[s]) * silu(z)
    """

    def __init__(self, dim, n_slots=64, spectral_radius=0.99, diag=False):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.spectral_radius = spectral_radius
        self.diag = diag

        # Input projection
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Recurrence: full matrix or diagonal
        if diag:
            # Diagonal A in (-1, 1) for stability
            self.A = nn.Parameter(torch.zeros(dim))
        else:
            self.W_h = nn.Parameter(torch.empty(dim, dim))

        # Slot combination weights
        self.C = nn.Parameter(torch.ones(n_slots) / n_slots)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)

        if self.diag:
            # Init A near 0 (sigmoid(0) = 0.5, so decay ~0.5)
            nn.init.zeros_(self.A)
        else:
            nn.init.orthogonal_(self.W_h)
            with torch.no_grad():
                self.W_h.mul_(self.spectral_radius * 0.5)

    def _get_normalized_Wh(self):
        """Spectral normalization for full W_h."""
        W = self.W_h
        u = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
        for _ in range(3):
            v = F.normalize(W.T @ u, dim=0)
            u = F.normalize(W @ v, dim=0)
        sigma = (u @ W @ v).item()
        if sigma > self.spectral_radius:
            return W * (self.spectral_radius / sigma)
        return W

    def _get_A(self):
        """Get diagonal recurrence in (0, spectral_radius) via sigmoid."""
        return torch.sigmoid(self.A) * self.spectral_radius

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated)
            z: [T, B, dim] input for gating
            h0: [B, n_slots, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, n_slots, dim] all hidden states
        """
        T, B_size, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B_size, self.n_slots, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available (full W_h only)
        if not self.diag and SLOT_CUDA_AVAILABLE and x.is_cuda:
            W_h = self._get_normalized_Wh()
            h, output = SlotElmanFunction.apply(
                self.training, x.contiguous(), z.contiguous(),
                h0.contiguous(), self.W_x.contiguous(),
                W_h.contiguous(), self.b.contiguous(), self.C.contiguous()
            )
            return output, h

        # PyTorch path (diagonal or fallback)
        if self.diag:
            A = self._get_A()  # [dim] in (0, spectral_radius)
        else:
            W_h = self._get_normalized_Wh()

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]  # [B, n_slots, dim]
            x_t = x[t]  # [B, dim]
            z_t = z[t]  # [B, dim]

            # W_x @ x is same for all slots - broadcast
            Wx = x_t @ self.W_x.T  # [B, dim]

            if self.diag:
                # Diagonal recurrence: A * h_prev (element-wise, broadcast A)
                Rh = A * h_prev  # [B, n_slots, dim]
            else:
                # Full W_h matmul: batch across slots
                h_flat = h_prev.reshape(B_size * self.n_slots, D)
                Rh_flat = h_flat @ W_h.T  # [B*n_slots, dim]
                Rh = Rh_flat.reshape(B_size, self.n_slots, D)

            # h_new = tanh(Wx + Rh + b), broadcast Wx across slots
            h_new = torch.tanh(Wx.unsqueeze(1) + Rh + self.b)  # [B, n_slots, dim]
            h_list.append(h_new)

            # Combine slots with learned weights: sum(C[s] * h_new[:, s, :])
            h_combined = (h_new * self.C.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # [B, dim]

            # Mamba2-style gating
            output = h_combined * F.silu(z_t)  # [B, dim]
            output_list.append(output)

        h = torch.stack(h_list, dim=0)  # [T+1, B, n_slots, dim]
        output = torch.stack(output_list, dim=0)  # [T, B, dim]
        return output, h


class SlotElman(nn.Module):
    """
    E2: Slot-Based Elman with multi-slot memory.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = conv1d(x) if use_conv   # Optional local context
        x = silu(x)                 # Pre-activation
        h = slot_cell(x, z)         # Multi-slot RNN with gated output
        output = out_proj(h)        # Project back to dim

    Recurrence options (via diag parameter):
        diag=False: Full W_h matmul (more expressive, slower)
        diag=True:  Diagonal A (like Mamba, much faster)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_slots=8,
        spectral_radius=0.99,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        diag=False,  # Full W_h with CUDA kernel is faster than Python loop
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots
        self.use_conv = use_conv
        self.diag = diag

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

        # Slot-based Elman cell
        self.cell = SlotElmanCell(
            self.d_inner,
            n_slots=n_slots,
            spectral_radius=spectral_radius,
            diag=diag
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
            h0: [B, n_slots, d_inner] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, n_slots, d_inner] final hidden state
        """
        B, T, D = x.shape

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run slot-based cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]  # [B, n_slots, d_inner]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, diag={self.diag}, LEVEL=2_SLOT'


if __name__ == "__main__":
    print("Testing SlotElman (E2) with cuBLAS GEMMs...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test basic
    model = SlotElman(dim=512, expansion=2.0, n_slots=8).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print(f"Testing forward with n_slots=8...")
    out, h = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape} (includes slots dimension)")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Compare parameter counts
    print("\n" + "=" * 60)
    print("Parameter comparison:")

    from mamba_gated_elman import MambaGatedElman
    e1 = MambaGatedElman(dim=512, expansion=2.0).to(device)
    e2 = SlotElman(dim=512, expansion=2.0, n_slots=8).to(device)

    e1_params = sum(p.numel() for p in e1.parameters())
    e2_params = sum(p.numel() for p in e2.parameters())

    print(f"E1 (Mamba-Gated): {e1_params:,} params")
    print(f"E2 (Slot, 8 slots): {e2_params:,} params")
    print(f"E2 memory: 8 independent hidden states per layer")

    print("\nE2 (Slot Elman with cuBLAS GEMMs) test passed!")
