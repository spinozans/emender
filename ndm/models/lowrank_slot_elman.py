"""
E3: Low-Rank Slot Elman - Independent low-rank W_h per slot

Architecture (matches Mamba2 structure):
    h_t[s] = tanh(W_x @ x + U_s @ (V_s @ h_prev[s]) + b)    for each slot s
    output = sum(C[s] * h_t[s]) * silu(z)

Key insight: Low-rank W_h_s = U_s @ V_s gives each slot unique dynamics
with O(2dr) compute instead of O(d²) per slot.

Mapping to Mamba2:
    - n_slots ↔ n_heads (number of independent state tracks)
    - rank ↔ d_state (compressed state dimension per head)
    - Each slot has its own recurrence dynamics (like head-specific queries)

With n_slots=8 and rank=d/4, e3 is ~4x faster than e2 for recurrence
while giving each slot unique dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    LOWRANK_SLOT_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'lowrank_slot_elman_forward')
except ImportError:
    LOWRANK_SLOT_CUDA_AVAILABLE = False

# Try to import Triton diagonal kernel
try:
    from ndm.kernels.diag_slot_triton import diag_slot_recurrence
    DIAG_TRITON_AVAILABLE = True
except ImportError:
    DIAG_TRITON_AVAILABLE = False


class LowRankSlotElmanFunction(torch.autograd.Function):
    """CUDA-accelerated low-rank slot elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, W_x, U, V, b, C):
        h, output, v = hasty_pytorch_lib.lowrank_slot_elman_forward(
            training, x, z, h0, W_x, U, V, b, C
        )
        ctx.save_for_backward(W_x, U, V, C, x, z, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, U, V, C, x, z, h, v = ctx.saved_tensors
        dx, dz, dW_x, dU, dV, db, dC = hasty_pytorch_lib.lowrank_slot_elman_backward(
            W_x, U, V, C, x, z, h, v, d_output.contiguous()
        )
        return None, dx, dz, None, dW_x, dU, dV, db, dC


class LowRankSlotElmanCell(nn.Module):
    """
    E3 Elman cell with per-slot recurrence.

    Options:
    - diag=False: Low-rank W_h = U_s @ V_s (more expressive, slower)
    - diag=True:  Diagonal A_s (fast! like Mamba)

    Recurrence:
    - Low-rank: h_t[s] = tanh(W_x @ x + U_s @ (V_s @ h_prev[s]) + b)
    - Diagonal: h_t[s] = tanh(W_x @ x + A_s * h_prev[s] + b)

    Output: sum(C[s] * h_t[s]) * silu(z)
    """

    def __init__(self, dim, n_slots=8, rank=None, spectral_radius=0.99, diag=False):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.spectral_radius = spectral_radius
        self.diag = diag
        # Default rank = 64 (fixed for speed; dim // 4 was too slow)
        self.rank = rank if rank is not None else 64

        # Shared input projection (same for all slots)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        if diag:
            # Per-slot diagonal recurrence: A_s in (0, spectral_radius)
            # Shape: [n_slots, dim] - unique decay per slot per dimension
            self.A_raw = nn.Parameter(torch.zeros(n_slots, dim))
        else:
            # Per-slot low-rank recurrence: W_h_s = U_s @ V_s
            self.U = nn.Parameter(torch.empty(n_slots, dim, self.rank))
            self.V = nn.Parameter(torch.empty(n_slots, self.rank, dim))

        # Slot combination weights
        self.C = nn.Parameter(torch.ones(n_slots) / n_slots)

        self._init_weights()

    def _init_weights(self):
        # Xavier for input projection
        nn.init.xavier_uniform_(self.W_x)

        if self.diag:
            # Init A_raw near 0 -> A = sigmoid(0) * radius = 0.5 * radius
            nn.init.zeros_(self.A_raw)
        else:
            # Initialize U and V for small spectral norm
            for s in range(self.n_slots):
                nn.init.orthogonal_(self.V[s])
                nn.init.orthogonal_(self.U[s])
                with torch.no_grad():
                    self.U[s].mul_(self.spectral_radius * 0.3)
                    self.V[s].mul_(0.5)

    def _get_A(self):
        """Get diagonal decay in (0, spectral_radius)."""
        return torch.sigmoid(self.A_raw) * self.spectral_radius

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated)
            z: [T, B, dim] input for gating
            h0: [B, n_slots, dim] initial hidden state (user-facing)

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, n_slots, B, dim] all hidden states (CUDA layout for batched GEMM)
        """
        T, B_size, D = x.shape

        # CUDA kernel uses [n_slots, B, dim] layout for efficient batched GEMM
        if h0 is None:
            h0_cuda = torch.zeros(self.n_slots, B_size, D, device=x.device, dtype=x.dtype)
        else:
            # Transpose from user [B, n_slots, dim] to CUDA [n_slots, B, dim]
            h0_cuda = h0.permute(1, 0, 2).contiguous()

        # Use CUDA kernel if available (low-rank only)
        if not self.diag and LOWRANK_SLOT_CUDA_AVAILABLE and x.is_cuda:
            h, output = LowRankSlotElmanFunction.apply(
                self.training, x.contiguous(), z.contiguous(),
                h0_cuda, self.W_x.contiguous(),
                self.U.contiguous(), self.V.contiguous(),
                self.b.contiguous(), self.C.contiguous()
            )
            return output, h

        # Use Triton for diagonal if available
        if self.diag and DIAG_TRITON_AVAILABLE and x.is_cuda:
            # Compute Wx = W_x @ x for all timesteps
            Wx = torch.matmul(x, self.W_x.T)  # [T, B, dim]

            if h0 is None:
                h0_triton = torch.zeros(B_size, self.n_slots, D, device=x.device, dtype=x.dtype)
            else:
                h0_triton = h0

            A = self._get_A()  # [n_slots, dim]
            output, h = diag_slot_recurrence(
                Wx.contiguous(), z.contiguous(), h0_triton.contiguous(),
                A.contiguous(), self.b.contiguous(), self.C.contiguous()
            )
            # h is [T+1, B, S, D], need to transpose to [T+1, S, B, D] for CUDA layout
            h = h.permute(0, 2, 1, 3).contiguous()
            return output, h

        # PyTorch fallback
        if h0 is None:
            h0_fallback = torch.zeros(B_size, self.n_slots, D, device=x.device, dtype=x.dtype)
        else:
            h0_fallback = h0

        if self.diag:
            A = self._get_A()  # [n_slots, dim]

        h_list = [h0_fallback]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]  # [B, n_slots, dim]
            x_t = x[t]  # [B, dim]
            z_t = z[t]  # [B, dim]

            # Shared W_x @ x
            Wx = x_t @ self.W_x.T  # [B, dim]

            if self.diag:
                # Diagonal recurrence: A_s * h_prev[:, s, :]
                Rh = A.unsqueeze(0) * h_prev  # [B, n_slots, dim]
                h_new = torch.tanh(Wx.unsqueeze(1) + Rh + self.b)
            else:
                # Low-rank recurrence for each slot
                h_new_list = []
                for s in range(self.n_slots):
                    h_s = h_prev[:, s, :]
                    Vh_s = h_s @ self.V[s].T
                    Uh_s = Vh_s @ self.U[s].T
                    h_new_s = torch.tanh(Wx + Uh_s + self.b)
                    h_new_list.append(h_new_s)
                h_new = torch.stack(h_new_list, dim=1)

            h_list.append(h_new)

            # Combine slots
            h_combined = (h_new * self.C.view(1, -1, 1)).sum(dim=1)

            # Gating
            output = h_combined * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        h = h.permute(0, 2, 1, 3).contiguous()
        output = torch.stack(output_list, dim=0)
        return output, h


class LowRankSlotElman(nn.Module):
    """
    E3: Slot Elman with per-slot dynamics.

    Options:
        - diag=True:  Diagonal A per slot (fast! uses Triton)
        - diag=False: Low-rank U @ V per slot (more expressive, slower)

    Architecture:
        x, z = split(in_proj(x))
        x = silu(x)
        h = slot_cell(x, z)  # per-slot recurrence
        output = out_proj(h)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_slots=8,
        rank=None,
        spectral_radius=0.99,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        diag=True,  # Default to diagonal for speed (3.9M tok/s!)
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots
        self.diag = diag
        # Default rank=64 for speed (d_inner/4 was too slow)
        self.rank = rank if rank is not None else 64
        self.use_conv = use_conv

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Optional conv1d
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # Slot cell (diagonal or low-rank)
        self.cell = LowRankSlotElmanCell(
            self.d_inner,
            n_slots=n_slots,
            rank=self.rank,
            spectral_radius=spectral_radius,
            diag=diag,
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

        # Project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)

        # Optional conv
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        # h_all is [T+1, n_slots, B, dim] from CUDA, get last and transpose to [B, n_slots, dim]
        h_final = h_all[-1].permute(1, 0, 2).contiguous()  # [n_slots, B, dim] -> [B, n_slots, dim]

        # Project output
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        mode = "diag" if self.diag else f"lowrank(r={self.rank})"
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, mode={mode}, LEVEL=3'


if __name__ == "__main__":
    print("Testing LowRankSlotElman (E3)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA kernel available: {LOWRANK_SLOT_CUDA_AVAILABLE}")

    # Create model matching Mamba2 structure
    model = LowRankSlotElman(dim=512, expansion=2.0, n_slots=8, rank=64).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print(f"n_slots={model.n_slots}, rank={model.rank}, d_inner={model.d_inner}")
    print(f"Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare with e2
    from slot_elman import SlotElman
    e2 = SlotElman(dim=512, expansion=2.0, n_slots=8).to(device)
    e2_params = sum(p.numel() for p in e2.parameters())
    print(f"E2 parameters: {e2_params:,}")
    print(f"E3 vs E2: {params / e2_params:.2f}x params")

    print("\nE3 (Low-Rank Slot Elman) test passed!")
