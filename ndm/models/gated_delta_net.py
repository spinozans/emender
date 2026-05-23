"""
Gated DeltaNet - ICLR 2025 "Gated Delta Networks: Improving Mamba2 with Delta Rule"

This module provides:
1. FLA wrapper: Uses fla library's optimized Triton kernels when available
2. Pure PyTorch fallback: Sequential implementation for testing/debugging

The Gated Delta Rule:
    S_t = α_t * S_{t-1} * (I - β_t * k_t * k_t^T) + β_t * v_t * k_t^T

Where:
- S_t is the [d, d] hidden state matrix
- α_t is the decay/forget gate
- β_t is the write strength
- k_t is the key (which slot to address)
- v_t is the value (what to write)

Key insight: The (I - β*kk^T) term is the "selective erase" from DeltaNet.
Combined with α decay, this gives fine-grained memory control.

Reference: https://github.com/NVlabs/GatedDeltaNet
           https://github.com/fla-org/flash-linear-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import FLA's optimized implementation
FLA_AVAILABLE = False
try:
    from fla.layers import DeltaNet as FLADeltaNet
    from fla.layers import GatedDeltaNet as FLAGatedDeltaNet
    FLA_AVAILABLE = True
except ImportError:
    try:
        from fla.layers.delta_net import DeltaNet as FLADeltaNet
        FLA_AVAILABLE = True
    except ImportError:
        pass


class GatedDeltaNetCell(nn.Module):
    """
    Pure PyTorch Gated DeltaNet cell for reference/fallback.

    Recurrence:
        α_t = sigmoid(x @ W_α)                 # decay gate
        β_t = sigmoid(x @ W_β)                 # write gate
        k_t = normalize(x @ W_k)               # key (address)
        v_t = x @ W_v                          # value (content)

        # Selective erase then write
        erase = β_t * outer(k_t, k_t)          # what to erase
        S_t = α_t * (S_{t-1} - S_{t-1} @ erase) + β_t * outer(v_t, k_t)

        # Output via query
        q_t = normalize(x @ W_q)
        out_t = S_t @ q_t
    """

    def __init__(self, dim, head_dim=64, num_heads=None):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads or dim // head_dim

        assert dim % self.num_heads == 0, f"dim {dim} not divisible by num_heads {self.num_heads}"

        # Per-head projections
        self.W_q = nn.Parameter(torch.empty(dim, dim))
        self.W_k = nn.Parameter(torch.empty(dim, dim))
        self.W_v = nn.Parameter(torch.empty(dim, dim))

        # Gates
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), 2.0))  # high initial retain

        self.W_beta = nn.Parameter(torch.empty(dim, dim))
        self.b_beta = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        for W in [self.W_q, self.W_k, self.W_v, self.W_alpha, self.W_beta]:
            nn.init.xavier_uniform_(W)

    def forward(self, x, S0=None):
        """
        Args:
            x: [T, B, D] input sequence
            S0: [B, H, head_dim, head_dim] initial state (or None)

        Returns:
            output: [T, B, D]
            S_final: [B, H, head_dim, head_dim]
        """
        T, B, D = x.shape
        H = self.num_heads
        d_h = self.head_dim

        # Initialize state
        if S0 is None:
            S = torch.zeros(B, H, d_h, d_h, device=x.device, dtype=x.dtype)
        else:
            S = S0.clone()

        # Batch compute projections
        x_flat = x.reshape(T * B, D)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, H, d_h)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, H, d_h)
        v_all = (x_flat @ self.W_v.T).reshape(T, B, H, d_h)
        alpha_all = torch.sigmoid(x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, H, d_h)
        beta_all = torch.sigmoid(x_flat @ self.W_beta.T + self.b_beta).reshape(T, B, H, d_h)

        # Normalize q and k
        q_all = F.normalize(q_all, dim=-1)
        k_all = F.normalize(k_all, dim=-1)

        outputs = []
        for t in range(T):
            q = q_all[t]  # [B, H, d_h]
            k = k_all[t]  # [B, H, d_h]
            v = v_all[t]  # [B, H, d_h]
            alpha = alpha_all[t].mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, H, 1, 1]
            beta = beta_all[t].mean(dim=-1, keepdim=True)  # [B, H, 1]

            # Selective erase: S @ (kk^T) gives component of S in k direction
            # S - β * S @ kk^T erases the k-direction proportionally
            k_outer = torch.einsum('bhd,bhe->bhde', k, k)  # [B, H, d_h, d_h]
            erase = torch.einsum('bhde,bhef->bhdf', S, k_outer)  # S @ kk^T

            # Write: outer(v, k)
            write = torch.einsum('bhd,bhe->bhde', v, k)  # [B, H, d_h, d_h]

            # Update: decay * (S - erase) + write
            S = alpha * (S - beta.unsqueeze(-1) * erase) + beta.unsqueeze(-1) * write

            # Output: S @ q
            out = torch.einsum('bhde,bhe->bhd', S, q)  # [B, H, d_h]
            out = out.reshape(B, D)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)  # [T, B, D]
        return output, S


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet layer compatible with elman framework.

    Uses FLA's optimized Triton kernels when available,
    falls back to pure PyTorch implementation otherwise.

    This is the "standard" Gated DeltaNet from ICLR 2025 -
    a known-good baseline for linear attention with memory.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=True,
        d_conv=4,
        head_dim=64,
        num_heads=None,
        use_fla=True,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.head_dim = head_dim
        self.num_heads = num_heads or self.d_inner // head_dim
        self.use_fla = use_fla and FLA_AVAILABLE

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        if self.use_fla:
            # Use FLA's optimized implementation
            self.cell = FLADeltaNet(
                hidden_size=self.d_inner,
                head_dim=head_dim,
                num_heads=self.num_heads,
            )
        else:
            # Pure PyTorch fallback
            self.cell = GatedDeltaNetCell(
                self.d_inner,
                head_dim=head_dim,
                num_heads=self.num_heads,
            )

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)

        if self.use_fla:
            # FLA expects [B, T, D] and returns [B, T, D]
            cell_out = self.cell(x_proj)[0]
            h_final = None  # FLA doesn't expose state by default
        else:
            # Our fallback expects [T, B, D]
            x_rnn = x_proj.permute(1, 0, 2).contiguous()
            cell_out, h_final = self.cell(x_rnn, h0)
            cell_out = cell_out.permute(1, 0, 2).contiguous()

        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        fla_str = "FLA" if self.use_fla else "PyTorch"
        return f'dim={self.dim}, d_inner={self.d_inner}, heads={self.num_heads}, backend={fla_str}, LEVEL=GATED_DELTA_NET'


# Vector state version (simpler, like E61/E62)
class GatedDeltaNetVector(nn.Module):
    """
    Simplified Gated DeltaNet with vector state (not matrix).

    This is essentially DeltaNet/GRU-style gating without the
    outer product state - simpler but less expressive.

    Recurrence:
        α_t = sigmoid(x @ W_α)     # decay
        β_t = sigmoid(x @ W_β)     # write strength
        v_t = tanh(x @ W_v)        # value

        h_t = α_t * h_{t-1} + β_t * v_t

    Note: This is NOT UTM-class (no nonlinear h-dependence).
    Use E63+ for UTM expressivity.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # Gates and value
        self.W_alpha = nn.Parameter(torch.empty(self.d_inner, self.d_inner))
        self.b_alpha = nn.Parameter(torch.full((self.d_inner,), 2.0))
        self.W_beta = nn.Parameter(torch.empty(self.d_inner, self.d_inner))
        self.b_beta = nn.Parameter(torch.zeros(self.d_inner))
        self.W_v = nn.Parameter(torch.empty(self.d_inner, self.d_inner))
        self.b_v = nn.Parameter(torch.zeros(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_beta)
        nn.init.xavier_uniform_(self.W_v)
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()  # [T, B, d_inner]

        # Batch projections
        x_flat = x_rnn.reshape(T * B, self.d_inner)
        alpha_all = torch.sigmoid(x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, self.d_inner)
        beta_all = torch.sigmoid(x_flat @ self.W_beta.T + self.b_beta).reshape(T, B, self.d_inner)
        v_all = torch.tanh(x_flat @ self.W_v.T + self.b_v).reshape(T, B, self.d_inner)

        if h0 is None:
            h = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
        else:
            h = h0

        outputs = []
        for t in range(T):
            h = alpha_all[t] * h + beta_all[t] * v_all[t]
            out = h * F.silu(h)
            outputs.append(out)

        cell_out = torch.stack(outputs, dim=1)  # [B, T, d_inner]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=GATED_DELTA_NET_VECTOR'


if __name__ == "__main__":
    print("Testing Gated DeltaNet...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"FLA available: {FLA_AVAILABLE}")

    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    # Test vector version (always works)
    print("\n--- Gated DeltaNet Vector (E61/E62-style) ---")
    model_vec = GatedDeltaNetVector(dim=512, expansion=2.0).to(device).bfloat16()
    out, h = model_vec(x)
    print(f"Output: {out.shape}")
    loss = out.sum()
    loss.backward()
    print(f"Backward passed!")
    params = sum(p.numel() for p in model_vec.parameters())
    print(f"Parameters: {params:,}")

    # Test matrix version
    print("\n--- Gated DeltaNet Matrix ---")
    model_mat = GatedDeltaNet(dim=512, expansion=2.0, use_fla=False).to(device).bfloat16()
    out, h = model_mat(x)
    print(f"Output: {out.shape}")
    loss = out.sum()
    loss.backward()
    print(f"Backward passed!")
    params = sum(p.numel() for p in model_mat.parameters())
    print(f"Parameters: {params:,}")

    if FLA_AVAILABLE:
        print("\n--- Gated DeltaNet with FLA backend ---")
        model_fla = GatedDeltaNet(dim=512, expansion=2.0, use_fla=True).to(device).bfloat16()
        out, h = model_fla(x)
        print(f"Output: {out.shape}")
        print(f"FLA backend active!")

    print("\n" + "=" * 60)
    print("Gated DeltaNet: S = α*(S - β*S@kk^T) + β*v@k^T")
    print("ICLR 2025 baseline for linear attention with memory")
    print("=" * 60)
