"""
E22: Structured Elman with State Attention (UTM class)

Extends E21 with periodic state attention to achieve Universal Turing Machine
computational class. The key insight: TC¹ (nonlinear RNN) is not UTM—you need
state-dependent routing where any state position can influence any other.

Architecture:
    # Per timestep (E21 base):
    H_t = SiLU(α_t * H_{t-1} + B_t @ X_t.T)

    # Every K timesteps (E22 routing):
    H_t = H_t + StateAttention(H_t)

    # Output:
    y_t = H_t.sum(dim=N)
    output = y_t * silu(z_t + y_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CUDA kernel
try:
    import sys
    sys.path.insert(0, '/home/erikg/elman/elman/cuda')
    import hasty_pytorch_lib
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False


class StructuredElmanAttentionFunction(torch.autograd.Function):
    """Custom autograd function for E22 CUDA kernel."""

    @staticmethod
    def forward(ctx, B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H0,
                nheads, d_state, mimo_rank, attn_period, attn_dim,
                nonlinearity_mode, training):
        """
        Args:
            B_proj: [T, B, nheads, d_state, mimo_rank]
            X_proj: [T, B, nheads, headdim, mimo_rank]
            alpha_raw: [T, B, nheads]
            alpha_bias: [nheads]
            z: [T, B, d_inner]
            W_q: [nheads, headdim, attn_dim]
            W_k: [nheads, headdim, attn_dim]
            W_v: [nheads, headdim, attn_dim]
            W_o: [nheads, attn_dim, headdim]
            H0: [B, nheads, d_state, headdim]
            nonlinearity_mode: 0=silu, 1=tanh, 2=linear
        """
        # Make tensors contiguous BEFORE forward to ensure backward gets same layout
        B_proj = B_proj.contiguous()
        X_proj = X_proj.contiguous()
        alpha_raw = alpha_raw.contiguous()
        alpha_bias = alpha_bias.contiguous()
        z = z.contiguous()
        W_q = W_q.contiguous()
        W_k = W_k.contiguous()
        W_v = W_v.contiguous()
        W_o = W_o.contiguous()
        H0 = H0.contiguous()

        output, H_final, H_all, y_cache = hasty_pytorch_lib.structured_elman_attention_forward(
            training, nheads, d_state, mimo_rank, attn_period, attn_dim, nonlinearity_mode,
            B_proj, X_proj, alpha_raw, alpha_bias, z,
            W_q, W_k, W_v, W_o, H0
        )

        if training:
            ctx.save_for_backward(B_proj, X_proj, alpha_raw, alpha_bias, z,
                                  W_q, W_k, W_v, W_o, H_all, y_cache)
            ctx.nheads = nheads
            ctx.d_state = d_state
            ctx.mimo_rank = mimo_rank
            ctx.attn_period = attn_period
            ctx.attn_dim = attn_dim
            ctx.nonlinearity_mode = nonlinearity_mode

        return output, H_final

    @staticmethod
    def backward(ctx, d_output, d_H_final):
        B_proj, X_proj, alpha_raw, alpha_bias, z, W_q, W_k, W_v, W_o, H_all, y_cache = ctx.saved_tensors

        dz, dB_proj, dX_proj, dalpha_raw, dW_q, dW_k, dW_v, dW_o = \
            hasty_pytorch_lib.structured_elman_attention_backward(
                ctx.nheads, ctx.d_state, ctx.mimo_rank,
                ctx.attn_period, ctx.attn_dim, ctx.nonlinearity_mode,
                B_proj, X_proj, alpha_raw, alpha_bias, z,
                W_q, W_k, W_v, W_o, H_all, y_cache,
                d_output.contiguous()
            )

        # dalpha_bias is sum of dalpha_raw over time and batch
        dalpha_bias = dalpha_raw.sum(dim=(0, 1))

        return dB_proj, dX_proj, dalpha_raw, dalpha_bias, dz, dW_q, dW_k, dW_v, dW_o, None, None, None, None, None, None, None, None


class StateAttentionOverN(nn.Module):
    """
    Attention over N state positions (most expressive routing).

    Each of the N state positions can attend to all other N positions.
    Cost: O(N² × d_k + N × P × d_k) per head.
    For N=32, d_k=32, P=64: ~100K FLOPs per head.
    """

    def __init__(self, headdim, d_attn=32):
        super().__init__()
        self.d_attn = d_attn
        self.scale = d_attn ** -0.5

        # Project from P (headdim) to d_attn
        self.W_q = nn.Linear(headdim, d_attn, bias=False)
        self.W_k = nn.Linear(headdim, d_attn, bias=False)
        self.W_v = nn.Linear(headdim, d_attn, bias=False)
        self.W_o = nn.Linear(d_attn, headdim, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Small init for residual path
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.zeros_(self.W_o.weight)  # Zero init for residual

    def forward(self, H):
        """
        H: [B, nheads, N, P] - state tensor
        Returns: H + delta (residual connection)
        """
        B, nheads, N, P = H.shape

        # [B, nheads, N, P] -> [B*nheads, N, P]
        H_flat = H.view(B * nheads, N, P)

        # Project to Q, K, V: [BH, N, d_k]
        Q = self.W_q(H_flat)
        K = self.W_k(H_flat)
        V = self.W_v(H_flat)

        # Attention scores: [BH, N, N]
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)

        # Apply attention: [BH, N, d_k]
        V_out = torch.bmm(attn, V)

        # Project back: [BH, N, P]
        H_delta = self.W_o(V_out)
        H_delta = H_delta.view(B, nheads, N, P)

        return H + H_delta


class StateAttentionOverHeads(nn.Module):
    """
    Attention over heads (cheaper alternative).

    Each head attends to other heads based on flattened N×P state.
    Cost: O(H² × d_k + H × NP × d_k)
    """

    def __init__(self, d_state, headdim, d_attn=32):
        super().__init__()
        self.d_attn = d_attn
        self.scale = d_attn ** -0.5

        state_dim = d_state * headdim  # NP
        self.W_q = nn.Linear(state_dim, d_attn, bias=False)
        self.W_k = nn.Linear(state_dim, d_attn, bias=False)
        self.W_v = nn.Linear(state_dim, d_attn, bias=False)
        self.W_o = nn.Linear(d_attn, state_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.zeros_(self.W_o.weight)

    def forward(self, H):
        """
        H: [B, nheads, N, P]
        """
        B, nheads, N, P = H.shape

        # Flatten state per head: [B, H, NP]
        H_flat = H.view(B, nheads, N * P)

        # Project: [B, H, d_k]
        Q = self.W_q(H_flat)
        K = self.W_k(H_flat)
        V = self.W_v(H_flat)

        # Attention: heads attend to each other
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)

        # Apply: [B, H, d_k]
        V_out = torch.bmm(attn, V)

        # Project back: [B, H, NP]
        H_delta = self.W_o(V_out)
        H_delta = H_delta.view(B, nheads, N, P)

        return H + H_delta


class StructuredElmanAttentionCell(nn.Module):
    """
    E22 Cell: E21 + periodic state attention.

    State: H ∈ ℝ^{nheads × d_state × headdim}

    When CUDA kernel is available, uses fused kernel for maximum efficiency.
    Falls back to Python for unsupported configurations.
    """

    def __init__(
        self,
        d_inner,
        nheads=16,
        d_state=32,
        mimo_rank=8,
        nonlinearity='silu',
        attn_period=8,
        attn_type='over_N',
        attn_dim=32,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.nheads = nheads
        self.headdim = d_inner // nheads
        self.d_state = d_state
        self.mimo_rank = mimo_rank
        self.nonlinearity = nonlinearity
        self.attn_period = attn_period
        self.attn_type = attn_type
        self.attn_dim = attn_dim

        assert d_inner % nheads == 0

        # E21: Decay bias
        self.alpha_bias = nn.Parameter(torch.full((nheads,), 2.2))

        # E22: State attention weights (stored directly for CUDA kernel)
        # W_q, W_k, W_v: [nheads, headdim, attn_dim]
        # W_o: [nheads, attn_dim, headdim]
        self.W_q = nn.Parameter(torch.empty(nheads, self.headdim, attn_dim))
        self.W_k = nn.Parameter(torch.empty(nheads, self.headdim, attn_dim))
        self.W_v = nn.Parameter(torch.empty(nheads, self.headdim, attn_dim))
        self.W_o = nn.Parameter(torch.empty(nheads, attn_dim, self.headdim))

        self._init_attn_weights()

        # Check if CUDA kernel supports this configuration
        self._use_cuda = (
            HAS_CUDA_KERNEL and
            attn_type == 'over_N' and
            d_state == 32 and
            self.headdim == 64 and
            attn_dim == 32 and
            mimo_rank in [4, 8, 16]
        )

        # Fallback attention module for non-CUDA path
        if not self._use_cuda:
            if attn_type == 'over_N':
                self.state_attn = StateAttentionOverN(self.headdim, attn_dim)
            else:
                self.state_attn = StateAttentionOverHeads(d_state, self.headdim, attn_dim)

    def _init_attn_weights(self):
        """Initialize attention weights with small values for residual path."""
        nn.init.normal_(self.W_q, std=0.02)
        nn.init.normal_(self.W_k, std=0.02)
        nn.init.normal_(self.W_v, std=0.02)
        nn.init.zeros_(self.W_o)  # Zero init for residual

    def _get_nonlinearity_mode(self):
        """Convert nonlinearity string to mode int for CUDA kernel."""
        if self.nonlinearity == 'silu':
            return 0
        elif self.nonlinearity == 'tanh':
            return 1
        else:  # linear
            return 2

    def forward(self, B_proj, X_proj, alpha_raw, z, H0=None, step_offset=0):
        """
        Args:
            B_proj: [T, B, nheads, d_state, mimo_rank]
            X_proj: [T, B, nheads, headdim, mimo_rank]
            alpha_raw: [T, B, nheads]
            z: [T, B, d_inner]
            H0: [B, nheads, d_state, headdim] initial state
            step_offset: for tracking attention period across chunks

        Returns:
            output: [T, B, d_inner]
            H_final: [B, nheads, d_state, headdim]
        """
        T, B, _, _, _ = B_proj.shape
        device, dtype = B_proj.device, B_proj.dtype

        if H0 is None:
            H0 = torch.zeros(B, self.nheads, self.d_state, self.headdim,
                           device=device, dtype=dtype)

        # Use CUDA kernel if available and supported
        if self._use_cuda and device.type == 'cuda' and step_offset == 0:
            return self._forward_cuda(B_proj, X_proj, alpha_raw, z, H0)

        # Fallback to Python loop
        return self._forward_python(B_proj, X_proj, alpha_raw, z, H0, step_offset)

    def _forward_cuda(self, B_proj, X_proj, alpha_raw, z, H0):
        """CUDA kernel path - fused and fast."""
        output, H_final = StructuredElmanAttentionFunction.apply(
            B_proj, X_proj, alpha_raw, self.alpha_bias, z,
            self.W_q, self.W_k, self.W_v, self.W_o, H0,
            self.nheads, self.d_state, self.mimo_rank,
            self.attn_period, self.attn_dim,
            self._get_nonlinearity_mode(), self.training
        )
        return output, H_final

    def _forward_python(self, B_proj, X_proj, alpha_raw, z, H0, step_offset):
        """Python fallback for unsupported configurations."""
        T, B = B_proj.shape[:2]

        H = H0
        output_list = []

        for t in range(T):
            global_t = step_offset + t

            # === E21: Nonlinear MIMO update ===

            # Decay: scalar per head
            alpha = torch.sigmoid(-F.softplus(alpha_raw[t] + self.alpha_bias))

            # MIMO update
            B_t = B_proj[t]  # [B, nheads, d_state, mimo_rank]
            X_t = X_proj[t]  # [B, nheads, headdim, mimo_rank]
            update = torch.einsum('bhnr,bhpr->bhnp', B_t, X_t)

            # Nonlinear state transition
            pre_act = alpha[:, :, None, None] * H + update
            if self.nonlinearity == 'silu':
                H = F.silu(pre_act)
            elif self.nonlinearity == 'tanh':
                H = torch.tanh(pre_act)
            else:  # linear
                H = pre_act

            # === E22: Periodic state attention ===
            if (global_t + 1) % self.attn_period == 0:
                if hasattr(self, 'state_attn'):
                    H = self.state_attn(H)
                else:
                    # Use direct weights for Python attention
                    H = self._apply_state_attention(H)

            # === Output ===
            y_t = H.sum(dim=2)  # [B, nheads, headdim]
            y_t = y_t.reshape(B, self.d_inner)

            # E18-A style gating
            z_t = z[t]
            gate = F.silu(z_t + y_t)
            out_t = y_t * gate

            output_list.append(out_t)

        output = torch.stack(output_list, dim=0)  # [T, B, d_inner]
        return output, H

    def _apply_state_attention(self, H):
        """Apply state attention using direct weights (Python path)."""
        B, nheads, N, P = H.shape
        d_k = self.attn_dim

        # [B, nheads, N, P] -> [B*nheads, N, P]
        H_flat = H.view(B * nheads, N, P)

        # W_q, W_k, W_v are [nheads, P, d_k] - expand for batch
        W_q = self.W_q.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nheads, P, d_k)
        W_k = self.W_k.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nheads, P, d_k)
        W_v = self.W_v.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nheads, P, d_k)
        W_o = self.W_o.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * nheads, d_k, P)

        # Q, K, V: [BH, N, d_k]
        Q = torch.bmm(H_flat, W_q)
        K = torch.bmm(H_flat, W_k)
        V = torch.bmm(H_flat, W_v)

        # Attention: [BH, N, N]
        scale = d_k ** -0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale
        attn = F.softmax(scores, dim=-1)

        # Apply: [BH, N, d_k]
        V_out = torch.bmm(attn, V)

        # Project back: [BH, N, P]
        H_delta = torch.bmm(V_out, W_o)
        H_delta = H_delta.view(B, nheads, N, P)

        return H + H_delta


class StructuredElmanAttention(nn.Module):
    """
    E22: Structured Elman with State Attention (UTM class).

    Extends E21 with periodic state attention for state-dependent routing.
    """

    def __init__(
        self,
        dim,
        expansion=2.0,
        nheads=16,
        d_state=32,
        mimo_rank=8,
        nonlinearity='silu',
        attn_period=8,
        attn_type='over_N',
        attn_dim=32,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.nheads = nheads
        self.headdim = self.d_inner // nheads
        self.d_state = d_state
        self.mimo_rank = mimo_rank
        self.nonlinearity = nonlinearity
        self.attn_period = attn_period
        self.attn_type = attn_type

        # Ensure divisibility
        if self.d_inner % nheads != 0:
            self.d_inner = (self.d_inner // nheads) * nheads
            self.headdim = self.d_inner // nheads

        # Combined input projection
        d_B = nheads * d_state * mimo_rank
        d_X = nheads * self.headdim * mimo_rank
        self.d_proj = self.d_inner + self.d_inner + d_B + d_X + nheads
        self.in_proj = nn.Linear(dim, self.d_proj, bias=False)

        # Cell with state attention
        self.cell = StructuredElmanAttentionCell(
            self.d_inner, nheads, d_state, mimo_rank, nonlinearity,
            attn_period, attn_type, attn_dim
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x, H0=None, step_offset=0, **kwargs):
        """
        Args:
            x: [B, T, dim]
            H0: [B, nheads, d_state, headdim] initial state
            step_offset: for tracking attention period across chunks

        Returns:
            output: [B, T, dim]
            H_final: [B, nheads, d_state, headdim]
        """
        B, T, D = x.shape

        # Combined projection
        proj = self.in_proj(x)

        # Split
        d_B = self.nheads * self.d_state * self.mimo_rank
        d_X = self.nheads * self.headdim * self.mimo_rank
        sizes = [self.d_inner, self.d_inner, d_B, d_X, self.nheads]

        x_path, z, B_flat, X_flat, alpha_raw = proj.split(sizes, dim=-1)

        # Reshape MIMO components
        B_proj = B_flat.view(B, T, self.nheads, self.d_state, self.mimo_rank)
        X_proj = X_flat.view(B, T, self.nheads, self.headdim, self.mimo_rank)

        # Transpose for cell: [T, B, ...]
        B_proj = B_proj.permute(1, 0, 2, 3, 4).contiguous()
        X_proj = X_proj.permute(1, 0, 2, 3, 4).contiguous()
        alpha_raw = alpha_raw.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, H_final = self.cell(B_proj, X_proj, alpha_raw, z_rnn, H0, step_offset)

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, H_final

    def extra_repr(self):
        state_size = self.nheads * self.d_state * self.headdim
        return (f'dim={self.dim}, d_inner={self.d_inner}, nheads={self.nheads}, '
                f'd_state={self.d_state}, mimo_rank={self.mimo_rank}, '
                f'nonlinearity={self.nonlinearity}, attn_period={self.attn_period}, '
                f'attn_type={self.attn_type}, state_size={state_size:,}, '
                f'LEVEL=22_UTM')


if __name__ == "__main__":
    print("Testing StructuredElmanAttention (E22)...")
    print("=" * 60)
    print(f"CUDA kernel available: {HAS_CUDA_KERNEL}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test configs - only CUDA-compatible configs for now
    configs = [
        # CUDA-compatible: nheads=16, headdim=64, d_state=32, mimo_rank=8, attn_dim=32
        {"dim": 1024, "nheads": 16, "expansion": 1.0, "attn_period": 8,
         "attn_type": "over_N", "mimo_rank": 8, "name": "E22 CUDA (K=8, over_N)"},
        # Different period
        {"dim": 1024, "nheads": 16, "expansion": 1.0, "attn_period": 4,
         "attn_type": "over_N", "mimo_rank": 8, "name": "E22 CUDA (K=4, over_N)"},
        # Different MIMO rank
        {"dim": 1024, "nheads": 16, "expansion": 1.0, "attn_period": 8,
         "attn_type": "over_N", "mimo_rank": 4, "name": "E22 CUDA (K=8, R=4)"},
    ]

    for cfg in configs:
        print(f"\n{cfg['name']}:")
        model = StructuredElmanAttention(
            dim=cfg['dim'],
            expansion=cfg['expansion'],
            nheads=cfg['nheads'],
            d_state=32,
            mimo_rank=cfg['mimo_rank'],
            attn_period=cfg['attn_period'],
            attn_type=cfg['attn_type'],
            attn_dim=32,
        ).to(device).bfloat16()

        # Check if using CUDA path
        using_cuda = model.cell._use_cuda
        print(f"  Using CUDA kernel: {using_cuda}")

        x = torch.randn(2, 64, cfg['dim'], device=device, dtype=torch.bfloat16)

        # Forward pass
        out, h = model(x)
        loss = out.sum()
        loss.backward()

        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}, Output: {out.shape}, H: {h.shape}")

        # Quick benchmark (forward only to avoid CUDA async issues)
        if device == 'cuda' and using_cuda:
            import time
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                for _ in range(100):
                    out, h = model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            toks = 2 * 64 * 100  # batch * seq * iters
            print(f"  Throughput: {toks / elapsed:.0f} tok/s (100 forward iters)")

    print("\nE22 (Structured Elman with State Attention) test passed!")
