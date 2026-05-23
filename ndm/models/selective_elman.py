"""
E11: Selective Multi-Scale EMA Elman (Mamba-inspired selectivity)

Architecture:
    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  -- same as E1

    # Selective decay (input modulates retention per bank)
    a_scale = x_t @ W_a           # [B, k]
    alpha_i = sigmoid(a_i + a_scale[:, i, None])  # [B, dim] per bank
    m_i_t = alpha_i * m_i_prev + (1 - alpha_i) * h_t

    # Selective read (input chooses which banks to read)
    w = softmax(x_t @ W_w)        # [B, k]
    memory_out = sum(w[:, i:i+1] * m_i)

    # Gated output (just 2 gates instead of k+1)
    out = h * silu(z_h) + memory_out * silu(z_m)

Key insight: Mamba-style selectivity with cheap projections.
- W_a: [dim, k] controls per-bank decay modulation
- W_w: [dim, k] controls which banks to read
- Saves ~15x parameters vs E10's k+1 separate gates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    SELECTIVE_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'selective_elman_forward')
except ImportError:
    SELECTIVE_CUDA_AVAILABLE = False


class SelectiveElmanFunction(torch.autograd.Function):
    """CUDA-accelerated selective EMA elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, m0, W_x, W_h, b, a, W_a, W_w):
        h, m, output, v, a_scale, read_weights = hasty_pytorch_lib.selective_elman_forward(
            training, x, z, h0, m0, W_x, W_h, b, a, W_a, W_w
        )
        ctx.save_for_backward(W_x, W_h, a, W_a, W_w, x, z, h, m, v, a_scale, read_weights)
        return h, m, output

    @staticmethod
    def backward(ctx, dh, dm, d_output):
        W_x, W_h, a, W_a, W_w, x, z, h, m, v, a_scale, read_weights = ctx.saved_tensors

        dx, dz, dW_x, dW_h, db, da, dW_a, dW_w = hasty_pytorch_lib.selective_elman_backward(
            W_x, W_h, a, W_a, W_w, x, z, h, m, v, a_scale, read_weights, d_output.contiguous()
        )

        return None, dx, dz, None, None, dW_x, dW_h, db, da, dW_a, dW_w


class SelectiveElmanCell(nn.Module):
    """
    E11 Selective Multi-Scale EMA Elman cell.

    Uses full dense W_h matrix (like E1) plus k EMA memory banks with:
    - Input-dependent decay (selective retention)
    - Input-dependent read weights (selective memory access)
    """

    def __init__(self, dim, n_banks=4, w_h_mode='spectral_norm', w_h_init_gain=1.0,
                 decay_init_range=(0.8, 0.99)):
        super().__init__()
        self.dim = dim
        self.n_banks = n_banks
        self.w_h_mode = w_h_mode

        # RNN weights (same as E1)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # EMA base decay logits [n_banks, dim]
        self.a = nn.Parameter(torch.empty(n_banks, dim))

        # Selective decay modulation: x @ W_a -> [B, k] per-bank scalars
        self.W_a = nn.Parameter(torch.empty(dim, n_banks))

        # Selective read weights: softmax(x @ W_w) -> [B, k] bank attention
        self.W_w = nn.Parameter(torch.empty(dim, n_banks))

        self._init_decay(decay_init_range)
        self._init_weights(w_h_init_gain)

    def _init_decay(self, decay_init_range):
        """Initialize decay logits to spread decays across the range."""
        low, high = decay_init_range
        decays = torch.linspace(low, high, self.n_banks)
        for i, decay in enumerate(decays):
            logit = torch.log(torch.tensor(decay / (1 - decay)))
            self.a.data[i] = logit + torch.randn(self.dim) * 0.1

        # Initialize modulation weights small (near identity at start)
        nn.init.zeros_(self.W_a)
        nn.init.zeros_(self.W_w)

    def _init_weights(self, w_h_init_gain):
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

    def forward(self, x, z, h0=None, m0=None):
        """
        Args:
            x: [T, B, dim] pre-activated input (also used for selectivity)
            z: [T, B, 2*dim] gates for h and memory_out
            h0: [B, dim] initial hidden state
            m0: [n_banks, B, dim] initial memory states

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
            m_final: [n_banks, B, dim] final memory states
        """
        T, B, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
        if m0 is None:
            m0 = torch.zeros(self.n_banks, B, self.dim, device=x.device, dtype=x.dtype)

        W_h = self.get_W_h()

        # Use CUDA kernel if available
        if SELECTIVE_CUDA_AVAILABLE and x.is_cuda:
            h, m, output = SelectiveElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                m0.contiguous(),
                self.W_x.contiguous(),
                W_h.contiguous(),
                self.b.contiguous(),
                self.a.contiguous(),
                self.W_a.contiguous(),
                self.W_w.contiguous()
            )
            return output, h[-1], m[-1]

        # PyTorch fallback
        h = h0
        m_list = [m0[i].clone() for i in range(self.n_banks)]  # List of [B, dim]
        outputs = []

        for t in range(T):
            x_t = x[t]  # [B, dim]
            z_t = z[t]  # [B, 2*dim]

            # RNN update (same as E1)
            raw = x_t @ self.W_x.T + h @ W_h.T + self.b
            h = torch.tanh(raw)

            # Selective decay: input modulates per-bank decay
            a_scale = x_t @ self.W_a  # [B, k]

            # EMA updates with selective decay (no inplace ops)
            new_m_list = []
            for i in range(self.n_banks):
                # alpha_i = sigmoid(base_a_i + modulation_i)
                alpha_i = torch.sigmoid(self.a[i:i+1, :] + a_scale[:, i:i+1])  # [B, dim]
                new_m_i = alpha_i * m_list[i] + (1 - alpha_i) * h
                new_m_list.append(new_m_i)
            m_list = new_m_list

            # Selective read: input chooses which banks to attend to
            read_weights = F.softmax(x_t @ self.W_w, dim=-1)  # [B, k]

            # Weighted sum of memory banks
            memory_out = torch.zeros_like(h)
            for i in range(self.n_banks):
                memory_out = memory_out + read_weights[:, i:i+1] * m_list[i]

            # Gated output: just 2 gates (h and memory)
            z_h = z_t[:, :self.dim]
            z_m = z_t[:, self.dim:]
            out = h * F.silu(z_h) + memory_out * F.silu(z_m)

            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        m = torch.stack(m_list, dim=0)  # [n_banks, B, dim]
        return output, h, m


class SelectiveElman(nn.Module):
    """
    E11: Selective Multi-Scale EMA Elman layer.

    Architecture:
        # Project input
        xz = in_proj(x)  # [B, T, dim + 2*dim]
        x_branch, z = split(xz)

        # Pre-activation
        x_branch = silu(x_branch)

        # Run selective cell
        output = selective_cell(x_branch, z)

        # Project back
        output = out_proj(output)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_banks=4,
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=1.0,
        decay_init_range=(0.8, 0.99),
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_banks = n_banks

        # Input projection: x_branch + z (just 2 gates now!)
        # x_branch: d_inner
        # z: 2 * d_inner (one gate for h, one for memory)
        total_proj = self.d_inner + 2 * self.d_inner
        self.in_proj = nn.Linear(dim, total_proj, bias=False)

        # Selective cell
        self.cell = SelectiveElmanCell(
            self.d_inner,
            n_banks=n_banks,
            w_h_mode=r_h_mode,
            w_h_init_gain=r_h_init_gain,
            decay_init_range=decay_init_range
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
            h0: tuple of (h0, m0) initial states, or None

        Returns:
            output: [B, T, dim] output sequence
            h_final: tuple of (h_final, m_final)
        """
        B, T, D = x.shape

        # Project and split
        xz = self.in_proj(x)  # [B, T, d_inner + 2*d_inner]

        x_branch = xz[:, :, :self.d_inner]
        z = xz[:, :, self.d_inner:]

        # Pre-activation (like Mamba2)
        x_branch = F.silu(x_branch)

        # Transpose for cell: [T, B, dim]
        x_branch = x_branch.permute(1, 0, 2).contiguous()
        z = z.permute(1, 0, 2).contiguous()

        # Handle initial state
        if h0 is not None:
            h0_hidden, m0 = h0
        else:
            h0_hidden, m0 = None, None

        # Run cell
        output, h_final, m_final = self.cell(x_branch, z, h0_hidden, m0)

        # Transpose back
        output = output.permute(1, 0, 2).contiguous()  # [B, T, d_inner]

        # Project back to dim
        output = self.dropout(output)
        output = self.out_proj(output)

        return output, (h_final, m_final)

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, '
                f'n_banks={self.n_banks}, LEVEL=11_SELECTIVE')


if __name__ == "__main__":
    print("Testing SelectiveElman (E11)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test layer
    model = SelectiveElman(dim=512, expansion=1.0, n_banks=32).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print(f"Inner dim: {model.d_inner}")
    print(f"Num memory banks: {model.n_banks}")
    print(f"Gate dim: 2 * {model.d_inner} = {2 * model.d_inner}")

    print("\nTesting forward...")
    out, (h_final, m_final) = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Hidden state: {h_final.shape}")
    print(f"Memory states: {m_final.shape}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Parameter count comparison
    cell = model.cell
    rnn_params = cell.W_x.numel() + cell.W_h.numel() + cell.b.numel()
    decay_params = cell.a.numel()
    selectivity_params = cell.W_a.numel() + cell.W_w.numel()
    proj_params = model.in_proj.weight.numel() + model.out_proj.weight.numel()

    print(f"\nParameters:")
    print(f"  RNN W matrices: {rnn_params:,}")
    print(f"  EMA decay logits ({model.n_banks} banks): {decay_params:,}")
    print(f"  Selectivity (W_a + W_w): {selectivity_params:,}")
    print(f"  Projections: {proj_params:,}")
    print(f"  Total: {sum(p.numel() for p in model.parameters()):,}")

    # Compare to E10 gate cost
    e10_gate_cost = model.d_inner * model.d_inner * (1 + model.n_banks)
    e11_gate_cost = model.d_inner * model.d_inner * 2 + selectivity_params
    print(f"\nGate parameter comparison:")
    print(f"  E10 would need: {e10_gate_cost:,} (dim² × (1+k))")
    print(f"  E11 uses:       {e11_gate_cost:,} (2×dim² + selectivity)")
    print(f"  Savings:        {e10_gate_cost - e11_gate_cost:,} ({100*(e10_gate_cost-e11_gate_cost)/e10_gate_cost:.1f}%)")

    # Show decay rates
    with torch.no_grad():
        decays = torch.sigmoid(cell.a).float().mean(dim=1)
        print(f"\nMean decay rates per bank: {decays.cpu().numpy()}")

    print("\nE11 (Selective Multi-Scale EMA Elman) test passed!")
