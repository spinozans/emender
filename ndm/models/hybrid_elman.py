"""
E9: Hybrid Elman - Small dense core + large diagonal memory

Architecture:
    Core: Small dense RNN with full W_h matrix (nonlinear mixing, complex dynamics)
    Memory: Large diagonal bank with learned decay (linear long-range storage)

    h_core_t = tanh(W_x @ x_core + W_h @ h_core_prev + b)
    h_mem_t = sigmoid(a) * h_mem_prev + x_mem
    out_core = h_core * silu(z_core)
    out_mem = h_mem * silu(z_mem)
    output = out_proj([out_core, out_mem])

Key insight: The core handles complex nonlinear interactions with O(core_dim²) compute,
while the memory provides cheap long-range storage with O(mem_dim) compute.
Total hidden state = core_dim + mem_dim, but compute ~ O(core_dim²).

Example config for 50M params:
    core_dim=256, mem_dim=2048 -> 2304 total hidden, ~65K core params per layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    HYBRID_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'hybrid_elman_forward')
except ImportError:
    HYBRID_CUDA_AVAILABLE = False


class HybridElmanFunction(torch.autograd.Function):
    """CUDA-accelerated hybrid elman autograd function."""

    @staticmethod
    def forward(ctx, training, x_core, z_core, x_mem, z_mem,
                h0_core, h0_mem, W_x_core, W_h, b_core, a_mem):
        h_core, h_mem, out_core, out_mem, v_core = hasty_pytorch_lib.hybrid_elman_forward(
            training, x_core, z_core, x_mem, z_mem,
            h0_core, h0_mem, W_x_core, W_h, b_core, a_mem
        )
        ctx.save_for_backward(W_x_core, W_h, a_mem, x_core, z_core, x_mem, z_mem,
                               h_core, h_mem, v_core)
        return h_core, h_mem, out_core, out_mem

    @staticmethod
    def backward(ctx, dh_core, dh_mem, d_out_core, d_out_mem):
        (W_x_core, W_h, a_mem, x_core, z_core, x_mem, z_mem,
         h_core, h_mem, v_core) = ctx.saved_tensors

        dx_core, dz_core, dx_mem, dz_mem, dW_x_core, dW_h, db_core, da_mem = \
            hasty_pytorch_lib.hybrid_elman_backward(
                W_x_core, W_h, a_mem, x_core, z_core, x_mem, z_mem,
                h_core, h_mem, v_core, d_out_core.contiguous(), d_out_mem.contiguous()
            )

        return None, dx_core, dz_core, dx_mem, dz_mem, None, None, dW_x_core, dW_h, db_core, da_mem


class HybridElmanCell(nn.Module):
    """
    E9 Hybrid Elman cell with dense core + diagonal memory.

    The core maintains complex nonlinear dynamics via full W_h matrix.
    The memory provides cheap long-range storage via diagonal decay.
    """

    def __init__(self, core_dim, mem_dim, w_h_mode='spectral_norm', w_h_init_gain=1.0,
                 memory_init_decay=0.9):
        super().__init__()
        self.core_dim = core_dim
        self.mem_dim = mem_dim
        self.w_h_mode = w_h_mode

        # Core RNN weights (small dense matrix)
        self.W_x_core = nn.Parameter(torch.empty(core_dim, core_dim))
        self.W_h = nn.Parameter(torch.empty(core_dim, core_dim))
        self.b_core = nn.Parameter(torch.zeros(core_dim))

        # Memory decay logits (will be sigmoid'd to get decay in (0, 1))
        # Initialize to achieve target decay rate
        init_logit = torch.log(torch.tensor(memory_init_decay / (1 - memory_init_decay)))
        self.a_mem = nn.Parameter(torch.full((mem_dim,), init_logit.item()))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        nn.init.xavier_uniform_(self.W_x_core)
        nn.init.xavier_uniform_(self.W_h, gain=w_h_init_gain)

    def get_W_h(self):
        """Get W_h with spectral normalization applied."""
        if self.w_h_mode == 'spectral_norm':
            target_radius = 0.99
            u = getattr(self, '_spectral_u', None)
            if u is None or u.shape[0] != self.core_dim:
                u = torch.randn(self.core_dim, device=self.W_h.device, dtype=self.W_h.dtype)
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

    def forward(self, x_core, z_core, x_mem, z_mem, h0_core=None, h0_mem=None):
        """
        Args:
            x_core: [T, B, core_dim] pre-activated core input
            z_core: [T, B, core_dim] core gate input
            x_mem: [T, B, mem_dim] memory input
            z_mem: [T, B, mem_dim] memory gate input
            h0_core: [B, core_dim] initial core hidden state
            h0_mem: [B, mem_dim] initial memory state

        Returns:
            out_core: [T, B, core_dim] gated core output
            out_mem: [T, B, mem_dim] gated memory output
            h_core_final: [B, core_dim] final core state
            h_mem_final: [B, mem_dim] final memory state
        """
        T, B, _ = x_core.shape

        if h0_core is None:
            h0_core = torch.zeros(B, self.core_dim, device=x_core.device, dtype=x_core.dtype)
        if h0_mem is None:
            h0_mem = torch.zeros(B, self.mem_dim, device=x_mem.device, dtype=x_mem.dtype)

        W_h = self.get_W_h()

        # Use CUDA kernel if available
        if HYBRID_CUDA_AVAILABLE and x_core.is_cuda:
            h_core, h_mem, out_core, out_mem = HybridElmanFunction.apply(
                self.training,
                x_core.contiguous(),
                z_core.contiguous(),
                x_mem.contiguous(),
                z_mem.contiguous(),
                h0_core.contiguous(),
                h0_mem.contiguous(),
                self.W_x_core.contiguous(),
                W_h.contiguous(),
                self.b_core.contiguous(),
                self.a_mem.contiguous()
            )
            return out_core, out_mem, h_core[-1], h_mem[-1]

        # PyTorch fallback
        h_core = h0_core
        h_mem = h0_mem
        out_core_list = []
        out_mem_list = []

        # Memory decay
        decay = torch.sigmoid(self.a_mem)

        for t in range(T):
            x_core_t = x_core[t]
            z_core_t = z_core[t]
            x_mem_t = x_mem[t]
            z_mem_t = z_mem[t]

            # Core update (nonlinear)
            raw = x_core_t @ self.W_x_core.T + h_core @ W_h.T + self.b_core
            h_core = torch.tanh(raw)
            out_c = h_core * F.silu(z_core_t)
            out_core_list.append(out_c)

            # Memory update (linear diagonal)
            h_mem = decay * h_mem + x_mem_t
            out_m = h_mem * F.silu(z_mem_t)
            out_mem_list.append(out_m)

        out_core = torch.stack(out_core_list, dim=0)
        out_mem = torch.stack(out_mem_list, dim=0)
        return out_core, out_mem, h_core, h_mem


class HybridElman(nn.Module):
    """
    E9: Hybrid Elman layer with dense core + diagonal memory.

    Architecture:
        # Project input to core and memory channels
        x_proj = in_proj(x)  # [B, T, 2*(core_dim + mem_dim)]
        x_core, z_core, x_mem, z_mem = split(x_proj)

        # Pre-activation for core
        x_core = silu(x_core)

        # Run hybrid cell
        out_core, out_mem = hybrid_cell(x_core, z_core, x_mem, z_mem)

        # Project back
        output = out_proj([out_core, out_mem])
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        core_ratio=0.125,  # Fraction of d_inner for core (rest is memory)
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=1.0,
        memory_init_decay=0.9,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Split d_inner into core and memory
        self.core_dim = max(64, int(self.d_inner * core_ratio))
        self.mem_dim = self.d_inner - self.core_dim

        # Input projection: project to 2*(core + mem) for x and z branches
        total_proj = 2 * (self.core_dim + self.mem_dim)
        self.in_proj = nn.Linear(dim, total_proj, bias=False)

        # Hybrid cell
        self.cell = HybridElmanCell(
            self.core_dim,
            self.mem_dim,
            w_h_mode=r_h_mode,
            w_h_init_gain=r_h_init_gain,
            memory_init_decay=memory_init_decay
        )

        # Output projection: project combined output back to dim
        self.out_proj = nn.Linear(self.core_dim + self.mem_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: tuple of (h0_core, h0_mem) initial states, or None

        Returns:
            output: [B, T, dim] output sequence
            h_final: tuple of (h_core_final, h_mem_final)
        """
        B, T, D = x.shape

        # Project and split into core/memory x and z branches
        xz = self.in_proj(x)  # [B, T, 2*(core + mem)]

        # Split: x_core, z_core, x_mem, z_mem
        x_core = xz[:, :, :self.core_dim]
        z_core = xz[:, :, self.core_dim:2*self.core_dim]
        x_mem = xz[:, :, 2*self.core_dim:2*self.core_dim + self.mem_dim]
        z_mem = xz[:, :, 2*self.core_dim + self.mem_dim:]

        # Pre-activation for core (like Mamba2)
        x_core = F.silu(x_core)

        # Transpose for cell: [T, B, dim]
        x_core = x_core.permute(1, 0, 2).contiguous()
        z_core = z_core.permute(1, 0, 2).contiguous()
        x_mem = x_mem.permute(1, 0, 2).contiguous()
        z_mem = z_mem.permute(1, 0, 2).contiguous()

        # Handle initial state
        if h0 is not None:
            h0_core, h0_mem = h0
        else:
            h0_core, h0_mem = None, None

        # Run hybrid cell
        out_core, out_mem, h_core_final, h_mem_final = self.cell(
            x_core, z_core, x_mem, z_mem, h0_core, h0_mem
        )

        # Concatenate outputs and transpose back
        combined = torch.cat([out_core, out_mem], dim=-1)  # [T, B, core+mem]
        combined = combined.permute(1, 0, 2).contiguous()   # [B, T, core+mem]

        # Project back to dim
        combined = self.dropout(combined)
        output = self.out_proj(combined)

        return output, (h_core_final, h_mem_final)

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, '
                f'core_dim={self.core_dim}, mem_dim={self.mem_dim}, LEVEL=9_HYBRID')


if __name__ == "__main__":
    print("Testing HybridElman (E9)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test layer
    model = HybridElman(dim=512, expansion=2.0, core_ratio=0.125).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print(f"Core dim: {model.core_dim}, Memory dim: {model.mem_dim}")
    print(f"Total hidden: {model.core_dim + model.mem_dim}")

    print("Testing forward...")
    out, (h_core, h_mem) = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Core state: {h_core.shape}, Memory state: {h_mem.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Parameter count
    core_params = model.cell.W_x_core.numel() + model.cell.W_h.numel() + model.cell.b_core.numel()
    mem_params = model.cell.a_mem.numel()
    proj_params = model.in_proj.weight.numel() + model.out_proj.weight.numel()

    print(f"\nParameters:")
    print(f"  Core W matrices: {core_params:,}")
    print(f"  Memory decay: {mem_params:,}")
    print(f"  Projections: {proj_params:,}")
    print(f"  Total: {sum(p.numel() for p in model.parameters()):,}")

    print("\nE9 (Hybrid Elman) test passed!")
