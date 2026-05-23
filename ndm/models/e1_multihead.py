"""
E1H: Multi-Head E1 (Mamba-Gated Elman with independent heads)

Like E1 but with H independent Elman heads, each with its own W_h recurrence.
Each head maintains a vector state of size n_state.

Architecture:
    x, z = split(in_proj(x))           # dim -> 2*H*n_state
    x = silu(x)                        # Pre-activation (like E1/Mamba2)
    x = reshape(x, [B, T, H, n_state])
    z = reshape(z, [B, T, H, n_state])

    # Pre-compute W_x for all timesteps (batched GEMM):
    pre_x = einsum('bthi,hij->bthj', x, W_x)  # [B, T, H, n_state]

    # Sequential recurrence per timestep:
    for t in range(T):
        h = tanh(pre_x[:, t] + einsum('bhi,hij->bhj', h_prev, W_h) + b)
        out[t] = h * silu(z[:, t])

    output = out_proj(reshape(out))     # H*n_state -> dim

Key differences from E1:
    - E1: single d_inner x d_inner state (one big GEMM per step)
    - E1H: H independent n_state x n_state states (batched small GEMMs per step)

Key differences from E88:
    - E88: matrix state (n_state x n_state per head), decay, L2 norm, separate q/k/v
    - E1H: vector state (n_state per head), no decay, no L2, single in_proj split
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to load CUDA kernel
try:
    import hasty_pytorch_lib
    HAS_CUDA = hasattr(hasty_pytorch_lib, 'e1h_forward')
except ImportError:
    HAS_CUDA = False


class E1HCUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E1H CUDA kernels."""

    @staticmethod
    def forward(ctx, training, pre_x, z, h0, W_h, b_h):
        """
        Args:
            pre_x: [B, T, H, N] pre-computed W_x @ silu(x)
            z: [B, T, H, N] gate values
            h0: [B, H, N] initial hidden state
            W_h: [H, N, N] per-head recurrence weight
            b_h: [H, N] per-head bias
        Returns:
            output: [B, T, H, N]
            h_final: [B, H, N]
        """
        # CUDA kernel expects [T, B, H, N]
        pre_x_t = pre_x.permute(1, 0, 2, 3).contiguous()
        z_t = z.permute(1, 0, 2, 3).contiguous()

        results = hasty_pytorch_lib.e1h_forward(
            training,
            pre_x_t,       # [T, B, H, N]
            z_t,           # [T, B, H, N]
            h0,            # [B, H, N]
            W_h,           # [H, N, N]
            b_h            # [H, N]
        )

        h_final = results[0]       # [B, H, N]
        output_t = results[1]      # [T, B, H, N]
        h_checkpoints = results[2] # [num_cp, B, H, N]

        # Transpose output back to [B, T, H, N]
        output = output_t.permute(1, 0, 2, 3).contiguous()

        if training:
            ctx.save_for_backward(pre_x_t, z_t, W_h, b_h, h_checkpoints)

        return output, h_final

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        pre_x_t, z_t, W_h, b_h, h_checkpoints = ctx.saved_tensors

        # d_output is [B, T, H, N], transpose to [T, B, H, N]
        d_output_t = d_output.permute(1, 0, 2, 3).contiguous()

        results = hasty_pytorch_lib.e1h_backward(
            pre_x_t,           # [T, B, H, N]
            z_t,               # [T, B, H, N]
            W_h,               # [H, N, N]
            b_h,               # [H, N]
            h_checkpoints,     # [num_cp, B, H, N]
            d_output_t         # [T, B, H, N]
        )

        d_pre_x_t = results[0]  # [T, B, H, N]
        d_z_t = results[1]      # [T, B, H, N]
        d_W_h_per_b = results[2] # [B, H, N, N] float32
        d_b_per_b = results[3]   # [B, H, N] float32

        # Transpose gradients back to [B, T, H, N]
        d_pre_x = d_pre_x_t.permute(1, 0, 2, 3).contiguous()
        d_z = d_z_t.permute(1, 0, 2, 3).contiguous()

        # Sum d_W_h and d_b across batch dimension
        d_W_h = d_W_h_per_b.sum(dim=0)  # [H, N, N]
        d_b = d_b_per_b.sum(dim=0)       # [H, N]

        # Return gradients: training, pre_x, z, h0, W_h, b_h
        return None, d_pre_x, d_z, None, d_W_h.to(W_h.dtype), d_b.to(b_h.dtype)


class E1MultiHead(nn.Module):
    """
    E1H: Multi-Head E1 layer.

    H independent Elman RNN heads with Mamba2-style split projection gating.
    Each head has its own W_x, W_h, b parameters operating on n_state-dimensional vectors.
    """

    def __init__(
        self,
        dim,
        n_heads=16,
        n_state=32,
        expansion=1.0,  # Ignored when n_heads and n_state are set
        dropout=0.0,
        mamba2_init=False,
        **kwargs  # Accept and ignore extra kwargs from LadderLM
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_state = n_state
        self.d_inner = n_heads * n_state

        # Mamba2-style: project to 2*d_inner, then split into x (RNN) and z (gate)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Per-head parameters: H independent n_state x n_state recurrences
        # h_i = tanh(x_i @ W_x_i.T + h_prev_i @ W_h_i.T + b_i)
        self.W_x = nn.Parameter(torch.empty(n_heads, n_state, n_state))
        self.W_h = nn.Parameter(torch.empty(n_heads, n_state, n_state))
        self.b = nn.Parameter(torch.zeros(n_heads, n_state))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
            nn.init.normal_(self.W_x, std=0.02)
            # Orthogonal W_h with spectral radius ~0.999 for each head
            for h in range(self.n_heads):
                W_h_fp32 = torch.empty(self.n_state, self.n_state, dtype=torch.float32)
                nn.init.orthogonal_(W_h_fp32)
                W_h_fp32.mul_(0.999)
                with torch.no_grad():
                    self.W_h.data[h].copy_(W_h_fp32.to(self.W_h.dtype))
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            # Per-head xavier
            for h in range(self.n_heads):
                nn.init.xavier_uniform_(self.W_x.data[h])
                nn.init.orthogonal_(self.W_h.data[h])
                self.W_h.data[h].mul_(0.9)

    def _forward_pytorch(self, x, h0):
        """Pure PyTorch forward (reference / fallback)."""
        B, T, D = x.shape
        H, N = self.n_heads, self.n_state

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*H*N]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, H*N]

        # Pre-activation (like E1/Mamba2)
        x_proj = F.silu(x_proj)

        # Reshape to per-head
        x_proj = x_proj.view(B, T, H, N)  # [B, T, H, N]
        z = z.view(B, T, H, N)  # [B, T, H, N]

        # Pre-compute W_x @ x for all timesteps (batched GEMM over heads and time)
        pre_x = torch.einsum('bthi,hij->bthj', x_proj, self.W_x)

        # Initial state
        if h0 is None:
            h_prev = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)
        else:
            h_prev = h0

        # Sequential recurrence
        outputs = []
        for t in range(T):
            wh = torch.einsum('bhi,hij->bhj', h_prev, self.W_h)
            h_new = torch.tanh(pre_x[:, t] + wh + self.b)
            out_t = h_new * F.silu(z[:, t])
            outputs.append(out_t)
            h_prev = h_new

        output = torch.stack(outputs, dim=1)  # [B, T, H, N]
        output = output.reshape(B, T, self.d_inner)
        output = self.dropout(output)
        output = self.out_proj(output)

        return output, h_prev

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, H, n_state] initial hidden state, or None

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, H, n_state] final hidden state
        """
        B, T, D = x.shape
        H, N = self.n_heads, self.n_state

        use_cuda = HAS_CUDA and x.is_cuda and x.dtype == torch.bfloat16

        # The E1H CUDA kernel corrupts GPU state at inference with trained
        # weights (async illegal memory access that surfaces at the next
        # cuBLAS call). Force the PyTorch fallback for all eval-mode runs
        # — it's numerically equivalent and fast enough for generation.
        if not self.training:
            use_cuda = False

        if not use_cuda:
            return self._forward_pytorch(x, h0)

        # CUDA path: compute projections in Python, recurrence in CUDA
        xz = self.in_proj(x)  # [B, T, 2*H*N]
        x_proj, z = xz.chunk(2, dim=-1)

        x_proj = F.silu(x_proj)

        x_proj = x_proj.view(B, T, H, N)
        z = z.view(B, T, H, N)

        # Pre-compute W_x @ x
        pre_x = torch.einsum('bthi,hij->bthj', x_proj, self.W_x)

        if h0 is None:
            h0 = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)

        # CUDA kernel handles the recurrence
        output, h_final = E1HCUDAFunction.apply(
            self.training, pre_x, z, h0, self.W_h, self.b
        )

        # Reshape and project
        output = output.reshape(B, T, self.d_inner)
        output = self.dropout(output)
        output = self.out_proj(output)

        return output, h_final

    def extra_repr(self):
        return (f'dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, '
                f'd_inner={self.d_inner}, LEVEL=E1H_MULTIHEAD')
