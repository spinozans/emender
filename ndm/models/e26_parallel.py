"""
E26: Parallel Dual-Memory Elman

Key insight: Separate "what" (content via GEMMs) from "where" (routing via attention).

Architecture:
  PARALLEL PHASE (batched cuBLAS):
    x_proj[0:T] = x[0:T] @ W_x.T   # One big GEMM for all timesteps

  SEQUENTIAL PHASE (cheap routing):
    for t in range(T):
      read = softmax(h_work @ tape.T) @ tape    # O(N×D) dots
      h_work = tanh(x_proj[t] + W_h @ h_work + read + b)
      tape = sparse_write(tape, h_work)          # O(N×D) dots

The W_h @ h_work is still sequential but unavoidable for RNN semantics.
Attention is cheap: O(N×D) vs O(D²) for GEMM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def e26_forward_step_python(
    x_proj_t: torch.Tensor,     # [B, D] - pre-projected input
    h_tape: torch.Tensor,       # [B, N, D] - tape memory
    h_work: torch.Tensor,       # [B, D] - working memory
    W_h: torch.Tensor,          # [D, D] - recurrence weight
    b_h: torch.Tensor,          # [D] - bias
    W_write: torch.Tensor,      # [D, D] - write projection
    scale: float                # 1/sqrt(D) for attention
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single E26 timestep.

    Returns: (h_work_new, h_tape_new, read_attn, write_attn)
    """
    B, N, D = h_tape.shape

    # Read attention: h_work @ tape.T -> [B, N]
    read_scores = torch.einsum('bd,bnd->bn', h_work.float(), h_tape.float()) * scale
    read_attn = F.softmax(read_scores, dim=-1).to(h_work.dtype)

    # Read value: weighted sum over tape
    read_val = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape.float()).to(h_work.dtype)

    # Update h_work: tanh(x_proj + W_h @ h_work + read + b)
    Rh = h_work @ W_h.T  # [B, D] - the sequential GEMM
    h_work_new = torch.tanh(x_proj_t + Rh + read_val + b_h)

    # Write attention
    write_val = h_work_new @ W_write.T  # [B, D]
    write_scores = torch.einsum('bd,bnd->bn', write_val.float(), h_tape.float()) * scale
    write_attn = F.softmax(write_scores, dim=-1).to(h_work.dtype)

    # Update tape: h_tape = h_tape * (1 - w) + write_val * w
    h_tape_new = (h_tape * (1 - write_attn.unsqueeze(-1)) +
                  write_val.unsqueeze(1) * write_attn.unsqueeze(-1))

    return h_work_new, h_tape_new, read_attn, write_attn


def e26_forward_python(
    x: torch.Tensor,            # [B, T, D] - input sequence
    h_tape_init: torch.Tensor,  # [B, N, D] - initial tape
    h_work_init: torch.Tensor,  # [B, D] - initial working memory
    W_h: torch.Tensor,          # [D, D]
    W_x: torch.Tensor,          # [D, D]
    b_h: torch.Tensor,          # [D]
    W_write: torch.Tensor       # [D, D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E26 forward pass.

    Returns: (h_work_all, h_tape_final, read_attn_all, write_attn_all)
    """
    B, T, D = x.shape
    N = h_tape_init.size(1)
    scale = 1.0 / (D ** 0.5)

    # PARALLEL PHASE: Pre-compute x_proj for ALL timesteps
    x_proj = x @ W_x.T  # [B, T, D] - one big GEMM

    # SEQUENTIAL PHASE: Routing
    h_tape = h_tape_init.clone()
    h_work = h_work_init.clone()

    h_work_list = []
    read_attn_list = []
    write_attn_list = []

    for t in range(T):
        h_work, h_tape, read_attn, write_attn = e26_forward_step_python(
            x_proj[:, t, :], h_tape, h_work, W_h, b_h, W_write, scale
        )
        h_work_list.append(h_work)
        read_attn_list.append(read_attn)
        write_attn_list.append(write_attn)

    h_work_all = torch.stack(h_work_list, dim=1)      # [B, T, D]
    read_attn_all = torch.stack(read_attn_list, dim=1)  # [B, T, N]
    write_attn_all = torch.stack(write_attn_list, dim=1)  # [B, T, N]

    return h_work_all, h_tape, read_attn_all, write_attn_all


class E26DualMemoryElmanFunction(torch.autograd.Function):
    """Autograd function for E26 with CUDA kernel support."""

    @staticmethod
    def forward(ctx, training, x_seq, h_tape_init, h_work_init,
                W_h, W_x, b_h, W_write, use_cuda=True):
        B, T, D = x_seq.shape
        N = h_tape_init.shape[1]
        scale = 1.0 / math.sqrt(D)

        # Check for CUDA kernel
        use_cuda_kernel = False
        if use_cuda and x_seq.is_cuda:
            try:
                import hasty_pytorch_lib
                if hasattr(hasty_pytorch_lib, 'e26_parallel_forward'):
                    use_cuda_kernel = True
            except ImportError:
                pass

        if use_cuda_kernel:
            # CUDA path
            import hasty_pytorch_lib
            h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
                hasty_pytorch_lib.e26_parallel_forward(
                    training, x_seq, h_tape_init, h_work_init,
                    W_h, W_x, b_h, W_write
                )
            ctx.use_cuda = True
        else:
            # Python fallback
            h_tape = h_tape_init.clone()
            h_work = h_work_init.clone()

            h_work_list = []
            h_tape_list = [h_tape_init]
            read_attn_list = []
            write_attn_list = []

            for t in range(T):
                h_work, h_tape, read_attn, write_attn = e26_forward_step_python(
                    x_seq[:, t, :] @ W_x.T, h_tape, h_work, W_h, b_h, W_write, scale
                )
                h_work_list.append(h_work)
                h_tape_list.append(h_tape)
                read_attn_list.append(read_attn)
                write_attn_list.append(write_attn)

            h_work_all = torch.stack(h_work_list, dim=1)  # [B, T, D]
            h_tape_all = torch.stack(h_tape_list, dim=1)  # [B, T+1, N, D]
            read_attn_all = torch.stack(read_attn_list, dim=1)  # [B, T, N]
            write_attn_all = torch.stack(write_attn_list, dim=1)  # [B, T, N]
            h_tape_final = h_tape_all[:, -1]
            ctx.use_cuda = False

        ctx.save_for_backward(x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                              W_h, W_x, b_h, W_write, h_work_init)
        ctx.scale = scale

        return h_work_all, h_tape_final

    @staticmethod
    def backward(ctx, d_h_work_all, d_h_tape_final):
        x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, \
            W_h, W_x, b_h, W_write, h_work_init = ctx.saved_tensors
        scale = ctx.scale

        B, T, D = x_seq.shape
        N = h_tape_all.shape[2]

        if ctx.use_cuda:
            # CUDA backward
            import hasty_pytorch_lib
            dx, dW_h, dW_x, db_h, dW_write = \
                hasty_pytorch_lib.e26_parallel_backward(
                    x_seq.contiguous(), h_work_all.contiguous(), h_work_init.contiguous(),
                    h_tape_all.contiguous(), read_attn_all.contiguous(), write_attn_all.contiguous(),
                    W_h.contiguous(), W_x.contiguous(), W_write.contiguous(),
                    d_h_work_all.contiguous(), d_h_tape_final.contiguous()
                )
        else:
            # Python backward
            dx, dW_h, dW_x, db_h, dW_write = e26_backward_python(
                x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                W_h, W_x, b_h, W_write, h_work_init, d_h_work_all, d_h_tape_final, scale
            )

        return None, dx, None, None, dW_h, dW_x, db_h, dW_write, None


def e26_backward_python(
    x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
    W_h, W_x, b_h, W_write, h_work_init, d_h_work_all, d_h_tape_final, scale
):
    """Python reference backward pass for E26."""
    B, T, D = x_seq.shape
    N = h_tape_all.shape[2]

    # Initialize gradients
    dW_h = torch.zeros_like(W_h)
    dW_x = torch.zeros_like(W_x)
    db_h = torch.zeros_like(b_h)
    dW_write = torch.zeros_like(W_write)
    dx = torch.zeros_like(x_seq)

    d_h_tape = d_h_tape_final.clone()
    d_h_work = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

    for t in range(T - 1, -1, -1):
        h_work_t = h_work_all[:, t]
        h_tape_t = h_tape_all[:, t]
        read_attn = read_attn_all[:, t]
        write_attn = write_attn_all[:, t]

        if t > 0:
            h_work_prev = h_work_all[:, t - 1]
        else:
            h_work_prev = h_work_init

        d_h_work_t = d_h_work_all[:, t] + d_h_work

        # === BACKWARD THROUGH TAPE WRITE ===
        d_write_value = (d_h_tape * write_attn[:, :, None]).sum(dim=1)

        write_value = h_work_t @ W_write.T
        d_write_attn = (d_h_tape * (write_value[:, None, :] - h_tape_t)).sum(dim=-1)

        d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])

        dW_write += d_write_value.T @ h_work_t
        d_h_work_from_write = d_write_value @ W_write

        # === SOFTMAX BACKWARD FOR WRITE ATTENTION ===
        # softmax backward: dz = p * (dp - sum(p * dp))
        p_dp_sum = (write_attn * d_write_attn).sum(dim=-1, keepdim=True)
        d_write_scores = write_attn * (d_write_attn - p_dp_sum) * scale

        d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)
        d_h_tape_from_write_attn = d_write_scores[:, :, None] * h_work_t[:, None, :]

        d_h_work_t_total = d_h_work_t + d_h_work_from_write + d_h_work_from_write_attn

        # === BACKWARD THROUGH WORKING MEMORY UPDATE ===
        d_pre_act = d_h_work_t_total * (1 - h_work_t ** 2)
        d_read = d_pre_act

        dW_h += d_pre_act.T @ h_work_prev
        d_h_work = d_pre_act @ W_h
        dW_x += d_pre_act.T @ x_seq[:, t]
        dx[:, t] = d_pre_act @ W_x
        db_h += d_pre_act.sum(dim=0)

        # === BACKWARD THROUGH READ ===
        d_read_attn = (d_read[:, None, :] * h_tape_t).sum(dim=-1)
        d_h_tape_from_read = d_read[:, None, :] * read_attn[:, :, None]

        # === SOFTMAX BACKWARD FOR READ ATTENTION ===
        p_dp_sum_read = (read_attn * d_read_attn).sum(dim=-1, keepdim=True)
        d_read_scores = read_attn * (d_read_attn - p_dp_sum_read) * scale

        d_h_work_from_read_attn = (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)
        d_h_work += d_h_work_from_read_attn
        d_h_tape_from_read_attn = d_read_scores[:, :, None] * h_work_prev[:, None, :]

        d_h_tape = d_h_tape_pre_write + d_h_tape_from_write_attn + d_h_tape_from_read + d_h_tape_from_read_attn

    return dx, dW_h, dW_x, db_h, dW_write


class E26DualMemoryElmanCell(nn.Module):
    """Single E26 cell for stacking."""

    def __init__(self, dim: int, n_slots: int = 8):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.scale = 1.0 / (dim ** 0.5)

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.9)
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_write)

    def forward(
        self,
        x: torch.Tensor,
        h_tape: Optional[torch.Tensor] = None,
        h_work: Optional[torch.Tensor] = None,
        use_cuda: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Handle both [B, T, D] and [T, B, D] formats
        if x.dim() == 3:
            if x.shape[0] > x.shape[1]:
                x = x.permute(1, 0, 2).contiguous()

        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        h_work_all, h_tape_final = E26DualMemoryElmanFunction.apply(
            self.training,
            x.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            self.W_h.contiguous(),
            self.W_x.contiguous(),
            self.b_h.contiguous(),
            self.W_write.contiguous(),
            use_cuda
        )

        h_work_final = h_work_all[:, -1]
        return h_work_all, h_tape_final, h_work_final


class E26DualMemoryElmanLM(nn.Module):
    """Full E26 model with embedding and output projection (standalone)."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        n_slots: int = 8,
        depth: int = 1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            E26DualMemoryElmanCell(dim, n_slots) for _ in range(depth)
        ])
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight  # Weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h, _, _ = layer(h)
        return self.out_proj(h)


class E26DualMemoryElman(nn.Module):
    """E26 layer compatible with LadderLM (Mamba2-style projections)."""

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_slots: int = 64,
        dropout: float = 0.0,
        w_h_init_scale: float = 0.9,
        mamba2_init: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots

        # Mamba2-style: project to 2*d_inner, then split (x and gate z)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # E26 cell
        self.cell = E26DualMemoryElmanCell(self.d_inner, n_slots)

        # Override W_h init scale
        with torch.no_grad():
            W_h_f32 = torch.empty_like(self.cell.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_f32)
            W_h_f32.mul_(w_h_init_scale)
            self.cell.W_h.copy_(W_h_f32.to(self.cell.W_h.dtype))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init: bool):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input
            h0: tuple of (h_tape [B, N, d_inner], h_work [B, d_inner]) or None

        Returns:
            output: [B, T, dim]
            h_final: tuple of (h_tape_final, h_work_final)
        """
        B, T, D = x.shape

        # Project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # E26 cell
        if h0 is not None:
            h_tape, h_work = h0
        else:
            h_tape, h_work = None, None

        h_work_all, h_tape_final, h_work_final = self.cell(x_proj, h_tape, h_work)

        # Gating with z (like E1/Mamba)
        output = h_work_all * F.silu(z)  # [B, T, d_inner]

        # Output projection
        output = self.dropout(output)
        output = self.out_proj(output)  # [B, T, dim]

        return output, (h_tape_final, h_work_final)

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, LEVEL=26_PARALLEL_DUAL_MEMORY'


if __name__ == '__main__':
    # Quick test
    B, T, D, N = 2, 8, 256, 8
    device = 'cuda'
    dtype = torch.bfloat16

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    W_h = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    W_x = torch.randn(D, D, device=device, dtype=dtype) * 0.01
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.01

    # Make W_h well-conditioned
    with torch.no_grad():
        W_h_f32 = torch.empty_like(W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_f32)
        W_h.copy_(W_h_f32.to(dtype) * 0.9)

    h_work_all, h_tape_final, read_attn, write_attn = e26_forward_python(
        x, h_tape, h_work, W_h, W_x, b_h, W_write
    )

    print(f"E26 Python forward test:")
    print(f"  h_work_all shape: {h_work_all.shape}")
    print(f"  h_tape_final shape: {h_tape_final.shape}")
    print(f"  read_attn shape: {read_attn.shape}")
    print(f"  write_attn shape: {write_attn.shape}")
    print(f"  h_work_all stats: min={h_work_all.min():.4f}, max={h_work_all.max():.4f}")
    print(f"  PASS" if not torch.isnan(h_work_all).any() else "  FAIL: NaN")
