"""
E23 Dual-Memory Elman - Optimized Attention Version

Uses torch.bmm (cuBLAS batched GEMM) for attention operations instead of einsum.
This provides ~2-4x speedup on attention by using tensor cores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DualMemoryElmanCellOptimized(nn.Module):
    """
    E23 cell with optimized attention using batched GEMM.

    Uses bmm for attention score computation and weighted read/write.
    """

    def __init__(self, dim, n_slots=64, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots

        # Weights
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_write = nn.Linear(dim, dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(dim))

        # Initialize W_h with small spectral radius
        with torch.no_grad():
            nn.init.orthogonal_(self.W_h.weight)
            self.W_h.weight.mul_(w_h_init_scale)

        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x_seq, h_tape=None, h_work=None):
        """
        Forward pass with optimized attention.

        Args:
            x_seq: [B, T, D] input sequence
            h_tape: [B, N, D] initial tape state (optional)
            h_work: [B, D] initial working memory (optional)

        Returns:
            h_work_all: [B, T, D] all working memory states
            h_tape_final: [B, N, D] final tape state
            h_work_final: [B, D] final working memory
        """
        B, T, D = x_seq.shape
        N = self.n_slots
        device = x_seq.device
        dtype = x_seq.dtype

        # Initialize states
        if h_tape is None:
            h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        # Precompute W_x @ x for all timesteps
        Wx_all = self.W_x(x_seq)  # [B, T, D]

        # Precompute transposed tape for bmm
        # h_tape is [B, N, D], we need [B, D, N] for score computation
        h_tape_T = h_tape.transpose(1, 2).contiguous()  # [B, D, N]

        h_work_list = []

        for t in range(T):
            # ==================================================================
            # 1. W_h @ h_work
            # ==================================================================
            Rh = self.W_h(h_work)  # [B, D]

            # ==================================================================
            # 2. Read attention using bmm (optimized)
            # ==================================================================
            # scores = h_work @ h_tape^T -> [B, N]
            # Using bmm: [B, 1, D] @ [B, D, N] -> [B, 1, N] -> [B, N]
            read_scores = torch.bmm(h_work.unsqueeze(1), h_tape_T).squeeze(1) * self.scale
            read_attn = F.softmax(read_scores, dim=-1)  # [B, N]

            # read = attn @ h_tape -> [B, D]
            # Using bmm: [B, 1, N] @ [B, N, D] -> [B, 1, D] -> [B, D]
            read = torch.bmm(read_attn.unsqueeze(1), h_tape).squeeze(1)

            # ==================================================================
            # 3. Update working memory
            # ==================================================================
            h_work = torch.tanh(Rh + Wx_all[:, t] + read + self.b_h)
            h_work_list.append(h_work)

            # ==================================================================
            # 4. W_write @ h_work_new -> write_val
            # ==================================================================
            write_val = self.W_write(h_work)  # [B, D]

            # ==================================================================
            # 5. Write attention using bmm (optimized)
            # ==================================================================
            # scores = h_work_new @ h_tape^T -> [B, N]
            write_scores = torch.bmm(h_work.unsqueeze(1), h_tape_T).squeeze(1) * self.scale
            write_attn = F.softmax(write_scores, dim=-1)  # [B, N]

            # ==================================================================
            # 6. Update tape
            # ==================================================================
            # h_tape = (1 - attn) * h_tape + attn * write_val
            h_tape = (1 - write_attn.unsqueeze(-1)) * h_tape + write_attn.unsqueeze(-1) * write_val.unsqueeze(1)

            # Update transposed view for next iteration
            h_tape_T = h_tape.transpose(1, 2).contiguous()

        h_work_all = torch.stack(h_work_list, dim=1)  # [B, T, D]

        return h_work_all, h_tape, h_work


class DualMemoryElmanOptimized(nn.Module):
    """
    E23 layer with optimized attention.

    Wraps DualMemoryElmanCellOptimized with input/output projections.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_slots=64,
        w_h_init_scale=0.9,
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots

        # Projections
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # RNN cell
        self.cell = DualMemoryElmanCellOptimized(
            dim=self.d_inner,
            n_slots=n_slots,
            w_h_init_scale=w_h_init_scale,
        )

        # Output gate
        self.gate_proj = nn.Linear(dim, self.d_inner, bias=False)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [B, T, dim] input
            hidden: tuple of (h_tape, h_work) or None

        Returns:
            output: [B, T, dim]
            hidden: tuple of (h_tape, h_work)
        """
        B, T, _ = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Parse hidden state
        if hidden is not None:
            h_tape, h_work = hidden
        else:
            h_tape = None
            h_work = None

        # Run cell
        h_work_all, h_tape_final, h_work_final = self.cell(x_proj, h_tape, h_work)

        # Output gating
        gate = F.silu(self.gate_proj(x))  # [B, T, d_inner]
        output = self.out_proj(h_work_all * gate)  # [B, T, dim]

        return output, (h_tape_final, h_work_final)

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, LEVEL=23_OPTIMIZED'


if __name__ == '__main__':
    import time

    B, T, D, N = 64, 512, 512, 32
    device = 'cuda'
    dtype = torch.bfloat16

    print("=" * 70)
    print(f"E23 Optimized vs Original: B={B}, T={T}, D={D}, N={N}")
    print("=" * 70)

    # Create cells
    opt_cell = DualMemoryElmanCellOptimized(dim=D, n_slots=N).to(device).to(dtype)

    from dual_memory_elman import DualMemoryElmanCell
    orig_cell = DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

    # Test data
    x_seq = torch.randn(B, T, D, device=device, dtype=dtype)
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = opt_cell(x_seq, h_tape.clone(), h_work.clone())
            _ = orig_cell(x_seq, h_tape.clone(), h_work.clone(), use_cuda=True)
    torch.cuda.synchronize()

    # Benchmark optimized
    n_iters = 10
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = opt_cell(x_seq, h_tape.clone(), h_work.clone())
    torch.cuda.synchronize()
    opt_time = (time.perf_counter() - t0) / n_iters * 1000

    # Benchmark original
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = orig_cell(x_seq, h_tape.clone(), h_work.clone(), use_cuda=True)
    torch.cuda.synchronize()
    orig_time = (time.perf_counter() - t0) / n_iters * 1000

    print(f"\nOptimized (bmm): {opt_time:.1f}ms ({B*T/(opt_time/1000)/1000:.1f}K tok/s)")
    print(f"Original (CUDA): {orig_time:.1f}ms ({B*T/(orig_time/1000)/1000:.1f}K tok/s)")
    print(f"Speedup: {orig_time/opt_time:.2f}x")
