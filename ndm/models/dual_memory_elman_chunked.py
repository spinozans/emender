"""
E23c: Chunked Dual-Memory Elman

Key insight: Decouple h_work evolution from tape reads to enable batching.

Architecture:
    h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)  # Pure RNN, no read!
    read_t = attention(h_work_t, tape)                   # Batched per chunk
    output_t = h_work_t + read_t                         # Additive read
    tape = write(h_work_chunk, tape)                     # Update at chunk boundary

Chunked processing (K timesteps per chunk):
    1. Pre-compute h_work for K steps (sequential RNN, but fuseable)
    2. Batch ALL read attentions: [K*B, D] @ [D, N] - ONE BIG GEMM
    3. Apply writes to tape (can also be batched within chunk)

Benefits:
    - Read attention: T tiny GEMMs -> T/K big GEMMs
    - Better tensor core utilization
    - Reduced kernel launch overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DualMemoryElmanChunkedCell(nn.Module):
    """
    E23c cell with chunked attention.

    Key difference from E23: h_work doesn't depend on read.
    This allows batching reads across timesteps within a chunk.
    """

    def __init__(self, dim, n_slots=64, chunk_size=64, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.chunk_size = chunk_size

        # RNN weights (no read in h_work update)
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_x = nn.Linear(dim, dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(dim))

        # Write projection
        self.W_write = nn.Linear(dim, dim, bias=False)

        # Initialize W_h with controlled spectral radius
        with torch.no_grad():
            W_h_fp32 = torch.empty(dim, dim, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(w_h_init_scale)
            self.W_h.weight.copy_(W_h_fp32.to(self.W_h.weight.dtype))

        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_write.weight)

        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x_seq, h_tape=None, h_work=None):
        """
        Forward pass with chunked attention.

        Args:
            x_seq: [B, T, D] input sequence
            h_tape: [B, N, D] initial tape state
            h_work: [B, D] initial working memory

        Returns:
            output: [B, T, D] outputs (h_work + read)
            h_tape_final: [B, N, D] final tape state
            h_work_final: [B, D] final working memory
        """
        B, T, D = x_seq.shape
        N = self.n_slots
        K = self.chunk_size
        device = x_seq.device
        dtype = x_seq.dtype

        # Initialize states
        if h_tape is None:
            h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        # Pre-compute W_x @ x for all timesteps (one big GEMM)
        Wx_all = self.W_x(x_seq)  # [B, T, D]

        outputs = []

        # Process in chunks
        for chunk_start in range(0, T, K):
            chunk_end = min(chunk_start + K, T)
            chunk_len = chunk_end - chunk_start

            # ============================================================
            # Step 1: Pre-compute h_work for this chunk (sequential RNN)
            # h_work_t = tanh(W_h @ h_work_{t-1} + W_x @ x_t + b)
            # ============================================================
            h_work_chunk = []
            h_work_cur = h_work

            for t in range(chunk_start, chunk_end):
                Rh = self.W_h(h_work_cur)  # [B, D]
                h_work_cur = torch.tanh(Rh + Wx_all[:, t] + self.b_h)
                h_work_chunk.append(h_work_cur)

            # Stack: [chunk_len, B, D]
            H_chunk = torch.stack(h_work_chunk, dim=0)

            # ============================================================
            # Step 2: Batch ALL read attentions from frozen tape
            # Use efficient batching: [B, K, D] @ [B, D, N] -> [B, K, N]
            # This avoids expanding tape K times!
            # ============================================================
            # Permute H_chunk: [K, B, D] -> [B, K, D]
            H_perm = H_chunk.permute(1, 0, 2).contiguous()

            # Attention scores: [B, K, D] @ [B, D, N] -> [B, K, N]
            tape_t = h_tape.transpose(1, 2).contiguous()  # [B, D, N]
            read_scores = torch.bmm(H_perm, tape_t) * self.scale  # [B, K, N]

            # Softmax over N
            read_attn = F.softmax(read_scores, dim=-1)  # [B, K, N]

            # Weighted read: [B, K, N] @ [B, N, D] -> [B, K, D]
            read_vals = torch.bmm(read_attn, h_tape)  # [B, K, D]

            # Permute back: [B, K, D] -> [K, B, D]
            read_vals = read_vals.permute(1, 0, 2)  # [chunk_len, B, D]

            # ============================================================
            # Step 3: Compute outputs (h_work + read)
            # ============================================================
            chunk_output = H_chunk + read_vals  # [chunk_len, B, D]
            outputs.append(chunk_output)

            # ============================================================
            # Step 4: Update tape with writes from this chunk
            # OPTIMIZED: Batch compute all writes, parallel tape update
            # ============================================================
            # Batch compute write_vals: [chunk_len, B, D]
            H_flat = H_chunk.view(chunk_len * B, D)
            write_vals = self.W_write(H_flat).view(chunk_len, B, D)

            # Batch compute write attentions from INITIAL tape (frozen)
            # [B, K, D] @ [B, D, N] -> [B, K, N]
            write_scores = torch.bmm(H_perm, tape_t) * self.scale  # [B, K, N]
            write_attn = F.softmax(write_scores, dim=-1)  # [B, K, N]
            write_attn = write_attn.permute(1, 0, 2)  # [K, B, N]

            # Parallel tape update formula:
            # tape_K = tape_0 * prod(1-a_i) + sum_i(v_i * a_i * prod_{j>i}(1-a_j))
            one_minus_a = 1 - write_attn  # [K, B, N]

            # Cumulative product from the END
            cumprod_rev = torch.flip(
                torch.cumprod(torch.flip(one_minus_a, [0]), dim=0),
                [0]
            )  # [K, B, N], cumprod_rev[i] = prod_{j=i}^{K-1}(1-a_j)

            # Coefficient for tape_0: prod_{i=0}^{K-1}(1-a_i)
            tape_0_coeff = cumprod_rev[0]  # [B, N]

            # Coefficients for v_i: a_i * prod_{j=i+1}^{K-1}(1-a_j)
            cumprod_shifted = torch.cat([
                cumprod_rev[1:],
                torch.ones(1, B, N, device=device, dtype=dtype)
            ], dim=0)  # [K, B, N]
            v_coeffs = write_attn * cumprod_shifted  # [K, B, N]

            # Compute final tape
            tape_0_term = h_tape * tape_0_coeff.unsqueeze(-1)  # [B, N, D]
            v_contrib = (write_vals.unsqueeze(2) * v_coeffs.unsqueeze(-1)).sum(dim=0)  # [B, N, D]
            h_tape = tape_0_term + v_contrib

            # Update h_work for next chunk
            h_work = h_work_cur

        # Concatenate outputs: [T, B, D] -> [B, T, D]
        output = torch.cat(outputs, dim=0).permute(1, 0, 2)

        return output, h_tape, h_work


class DualMemoryElmanChunked(nn.Module):
    """
    E23c layer with chunked attention and Mamba2-style wrapping.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_slots=64,
        chunk_size=64,
        w_h_init_scale=0.9,
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots
        self.chunk_size = chunk_size

        # Projections (Mamba2-style)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # RNN cell
        self.cell = DualMemoryElmanChunkedCell(
            dim=self.d_inner,
            n_slots=n_slots,
            chunk_size=chunk_size,
            w_h_init_scale=w_h_init_scale,
        )

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

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Parse hidden state
        if hidden is not None:
            h_tape, h_work = hidden
        else:
            h_tape = None
            h_work = None

        # Run chunked cell
        output, h_tape_final, h_work_final = self.cell(x_proj, h_tape, h_work)

        # Gate with z (Mamba2-style)
        output = output * F.silu(z)

        # Project back
        output = self.out_proj(output)

        return output, (h_tape_final, h_work_final)

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, chunk_size={self.chunk_size}, LEVEL=23c_CHUNKED'


if __name__ == '__main__':
    import time

    B, T, D, N = 32, 512, 768, 64
    K = 64  # chunk size
    device = 'cuda'
    dtype = torch.bfloat16

    print("=" * 70)
    print(f"E23c Chunked vs E23 Original: B={B}, T={T}, D={D}, N={N}, K={K}")
    print("=" * 70)

    # Create cells
    chunked_cell = DualMemoryElmanChunkedCell(dim=D, n_slots=N, chunk_size=K).to(device).to(dtype)

    from dual_memory_elman import DualMemoryElmanCell
    original_cell = DualMemoryElmanCell(dim=D, n_slots=N).to(device).to(dtype)

    # Copy weights for fair comparison
    with torch.no_grad():
        chunked_cell.W_h.weight.copy_(original_cell.W_h)
        chunked_cell.W_x.weight.copy_(original_cell.W_x)
        chunked_cell.b_h.copy_(original_cell.b_h)
        chunked_cell.W_write.weight.copy_(original_cell.W_write)

    # Test data
    x_seq = torch.randn(B, T, D, device=device, dtype=dtype)
    h_tape = torch.zeros(B, N, D, device=device, dtype=dtype)
    h_work = torch.zeros(B, D, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = chunked_cell(x_seq, h_tape.clone(), h_work.clone())
            _ = original_cell(x_seq, h_tape.clone(), h_work.clone())
    torch.cuda.synchronize()

    # Benchmark chunked
    n_iters = 10
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = chunked_cell(x_seq, h_tape.clone(), h_work.clone())
    torch.cuda.synchronize()
    chunked_time = (time.perf_counter() - t0) / n_iters * 1000

    # Benchmark original
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = original_cell(x_seq, h_tape.clone(), h_work.clone())
    torch.cuda.synchronize()
    orig_time = (time.perf_counter() - t0) / n_iters * 1000

    print(f"\nChunked (K={K}): {chunked_time:.1f}ms ({B*T/(chunked_time/1000)/1000:.1f}K tok/s)")
    print(f"Original: {orig_time:.1f}ms ({B*T/(orig_time/1000)/1000:.1f}K tok/s)")
    print(f"Speedup: {orig_time/chunked_time:.2f}x")

    # Verify outputs are different (expected - different architecture!)
    with torch.no_grad():
        out_chunked, _, _ = chunked_cell(x_seq, h_tape.clone(), h_work.clone())
        out_orig, _, _ = original_cell(x_seq, h_tape.clone(), h_work.clone())
    print(f"\nNote: Outputs differ (expected - h_work doesn't use read in E23c)")
    print(f"Output diff: {torch.mean(torch.abs(out_chunked - out_orig)).item():.4f}")
