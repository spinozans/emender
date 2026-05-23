"""
E6: Diagonal Elman - Per-channel scalar recurrence + low-rank mixing.

Architecture:
    h_t = sigmoid(a) * h_{t-1} + (1 - sigmoid(a)) * x_t  # per-channel EMA
    y_t = U @ V @ h_t * silu(x_t)  # low-rank cross-channel mix

Key insight: Diagonal recurrence is O(dim), mixing is O(dim * rank).
Allows MASSIVE depth: 743 layers at 50M with rank=64!

This is essentially a learnable EMA per channel with cross-channel mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    import hasty_pytorch_lib
    HASTY_AVAILABLE = True
except ImportError:
    HASTY_AVAILABLE = False


class DiagonalElmanFunction(Function):
    """CUDA-backed E6 forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, gate_logit, U, V, training):
        h, output = hasty_pytorch_lib.diagonal_elman_forward(
            training, x, h0, gate_logit, U, V
        )
        if training:
            ctx.save_for_backward(gate_logit, U, V, x, h)
        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_h):
        gate_logit, U, V, x, h = ctx.saved_tensors
        dx, d_gate_logit, dU, dV = hasty_pytorch_lib.diagonal_elman_backward(
            gate_logit, U, V, x, h, grad_output.contiguous()
        )
        return dx, None, d_gate_logit, dU, dV, None


class DiagonalElmanCell(nn.Module):
    """
    Diagonal Elman cell - per-channel scalar recurrence.

    h[i]_t = gate[i] * h[i]_{t-1} + (1 - gate[i]) * x[i]_t
    """

    def __init__(self, dim, rank=64, gate_init=-2.0):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # Per-channel gate (controls decay vs input)
        # Initialized to favor input slightly
        self.gate_logit = nn.Parameter(torch.full((dim,), gate_init))

        # Low-rank cross-channel mixing
        self.U = nn.Parameter(torch.empty(dim, rank))
        self.V = nn.Parameter(torch.empty(rank, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        with torch.no_grad():
            self.U.mul_(0.5)
            self.V.mul_(0.5)

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states (including h0)
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # CUDA kernel path
        if HASTY_AVAILABLE and x.is_cuda:
            output, h = DiagonalElmanFunction.apply(
                x, h0, self.gate_logit, self.U, self.V, self.training
            )
            return output, h

        # PyTorch fallback
        # Compute gates once
        gate = torch.sigmoid(self.gate_logit)  # [dim]

        h_list = [h0]
        out_list = []
        h_prev = h0

        for t in range(T):
            # Per-channel gated update (like EMA)
            h_new = gate * h_prev + (1 - gate) * x[t]
            h_list.append(h_new)

            # Low-rank cross-channel mixing
            # h -> V -> rank -> U -> dim
            Vh = h_new @ self.V.T  # [B, rank]
            mixed = Vh @ self.U.T  # [B, dim]

            # Gate with input
            silu_gate = F.silu(x[t])
            output = mixed * silu_gate
            out_list.append(output)

            h_prev = h_new

        h = torch.stack(h_list, dim=0)
        output = torch.stack(out_list, dim=0)
        return output, h


class DiagonalElman(nn.Module):
    """
    E6: Diagonal Elman layer.

    Per-channel recurrence + low-rank cross-channel mixing.
    Super cheap: ~67k params/layer -> 743 layers at 50M!
    """

    def __init__(
        self,
        dim,
        rank=None,
        gate_init=-2.0,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = dim

        # Default rank
        if rank is None:
            rank = max(16, dim // 8)
        self.rank = rank

        # Diagonal recurrence + mixing cell
        self.cell = DiagonalElmanCell(dim, rank=rank, gate_init=gate_init)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, dim] final hidden state
        """
        B, T, D = x.shape

        # Transpose for cell [B, T, D] -> [T, B, D]
        x_rnn = x.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, h0)
        h_final = h_all[-1]

        # Transpose back [T, B, D] -> [B, T, D]
        output = cell_out.permute(1, 0, 2).contiguous()
        output = self.dropout(output)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, rank={self.rank}, LEVEL=6'


# For compatibility with model registry
DIAGONAL_ELMAN_AVAILABLE = True


if __name__ == "__main__":
    print("Testing DiagonalElman (E6)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dim = 512
    model = DiagonalElman(dim=dim, rank=64).to(device).bfloat16()
    x = torch.randn(32, 512, dim, device=device, dtype=torch.bfloat16)

    print(f"Model: {model.extra_repr()}")
    print(f"CUDA kernel: {HASTY_AVAILABLE}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Compare depths
    print(f"\nFor 50M model:")
    embed = 256 * dim
    target = 50_000_000
    depth = (target - embed) // params
    print(f"  Depth: {depth} layers (!)")
    print(f"  Compare to E1: ~38 layers")

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.mean()
    loss.backward()
    print("Backward passed!")

    # Benchmark
    import time

    model = DiagonalElman(dim=dim, rank=64).to(device).bfloat16()
    x = torch.randn(32, 512, dim, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        out, h = model(x)
        out.mean().backward()
        model.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        out, h = model(x)
        out.mean().backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 20 * 1000

    tok_per_sec = 32 * 512 / (elapsed / 1000)
    print(f"\nBenchmark: {elapsed:.1f}ms, {tok_per_sec / 1e3:.1f}k tok/s")
