"""
E5: Pure Low-Rank Elman - No projections, all low-rank on full dim.

Architecture:
    h_t = tanh(U_h @ V_h @ h_{t-1} + U_x @ V_x @ x_t + b)
    y_t = h_t * silu(U_z @ V_z @ x_t)

Key insight: No in_proj/out_proj. Hidden state IS dim.
All matrices factored as U @ V (low-rank).

With rank=64, dim=512:
- 197k params/layer (vs 1.3M for E1)
- 252 layers for 50M (vs 38 for E1)
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


class PureLowRankElmanFunction(Function):
    """CUDA-backed E5 forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b, training):
        h, output, v = hasty_pytorch_lib.pure_lowrank_elman_forward(
            training, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b
        )
        if training:
            ctx.save_for_backward(U_h, V_h, U_x, V_x, U_z, V_z, x, h, v)
        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_h):
        U_h, V_h, U_x, V_x, U_z, V_z, x, h, v = ctx.saved_tensors
        dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db = hasty_pytorch_lib.pure_lowrank_elman_backward(
            U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, grad_output.contiguous()
        )
        return dx, None, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db, None


class PureLowRankElmanFusedFunction(Function):
    """CUDA-backed E5 forward/backward with fused kernel optimization.

    Fuses tanh + gate into single kernel, reducing kernel launches by 25%.
    """

    @staticmethod
    def forward(ctx, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b, training):
        h, output, v = hasty_pytorch_lib.pure_lowrank_elman_forward_fused(
            training, x, h0, U_h, V_h, U_x, V_x, U_z, V_z, b
        )
        if training:
            ctx.save_for_backward(U_h, V_h, U_x, V_x, U_z, V_z, x, h, v)
        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_h):
        U_h, V_h, U_x, V_x, U_z, V_z, x, h, v = ctx.saved_tensors
        dx, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db = hasty_pytorch_lib.pure_lowrank_elman_backward_fused(
            U_h, V_h, U_x, V_x, U_z, V_z, x, h, v, grad_output.contiguous()
        )
        return dx, None, dU_h, dV_h, dU_x, dV_x, dU_z, dV_z, db, None


class CUDAGraphCache:
    """Cache for CUDA graphs keyed by (batch_size, seq_len)."""

    def __init__(self):
        self.graphs = {}  # (B, T) -> (graph, static_inputs, static_outputs)

    def get(self, key):
        return self.graphs.get(key)

    def set(self, key, value):
        self.graphs[key] = value

    def clear(self):
        self.graphs.clear()


# Global graph cache (per-module caching handled via module attribute)
_global_graph_cache = CUDAGraphCache()


class PureLowRankElmanCell(nn.Module):
    """
    Pure low-rank Elman cell.

    h_t = tanh(U_h @ V_h @ h_{t-1} + U_x @ V_x @ x_t + b)
    """

    def __init__(self, dim, rank=64, use_fused=True, use_cuda_graph=False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.use_fused = use_fused
        self.use_cuda_graph = use_cuda_graph

        # Low-rank recurrence: W_h ≈ U_h @ V_h
        self.U_h = nn.Parameter(torch.empty(dim, rank))
        self.V_h = nn.Parameter(torch.empty(rank, dim))

        # Low-rank input: W_x ≈ U_x @ V_x
        self.U_x = nn.Parameter(torch.empty(dim, rank))
        self.V_x = nn.Parameter(torch.empty(rank, dim))

        # Low-rank gate: W_z ≈ U_z @ V_z
        self.U_z = nn.Parameter(torch.empty(dim, rank))
        self.V_z = nn.Parameter(torch.empty(rank, dim))

        self.b = nn.Parameter(torch.zeros(dim))

        # CUDA Graph state
        self._graph_cache = {}  # (B, T, training) -> (graph, static_x, static_h0, static_out, static_h)
        self._graphed_forward = None

        self._init_weights()

    def _init_weights(self):
        # Initialize for stable gradients
        for U, V in [(self.U_h, self.V_h), (self.U_x, self.V_x), (self.U_z, self.V_z)]:
            nn.init.orthogonal_(U)
            nn.init.orthogonal_(V)
            # Scale so U @ V has reasonable norm
            with torch.no_grad():
                U.mul_(0.5)
                V.mul_(0.5)

    def _run_kernel(self, x, h0):
        """Run the CUDA kernel (fused or non-fused)."""
        if self.use_fused:
            return PureLowRankElmanFusedFunction.apply(
                x, h0, self.U_h, self.V_h, self.U_x, self.V_x,
                self.U_z, self.V_z, self.b, self.training
            )
        else:
            return PureLowRankElmanFunction.apply(
                x, h0, self.U_h, self.V_h, self.U_x, self.V_x,
                self.U_z, self.V_z, self.b, self.training
            )

    def _run_with_cuda_graph(self, x, h0):
        """Run forward pass with CUDA Graph capture/replay.

        Note: Returns views of static buffers. Caller must not modify outputs
        after subsequent calls, or clone if needed.
        """
        T, B, D = x.shape
        cache_key = (B, T, self.training)

        if cache_key not in self._graph_cache:
            # First call with this shape - capture the graph
            # Allocate static buffers
            static_x = torch.empty_like(x)
            static_h0 = torch.empty_like(h0)

            # Warmup run (required before capture)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                static_x.copy_(x)
                static_h0.copy_(h0)
                static_out, static_h = self._run_kernel(static_x, static_h0)
            torch.cuda.current_stream().wait_stream(s)

            # Capture the graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out, static_h = self._run_kernel(static_x, static_h0)

            self._graph_cache[cache_key] = (graph, static_x, static_h0, static_out, static_h)

        # Replay the graph
        graph, static_x, static_h0, static_out, static_h = self._graph_cache[cache_key]
        static_x.copy_(x)
        static_h0.copy_(h0)
        graph.replay()

        # Return views - no clone for maximum performance
        # If caller needs to preserve outputs across calls, they should clone
        return static_out, static_h

    def forward(self, x, h0=None):
        """
        Args:
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # CUDA kernel path
        if HASTY_AVAILABLE and x.is_cuda:
            # CUDA Graph path (inference only - training needs gradient flow)
            if self.use_cuda_graph and not self.training:
                return self._run_with_cuda_graph(x, h0)
            # Standard kernel path
            return self._run_kernel(x, h0)

        # PyTorch fallback
        h_list = [h0]
        out_list = []

        h_prev = h0
        for t in range(T):
            x_t = x[t]

            # Low-rank recurrence
            Vh = h_prev @ self.V_h.T  # [B, rank]
            Uh = Vh @ self.U_h.T      # [B, dim]

            # Low-rank input
            Vx = x_t @ self.V_x.T     # [B, rank]
            Ux = Vx @ self.U_x.T      # [B, dim]

            # Combine and activate
            pre = Uh + Ux + self.b
            h_new = torch.tanh(pre)
            h_list.append(h_new)

            # Low-rank gate
            Vz = x_t @ self.V_z.T     # [B, rank]
            Uz = Vz @ self.U_z.T      # [B, dim]
            gate = F.silu(Uz)

            # Gated output
            out = h_new * gate
            out_list.append(out)

            h_prev = h_new

        h = torch.stack(h_list, dim=0)
        output = torch.stack(out_list, dim=0)
        return output, h


class PureLowRankElman(nn.Module):
    """
    E5: Pure Low-Rank Elman layer.

    No projections - hidden state IS dim.
    All operations are low-rank.
    """

    def __init__(
        self,
        dim,
        rank=None,
        dropout=0.0,
        use_fused=True,
        use_cuda_graph=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = dim  # No expansion!
        self.use_fused = use_fused
        self.use_cuda_graph = use_cuda_graph

        # Default rank for ~200k params/layer
        if rank is None:
            rank = max(16, dim // 8)
        self.rank = rank

        # Just the cell, no projections
        self.cell = PureLowRankElmanCell(dim, rank=rank, use_fused=use_fused, use_cuda_graph=use_cuda_graph)

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
        return f'dim={self.dim}, rank={self.rank}, fused={self.use_fused}, cuda_graph={self.use_cuda_graph}, LEVEL=5'


PureLowRankElmanCell = PureLowRankElmanCell  # export


if __name__ == "__main__":
    print("Testing PureLowRankElman (E5)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test model
    dim = 512
    model = PureLowRankElman(dim=dim, rank=64).to(device).bfloat16()
    x = torch.randn(32, 512, dim, device=device, dtype=torch.bfloat16)

    print(f"Model: {model.extra_repr()}")
    print(f"CUDA kernel: {HASTY_AVAILABLE}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Compare to E1
    print(f"\nFor 50M model:")
    embed = 256 * dim
    target = 50_000_000
    depth = (target - embed) // params
    print(f"  Depth: {depth} layers")
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

    model = PureLowRankElman(dim=dim, rank=64).to(device).bfloat16()
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
