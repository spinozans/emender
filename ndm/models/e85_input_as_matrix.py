"""
E85: Input-As-Matrix Layer

Key insight: dim = n_state^2 (e.g., n_state=32 -> dim=1024)
The input vector IS the transformation matrix - no projection matrices needed.
All ops are n_state x n_state matmuls -> fits in shared memory.

Architecture:
    # Input is directly the transformation matrix
    A = x.view(B, n_state, n_state)  # Input IS the matrix

    # State update via matmul (all in shared memory)
    M = M + scale * (A @ M)

    # Normalize for stability
    M = M / (M.norm() + eps)

    # Output = flattened state
    output = ln(M.view(B, dim))

Properties:
- No W_k, W_v, W_q projections needed - input IS the matrix
- All operations are n_state x n_state matmuls
- Perfect for shared memory (32x32 = 1024 = 4KB @ fp32)
- Learnable scale for stability control
- LayerNorm on flattened output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel (placeholder for future implementation)
E85_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e85_input_as_matrix_forward') and hasattr(hasty_pytorch_lib, 'e85_input_as_matrix_backward'):
        E85_CUDA_AVAILABLE = True
except ImportError:
    pass


class E85CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E85 CUDA kernel (placeholder for future implementation)."""

    @staticmethod
    def forward(ctx, x, M_init, scale, training):
        """
        Args:
            x: [T, B, dim] input (dim = n_state^2)
            M_init: [B, n_state, n_state] initial state matrix
            scale: scalar scale parameter
            training: bool

        Returns:
            output: [T, B, dim] output
            M_final: [B, n_state, n_state] final state matrix
        """
        # Placeholder - will be implemented when CUDA kernel is available
        raise NotImplementedError("CUDA kernel not yet implemented")

    @staticmethod
    def backward(ctx, d_output, d_M):
        raise NotImplementedError("CUDA kernel not yet implemented")


class E85InputAsMatrixCell(nn.Module):
    """
    E85 Input-As-Matrix cell.

    dim = n_state^2 (e.g., n_state=32 -> dim=1024)
    Input vector IS the transformation matrix - no projections needed.
    All ops are n_state x n_state matmuls -> fits in shared memory.

    Args:
        n_state: State matrix dimension (n_state x n_state matrix)
        init_scale: Initial scale value for stability (default 0.1)
        eps: Epsilon for numerical stability (default 1e-6)
    """

    def __init__(
        self,
        n_state: int = 32,
        init_scale: float = 0.1,
        eps: float = 1e-6,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.n_state = n_state
        self.dim = n_state * n_state
        self.eps = eps
        self.use_cuda = use_cuda and E85_CUDA_AVAILABLE

        # Learnable scale for stability
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # LayerNorm on flattened output
        self.ln = nn.LayerNorm(self.dim)

    def forward(
        self,
        x: torch.Tensor,
        M: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - simple reference implementation (loop over timesteps).

        Args:
            x: [T, B, dim] input (dim = n_state^2)
            M: [B, n_state, n_state] initial state matrix (optional)

        Returns:
            output: [T, B, dim] LayerNorm'd flattened state at each step
            M: [B, n_state, n_state] final state matrix
        """
        T, B, D = x.shape
        n = self.n_state
        device = x.device
        dtype = x.dtype

        assert D == self.dim, f"Input dim {D} != expected {self.dim} (n_state^2 = {n}^2)"

        # Initialize state if not provided
        if M is None:
            M = torch.zeros(B, n, n, device=device, dtype=dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda and E85_CUDA_AVAILABLE:
            return self._forward_cuda(x, M)

        # Python reference implementation
        return self._forward_python(x, M)

    def _forward_cuda(
        self,
        x: torch.Tensor,
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CUDA-accelerated forward pass (placeholder)."""
        # Will be implemented when CUDA kernel is available
        # For now, fall back to Python
        return self._forward_python(x, M)

    def _forward_python(
        self,
        x: torch.Tensor,
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure Python implementation - reference for CUDA kernel.

        Loop over timesteps. Each step:
        1. Reshape input to matrix A
        2. Update: M = M + scale*(A @ M)
        3. Normalize: M = M / ||M||
        4. Output: LayerNorm(flatten(M))
        """
        T, B, _ = x.shape
        n = self.n_state

        outputs = []

        for t in range(T):
            # Input IS the transformation matrix
            A = x[t].view(B, n, n)  # [B, n, n]

            # State update via matmul (all in shared memory for n=32)
            # M = M + scale*(A @ M)
            AM = torch.bmm(A, M)  # [B, n, n]
            M = M + self.scale * AM

            # Normalize for stability
            M_norm = M.norm(dim=(-2, -1), keepdim=True) + self.eps
            M = M / M_norm

            # Output = LayerNorm(flattened state)
            out = self.ln(M.view(B, self.dim))
            outputs.append(out)

        output = torch.stack(outputs, dim=0)  # [T, B, dim]
        return output, M

    def _forward_cuda_ready(
        self,
        x: torch.Tensor,
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CUDA-ready implementation that can be swapped in later.

        This version pre-computes what it can for better kernel fusion:
        - All inputs reshaped at once
        - Scale applied uniformly
        - Explicit memory layout for coalesced access
        """
        T, B, _ = x.shape
        n = self.n_state
        device = x.device
        dtype = x.dtype

        # Pre-reshape all inputs to matrices [T, B, n, n]
        A_all = x.view(T, B, n, n).contiguous()

        # Allocate output buffer
        outputs = torch.empty(T, B, self.dim, device=device, dtype=dtype)

        # Get scale value (for potential kernel fusion)
        scale = self.scale

        # Sequential update (inherently sequential due to state dependency)
        for t in range(T):
            A = A_all[t]
            # State update: M = M + scale*(A @ M)
            AM = torch.bmm(A, M)
            M = M + scale * AM

            # Normalize
            M_norm = M.norm(dim=(-2, -1), keepdim=True) + self.eps
            M = M / M_norm

            # Store flattened output (will apply LayerNorm after)
            outputs[t] = M.view(B, self.dim)

        # Apply LayerNorm to all outputs at once (can be fused in kernel)
        outputs = self.ln(outputs)

        return outputs, M


class E85InputAsMatrix(nn.Module):
    """
    E85: Input-As-Matrix Layer - Full layer wrapper.

    dim = n_state^2 (e.g., n_state=32 -> dim=1024)
    Input vector IS the transformation matrix - no projections needed.
    All ops are n_state x n_state matmuls -> fits in shared memory.

    Use this class for standalone E85 testing.

    Args:
        n_state: State matrix dimension (default 32, gives dim=1024)
        dropout: Dropout rate (default 0.0)
        use_cuda: Use CUDA kernel if available (default True)
    """

    def __init__(
        self,
        n_state: int = 32,
        dropout: float = 0.0,
        use_cuda: bool = True,
        **kwargs
    ):
        super().__init__()
        self.n_state = n_state
        self.dim = n_state * n_state
        self.use_cuda = use_cuda

        self.cell = E85InputAsMatrixCell(
            n_state=n_state,
            use_cuda=use_cuda
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        M: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, B, dim] input (dim = n_state^2)
            M: [B, n_state, n_state] initial state matrix

        Returns:
            output: [T, B, dim] output
            M: [B, n_state, n_state] final state matrix
        """
        output, M = self.cell(x, M)
        output = self.dropout(output)
        return output, M

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize hidden state as zero matrix."""
        return torch.zeros(batch_size, self.n_state, self.n_state, device=device, dtype=dtype)


class E85InputAsMatrixLayer(nn.Module):
    """
    E85: Input-As-Matrix Layer - Wrapper for use in ladder_lm.py.

    This wraps E85InputAsMatrixCell with the standard interface expected by LadderLM:
    - forward(x, hidden) where x is [B, T, dim] and hidden is [B, n_state, n_state]
    - Returns (output, new_hidden)

    Since dim = n_state^2, this layer is designed for use when the model dimension
    matches n_state^2 (e.g., dim=1024 for n_state=32).

    Args:
        dim: Model dimension (must equal n_state^2)
        n_state: State matrix dimension (default: sqrt(dim))
        expansion: Unused (for interface compatibility)
        dropout: Dropout rate (default 0.0)
        use_cuda: Use CUDA kernel if available (default True)
        use_conv: Use 1D convolution (default False)
        d_conv: Convolution kernel size (default 4)
        **kwargs: Additional arguments (ignored for compatibility)
    """

    def __init__(
        self,
        dim: int,
        n_state: Optional[int] = None,
        expansion: float = 1.0,
        dropout: float = 0.0,
        use_cuda: bool = True,
        use_conv: bool = False,
        d_conv: int = 4,
        **kwargs
    ):
        super().__init__()

        # Infer n_state from dim if not provided
        if n_state is None:
            n_state = int(dim ** 0.5)
            if n_state * n_state != dim:
                raise ValueError(f"dim={dim} is not a perfect square. E85 requires dim = n_state^2")

        self.dim = dim
        self.n_state = n_state
        self.use_conv = use_conv
        self.use_cuda = use_cuda

        assert n_state * n_state == dim, f"n_state^2 ({n_state}^2={n_state*n_state}) must equal dim ({dim})"

        # Optional convolution
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=dim,
                bias=True,
            )

        # Pre-activation (like other E-series models)
        # Since there's no projection, we apply silu directly
        self.pre_silu = True

        # The core cell
        self.cell = E85InputAsMatrixCell(
            n_state=n_state,
            use_cuda=use_cuda
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with LadderLM interface.

        Args:
            x: [B, T, dim] input sequence
            hidden: [B, n_state, n_state] initial state matrix (optional)

        Returns:
            output: [B, T, dim] output sequence
            hidden: [B, n_state, n_state] final state matrix
        """
        B, T, D = x.shape

        # Optional conv1d
        if self.use_conv:
            x_conv = x.transpose(1, 2)  # [B, dim, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal padding
            x = x_conv.transpose(1, 2)  # [B, T, dim]

        # Pre-activation
        if self.pre_silu:
            x = F.silu(x)

        # Transpose for RNN cell: [B, T, D] -> [T, B, D]
        x_rnn = x.transpose(0, 1).contiguous()

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                B, self.n_state, self.n_state,
                device=x.device, dtype=x.dtype
            )

        # Run cell
        cell_out, M_final = self.cell(x_rnn, hidden)

        # Transpose back: [T, B, D] -> [B, T, D]
        output = cell_out.transpose(0, 1).contiguous()

        # Dropout
        output = self.dropout(output)

        return output, M_final

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize hidden state as zero matrix [B, n_state, n_state]."""
        return torch.zeros(batch_size, self.n_state, self.n_state, device=device, dtype=dtype)

    def get_num_params(self):
        """Count parameters."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f'dim={self.dim}, n_state={self.n_state}, LEVEL=85_INPUT_AS_MATRIX'


if __name__ == "__main__":
    print("Testing E85 (Input-As-Matrix Layer)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E85_CUDA_AVAILABLE}")

    # Test dimensions: n_state=32 -> dim=1024
    n_state = 32
    dim = n_state * n_state
    B, T = 4, 32

    print(f"\n--- E85 Cell Test (n_state={n_state}, dim={dim}) ---")

    # Test the cell directly
    cell = E85InputAsMatrixCell(n_state=n_state).to(device).bfloat16()
    x = torch.randn(T, B, dim, device=device, dtype=torch.bfloat16)

    print(f"Input: {x.shape}")
    print(f"Cell params: {sum(p.numel() for p in cell.parameters()):,}")

    # Forward
    out, M = cell(x)
    print(f"Output: {out.shape}")
    print(f"State matrix: {M.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    print(f"\n--- E85 Layer Test (LadderLM interface) ---")

    # Test the layer wrapper
    layer = E85InputAsMatrixLayer(dim=dim, n_state=n_state).to(device).bfloat16()
    x_layer = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16)

    print(f"Input: {x_layer.shape}")
    print(f"Layer params: {layer.get_num_params():,}")

    # Forward with hidden state
    hidden = layer.init_hidden(B, device, torch.bfloat16)
    out_layer, hidden_new = layer(x_layer, hidden)
    print(f"Output: {out_layer.shape}")
    print(f"Hidden: {hidden_new.shape}")

    # Forward without hidden state (auto-initialize)
    out_layer2, _ = layer(x_layer)
    print(f"Output (no initial hidden): {out_layer2.shape}")

    # Backward
    loss = out_layer.sum()
    loss.backward()
    print("Backward passed!")

    # Test gradient flow
    print(f"\n--- Gradient flow check ---")
    layer.zero_grad()
    x_layer_grad = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    out_grad, _ = layer(x_layer_grad)
    loss = out_grad.sum()
    loss.backward()
    print(f"Input gradient norm: {x_layer_grad.grad.norm().item():.6e}")
    print(f"Input gradient max: {x_layer_grad.grad.abs().max().item():.6e}")
    print(f"Scale gradient: {layer.cell.scale.grad.item():.6e}")
    print(f"LN weight gradient norm: {layer.cell.ln.weight.grad.norm().item():.6e}")

    print(f"\n--- Memory usage ---")
    if device == 'cuda':
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    print("\n" + "=" * 60)
    print("E85: Input-As-Matrix Layer")
    print("dim = n_state^2 (no projections needed)")
    print("A = x.view(n_state, n_state)    # Input IS the matrix")
    print("M = (1-s)*M + s*A + s*(A @ M)   # State update (s=scale)")
    print("M = M / ||M||                 # Normalize")
    print("out = LayerNorm(flatten(M))   # Output")
    print("=" * 60)
