"""
CUDA GRU Language Model using custom CUDA kernel instead of cuDNN.

Uses the same model structure as gru_baseline.py (embedding, layer norms,
input/output projections, residual connections, LM head) but with our
custom CUDA GRU kernel for the recurrence.

CUDA Kernel API:
    Forward: returns [h_all, z_cache, h_tilde_cache, r_h_cache]
    Backward: returns [dx, dW_zr, dW_h, dU_zr, dU_h, db_zr, db_h]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    GRU_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'gru_forward')
except ImportError:
    GRU_CUDA_AVAILABLE = False


class CudaGRUFunction(torch.autograd.Function):
    """CUDA-accelerated GRU autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W_zr, U_zr, W_h, U_h, b_zr, b_h):
        """
        Forward pass using custom CUDA kernel.

        Args:
            training: bool - whether in training mode
            x: [T, B, dim] - input sequence
            h0: [B, dim] - initial hidden state
            W_zr: [2*dim, dim] - input weights for z,r gates
            U_zr: [2*dim, dim] - recurrent weights for z,r gates
            W_h: [dim, dim] - input weights for candidate
            U_h: [dim, dim] - recurrent weights for candidate
            b_zr: [2*dim] - biases for z,r gates
            b_h: [dim] - bias for candidate

        Returns:
            h: [T, B, dim] - hidden states at each timestep
        """
        h, z_cache, h_tilde_cache, r_h_cache = hasty_pytorch_lib.gru_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            W_zr.contiguous(),
            U_zr.contiguous(),
            W_h.contiguous(),
            U_h.contiguous(),
            b_zr.contiguous(),
            b_h.contiguous()
        )
        ctx.save_for_backward(W_zr, W_h, U_zr, U_h, b_zr, b_h, x, h, z_cache, h_tilde_cache, r_h_cache)
        # h is [T+1, B, dim] (includes h0), return h[1:] for [T, B, dim]
        return h[1:]

    @staticmethod
    def backward(ctx, d_h):
        """
        Backward pass using custom CUDA kernel.

        Args:
            d_h: [T, B, dim] - gradient of loss w.r.t. hidden states

        Returns:
            Gradients for all inputs (None for training, dx, None for h0, dW_zr, dU_zr, dW_h, dU_h, db_zr, db_h)
        """
        W_zr, W_h, U_zr, U_h, b_zr, b_h, x, h, z_cache, h_tilde_cache, r_h_cache = ctx.saved_tensors

        # CUDA backward returns: dx, dW_zr, dW_h, dU_zr, dU_h, db_zr, db_h
        dx, dW_zr, dW_h, dU_zr, dU_h, db_zr, db_h = hasty_pytorch_lib.gru_backward(
            W_zr.contiguous(),
            W_h.contiguous(),
            U_zr.contiguous(),
            U_h.contiguous(),
            b_zr.contiguous(),
            b_h.contiguous(),
            x.contiguous(),
            h.contiguous(),
            z_cache.contiguous(),
            h_tilde_cache.contiguous(),
            r_h_cache.contiguous(),
            d_h.contiguous()
        )

        # Return gradients matching forward signature:
        # forward(ctx, training, x, h0, W_zr, U_zr, W_h, U_h, b_zr, b_h)
        # CUDA returns: dx, dW_zr, dW_h, dU_zr, dU_h, db_zr, db_h
        # Forward order: training, x, h0, W_zr, U_zr, W_h, U_h, b_zr, b_h
        # Backward return: None, dx, None, dW_zr, dU_zr, dW_h, dU_h, db_zr, db_h
        # Note: CUDA has (dW_zr, dW_h, dU_zr, dU_h) but forward has (W_zr, U_zr, W_h, U_h)
        # So swap dW_h<->dU_zr to match forward order
        return None, dx, None, dW_zr, dU_zr, dW_h, dU_h, db_zr, db_h


def gru_forward_python(x, h0, W_zr, U_zr, W_h, U_h, b_zr, b_h):
    """
    Python reference implementation for GRU forward pass.

    GRU equations:
        z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1} + b_z)  # update gate
        r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1} + b_r)  # reset gate
        h_tilde_t = tanh(W_h @ x_t + U_h @ (r_t * h_{t-1}) + b_h)  # candidate
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t  # output

    Args:
        x: [T, B, dim] - input sequence
        h0: [B, dim] - initial hidden state
        W_zr: [2*dim, dim] - input weights for z,r gates
        U_zr: [2*dim, dim] - recurrent weights for z,r gates
        W_h: [dim, dim] - input weights for candidate
        U_h: [dim, dim] - recurrent weights for candidate
        b_zr: [2*dim] - biases for z,r gates
        b_h: [dim] - bias for candidate

    Returns:
        h: [T, B, dim] - hidden states at each timestep
    """
    T, B, dim = x.shape

    # Split weights and biases for z and r gates
    W_z, W_r = W_zr[:dim], W_zr[dim:]
    U_z, U_r = U_zr[:dim], U_zr[dim:]
    b_z, b_r = b_zr[:dim], b_zr[dim:]

    h_prev = h0
    h_list = []

    for t in range(T):
        x_t = x[t]  # [B, dim]

        # Gates: z (update), r (reset)
        z_t = torch.sigmoid(x_t @ W_z.T + h_prev @ U_z.T + b_z)
        r_t = torch.sigmoid(x_t @ W_r.T + h_prev @ U_r.T + b_r)

        # Candidate hidden state
        h_tilde_t = torch.tanh(x_t @ W_h.T + (r_t * h_prev) @ U_h.T + b_h)

        # New hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde_t

        h_list.append(h_t)
        h_prev = h_t

    return torch.stack(h_list, dim=0)  # [T, B, dim]


class CudaGRUCell(nn.Module):
    """
    GRU cell using custom CUDA kernel.

    Wraps the CUDA kernel with proper weight initialization and
    torch.autograd.Function for gradient computation.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Gate weights (z=update, r=reset)
        self.W_zr = nn.Parameter(torch.empty(2 * dim, dim))  # input weights
        self.U_zr = nn.Parameter(torch.empty(2 * dim, dim))  # recurrent weights
        self.b_zr = nn.Parameter(torch.zeros(2 * dim))       # biases

        # Candidate weights
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.U_h = nn.Parameter(torch.empty(dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to PyTorch GRU."""
        # Input weights: normal initialization
        nn.init.normal_(self.W_zr, std=0.02)
        nn.init.normal_(self.W_h, std=0.02)

        # Recurrent weights: orthogonal initialization for stable gradients
        # Initialize in fp32 then copy for bf16 compatibility
        U_zr_fp32 = torch.empty(2 * self.dim, self.dim, dtype=torch.float32)
        nn.init.orthogonal_(U_zr_fp32[:self.dim])  # U_z
        nn.init.orthogonal_(U_zr_fp32[self.dim:])  # U_r
        with torch.no_grad():
            self.U_zr.copy_(U_zr_fp32.to(self.U_zr.dtype))

        U_h_fp32 = torch.empty(self.dim, self.dim, dtype=torch.float32)
        nn.init.orthogonal_(U_h_fp32)
        with torch.no_grad():
            self.U_h.copy_(U_h_fp32.to(self.U_h.dtype))

    def forward(self, x, h0=None):
        """
        Forward pass through GRU cell.

        Args:
            x: [T, B, dim] - input sequence
            h0: [B, dim] - initial hidden state (default: zeros)

        Returns:
            h: [T, B, dim] - hidden states at each timestep
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if GRU_CUDA_AVAILABLE and x.is_cuda:
            return CudaGRUFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                self.W_zr.contiguous(),
                self.U_zr.contiguous(),
                self.W_h.contiguous(),
                self.U_h.contiguous(),
                self.b_zr.contiguous(),
                self.b_h.contiguous()
            )

        # Python fallback
        return gru_forward_python(
            x, h0, self.W_zr, self.U_zr, self.W_h, self.U_h, self.b_zr, self.b_h
        )


class CudaGRU(nn.Module):
    """
    GRU layer using custom CUDA kernel.

    Similar interface to nn.GRU but uses our custom CUDA implementation.
    """

    def __init__(self, input_size, hidden_size, batch_first=True, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # The cell handles all the weights
        self.cell = CudaGRUCell(hidden_size)

        # If input_size != hidden_size, we need an input projection
        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
        else:
            self.input_proj = None

    def forward(self, x, h0=None):
        """
        Forward pass through GRU layer.

        Args:
            x: [B, T, input_size] if batch_first else [T, B, input_size]
            h0: [1, B, hidden_size] - initial hidden state (nn.GRU format)

        Returns:
            output: [B, T, hidden_size] if batch_first else [T, B, hidden_size]
            h_n: [1, B, hidden_size] - final hidden state
        """
        if self.batch_first:
            x = x.permute(1, 0, 2).contiguous()  # [B, T, D] -> [T, B, D]

        T, B, _ = x.shape

        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Handle h0 format (nn.GRU uses [1, B, D])
        if h0 is not None:
            h0 = h0.squeeze(0)  # [1, B, D] -> [B, D]

        # Run GRU cell
        h = self.cell(x, h0)  # [T, B, hidden_size]

        # Get final hidden state: h[-1] is [B, hidden_size], need [1, B, hidden_size]
        h_n = h[-1].unsqueeze(0)  # [1, B, hidden_size]

        if self.batch_first:
            h = h.permute(1, 0, 2).contiguous()  # [T, B, D] -> [B, T, D]

        return h, h_n


class CudaGRULM(nn.Module):
    """
    GRU Language Model using custom CUDA kernel.

    Same structure as GRULM (gru_baseline.py) but uses CudaGRU layers
    instead of nn.GRU (cuDNN).

    Architecture:
        - Token embedding
        - Stack of [LayerNorm -> input_proj -> CudaGRU -> output_proj + residual]
        - Final LayerNorm -> LM head
    """

    def __init__(
        self,
        vocab_size=256,
        dim=512,
        depth=12,
        expansion_factor=1.0,
        dropout=0.0,
        tie_weights=True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.dim_inner = int(dim * expansion_factor)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Pre-normalization layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(depth)
        ])

        # Input projections (dim -> dim_inner)
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, self.dim_inner, bias=False) for _ in range(depth)
        ])

        # GRU layers using custom CUDA kernel
        self.grus = nn.ModuleList([
            CudaGRUCell(self.dim_inner)
            for _ in range(depth)
        ])

        # Output projections (dim_inner -> dim)
        self.output_projs = nn.ModuleList([
            nn.Linear(self.dim_inner, dim, bias=False) for _ in range(depth)
        ])

        # Final norm and output
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

        for i in range(self.depth):
            # Input/output projections
            nn.init.normal_(self.input_projs[i].weight, std=0.02)
            nn.init.zeros_(self.output_projs[i].weight)

    def forward(
        self,
        x,
        return_loss=False,
        return_prev_hiddens=False,
        prev_hiddens=None,
        prev_conv_buffers=None,
        actual_length=None,
        doc_boundaries=None,
    ):
        """Forward pass with LadderLM-compatible interface."""
        if return_loss:
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        B, T = inp.shape
        device = inp.device
        dtype = next(self.parameters()).dtype

        # Embed
        h = self.embedding(inp)  # [B, T, dim]

        # Initialize hidden states
        if prev_hiddens is None:
            hiddens = [torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
                       for _ in range(self.depth)]
        else:
            hiddens = [ph if ph is not None else
                       torch.zeros(B, self.dim_inner, device=device, dtype=dtype)
                       for ph in prev_hiddens]

        next_hiddens = []

        # GRU layers with pre-norm + residual
        for i, (ln, in_proj, gru, out_proj) in enumerate(zip(
            self.layer_norms, self.input_projs, self.grus, self.output_projs
        )):
            residual = h
            h = ln(h)
            h = in_proj(h)  # [B, T, dim_inner]

            # Transpose for GRU cell: [B, T, D] -> [T, B, D]
            h_gru = h.permute(1, 0, 2).contiguous()

            # Run GRU cell
            h_gru = gru(h_gru, hiddens[i])  # [T, B, dim_inner]

            # Store final hidden state for TBPTT (from GRU internal state, not projected output)
            next_hiddens.append(h_gru[-1].contiguous())

            # Transpose back: [T, B, D] -> [B, T, D]
            h = h_gru.permute(1, 0, 2).contiguous()

            h = out_proj(h)
            h = residual + h

        # Output
        h = self.norm(h)
        logits = self.lm_head(h)

        if return_loss:
            # Mask out padded positions if actual_length is provided
            if actual_length is not None:
                positions = torch.arange(target.size(1), device=device).unsqueeze(0)
                valid_mask = positions < (actual_length.unsqueeze(1) - 1)
                target = target.clone()
                target[~valid_mask] = -100

            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target.reshape(-1),
                ignore_index=-100,
            )
            if return_prev_hiddens:
                return loss, (next_hiddens, None)
            return loss

        if return_prev_hiddens:
            return logits, (next_hiddens, None)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f'CUDA GRU baseline, dim={self.dim}, depth={self.depth}'


def create_cuda_gru_model(
    target_params: str = "100m",
    vocab_size: int = 50281,
):
    """Create a CUDA GRU model with approximately target_params parameters."""
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    # Configs tuned to match parameter counts
    # GRU has 3 gates: reset, update, candidate
    # Per layer: 3 * (dim_inner * dim_inner + dim_inner * dim_inner) + biases
    configs = {
        20_000_000: (256, 12, 1.0),    # dim, depth, expansion
        50_000_000: (384, 16, 1.0),
        100_000_000: (512, 20, 1.0),
        200_000_000: (768, 20, 1.0),
        350_000_000: (1024, 22, 1.0),
        500_000_000: (1280, 24, 1.0),
        700_000_000: (1536, 26, 1.0),
        1_000_000_000: (1920, 28, 1.0),
    }

    closest = min(configs.keys(), key=lambda x: abs(x - target_count))
    dim, depth, expansion = configs[closest]

    model = CudaGRULM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        expansion_factor=expansion,
    )

    print(f"Created CUDA GRU model: dim={dim}, depth={depth}, params={model.get_num_params():,}")
    return model


if __name__ == "__main__":
    print("Testing CudaGRULM...")
    print(f"CUDA kernel available: {GRU_CUDA_AVAILABLE}")

    model = CudaGRULM(vocab_size=50281, dim=256, depth=4).cuda().bfloat16()
    x = torch.randint(0, 50281, (2, 32), device='cuda')
    loss = model(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    print(f"Params: {model.get_num_params():,}")

    # Test with prev_hiddens
    loss, (hiddens, _) = model(x, return_loss=True, return_prev_hiddens=True)
    print(f"Got {len(hiddens)} hidden states")

    # Second forward with hiddens
    loss2, _ = model(x, return_loss=True, return_prev_hiddens=True, prev_hiddens=hiddens)
    print(f"Loss with prev_hiddens: {loss2.item():.4f}")

    # Test backward
    loss.backward()
    print("Backward pass completed!")
