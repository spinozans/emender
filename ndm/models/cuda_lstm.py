"""
CudaLSTMLM - LSTM Language Model using custom CUDA kernel

Uses our BF16-optimized LSTM kernel instead of cuDNN to avoid the
4-16x performance regression cuDNN has for GRU/LSTM in bfloat16.

The CUDA kernel API:
    Forward returns: [h_all, c_all, f_cache, i_cache, o_cache, c_tilde_cache, tanh_c_cache]
    Backward returns: [dx, dW_fio, dW_c, dU_fio, dU_c, db_fio, db_c]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CUDA kernel
try:
    import hasty_pytorch_lib
    CUDA_LSTM_AVAILABLE = hasattr(hasty_pytorch_lib, 'lstm_forward')
except ImportError:
    CUDA_LSTM_AVAILABLE = False


class CudaLSTMFunction(torch.autograd.Function):
    """CUDA-accelerated LSTM autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, c0, W_fio, U_fio, W_c, U_c, b_fio, b_c):
        """
        Args:
            training: bool
            x: [T, B, dim] input sequence
            h0: [B, dim] initial hidden state
            c0: [B, dim] initial cell state
            W_fio: [3*dim, dim] input weights for f, i, o gates
            U_fio: [3*dim, dim] recurrent weights for f, i, o gates
            W_c: [dim, dim] input weights for candidate cell
            U_c: [dim, dim] recurrent weights for candidate cell
            b_fio: [3*dim] biases for f, i, o gates
            b_c: [dim] bias for candidate cell

        Returns:
            h_all: [T+1, B, dim] all hidden states (h_all[0] = h0)
            c_all: [T+1, B, dim] all cell states (c_all[0] = c0)
        """
        # Need caches for backward if any input requires grad
        # Note: torch.is_grad_enabled() is False inside autograd.Function.forward
        # so we only check requires_grad on inputs
        needs_grad = any(
            t.requires_grad for t in [x, W_fio, U_fio, W_c, U_c, b_fio, b_c]
        )

        result = hasty_pytorch_lib.lstm_forward(
            needs_grad,  # Use needs_grad instead of training to decide caching
            x.contiguous(),
            h0.contiguous(),
            c0.contiguous(),
            W_fio.contiguous(),
            U_fio.contiguous(),
            W_c.contiguous(),
            U_c.contiguous(),
            b_fio.contiguous(),
            b_c.contiguous()
        )
        # result = [h_all, c_all, f_cache, i_cache, o_cache, c_tilde_cache, tanh_c_cache]
        h_all, c_all, f_cache, i_cache, o_cache, c_tilde_cache, tanh_c_cache = result

        if needs_grad:
            ctx.save_for_backward(
                W_fio, W_c, U_fio, U_c,
                x, h_all, c_all,
                f_cache, i_cache, o_cache, c_tilde_cache, tanh_c_cache
            )

        # h_all/c_all are [T+1, B, dim] (include h0/c0), return [1:] for [T, B, dim]
        return h_all[1:], c_all[1:]

    @staticmethod
    def backward(ctx, dh_all, dc_all):
        """
        Args:
            dh_all: [T, B, dim] gradients on hidden states (T timesteps, excluding h0)
            dc_all: [T, B, dim] gradients on cell states (T timesteps, excluding c0)

        Returns:
            Gradients for: training, x, h0, c0, W_fio, U_fio, W_c, U_c, b_fio, b_c
        """
        (W_fio, W_c, U_fio, U_c,
         x, h_all, c_all,
         f_cache, i_cache, o_cache, c_tilde_cache, tanh_c_cache) = ctx.saved_tensors

        T = x.size(0)

        # Pass ALL hidden state gradients to CUDA backward, not just the final one
        # dh_all is [T, B, dim] - gradients from the output at every timestep
        dh_all_contiguous = dh_all.contiguous() if dh_all is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        # d_c_final is the gradient on the final cell state (optional, for sequence-to-one tasks)
        d_c_final = dc_all[-1].contiguous() if dc_all is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        result = hasty_pytorch_lib.lstm_backward(
            W_fio.contiguous(),
            W_c.contiguous(),
            U_fio.contiguous(),
            U_c.contiguous(),
            x.contiguous(),
            h_all.contiguous(),
            c_all.contiguous(),
            f_cache.contiguous(),
            i_cache.contiguous(),
            o_cache.contiguous(),
            c_tilde_cache.contiguous(),
            tanh_c_cache.contiguous(),
            dh_all_contiguous,
            d_c_final
        )
        # result = [dx, dW_fio, dW_c, dU_fio, dU_c, db_fio, db_c]
        dx, dW_fio, dW_c, dU_fio, dU_c, db_fio, db_c = result

        # Return gradients: None for training, then x, h0, c0, W_fio, U_fio, W_c, U_c, b_fio, b_c
        # h0/c0 gradients would require additional computation - return None for now
        return None, dx, None, None, dW_fio, dU_fio, dW_c, dU_c, db_fio, db_c


class CudaLSTMCell(nn.Module):
    """
    Single CUDA LSTM cell wrapping the kernel.

    Weights layout:
        W_fio: [3*dim, dim] - weights for forget, input, output gates
        U_fio: [3*dim, dim] - recurrent weights for f, i, o gates
        W_c: [dim, dim] - weights for candidate cell
        U_c: [dim, dim] - recurrent weights for candidate
        b_fio: [3*dim] - biases for f, i, o gates
        b_c: [dim] - bias for candidate
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gate weights (f, i, o stacked)
        self.W_fio = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.U_fio = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.b_fio = nn.Parameter(torch.zeros(3 * hidden_size))

        # Candidate cell weights
        self.W_c = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal for recurrent, normal for input."""
        # Input weights: normal distribution
        nn.init.normal_(self.W_fio, std=0.02)
        nn.init.normal_(self.W_c, std=0.02)

        # Recurrent weights: orthogonal initialization (better gradient flow)
        # Initialize each gate's recurrent weights separately
        for i in range(3):  # f, i, o gates
            nn.init.orthogonal_(self.U_fio.data[i * self.hidden_size:(i + 1) * self.hidden_size])
        nn.init.orthogonal_(self.U_c)

        # Biases: zero, except forget gate bias = 1 for better gradient flow
        nn.init.zeros_(self.b_fio)
        self.b_fio.data[:self.hidden_size].fill_(1.0)  # Forget gate bias = 1
        nn.init.zeros_(self.b_c)

    def forward(self, x, h0=None, c0=None):
        """
        Args:
            x: [T, B, input_size] input sequence
            h0: [B, hidden_size] initial hidden state
            c0: [B, hidden_size] initial cell state

        Returns:
            h_out: [T, B, hidden_size] output hidden states (excluding h0)
            h_final: [B, hidden_size] final hidden state
            c_final: [B, hidden_size] final cell state
        """
        T, B, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        if c0 is None:
            c0 = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)

        if CUDA_LSTM_AVAILABLE and x.is_cuda:
            h_out, c_out = CudaLSTMFunction.apply(
                self.training,
                x, h0, c0,
                self.W_fio, self.U_fio,
                self.W_c, self.U_c,
                self.b_fio, self.b_c
            )
            # CudaLSTMFunction now returns [T, B, dim] (already sliced)
            h_final = h_out[-1]  # [B, dim]
            c_final = c_out[-1]  # [B, dim]
            return h_out, h_final, c_final

        # PyTorch fallback for CPU/debugging
        h_list = [h0]
        c_list = [c0]

        for t in range(T):
            h_prev = h_list[-1]
            c_prev = c_list[-1]
            x_t = x[t]

            # Gates: f, i, o
            gates_pre = x_t @ self.W_fio.T + h_prev @ self.U_fio.T + self.b_fio
            f = torch.sigmoid(gates_pre[:, :self.hidden_size])
            i = torch.sigmoid(gates_pre[:, self.hidden_size:2 * self.hidden_size])
            o = torch.sigmoid(gates_pre[:, 2 * self.hidden_size:])

            # Candidate cell
            c_tilde = torch.tanh(x_t @ self.W_c.T + h_prev @ self.U_c.T + self.b_c)

            # Cell state
            c_new = f * c_prev + i * c_tilde

            # Hidden state
            h_new = o * torch.tanh(c_new)

            h_list.append(h_new)
            c_list.append(c_new)

        h_out = torch.stack(h_list[1:], dim=0)  # [T, B, dim]
        h_final = h_list[-1]
        c_final = c_list[-1]
        return h_out, h_final, c_final


class CudaLSTM(nn.Module):
    """
    LSTM layer using our custom CUDA kernel.

    Similar interface to nn.LSTM but uses CudaLSTMCell internally.
    """

    def __init__(self, input_size, hidden_size, batch_first=True, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.cell = CudaLSTMCell(input_size, hidden_size)

    def forward(self, x, hx=None):
        """
        Args:
            x: Input tensor
               If batch_first: [B, T, input_size]
               Else: [T, B, input_size]
            hx: Tuple of (h0, c0) where each is [1, B, hidden_size] or [B, hidden_size]

        Returns:
            output: [B, T, hidden_size] if batch_first else [T, B, hidden_size]
            (hn, cn): Final states, each [1, B, hidden_size]
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # [B, T, D] -> [T, B, D]

        B = x.size(1)

        h0, c0 = None, None
        if hx is not None:
            h0, c0 = hx
            # Handle [1, B, D] format from nn.LSTM
            if h0.dim() == 3:
                h0 = h0.squeeze(0)
            if c0.dim() == 3:
                c0 = c0.squeeze(0)

        h_out, h_final, c_final = self.cell(x, h0, c0)

        if self.batch_first:
            h_out = h_out.transpose(0, 1)  # [T, B, D] -> [B, T, D]

        # Return in nn.LSTM format: [1, B, D]
        return h_out, (h_final.unsqueeze(0), c_final.unsqueeze(0))


class CudaLSTMLM(nn.Module):
    """
    LSTM Language Model using our custom CUDA LSTM kernel.

    Same interface as LSTMLM but uses CudaLSTM instead of nn.LSTM.
    This avoids cuDNN's bfloat16 performance regression.
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

        # CUDA LSTM layers
        self.lstms = nn.ModuleList([
            CudaLSTM(
                input_size=self.dim_inner,
                hidden_size=self.dim_inner,
                batch_first=True,
                bias=True
            )
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

            # LSTM cell weights are initialized in CudaLSTMCell._init_weights()

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
        """Forward pass with LadderLM-compatible interface.

        prev_hiddens: list of (h, c) tuples for each layer
        """
        if return_loss:
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        B, T = inp.shape
        device = inp.device
        dtype = next(self.parameters()).dtype

        # Embed
        h = self.embedding(inp)

        # Initialize hidden states
        if prev_hiddens is None:
            hiddens = [
                (torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype),
                 torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype))
                for _ in range(self.depth)
            ]
        else:
            hiddens = []
            for ph in prev_hiddens:
                if ph is not None and isinstance(ph, tuple):
                    h_prev, c_prev = ph
                    hiddens.append((
                        h_prev.unsqueeze(0) if h_prev.dim() == 2 else h_prev,
                        c_prev.unsqueeze(0) if c_prev.dim() == 2 else c_prev
                    ))
                else:
                    hiddens.append((
                        torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype),
                        torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype)
                    ))

        next_hiddens = []

        # LSTM layers with pre-norm + residual
        for i, (ln, in_proj, lstm, out_proj) in enumerate(zip(
            self.layer_norms, self.input_projs, self.lstms, self.output_projs
        )):
            residual = h
            h = ln(h)
            h = in_proj(h)
            h, (hn, cn) = lstm(h, hiddens[i])
            h = out_proj(h)
            h = residual + h
            next_hiddens.append((hn.squeeze(0), cn.squeeze(0)))

        # Output
        h = self.norm(h)
        logits = self.lm_head(h)

        if return_loss:
            # Mask out padded positions if actual_length is provided
            if actual_length is not None:
                # Create mask: valid positions are 0 to actual_length-2 (shifted by 1 for targets)
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
        return f'CudaLSTM LM, dim={self.dim}, depth={self.depth}, dim_inner={self.dim_inner}'


def create_cuda_lstm_model(
    target_params: str = "100m",
    vocab_size: int = 50281,
):
    """Create a CudaLSTM model with approximately target_params parameters."""
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    # Configs tuned to match parameter counts
    # LSTM has 4 gates: forget, input, output, candidate
    # Per layer: 4 * (dim_inner * dim_inner + dim_inner * dim_inner) + biases
    configs = {
        20_000_000: (256, 10, 1.0),    # dim, depth, expansion
        50_000_000: (384, 14, 1.0),
        100_000_000: (512, 18, 1.0),
        200_000_000: (768, 18, 1.0),
        350_000_000: (1024, 20, 1.0),
        500_000_000: (1280, 22, 1.0),
        700_000_000: (1536, 24, 1.0),
        1_000_000_000: (1920, 26, 1.0),
    }

    closest = min(configs.keys(), key=lambda x: abs(x - target_count))
    dim, depth, expansion = configs[closest]

    model = CudaLSTMLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        expansion_factor=expansion,
    )

    print(f"Created CudaLSTM model: dim={dim}, depth={depth}, params={model.get_num_params():,}")
    return model


if __name__ == "__main__":
    print("Testing CudaLSTMLM...")
    print(f"CUDA LSTM kernel available: {CUDA_LSTM_AVAILABLE}")

    model = CudaLSTMLM(vocab_size=256, dim=256, depth=4).cuda().bfloat16()
    x = torch.randint(0, 256, (2, 32), device='cuda')

    print("\nTesting forward...")
    loss = model(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    print(f"Params: {model.get_num_params():,}")

    print("\nTesting backward...")
    loss.backward()
    print("Backward passed!")

    # Test with prev_hiddens
    print("\nTesting with prev_hiddens...")
    loss, (hiddens, _) = model(x, return_loss=True, return_prev_hiddens=True)
    print(f"Got {len(hiddens)} hidden states (h, c tuples)")

    # Second forward with hiddens
    loss2, _ = model(x, return_loss=True, return_prev_hiddens=True, prev_hiddens=hiddens)
    print(f"Loss with prev_hiddens: {loss2.item():.4f}")

    print("\nCudaLSTMLM test passed!")
