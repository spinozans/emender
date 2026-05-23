"""
LSTM baseline wrapper for comparison with Elman ladder models.

Uses PyTorch's nn.LSTM (cuDNN backend) for maximum performance.
LSTM has more stepwise nonlinearity than GRU (4 gates vs 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMLM(nn.Module):
    """
    Standard LSTM Language Model with same interface as LadderLM.

    Uses cuDNN-optimized LSTM for maximum performance.
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

        # LSTM layers (cuDNN backend)
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.dim_inner,
                hidden_size=self.dim_inner,
                num_layers=1,
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

            # LSTM weights
            for name, param in self.lstms[i].named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Initialize forget gate bias to 1 for better gradient flow
                    if 'bias_ih' in name or 'bias_hh' in name:
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

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
        return f'LSTM baseline, dim={self.dim}, depth={self.depth}'


def create_lstm_model(
    target_params: str = "100m",
    vocab_size: int = 50281,
):
    """Create an LSTM model with approximately target_params parameters."""
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    # Configs tuned to match parameter counts
    # LSTM has 4 gates: input, forget, cell, output
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

    model = LSTMLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        expansion_factor=expansion,
    )

    print(f"Created LSTM model: dim={dim}, depth={depth}, params={model.get_num_params():,}")
    return model


if __name__ == "__main__":
    print("Testing LSTMLM...")
    model = LSTMLM(vocab_size=50281, dim=256, depth=4).cuda().bfloat16()
    x = torch.randint(0, 50281, (2, 32), device='cuda')
    loss = model(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    print(f"Params: {model.get_num_params():,}")

    # Test with prev_hiddens
    loss, (hiddens, _) = model(x, return_loss=True, return_prev_hiddens=True)
    print(f"Got {len(hiddens)} hidden states (h, c tuples)")

    # Second forward with hiddens
    loss2, _ = model(x, return_loss=True, return_prev_hiddens=True, prev_hiddens=hiddens)
    print(f"Loss with prev_hiddens: {loss2.item():.4f}")
