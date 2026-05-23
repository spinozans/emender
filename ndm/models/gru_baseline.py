"""
GRU baseline wrapper for comparison with Elman ladder models.

Uses PyTorch's nn.GRU (cuDNN backend) for maximum performance.
This provides a standard nonlinear RNN baseline for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GRULM(nn.Module):
    """
    Standard GRU Language Model with same interface as LadderLM.

    Uses cuDNN-optimized GRU for maximum performance.
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

        # GRU layers (cuDNN backend)
        self.grus = nn.ModuleList([
            nn.GRU(
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

            # GRU weights
            for name, param in self.grus[i].named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

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
        h = self.embedding(inp)

        # Initialize hidden states
        if prev_hiddens is None:
            hiddens = [torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype)
                      for _ in range(self.depth)]
        else:
            hiddens = [ph.unsqueeze(0) if ph is not None else
                      torch.zeros(1, B, self.dim_inner, device=device, dtype=dtype)
                      for ph in prev_hiddens]

        next_hiddens = []

        # GRU layers with pre-norm + residual
        for i, (ln, in_proj, gru, out_proj) in enumerate(zip(
            self.layer_norms, self.input_projs, self.grus, self.output_projs
        )):
            residual = h
            h = ln(h)
            h = in_proj(h)
            h, hn = gru(h, hiddens[i])
            h = out_proj(h)
            h = residual + h
            next_hiddens.append(hn.squeeze(0))

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
        return f'GRU baseline, dim={self.dim}, depth={self.depth}'


def create_gru_model(
    target_params: str = "100m",
    vocab_size: int = 50281,
):
    """Create a GRU model with approximately target_params parameters."""
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

    model = GRULM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        expansion_factor=expansion,
    )

    print(f"Created GRU model: dim={dim}, depth={depth}, params={model.get_num_params():,}")
    return model


if __name__ == "__main__":
    print("Testing GRULM...")
    model = GRULM(vocab_size=50281, dim=256, depth=4).cuda().bfloat16()
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
