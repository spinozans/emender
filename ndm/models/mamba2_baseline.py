"""
Mamba2 baseline wrapper for comparison.

This wraps the mamba_ssm package to provide the same interface as LadderLM,
allowing direct comparison between Elman levels and Mamba2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
    MAMBA2_AVAILABLE = True
except ImportError:
    MAMBA2_AVAILABLE = False
    Mamba2 = None


class Mamba2LM(nn.Module):
    """
    Mamba2 Language Model with same interface as LadderLM.

    Uses Mamba2 blocks instead of Elman layers for direct comparison.
    """

    def __init__(
        self,
        vocab_size=256,
        dim=512,
        depth=12,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        dropout=0.0,
        loss_chunk_size=0,
    ):
        super().__init__()
        self.loss_chunk_size = loss_chunk_size

        if not MAMBA2_AVAILABLE:
            raise ImportError("mamba_ssm not installed. Install with: pip install mamba-ssm")

        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Pre-normalization layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(depth)
        ])

        # Mamba2 layers
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
            for _ in range(depth)
        ])

        # Final norm and output
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie weights

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

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

        # Embed
        x = self.embedding(inp)

        # Mamba2 layers with pre-norm + residual
        for ln, layer in zip(self.layer_norms, self.layers):
            residual = x
            x = ln(x)
            x = layer(x)
            x = residual + x

        # Output
        x = self.norm(x)

        if return_loss:
            if actual_length is not None:
                device = x.device
                positions = torch.arange(target.size(1), device=device).unsqueeze(0)
                valid_mask = positions < (actual_length.unsqueeze(1) - 1)
                target = target.clone()
                target[~valid_mask] = -100

            T_total = x.size(1)
            loss_chunk = getattr(self, 'loss_chunk_size', 0)
            if loss_chunk > 0 and T_total > loss_chunk:
                total_sum = x.new_zeros(())
                total_count = 0
                for t0 in range(0, T_total, loss_chunk):
                    t1 = min(t0 + loss_chunk, T_total)
                    logits_c = self.lm_head(x[:, t0:t1])
                    target_c = target[:, t0:t1]
                    chunk_loss_sum = F.cross_entropy(
                        logits_c.reshape(-1, self.vocab_size),
                        target_c.reshape(-1),
                        ignore_index=-100,
                        reduction='sum',
                    )
                    total_sum = total_sum + chunk_loss_sum
                    total_count = total_count + (target_c != -100).sum()
                loss = total_sum / total_count.clamp(min=1)
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    target.reshape(-1),
                    ignore_index=-100,
                )
            if return_prev_hiddens:
                return loss, (None, None)
            return loss

        logits = self.lm_head(x)
        if return_prev_hiddens:
            return logits, (None, None)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f'Mamba2 baseline, dim={self.dim}, depth={self.depth}'


def create_mamba2_model(
    target_params: str = "100m",
    vocab_size: int = 256,
    expand: int = 2,
):
    """Create a Mamba2 model with approximately target_params parameters."""
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    # Search for dim/depth that hits target param count
    # Mamba2 layer params scale with expand factor
    d_state = 64
    headdim = 64

    # For expand=1, dim must be divisible by headdim
    # For expand=2, dim*2 must be divisible by headdim
    dim_step = headdim if expand == 1 else 32

    best_config = None
    best_diff = float('inf')

    for depth in range(6, 36, 2):
        for dim in range(256, 2048, dim_step):
            # Check Mamba2 constraint: d_ssm = dim * expand must be divisible by headdim
            if (dim * expand) % headdim != 0:
                continue

            # Estimate params (scales with expand factor)
            embed_params = vocab_size * dim * 2  # embedding + lm_head
            # Mamba2 layer params scale roughly as: expand * 3 * dim^2 per layer
            layer_params = depth * expand * 3.05 * dim * dim
            total = embed_params + layer_params

            diff = abs(total - target_count)
            if diff < best_diff:
                best_diff = diff
                best_config = (dim, depth)

    dim, depth = best_config

    # Use larger d_state for bigger models
    if target_count >= 500_000_000:
        d_state = 128

    model = Mamba2LM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        d_state=d_state,
        expand=expand,
        headdim=headdim,
    )

    actual_params = model.get_num_params()
    print(f"Created Mamba2 model: dim={dim}, depth={depth}, expand={expand}, params={actual_params:,}")
    return model


if __name__ == "__main__":
    if not MAMBA2_AVAILABLE:
        print("Mamba2 not available. Install with: pip install mamba-ssm")
    else:
        print("Testing Mamba2LM...")
        model = Mamba2LM(vocab_size=256, dim=256, depth=4).cuda().bfloat16()
        x = torch.randint(0, 256, (2, 32), device='cuda')
        loss = model(x, return_loss=True)
        print(f"Loss: {loss.item():.4f}")
        print(f"Params: {model.get_num_params():,}")
