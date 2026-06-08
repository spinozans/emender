"""HybridLadderLM — LadderLM with per-layer architecture.

Accepts a `layer_pattern` (list of level names) instead of a single `level`.
Each entry is one layer's architecture. The pattern repeats to fill `depth`.

Examples:
  ['E88', 'fla-gdn']                  → alternating E88/FLA every layer
  ['E88', 'E88', 'fla-gdn', 'fla-gdn'] → 2 E88 then 2 FLA, repeat
  ['E88']                             → all E88 (equivalent to LadderLM(level='E88'))

Each layer can have different shape kwargs by passing `layer_kwargs`
(list-of-dicts, one per pattern entry).

Same residual + RMSNorm wrapping as LadderLM.
"""
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from .ladder_lm import RMSNorm, get_ladder_level, SwiGLUMLP, round_mlp_hidden


def _is_m2rnn_level(level: str) -> bool:
    return level in ('m2rnn', 'm2rnn-paper')


def _build_m2rnn_layer(level: str, dim: int, kwargs: Dict[str, Any]) -> nn.Module:
    from .m2rnn_baseline import M2RNNLayer

    paper_shape = level == 'm2rnn-paper'
    n_state = kwargs.get('n_state', 16)
    return M2RNNLayer(
        dim=dim,
        n_heads=kwargs.get('n_heads', 4),
        n_state=n_state,
        expansion=kwargs.get('expansion', 1.0),
        paper_shape=paper_shape,
        k_head_dim=kwargs.get('k_head_dim', 64 if paper_shape else None),
        v_head_dim=kwargs.get('v_head_dim', n_state if paper_shape else None),
        num_q_heads=kwargs.get('num_q_heads', 1 if paper_shape else None),
        num_k_heads=kwargs.get('num_k_heads', 1 if paper_shape else None),
        num_v_heads=kwargs.get('num_v_heads', None),
        num_f_heads=kwargs.get('num_f_heads', None),
        num_g_heads=kwargs.get('num_g_heads', None),
        num_weight_heads=kwargs.get('num_weight_heads', None),
        use_gate=kwargs.get('use_gate', True),
        use_residual=kwargs.get('use_residual', True),
        state_weight_trainable=kwargs.get('state_weight_trainable', True),
        use_conv=kwargs.get('use_conv', paper_shape),
        d_conv=kwargs.get('d_conv', 4),
        output_norm=kwargs.get('output_norm', paper_shape),
        normalize_qk=kwargs.get('normalize_qk', False),
        dropout=kwargs.get('dropout', 0.0),
        gradient_clipping=kwargs.get('gradient_clipping', 1.0 if paper_shape else None),
        linear_state=kwargs.get('linear_state', False),
    )


class HybridLadderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        depth: int = 12,
        layer_pattern: Optional[List[str]] = None,
        layer_kwargs: Optional[List[Dict[str, Any]]] = None,
        # Defaults shared across all layers (overridden by per-layer kwargs)
        n_state: int = 16,
        n_heads: int = 4,
        expansion: float = 1.0,
        rank: Optional[int] = None,
        use_gate: bool = True,
        gate_activation: str = 'silu',
        use_triton_e88: bool = False,
        dropout: float = 0.0,
        mlp_ratio: float = 0.0,
        **extra_kwargs,
    ):
        super().__init__()
        if layer_pattern is None:
            layer_pattern = ['E88']
        if layer_kwargs is None:
            layer_kwargs = [{}] * len(layer_pattern)
        assert len(layer_pattern) == len(layer_kwargs), \
            f"layer_pattern ({len(layer_pattern)}) and layer_kwargs ({len(layer_kwargs)}) must match"

        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.layer_pattern = layer_pattern
        self.use_triton_e88 = use_triton_e88
        self.disable_autocast = any(_is_m2rnn_level(level) for level in layer_pattern)

        self.embed = nn.Embedding(vocab_size, dim)

        self.layer_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])

        layers = []
        actual_pattern = []
        for i in range(depth):
            level = layer_pattern[i % len(layer_pattern)]
            kw = layer_kwargs[i % len(layer_kwargs)]
            actual_pattern.append(level)

            base_kwargs = {
                'dim': dim,
                'n_state': n_state,
                'n_heads': n_heads,
                'expansion': expansion,
                'use_gate': use_gate,
                'gate_activation': gate_activation,
                'dropout': dropout,
            }
            if isinstance(level, str) and (level.startswith('E88') or level == 'E97'):
                base_kwargs['use_triton'] = use_triton_e88
            if rank is not None:
                base_kwargs['rank'] = rank
            base_kwargs.update(kw)

            if _is_m2rnn_level(level):
                layers.append(_build_m2rnn_layer(level, dim, base_kwargs))
                continue

            LayerClass = get_ladder_level(level)
            try:
                layer = LayerClass(**base_kwargs)
            except TypeError as e:
                # Some layer classes don't accept all of these kwargs.
                # Try with a smaller set.
                minimal = {'dim': dim}
                for k in ('n_state', 'n_heads', 'expansion', 'rank', 'use_gate',
                          'gate_activation', 'dropout'):
                    if k in base_kwargs:
                        try:
                            test = LayerClass(**minimal, **{k: base_kwargs[k]})
                            minimal[k] = base_kwargs[k]
                        except TypeError:
                            pass
                layer = LayerClass(**minimal, **kw)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.actual_pattern = actual_pattern

        # Optional post-mixer SwiGLU MLP per block (transformer-style mixer+MLP).
        # When mlp_ratio>0 each block becomes:  h = h + mixer(norm1(h)); h = h + mlp(norm2(h))
        # This supplies the FIXED O(depth) nonlinear readout depth that the
        # capability-gap study (task capability-gap-research) requires present in
        # BOTH arms — the whole question is whether nonlinearity-IN-TIME still
        # separates once this fixed-depth MLP is available. mlp_ratio=0 (default)
        # preserves the original mixer-only behaviour for all prior experiments.
        self.mlp_ratio = float(mlp_ratio)
        if self.mlp_ratio > 0.0:
            self.mlp_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])
            self.mlps = nn.ModuleList([
                SwiGLUMLP(dim, round_mlp_hidden(dim, self.mlp_ratio), dropout=dropout)
                for _ in range(depth)
            ])
        else:
            self.mlp_norms = None
            self.mlps = None

        # Per-layer flag: True for E88/E97-family recurrent mixers (the only ones
        # with a fused bf16 Triton fwd/bwd kernel). Used by forward() to feed those
        # layers bf16 so the fused split-edit kernel actually engages — otherwise
        # RMSNorm emits fp32 under autocast, the layer's `x.dtype == bfloat16` gate
        # fails, and use_triton_e88 silently falls back to the eager T-scan.
        self._is_e88_layer = [
            bool(isinstance(level, str) and (level.startswith('E88') or level == 'E97'))
            for level in actual_pattern
        ]
        # When True, cast E88/E97 layer inputs to bf16 under autocast (required for
        # the fused kernel; production train.py/LadderLM gets bf16 for free because
        # the whole model is cast to bf16 via --bf16). Defaults on with the fused
        # path; the parity harness can flip it on the eager model for a bf16-vs-bf16
        # comparison. No effect under --disable_autocast (autocast disabled).
        self.cast_recurrent_bf16 = bool(use_triton_e88)

        self.out_norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        # Tie output to embedding (saves params)
        self.out_proj.weight = self.embed.weight

    def forward(self, x: torch.Tensor, return_loss: bool = False, targets: Optional[torch.Tensor] = None):
        h = self.embed(x)  # [B, T, dim]
        for i, (ln, layer) in enumerate(zip(self.layer_norms, self.layers)):
            normed = ln(h)
            # Feed E88/E97-family layers bf16 so the fused split-edit Triton kernel
            # engages (its forward gate requires x.dtype == bfloat16). RMSNorm emits
            # fp32 under autocast, so without this cast use_triton_e88 is inert.
            if (self.cast_recurrent_bf16 and self._is_e88_layer[i]
                    and normed.is_cuda and normed.dtype == torch.float32
                    and torch.is_autocast_enabled()):
                normed = normed.to(torch.bfloat16)
            out = layer(normed)
            if isinstance(out, tuple):
                out = out[0]
            h = h + out  # residual
            if self.mlps is not None:
                h = h + self.mlps[i](self.mlp_norms[i](h))  # post-mixer SwiGLU MLP
        h = self.out_norm(h)
        logits = self.out_proj(h)
        if return_loss:
            tgt = targets if targets is not None else x
            import torch.nn.functional as F
            return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                    tgt[:, 1:].reshape(-1))
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Self-test
# ============================================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hybrid: alternating E88 and FLA-GDN
    model = HybridLadderLM(
        vocab_size=4, dim=128, depth=4,
        layer_pattern=['E88', 'fla-gdn'],
        n_state=16, n_heads=4,
    ).to(device)
    print(f"layer pattern: {model.actual_pattern}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randint(0, 4, (2, 32), device=device)
    logits = model(x)
    print(f"logits: {tuple(logits.shape)}")

    # Backward
    loss = model(x, return_loss=True)
    loss.backward()
    print(f"loss: {loss.item():.4f} — backward OK")
