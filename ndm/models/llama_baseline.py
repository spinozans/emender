"""
Llama-style Transformer baseline for comparison with Elman ladder models.

Standard architecture:
- RMSNorm (pre-norm)
- RoPE (Rotary Position Embedding)
- SwiGLU FFN
- Causal attention with Flash Attention support

This provides an attention baseline for fair comparison with recurrent models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, T, D]
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """Precompute rotary embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to queries and keys."""
    # xq, xk: [B, T, n_heads, head_dim]
    # freqs_cis: [T, head_dim//2] complex
    B, T, H, D = xq.shape

    # Reshape to complex
    xq_ = torch.view_as_complex(xq.float().reshape(B, T, H, D // 2, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(B, T, H, D // 2, 2))

    # Apply rotation
    freqs_cis = freqs_cis[:T].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D//2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaAttention(nn.Module):
    """Multi-head attention with RoPE and optional Flash Attention."""

    def __init__(self, dim, n_heads, head_dim=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else dim // n_heads

        # Q, K, V projections
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        for module in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x, freqs_cis, mask=None):
        """
        Args:
            x: [B, T, D] input
            freqs_cis: [max_seq_len, head_dim//2] rotary frequencies
            mask: Optional attention mask

        Returns:
            output: [B, T, D]
        """
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Attention
        if FLASH_ATTN_AVAILABLE and x.is_cuda and self.head_dim <= 256:
            # Flash Attention expects [B, T, H, D]
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )
        elif x.is_cuda:
            # Use PyTorch's built-in SDPA which dispatches to FlashAttention v2 / mem-efficient
            # backend automatically. O(T) memory like flash_attn package.
            # SDPA expects [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            output = output.transpose(1, 2).contiguous()  # [B, T, H, D]
        else:
            # CPU fallback: standard attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is None:
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout)
            output = torch.matmul(attn, v)
            output = output.transpose(1, 2).contiguous()

        # Project output
        output = output.view(B, T, -1)
        output = self.wo(output)

        return output


class LlamaFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Llama-style)."""

    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        # Llama uses 8/3 * dim for hidden, rounded to multiple of 256
        if hidden_dim is None:
            hidden_dim = int(8 / 3 * dim)
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in [self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        # SwiGLU: output = down(silu(gate(x)) * up(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class LlamaBlock(nn.Module):
    """Single Llama transformer block: attention + FFN with pre-norm."""

    def __init__(self, dim, n_heads, head_dim=None, hidden_dim=None, dropout=0.0):
        super().__init__()

        self.attention_norm = RMSNorm(dim)
        self.attention = LlamaAttention(dim, n_heads, head_dim, dropout)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = LlamaFFN(dim, hidden_dim, dropout)

    def forward(self, x, freqs_cis):
        # Attention with residual
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        # FFN with residual
        out = h + self.ffn(self.ffn_norm(h))
        return out


class LlamaLayer(nn.Module):
    """
    Wrapper for LlamaBlock that matches the interface expected by LadderLM.

    forward(x, h0=None) -> (output, h_final)

    For transformers, h_final is None since there's no recurrent state.
    """

    def __init__(self, dim, n_heads=None, head_dim=64, hidden_dim=None,
                 dropout=0.0, max_seq_len=65536, **kwargs):
        super().__init__()

        # Default n_heads based on dim and head_dim
        if n_heads is None:
            n_heads = max(1, dim // head_dim)

        self.block = LlamaBlock(dim, n_heads, head_dim, hidden_dim, dropout)

        # Precompute RoPE frequencies
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, max_seq_len),
            persistent=False
        )

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: Ignored (for interface compatibility)

        Returns:
            output: [B, T, dim] output sequence
            h_final: None (no recurrent state)
        """
        output = self.block(x, self.freqs_cis)
        return output, None


class LlamaLM(nn.Module):
    """
    Llama-style Transformer Language Model.

    Same interface as LadderLM for direct comparison.
    """

    def __init__(
        self,
        vocab_size=256,
        dim=512,
        depth=12,
        n_heads=None,
        head_dim=64,
        hidden_dim=None,
        dropout=0.0,
        max_seq_len=65536,
        tie_weights=True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth

        # Default n_heads based on dim and head_dim
        if n_heads is None:
            n_heads = max(1, dim // head_dim)
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Precompute RoPE frequencies (shared across layers)
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, max_seq_len),
            persistent=False
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            LlamaBlock(dim, n_heads, head_dim, hidden_dim, dropout)
            for _ in range(depth)
        ])

        # Final norm and output
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

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

        B, T = inp.shape
        device = inp.device

        # Embed
        h = self.embedding(inp)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, self.freqs_cis)

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
                return loss, (None, None)
            return loss

        if return_prev_hiddens:
            return logits, (None, None)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f'Llama Transformer, dim={self.dim}, depth={self.depth}, n_heads={self.n_heads}'


def count_llama_params(dim, depth, vocab_size=256, head_dim=64, hidden_dim=None):
    """Count Llama parameters analytically.

    Per layer:
    - Attention: 4 * dim * (n_heads * head_dim) for Q, K, V, O projections
    - FFN: 3 * dim * hidden_dim for gate, up, down projections
    - Norms: 2 * dim for attention_norm and ffn_norm

    Args:
        dim: Model dimension
        depth: Number of layers
        vocab_size: Vocabulary size
        head_dim: Head dimension (determines n_heads = dim // head_dim)
        hidden_dim: FFN hidden dimension (default: 8/3 * dim rounded to 256)

    Returns:
        Total parameter count
    """
    n_heads = max(1, dim // head_dim)
    attn_dim = n_heads * head_dim

    if hidden_dim is None:
        hidden_dim = int(8 / 3 * dim)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

    per_layer = (
        4 * dim * attn_dim +      # Q, K, V, O projections
        3 * dim * hidden_dim +    # FFN: gate, up, down
        2 * dim                   # RMSNorm weights (attention + ffn)
    )

    total = (
        vocab_size * dim +        # embedding (tied with lm_head)
        per_layer * depth +       # layers
        dim                       # final norm
    )

    return total


def find_llama_config(target_params, target_depth=20, vocab_size=256, head_dim=64):
    """Find dim to hit target_params at target_depth.

    Returns:
        (dim, depth, params) tuple
    """
    def align_to_128(x):
        return ((x + 63) // 128) * 128

    depth = target_depth

    # Binary search for dim
    low, high = 128, 2048
    while low < high:
        mid = align_to_128((low + high) // 2)
        params = count_llama_params(mid, depth, vocab_size, head_dim)
        if params < target_params:
            low = mid + 128
        else:
            high = mid

    dim = align_to_128(low)
    params = count_llama_params(dim, depth, vocab_size, head_dim)

    # Also try one step down
    if dim > 128:
        dim2 = dim - 128
        params2 = count_llama_params(dim2, depth, vocab_size, head_dim)
        if abs(params2 - target_params) < abs(params - target_params):
            dim, params = dim2, params2

    return (dim, depth, params)


def create_llama_model(
    target_params: str = "100m",
    vocab_size: int = 256,
    head_dim: int = 64,
    target_depth: int = 20,
):
    """Create a Llama model with approximately target_params parameters."""
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    dim, depth, expected_params = find_llama_config(
        target_count, target_depth, vocab_size, head_dim
    )

    model = LlamaLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        head_dim=head_dim,
    )

    actual_params = model.get_num_params()
    print(f"Created Llama model: dim={dim}, depth={depth}, n_heads={model.n_heads}, params={actual_params:,}")
    return model


if __name__ == "__main__":
    print("Testing LlamaLM...")
    print(f"Flash Attention available: {FLASH_ATTN_AVAILABLE}")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test small model
    model = LlamaLM(vocab_size=256, dim=256, depth=4, head_dim=64).to(device).bfloat16()
    x = torch.randint(0, 256, (2, 32), device=device)

    print("Testing forward...")
    loss = model(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    print(f"Params: {model.get_num_params():,}")

    print("\nTesting backward...")
    loss.backward()
    print("Backward passed!")

    # Test with return_prev_hiddens
    loss, (hiddens, _) = model(x, return_loss=True, return_prev_hiddens=True)
    print(f"Hidden states: {hiddens}")  # Should be None for transformer

    # Test LlamaLayer interface
    print("\nTesting LlamaLayer interface...")
    layer = LlamaLayer(dim=256, head_dim=64).to(device).bfloat16()
    x_seq = torch.randn(2, 32, 256, device=device, dtype=torch.bfloat16)
    out, h_final = layer(x_seq)
    print(f"Input: {x_seq.shape}, Output: {out.shape}, h_final: {h_final}")

    # Test param counting
    print("\n" + "=" * 60)
    print("Testing param counting...")
    for dim in [256, 512, 768]:
        analytical = count_llama_params(dim, depth=20, vocab_size=256)
        model = LlamaLM(vocab_size=256, dim=dim, depth=20)
        actual = model.get_num_params()
        print(f"dim={dim}: analytical={analytical:,}, actual={actual:,}, diff={actual-analytical}")

    # Test config finding for 100M
    print("\n" + "=" * 60)
    print("Finding 100M config...")
    dim, depth, params = find_llama_config(100_000_000, target_depth=20)
    print(f"100M target: dim={dim}, depth={depth}, params={params:,}")

    print("\nLlamaLM tests passed!")
