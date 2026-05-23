"""
FLA (Flash Linear Attention) GatedDeltaNet wrapper for LadderLM benchmarking.

This wraps the fla library's GatedDeltaNet for use with the elman framework,
providing a fair comparison between our E-series models and this ICLR 2025 baseline.

GatedDeltaNet combines:
- Delta rule for selective memory updates (DeltaNet)
- Gating mechanism for decay control (Mamba2-style)
- Linear attention with O(n) complexity

Reference:
    Gated Delta Networks: Improving Mamba2 with Delta Rule (ICLR 2025)
    https://github.com/NVlabs/GatedDeltaNet
    https://github.com/fla-org/flash-linear-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import FLA's GatedDeltaNet
try:
    from fla.layers import GatedDeltaNet as FLAGatedDeltaNet
    FLA_GDN_AVAILABLE = True
except ImportError:
    FLA_GDN_AVAILABLE = False
    FLAGatedDeltaNet = None


class FLAGatedDeltaNetLayer(nn.Module):
    """
    Wrapper for FLA's GatedDeltaNet that matches the elman model interface.

    This provides a fair apples-to-apples comparison with E-series models
    by using FLA's highly optimized Triton kernels.

    The interface matches other E-series layers:
    - forward(x, h0=None) -> (output, h_final)
    - x has shape [B, T, D]
    - output has shape [B, T, D]

    FLA's GatedDeltaNet uses chunked linear attention with:
    - Per-head Q, K, V projections
    - Short convolutions on Q, K, V (like Mamba2)
    - Gated output projection
    - Fused RMSNorm on output
    """

    def __init__(
        self,
        dim,
        expansion=2.0,  # expand_v in FLA terms
        dropout=0.0,
        head_dim=128,  # FLA default is 256, but 128 is more comparable
        num_heads=None,
        use_conv=None,  # If None, default to True for FLA (crucial for performance)
        d_conv=4,  # conv_size in FLA
        mamba2_init=False,
        **kwargs  # Absorb unused args from LadderLM
    ):
        # FLA GatedDeltaNet REQUIRES short convolutions for good performance
        # Override use_conv to True unless explicitly set to False
        if use_conv is None or use_conv is False:
            use_conv = True  # FLA warns strongly against disabling this
        super().__init__()

        if not FLA_GDN_AVAILABLE:
            raise ImportError(
                "FLA GatedDeltaNet not available. Install with: "
                "pip install flash-linear-attention"
            )

        self.dim = dim
        self.expansion = expansion
        self.head_dim = head_dim

        # Compute num_heads if not specified
        if num_heads is None:
            num_heads = max(1, dim // head_dim)

        # Ensure dim is divisible by num_heads
        if dim % num_heads != 0:
            # Adjust num_heads to be a divisor
            for nh in range(num_heads, 0, -1):
                if dim % nh == 0:
                    num_heads = nh
                    break

        self.num_heads = num_heads
        self.actual_head_dim = dim // num_heads

        # Create FLA's GatedDeltaNet
        # Note: FLA's GatedDeltaNet already has in/out projections built in
        # layer_idx is required for FLA's Cache to populate/retrieve per-layer state
        # for stateful autoregressive generation. LadderLM sets it post-construction
        # via set_layer_idx() so every layer has a unique index.
        self.gdn = FLAGatedDeltaNet(
            hidden_size=dim,
            expand_v=expansion,
            head_dim=self.actual_head_dim,
            num_heads=num_heads,
            use_gate=True,  # Always use gating
            use_short_conv=use_conv,
            conv_size=d_conv,
            mode='chunk',  # Chunked linear attention (most efficient)
            layer_idx=0,  # Placeholder; LadderLM overwrites via set_layer_idx()
        )

        # Dropout after GDN output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def set_layer_idx(self, idx):
        """Set the layer index used by FLA's Cache for stateful inference."""
        self.gdn.layer_idx = idx

    def forward(self, x, h0=None, **kwargs):
        """
        Forward pass matching elman interface.

        Args:
            x: [B, T, D] input tensor
            h0: Initial state. May be None, or a per-layer FLA `Cache` object
                returned from a prior call. Each LadderLM layer carries its
                own single-entry Cache; FLA's internal `layer_idx` is set to
                0 within a per-layer cache (overriding LadderLM's per-layer
                index, which is only used if we ever shared a Cache).

        Returns:
            output: [B, T, D] output tensor
            h_final: Per-layer FLA Cache (None during training).
        """
        # Training: no state, chunked mode, sequences independent.
        if self.training:
            output, _, _ = self.gdn(x, use_cache=False)
            output = self.dropout(output)
            return output, None

        # Inference: allocate an empty Cache for the first call so FLA will
        # populate it (its forward only calls cache.update() if non-None).
        # For subsequent calls, reuse the Cache passed in as h0.
        if h0 is None:
            from fla.models.utils import Cache
            cache = Cache()
            # Within this per-layer cache we index at position 0
            saved_idx = self.gdn.layer_idx
            self.gdn.layer_idx = 0
            output, _, new_cache = self.gdn(x, past_key_values=cache, use_cache=True)
            self.gdn.layer_idx = saved_idx
        else:
            saved_idx = self.gdn.layer_idx
            self.gdn.layer_idx = 0
            output, _, new_cache = self.gdn(x, past_key_values=h0, use_cache=True)
            self.gdn.layer_idx = saved_idx
        output = self.dropout(output)
        return output, new_cache

    def extra_repr(self):
        return (
            f'dim={self.dim}, expansion={self.expansion}, '
            f'num_heads={self.num_heads}, head_dim={self.actual_head_dim}, '
            f'LEVEL=FLA_GATED_DELTA_NET'
        )


# Note: We use FLAGatedDeltaNetLayer as the primary class name
# to avoid confusion with the imported FLAGatedDeltaNet from fla.layers


def count_fla_gdn_params(dim, depth, vocab_size=256, expansion=2.0):
    """
    Count FLA GatedDeltaNet parameters for config search.

    FLA's GatedDeltaNet has:
    - q_proj: dim x dim
    - k_proj: dim x dim
    - v_proj: dim x (dim * expand_v)
    - a_proj: dim x num_heads (alpha gate)
    - b_proj: dim x num_heads (beta gate)
    - q_conv1d: dim (depthwise)
    - k_conv1d: dim (depthwise)
    - v_conv1d: dim * expand_v (depthwise)
    - g_proj: dim x (dim * expand_v) (output gate)
    - o_norm: 2 * (dim * expand_v / num_heads) (FusedRMSNormGated per head)
    - o_proj: (dim * expand_v) x dim
    - layer_norm: dim
    """
    d_inner = int(dim * expansion)
    num_heads = max(1, dim // 128)  # Match our default head_dim=128

    per_layer = (
        dim * dim +              # q_proj
        dim * dim +              # k_proj
        dim * d_inner +          # v_proj
        dim * num_heads +        # a_proj (alpha)
        dim * num_heads +        # b_proj (beta)
        dim * 4 +                # q_conv1d (kernel_size=4, depthwise)
        dim * 4 +                # k_conv1d
        d_inner * 4 +            # v_conv1d
        dim * d_inner +          # g_proj (output gate)
        2 * d_inner +            # o_norm (weight + bias like params)
        d_inner * dim +          # o_proj
        dim                      # layer RMSNorm weight
    )

    total = (
        vocab_size * dim +       # embedding
        per_layer * depth +      # layers
        dim                      # final norm
    )
    return total


if __name__ == "__main__":
    print("Testing FLA GatedDeltaNet wrapper...")
    print("=" * 60)

    if not FLA_GDN_AVAILABLE:
        print("FLA not available! Install with: pip install flash-linear-attention")
        exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test basic forward/backward
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    model = FLAGatedDeltaNetLayer(
        dim=512,
        expansion=2.0,
        num_heads=4,
    ).to(device).bfloat16()

    print(f"\nModel:\n{model}")

    out, h = model(x)
    print(f"\nOutput shape: {out.shape}")
    print(f"Hidden state: {h}")

    # Test backward
    loss = out.sum()
    loss.backward()
    print(f"Backward passed!")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Parameter count estimate
    print("\nParameter count verification:")
    est_params = count_fla_gdn_params(512, 1) - 256 * 512 - 512  # Remove embedding/final norm
    print(f"Estimated per-layer params: {est_params:,}")
    print(f"Actual params: {params:,}")

    print("\n" + "=" * 60)
    print("FLA GatedDeltaNet: Optimized ICLR 2025 baseline")
    print("Uses Triton kernels for chunked linear attention")
    print("=" * 60)
