"""Session-store adapter for the optional external GDN2 eval block. No vendored model/kernels.

Keep all persistent state under CachedByteHost's byte/ownership contract. The native layer (not
this adapter) computes the recurrence. Short-convolution step mutates its input, so a dispatched
batch gets a private gather of *all four* state components, in request order.
"""

from collections import Counter
import torch


class LayerExchange:
    """One native layer's get/update during one call. Never a shared global FLA cache."""

    def __init__(self, j, slots, arena, prompt, length, stats):
        self.j = j
        self.slots = slots
        self.arena = arena
        self.prompt = prompt
        self.length = length
        self.stats = stats
        self.read = False
        self.wrote = False

    def __len__(self):
        return 0 if self.prompt else self.j + 1

    def __getitem__(self, j):
        if self.read or self.wrote or self.prompt or j != self.j:
            raise ValueError("wrong recurrent layer read")
        self.read = True

        def get(k):
            x = self.arena[k][:, j].index_select(0, self.slots)
            self.stats["gather_bytes"] += x.numel() * x.element_size()
            return x

        return dict(recurrent_state=get("recurrent"), conv_state=tuple(get(k) for k in ("q", "k", "v")))

    def update(self, *, layer_idx, recurrent_state, conv_state, offset):
        if (
            self.wrote
            or (not self.prompt) != self.read
            or layer_idx != self.j
            or offset != self.length
            or not isinstance(conv_state, (list, tuple))
            or len(conv_state) != 3
        ):
            raise ValueError("wrong recurrent cache update")
        for k, x in zip(("recurrent", "q", "k", "v"), (recurrent_state, *conv_state), strict=True):
            dst = self.arena[k][:, layer_idx]
            if (
                not isinstance(x, torch.Tensor)
                or x.shape != (len(self.slots), *dst.shape[1:])
                or x.dtype != dst.dtype
                or x.device != dst.device
            ):
                raise ValueError(f"incompatible native final state {k}: {getattr(x, 'shape', None)}")
            dst.index_copy_(0, self.slots, x)
            self.stats["scatter_bytes"] += x.numel() * x.element_size()
        self.wrote = True


class RecurrentHost:
    """FP32 GDN2ExternalMLPLayer + outer residual. Default is the short recurrent clock."""

    max_context = 64

    def __init__(self, net, *, chunk_prefill=False, segmented_prefill=False):
        if (
            type(chunk_prefill) is not bool
            or type(segmented_prefill) is not bool
            or chunk_prefill
            and segmented_prefill
        ):
            raise ValueError("select at most one explicit long prefill method")
        self.max_context = 2048 if chunk_prefill or segmented_prefill else 64
        self.segmented_prefill = segmented_prefill
        self.prefill_work = Counter()
        self.net = net
        self.device = net.embed.weight.device
        self.width = net.embed.embedding_dim
        self.shapes = {}
        self.stats = Counter()
        if (
            net.training
            or net.embed.num_embeddings != 256
            or net.head.out_features != 256
            or any(p.dtype != torch.float32 or p.device != self.device for p in net.parameters())
        ):
            raise ValueError("requires FP32 eval byte host")
        depth = len(net.blocks)
        for j, block in enumerate(net.blocks):
            core = block.layer.gdn2.gdn2
            if chunk_prefill and getattr(core, "mode", None) != "chunk":
                raise ValueError("long prefill requires native chunk mode")
            if not core.use_short_conv or core.layer_idx != j:
                raise ValueError("GDN2 layer/cache binding")
            shapes = {"recurrent": (depth, core.num_v_heads, core.head_k_dim, core.head_v_dim)}
            for k in ("q", "k", "v"):
                c = getattr(core, f"{k}_conv1d")
                shapes[k] = (depth, c.out_channels, c.kernel_size[0])
            if self.shapes and shapes != self.shapes:
                raise ValueError("heterogeneous GDN2 arena not supported")
            self.shapes = shapes
        if not self.shapes:
            raise ValueError("empty GDN2")

    def spec(self, context):
        return {k: (shape, torch.float32) for k, shape in self.shapes.items()}

    def encode(self, arena, slots, old, ids, prompt):
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError("FP32 recurrent cache required")
        if prompt and self.segmented_prefill:
            # old is validated by the session owner. Intermediate states stay in this
            # admission's slots. A later piece is continuation, not a fresh prompt.
            for i, part in enumerate(ids.split(64, dim=1)):
                result = self._piece(arena, slots, part, i == 0)
                self.prefill_work["calls"] += 1
                self.prefill_work["tokens"] += part.numel()
            return result
        return self._piece(arena, slots, ids, prompt)

    def frontiers(self, arena, slots, ids, *, fresh):
        """Every output of a known short causal piece, for a frozen training-feature producer only."""
        if (
            not self.segmented_prefill
            or self.net.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled(self.device.type)
            or type(fresh) is not bool
            or ids.ndim != 2
            or ids.shape[0] != len(slots)
            or not 0 < ids.shape[1] <= 64
        ):
            raise ValueError("invalid causal recording piece")
        return self._piece(arena, slots, ids, fresh, all_frontiers=True)

    def _piece(self, arena, slots, ids, prompt, *, all_frontiers=False):
        x = self.net.embed(ids)
        for j, block in enumerate(self.net.blocks):
            exchange = LayerExchange(j, slots, arena, prompt, ids.shape[1], self.stats)
            delta, final = block.layer(block.norm(x), exchange)
            if final is not exchange or not exchange.wrote:
                raise ValueError("native layer did not return its bound cache")
            x = x + delta
        h = self.net.norm(x if all_frontiers else x[:, -1])
        return h, self.net.head(h).float().log_softmax(-1)
