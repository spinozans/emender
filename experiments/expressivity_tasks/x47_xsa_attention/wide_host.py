"""Four-copy dense *cost probe*. No training, no new independent model capacity claimed."""

from __future__ import annotations
from collections import Counter
from math import prod
import torch
from torch import nn


def copies(src, dst, module, field, r):
    if (
        src.ndim != dst.ndim
        or not src.ndim
        or any(d % s or d // s not in (1, r) for s, d in zip(src.shape, dst.shape))
    ):
        raise ValueError("invalid replicated parameter geometry")
    reps = tuple(d // s for s, d in zip(src.shape, dst.shape))
    factor = 1
    if isinstance(module, nn.Linear) and field in ("weight", "bias"):
        if field == "weight":
            factor = reps[1]
    elif isinstance(module, nn.Embedding) and field == "weight" and reps[0] == 1:
        pass
    elif (
        isinstance(module, nn.Conv1d)
        and module.groups == module.in_channels == module.out_channels
        and field == "weight"
        and reps[1:] == (1, 1)
    ):
        pass
    elif (
        (isinstance(module, (nn.RMSNorm, nn.LayerNorm)) or type(module).__name__ == "FusedRMSNormSwishGate")
        and field in ("weight", "bias")
        and src.ndim == 1
    ):
        pass
    elif type(module).__name__ == "GatedDeltaNet2" and field in ("A_log", "dt_bias") and src.ndim == 1:
        pass
    else:
        raise ValueError(f"unsupported replication rule: {type(module).__name__}.{field}")
    return reps, factor


@torch.no_grad()
def expand_into(small, wide, r):
    if type(r) is not int or r < 2 or r & (r - 1):
        raise ValueError("power-of-two expansion required")
    p = dict(small.named_parameters())
    q = dict(wide.named_parameters())
    modules = dict(wide.named_modules())
    srcmods = dict(small.named_modules())
    if p.keys() != q.keys() or small.state_dict().keys() != p.keys() or wide.state_dict().keys() != q.keys():
        raise ValueError("parameter or buffer coverage")
    srcids = {v.untyped_storage().data_ptr() for v in p.values()}
    checked = 0
    storage = set()
    rules = Counter()
    for key, x in p.items():
        y = q[key]
        prefix, _, field = key.rpartition(".")
        mod = modules[prefix]
        if type(srcmods[prefix]) is not type(mod):
            raise ValueError("changed parameter owner type")
        ptr = y.untyped_storage().data_ptr()
        if (
            x.device != y.device
            or y.dtype != x.dtype
            or x.dtype != torch.float32
            or ptr in storage
            or ptr in srcids
            or not y.is_contiguous()
            or y.untyped_storage().nbytes() != y.numel() * 4
        ):
            raise ValueError("not independent dense FP32 storage")
        storage.add(ptr)
        reps, scale = copies(x, y, mod, field, r)
        layout = tuple(n for rep, s in zip(reps, x.shape) for n in (rep, s))
        pattern = x.reshape(tuple(n for s in x.shape for n in (1, s)))
        if scale != 1:
            pattern = pattern / scale
        target = y.view(layout)
        target.copy_(pattern)
        if not torch.eq(target, pattern).all().item():
            raise ValueError("inexact parameter copy")
        checked += y.numel()
        rules[f"{type(mod).__name__}.{field}, input divisor {scale}"] += 1
    return dict(
        parameters_verified=checked, tensors=len(q), distinct_storages=len(storage), rules=dict(rules)
    )


class Block(nn.Module):
    def __init__(self, old, j, r):
        super().__init__()
        from ndm.models.external_gdn2 import GDN2ExternalMLPLayer

        layer = old.layer
        core = layer.gdn2.gdn2
        d = r * layer.dim
        self.norm = nn.RMSNorm(d, eps=old.norm.eps)
        self.layer = GDN2ExternalMLPLayer(
            d,
            expansion=core.expand_v,
            head_dim=core.head_k_dim,
            num_heads=r * core.num_heads,
            gdn2_mlp_ratio=layer.mlp_hidden_dim / layer.dim,
            gdn2_mlp_multiple=1,
            use_conv=True,
            d_conv=core.q_conv1d.kernel_size[0],
        )
        self.layer.set_layer_idx(j)

    def forward(self, x, ids):
        return x + self.layer(self.norm(x))[0]


class WideHost(nn.Module):
    def __init__(self, small, r):
        super().__init__()
        self.width = r * small.width
        self.embed = nn.Embedding(256, self.width)
        self.norm = nn.RMSNorm(self.width, eps=small.norm.eps)
        self.head = nn.Linear(self.width, 256, bias=False)
        self.blocks = nn.ModuleList([Block(b, j, r) for j, b in enumerate(small.blocks)])


def build(small):
    with torch.device("meta"):
        wide = WideHost(small, 4)
    if sum(x.numel() for x in wide.parameters()) != 2221786584:
        raise ValueError("wrong wide shape")
    wide.to_empty(device=small.embed.weight.device).eval().requires_grad_(False)
    return wide, expand_into(small, wide, 4)
