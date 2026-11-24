"""Compact read ABI. Device slots are NOT archive positions. See README.md.

The native/provider layer establishes scope, membership, completeness and value
identity. This layer checks the numeric page addresses once, independently of
model forwards. Treat a bound page's index tensor as read-only.
"""
from dataclasses import dataclass

import torch

MAX_FAN = 64
MAX_WIDTH = 128


@dataclass(frozen=True)
class BoundPage:
    ids: torch.Tensor  # R x F contiguous int32; one row per (query, head, bank)
    records: int       # K,V working-set size, not size of the indexed collection

    @classmethod
    def from_cpu(cls, ids, records, *, device='cpu'):
        if not isinstance(records, int) or not 0 <= records <= 2**31 - 1:
            raise ValueError('invalid device table size')
        if (ids.device.type != 'cpu' or ids.dtype not in (torch.int32, torch.int64)
                or ids.ndim != 2 or ids.shape[0] < 1 or not 1 <= ids.shape[1] <= MAX_FAN):
            raise ValueError('expected nonempty CPU integer page with bounded width')
        # Check *before* narrowing: a large logical archive offset must not
        # quietly wrap to a valid local slot. Only -1 denotes padding.
        if ids.min().item() < -1 or ids.max().item() >= records:
            raise ValueError('out-of-range compact slot; bind logical addresses first')
        return cls(ids.to(device=device, dtype=torch.int32, copy=True).contiguous(), records)


def check_layout(q, k, v, page, bias):
    ids = page.ids
    if (q.ndim != 2 or k.ndim != 2 or k.shape != v.shape or ids.ndim != 2
            or ids.shape != bias.shape or ids.shape[0] != q.shape[0] or not q.shape[0]
            or k.shape != (page.records, q.shape[1]) or not 1 <= ids.shape[1] <= MAX_FAN
            or not 1 <= q.shape[1] <= MAX_WIDTH or ids.dtype != torch.int32
            or q.dtype != k.dtype or k.dtype != v.dtype or not q.is_floating_point()
            or not bias.is_floating_point()):
        raise ValueError('incompatible queries, key/value table, log priors or bound page')
    if not all(x.device == q.device and x.is_contiguous() for x in (q, k, v, ids, bias)):
        raise ValueError('read expects contiguous buffers on the same device')


def reference_read(q, k, v, page, bias):
    """Vectorized eager baseline; 'bias' is log prior mass, not a mask.

    One normalization per bank row. Caller mixes returned rows across banks.
    Duplicate slots are legal (additive evidence); distinct source slots must
    not alias unless they really use the same K and V. FP32 output/accumulation
    unless the reference is run in FP64. No whole-index tensor materialization.
    """
    check_layout(q, k, v, page, bias)
    acc = torch.float64 if q.dtype == torch.float64 else torch.float32
    if not page.records:
        # Keep the all-empty result connected to the four zero adjoints.
        return q.to(acc) * 0 + (k.to(acc).sum() + v.to(acc).sum() + (bias.to(acc)*0).sum())*0
    ids = page.ids.long()
    valid = ids >= 0
    slots = ids.clamp_min(0)
    # Gather FIRST, so fp32 casting for BF16 never copies the entire corpus table.
    keys, values = k[slots].to(acc), v[slots].to(acc)
    scores = bias.to(acc) + (q.to(acc).unsqueeze(1) * keys).sum(-1) * q.shape[1]**-.5
    scores = scores.masked_fill(~valid, -torch.inf)
    safe = torch.where(valid.any(1, keepdim=True), scores, 0.)
    weights = safe.softmax(1).masked_fill(~valid, 0.)
    return (weights.unsqueeze(2) * values).sum(1)
