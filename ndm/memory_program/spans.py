"""Bounded document span → real text and optional differentiable byte values.

The application owns admission, immutable file binding / live watermark, and verifies its
locator. A Span is an address, not a grant to open a file. Never dispatch by its source name.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import accumulate

import torch
from torch.nn.functional import embedding_bag

from .runtime import Field, Hit, Reader, Span, Unavailable


def mean_byte_values(fragments, weight):
    """One differentiable embedding per actual nonempty fragment. No padding or result cache."""
    if not fragments or weight.ndim != 2 or len(weight) < 256 or any(not f for f in fragments):
        raise ValueError("invalid byte encoding")
    offsets = [0, *accumulate(map(len, fragments))]
    data = torch.tensor(list(b"".join(fragments)), device=weight.device, dtype=torch.long)
    return embedding_bag(
        data, weight[:256], torch.tensor(offsets, device=weight.device), mode="mean", include_last_offset=True
    )


class SpanValues:
    """bounds(doc) → [lo, hi) document-relative visible interval or None. Read only within it."""

    def __init__(self, source, version, bounds, read, *, scope, encode=None, width=16, **binding):
        if scope != "whole":
            raise PermissionError("partial span scope unsupported")
        if not 1 <= width <= 256:
            raise ValueError("invalid passage width")
        self.source, self.version = source, version
        self.bounds, self.read, self.encode = bounds, read, encode
        self.width, self.binding = width, binding

    def reader(self, *, following=False):
        fields = (
            (Field("first", "byte"), Field("text", "text"), Field("region", "span"))
            if following
            else (Field("text", "text"),)
        )

        def many(keys, cap):
            out = []
            fragments = []
            positions = []
            for k in keys:
                if not isinstance(k, Span) or (k.source, k.version) != (self.source, self.version):
                    out.append(Unavailable("foreign span"))
                    continue
                extent = self.bounds(k.document)
                if extent is None or not extent[0] <= k.start < k.stop <= extent[1]:
                    out.append(Unavailable("outside visible document"))
                    continue
                if k.stop - k.start > 256:
                    out.append(Unavailable("span width"))
                    continue
                a, b = (k.stop, min(k.stop + self.width, extent[1])) if following else (k.start, k.stop)
                if a == b:
                    out.append(())
                    continue  # no visible continuation; never emit END or padding
                if cap < 1:
                    out.append(Unavailable("span row budget"))
                    continue
                text = self.read(k.document, a, b - a)
                if not isinstance(text, bytes) or len(text) != b - a:
                    raise ValueError("incomplete span read")
                bindings = (text[:1], text, replace(k, start=a, stop=b)) if following else (text,)
                positions.append(len(out))
                fragments.append(text)
                out.append((Hit(bindings),))
            if self.encode is not None and fragments:
                for i, v in zip(positions, self.encode(fragments), strict=True):
                    out[i] = (replace(out[i][0], value=v),)
            return out

        return Reader(
            self.source,
            self.version,
            "following" if following else "contents",
            "span",
            fields,
            many,
            **self.binding,
        )
