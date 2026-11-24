"""Exclusive frozen GDN2 feature producer. Never put a memory program or unknown output in this prefill."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from math import prod
from threading import get_ident
import torch
from .recurrent_host import RecurrentHost


class SegmentRecorder:
    """One private, fixed-capacity recurrent store. Every block starts from fresh native state.

    byte_budget limits the four persistent state tensors, not transient activations or output storage.
    A binding describes one length bucket of observed, document-bounded bytes. This interface cannot
    generate future inputs, share a live serving admission, or propagate gradients through the host.
    """

    def __init__(self, net, capacity, context, *, byte_budget):
        if (
            any(type(v) is not int or v < 1 for v in (capacity, context, byte_budget))
            or context > 2048
            or net.training
            or any(p.requires_grad for p in net.parameters())
        ):
            raise ValueError("invalid frozen recorder model/bounds")
        self.host = RecurrentHost(net, segmented_prefill=True)
        self.capacity = capacity
        self.context = context
        self.closed = False
        self.busy = False
        self.thread = get_ident()
        self.work = Counter()
        self.stamp = self._stamp()
        spec = self.host.spec(context)
        self.tensor_bytes = capacity * 4 * sum(prod(s) for s, dtype in spec.values())
        if self.tensor_bytes > byte_budget or any(dt != torch.float32 for _, dt in spec.values()):
            raise ValueError("frozen recorder state budget/precision")
        self.state = {
            k: torch.zeros((capacity, *s), device=self.host.device, dtype=dt) for k, (s, dt) in spec.items()
        }
        self.stream = torch.cuda.current_stream(self.host.device) if self.host.device.type == "cuda" else None

    def _stamp(self):
        return tuple(
            (id(p), p._version, p.device, p.dtype, p.requires_grad) for p in self.host.net.parameters()
        )

    def _owner(self):
        return self.thread == get_ident() and (
            self.stream is None or self.stream == torch.cuda.current_stream(self.host.device)
        )

    @torch.no_grad()
    def record(self, binding, words, *, byte_limit):
        binding.layout(byte_limit)
        n, t, d = binding.shape
        if (
            self.closed
            or self.busy
            or not self._owner()
            or self.host.net.training
            or self._stamp() != self.stamp
            or torch.is_autocast_enabled(self.host.device.type)
            or binding.width > self.host.width
            or t > self.context
            or len(words) != n
            or any(
                type(w) is not bytes or len(w) != t or sha256(w).hexdigest() != r.input_sha
                for w, r in zip(words, binding.rows, strict=True)
            )
        ):
            raise ValueError("invalid frozen recorder or changed input")
        self.busy = True
        try:
            out = torch.empty((n, t, d), dtype=torch.float32, device="cpu")
            for pos in range(0, n, self.capacity):
                ws = words[pos : pos + self.capacity]
                b = len(ws)
                slots = torch.arange(b, device=self.host.device)
                buf = torch.empty((b, t, d), dtype=torch.float32, device=self.host.device)
                for j in range(0, t, 64):
                    stop = min(j + 64, t)
                    ids = torch.tensor(
                        [list(w[j:stop]) for w in ws], device=self.host.device, dtype=torch.long
                    )
                    h, p = self.host.frontiers(self.state, slots, ids, fresh=j == 0)
                    if (
                        h.shape != (b, stop - j, self.host.width)
                        or p.shape != (b, stop - j, 256)
                        or h.dtype != p.dtype
                        or p.dtype != torch.float32
                    ):
                        raise ValueError("bad recorded segment")
                    buf[:, j:stop, : binding.width].copy_(h[:, :, : binding.width])
                    buf[:, j:stop, binding.width :].copy_(p)
                    self.work["pieces"] += 1
                    self.work["observed"] += ids.numel()
                out[pos : pos + b].copy_(buf)
                for s in self.state.values():
                    s[:b].zero_()
                self.work["released_rows"] += b
            if (
                not out.isfinite().all().item()
                or out[:, :, binding.width :].logsumexp(-1).abs().max().item() > 5e-5
            ):
                raise ValueError("nonfinite/unnormalized frozen packet")
            return out
        except BaseException:
            # An interrupted update could have partially mutated a layer. Drop this store.
            self.closed = True
            for s in self.state.values():
                s.zero_()
            self.state.clear()
            raise
        finally:
            self.busy = False

    def close(self):
        if self.closed:
            return
        if not self._owner() or self.busy:
            raise ValueError("cannot close active or foreign recorder")
        if any(v.count_nonzero().item() for v in self.state.values()):
            raise ValueError("unreleased segment state")
        self.state.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
