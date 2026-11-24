"""Record a frozen byte host as served, without handing it a future observation.

This deliberately uses one physical frontier per observation. It is not a fast full-sequence
training kernel or a claim that training- and inference-mode features are interchangeable.
"""

from __future__ import annotations

from hashlib import sha256

import torch


@torch.no_grad()
def record_frontiers(arena, binding, words, *, byte_limit):
    """Return owned CPU [docs,time,H+256] on success. The caller owns the entire empty arena.

    The binding pins the caller's chosen model/bridge as well as each document/offset/input hash.
    The bridge here is specifically the first `width` hidden coordinates + normalized host logp.
    Split ragged batches by true length before calling; never pad them or carry a slot across rows.
    """
    binding.layout(byte_limit)
    rows = binding.rows
    n, t, d = binding.shape
    if (
        not arena.empty
        or arena.net.training
        or any(p.requires_grad for p in arena.net.parameters())
        or binding.width > arena.backend.width
        or t > arena.context
        or len(words) != n
    ):
        raise ValueError("incompatible frozen arena or packet geometry")
    for w, r in zip(words, rows, strict=True):
        if type(w) is not bytes or len(w) != t or sha256(w).hexdigest() != r.input_sha:
            raise ValueError("wrong frozen packet observations")
    result = torch.empty((n, t, d), device="cpu", dtype=torch.float32)
    for first in range(0, n, arena.capacity):
        batch = words[first : first + arena.capacity]
        b = len(batch)
        try:
            hs = [arena.admit(f"window:{r.document}:{r.start}") for r in rows[first : first + b]]
            buf = torch.empty((b, t, d), device=arena.device, dtype=torch.float32)
            for j in range(t):
                # Only w[j] crosses the host boundary. The policy later consumes these frontiers in order.
                if j == 0:
                    h, lp = arena.prefill(hs, [w[:1] for w in batch])
                else:
                    h, lp = arena.advance(hs, [j] * b, [w[j] for w in batch])
                if (
                    h.dtype != torch.float32
                    or lp.dtype != torch.float32
                    or h.shape != (b, arena.backend.width)
                    or lp.shape != (b, 256)
                ):
                    raise ValueError("invalid host frontier")
                buf[:, j, : binding.width].copy_(h[:, : binding.width])
                buf[:, j, binding.width :].copy_(lp)
            result[first : first + b].copy_(buf)
            arena.release(hs)
        except BaseException:
            arena.clear()
            raise
    if (
        not arena.empty
        or not result.isfinite().all().item()
        or result[:, :, binding.width :].logsumexp(-1).abs().max().item() > 5e-5
    ):
        raise ValueError("invalid recorded host packet")
    return result
