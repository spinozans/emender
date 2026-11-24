"""Conformance oracle shared by the deterministic CPU double and retained native GPU host."""

from collections import Counter, defaultdict
import torch
from ndm.memory_program.host_cache import CachedByteHost
from ndm.memory_program.recurrent_host import RecurrentHost


def make(net, b):
    return CachedByteHost(RecurrentHost(net), b, 64, 128 << 20)


def stock(net, words, *, limit=64):
    """Independent from LayerExchange: original block.forward, no retained state."""
    groups = defaultdict(list)
    for i, w in enumerate(words):
        if not 0 < len(w) <= limit <= 2048:
            raise ValueError("outside native recurrent clock")
        groups[len(w)].append(i)
    order = [None] * len(words)
    for ix in groups.values():
        ids = torch.tensor([list(words[i]) for i in ix], device=net.embed.weight.device, dtype=torch.long)
        x = net.embed(ids)
        for block in net.blocks:
            x = block(x, ids)
        h = net.norm(x)[:, -1].contiguous()
        lp = net.head(h).float().log_softmax(-1)
        if len(groups) == 1:
            return h, lp
        for i, y, p in zip(ix, h, lp, strict=True):
            order[i] = (y, p)
    return tuple(torch.stack(v) for v in zip(*order, strict=True))


@torch.no_grad()
def check(net):
    arena = make(net, 3)
    ss = [arena.admit(f"qualification:{i}") for i in range(3)]
    words = [b"\x00", b"line\x1eone;end\xff", bytes((31 * i + 7) % 256 for i in range(63))]
    err = Counter()
    seen = Counter()

    def compare(out, idx):
        for j, i in enumerate(idx):
            h, p = stock(net, [words[i]])
            for key, x, y in [("hidden", out[0][j], h[0]), ("byte_logp", out[1][j], p[0])]:
                torch.testing.assert_close(
                    x, y, atol=5e-4, rtol=2e-4, msg=lambda msg: f"{key} stream={i}, t={len(words[i])}: {msg}"
                )
                err[key] = max(err[key], (x - y).abs().max().item())
                seen[key] += 1
            z = out[1][j]
            assert z.isfinite().all() and z.logsumexp(0).abs().item() < 1e-5 and z.argmax() == p[0].argmax()

    compare(arena.prefill(ss, words), [0, 1, 2])
    compare(arena.read([ss[2], ss[0]], [63, 1]), [2, 0])
    for ids, bs in [([2, 0, 1], [30, 255, 0]), ([1, 0], [96, 0])]:
        out = arena.advance([ss[i] for i in ids], [len(words[i]) for i in ids], bs)
        for i, b in zip(ids, bs, strict=True):
            words[i] += bytes([b])
        compare(out, ids)
    old = ss[1]
    arena.release([old])
    assert all(t[old.slot].count_nonzero().item() == 0 for t in arena.slots.values())
    ss[1] = arena.admit("different-document")
    words[1] = b"\xff\x1e\x00"
    compare(arena.prefill([ss[1]], [words[1]]), [1])
    compare(arena.read([ss[2]], [64]), [2])
    for fn in [lambda: arena.advance([ss[2]], [64], [7]), lambda: arena.read([old], [3])]:
        try:
            fn()
            raise AssertionError("bad frontier/admission accepted")
        except ValueError:
            pass
    arena.clear()
    assert all(t.count_nonzero().item() == 0 for t in arena.slots.values())
    return dict(
        checks=seen,
        max_error=err,
        all_argmax_match=True,
        all_released_zero=True,
        stats=arena.stats,
        shapes=arena.backend.shapes,
        tensor_bytes=arena.tensor_bytes,
        traffic=arena.backend.stats,
    )
