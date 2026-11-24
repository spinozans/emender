"""Incremental byte host conformance; synthetic byte strings, not service timings."""

from collections import Counter
import torch
from ndm.memory_program.host_cache import AbsoluteKV, CachedByteHost
from .serving_bench import causal_last


def make_cache(net, capacity, context=128, budget=16 << 20):
    return CachedByteHost(AbsoluteKV(net), capacity, context, budget)


@torch.no_grad()
def check_cache(net):
    device = net.embed.weight.device
    a = make_cache(net, 3)
    ss = [a.admit(f"fixture:{i}") for i in range(3)]
    words = [b"\x00", bytes([30, 255, 0, 127, 89]), bytes((j * 31 + 7) % 256 for j in range(127))]
    err = Counter()
    checks = Counter()

    def compare(out, idx):
        # Stock full-prefix separately per document; never a padded target.
        for j, i in enumerate(idx):
            w = words[i]
            h, p = causal_last(net, [w], device)
            z = net(torch.tensor([list(w)], device=device))[:, -1].float().log_softmax(-1)
            for key, x, y in [
                ("hidden", out[0][j], h[0]),
                ("byte_logp", out[1][j], z[0]),
                ("full_control", p, z),
            ]:
                torch.testing.assert_close(
                    x, y, atol=5e-5, rtol=2e-5, msg=lambda msg: f"{key} stream={i} visible={len(w)}: {msg}"
                )
                err[key] = max(err[key], (x - y).abs().max().item())
                checks[key] += 1

    compare(a.prefill(ss, words), [0, 1, 2])
    before = a.stats.copy()
    compare(a.read([ss[2], ss[0]], [127, 1]), [2, 0])
    assert a.stats == before
    for ids, bs in [([2, 0, 1], [0, 255, 30]), ([1, 0], [0, 255]), ([0], [30])]:
        hs = [ss[i] for i in ids]
        old = [len(words[i]) for i in ids]
        out = a.advance(hs, old, bs)
        for i, b in zip(ids, bs, strict=True):
            words[i] += bytes([b])
        compare(out, ids)
    try:
        a.advance([ss[2]], [128], [0])
        raise AssertionError("accepted 129th byte")
    except ValueError:
        pass
    a.release([ss[1]])
    assert all(t[ss[1].slot].count_nonzero().item() == 0 for t in a.slots.values())
    ss[1] = a.admit("new-window")
    words[1] = b"\xff\x1e"
    compare(a.prefill([ss[1]], [words[1]]), [1])
    compare(a.read([ss[2]], [128]), [2])
    a.clear()
    assert all(t.count_nonzero().item() == 0 for t in a.slots.values())
    return dict(
        max_error=dict(err),
        checks=dict(checks),
        stats=dict(a.stats),
        tensor_bytes=a.tensor_bytes,
        all_released_zero=True,
    )
