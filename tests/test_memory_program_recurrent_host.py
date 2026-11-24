"""CPU cache protocol double, not a second implementation of native GDN2 or a speed test."""

from types import SimpleNamespace as NS
import pytest
import torch
from torch import nn
from experiments.expressivity_tasks.x47_xsa_attention.large_host_check import check, make
from ndm.memory_program.recurrent_host import LayerExchange, RecurrentHost
from ndm.memory_program.host_cache import CachedByteHost


class Inner(nn.Module):
    def __init__(self, j):
        super().__init__()
        conv = NS(out_channels=4, kernel_size=(3,))
        core = NS(
            layer_idx=j,
            num_v_heads=1,
            head_k_dim=2,
            head_v_dim=2,
            q_conv1d=conv,
            k_conv1d=conv,
            v_conv1d=conv,
            use_short_conv=True,
        )
        self.gdn2 = NS(gdn2=core)
        self.j = j

    def forward(self, x, cache=None):
        b, t, d = x.shape
        if cache is not None and len(cache) > self.j:
            prior = cache[self.j]
            state = prior["recurrent_state"]
            cv = list(prior["conv_state"])
        else:
            state = x.new_zeros(b, 1, 2, 2)
            cv = [x.new_zeros(b, 4, 3) for _ in range(3)]
        out = []
        for row in x.unbind(1):
            for i, v in enumerate(cv):
                v.copy_(torch.cat([v[:, :, 1:], row.unsqueeze(2) * (i + 1)], 2))
            state = 0.25 * state + row.reshape_as(state)
            out.append(torch.tanh(state.flatten(1) + sum(v[:, :, i] for i, v in enumerate(cv))))
        if cache is not None:
            cache.update(layer_idx=self.j, recurrent_state=state, conv_state=tuple(cv), offset=t)
        return torch.stack(out, 1), cache


class Block(nn.Module):
    def __init__(self, j):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.layer = Inner(j)

    def forward(self, x, ids):
        return x + self.layer(self.norm(x))[0]


def net():
    torch.manual_seed(7620)
    m = nn.Module()
    m.embed = nn.Embedding(256, 4)
    m.blocks = nn.ModuleList([Block(i) for i in range(3)])
    m.norm = nn.LayerNorm(4)
    m.head = nn.Linear(4, 256)
    return m.eval()


def test_wiring_preserves_every_convolution_and_recurrent_component_across_clients():
    r = check(net())
    assert (
        r["checks"] == {"hidden": 12, "byte_logp": 12}
        and r["traffic"]["gather_bytes"] > 0
        and r["all_released_zero"]
    )


@torch.no_grad()
def test_long_host_is_explicit_mode_and_frontier_not_kernel_step_length():
    from experiments.expressivity_tasks.x47_xsa_attention.large_host_check import stock

    n = net()
    with pytest.raises(ValueError, match="chunk mode"):
        RecurrentHost(n, chunk_prefill=True)
    for b in n.blocks:
        b.layer.gdn2.gdn2.mode = "chunk"
    for long in (False, True):
        backend = RecurrentHost(n, chunk_prefill=long)
        capacity = 67 if long else 64
        a = CachedByteHost(backend, 2, capacity, 1 << 20)
        s = a.admit("test")
        word = bytes(range(64))
        a.prefill([s], [word])
        if long:
            for j in range(64, 67):
                h, p = a.advance([s], [j], [255])
                word += b"\xff"
                u, v = stock(n, [word], limit=2048)
                torch.testing.assert_close(h, u)
                torch.testing.assert_close(p, v)
        with pytest.raises(ValueError, match="frontier"):
            a.advance([s], [capacity], [0])
        a.release([s])
        assert all(not t.any() for t in a.slots.values())


@torch.no_grad()
def test_segmented_prompt_preserves_full_state_on_ragged_reorder_and_reuse():
    n = net()
    a = CachedByteHost(RecurrentHost(n, segmented_prefill=True), 2, 160, 1 << 20)
    b = CachedByteHost(RecurrentHost(n, segmented_prefill=True), 2, 160, 1 << 20)
    hs = [a.admit(str(i)) for i in range(2)]
    ref = [b.admit(str(i)) for i in range(2)]
    ws = [bytes(range(65)), bytes(range(159))]
    a.prefill(hs, ws)
    b.prefill(ref, [w[:1] for w in ws])
    assert a.backend.prefill_work == {"calls": 5, "tokens": 224}
    for j in range(1, 159):
        ix = [i for i, w in enumerate(ws) if j < len(w)]
        b.advance([ref[i] for i in ix], [j] * len(ix), [ws[i][j] for i in ix])
    for k, t in a.slots.items():
        torch.testing.assert_close(t, b.slots[k])
    a.advance(hs[::-1], [159, 65], [255, 0])
    b.advance(ref[::-1], [159, 65], [255, 0])
    with pytest.raises(ValueError):
        a.advance([hs[1]], [160], [0])
    old = hs[0]
    a.release([old])
    new = a.admit("new")
    a.prefill([new], [b"x" * 130])
    with pytest.raises(ValueError):
        a.read([old], [66])
    for k, t in a.slots.items():
        torch.testing.assert_close(t[hs[1].slot], b.slots[k][ref[1].slot])
    a.clear()
    assert a.empty and all(not t.any() for t in a.slots.values())


def test_layer_exchange_rejects_wrong_layer_dtype_and_missing_conv_state():
    a = make(net(), 2)
    idx = torch.tensor([1])
    e = LayerExchange(0, idx, a.slots, False, 1, a.backend.stats)
    with pytest.raises(ValueError):
        e[1]
    o = e[0]
    cv = o["conv_state"]
    s = o["recurrent_state"]
    s.fill_(1)
    assert not a.slots["recurrent"].any()  # native mutation of gather not alias on the owner
    with pytest.raises(ValueError):
        e.update(layer_idx=1, offset=1, recurrent_state=s, conv_state=cv)
    with pytest.raises(ValueError):
        e.update(layer_idx=0, offset=2, recurrent_state=s, conv_state=cv)
    with pytest.raises(ValueError):
        e.update(layer_idx=0, offset=1, recurrent_state=s, conv_state=cv[:2])
    with pytest.raises(ValueError):
        e.update(layer_idx=0, offset=1, recurrent_state=s.double(), conv_state=cv)
    e.update(layer_idx=0, offset=1, recurrent_state=s, conv_state=cv)
    assert a.slots["recurrent"][1, 0].eq(1).all() and not a.slots["recurrent"][0].any()
    with pytest.raises(ValueError):
        e.update(layer_idx=0, offset=1, recurrent_state=s, conv_state=cv)


@torch.no_grad()
def test_segment_recording_keeps_every_position_causal_across_pieces_and_capacity_reuse():
    from ndm.memory_program.segment_packet import SegmentRecorder
    from ndm.memory_program.stream_packet import record_frontiers
    from ndm.memory_program.frozen_packet import Window, PacketBinding

    n = net().requires_grad_(False)
    w = [bytes((i + j * j) % 256 for j in range(129)) for i in (0, 31, 255)]
    b = PacketBinding(
        "b" * 64, "c" * 64, "train", 4, tuple(Window.observe(i, 21, s) for i, s in enumerate(w))
    )
    a = CachedByteHost(RecurrentHost(n, segmented_prefill=True), 2, 130, 1 << 20)
    with SegmentRecorder(n, 2, 130, byte_budget=1 << 20) as rec:
        x = rec.record(b, w, byte_limit=1 << 20)
        y = record_frontiers(a, b, w, byte_limit=1 << 20)
        torch.testing.assert_close(x, y, rtol=2e-5, atol=5e-5)
        assert (
            rec.work == {"pieces": 6, "observed": 387, "released_rows": 3}
            and all(not z.any() for z in rec.state.values())
            and x.shape == (3, 129, 260)
        )
    assert rec.closed and not rec.state


@torch.no_grad()
def test_segment_recorder_checks_bounds_before_read_and_drops_partial_native_state():
    from ndm.memory_program.segment_packet import SegmentRecorder
    from ndm.memory_program.frozen_packet import Window, PacketBinding

    n = net().requires_grad_(False)
    w = b"ab" * 33
    b = PacketBinding("b" * 64, "c" * 64, "train", 4, (Window.observe(2, 7, w),))
    with pytest.raises(ValueError, match="budget"):
        SegmentRecorder(n, 1, 128, byte_budget=2)
    r = SegmentRecorder(n, 1, 128, byte_budget=1 << 16)
    store = list(r.state.values())
    old = r.host.frontiers
    with pytest.raises(ValueError):
        r.record(b, [w[::-1]], byte_limit=1 << 20)
    assert not r.work and not r.closed

    def injected(*args, **kw):
        out = old(*args, **kw)
        if not kw["fresh"]:
            raise RuntimeError("partial layer write")
        return out

    r.host.frontiers = injected
    with pytest.raises(RuntimeError):
        r.record(b, [w], byte_limit=1 << 20)
    assert r.closed and not r.state and all(not t.any() for t in store)
