from dataclasses import replace
from hashlib import sha256

import pytest
import torch
from ndm.memory_program import Session
from ndm.memory_program.frozen_packet import Window, PacketBinding, FrozenPacket, write_packet


def sample():
    rows = (Window.observe(7, 100, b"\x00\x1e\xff"), Window.observe(11, 31, b"bar"))
    spec = PacketBinding("a" * 64, "b" * 64, "train", 4, rows)
    x = torch.tensor([[0, 30, 255], [98, 97, 114]])
    return spec, x, torch.arange(2 * 3 * 260).view(spec.shape).float()


def test_packet_is_exact_bound_to_fresh_origins_and_no_mutable_alias(tmp_path):
    spec, x, v = sample()
    path = tmp_path / "data"
    sha, size = write_packet(path, spec, v, byte_limit=16384)
    assert (
        path.stat().st_size == size
        and sha == sha256(path.read_bytes()).hexdigest()
        and path.stat().st_mode & 0o777 == 0o600
    )
    bound = FrozenPacket.load(path, spec, sha, byte_limit=size)
    ss = [Session(document=r.document, start=r.start) for r in spec.rows]
    encode = bound.encoder(ss)
    h, p = encode(x)
    assert torch.equal(v, torch.cat((h, p), -1)) and not h.requires_grad
    h.fill_(100)
    assert torch.equal(encode(x)[0], v[:, :, :4])
    v.zero_()
    assert encode(x)[0].abs().max() > 0
    for ids in [x.flip(0), x + 256, x.int(), x[:, :2]]:
        with pytest.raises(ValueError):
            encode(ids)
    with pytest.raises(ValueError):
        bound.encoder(ss[::-1])(x)
    ss[0].observe(b"x")
    with pytest.raises(ValueError):
        encode(x)


def test_verified_row_subset_checks_actual_new_session_and_full_observed_bytes():
    spec, x, v = sample()
    packet = FrozenPacket(spec, v)
    ss = [Session(document=r.document, start=r.start) for r in spec.rows]
    for ix in ([1], [1, 0]):
        select = packet.encoder([ss[i] for i in ix], row_indices=ix)
        h, p = select(x[ix])
        assert torch.equal(torch.cat([h, p], -1), v[ix])
        h.zero_()
        assert torch.equal(select(x[ix])[0], v[ix, :, :4])
    for ix in ([0, 0], [-1], [2], [], [False]):
        with pytest.raises(ValueError):
            packet.encoder(ss, row_indices=ix)
    with pytest.raises(ValueError):
        packet.encoder(ss[:1], row_indices=[1])(x[1:])
    with pytest.raises(ValueError):
        packet.encoder(ss[1:], row_indices=[1])(x[:1])
    ss[1].observe(b"x")
    with pytest.raises(ValueError):
        packet.encoder(ss[1:], row_indices=[1])(x[1:])


def test_packet_rejects_stale_model_source_offsets_split_rows_truncation_and_bit_corruption(tmp_path):
    spec, x, v = sample()
    p = tmp_path / "values"
    h, size = write_packet(p, spec, v, byte_limit=12000)
    for b in [
        replace(spec, encoder_sha="c" * 64),
        replace(spec, source_sha="d" * 64),
        replace(spec, split="dev"),
        replace(spec, rows=spec.rows[::-1]),
        replace(spec, rows=(replace(spec.rows[0], start=99), spec.rows[1])),
    ]:
        with pytest.raises(ValueError):
            FrozenPacket.load(p, b, h, byte_limit=12000)
    with pytest.raises(ValueError):
        FrozenPacket.load(p, spec, "0" * 64, byte_limit=size)
    raw = p.read_bytes()
    for buf in [raw[:-1], raw + b"\x00", raw[:-1] + bytes([raw[-1] ^ 1])]:
        p.write_bytes(buf)
        with pytest.raises(ValueError):
            FrozenPacket.load(p, spec, h, byte_limit=size)


def test_packet_cannot_publish_partial_oversize_trainable_or_nonfinite_values(tmp_path):
    spec, _, v = sample()
    p = tmp_path / "values"
    bad = v.clone()
    bad[0, 2, 0] = torch.nan
    for x, cap in [
        (v, 24),
        (v.clone().requires_grad_(), 16384),
        (bad, 16384),
        (v.half(), 16384),
        (v[:, 1:], 16384),
    ]:
        with pytest.raises(ValueError):
            write_packet(p, spec, x, byte_limit=cap)
        assert not list(tmp_path.iterdir())
    h, size = write_packet(p, spec, v, byte_limit=16384)
    with pytest.raises(FileExistsError):
        write_packet(p, spec, v + 1, byte_limit=16384)
    with pytest.raises(ValueError):
        FrozenPacket.load(p, spec, h, byte_limit=size - 1)
    assert list(tmp_path.iterdir()) == [p] and sha256(p.read_bytes()).hexdigest() == h
    with pytest.raises(ValueError):
        replace(spec, rows=(spec.rows[0], spec.rows[0]))
    with pytest.raises(ValueError):
        replace(spec, rows=(replace(spec.rows[0], length=4), spec.rows[1]))


class OneByte:
    width = 4
    device = torch.device("cpu")
    max_context = 16

    def __init__(self):
        self.net = torch.nn.Module().eval()
        self.calls = []

    def spec(self, _):
        return {"sum": ((1,), torch.long)}

    def encode(self, store, slots, old, ids, prompt):
        assert ids.shape == (len(slots), 1) and all(v == old[0] for v in old)
        self.calls.append((prompt, old[0], ids.tolist()))
        result = (ids[:, 0] + store["sum"][slots, 0]) % 256
        store["sum"][slots, 0] = result
        return result[:, None].expand(-1, 4).float(), torch.zeros(len(slots), 256).scatter_(
            1, result[:, None], 2.0
        ).log_softmax(1)


def test_record_frontiers_consumes_only_actual_bytes_and_resets_at_capacity():
    from ndm.memory_program.host_cache import CachedByteHost
    from ndm.memory_program.stream_packet import record_frontiers

    x = OneByte()
    arena = CachedByteHost(x, 2, 16, 65536)
    spec, _, _ = sample()
    words = [b"\x00\x1e\xff", b"bar", b"foo"]
    spec = replace(spec, rows=spec.rows + (Window.observe(15, 20, words[2]),))
    out = record_frontiers(arena, spec, words, byte_limit=16384)
    assert arena.empty and all(not z.count_nonzero() for z in arena.slots.values())
    assert [
        (j == 0, j, [[w[j]] for w in block]) for block in (words[:2], words[2:]) for j in range(3)
    ] == x.calls
    expected = torch.tensor([list(w) for w in words]).cumsum(1) % 256
    assert (
        torch.equal(out[:, :, 0], expected.float())
        and torch.equal(out[:, :, 4:].argmax(-1), expected)
        and out.shape == (3, 3, 260)
        and not out.requires_grad
    )
    short = [w[:2] for w in words]
    s = replace(spec, rows=tuple(Window.observe(r.document, r.start, w) for r, w in zip(spec.rows, short)))
    other = record_frontiers(arena, s, short, byte_limit=16384)
    assert torch.equal(other, out[:, :2])


def test_invalid_frozen_stream_never_displaces_another_admission():
    from ndm.memory_program.host_cache import CachedByteHost
    from ndm.memory_program.stream_packet import record_frontiers

    x = OneByte()
    arena = CachedByteHost(x, 2, 16, 65536)
    spec, _, _ = sample()
    ws = [b"\x00\x1e\xff", b"bar"]
    t = arena.admit("other")
    with pytest.raises(ValueError):
        record_frontiers(arena, spec, ws, byte_limit=16384)
    assert not arena.empty and not x.calls
    arena.release([t])
    for words, cap in [(ws[::-1], 16384), (ws, 10)]:
        with pytest.raises(ValueError):
            record_frontiers(arena, spec, words, byte_limit=cap)
        assert arena.empty and not x.calls
