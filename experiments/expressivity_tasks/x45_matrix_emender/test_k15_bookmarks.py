from pathlib import Path
import torch
from .k15_bookmarks import BookmarkPort, BookmarkSession


def test_recalled_bookmark_native_address_and_delayed_gradient():
    raw = bytes(range(80))
    ids = torch.tensor([list(raw)])
    h = torch.randn(1, len(raw), 32)
    from .k15_composition import ComposedReadMixer
    from .schedule import RoutingPolicy
    mixer = ComposedReadMixer(32, 2, 16, 8, 8, 8, RoutingPolicy())
    port = BookmarkPort(32, 16)
    s = BookmarkSession(Path.home()/'sxgc/xsa/target/debug/xsa')
    try:
        out, trace, _, writes = port(mixer, h, ids, s, [(0, 1), (1, 0), (0, 2), (1, 3)], True)
        assert [x[0] for x in trace[2]['calls']] == ['byte 16', f'locate {raw[4:17].hex()} 0 48 48', 'byte 17']
        assert torch.autograd.grad(out[:, 32:48].sum(), writes[0])[0].norm() > 0
        assert out[:, :16].count_nonzero() == 0
    finally:
        s.close()
