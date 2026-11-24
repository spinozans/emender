import pytest
import torch
from ndm.memory_program.serving import place_inputs


def test_same_device_host_packet_is_identity_and_preserves_gradients():
    h = torch.randn(2, 6, requires_grad=True)
    p = torch.randn(2, 256, requires_grad=True)
    x, y = place_inputs(h, p, width=6, device="cpu")
    assert x is h and y is p
    (x.sum() + 2 * y.sum()).backward()
    assert h.grad.eq(1).all() and p.grad.eq(2).all()


def test_host_packet_refuses_extra_rows_mixed_precision_and_unregistered_direction():
    h = torch.randn(2, 6)
    p = torch.randn(2, 256)
    for x, y in [(h.double(), p), (h, p[:1]), (h[:, :3], p), (h, p[:, :4]), (h[:0], p[:0])]:
        with pytest.raises(ValueError, match="packet"):
            place_inputs(x, y, width=6, device="cpu")
    with pytest.raises(ValueError, match="placement"):
        place_inputs(h, p, width=6, device="meta")
