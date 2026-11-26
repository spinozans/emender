"""Pure arithmetic and invented geometry, no broker access or actual panel files."""

import pytest
import torch

from .device_gate import capacity
from .device_meter import invented


def test_gate_requires_every_closed_charge_and_both_remaining_budgets():
    end = 1795712280
    s = dict(
        deadline_epoch=end,
        aggregate_card_seconds_upper=43200,
        max_cards_at_once=1,
        visible_allowlist="0,1",
        remote=False,
        auto_retry=False,
    )
    c = {"meter": 29, "natural-meter": 39, "panel": 40000}
    assert capacity(s, end - 190, c) == 180 and capacity(s, end - 129, c) == 129
    assert capacity(s, end - 250, dict(c, panel=43000)) == 132
    assert capacity(s, end - 250, dict(c, meter=725)) == 136
    for cs, now in (
        ({"meter": 1}, end - 500),
        (dict(c, panel=43136), end - 500),
        (dict(c, meter=780), end - 500),
        (c, end - 100),
        (dict(c, panel=0), end - 500),
    ):
        with pytest.raises(RuntimeError):
            capacity(s, now, cs)
    with pytest.raises(RuntimeError):
        capacity(dict(s, max_cards_at_once=2), end - 500, c)


def test_actual_meter_packet_small_cpu_smoke():
    m, a, req, y, _ = invented(3, 16, 7, "cpu")
    b = a.batch(req)
    opt = torch.optim.AdamW(m.parameters())
    z = b.predict(m, mode="indexed")
    loss = -z.probabilities[range(len(y)), y].log().mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss) and a.layout.rows == 48
