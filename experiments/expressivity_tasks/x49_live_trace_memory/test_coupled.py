"""Literal host/head gradient and optimizer replay on full-length synthetic document windows."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from ndm.memory_program.direct import DirectTrust
from experiments.expressivity_tasks.x48_muon_continuation.learning import hybrid
from experiments.expressivity_tasks.x48_muon_continuation.train import save, restore
from experiments.expressivity_tasks.x49_live_trace_memory.coupled import (
    PEAK,
    head_groups,
    step,
)
from experiments.expressivity_tasks.x49_live_trace_memory.values import LiveValueHead
from experiments.expressivity_tasks.x49_live_trace_memory.trace import TraceHead


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 8)
        self.out = nn.Linear(8, 256)

    def forward(self, x):
        h = (self.emb(x) * 0.01).cumsum(1).tanh()
        return torch.cat([h, self.out(h)], -1)


def fresh(arm):
    m = Tiny().train()
    c = LiveValueHead(DirectTrust(8)).train()
    if arm in ("F", "C"):
        c = TraceHead(c, learned=arm == "C")
    return (
        m,
        c,
        (
            hybrid(([m.emb.weight, m.out.weight], [m.out.bias], [], []), PEAK[0])[0],
            hybrid(head_groups(c), PEAK[1])[0],
        ),
    )


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


@pytest.mark.parametrize("arm", ["A", "B", "F", "C"])
def test_two_coupled_full_window_updates_and_exact_checkpoint_successor(tmp_path, arm):
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4901)
            m, c, opts = fresh(arm)
            m0 = deepcopy(m.state_dict())
            c0 = deepcopy(c.state_dict())
            bind = dict(
                schema="x49-fixture-v1",
                arm=arm,
                seed=4901,
                source="synthetic",
                horizon=4096,
            )
            w1 = (b"\xffAkey:key_1\0\x1e" * 350)[:4097]
            w2 = (b"xkey_2 key:key_2=" * 300)[:4097]
            assert len(w1) == len(w2) == 4097
            kw = dict(mode=arm, seed=4901, width=8)
            r = step(m, c, opts, w1, document=23, index=1, **kw)
            assert r["work"]["private_writes"] == (4095 if arm in ("F", "C") else 0)
            if arm in ("F", "C"):
                assert torch.equal(c0["ask.weight"], c.ask.weight) == (arm == "F")
                assert torch.equal(c0["filing.weight"], c.filing.weight) == (arm == "F")
            assert r["targets"] == 4096
            assert not equal(m0, m.state_dict()) and not equal(c0, c.state_dict())
            ck = save(tmp_path, bind, 1, m, c, opts)
            r2 = step(m, c, opts, w2, document=24, index=2, **kw)
            m2, c2, o2 = fresh(arm)
            with pytest.raises(ValueError, match="foreign"):
                restore(tmp_path, dict(bind, seed=4902), 1, m2, c2, o2)
            assert restore(tmp_path, bind, 1, m2, c2, o2) == ck["sha256"]
            assert step(m2, c2, o2, w2, document=24, index=2, **kw) == r2
            for a, b in zip((m, c, *opts), (m2, c2, *o2), strict=True):
                assert equal(a.state_dict(), b.state_dict())
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
