from collections import Counter
from types import SimpleNamespace as N
import pytest
from ndm.memory_program.resident_host import validate_buckets, _Buckets


def test_bucket_admission_refuses_unbounded_and_ghost_widths():
    assert validate_buckets(16, [16, 1], 64) == (1, 16)
    for widths in ([], [0], [True], [17], [1, 1], [1, 2, 3, 4, 5], iter([1])):
        with pytest.raises(ValueError):
            validate_buckets(16, widths, 64)
    with pytest.raises(ValueError):
        validate_buckets(16, [1], 0)


def test_private_budget_failure_closes_every_unpublished_graph(monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kw: object())

    def graph(n):
        g = N(prepared=False, setup={"private_pool_reserved": n})

        def build(*args, **kwargs):
            g.prepared = True
            calls.append(("build", kwargs["warm_stream"]))

        def close():
            g.prepared = False
            calls.append(("close", n))

        g.build = build
        g.close = close
        return g

    fake = N(ready=False, device="cpu", buckets={1: graph(3), 16: graph(7)})
    with pytest.raises(ValueError, match="pool budget"):
        _Buckets.build(fake, N(empty=True), 9)
    assert (
        not fake.ready
        and not fake.buckets
        and calls[0][1] is calls[1][1]
        and calls[-2:] == [("close", 3), ("close", 7)]
    )


def test_retire_every_width_not_just_the_last_used():
    seen = []
    fake = N(work=Counter(), buckets={b: N(retire=lambda b=b: seen.append(b)) for b in (1, 16)})
    _Buckets.retire(fake)
    assert seen == [1, 16] and fake.work["retirements"] == 1
