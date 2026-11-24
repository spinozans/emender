from itertools import product
import pytest
import torch
from ndm.memory_program.resident_host import bounded_covers, execute_cover


def test_graph_covers_are_exact_shortest_and_bounded_not_greedy_padding():
    for widths in [(1, 4, 8, 16), (4, 5, 8), (1, 7, 8)]:
        for limit in (1, 2, 4, 6):
            got = bounded_covers(20, widths, limit)
            want = {}
            for d in range(1, limit + 1):
                for x in product(sorted(widths, reverse=True), repeat=d):
                    if sum(x) <= 20:
                        want.setdefault(sum(x), x)
            assert got == want
    assert bounded_covers(16, (1, 4, 8, 16))[15] == (8, 4, 1, 1, 1)
    assert bounded_covers(16, (1, 6, 8))[12] == (6, 6)
    assert 3 not in bounded_covers(16, (1, 4), 2)
    for parts in (0, 7, True):
        with pytest.raises(ValueError):
            bounded_covers(16, (1, 2), parts)


def test_graph_cover_visits_real_uneven_slots_once_and_reassembles_in_request_order():
    seen = []

    class G:
        def __init__(self, n):
            self.n = n

        def encode(self, a, s, old, ids, prompt):
            assert len(s) == self.n and not prompt
            for slot, t, b in zip(s.tolist(), old, ids[:, 0].tolist(), strict=True):
                seen.append((slot, t, b))
                a[slot] = t + b
            return a[s, None].clone(), ids.clone()

    slots = torch.tensor([6, 2, 8, 1, 4, 7, 3])
    old = [9, 19, 5, 8, 22, 70, 8]
    ids = torch.arange(7)[:, None]
    a = torch.full((10,), -1)
    pool = {n: G(n) for n in (1, 4, 8)}
    plan = bounded_covers(9, tuple(pool))[7]
    h, p = execute_cover(plan, pool, a, slots, old, ids)
    assert seen == list(zip(slots.tolist(), old, ids[:, 0].tolist()))
    assert h[:, 0].tolist() == [v + i for i, v in enumerate(old)] and torch.equal(p, ids)
    assert a[[0, 5, 9]].eq(-1).all()
    with pytest.raises(ValueError):
        execute_cover((8,), pool, a, slots, old, ids)
