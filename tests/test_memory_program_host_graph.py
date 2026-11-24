"""CPU allocation/identity checks only; actual graph replay is a leased CUDA gate."""

import pytest
import torch
from ndm.memory_program.host_cache import AbsoluteKV, Admission, CachedByteHost
from ndm.memory_program.host_graph import StepGraph
from experiments.expressivity_tasks.x43_fork_resolved.run import Control


def test_charged_private_rows_never_admissible_even_after_release():
    backend = AbsoluteKV(Control(2000324).eval())
    tmp = CachedByteHost(backend, 1, 16, 1 << 20)
    slot = tmp.tensor_bytes
    with pytest.raises(ValueError, match="budget"):
        CachedByteHost(backend, 2, 16, slot * 3 - 1, reserve=1)
    a = CachedByteHost(backend, 2, 16, slot * 3, reserve=1)
    assert a.empty and a.tensor_bytes == 3 * slot
    x = a.admit("x")
    a.admit("y")
    assert not a.empty
    with pytest.raises(ValueError, match="full"):
        a.admit("reserved")
    with pytest.raises(ValueError, match="foreign"):
        a.release([Admission(2, "fake", 100)])
    with torch.no_grad():
        for t in a.slots.values():
            t[2].fill_(7)
        a.prefill([x], [b"\xff"])
        a.release([x])
        x2 = a.admit("z")
        a.prefill([x2], [b"\x00"])
        a.clear()
        assert a.empty and all(t[2].eq(7).all() and not t[:2].any() for t in a.slots.values())


def test_cpu_graph_cannot_masquerade_as_accelerated_host():
    with pytest.raises(ValueError, match="requires CUDA"):
        StepGraph(Control(2000324).eval())
