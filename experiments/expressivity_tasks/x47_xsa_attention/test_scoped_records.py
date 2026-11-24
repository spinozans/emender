"""Record teardown/authority gates. F6 text and positional worker are real; small m20 fixture refuses unmaterialized edges."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
import json
import os
from weakref import ref

import numpy as np
import pytest
import torch
from ndm.memory_program.branch_domain import BranchDomain
from ndm.memory_program.runtime import Unavailable
from . import scoped_records as lib
from .b1_mount import ArchiveDescriptor, HERE, ROOT
from .workbench_native import EXE

pytestmark = pytest.mark.skipif(not EXE.exists(), reason="local registered positional binary")


class Pages:
    root = nodes = 1

    def __init__(self, desc):
        self.desc = desc
        self.closes = 0

    def rows(self, ids):
        assert tuple(ids) == (1,)
        return [dict(h=np.ones(257, np.uint64), child=np.zeros(256, np.uint32))]

    @contextmanager
    def open(self):
        try:
            yield self, lambda: None
        finally:
            self.closes += 1


def mount():
    c = json.loads((HERE / "b1_mounts.json").read_text())["F6"]
    d = ArchiveDescriptor("F6", c, ROOT)
    p = Pages(d)
    return d, c["files"][1], p, BranchDomain(d.name, d.version, p.open, scope="whole", max_grants=2)


class Encoder:
    def __call__(self, texts):
        return [torch.tensor([sum(v)], dtype=torch.float32) for v in texts]


def test_real_positive_reads_and_retained_endpoints_do_not_own_backing():
    def issued():
        d, receipt, p, domain = mount()
        enc = Encoder()
        weak = list(map(ref, (d, d.starts, enc)))
        with domain, domain.borrow(d.name, d.version, scope="whole") as g:
            with lib.ScopedRecords(d, receipt, branches=g, scope="whole", encode=enc) as state:
                rs = state.readers()
                key = d.text.read_bytes()[:8]
                loc, content, following = rs
                hits = loc.read_many((key,), 8)[0]
                assert hits and not isinstance(hits, Unavailable)
                spans = tuple(x.keys[0] for x in hits)
                xs = content.read_many(spans, 8)
                ys = following.read_many(spans, 8)
                assert all(v[0].keys == (key,) and v[0].value.item() == sum(key) for v in xs)
                assert any(y and y[0].keys[0] == y[0].keys[1][:1] for y in ys)
                wrong = replace(spans[0], version="unknown")
                assert isinstance(content.read_many((wrong,), 8)[0], Unavailable)
            assert state.final["rc"] == 0 and state.final["fd_closed"] and state.closed
        assert p.closes == 1
        return weak, rs

    weak, rs = issued()
    # No gc.collect here. The small handles are intentionally still strongly reachable.
    assert all(w() is None for w in weak)
    for op in rs:
        with pytest.raises(ValueError, match="retired"):
            op.read_many((), 8)


@pytest.mark.parametrize("fault", ["scope", "excluded", "copy", "expired"])
def test_wrong_authority_rejected_before_companion_opens_anything(monkeypatch, fault):
    d, r, p, domain = mount()
    monkeypatch.setattr(lib, "FilePins", lambda *a: pytest.fail("opened before authority"))
    with domain:
        with domain.borrow(d.name, d.version, scope="whole") as g:
            if fault != "expired":
                with pytest.raises((ValueError, PermissionError)):
                    lib.ScopedRecords(
                        copy(d) if fault == "copy" else d,
                        r,
                        branches=g,
                        scope="part" if fault == "scope" else "whole",
                        excluded_docs=(1,) if fault == "excluded" else (),
                    )
        if fault == "expired":
            with pytest.raises(ValueError, match="expired"):
                lib.ScopedRecords(d, r, branches=g, scope="whole")
    assert p.closes == 1


def test_one_companion_per_admission_wrong_thread_cannot_read_or_revoke():
    d, r, _, domain = mount()
    with domain, domain.borrow(d.name, d.version, scope="whole") as g:
        with lib.ScopedRecords(d, r, scope="whole", branches=g) as scope:
            endpoint = scope.readers()[1]
            with ThreadPoolExecutor(1) as th:
                for method, args in [(endpoint.read_many, ((), 0)), (scope.close, ())]:
                    with pytest.raises(ValueError, match="foreign"):
                        th.submit(method, *args).result()
            assert endpoint.read_many((), 0) == []
            with pytest.raises(ValueError, match="companion unavailable"):
                lib.ScopedRecords(d, r, scope="whole", branches=g)
        with pytest.raises(ValueError, match="companion unavailable"):
            lib.ScopedRecords(d, r, scope="whole", branches=g)


def test_parent_return_revokes_forgotten_child_before_next_same_version_grant():
    d, r, _, domain = mount()
    with domain:
        with domain.borrow(d.name, d.version, scope="whole") as g:
            a = lib.ScopedRecords(d, r, scope="whole", branches=g)
            old = a.readers()[2]
        assert a.closed and a.final["rc"] == 0 and a.final["fd_closed"]
        with (
            domain.borrow(d.name, d.version, scope="whole") as g,
            lib.ScopedRecords(d, r, scope="whole", branches=g) as b,
        ):
            with pytest.raises(ValueError, match="retired"):
                old.read_many((b"foreign",), 8)
            assert b.pid != a.pid and b.readers()[2].read_many((), 0) == []
    a.close()


@pytest.mark.parametrize("fault", ["dead", "reply"])
def test_ambiguous_position_or_dead_worker_expires_even_cached_or_empty_reads(monkeypatch, fault):
    d, r, _, domain = mount()
    if fault == "reply":
        monkeypatch.setattr(lib.ArchiveRecords, "locate", lambda *xs: [])  # incomplete one-key response
    with domain, domain.borrow(d.name, d.version, scope="whole") as g:
        with lib.ScopedRecords(d, r, scope="whole", branches=g) as state:
            rs = state.readers()
            if fault == "dead":
                os.kill(state.pid, 9)
                os.waitid(os.P_PID, state.pid, os.WEXITED | os.WNOWAIT)
            with pytest.raises(ValueError):
                rs[0].read_many((b"x",), 8)
            assert (
                state.closed
                and state.final["fd_closed"]
                and state.final["rc"] == (-9 if fault == "dead" else 0)
            )
            with pytest.raises(ValueError, match="retired"):
                rs[1].read_many((), 0)


def test_failed_final_verification_invalidates_and_reaps_both_owners(monkeypatch):
    d, r, p, domain = mount()
    with pytest.raises(ValueError, match="final identity changed"):
        with domain, domain.borrow(d.name, d.version, scope="whole") as g:
            child = lib.ScopedRecords(d, r, scope="whole", branches=g)
            rs = child.readers()

            def bad(_):
                raise ValueError("final identity changed")

            monkeypatch.setattr(lib.ArchiveRecords, "verify", bad)
            # Leave subordinate open: parent expiry is responsible even when final verify raises.
    assert (
        domain.closed
        and p.closes == 1
        and child.closed
        and child.final["rc"] == 0
        and child.final["fd_closed"]
    )
    with pytest.raises(ValueError, match="retired"):
        rs[0].read_many((), 8)
