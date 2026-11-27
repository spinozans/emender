"""SAL-3 oracles: the transport-side binding must reproduce the SAL-1/2
count-law semantics exactly, from a synthetic consult row stream to the
driver's batch layout. Invented data only; natural memberships stay sealed."""

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from . import sal3
from .ladder import MAX_DEPTH
from .model import SalLM, SalPack


def _archive(seed=7, n=16, w=300):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, w), dtype=np.uint8)


def _docs(seed=3, n=4):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 4097), dtype=np.uint8)


def _flat_stats(archive, docs):
    ci = type("C", (), {})  # placeholder; CountIndex built below for real
    from .ladder import CountIndex

    ci = CountIndex(archive, depth=MAX_DEPTH, rowids=False)
    counts, hists, m_ex, h_ex = ci.stats(docs)
    expect = docs.shape[0] * 4096
    flat_counts = counts.reshape(expect, MAX_DEPTH + 1)
    flat_hists = hists.reshape(expect, MAX_DEPTH + 1, 256)
    dist = np.tile(np.arange(1, 4097), docs.shape[0])
    return ci, flat_counts, flat_hists, m_ex, h_ex, dist, expect


def test_pass_parser_and_telescope_match_stats(tmp_path):
    a = _archive()
    docs = _docs()
    ci, fc, fh, m_ex, h_ex, dist, expect = _flat_stats(a, docs)
    classes, hists = {}, {}
    for d in sal3.DEPTHS:
        path = tmp_path / f"rows{d}.bin"
        sal3.encode_pass(path, d, fc, fh, dist)
        c, h = sal3.pass_arrays(path, d, expect)
        assert np.array_equal(c, fc[:, d])
        assert np.array_equal(h, fh[:, d])
        classes[d], hists[d] = c, h
    m, h = sal3.telescope(classes, hists, ci.rows, ci.root, expect)
    assert np.array_equal(m, m_ex.reshape(expect, -1).astype(np.int64))
    assert np.array_equal(
        h.astype(np.float32), h_ex.reshape(expect, MAX_DEPTH + 1, 256)
    )


def test_pass_parser_drops_truncated_spines(tmp_path):
    a = _archive(seed=11)
    docs = _docs(seed=5)
    ci, fc, fh, m_ex, h_ex, dist, expect = _flat_stats(a, docs)
    for d in (2, 4, 8):
        path = tmp_path / f"t{d}.bin"
        sal3.encode_pass(path, d, fc, fh, dist)
        c, h = sal3.pass_arrays(path, d, expect)
        near = np.tile(np.arange(1, 4097), docs.shape[0]) < d
        assert (c[near] == 0).all() and (h[near] == 0).all()
        assert (c[~near] == fc[:, d][~near]).all()


def test_boundary_inexact_row_fails_closed(tmp_path):
    rows = bytearray(b"XSACONS1" + struct.pack("<II", 3, 0))
    # one exact row, then one conflated (flag bit 4) row
    rows += struct.pack("<IHHQQQQ", 1, 8, 0, 0, 0, 8, 0)
    rows += struct.pack("<BQ", 65, 8) + struct.pack("<QQQ", 0, 0, 8)
    rows += struct.pack("<IHHQQQQ", 1, 8, 16, 0, 0, 8, 0)
    rows += struct.pack("<BQ", 66, 8) + struct.pack("<QQQ", 0, 0, 8)
    path = tmp_path / "bad.bin"
    path.write_bytes(rows)
    with pytest.raises(RuntimeError, match="boundary-inexact"):
        sal3.pass_arrays(path, 8, 2)


def _batches_from(zs, n):
    out = []
    for i in range(n):
        out.append(
            dict(
                ids=torch.as_tensor(zs[f"b{i}_ids"], dtype=torch.long),
                targets=torch.as_tensor(zs[f"b{i}_targets"], dtype=torch.long),
                m=torch.as_tensor(zs[f"b{i}_m"], dtype=torch.float32),
                h=torch.as_tensor(zs[f"b{i}_h"]),
                local_succ=torch.as_tensor(zs[f"b{i}_local_succ"], dtype=torch.long),
            )
        )
    return out


def test_build_batches_match_dataset(tmp_path):
    a = _archive()
    docs = _docs()
    ci, fc, fh, m_ex, h_ex, dist, expect = _flat_stats(a, docs)
    classes = {d: fc[:, d] for d in sal3.DEPTHS}
    hists = {d: fh[:, d] for d in sal3.DEPTHS}
    m, h = sal3.telescope(classes, hists, ci.rows, ci.root, expect)
    old = dict(sal3.N3)
    try:
        sal3.N3 = {"fit": 4, "probe": 4}
        # SAL-3 binds true-document contexts (registered): window-start
        # targets carry full-depth statistics where SAL-1/2 truncated at
        # the window edge, so m/h compare against the doc-level telescope
        # at the mapped indices, while ids/targets/local_succ must equal
        # the driver's window layout exactly.
        for role, shuffle in (("probe", False), ("fit", True)):
            zs = sal3.build_batches(role, docs, m, h)
            n = len(zs) // 5
            pack = SalPack(a, SalLM(99), batch=16, bank=False)
            want = pack.dataset(docs, shuffle=shuffle)
            assert n == len(want)
            got = _batches_from(zs, n)
            order = (
                np.arange(256)
                if not shuffle
                else np.random.default_rng(5750).permutation(256)
            )
            for bi, (g, w) in enumerate(zip(got, want)):
                assert torch.equal(g["ids"], w["ids"])
                assert torch.equal(g["targets"], w["targets"])
                assert torch.equal(g["local_succ"], w["local_succ"])
                sel = order[bi * 16 : bi * 16 + 16]
                base = (sel // 64) * 4096 + (sel % 64) * 64
                idx = np.repeat(base, 64) + np.tile(np.arange(64), 16)
                assert np.array_equal(g["m"].numpy(), m[idx].astype(np.float32))
                assert np.array_equal(g["h"].numpy(), h[idx])
    finally:
        sal3.N3 = old


def test_run_loader_round_trip(tmp_path, monkeypatch):
    from . import run
    from ..window import put as wput
    from .settings import sha as wsha

    a = _archive()
    docs = _docs()
    ci, fc, fh, m_ex, h_ex, dist, expect = _flat_stats(a, docs)
    classes = {d: fc[:, d] for d in sal3.DEPTHS}
    hists = {d: fh[:, d] for d in sal3.DEPTHS}
    m, h = sal3.telescope(classes, hists, ci.rows, ci.root, expect)
    old = dict(sal3.N3)
    try:
        sal3.N3 = {"probe": 4}
        zs = sal3.build_batches("probe", docs, m, h)
        zpath = tmp_path / "bound-probe.npz"
        np.savez(zpath, **zs)
        wput(tmp_path / "rows.json", dict(placeholder=1))
        wput(
            tmp_path / "bound-probe.json",
            dict(
                batches=len(zs) // 5,
                batches_sha=wsha(zpath),
                rows_sha=wsha(tmp_path / "rows.json"),
                targets=expect,
            ),
        )
        monkeypatch.setattr(run, "DATA", tmp_path)
        monkeypatch.setattr(run, "JOB", tmp_path)
        monkeypatch.setattr(run, "load", lambda role: docs)
        pack = SalPack(a, SalLM(98), batch=16, bank=False)
        want = pack.dataset(docs, shuffle=False)
        _, batches, setup = run.bind_all(pack, "probe", False)
        assert setup["bound_from_index"] is True
        assert len(batches) == len(want)
        for bi, (g, w) in enumerate(zip(batches, want)):
            for k in (
                "ids",
                "targets",
                "local_succ",
                "local_rows",
                "pair_owner",
                "pair_row",
                "spans",
            ):
                assert torch.equal(g[k], w[k]), k
            assert g["rows"] == w["rows"] and g["lengths"] == w["lengths"]
            assert g["core"].shape[1] == 0 and not g["core_valid"].any()
            idx = np.repeat(np.arange(bi * 16, bi * 16 + 16) * 64, 64) + np.tile(
                np.arange(64), 16
            )
            assert np.array_equal(g["m"].numpy(), m[idx].astype(np.float32))
            assert np.array_equal(g["h"].numpy(), h[idx])
    finally:
        sal3.N3 = old


def test_sal3_wiring():
    sal3.rebind()
    from . import data as d, gate as g, run as r

    assert sal3.sal_data is d and sal3.sal_gate is g and sal3.sal_run is r
    assert g.READY_KEYS == ("rows", "fit", "container")
    assert g.ARMS == ("local", "xsacount") and g.CAMPAIGN_CAP == 6000
    assert sal3.sal_run.PROBE_ENTRY.endswith(".sal3")
    assert sal3.sal_run.BIND_ENTRY[0].endswith(".sal3")
    assert callable(sal3.sal_run.DICT_BYTES)
    assert sal3.sal_run.N == sal3.N3
    here = Path(sal3.__file__).parent
    proto = (here / "PROTOCOL-SAL3.md").read_text()
    assert "**6,000 card-seconds**" in proto
    assert "15419c427b5bdc9fb63548901ca6cf83d54f486da890f66faa55021f0a0dc053" in proto
    assert sal3.CONTAINER_SHA == (
        "15419c427b5bdc9fb63548901ca6cf83d54f486da890f66faa55021f0a0dc053"
    )
    assert sal3.N3 == {"fit": 512, "probe": 128}
    assert sal3.DEPTHS == tuple(range(1, 9))
    run_src = (here / "run.py").read_text()
    assert 'f"{arms[1]}:{arms[0]}"' in run_src
    script = (here / "leased3.sh").read_text()
    assert "output/retrieved-span-effects/sal-3/device" in script
    assert '-m "$mod.sal3" prelease' in script
    assert (here / "sal3run.py").exists()
    gate_src = (here / "gate.py").read_text()
    assert '"sal3": "leased3.sh"' in gate_src
    # the exact-depth-8 class needs the ninth pass
    assert sal3.LAW_DEPTH == 8 and max(sal3.DEPTHS) == 8


def test_archiveless_pack_refusals():
    with pytest.raises(ValueError, match="archive-less"):
        SalPack(None, SalLM(1), batch=4, bank=True, rows=128)
    with pytest.raises(ValueError, match="archive-less"):
        SalPack(None, SalLM(2), batch=4, bank=False)
    p = SalPack(None, SalLM(3), batch=4, bank=False, rows=128)
    assert p.rows == 128 and p.counts is None and p.index is None
