"""SAL-2 oracles: archive scan, count-table cache, prebuilt-counts parity,
campaign wiring. Invented data only; natural memberships stay sealed."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from . import sal2
from .ladder import CountIndex
from .model import SalLM, SalPack


def _archive(seed=7, n=12, w=200):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, w), dtype=np.uint8)


def _big_archive(seed=11, n=6, w=400):
    return _archive(seed=seed, n=n, w=w)


def _live(seed=3, n=4):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 65), dtype=np.uint8)


def test_cache_roundtrip_stats_parity(tmp_path):
    a = _archive()
    x = _live()
    full = CountIndex(a, depth=4, rowids=True)
    ci = CountIndex(a, depth=4, rowids=False)
    ci.save(tmp_path / "cache", meta=dict(archive_sha="deadbeef01"))
    loaded = CountIndex.load(tmp_path / "cache", meta=dict(archive_sha="deadbeef01"))
    for got in (ci, loaded):
        c1, h1, m1, hh1 = full.stats(x)
        c2, h2, m2, hh2 = got.stats(x)
        assert np.array_equal(c1, c2)
        assert np.array_equal(h1, h2)
        assert np.array_equal(m1, m2)
        assert np.array_equal(hh1, hh2)
    with pytest.raises(RuntimeError):
        loaded.matched_rows(0, 10, 2)


def test_cache_binding_refusals(tmp_path):
    a = _archive()
    ci = CountIndex(a, depth=3, rowids=False)
    ci.save(tmp_path / "cache", meta=dict(archive_sha="aa"))
    with pytest.raises(ValueError, match="binding mismatch"):
        CountIndex.load(tmp_path / "cache", meta=dict(archive_sha="bb"))
    with pytest.raises(RuntimeError, match="not empty"):
        ci.save(tmp_path / "cache", meta=dict(archive_sha="aa"))


def test_salpack_prebuilt_counts_parity(tmp_path):
    from .ladder import MAX_DEPTH

    a = _big_archive()
    x = _live()
    ci = CountIndex(a, depth=MAX_DEPTH, rowids=False)
    ci.save(tmp_path / "cache", meta=dict(archive_sha="k1"))
    loaded = CountIndex.load(tmp_path / "cache", meta=dict(archive_sha="k1"))
    torch.manual_seed(0)
    m1 = SalLM(101)
    torch.manual_seed(0)
    m2 = SalLM(101)
    pack1 = SalPack(a, m1, batch=4, depth=MAX_DEPTH, bank=False)
    pack2 = SalPack(a, m2, batch=4, depth=MAX_DEPTH, bank=False, counts=loaded)
    b1 = pack1.bind(x)
    b2 = pack2.bind(x)
    assert b1["core"].shape[1] == 0 and not b1["core_valid"].any()
    for arm in ("local", "skeleton"):
        out1 = m1(b1, arm)
        out2 = m2(b2, arm)
        assert torch.allclose(out1, out2, atol=0, rtol=0)
    # geometry refusals: wrong depth / bank needs row ids
    with pytest.raises(RuntimeError, match="geometry"):
        SalPack(
            a,
            SalLM(102),
            batch=4,
            depth=MAX_DEPTH,
            bank=False,
            counts=CountIndex(a, depth=MAX_DEPTH - 1, rowids=False),
        )
    # bank needs row ids: a complete-tree geometry archive still refuses a
    # rowids=False cache rather than silently building an unusable bank
    a2 = _archive(seed=13, n=4, w=129)
    ci2 = CountIndex(a2, depth=MAX_DEPTH, rowids=False)
    ci2.save(tmp_path / "cache2", meta=dict(archive_sha="k2"))
    loaded2 = CountIndex.load(tmp_path / "cache2", meta=dict(archive_sha="k2"))
    with pytest.raises(RuntimeError, match="geometry"):
        SalPack(a2, SalLM(103), batch=4, depth=MAX_DEPTH, bank=True, counts=loaded2)


def _write_pile(tmp_path, docs):
    content = b"\x1e".join(docs) + b"\x1e"
    p = tmp_path / "pile"
    p.write_bytes(content)
    st = os.stat(p)
    return dict(path=str(p), inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns)


def _expected_rows(docs):
    rows, seen = [], set()
    pos = 0
    for raw in docs:
        n = len(raw)
        if n > 1:
            h = hashlib.sha256(raw).hexdigest()
            if int(h, 16) % 100 >= 2 and n >= sal2.ARCHIVE_MIN_LEN and h not in seen:
                seen.add(h)
                j = int(h[16:32], 16) % ((n - 1) // sal2.WINDOW_T)
                rows.append(
                    dict(
                        doc=pos,
                        offset=pos,
                        length=n,
                        split="train",
                        sha256=h,
                        start=j * sal2.WINDOW_T,
                    )
                )
        pos += n + 1
    return rows


def test_archive_rows_selects_trains_in_order(tmp_path):
    rng = np.random.default_rng(5)
    docs = [bytes(rng.integers(65, 91, 4200, dtype=np.uint8)) for _ in range(6)]
    docs.append(docs[2])  # duplicate content: must be skipped
    docs.insert(0, b"tiny")  # below minimum length
    source = _write_pile(tmp_path, docs)
    want = _expected_rows(docs)
    need_n = len(want) - 1
    rows, scan = sal2.archive_rows(source, count=need_n)
    assert rows == want[:need_n]
    assert scan["distinct"] == need_n
    assert scan["region_end"] <= source["size"]
    for r in rows:
        assert r["start"] % 4096 == 0 and r["start"] + 4096 <= r["length"]
    # requesting more than the inventory refuses
    with pytest.raises(RuntimeError, match="insufficient"):
        sal2.archive_rows(source, count=len(want) + 1)
    # a changed source refuses
    with open(source["path"], "ab") as f:
        f.write(b"\x1e" + b"y" * 5000)
    with pytest.raises(RuntimeError, match="source changed"):
        sal2.archive_rows(source, count=1)


def test_archive_rows_cap(tmp_path):
    rng = np.random.default_rng(9)
    docs = [bytes(rng.integers(65, 91, 4200, dtype=np.uint8)) for _ in range(3)]
    source = _write_pile(tmp_path, docs)
    with pytest.raises(RuntimeError, match="cap overrun"):
        sal2.archive_rows(source, count=3, cap=100)


def test_sal2_wiring():
    sal2.rebind()
    from . import data as d, gate as g, run as r

    assert sal2.sal_data is d and sal2.sal_gate is g and sal2.sal_run is r
    assert d.DATA == sal2.DATA2 and g.JOB == sal2.JOB2
    assert g.N == sal2.N2 and g.CAMPAIGN_CAP == sal2.CAP2
    assert g.ARMS == ("local", "skeleton")
    assert g.BASE == sal2.B2
    assert g.qualified is sal2.qualified
    here = Path(sal2.__file__).parent
    script = (here / "leased2.sh").read_text()
    assert "output/retrieved-span-effects/sal-2/device" in script
    assert '-m "$mod.sal2" prelease' in script
    assert 'modname=$(cat "$d/module")' in script
    assert (here / "sal2run.py").exists()
    gate_src = (here / "gate.py").read_text()
    assert '"sal2run"' in gate_src and "leased2.sh" in gate_src
    run_src = (here / "run.py").read_text()
    assert "counts-cache" in run_src and "archive_sha" in run_src
    # the archive geometry is the registered 512:1 design point
    assert sal2.N2["build"] * 4096 == sal2.N2["fit"] * 4096 * 512
    assert sal2.N2["build"] * (4097 - 1) == 2**30
