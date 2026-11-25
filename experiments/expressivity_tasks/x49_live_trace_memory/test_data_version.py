"""A v1 refusal must never turn into an implicit reduced-quota extraction."""

from copy import deepcopy
import json
import os
import pytest
from experiments.expressivity_tasks.x49_live_trace_memory import panels as p
from experiments.expressivity_tasks.x49_live_trace_memory.test_panels import metadata


def test_v2_is_explicit_and_only_changes_calibration(tmp_path):
    old = p.protocol_quotas("v1")
    new = p.protocol_quotas("v2")
    assert new == dict(old, calibration=110) and old == dict(
        train=1024, calibration=128, final_test=128
    )
    assert (
        p.data_dir(tmp_path, "v1") == tmp_path
        and p.data_dir(tmp_path, "v2") == tmp_path / "v2"
    )
    with pytest.raises(ValueError):
        p.data_dir(tmp_path, "auto")


def test_independent_member_and_window_witness():
    pairs, end = metadata()
    rows = [r for r, _ in pairs]
    excluded = {rows[0]["sha256"]}
    q = dict.fromkeys(p.N, 2)
    selected, counts = p.select(rows, excluded, q, end)
    p.verify_selection(rows, excluded, selected, q, counts)
    for field, value in [
        ("start", 17),
        ("sha256", rows[0]["sha256"]),
        ("doc", -1),
        ("length", 0),
    ]:
        changed = deepcopy(selected)
        changed["calibration"][0][field] = value
        with pytest.raises(ValueError):
            p.verify_selection(rows, excluded, changed, q, counts)
    with pytest.raises(ValueError):
        p.verify_selection(rows, excluded, selected, q, dict(counts, train=-1))


def test_freeze_and_extract_reads_only_verified_train_documents(tmp_path, monkeypatch):
    # A small fixture uses the real file/receipt path without reading Pile.
    monkeypatch.setattr(p, "MIN_OFFSET", 16)
    pairs, end = metadata()
    rows = [r for r, _ in pairs]
    q = dict.fromkeys(p.N, 2)
    panel, counts = p.select(rows, set(), q, end)
    blob = tmp_path / "fixture"
    with blob.open("xb") as f:
        for r, raw in pairs:
            f.seek(r["byte_offset"])
            f.write(raw)
    st = blob.stat()
    reg = dict(
        schema="x49-v2",
        amendment_sha=p.V2_SPEC,
        panels=panel,
        source=dict(
            path=str(blob), inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns
        ),
        eligible=counts,
        excluded_unique=0,
    )

    def fixture(protocol):
        assert protocol == "v2"
        return reg

    monkeypatch.setattr(p, "reserve_data", fixture)
    root = tmp_path / "out"
    note = p.freeze(root, protocol="v2")
    dst = root / "v2"
    assert note["payloads_opened"] == 0 and not (root / "reservation.json").exists()
    f = dst / "reservation.json"
    frozen = f.read_bytes()
    bad = deepcopy(reg)
    bad["panels"]["train"][0] = dict(reg["panels"]["final_test"][0], split="train")
    f.write_text(json.dumps(bad))
    reads = []
    pread = os.pread

    def measured(fd, n, off):
        reads.append((off, n))
        return pread(fd, n, off)

    monkeypatch.setattr(os, "pread", measured)
    with pytest.raises(ValueError, match="before source open"):
        p.extract_train(root, protocol="v2")
    assert not reads
    f.write_bytes(frozen)
    with pytest.raises(ValueError, match="already exists"):
        p.freeze(root, protocol="v2")
    ident = p.extract_train(root, protocol="v2")
    assert reads == [(r["offset"], r["length"]) for r in panel["train"]]
    assert (
        ident["schema"] == "x49-train-v2"
        and ident["targets"] == 2 * p.T
        and ident["evaluation_windows_read"] == 0
    )
    assert (
        ident["reservation_sha"] == note["reservation_sha"]
        and not (root / "train").exists()
    )
    with pytest.raises(ValueError, match="already exists"):
        p.extract_train(root, protocol="v2")
    from experiments.expressivity_tasks.x49_live_trace_memory import train_stream as ts

    monkeypatch.setattr(ts, "V2_QUOTAS", q)
    monkeypatch.setattr(ts, "V2_IDENTITY", p.sha(dst / "train/identity.json"))
    stream = ts.TrainStream(root)
    for seed in p.SEEDS:
        for k, i in enumerate(p.order(panel["train"], seed), 1):
            r = panel["train"][i]
            raw = dict((row["document"], w) for row, w in pairs)[r["doc"]]
            assert stream.sample(seed, k) == (r["doc"], p.window(raw, r))
        with pytest.raises(ValueError):
            stream.sample(seed, 3)
    target = dst / "train/windows.u8"
    w = target.read_bytes()
    target.write_bytes(w[:-1] + bytes([w[-1] ^ 1]))
    with pytest.raises(ValueError, match="windows.u8"):
        ts.TrainStream(root)
