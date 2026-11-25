"""No corpus access: JT native calibration rank, duplicate choice, minimal crossing and spent LS membership."""

from copy import deepcopy
from hashlib import sha256
import pytest
from experiments.expressivity_tasks.x48_muon_continuation.data import write_json, sha
from experiments.expressivity_tasks.x49_live_trace_memory import (
    transfer_data as data,
    short_phase,
)


def record(i, off, n, split="calibration"):
    return dict(
        document=off,
        byte_offset=off,
        byte_length=n,
        split=split,
        sha256=f"{i * 100 + {'calibration': 0, 'final_test': 1, 'train': 2}[split]:064x}",
    )


def test_only_whole_unspent_calibration_rank_dedup_and_crossing(monkeypatch):
    records = [record(i, 5000 + i * 8192, 31 + i) for i in range(18)]
    early = records[3]
    records.extend(
        [
            dict(early, document=149000, byte_offset=149000),
            record(40, 3, 100),
            record(50, 180000, 2, "final_test"),
            record(51, 181000, 4096, "train"),
            record(52, 191000, 4097),
            record(53, 202000, 1),
        ]
    )
    ex = {records[7]["sha256"]}
    goal = 133
    lo = 4000
    size = 300000
    clean = sorted(
        [x for i, x in enumerate(records[:18]) if i != 7],
        key=lambda r: sha256(
            ("x49-jt-short-cal-v1:" + r["sha256"]).encode()
        ).hexdigest(),
    )
    rows, n, nt = data.choose(records[::-1], ex, size, goal, lo)
    assert (
        n == 17
        and nt == sum(r["byte_length"] - 1 for r in clean)
        and [x["offset"] for x in rows]
        == [x["byte_offset"] for x in clean[: len(rows)]]
    )
    monkeypatch.setattr(data, "MIN_OFFSET", lo)
    monkeypatch.setattr(data, "GOAL", goal)
    book = dict(
        spec=data.SPEC,
        source=dict(size=size),
        rows=rows,
        eligible_documents=n,
        eligible_targets=nt,
    )
    got, total = data.checked_rows(book)
    assert got == rows and total >= goal > total - rows[-1]["length"] + 1
    # Include the duplicate in the full supply; traversal order must not choose the later descriptor.
    full = data.choose(records[::-1], ex, size, nt, lo)[0]
    assert (
        next(x for x in full if x["sha256"] == early["sha256"])["offset"]
        == early["byte_offset"]
    )
    with pytest.raises(ValueError, match="insufficient"):
        data.choose(records, ex, size, nt + 1, lo)
    with pytest.raises(ValueError, match="duplicate length"):
        data.choose(records + [dict(early, byte_length=99)], ex, size, goal, lo)
    bad = deepcopy(book)
    bad["rows"].reverse()
    with pytest.raises(ValueError, match="rank crossing"):
        data.checked_rows(bad)
    bad = deepcopy(book)
    bad["rows"][0]["split"] = "final_test"
    with pytest.raises(ValueError, match="calibration membership"):
        data.checked_rows(bad)
    with pytest.raises(ValueError, match="unknown frozen"):
        short_phase.study("another")


def test_selection_rederives_parent_and_excludes_every_ls_role(tmp_path, monkeypatch):
    ls = data.ls
    pre = {f"{101 + i * 100:064x}" for i in range(42555)}
    used = [
        dict(
            doc=i,
            offset=i,
            length=4097,
            split="train",
            start=0,
            sha256=f"{102 + i * 100:064x}",
        )
        for i in range(384)
    ]
    records = [record(i, 5000 + i * 100, 40 + i) for i in range(9)]
    base = dict(
        source=dict(size=100000),
        manifest_sha="fixture",
        roles=dict(fit=used[:256], probe=used[256:]),
        spec=ls.SPEC,
        schema="x49-ls-native-train",
    )
    f = tmp_path / "reservation.json"
    write_json(f, base)
    meta = {"rows_sha": sha(f)}
    monkeypatch.setattr(data, "parent", lambda: {"data": meta})
    monkeypatch.setattr(ls, "BASE", tmp_path)
    monkeypatch.setattr(ls, "MIN_OFFSET", 0)
    monkeypatch.setattr(data, "MIN_OFFSET", 0)
    monkeypatch.setattr(data, "GOAL", 100)
    monkeypatch.setattr(ls, "metadata", lambda: meta)
    # Real LS role validator also enforces SHA rank for BOTH roles. Sort our fake TRAIN rows accordingly.
    used.sort(key=lambda r: sha256(("x49-live-v1:" + r["sha256"]).encode()).hexdigest())
    base["roles"] = dict(fit=used[:256], probe=used[256:])
    write_json(f, base)
    meta["rows_sha"] = sha(f)
    monkeypatch.setattr(ls, "selection_context", lambda: (base, records, pre))
    book = data.selection()
    assert (
        book["excluded_unique"] == 42939
        and book["excluded_sha"]
        == data.fingerprint(sorted(pre | {r["sha256"] for r in used}))
        and book["ls_rows_sha"] == meta["rows_sha"]
        and data.checked_rows(book)[1] >= 100
    )
    bad = deepcopy(base)
    bad["roles"]["probe"][0] = bad["roles"]["fit"][0]
    monkeypatch.setattr(ls, "selection_context", lambda: (bad, records, pre))
    with pytest.raises(ValueError, match="spent LS cohorts"):
        data.selection()
    write_json(f, bad)
    meta["rows_sha"] = sha(f)
    with pytest.raises(ValueError, match="overlapping FIT/PROBE"):
        data.selection()
