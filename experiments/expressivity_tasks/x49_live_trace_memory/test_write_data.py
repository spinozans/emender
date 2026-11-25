"""Synthetic-only W membership, real-byte shift/identity and short-window negative cases."""

from hashlib import sha256
import pytest
from experiments.expressivity_tasks.x48_muon_continuation.data import load, sha
from experiments.expressivity_tasks.x49_live_trace_memory import write_data as d


def test_short_rank_is_not_length_or_offset_selection_or_long_test_reuse():
    vals = []
    off = 0
    for idx, n in enumerate(
        [2, 17, 129, 4096, 4097, 2000, 42, 512, 4000, 16, 1401, 310]
    ):
        vals.append(
            dict(
                document=off,
                byte_offset=off,
                byte_length=n,
                split="final_test",
                sha256=f"{100 * idx + 1:064x}",
            )
        )
        off += n + 1
    duplicate = dict(vals[3], document=off, byte_offset=off)
    ex = {vals[1]["sha256"], vals[8]["sha256"]}
    v, n, nt = d.choose_short(vals + [duplicate], ex, 3000, min_offset=0)
    rank = sorted(
        [i for i in vals if i["sha256"] not in ex and i["byte_length"] <= 4096],
        key=lambda r: sha256(b"x49-w-short-v1:" + r["sha256"].encode()).digest(),
    )
    assert (
        [x["doc"] for x in v] == [i["document"] for i in rank[: len(v)]]
        and n == len(rank)
        and nt == sum(i["byte_length"] - 1 for i in rank)
        and sum(x["length"] - 1 for x in v[:-1])
        < 3000
        <= sum(x["length"] - 1 for x in v)
    )
    assert all(x["start"] == 0 for x in v)
    with pytest.raises(ValueError, match="insufficient"):
        d.choose_short(vals, ex, 10**8, min_offset=0)
    with pytest.raises(ValueError, match="malformed"):
        d.choose_short([dict(vals[0], sha256="0" * 64)], set(), 1, min_offset=0)


def test_train_extraction_checks_all_raw_shifts_no_short_payload_or_duplicate(
    tmp_path, monkeypatch
):
    folder = tmp_path / "reg"
    source = tmp_path / "toy"
    rows = []
    with source.open("wb") as out:
        i = 0
        while len(rows) < 1024:
            w = ((i.to_bytes(4, "little") + b"\xff\0\x1e_test") * 420)[:4097]
            i += 1
            hh = sha256(w).hexdigest()
            if int(hh, 16) % 100 < 2:
                continue
            pos = out.tell()
            out.write(w)
            rows.append(
                dict(
                    doc=pos, offset=pos, length=4097, split="train", start=0, sha256=hh
                )
            )
    st = source.stat()
    fake = dict(
        source=dict(
            path=str(source), inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns
        ),
        eligible_train=25491,
        eligible_short_final=1040,
        panels={
            "train": rows,
            "final_test": [dict(length=4001, offset=10**15, split="final_test")],
        },
    )
    monkeypatch.setattr(d, "BASE", folder)
    monkeypatch.setattr(d, "ROWS", folder / "reservation.json")
    monkeypatch.setattr(d, "selection", lambda: fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("GPU_LEASE_HELD", raising=False)
    q = d.prepare()
    assert (
        q["train_raw_source_bytes"] == 1024 * 4097
        and q["train_targets"] == 1024 * 4096
        and q["evaluation_windows_opened"] == 0
    )
    stream = d.WriteStream(q["train_identity"])
    doc, raw = stream.sample(4901, 1)
    assert sha256(raw).hexdigest() == next(r["sha256"] for r in rows if r["doc"] == doc)
    with pytest.raises(ValueError):
        stream.sample(4901, 1025)
    with pytest.raises(ValueError):
        d.WriteStream("wrong")
    assert (
        sha(folder / "reservation.json") == load(folder / "reserved.json")["rows_sha"]
    )
    p = folder / "train/windows.u8"
    with p.open("r+b") as f:
        x = f.read(1)
        f.seek(0)
        f.write(bytes([x[0] ^ 1]))
    with pytest.raises(ValueError, match="truncated"):
        d.WriteStream(q["train_identity"])
    with pytest.raises(FileExistsError):
        d.prepare()
