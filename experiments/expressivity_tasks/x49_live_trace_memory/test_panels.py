"""Synthetic split/shift and fail-closed tests. No access to evaluation text."""

import pytest
from experiments.expressivity_tasks.x49_live_trace_memory import panels as p


def metadata():
    items = []
    counts = {s: 0 for s in p.N}
    off = p.MIN_OFFSET
    for i in range(10000):
        raw = bytes((i % 256,)) * (p.T * 2) + i.to_bytes(4, "little")
        h = p.digest(raw)
        split = {0: "calibration", 1: "final_test"}.get(int(h, 16) % 100, "train")
        if counts[split] >= 3:
            continue
        r = dict(
            document=off, byte_offset=off, byte_length=len(raw), sha256=h, split=split
        )
        items.append((r, raw))
        counts[split] += 1
        off += len(raw) + 1
        if min(counts.values()) == 3:
            break
    assert min(counts.values()) == 3
    return items, off


def test_source_membership_order_and_shift():
    pairs, off = metadata()
    rows = [r for r, _ in pairs]
    orig = {r["sha256"]: raw for r, raw in pairs}
    dup = dict(rows[0], document=off, byte_offset=off)
    rows.append(dup)
    end = off + dup["byte_length"]
    bad = {rows[1]["sha256"]}
    panels, counts = p.select(rows, bad, dict.fromkeys(p.N, 2), end)
    assert sum(counts.values()) == 8 and all(len(v) == 2 for v in panels.values())
    flat = [r for v in panels.values() for r in v]
    assert len({r["sha256"] for r in flat} - bad) == 6 and all(
        r["doc"] != off for r in flat
    )
    for s, v in panels.items():
        expected = sorted(
            {r["sha256"] for r in rows if r["split"] == s} - bad,
            key=lambda h: p.digest(("x49-live-v1:" + h).encode()),
        )[:2]
        assert [r["sha256"] for r in v] == expected
        for r in v:
            if s != "train":
                with pytest.raises(ValueError, match="TRAIN"):
                    p.window(orig[r["sha256"]], r)
            else:
                w = p.window(orig[r["sha256"]], r)
                j = r["start"]
                assert w == orig[r["sha256"]][j : j + p.T + 1]
                with pytest.raises(ValueError, match="document"):
                    raw = orig[r["sha256"]]
                    p.window(bytes([raw[0] ^ 1]) + raw[1:], r)
    with pytest.raises(ValueError, match="insufficient"):
        p.select(rows, bad, p.N, end)
    with pytest.raises(ValueError, match="seed"):
        p.order(rows, -1)


def test_reject_identity_and_conflicting_splits():
    pairs, end = metadata()
    rows = [r for r, _ in pairs]
    r = rows[-1]
    q = dict.fromkeys(p.N, 1)
    with pytest.raises(ValueError, match="identity"):
        p.select(rows + [r], set(), q, end)
    with pytest.raises(ValueError, match="split"):
        p.select(
            [
                *rows[:-1],
                dict(r, split="train" if r["split"] != "train" else "final_test"),
            ],
            set(),
            q,
            end,
        )
    with pytest.raises(ValueError, match="duplicate"):
        p.select(
            rows + [dict(r, document=end, byte_offset=end, byte_length=p.T + 99)],
            set(),
            q,
            end + p.T + 99,
        )
    assert not list(
        p.membership(
            {"source": {"sha256": "a" * 64}, "document": 123, "sha256": "b" * 64}
        )
    )
