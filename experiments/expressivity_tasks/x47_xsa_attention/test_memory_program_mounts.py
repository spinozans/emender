"""Scope/visibility/value contracts, plus independent known-positive native fixtures (not profiling)."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from ndm.memory_program import MaterializedBranches, Span, Unavailable
from ndm.memory_program.spans import SpanValues, mean_byte_values
from .b1_mount import ArchiveDescriptor, HERE, ROOT
from .memory_program_mounts import ArchiveRecords, LiveRecords
from .w3_data import Pages
from .workbench_native import EXE


def test_span_value_admission_boundary_encoding_and_gradient():
    read = []
    raw = b"\x00\x1e\xff"
    weight = torch.nn.Parameter(torch.arange(256 * 2, dtype=torch.float).view(256, 2))

    def fetch(d, a, n):
        read.append((d, a, n))
        return raw[a : a + n]

    args = ("S", "v1", lambda d: (0, 3) if d == 17 else None, fetch)
    with pytest.raises(PermissionError):
        SpanValues(*args, scope="one-region")
    v = SpanValues(*args, scope="whole", encode=lambda ts: mean_byte_values(ts, weight))
    a = Span("S", "v1", 17, 0, 1)
    invalid = [replace(a, document=18), replace(a, version="v0"), replace(a, source="Q"), replace(a, stop=4)]
    out = v.reader(following=True).read_many((*invalid, replace(a, stop=3), a, a), 8)
    assert all(isinstance(x, Unavailable) for x in out[:4]) and out[4] == () and read == [(17, 1, 2)] * 2
    h = out[-1][0]
    assert h.keys == (b"\x1e", b"\x1e\xff", replace(a, start=1, stop=3))
    torch.testing.assert_close(h.value, (weight[30] + weight[255]) / 2)
    (h.value + out[-2][0].value).sum().backward()
    assert weight.grad.nonzero()[:, 0].tolist() == [30, 30, 255, 255]
    assert (weight.grad[[30, 255]] == 1).all()
    assert v.reader().read_many((a,), 8)[0][0].keys == (b"\x00",)
    short = SpanValues("s", "v", lambda d: (0, 1), lambda *xs: b"", scope="whole")
    with pytest.raises(ValueError, match="incomplete"):
        short.reader().read_many((Span("s", "v", 1, 0, 1),), 1)


@pytest.mark.skipif(not EXE.exists(), reason="local SXGC binary")
def test_live_native_locations_and_follow_are_limited_by_watermark():
    live = LiveRecords(5, 900)
    try:
        live.advance(b"aba\x00\xffaba")
        _, loc, contents, following = live.readers()
        a, b = (hit.keys[0] for hit in loc.read_many((b"aba",), 2)[0])
        assert (a.start, b.start) == (900, 905)
        assert isinstance(loc.read_many((b"a",), 2)[0], Unavailable)  # proof before issuing a fatal overcap
        assert following.read_many((b,), 4) == [()]
        live.advance(b"\x1e")
        assert following.read_many((b,), 4)[0][0].keys[:2] == (b"\x1e", b"\x1e")
        assert contents.read_many((a,), 4)[0][0].keys == (b"aba",)
        # Even with the latest watermark a different version/other document isn't admissible.
        assert isinstance(following.read_many((replace(a, version="stale"),), 4)[0], Unavailable)
        assert live.stats["locate_rpcs"] == 1
    finally:
        live.close()


@pytest.mark.skipif(not EXE.exists(), reason="local SXGC binary and registered F6")
def test_verified_archive_records_never_read_next_document():
    conf = json.loads((HERE / "b1_mounts.json").read_text())["F6"]
    desc = ArchiveDescriptor("F6", conf, ROOT)
    with pytest.raises(PermissionError):
        ArchiveRecords(None, None, scope="excluded")
    with pytest.raises(ValueError, match="file changed"):
        ArchiveRecords(desc, {**conf["files"][1], "inode": 0}, scope="whole")
    archive = ArchiveRecords(desc, conf["files"][1], scope="whole")
    try:
        raw = desc.text.read_bytes()
        cut = int(desc.starts[1])
        key = raw[:8]
        crossing = raw[cut - 8 : cut + 8]
        loc, contents, following = archive.readers()
        found, cross, over, miss = loc.read_many(
            (key, crossing, b" the quick brown fox ", b"\x00\x1e\xff"), 8
        )
        starts = desc.starts
        expected = []
        at = raw.find(key)
        while at != -1:
            doc = int(starts.searchsorted(at, "right")) - 1
            j = at - int(starts[doc])
            hi = archive.bounds(doc)[1]
            if j + len(key) <= hi:
                expected.append(Span("F6", desc.version, doc, j, j + len(key)))
            at = raw.find(key, at + 1)
        assert expected and {h.keys[0] for h in found} == set(expected)
        assert all(h.keys[0].start + int(starts[h.keys[0].document]) != cut - 8 for h in cross)
        assert isinstance(over, Unavailable) and miss == ()
        x = Span("F6", desc.version, 0, cut - 1, cut)
        assert following.read_many((x,), 1) == [()]
        assert contents.read_many((x,), 1)[0][0].keys == (raw[cut - 1 : cut],)
        assert (
            following.read_many((replace(x, start=cut - 2, stop=cut - 1),), 1)[0][0].keys[1]
            == raw[cut - 1 : cut]
        )
    finally:
        archive.close()


@pytest.mark.skipif(not EXE.exists(), reason="pinned local P100 mount and workers")
def test_p100_existing_external_positive_coordinates_and_contents():
    conf = json.loads((HERE / "b1_mounts.json").read_text())["P100"]
    pages = Pages()
    try:
        bank = MaterializedBranches("P100", pages.desc.version, pages, scope="whole")
        archive = ArchiveRecords(pages.desc, conf["files"][1], scope="whole", branches=bank)
        try:
            registration = (
                HERE.parent / "x43_fork_resolved/receipts/p100_query_verification/registration.json"
            )
            pats = json.loads(registration.read_text())["patterns"]
            keys = [bytes.fromhex(pats[i]["hex"]) for i in (2, 6)]
            old = [json.loads(line) for line in (HERE / "receipts/b2/samples.jsonl").read_text().splitlines()]
            ref = next(
                row
                for row in old
                if row["bank"] == "P100" and row["ids"] == [2, 6] and row["mode"] == "original"
            )
            assert ref["pattern_hashes"] == [hashlib.sha256(k).hexdigest() for k in keys]
            loc, contents, following = archive.readers()
            first, second, common = loc.read_many((*keys, b"the "), 8)
            assert (
                isinstance(common, Unavailable)
                and archive.stats["certified_overcap"] == 1
                and archive.stats["worker_keys"] == 2
            )
            assert [len(first), len(second)] == [2, 1]
            for row, k, expected in zip((first, second), keys, ref["positions"], strict=True):
                spans = [h.keys[0] for h in row]
                absolute = [int(pages.desc.starts[s.document]) + s.start for s in spans]
                assert sorted(absolute) == sorted(expected)
                for s, pos, original, future in zip(
                    spans, absolute, contents.read_many(spans, 8), following.read_many(spans, 8), strict=True
                ):
                    assert original[0].keys == (k,)
                    with pages.desc.text.open("rb") as f:
                        f.seek(pos + len(k))
                        n = min(16, archive.bounds(s.document)[1] - s.stop)
                        text = f.read(n)
                    assert text and future[0].keys == (
                        text[:1],
                        text,
                        replace(s, start=s.stop, stop=s.stop + n),
                    )
        finally:
            archive.close()
    finally:
        pages.close()
