"""No natural source. Independent literal counts, all-byte distributions and causal prefixes."""

from collections import Counter
import hashlib
import re

import numpy as np
import pytest

from . import data as d, study as s
from .archive import Atlas, Builder, Service, pieces
from .reader import ARMS, Reader, reader, INSIDE, PRE, REFRESH, bands

RAW = (
    b"left-New Yellow!city,",
    b"x:New Yellow!city,",
    b":New Yonder!shore,",
    b":New Zulu!home,",
    b":Old Yellow!high,",
    b":Old Yonder!road,",
    b":Old Yonder!road,",
    b"\xff\x00a1_2.\x80abc_suffix",
    b"pure_open_fragment",
)
CLOSED = re.compile(rb"[A-Za-z0-9_]*[^A-Za-z0-9_]")


def literal_tokens(raw):
    return tuple(CLOSED.findall(raw)[1:])


@pytest.fixture(scope="module")
def atlas(tmp_path_factory):
    b = Builder()
    for w in RAW:
        b.add(w)
    f = tmp_path_factory.mktemp("invented") / "image.npz"
    b.save(f)
    return Atlas(f)


def counts(history, docs):
    for n in range(len(history), 0, -1):
        followers = Counter()
        occurs = False
        for doc in docs:
            for j in range(n, len(doc) + 1):
                if doc[j - n : j] == history[-n:]:
                    occurs = True
                    if j < len(doc):
                        followers[doc[j]] += 1
        if occurs:
            return n, followers
    return 0, Counter(t for doc in docs for t in doc)


def base(raws, x):
    hist = np.zeros((3, 256))
    n = len(x)
    for raw in raws:
        for i in range(1, len(raw)):
            c = raw[i]
            hist[0, c] += 1
            if raw[i - 1] == x[-1]:
                hist[1, c] += 1
            if i >= 2 and n >= 2 and raw[i - 2 : i] == x[-2:]:
                hist[2, c] += 1
    p = (hist[0] + 0.5) / (hist[0].sum() + 128)
    for h in hist[1 : 2 if n == 1 else 3]:
        p = (h + 16 * p) / (h.sum() + 16)
    return p


def independent(x, arm):
    docs = [literal_tokens(w) for w in RAW]
    B = base(RAW, x)
    hits = list(CLOSED.finditer(x))
    if not hits or arm == "BYTE":
        return B
    tail = x[hits[-1].end() :]
    history = tuple(m.group() for m in hits[1:])
    _, local = counts(history, docs)
    root = Counter(t for doc in docs for t in doc)
    if tail and arm == "FIRST":
        return B
    r = np.zeros(256)
    for token, num in root.items():
        if token.startswith(tail):
            assert len(token) > len(tail)
            mass = (
                num if tail and arm == "LEX" else local[token] + 4 * num / root.total()
            )
            r[token[len(tail)]] += mass
    return (B + r / r.sum()) / 2 if r.sum() else B


def test_lossless_closed_spelling_rule_and_boundaries():
    for raw in [
        *RAW,
        b"ab:xyz_",
        b"\0\xff\x00",
        b"",
        b"justWords",
        b"  abc\xc3\xa9def_",
    ]:
        assert tuple(pieces(raw)) == literal_tokens(raw)
        words = list(pieces(raw))
        assert all(CLOSED.fullmatch(w) for w in words)
        assert not any(w.startswith(p) for w in words for p in words if w != p)
    assert tuple(pieces(b"left:New York!tail")) == (b"New ", b"York!")


@pytest.mark.parametrize("arm", ARMS)
def test_every_byte_matches_occurrence_scan_and_independent_ngram(arm, atlas):
    m = reader(arm, atlas)
    for x in [
        b"torn-New Yellow!city,\xff\0Old Yonder!",
        b"partial",
        b"\x00NeverSeen_Y99!Old Yzip??",
        b":New Zulu!home,ZZ\xff",
    ]:
        z = m.predict(x)
        ref = np.stack([independent(x[: t + 1], arm) for t in range(len(x))])
        np.testing.assert_allclose(z.pdf, ref, rtol=3e-14, atol=2e-15)
        for t in range(len(x)):
            cut = m.predict(x[: t + 1])
            np.testing.assert_array_equal(cut.pdf[-1], z.pdf[t])
            if arm != "BYTE":
                np.testing.assert_array_equal(cut.route[-1], z.route[t])
                assert cut.counts["reads"] == len(CLOSED.findall(x[: t + 1]))


def test_same_first_real_lookup_then_distinct_within_token_information(atlas):
    x = b":New Y"
    one, lex, full = (reader(a, atlas).predict(x) for a in ("FIRST", "LEX", "SPELL"))
    np.testing.assert_array_equal(one.pdf[-2], lex.pdf[-2])
    np.testing.assert_array_equal(one.pdf[-2], full.pdf[-2])
    assert (
        max(abs(full.pdf[-1] - lex.pdf[-1])) > 0.001
        and full.counts["reads"] == 2
        and full.counts["known_commits"] == 1
        and full.counts["context_supported_gap"] > 0
    )
    for p in (full, one, lex):
        assert np.all(p.route[-1] == -1) and np.array_equal(p.route, full.route)
    assert (
        one.counts["wire_bytes"] == 2 + 1024 and full.counts["wire_bytes"] == 2 + 3 * 8
    )


def test_only_content_commits_drive_navigation_and_root_tags(atlas):
    serv = Service(atlas, True, capacity=2)
    model = Reader(
        "SPELL",
        atlas.base,
        service=serv,
        root_first=atlas.lex.first_counts,
        lex=atlas.lex,
    )
    orig = serv.refresh
    asked = []

    def audited(h, token, c):
        asked.append(token)
        return orig(h, token, c)

    serv.refresh = audited
    x = b"abcdefgh\xffNew Yonder!Unseen?zz"
    a = model.predict(x)
    assert (
        asked == [None, b"New ", b"Yonder!", b"Unseen?"]
        and a.counts["root_reads"] == 2
        and a.counts["known_commits"] == 2
    )
    assert not reader("BYTE", atlas).predict(x).counts
    with pytest.raises(ValueError):
        Reader(
            "FIRST",
            atlas.base,
            service=serv,
            root_first=atlas.lex.first_counts,
            lex=atlas.lex,
        )
    with pytest.raises(ValueError):
        Reader("BYTE", atlas.base, service=serv)


def test_mounted_index_matches_shifted_counts_and_structural_end(atlas):
    assert atlas.geometry["byte_targets"] == sum(
        len(w) - 1 for w in RAW
    ) and atlas.geometry["tokens"] == sum(len(literal_tokens(w)) for w in RAW)
    ids = atlas.ids
    h = atlas.index.seek([ids[b"Yonder!"], ids[b"shore,"]])
    assert atlas.index.branch(h).ends == 1 and not atlas.index.branch(h).edges
    assert atlas.index.follow(h, ids[b"Old "]) is None and atlas.index.advance(
        h, ids[b"Old "]
    ) == atlas.index.seek([ids[b"Old "]])


def test_paired_dose_denominators_exact_controls_and_bootstrap(atlas):
    words = [
        (w * 50)[:65]
        for w in (b":New Yellow!city,", b"NoDelimiter", b"\0\xff", b"a:Old Yonder!")
    ] * 4
    tick = []
    facts, ds = s.phase(atlas, words, tick.append)
    z = s.summarize(ds, facts, draws=513)
    alt = s.summarize(ds, facts, draws=513, use_counts=False)
    s.check_work(facts, z["targets"])
    assert (
        tick == [16]
        and z["targets"]["all"] == 1024
        and z["changed_gap_predictions"] > 0
    )
    labels = np.stack([bands(w[:-1]) for w in words])
    assert z["targets"]["refresh"] == int((labels == REFRESH).sum()) and z["targets"][
        "pre"
    ] == int((labels == PRE).sum())
    k = ARMS.index("SPELL")
    mean = float(ds["loss"][:, k, INSIDE].sum() / (labels == INSIDE).sum())
    assert z["absolute"]["SPELL"]["inside"] == pytest.approx(mean)
    for key, bs in z["contrasts"].items():
        for b, v in bs.items():
            np.testing.assert_allclose(
                v["ci95"], alt["contrasts"][key][b]["ci95"], rtol=3e-13, atol=1e-15
            )
    # No invented wins when a band is unexposed.
    f, v = s.phase(atlas, [b"NoDelimiter"] * 16, lambda _: None)
    empty = s.summarize(v, f, draws=31)
    s.check_work(f, empty["targets"])
    assert empty["absolute"]["SPELL"]["inside"] is None and not empty["spelling_win"]


def test_spent_prefix_no_skip_or_overlap_after_three_studies():
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import select

    records = []
    for i in range(34):
        h = hashlib.sha256(str(i).encode()).hexdigest()
        off = d.MIN_OFFSET + i * 10000
        if int(h, 16) % 100 >= 2:
            records.append(
                dict(
                    sha256=h,
                    document=off,
                    byte_offset=off,
                    byte_length=5000,
                    split="train",
                )
            )
    size = off + 10000
    ex = {records[-1]["sha256"]}
    rows = select(records, ex, {"train": 22}, size)[0]["train"]
    spent = rows[:3] + rows[3:9] + rows[9:16]
    fresh, excluded, _ = d.p52.last.next_block(records, ex, spent, 6, size)
    assert fresh == rows[16:22] and len(excluded) == len(ex) + 16
    for wrong in (spent[1:] + rows[16:17], spent + spent[:1]):
        with pytest.raises(ValueError):
            d.p52.last.next_block(records, ex, wrong, 6, size)


def test_probe_cannot_open_before_image_and_independent_witness(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "RUN", tmp_path)
    monkeypatch.setattr(d, "qualified", lambda: None)
    with pytest.raises(FileNotFoundError, match="frozen.json"):
        list(d.windows("probe"))
    monkeypatch.setattr(d, "seal", lambda: {})
    with pytest.raises(FileNotFoundError, match="freeze-ack.json"):
        list(d.windows("probe"))
    assert not list(tmp_path.iterdir())
