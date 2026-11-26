"""Invented native tokens and source bytes only. No language corpus or fitted weights."""

from dataclasses import replace
import os

import pytest
import torch

from experiments.expressivity_tasks.retrieved_span_effects.natural_model import (
    Shape,
    TokenLM,
    HOSTS,
    ARMS,
)
from experiments.expressivity_tasks.retrieved_span_effects.token_postings import literal
from .archive import Archive, digest
from .session import FrozenHost


S = Shape(vocab=41, width=12, heads=2, key=4, time=7, prefix=3, hits=4, page=5)
DOCS = [
    [2, 3, 5, 11, 2, 3, 5, 7, 11, 8, 9],
    [2, 3, 4, 13, 16, 5, 7, 3, 2, 14],
    [2, 3, 5, 11, 2, 21, 20, 25],
    [5, 7, 2, 17, 6, 1],
    [2, 3, 5, 11, 2],
]
TOK = digest(b"invented length-prefixed byte spellings/v1")
LIVE = digest(b"provisioned disjoint stream identity")


def spell(i):
    return bytes([127 + i]) + b"\xff" * (i % 3)


def raw(doc):
    return b"".join(map(spell, doc))


def encode(buf):
    at = 0
    while at < len(buf):
        i = buf[at] - 127
        yield i
        at += len(spell(i))


def build(docs=DOCS):
    return Archive(
        map(raw, docs), encode, spell, tokenizer_sha256=TOK, vocab=S.vocab, page=S.page
    )


def test_exact_occurrence_handles_and_independent_membership():
    a = build()
    prefix = [2, 3, 5]
    got = a.locate(prefix, S.hits)
    rows, v = literal(DOCS, prefix, S.page, S.hits)
    assert [(h.ordinal, h.mass) for h in got] == [
        (i, w) for i, w in zip(rows, v, strict=True) if i >= 0
    ]
    sources = {digest(raw(d)): (d, raw(d)) for d in DOCS}
    for h in got:
        p = h.span
        doc, buf = sources[p.source_sha256]
        t = a.tokens(h).tolist()
        assert doc[p.token_start : p.token_end] == t and buf[
            p.byte_start : p.byte_end
        ] == raw(t)
    assert (
        got[0].span.source_sha256 == digest(raw(DOCS[0]))
        and got[0].span.token_end == 5
        and got[0].span.byte_end > 5
    )
    d = build(DOCS[::-1])
    assert d.version != a.version and build().version == a.version
    for bad in (
        replace(got[0], archive_version=d.version),
        replace(got[0], ordinal=-1),
        replace(got[0], span=replace(got[0].span, byte_end=3)),
    ):
        with pytest.raises(ValueError):
            a.tokens(bad)
    for doc in DOCS:  # includes a document contributing no new canonical page
        with pytest.raises(ValueError):
            a.check_live(digest(raw(doc)))
    assert not a.locate([39, 40], 4)
    with pytest.raises(ValueError, match="source bytes"):
        Archive(
            [b"a"],
            lambda b: [1],
            spell,
            tokenizer_sha256=TOK,
            vocab=S.vocab,
            page=S.page,
        )


OBS = [2, 3, 5, 7, 8, 14, 11, 5, 3, 2, 10, 2, 7, 21, 33, 33, 33, 3, 2, 4, 3, 2, 3]


def offline(model, obs):
    # Construct the packet separately with occurrence enumeration, then use the
    # unchanged training forward in the CPU Triton interpreter as logit reference.
    sh = model.sh
    pages = list(
        dict.fromkeys(
            tuple(d[i : i + sh.page]) for d in DOCS for i in range(0, len(d), sh.page)
        )
    )
    zs = []
    for a in range(0, len(obs), sh.time):
        w = obs[a : a + sh.time]
        n = len(w)
        ids = torch.tensor([w + [0] * (sh.time - n)])
        p = torch.zeros(1, sh.hits, sh.page, dtype=torch.int32)
        ns = torch.full((1, sh.hits), -1, dtype=torch.int32)
        votes = torch.zeros_like(ns)
        if n >= sh.prefix:
            found, mass = literal(DOCS, w[: sh.prefix], sh.page, sh.hits)
            for i, (key, v) in enumerate(zip(found, mass, strict=True)):
                if key >= 0:
                    t = pages[key]
                    ns[0, i] = len(t)
                    p[0, i, : len(t)] = torch.tensor(t)
                    votes[0, i] = v
        with torch.no_grad():
            zs.append(
                model(ids, torch.tensor([n], dtype=torch.int32), p, ns, votes)[
                    0, :n, : sh.vocab
                ]
            )
    return torch.cat(zs)


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("arm", ARMS)
def test_observations_and_compiled_reuse_match_training_logits(host, arm):
    if os.getenv("TRITON_INTERPRET") != "1" or os.getenv("CUDA_VISIBLE_DEVICES") != "":
        pytest.skip("requires explicit CPU Triton interpreter")
    torch.manual_seed(5507)
    m = TokenLM(S, host, arm).eval()
    expected = offline(m, OBS)
    img = build()
    refs = []
    for budget in (0, 65536):
        srv = FrozenHost(m, img, tokenizer_sha256=TOK, cache_bytes=budget)
        sess = srv.session(LIVE)
        out = []
        for j, tok in enumerate(OBS):
            z = sess.observe(tok)
            assert z.shape == (S.vocab,) and z.isfinite().all()
            out.append(z)
            assert sess.observations == j + 1 and sess.position == j % S.time + 1
        assert (
            sess.lookups == 3
            and not sess.last_hits
            and srv.cache_payload_bytes <= budget
        )
        torch.testing.assert_close(torch.stack(out), expected, rtol=3e-5, atol=3e-6)
        rows = srv.encodes
        sess.reset(digest(b"another disjoint stream"))
        res = torch.stack([sess.observe(i) for i in OBS[:7]])
        torch.testing.assert_close(res, expected[:7], rtol=3e-5, atol=3e-6)
        assert sess.lookups == 1
        if arm == "local":
            assert rows == srv.encodes == srv.reuses == 0
        elif budget:
            assert rows == srv.encodes and srv.reuses >= 4
        else:
            assert srv.encodes == rows + 4 and not srv.reuses
        refs.append(srv.binding)
    assert refs[0] == refs[1]


def test_snapshot_eviction_and_failed_admission_dont_change_live_history():
    torch.manual_seed(5502)
    m = TokenLM(S, "gdn2", "g-effect").eval()
    img = build()
    with pytest.raises(ValueError):
        FrozenHost(m, img, tokenizer_sha256=digest(b"other tokenizer"))
    # Exactly one key plus two square maps per head fits. Working scratch is not LRU storage.
    one = (S.key + 2 * S.heads * S.key * S.key) * 4
    srv = FrozenHost(m, img, tokenizer_sha256=TOK, cache_bytes=one)
    s = srv.session(LIVE)
    first = [s.observe(t) for t in OBS[:7]]
    assert s.observations == 7 and srv.cache_payload_bytes == one
    for bad in (-1, S.vocab, True):
        with pytest.raises(ValueError):
            s.observe(bad)
    with pytest.raises(ValueError):
        s.reset(digest(raw(DOCS[-1])))
    assert s.observations == 7 and s.position == 7
    with torch.no_grad():
        m.embedding.weight[2] += 5.0
    assert FrozenHost(m, img, tokenizer_sha256=TOK).binding != srv.binding
    prior = srv.encodes
    s.reset(LIVE)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        again = [s.observe(t) for t in OBS[:7]]
    torch.testing.assert_close(torch.stack(first), torch.stack(again), rtol=0, atol=0)
    assert srv.encodes == prior + 4 and srv.cache_payload_bytes == one
