"""Only invented token sequences. Brute-force occurrence scans are the oracle."""

from collections import Counter
from itertools import product
import random

import pytest

from .demo import CITY, DOCS, JERSEY, NEW, OLD, PIECES, ROAD, YORK, tokens
from .index import TokenIndex, spelling_step


def scan(docs, key):
    counts = Counter()
    ends = 0
    for doc in docs:
        for end in range(len(key), len(doc) + 1):
            if doc[end - len(key) : end] == key:
                if end == len(doc):
                    ends += 1
                else:
                    counts[doc[end]] += 1
    return counts, ends


def compare(answer, expected):
    counts, ends = expected
    assert [(e.token, e.count) for e in answer.edges] == sorted(
        counts.items()
    ) and answer.ends == ends


@pytest.mark.parametrize("alpha", [(0, 1, 2), (0, 256, 70000)])
def test_direct_occurrences_and_children_even_when_many_lengths_share_states(alpha):
    rng = random.Random(271)
    docs = [tuple(rng.choices(alpha, k=n)) for n in (0, 0, 1, 2, 5, 12, 17)]
    docs.extend((alpha * 5, alpha * 5))
    idx = TokenIndex(docs)
    for n in range(6):
        for key in product(alpha, repeat=n):
            counts, ends = scan(docs, key)
            h = idx.seek(key)
            if counts or ends:
                assert h.length == n
                ans = idx.branch(h)
                compare(ans, (counts, ends))
                assert idx.suffix(h) == idx.seek(key[1:])
                for edge in ans.edges:
                    nxt = key + (edge.token,)
                    assert edge.child == idx.follow(h, edge.token) == idx.seek(nxt)
                    compare(idx.branch(edge.child), scan(docs, nxt))
            else:
                # seek is suffix backoff; follow is exact. They cannot silently alias.
                assert n > 0 and h.length < n
                at = idx.root
                for t in key:
                    at = idx.follow(at, t)
                    if at is None:
                        break
                assert at is None


def test_unknown_tokens_and_reset_cannot_look_across_documents():
    docs = [(0, 500), (701, 0), (), (0, 500)]
    idx = TokenIndex(docs)
    right = idx.seek((0, 500))
    assert idx.branch(right).ends == 2 and not idx.branch(right).edges
    assert idx.follow(right, 701) is None and idx.advance(right, 701) == idx.seek([701])
    assert idx.seek([500, 701]).length == 1
    h = idx.seek([0, 555, 0, 500])
    assert h == right and idx.advance(h, 99) == idx.root
    compare(idx.branch(idx.root), scan(docs, ()))
    empty = TokenIndex([])
    compare(empty.branch(empty.root), ({}, 0))
    idx2 = TokenIndex(docs)
    for call in (
        lambda: idx2.follow(h, 0),
        lambda: idx2.branch(h),
        lambda: idx2.advance(h, 0),
        lambda: idx2.suffix(h),
    ):
        with pytest.raises(ValueError, match="foreign"):
            call()
    for token in (-1, 1.2, True):
        with pytest.raises(ValueError):
            idx.follow(idx.root, token)
        with pytest.raises(ValueError):
            TokenIndex([[token]])


def test_streaming_advance_is_longest_matching_suffix_and_suffix_drops_one_token():
    rng = random.Random(21)
    alpha = [0, 71, 300, 4096]
    docs = [tuple(rng.choices(alpha, k=83)), (5,) * 19, tuple(alpha * 8)]
    idx = TokenIndex(docs)
    known = {
        doc[i:j]
        for doc in docs
        for i in range(len(doc) + 1)
        for j in range(i, len(doc) + 1)
    }
    for stream in [
        tuple(rng.choices(alpha + [999999], k=80)),
        (5,) * 25,
        tuple(alpha * 12),
    ]:
        h = idx.root
        for j, t in enumerate(stream, 1):
            h = idx.advance(h, t)
            text = stream[:j]
            best = next(text[i:] for i in range(j + 1) if text[i:] in known)
            assert h.length == len(best)
            compare(idx.branch(h), scan(docs, best))
            assert idx.suffix(h) == idx.seek(best[1:])


def test_toy_token_commits_without_requesting_the_next_observation():
    def bounded():
        yield from b"New York"
        raise AssertionError("peeked beyond committed token")

    stream = tokens(bounded())
    assert next(stream) == NEW and next(stream) == YORK


def test_full_token_fanout_preserves_information_lost_in_the_next_byte():
    assert all(
        not a.startswith(b) for a in PIECES.values() for b in PIECES.values() if a != b
    )
    assert all(b"".join(PIECES[t] for t in tokens(doc)) == doc for doc in DOCS)
    idx = TokenIndex(map(tokens, DOCS))
    h = idx.seek(tokens(b"New"))
    new = idx.branch(h)
    old = idx.branch(idx.seek([OLD]))
    assert new == idx.branch(idx.seek([NEW]))
    compare(new, ({YORK: 2, JERSEY: 1}, 0))
    compare(old, ({YORK: 1, JERSEY: 2}, 0))
    assert (
        spelling_step(new, PIECES, b"").continuing_bytes
        == spelling_step(old, PIECES, b"").continuing_bytes
        == {32: 3}
    )
    assert spelling_step(new, PIECES, b" ").continuing_bytes == {
        ord("Y"): 2,
        ord("J"): 1,
    }
    assert spelling_step(old, PIECES, b" ").continuing_bytes == {
        ord("Y"): 1,
        ord("J"): 2,
    }
    step = spelling_step(new, PIECES, b" York")
    assert (
        len(step.completed) == 1
        and not step.continuing_bytes
        and step.document_ends == 0
    )
    y = step.completed[0].child
    assert y == idx.advance(h, YORK) == idx.follow(h, YORK)
    compare(idx.branch(y), ({CITY: 2}, 0))
    compare(idx.branch(idx.suffix(y)), ({CITY: 2, ROAD: 1}, 0))
    assert idx.follow(y, ROAD) is None and idx.advance(y, ROAD) == idx.seek(
        [YORK, ROAD]
    )
    compare(
        idx.branch(h), ({YORK: 2, JERSEY: 1}, 0)
    )  # no mutation from hypothetical choice


def test_spelling_prefix_collision_keeps_completed_and_unfinished_mass_separate():
    idx = TokenIndex([(0, 4), (1,), (2,), ()])
    a = idx.branch(idx.root)
    lex = {0: b"a", 1: b"ab", 2: b"a", 4: b"X"}  # two IDs decode to the same bytes
    assert spelling_step(a, lex, b"").document_ends == 4
    at = spelling_step(a, lex, b"a")
    assert (
        at.continuing_bytes == {98: 1}
        and at.document_ends == 0
        and [(e.token, e.count) for e in at.completed] == [(0, 1), (2, 1)]
    )
    compare(idx.branch(at.completed[0].child), ({4: 1}, 0))
    compare(idx.branch(at.completed[1].child), ({}, 1))
    later = spelling_step(a, lex, b"ab")
    assert [e.token for e in later.completed] == [1]
    assert [(e.token, tail) for e, tail in later.needs_successor] == [
        (0, b"b"),
        (2, b"b"),
    ]
    miss = spelling_step(a, lex, b"zz")
    assert not (
        miss.completed
        or miss.continuing_bytes
        or miss.needs_successor
        or miss.document_ends
    )
    with pytest.raises(ValueError):
        spelling_step(a, {**lex, 0: b""}, b"")
    with pytest.raises(ValueError):
        list(tokens(b"New J"))
