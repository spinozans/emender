"""A mounted reference collection. Handle resolution names both token and byte ranges.

Build from complete authorized reference documents, not a live transcript or targets.
Exact duplicate pages have one canonical occurrence; all source documents remain in
its exclusion set. No normalization of raw bytes or invented spelling after lookup.
"""

from dataclasses import dataclass
import hashlib
from operator import index as integer
import re

import numpy as np

from experiments.expressivity_tasks.retrieved_span_effects.token_postings import (
    BASE,
    index,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def checked_ref(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256 identity")
    return value


def token_id(t, vocab):
    if isinstance(t, (bool, np.bool_)):
        raise ValueError("a boolean is not a token ID")
    t = integer(t)
    if not 0 <= t < vocab:
        raise ValueError("token outside the mounted vocabulary")
    return t


@dataclass(frozen=True)
class Span:
    source_sha256: str
    token_start: int
    token_end: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class Hit:
    archive_version: str
    ordinal: int
    mass: int
    span: Span


class Archive:
    def __init__(self, documents, encode, spelling, *, tokenizer_sha256, vocab, page):
        """`encode(raw)` gives native IDs; `spelling(id)` gives their exact bytes."""
        if not 0 < vocab < BASE or page <= 0:
            raise ValueError("invalid archive geometry")
        self.tokenizer = checked_ref(tokenizer_sha256)
        self.vocab = vocab
        self.page = page
        version = hashlib.sha256(
            f"token-pages/1:{self.tokenizer}:{vocab}:{page}:".encode()
        )
        rows = []
        sources = set()
        first = {}
        for raw in documents:
            if not isinstance(raw, bytes) or not raw:
                raise ValueError(
                    "expected a nonempty complete reference document in bytes"
                )
            source = digest(raw)
            if source in sources:
                raise ValueError("duplicate source document")
            sources.add(source)
            ids = np.array([token_id(t, vocab) for t in encode(raw)], dtype="<i4")
            ends = [0]
            for t in ids:
                piece = spelling(int(t))
                if (
                    not isinstance(piece, bytes)
                    or not piece
                    or not raw.startswith(piece, ends[-1])
                ):
                    raise ValueError("token spelling does not reproduce source bytes")
                ends.append(ends[-1] + len(piece))
            if ends[-1] != len(raw):
                raise ValueError("uncovered source bytes")
            version.update(bytes.fromhex(source))
            version.update(hashlib.sha256(ids.tobytes()).digest())
            rows.append(ids)
            for a in range(0, len(ids), page):
                b = min(a + page, len(ids))
                first.setdefault(
                    ids[a:b].tobytes(), Span(source, a, b, ends[a], ends[b])
                )
        self._image = im = index(rows, page)
        self._spans = tuple(first.values())
        self.sources = frozenset(sources)
        self.version = version.hexdigest()
        if len(im.lens) != len(first) or any(
            im.pages[i, : im.lens[i]].astype("<i4", copy=False).tobytes() != key
            for i, key in enumerate(first)
        ):
            raise ValueError("provenance order differs from indexed page order")
        for arr in vars(im).values():
            arr.flags.writeable = False

    @property
    def index_payload_bytes(self):
        return sum(x.nbytes for x in vars(self._image).values())

    def check_live(self, source_sha256):
        if checked_ref(source_sha256) in self.sources:
            raise ValueError(
                "a live document may not be a member of its reference archive"
            )

    def locate(self, prefix, hits):
        if not 0 < hits <= 16 or not 0 < len(prefix) <= 64:
            raise ValueError("expected 1..64 observed tokens and 1..16 hits")
        ids = [1 + token_id(t, self.vocab) for t in prefix]
        grams = []
        for end, x in enumerate(ids):
            grams.append(x)
            if end:
                grams.append(x + BASE * ids[end - 1])
            if end >= 2:
                grams.append(x + BASE * ids[end - 1] + BASE**2 * ids[end - 2])
        im = self._image
        score = np.zeros(len(im.lens), np.int64)
        keys, rep = np.unique(np.asarray(grams, np.int64), return_counts=True)
        for k, i, r in zip(keys, np.searchsorted(im.codes, keys), rep, strict=True):
            if i < len(im.codes) and im.codes[i] == k:
                # Rows within one posting list are distinct; repeated query grams vote repeatedly.
                score[im.postings[im.offsets[i] : im.offsets[i + 1]]] += (
                    int(im.vote[i]) * r
                )
        positive = np.flatnonzero(score)
        rank = positive[np.lexsort((positive, -score[positive]))[:hits]]
        return tuple(
            Hit(self.version, int(i), int(score[i]), self._spans[i]) for i in rank
        )

    def tokens(self, hit):
        if (
            hit.archive_version != self.version
            or not 0 <= hit.ordinal < len(self._spans)
            or hit.span != self._spans[hit.ordinal]
        ):
            raise ValueError("stale or invalid span handle")
        i = hit.ordinal
        return self._image.pages[i, : self._image.lens[i]].copy()
