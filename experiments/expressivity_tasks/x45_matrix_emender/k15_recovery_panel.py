"""Pinned manifest-order document panels shared by the two recovery studies."""
import hashlib
from itertools import islice


def panel(stream, split):
    skip = 128 if split == 'calibration' else 0
    docs = (d for d in stream.documents(split) if d['length'] >= 1025)
    selected = list(islice(docs, skip, skip+64))
    assert len({d['doc'] for d in selected}) == 64
    with stream.source.open('rb') as f:
        for d in selected:
            f.seek(d['offset'])
            text = f.read(d['length'])
            assert hashlib.sha256(text).hexdigest() == d['sha256']
            yield d['doc'], text[:1025]
