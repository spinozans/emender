"""Hash-checked, read-only C14 byte windows; no text is written to disk."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class ByteStream:
    def __init__(self, source, offsets):
        self.source = Path(source)
        self.offsets = Path(offsets)

    def windows(self, split, context=2048):
        with self.source.open('rb') as raw, self.offsets.open() as index:
            for line in index:
                doc = json.loads(line)
                if doc['split'] != split or doc.get('truncated'):
                    continue
                raw.seek(doc['byte_offset'])
                text = raw.read(doc['byte_length'])
                if len(text) != doc['byte_length'] or hashlib.sha256(text).hexdigest() != doc['sha256']:
                    raise ValueError('document hash mismatch: ' + str(doc['source_index']))
                for start in range(0, len(text)-1, context):
                    piece = np.frombuffer(text[start:start+context+1], dtype=np.uint8).astype(np.int64)
                    yield doc['source_index'], start, piece[:-1], piece[1:]

    def batches(self, split, context=2048, batch=64):
        rows = []
        for row in self.windows(split, context):
            rows.append(row)
            if len(rows) == batch:
                yield rows
                rows = []
        if rows:
            yield rows

    def slope_windows(self):
        """Length-only intervention: identical document/start for every probe."""
        anchor = next(row for row in self.windows('valid', 4096) if len(row[2]) == 4096)
        return [(anchor[0], anchor[1], anchor[2][:length], anchor[3][:length])
                for length in (512, 1024, 2048, 4096)]
