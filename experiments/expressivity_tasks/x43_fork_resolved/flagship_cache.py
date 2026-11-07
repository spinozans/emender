"""Versioned T2048 sparse codec with 16-bit consultation context lengths.

A bounded engineering benchmark, never a substitute for complete train/valid
caches. Uses the unchanged, fixture-validated selector and frozen source order.
"""
import argparse
from collections import Counter, OrderedDict
import gzip
import json
import math
import os
from pathlib import Path
import resource
import struct
import time

from .c14_branch_serve import BranchServer, ROOT, XSA, exact, select, source_windows
from .c14_feature_extract import digest

MAGIC = b'FLAGSHIPSPARSE2\n'
WINDOW = struct.Struct('<QQH')
ROW = struct.Struct('<HBBH')  # context length, censored, EOF, number of branches
PAIR = struct.Struct('<BQ')


class QueryReuse:
    """Bounded LRU; eviction changes query cost only, never counts or features."""
    def __init__(self, query, capacity):
        if capacity < 0:
            raise ValueError('negative capacity')
        self.query = query
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, patterns):
        missing = list(dict.fromkeys(p for p in patterns if p not in self.cache))
        replies = self.query(missing) if missing else []
        if len(replies) != len(missing):
            raise ValueError('response cardinality mismatch')
        fetched = dict(zip(missing, replies))
        # Resolve all results before eviction, even for batches above capacity.
        result = []
        for pattern in patterns:
            if pattern in self.cache:
                result.append(self.cache[pattern])
                self.cache.move_to_end(pattern)
                self.hits += 1
            else:
                result.append(fetched[pattern])
        self.misses += len(missing)
        if self.capacity:
            self.cache.update(fetched)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return result


def write_window(out, doc, start, rows):
    if not 1 <= len(rows) <= 2048 or start % 2048:
        raise ValueError('invalid window geometry')
    out.write(WINDOW.pack(doc, start, len(rows)))
    for position, row in enumerate(rows, 1):
        if not 0 <= row['context_bytes'] <= min(position, 1024) or row['eof'] not in (0, 1):
            raise ValueError('invalid row geometry')
        pairs = row['branches']
        if len(pairs) > 256 or len({b for b, _ in pairs}) != len(pairs):
            raise ValueError('invalid branches')
        out.write(ROW.pack(row['context_bytes'], row['censored'], row['eof'], len(pairs)))
        for byte, count in pairs:
            if not 0 <= byte <= 255 or not 1 <= count < 2**64:
                raise ValueError('invalid count')
            out.write(PAIR.pack(byte, count))


def read_windows(stream):
    if exact(stream, len(MAGIC)) != MAGIC:
        raise ValueError('wrong sparse cache version')
    while True:
        header = stream.read(WINDOW.size)
        if not header:
            return
        if len(header) != WINDOW.size:
            raise EOFError('truncated window header')
        doc, start, length = WINDOW.unpack(header)
        if not 1 <= length <= 2048 or start % 2048:
            raise ValueError('invalid window geometry')
        rows = []
        for position in range(1, length+1):
            k, censored, eof, size = ROW.unpack(exact(stream, ROW.size))
            if k > min(position, 1024) or censored > 1 or eof > 1 or size > 256 or (censored and not k):
                raise ValueError('invalid sparse row')
            pairs = [PAIR.unpack(exact(stream, PAIR.size)) for _ in range(size)]
            if len({b for b, _ in pairs}) != size or any(c == 0 for _, c in pairs):
                raise ValueError('invalid sparse branches')
            if k == 0 and (pairs or eof):
                raise ValueError('empty context has counts')
            total = sum(c for b, c in pairs if b != 30)
            rows.append(dict(context_bytes=k, censored=bool(censored),
                context_count=sum(c for _, c in pairs)+eof, eof=eof,
                terminal_count=sum(c for b, c in pairs if b == 30), branches=pairs,
                support_size=total, distinct_branches=sum(b != 30 for b, _ in pairs),
                entropy=-sum(c/total*math.log(c/total) for b, c in pairs if b != 30) if total else 0.,
                coverage=bool(total)))
        yield doc, start, rows

