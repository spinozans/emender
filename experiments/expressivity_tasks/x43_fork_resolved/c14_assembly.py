"""Exact C14 assembly into bounded, memory-mappable numpy shards.

No corpus/query semantics change. Float64 preserves Python's registered arithmetic;
libm log1p and integer true division deliberately remain scalar for bit equality.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import c14_branch_serve as serving
from .c14_feature_extract import digest


def assemble(rows):
    """Scatter sparse probabilities into one contiguous batch, no dense row lists."""
    result = np.zeros((len(rows), 260), dtype='<f8')
    indices, values = [], []
    for i, row in enumerate(rows):
        total = row['support_size']
        if total:
            for byte, count in row['branches']:
                if byte != 30:
                    indices.append(i*260+byte)
                    values.append(count/total)
            result[i, 256:] = (math.log1p(total), row['entropy'],
                float(row['distinct_branches'] > 1), math.log1p(row['context_bytes']))
    result.ravel()[indices] = values
    return result


class Shards:
    """New directories only; manifest written only after all shards complete.

    Bound each shard to the caller's selector batch. np.load(..., mmap_mode='r')
    reads features without JSON parsing; identities are uint64 document/start/pos.
    """
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.shards = []

    def write(self, rows, identities):
        if len(rows) != len(identities):
            raise ValueError('identity cardinality mismatch')
        name = f'{len(self.shards):06d}'
        array = assemble(rows)
        ids = np.asarray(identities, dtype='<u8').reshape((-1, 3))
        paths = [self.directory/(name+'.features.npy'), self.directory/(name+'.ids.npy')]
        np.save(paths[0], array, allow_pickle=False)
        np.save(paths[1], ids, allow_pickle=False)
        self.shards.append(dict(positions=len(rows), files=[dict(path=p.name,
            sha256=digest(p), bytes=p.stat().st_size) for p in paths]))
        return array

    def finish(self, provenance):
        manifest = dict(version=1, dtype='<f8', geometry=260,
            identities=['document', 'start', 'position'], shards=self.shards,
            provenance=provenance)
        (self.directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest


def replay(reference, output, batch=16384):
    """Convert the existing oracle, validating every feature bit and identity."""
    if batch < 1:
        raise ValueError('positive batch required')
    writer = Shards(output)
    timings = Counter()
    n = 0
    old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
    begin = time.perf_counter()
    with gzip.open(reference, 'rt') as source:
        exhausted = False
        while not exhausted:
            rows = []
            at = time.perf_counter()
            for _ in range(batch):
                line = source.readline()
                if not line:
                    exhausted = True
                    break
                rows.append(json.loads(line))
            timings['gzip_read_and_json_parse'] += time.perf_counter()-at
            if not rows:
                break
            at = time.perf_counter()
            oracle = np.asarray([serving.features(row) for row in rows], dtype='<f8')
            timings['old_feature_lists_and_array'] += time.perf_counter()-at
            ids = [(r['document'], r['start'], r['position']) for r in rows]
            at = time.perf_counter()
            actual = writer.write(rows, ids)
            timings['new_assembly_save_and_hash'] += time.perf_counter()-at
            if actual.tobytes() != oracle.tobytes():
                raise ValueError(f'feature bits differ in batch at {n}')
            # Verify persisted arrays and identity order, not only in-memory output.
            name = f'{len(writer.shards)-1:06d}'
            stored = np.load(writer.directory/(name+'.features.npy'), mmap_mode='r')
            stored_ids = np.load(writer.directory/(name+'.ids.npy'), mmap_mode='r')
            if stored.tobytes() != oracle.tobytes() or not np.array_equal(stored_ids, ids):
                raise ValueError('persisted shard differs')
            old_hash.update(oracle.tobytes()); new_hash.update(actual.tobytes())
            n += len(rows)
    summary = dict(status='passed', positions=n, seconds=time.perf_counter()-begin,
        stage_seconds=dict(timings), assembly_features_per_second=n/timings['new_assembly_save_and_hash'],
        oracle_feature_sha256=old_hash.hexdigest(), assembled_feature_sha256=new_hash.hexdigest(),
        reference_sha256=digest(reference), python=__import__('sys').executable,
        scope='oracle replay and assembly only; not native extraction throughput')
    writer.finish(summary)
    return summary


def profile(output, windows=128, jobs=32):
    """Disjoint wall stages; decode is nested within query, never added twice."""
    output.mkdir(parents=True, exist_ok=False)
    timings = Counter()
    original = serving.decode
    def decode(*args):
        at = time.perf_counter()
        try:
            return original(*args)
        finally:
            timings['decode_nested_in_query'] += time.perf_counter()-at
    serving.decode = decode
    begin = time.perf_counter()
    at = time.perf_counter()
    server = serving.BranchServer(output/'native.stderr', jobs=jobs)
    timings['server_constructor'] = time.perf_counter()-at
    def query(patterns):
        at = time.perf_counter()
        try:
            return server.query(patterns)
        finally:
            timings['query_including_decode_and_pipe_wait'] += time.perf_counter()-at
    try:
        at = time.perf_counter()
        chunk = list(serving.source_windows(windows))
        timings['source_hash_and_windowing'] = time.perf_counter()-at
        lefts = [s for _, _, ss in chunk for s in ss]
        at = time.perf_counter()
        rows, _, queries = serving.select(lefts, query)
        timings['selector_excluding_query'] = time.perf_counter()-at-timings['query_including_decode_and_pipe_wait']
        at = time.perf_counter()
        with gzip.open(output/'old.jsonl.gz', 'wt') as out:
            for row in rows:
                out.write(json.dumps(row, separators=(',', ':'))+'\n')
        timings['old_json_and_gzip_write'] = time.perf_counter()-at
        at = time.perf_counter()
        array = assemble(rows)
        timings['new_feature_assembly'] = time.perf_counter()-at
        at = time.perf_counter()
        np.save(output/'features.npy', array, allow_pickle=False)
        ids = [(doc, start, pos) for doc, start, ss in chunk for pos in range(1, len(ss)+1)]
        np.save(output/'ids.npy', np.asarray(ids, dtype='<u8'), allow_pickle=False)
        timings['new_npy_write'] = time.perf_counter()-at
    finally:
        at = time.perf_counter()
        try:
            server.close()
        finally:
            timings['shutdown'] = time.perf_counter()-at
            serving.decode = original
    result = dict(positions=len(rows), native_queries=queries, queries_per_position=queries/len(rows),
        stage_seconds=dict(timings), seconds=time.perf_counter()-begin, jobs=jobs,
        xsa_sha256=digest(serving.XSA), selector_sha256=digest(serving.__file__),
        source_receipt_sha256=digest(serving.ROOT/'receipts/c14_dense/pile_span.json'),
        caveat='query residual includes native work, pipe wait, framing, writer and scheduling; not pure native time')
    (output/'profile.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def extract(output, windows, jobs):
    """Bounded native extraction directly to shards, without an intermediate JSON cache."""
    writer = Shards(output)
    begin = time.perf_counter()
    server = serving.BranchServer(output/'native.stderr', jobs=jobs)
    positions = queries = 0
    try:
        chunk = []
        def flush():
            nonlocal positions, queries
            rows, _, count = serving.select([s for _, _, ss in chunk for s in ss], server.query)
            ids = [(doc, start, pos) for doc, start, ss in chunk for pos in range(1, len(ss)+1)]
            writer.write(rows, ids)
            positions += len(rows)
            queries += count
        for window in serving.source_windows(windows):
            chunk.append(window)
            if len(chunk) == 128:
                flush()
                chunk = []
        if chunk:
            flush()
    finally:
        server.close()
    seconds = time.perf_counter()-begin
    summary = dict(status='passed', positions=positions, native_queries=queries,
        seconds=seconds, features_per_second=positions/seconds, jobs=jobs,
        xsa_sha256=digest(serving.XSA), selector_sha256=digest(serving.__file__),
        assembly_sha256=digest(__file__),
        source_receipt_sha256=digest(serving.ROOT/'receipts/c14_dense/pile_span.json'),
        scope='bounded training prefix; not complete train/valid cache', gpu_used=False)
    writer.finish(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=128)
    parser.add_argument('--jobs', type=int, default=32)
    args = parser.parse_args()
    if args.windows < 1 or args.jobs < 1:
        raise ValueError('positive bounds required')
    if args.reference and args.extract:
        parser.error('--reference and --extract are mutually exclusive')
    result = (replay(args.reference, args.output) if args.reference else
        extract(args.output, args.windows, args.jobs) if args.extract else
        profile(args.output, args.windows, args.jobs))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
