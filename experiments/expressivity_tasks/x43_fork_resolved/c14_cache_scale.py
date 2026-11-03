"""Measure exact cross-chunk query reuse and sparse binary C14 cache storage.

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

MAGIC = b'C14SPARSE1\n'
WINDOW = struct.Struct('<QQH')
ROW = struct.Struct('<BBBH')  # context length, censored, EOF, number of branches
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
    if not 1 <= len(rows) <= 128 or start % 128:
        raise ValueError('invalid window geometry')
    out.write(WINDOW.pack(doc, start, len(rows)))
    for position, row in enumerate(rows, 1):
        if not 0 <= row['context_bytes'] <= position or row['eof'] not in (0, 1):
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
        if not 1 <= length <= 128 or start % 128:
            raise ValueError('invalid window geometry')
        rows = []
        for position in range(1, length+1):
            k, censored, eof, size = ROW.unpack(exact(stream, ROW.size))
            if k > position or censored > 1 or eof > 1 or size > 256 or (censored and not k):
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


def benchmark(output, windows, jobs, capacity):
    # New destination only: never overwrite a completed cache or dense evidence.
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    server = BranchServer(output/'native.stderr', jobs=jobs)
    reuse = QueryReuse(server.query, capacity)
    distribution = Counter()
    support = {str(k): Counter(eligible=0, supported=0) for k in (16, 64, 256, 1024)}
    n = nw = covered = terminal = censored = 0
    path = output/'features.sparse.gz'
    try:
        with gzip.open(path, 'wb', compresslevel=1) as out:
            out.write(MAGIC)
            chunk = []
            def flush():
                nonlocal n, nw, covered, terminal, censored
                rows, rates, _ = select([s for _, _, ss in chunk for s in ss], reuse)
                at = 0
                for doc, start, ss in chunk:
                    selected = rows[at:at+len(ss)]
                    write_window(out, doc, start, selected)
                    for row in selected:
                        n += 1
                        covered += row['coverage']
                        terminal += row['terminal_count'] > 0
                        censored += row['censored']
                        distribution[row['context_bytes']] += 1
                    nw += 1
                    at += len(ss)
                for k in support:
                    support[k].update(rates[k])
                print(json.dumps(dict(windows=nw, positions=n, seconds=time.monotonic()-begin,
                    native_queries=server.ident, reuse_hits=reuse.hits)), flush=True)
            for window in source_windows(windows):
                chunk.append(window)
                if len(chunk) == 128:
                    flush()
                    chunk = []
            if chunk:
                flush()
    finally:
        server.close()
    elapsed = time.monotonic()-begin
    summary = dict(status='passed', scope='bounded training prefix; NOT complete train/valid cache',
        windows=nw, positions=n, seconds=elapsed, features_per_second=n/elapsed,
        native_queries=server.ident, native_calls=server.calls, reuse_hits=reuse.hits,
        reuse_capacity=capacity, jobs=jobs, cpu_affinity_count=len(os.sched_getaffinity(0)),
        native_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
        coverage_positions=covered, terminal_positions=terminal, censored_positions=censored,
        maximal_context_histogram=dict(sorted(distribution.items())), support=support,
        feature_path=str(path), feature_sha256=digest(path), feature_bytes=path.stat().st_size,
        xsa_sha256=digest(XSA), source_span_sha256=digest(ROOT/'receipts/c14_dense/pile_span.json'),
        eval_construction_sha256=digest(ROOT/'receipts/c14_dense/eval_construction.json'),
        extractor_sha256=digest(__file__), selector_sha256=digest(ROOT/'c14_branch_serve.py'),
        python=__import__('sys').executable, gpu_used=False, materialized_corpus_bytes=0,
        compressed_bytes_per_position=path.stat().st_size/n)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def compare(directory, reference):
    """Exact per-position replay versus the frozen JSON preflight, no tolerances."""
    summary = json.loads((directory/'summary.json').read_text())
    if digest(directory/'features.sparse.gz') != summary['feature_sha256']:
        raise ValueError('cache identity changed')
    n = nw = 0
    with gzip.open(directory/'features.sparse.gz', 'rb') as sparse, gzip.open(reference, 'rt') as old:
        for doc, start, rows in read_windows(sparse):
            nw += 1
            for position, row in enumerate(rows, 1):
                line = old.readline()
                if not line:
                    raise ValueError('reference exhausted')
                expected = json.loads(line)
                actual = json.loads(json.dumps(dict(document=doc, start=start, position=position, **row)))
                if actual != expected:
                    raise ValueError(f'cache mismatch at {doc}/{start}/{position}')
                n += 1
        if old.readline():
            raise ValueError('sparse cache exhausted')
    if (n, nw) != (summary['positions'], summary['windows']):
        raise ValueError('summary cardinality mismatch')
    result = dict(status='passed', positions=n, windows=nw, exact_row_equality=True,
        sparse_sha256=summary['feature_sha256'], reference_sha256=digest(reference),
        scope='exact cache arithmetic and identity replay; native oracle is separate')
    (directory/'parity.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=8192)
    parser.add_argument('--jobs', type=int, default=128)
    parser.add_argument('--capacity', type=int, default=250000)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(args.output, args.compare)))
    else:
        if args.windows < 1 or args.jobs < 1 or args.capacity < 0:
            raise ValueError('invalid extraction bounds')
        print(json.dumps(benchmark(args.output, args.windows, args.jobs, args.capacity)))


if __name__ == '__main__':
    main()
