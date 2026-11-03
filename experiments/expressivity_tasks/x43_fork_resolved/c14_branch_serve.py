"""Exact framed branch transport and bounded, causal maximal-suffix extraction.

This measures the complete selector + 256-branch feature workflow, not just
queries at contexts that have already been selected. Corpus text is never saved.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
import resource
from pathlib import Path
import struct
import subprocess
import threading
import time

from .c14_feature_extract import digest, INDEX, N

ROOT = Path(__file__).resolve().parent
XSA = Path('/home/erikg/emender-scratch/c14-xsa-serving/xsa/target/release/xsa')


def exact(stream, n):
    parts = bytearray()
    while len(parts) < n:
        chunk = stream.read(n-len(parts))
        if not chunk:
            raise EOFError('truncated branch response')
        parts.extend(chunk)
    return bytes(parts)


def decode(payload, expected_id):
    if len(payload) < 50:
        raise ValueError('short branch response')
    ident, lo, hi, width, eof, other, size = struct.unpack_from('<6QH', payload)
    if ident != expected_id or hi < lo or hi > N or width != hi-lo:
        raise ValueError('branch ID or interval mismatch')
    if size > 256 or len(payload) != 50+size*9 or other or eof > 1:
        raise ValueError('incomplete branch distribution')
    pairs = [struct.unpack_from('<BQ', payload, 50+9*j) for j in range(size)]
    if len({b for b, _ in pairs}) != size or any(c == 0 for _, c in pairs):
        raise ValueError('duplicate or zero branch')
    if sum(c for _, c in pairs)+eof != width:
        raise ValueError('branch closure mismatch')
    return dict(width=width, eof=eof, branches=pairs)


class BranchServer:
    def __init__(self, log, jobs=32, xsa=XSA, index=INDEX):
        if jobs < 1:
            raise ValueError('jobs must be positive')
        self.log = open(log, 'wb')
        self.proc = subprocess.Popen([str(xsa), 'branches', str(index), '--serve',
            '-j', str(jobs), '-k', '256'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log,
            env={**os.environ, 'XSA_BRANCH_GATHER': '0', 'RAYON_NUM_THREADS': str(jobs)})
        self.ident = 0
        self.calls = 0

    def query(self, patterns):
        # Concurrent pipe writer prevents request/response backpressure deadlock.
        # Native workers are CPU query workers, not orchestration agents.
        errors = []
        def send():
            try:
                for pattern in patterns:
                    self.proc.stdin.write(struct.pack('<I', len(pattern))+pattern)
                self.proc.stdin.flush()
            except BaseException as exc:
                errors.append(exc)
        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        rows = []
        try:
            for _ in patterns:
                length, = struct.unpack('<I', exact(self.proc.stdout, 4))
                if not 50 <= length <= 50+256*9:
                    raise ValueError('invalid branch frame length')
                rows.append(decode(exact(self.proc.stdout, length), self.ident))
                self.ident += 1
        except BaseException:
            self.proc.kill()
            raise
        finally:
            writer.join()
        if errors:
            raise errors[0]
        self.calls += 1
        return rows

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            status = self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise
        finally:
            self.log.close()
        if status:
            raise RuntimeError(f'branch server exit {status}')


def select(lefts, query):
    """Binary search monotone suffix support; query every emitted branch exactly.

    Memory bounded by the caller's chunk. Empty context maps to zero, and is
    never queried as the global corpus histogram. A separator in a natural
    Pile context cuts off Wikipedia context at that separator, preventing seams.
    """
    lefts = [s.rsplit(b'\x1e', 1)[-1][-1024:] for s in lefts]
    cache = {}
    def fetch(patterns):
        missing = list(dict.fromkeys(p for p in patterns if p and p not in cache))
        if missing:
            replies = query(missing)
            if len(replies) != len(missing):
                raise ValueError('response cardinality mismatch')
            cache.update(zip(missing, replies))
    fetch(lefts)
    lo = [len(s) if s and cache[s]['width'] else 0 for s in lefts]
    hi = [len(s)+1 if k else len(s) for s, k in zip(lefts, lo)]
    while any(b-a > 1 for a, b in zip(lo, hi)):
        probes = [(i, (a+b)//2) for i, (a, b) in enumerate(zip(lo, hi)) if b-a > 1]
        fetch([lefts[i][-k:] for i, k in probes])
        for i, k in probes:
            if cache[lefts[i][-k:]]['width']:
                lo[i] = k
            else:
                hi[i] = k
    fetch([s[-k:] for s, k in zip(lefts, lo) if k])
    rates = {}
    for k in (16, 64, 256, 1024):
        # Exact support queries, rather than inference from censored maxima.
        eligible = [s for s in lefts if len(s) >= k]
        fetch([s[-k:] for s in eligible])
        rates[str(k)] = dict(eligible=len(eligible), supported=sum(cache[s[-k:]]['width'] > 0 for s in eligible))
    result = []
    for s, k in zip(lefts, lo):
        row = cache[s[-k:]] if k else dict(width=0, eof=0, branches=[])
        pairs = row['branches']
        terminal = sum(c for b, c in pairs if b == 30)
        counts = [(b, c) for b, c in pairs if b != 30]
        total = sum(c for _, c in counts)
        entropy = -sum((c/total)*math.log(c/total) for _, c in counts) if total else 0.
        result.append(dict(context_bytes=k, censored=bool(k and k == len(s)),
            context_count=row['width'], eof=row['eof'], terminal_count=terminal,
            branches=pairs, support_size=total, distinct_branches=len(counts),
            entropy=entropy, coverage=bool(total)))
    return result, rates, len(cache)


def features(row):
    """Registered 260 float inputs; zero ALL inputs when nonterminal support is empty."""
    total = row['support_size']
    values = [0.]*260
    if total:
        for byte, count in row['branches']:
            if byte != 30:
                values[byte] = count/total
        values[256:] = [math.log1p(total), row['entropy'],
            float(row['distinct_branches'] > 1), math.log1p(row['context_bytes'])]
    return values


def source_windows(limit):
    """Frozen dense source-order training windows; verify every used document."""
    receipt = json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    offsets = Path(receipt['offsets_path'])
    if digest(offsets) != receipt['offsets_sha256']:
        raise ValueError('dense offset identity changed')
    source = Path(receipt['source'])
    st = source.stat()
    if (st.st_size, st.st_ino, st.st_mtime_ns) != (receipt['source_size'], receipt['source_inode'], receipt['source_mtime_ns']):
        raise ValueError('dense source identity changed')
    n = 0
    with source.open('rb') as raw, offsets.open() as index:
        for line in index:
            doc = json.loads(line)
            if doc['split'] != 'train':
                continue
            raw.seek(doc['byte_offset'])
            data = raw.read(doc['byte_length'])
            if hashlib.sha256(data).hexdigest() != doc['sha256']:
                raise ValueError('document hash mismatch')
            for start in range(0, len(data)-1, 128):
                length = min(128, len(data)-1-start)
                yield doc['source_index'], start, [data[start:start+i] for i in range(1, length+1)]
                n += 1
                if limit is not None and n >= limit:
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, required=True, help='Explicit bounded benchmark, never a full-cache claim')
    parser.add_argument('--jobs', type=int, default=32)
    args = parser.parse_args()
    if args.windows < 1:
        raise ValueError('windows must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    server = BranchServer(args.output/'native.stderr', jobs=args.jobs)
    distribution = Counter()
    rates = {str(k): dict(eligible=0, supported=0) for k in (16,64,256,1024)}
    contexts = positions = windows = covered = terminal = censored = 0
    cache = args.output/'features.jsonl.gz'
    try:
        with gzip.open(cache, 'wt') as out:
            chunk = []
            def flush():
                nonlocal contexts, positions, windows, covered, terminal, censored
                lefts = [s for _, _, ss in chunk for s in ss]
                rows, measured, queries = select(lefts, server.query)
                contexts += queries
                at = 0
                for doc, start, ss in chunk:
                    for pos, row in enumerate(rows[at:at+len(ss)], 1):
                        out.write(json.dumps(dict(document=doc, start=start, position=pos, **row), separators=(',', ':'))+'\n')
                        positions += 1
                        covered += row['coverage']
                        terminal += row['terminal_count'] > 0
                        censored += row['censored']
                        distribution[row['context_bytes']] += 1
                    at += len(ss)
                    windows += 1
                for k in rates:
                    for key in rates[k]:
                        rates[k][key] += measured[k][key]
                print(json.dumps(dict(windows=windows, positions=positions, seconds=time.monotonic()-begin)), flush=True)
            for window in source_windows(args.windows):
                chunk.append(window)
                if len(chunk) == 128:
                    flush()
                    chunk = []
            if chunk:
                flush()
    finally:
        server.close()
    elapsed = time.monotonic()-begin
    summary = dict(status='passed', scope='bounded training prefix benchmark; NOT full train/valid cache',
        windows=windows, positions=positions, seconds=elapsed, features_per_second=positions/elapsed,
        native_queries=server.ident, distinct_queries_in_chunks=contexts, native_calls=server.calls,
        jobs=args.jobs, cpu_affinity_count=len(os.sched_getaffinity(0)),
        native_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
        coverage_positions=covered, terminal_positions=terminal, censored_positions=censored,
        maximal_context_histogram=dict(sorted(distribution.items())), support=rates,
        feature_sha256=digest(cache), feature_bytes=cache.stat().st_size, xsa_sha256=digest(XSA),
        source_span_sha256=digest(ROOT/'receipts/c14_dense/pile_span.json'),
        eval_construction_sha256=digest(ROOT/'receipts/c14_dense/eval_construction.json'),
        feature_geometry='256 probabilities + log1p nonterminal outgoing count + entropy + fork bit + log1p context length',
        terminal_policy='byte 30 measured, excluded from probabilities; context cannot span Wikipedia separator',
        gpu_used=False, materialized_corpus_bytes=0, python=__import__('sys').executable,
        phase1_comparison='selected token-boundary interiors versus source-order raw-byte prefix; descriptive, different sampling')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

if __name__ == '__main__':
    main()
