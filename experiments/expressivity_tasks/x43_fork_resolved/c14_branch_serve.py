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

SPINE_CAP = 1024
"""Registered left-context cap in bytes; the fixed 1024-byte spine."""

SEPARATOR = b'\x1e'
"""The Pile's framing byte. Content, never a boundary: at vocabulary 256 no
in-band delimiter is safe by construction (SXGC_DOCUMENT_BOUNDARIES_NOTE.md)."""

SPINE_POLICY_REGISTERED = 'registered-1024'
SPINE_POLICY_SHORT = 'amended-short-8-16'
SPINE_POLICY = SPINE_POLICY_REGISTERED
SHORT_SPINE_BYTES = (8, 16)


def spine_left(policy=None):
    """Left-context cap in bytes, resolved from the spine policy flag.

    `registered-1024` is the registered feature definition and the default: every
    registered board's numbers depend on the fixed 1024-byte spine. The
    2026-11-10 short-spine amendment (RESULT_CONSULT_BOUNDARY_HARDENING.md) sets
    the flag to `amended-short-8-16` — an 8..16-byte spine with support-weighted
    relax. That policy is NOT implemented and refuses here rather than silently
    training a different feature: activation needs the relax implementation and a
    new registration.
    """
    policy = SPINE_POLICY if policy is None else policy
    if policy == SPINE_POLICY_REGISTERED:
        return SPINE_CAP
    if policy == SPINE_POLICY_SHORT:
        raise NotImplementedError('short-spine amendment ' + repr(SHORT_SPINE_BYTES) +
            ' has no implementation; no registered board may consume it without a new registration')
    raise ValueError('unknown spine policy ' + repr(policy))


class EndBranch:
    """Witness sentinel: the position's own occurrence ends the corpus (END branch)."""
    __slots__ = ()

    def __repr__(self):
        return 'EndBranch()'


END_BRANCH = EndBranch()


def consult_context(text, base, off, left=None):
    """The consult context for the position at absolute corpus offset `off`.

    `text` is the document's bytes starting at `base` — the window manifest's
    `byte_offset` for that document. `left` defaults to the spine policy flag
    (`spine_left()`). A caller that has already trimmed at a
    stricter in-document floor (a window start) passes that floor as `base` and
    the bytes from it; the function never reads outside `text`.

    The context is `corpus[max(base, off - left) .. off]`, so it can never leave
    the document, and a document shorter than `left` contributes only its own
    bytes. The last 0x1E is then cut as a belt-and-braces extra that can only
    shorten: on the pile it does shorten inside the documents carrying an
    internal 0x1E (6,979 in the spine region, 2,883 in the C14 region), which is
    a deliberate conservative loss and never a boundary. `off == base` is the
    document's first byte and yields the empty context, whose histogram is empty
    by rule.
    """
    if not isinstance(text, bytes):
        raise TypeError('document bytes must be bytes')
    left = spine_left() if left is None else left
    if left < 1:
        raise ValueError('left must be positive')
    if base < 0 or off < base:
        raise ValueError('invalid document geometry')
    depth = off - base
    if depth > len(text):
        raise ValueError('document bytes do not reach the query position')
    return text[max(0, depth-left):depth].rsplit(SEPARATOR, 1)[-1]


def exclude_occurrence(reply, next_byte, count=1):
    """`reply` with the query position's own occurrence removed, as arithmetic.

    `next_byte` is what that occurrence contributed: an int 0..255 for a content
    branch, or END_BRANCH for the END branch. `count` is how many times it was
    counted (1 per container position). Being pure arithmetic on a raw histogram,
    it can be applied BEFORE any support or width test.

    A witness absent from the raw counts is an error, never a silent no-op: a
    skipped exclusion is leakage, not a smaller count. The emitted branch order is
    kept; a branch whose count reaches zero is dropped.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('count must be a positive integer')
    width = reply['width'] - count
    if width < 0:
        raise ValueError('source occurrence absent from the raw counts')
    if next_byte is END_BRANCH:
        eof = reply['eof'] - count
        if eof < 0:
            raise ValueError('source occurrence absent from the raw counts')
        branches = list(reply['branches'])
    else:
        if not isinstance(next_byte, int) or isinstance(next_byte, bool) or not 0 <= next_byte <= 255:
            raise TypeError('next_byte must be a byte or END_BRANCH')
        if dict(reply['branches']).get(next_byte, 0) < count:
            raise ValueError('source occurrence absent from the raw counts')
        eof = reply['eof']
        branches = []
        for byte, counted in reply['branches']:
            if byte != next_byte:
                branches.append((byte, counted))
            elif counted > count:
                branches.append((byte, counted-count))
    return dict(width=width, eof=eof, branches=branches)


def is_source_occurrence(hit, doc_id, offset_in_doc):
    """The filter rule that attributes a located hit to the query position.

    `xsa query <container> <context> --docs DIR` returns one row per occurrence
    carrying `doc_id` (0-based container document id) and `offset_in_doc` (the
    occurrence's byte offset inside that document). A row IS the query position's
    own occurrence iff BOTH equal the query position's own pair, `hit['doc_id'] ==
    doc_id and hit['offset_in_doc'] == offset_in_doc`; a container position is
    unique, so the pair identifies at most one row. Other occurrences inside the
    same document are NOT the position's own occurrence and stay in the counts
    (removing those is `--doc-restrict`, a different question). A hit missing
    either field raises rather than quietly failing to exclude.
    """
    return hit['doc_id'] == doc_id and hit['offset_in_doc'] == offset_in_doc


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


def select(lefts, query, *, spans=None, excludes=None, left=None):
    """Binary search monotone suffix support; query every emitted branch exactly.

    Memory bounded by the caller's chunk. Empty context maps to zero, and is
    never queried as the global corpus histogram.

    Two opt-in hardenings, both OFF by default so every registered board keeps
    its recorded numbers (RESULT_CONSULT_BOUNDARY_HARDENING.md):

    `left` is the context cap, resolved from the spine policy flag when None.

    `spans` — a parallel sequence of `(base, off)` container offsets. With it,
    `lefts[i]` is the document's bytes from `base` and the context is cut by
    `consult_context` at the document base instead of scanning for 0x1E.
    `excludes` — a parallel sequence of the query position's own continuation:
    an int byte, END_BRANCH, or None when the position's document is known to be
    outside the index by span geometry. Excluded counts are applied BEFORE every
    support and width test, so `context_bytes` is the longest depth whose
    POST-exclusion width is nonzero.
    """
    left = spine_left() if left is None else left
    if spans is None:
        contexts = [s.rsplit(SEPARATOR, 1)[-1][-left:] for s in lefts]
    else:
        if len(spans) != len(lefts):
            raise ValueError('span cardinality mismatch')
        contexts = [consult_context(s, base, off, left) for s, (base, off) in zip(lefts, spans)]
    if excludes is not None and len(excludes) != len(contexts):
        raise ValueError('exclusion cardinality mismatch')
    cache = {}
    def fetch(patterns):
        missing = list(dict.fromkeys(p for p in patterns if p and p not in cache))
        if missing:
            replies = query(missing)
            if len(replies) != len(missing):
                raise ValueError('response cardinality mismatch')
            cache.update(zip(missing, replies))
    def histogram(i, k):
        # Per position, because the exclusion belongs to the position and not to
        # the shared pattern: the same context can be supported for one position
        # and source-only for another.
        if not k:
            return dict(width=0, eof=0, branches=[])
        row = cache[contexts[i][-k:]]
        if excludes is None or excludes[i] is None:
            return row
        return exclude_occurrence(row, excludes[i])
    fetch(contexts)
    lo = [len(s) if s and histogram(i, len(s))['width'] else 0 for i, s in enumerate(contexts)]
    hi = [len(s)+1 if k else len(s) for s, k in zip(contexts, lo)]
    while any(b-a > 1 for a, b in zip(lo, hi)):
        probes = [(i, (a+b)//2) for i, (a, b) in enumerate(zip(lo, hi)) if b-a > 1]
        fetch([contexts[i][-k:] for i, k in probes])
        for i, k in probes:
            if histogram(i, k)['width']:
                lo[i] = k
            else:
                hi[i] = k
    fetch([s[-k:] for s, k in zip(contexts, lo) if k])
    rates = {}
    for k in (16, 64, 256, 1024):
        # Exact support queries, rather than inference from censored maxima.
        eligible = [i for i, s in enumerate(contexts) if len(s) >= k]
        fetch([contexts[i][-k:] for i in eligible])
        rates[str(k)] = dict(eligible=len(eligible), supported=sum(histogram(i, k)['width'] > 0 for i in eligible))
    result = []
    for i, (s, k) in enumerate(zip(contexts, lo)):
        row = histogram(i, k)
        pairs = row['branches']
        terminal = sum(c for b, c in pairs if b == 30)
        counts = [(b, c) for b, c in pairs if b != 30]
        total = sum(c for _, c in counts)
        entropy = -sum((c/total)*math.log(c/total) for _, c in counts) if total else 0.
        entry = dict(context_bytes=k, censored=bool(k and k == len(s)),
            context_count=row['width'], eof=row['eof'], terminal_count=terminal,
            branches=pairs, support_size=total, distinct_branches=len(counts),
            entropy=entropy, coverage=bool(total))
        if excludes is not None:
            # Absent means the counts are the raw histogram, as registered.
            entry['exclude_source'] = True
        result.append(entry)
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
