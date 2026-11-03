"""Native 100GB regression probes and independent full-byte fixture oracle."""
import argparse
from collections import Counter
import json
import random
from pathlib import Path
import subprocess
from .c14_branch_serve import BranchServer, XSA, ROOT
from .c14_feature_extract import digest


def expected(corpus, pattern):
    counts = Counter(); eof = width = 0
    at = corpus.find(pattern)
    while at >= 0:
        width += 1
        end = at+len(pattern)
        if end == len(corpus):
            eof += 1
        else:
            counts[corpus[end]] += 1
        at = corpus.find(pattern, at+1)
    return width, eof, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repro = (ROOT/'receipts/c14_bridge_fixed/repro.patterns').read_bytes().split(b'\x1e')[:-1]
    patterns = repro+[b'the ', b'-\x1e\x00\xff\n', b'\x00\x1e']
    server = BranchServer(args.output/'regression.stderr')
    try:
        rows = server.query(patterns)
    finally:
        server.close()
    assert [r['width'] for r in rows[:3]] == [0, 0, 2493]
    assert rows[3]['width'] == 313603799 and len(rows[3]['branches']) == 137
    (args.output/'regression.json').write_text(json.dumps(dict(status='passed',
        probes=[dict(pattern_hex=p.hex(), **r) for p,r in zip(patterns, rows)],
        xsa_sha256=digest(XSA), corpus_reads_at_query_time=False), indent=2)+'\n')
    args.scratch.mkdir(parents=True, exist_ok=True)
    # Index format reserves an absent byte for END. Exercise all 256 query
    # bytes against a corpus with 255 present symbols, including NUL/30/255.
    alphabet = bytes(b for b in range(256) if b != 254)
    rng = random.Random(3401)
    corpus = bytes(rng.choice(alphabet) for _ in range(1_200_000))+b'-\x00\x1e\xff\nend'
    raw = args.scratch/'fixture.bin'
    raw.write_bytes(corpus)
    command = [str(XSA), 'build', '--tape', '--input', str(raw), '--scratch',
        str(args.scratch/'index'), '--threads', '4', '--memory-gb', '16',
        '--timeout', '600', '--no-chi']
    with (args.output/'fixture_build.stdout').open('wb') as out, (args.output/'fixture_build.stderr').open('wb') as err:
        proc = subprocess.run(command, stdout=out, stderr=err)
    if proc.returncode:
        raise RuntimeError(f'fixture build failed: {proc.returncode}')
    patterns = [bytes([b]) for b in range(256)]+[b'-\x00\x1e\xff\n', b'\x00\x1e', b'\xff\nend', b'absent', corpus[-128:]]
    server = BranchServer(args.output/'fixture.stderr', jobs=4, index=args.scratch/'index/fixture.sxi2')
    try:
        rows = server.query(patterns)
    finally:
        server.close()
    for pattern, row in zip(patterns, rows):
        width, eof, counts = expected(corpus, pattern)
        assert (row['width'], row['eof'], Counter(dict(row['branches']))) == (width, eof, counts), pattern.hex()
    (args.output/'fixture.json').write_text(json.dumps(dict(status='passed',
        patterns=len(patterns), all_256_query_bytes=True, present_symbols=255,
        missing_byte=254, corpus_sha256=digest(raw),
        xsa_sha256=digest(XSA), index_sha256=digest(args.scratch/'index/fixture.sxi2'),
        oracle='independent overlapping bytes.find next-byte histogram; all branches, width and EOF exact',
        special_cases='NUL, byte30, ff, newline, leading dash, absent, EOF'), indent=2)+'\n')

if __name__ == '__main__':
    main()
