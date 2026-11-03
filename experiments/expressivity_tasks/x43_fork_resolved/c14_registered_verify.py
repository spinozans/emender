"""Verify resumable cache hashes, full banked-oracle prefix and native fixtures."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from .c14_assembly import assemble
from .c14_branch_serve import BranchServer, ROOT, XSA, features, select
from .c14_cache_scale import read_windows
from .c14_feature_extract import digest
from .c14_registered_cache import checked_resume
from .c14_serving_verify import expected


def verify(cache, output):
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(s) for s in (cache/'journal.jsonl').read_text().splitlines()]
    shards = [r for r in records if r['event'] == 'shard_complete']
    identity = records[0]['identity']
    windows, positions = checked_resume(cache, identity)
    reference = Path('/home/erikg/c14_consultation_features/serving_benchmark_8192/features.jsonl.gz')
    binding = json.loads((ROOT/'receipts/c14_cache_scale/parity.json').read_text())
    if digest(reference) != binding['reference_sha256']:
        raise ValueError('oracle identity changed')
    count = 0
    old_hash = hashlib.sha256(); new_hash = hashlib.sha256()
    with gzip.open(reference, 'rt') as oracle:
        for shard in shards:
            with (cache/shard['file']).open('rb') as source:
                for doc, start, rows in read_windows(source):
                    if count >= binding['positions']:
                        break
                    old_rows = []
                    for position, row in enumerate(rows, 1):
                        old = json.loads(oracle.readline())
                        actual = json.loads(json.dumps(dict(document=doc, start=start, position=position, **row)))
                        if actual != old:
                            raise ValueError(f'oracle mismatch at {count}')
                        old_rows.append(old)
                        count += 1
                    import numpy as np
                    old_hash.update(np.asarray([features(r) for r in old_rows], dtype='<f8').tobytes())
                    new_hash.update(assemble(rows).tobytes())
            if count >= binding['positions']:
                break
    if count != binding['positions'] or old_hash.hexdigest() != new_hash.hexdigest():
        raise ValueError('full oracle feature parity incomplete')
    fixture = Path('/home/erikg/emender-scratch/c14-serving-fixture-random')
    corpus = (fixture/'fixture.bin').read_bytes()
    frozen = json.loads((ROOT/'receipts/c14_assembly/fixture.json').read_text())
    if (digest(fixture/'fixture.bin') != frozen['corpus_sha256'] or
        digest(fixture/'index/fixture.sxi2') != frozen['index_sha256'] or
        digest(XSA) != frozen['xsa_sha256']):
        raise ValueError('fixture/native identity changed')
    patterns = [bytes([b]) for b in range(256)]+[b'-\x00\x1e\xff\n', b'\x00\x1e', b'\xff\nend', b'absent', corpus[-128:]]
    server = BranchServer(output/'fixture.stderr', jobs=4, index=fixture/'index/fixture.sxi2')
    try:
        native = server.query(patterns)
        selected, _, _ = select(patterns, server.query)
    finally:
        server.close()
    oracle_selected = []
    for pattern, row in zip(patterns, native):
        width, eof, counts = expected(corpus, pattern)
        if (row['width'], row['eof'], Counter(dict(row['branches']))) != (width, eof, counts):
            raise ValueError('native fixture mismatch')
        s = pattern.rsplit(b'\x1e', 1)[-1][-1024:]
        k = max((k for k in range(1, len(s)+1) if s[-k:] in corpus), default=0)
        width, eof, counts = expected(corpus, s[-k:]) if k else (0, 0, Counter())
        # Assemble independent sparse oracle rows using exact byte histograms.
        import math
        total = sum(c for b, c in counts.items() if b != 30)
        entropy = -sum(c/total*math.log(c/total) for b, c in sorted(counts.items()) if b != 30) if total else 0.
        oracle_selected.append(dict(branches=sorted(counts.items()), support_size=total,
            entropy=entropy, distinct_branches=sum(b != 30 for b in counts), context_bytes=k))
    if assemble(selected).tobytes() != assemble(oracle_selected).tobytes():
        raise ValueError('native fixture feature bits differ')
    result = dict(status='passed', cached_windows=windows, cached_positions=positions,
        exact_oracle_positions=count, oracle_feature_sha256=old_hash.hexdigest(),
        cache_feature_sha256=new_hash.hexdigest(), independent_native_fixture_patterns=len(patterns),
        fixture_selected_feature_bits_exact=True, journal_sha256=digest(cache/'journal.jsonl'),
        runner_sha256=identity['runner_sha256'], complete_emender_cache=False,
        no_training_started=True, scope='partial Emender prefix verification, not full C14 acceptance')
    (output/'parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(a.cache, a.output)
