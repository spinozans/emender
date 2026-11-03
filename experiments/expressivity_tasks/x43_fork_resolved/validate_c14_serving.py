"""Replay bounded serving-cache arithmetic and immutable dense dependencies."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
from .c14_branch_serve import ROOT, XSA, features
from .c14_feature_extract import digest


def validate(directory):
    summary = json.loads((directory/'summary.json').read_text())
    assert summary['scope'] == 'bounded training prefix benchmark; NOT full train/valid cache'
    assert digest(directory/'features.jsonl.gz') == summary['feature_sha256']
    assert digest(XSA) == summary['xsa_sha256']
    assert digest(ROOT/'receipts/c14_dense/pile_span.json') == summary['source_span_sha256']
    assert digest(ROOT/'receipts/c14_dense/eval_construction.json') == summary['eval_construction_sha256']
    construction = json.loads((ROOT/'receipts/c14_dense/eval_construction.json').read_text())
    assert digest(construction['windows_path']) == construction['windows_sha256']
    histogram = Counter(); n = covered = terminal = censored = windows = 0
    previous = None
    strata = {k: Counter() for k in ('window_reset_1_16', 'window_interior_17_128')}
    documents = set()
    with gzip.open(directory/'features.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            ident = (row['document'], row['start'])
            if ident != previous:
                assert row['position'] == 1
                windows += 1
                previous = ident
                position = 0
            assert row['position'] == position+1
            position = row['position']
            assert 1 <= position <= 128 and row['start'] % 128 == 0
            assert 0 <= row['context_bytes'] <= position
            pairs = row['branches']
            assert len({b for b,c in pairs}) == len(pairs)
            assert all(0 <= b < 256 and c > 0 for b,c in pairs)
            assert sum(c for _,c in pairs)+row['eof'] == row['context_count']
            assert row['terminal_count'] == sum(c for b,c in pairs if b == 30)
            continuation = [(b,c) for b,c in pairs if b != 30]
            total = sum(c for _,c in continuation)
            assert row['support_size'] == total
            assert row['distinct_branches'] == len(continuation)
            assert row['coverage'] == bool(total)
            entropy = -sum(c/total*math.log(c/total) for _,c in continuation) if total else 0.
            assert abs(entropy-row['entropy']) < 1e-12
            values = features(row)
            assert values[30] == 0 and len(values) == 260
            assert abs(sum(values[:256])-float(bool(total))) < 1e-12
            if not total:
                assert values == [0.]*260
            n += 1
            covered += row['coverage']
            terminal += row['terminal_count'] > 0
            censored += row['censored']
            histogram[row['context_bytes']] += 1
            documents.add(row['document'])
            stratum = 'window_reset_1_16' if position <= 16 else 'window_interior_17_128'
            strata[stratum].update(positions=1, covered=bool(total), full_history_supported=(row['context_bytes'] == position), context_bytes=row['context_bytes'])
    assert (n, windows, covered, terminal, censored) == tuple(summary[k] for k in
        ('positions','windows','coverage_positions','terminal_positions','censored_positions'))
    assert {str(k):v for k,v in histogram.items()} == summary['maximal_context_histogram']
    return dict(status='passed', scope=summary['scope'], positions=n, windows=windows,
        cache_sha256=summary['feature_sha256'], immutable_dense_eval_windows_sha256=construction['windows_sha256'],
        paired_windows_changed=False, full_training_cache=False, full_validation_cache=False,
        documents=len(documents), within_window_coverage={k:dict(v) for k,v in strata.items()},
        note='Arithmetic replay and dependency hashes; native-query correctness is separately fixture-validated')


def main():
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path)
    args = p.parse_args()
    result = validate(args.directory)
    (args.directory/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
