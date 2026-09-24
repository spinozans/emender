#!/usr/bin/env python3
"""Read-only aggregate audit of sealed analysis authority target starts."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import tiktoken
from ndm.e97_agent_protocol import E97_PI_AGENT_ANALYSIS_SYSTEM_V1
from ndm.e97_atomic import publish_bytes_no_replace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_bytes = (args.authority / 'manifest.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == args.manifest_sha256
    manifest = json.loads(manifest_bytes)
    paths = {}
    for key in ('tokens', 'mask', 'index', 'metadata'):
        d = manifest['outputs'][key]
        assert Path(d['path']).name == d['path']
        path = args.authority / d['path']
        assert path.stat().st_size == d['bytes']
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == d['sha256']
        paths[key] = path
    tokens = np.memmap(paths['tokens'], dtype='<u4', mode='r')
    masks = np.memmap(paths['mask'], dtype='u1', mode='r')
    index = paths['index'].read_bytes()
    row_index = struct.Struct('<QQQB7x')
    encoding = tiktoken.get_encoding('p50k_base')
    expected_system = 'System:\n' + E97_PI_AGENT_ANALYSIS_SYSTEM_V1 + '\n\nUser:\n'
    counts = Counter()
    source_targets = Counter()
    with paths['metadata'].open() as metadata:
        for number, line in enumerate(metadata):
            row = json.loads(line)
            offset, length, target_count, split = row_index.unpack_from(index, number * row_index.size)
            assert length == row['tokens'] and target_count == row['targets']
            source_targets[row['source']] += target_count
            counts['total_records'] += 1
            if row['source'] != 'agent':
                continue
            counts['agent_records'] += 1
            record = tokens[offset:offset+length]
            mask = masks[offset:offset+length]
            assert int(mask.sum()) == target_count
            assert set(np.unique(mask).tolist()) <= {0, 1}
            prefix = encoding.decode(record[:min(length, 512)].tolist())
            if prefix.startswith(expected_system):
                counts['agent_records_exact_analysis_system'] += 1
            starts = np.flatnonzero((mask == 1) & np.concatenate(([True], mask[:-1] == 0)))
            for start in starts:
                text = encoding.decode(record[start:min(start+16, length)].tolist())
                counts['target_run_starts'] += 1
                if text.startswith('Analysis: '):
                    counts['analysis_target_run_starts'] += 1
                    tail = encoding.decode(record[max(0, start-16):start].tolist())
                    if tail.endswith('Assistant:\n'):
                        counts['analysis_starts_after_assistant_header'] += 1
                    if start == starts[0]:
                        history = encoding.decode(record[:start].tolist())
                        if encoding.encode_ordinary(history) == record[:start].tolist():
                            counts['first_analysis_prefix_tokenization_exact'] += 1
                elif text == '\x1e':
                    counts['record_separator_only_starts'] += 1
                else:
                    counts['unexpected_target_run_starts'] += 1
    passed = (counts['agent_records'] > 0
              and counts['agent_records_exact_analysis_system'] == counts['agent_records']
              and counts['unexpected_target_run_starts'] == 0
              and counts['analysis_target_run_starts'] == counts['analysis_starts_after_assistant_header']
              and counts['first_analysis_prefix_tokenization_exact'] == counts['agent_records'])
    result = {'schema': 'emender-e97-analysis-training-boundary-audit-v1',
              'status': 'passed' if passed else 'failed', 'counts': dict(counts),
              'source_authority_targets': dict(source_targets),
              'authority_manifest_sha256': args.manifest_sha256,
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'system_prompt_sha256': hashlib.sha256(E97_PI_AGENT_ANALYSIS_SYSTEM_V1.encode()).hexdigest(),
              'scope': 'agent target-run starts, system bytes and first-target prefix tokenization only; not a loss, gradient, full-turn serialization or capability audit'}
    publish_bytes_no_replace(args.output, (json.dumps(result, indent=2, sort_keys=True)+'\n').encode(), mode=0o600)
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
