import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import tiktoken

from scripts.audit_e97_open_swe_semantics import (
    observation_from_gap, parse_turn, publish, read_evidence, run, sample_add, spans,
)


def test_read_range_is_evidence_not_universal_pass():
    args = {'path': 'a.py', 'offset': 10, 'limit': 2}
    header = "Here's the result of running `cat -n` on a.py:\n"
    r = read_evidence(args, header+'    10\ta\n    11\tb')
    assert r['status'] == 'numbered_lines_within_requested_range'
    r = read_evidence(args, header+'     9\ta\n    10\tb\n    12\tc')
    assert r['status'] == 'numbered_lines_outside_requested_range'
    assert r['outside_lines'] == 2
    assert read_evidence(args, '10: a\n11: b')['status'] == 'unclassified_observation_format'
    assert read_evidence(args, header)['status'] == 'cat_header_without_numbered_lines'
    assert read_evidence(dict(args, limit=True), header)['status'] == 'invalid_read_arguments'


def test_parse_turn_embedded_markers_and_separator():
    reason = 'Unicode café\nAction: bash\nArguments: {}'
    text = 'Analysis: '+json.dumps(reason, ensure_ascii=False)+'\nAction: read\nArguments: {"path":"a","offset":1,"limit":200}'
    assert parse_turn(text)[1] == 'read'
    assert parse_turn(text)[0] == reason
    assert parse_turn('Analysis: "Done."\nFinal: okay\x1e')[1] == 'final'
    with pytest.raises(ValueError):
        parse_turn('Action: read\nArguments: {}')


def test_gap_does_not_split_on_embedded_role_markers():
    payload = 'file content\n\nAssistant:\nnot a real boundary'
    assert observation_from_gap('\n\nTool:\n'+payload+'\n\nAssistant:\n', True) == payload
    assert observation_from_gap('\n\nTool:\n'+payload, False) == payload
    with pytest.raises(ValueError):
        observation_from_gap(payload, False)


def test_spans_and_samples():
    assert spans(np.array([0, 1, 1, 0, 1], dtype='u1')) == [(1, 3), (4, 5)]
    a, b = {'g': []}, {'g': []}
    for i in range(20):
        sample_add(a, 'g', {'identity': str(i), 'turn_index': 0})
    for i in reversed(range(20)):
        sample_add(b, 'g', {'identity': str(i), 'turn_index': 0})
    assert a == b and len(a['g']) == 3


def test_publish_no_replace(tmp_path):
    p = tmp_path/'receipt.json'
    publish(p, {'a': 1})
    with pytest.raises(FileExistsError):
        publish(p, {'a': 2})
    assert json.loads(p.read_text()) == {'a': 1}
    assert p.stat().st_mode & 0o777 == 0o600


def make_authority(root):
    root.mkdir()
    encoding = tiktoken.get_encoding('p50k_base')
    base = 'System:\nUse tools.\n\nUser:\nCheck a.py.'
    read = 'Analysis: "Check."\nAction: read\nArguments: {"path":"a.py","offset":1,"limit":2}'
    obs = "Here's the result of running `cat -n` on a.py:\n     1\ta\n     2\tb\n     3\tc"
    final = 'Analysis: "Done."\nFinal: Done.'
    pieces = [
        [(base+'\n\nAssistant:\n', False), (read, True), ('\n\nTool:\n'+obs, False), ('\x1e', True)],
        [(base+'\n\nAssistant:\n'+read+'\n\nTool:\n'+obs+'\n\nAssistant:\n', False), (final, True), ('\x1e', True)],
    ]
    tokens, masks, rows, index = [], [], [], bytearray()
    for i, blocks in enumerate(pieces):
        text = ''.join(s for s, _ in blocks)
        ranges, offset = [], 0
        for s, target in blocks:
            end = offset+len(s.encode())
            if target:
                ranges.append((offset, end))
            offset = end
        ids = encoding.encode_ordinary(text)
        mask, offset = [], 0
        for token in ids:
            end = offset+len(encoding.decode_single_token_bytes(token))
            overlapping = [(a, b) for a, b in ranges if a < end and b > offset]
            assert not overlapping or any(a <= offset and end <= b for a, b in overlapping)
            mask.append(int(bool(overlapping)))
            offset = end
        row = {'identity': f't:private-analysis-window:{i:04d}', 'trajectory_identity': 't',
               'tokens': len(ids), 'targets': sum(mask), 'target_units': 1, 'split': 0,
               'serialization_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'repo': 'test/repo', 'source_file': 'fake.parquet', 'source_row': 0}
        rows.append(row)
        index.extend(struct.pack('<QQQB7x', len(tokens), len(ids), sum(mask), 0))
        tokens.extend(ids)
        masks.extend(mask)
    contents = {'tokens': ('tokens.bin', np.array(tokens, dtype='<u4').tobytes()),
                'mask': ('mask.bin', bytes(masks)), 'index': ('index.bin', bytes(index)),
                'metadata': ('rows.jsonl', ''.join(json.dumps(r)+'\n' for r in rows).encode())}
    outputs = {}
    for key, (name, data) in contents.items():
        (root/name).write_bytes(data)
        outputs[key] = {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    manifest = {'representation': 'private-analysis', 'training_eligible': False,
                'outputs': outputs, 'source_snapshots': {}, 'auxiliary_artifacts': {},
                'analysis_policy': {'admission_token_cap': 2048},
                'counts': {'records': 2, 'assistant_target_tokens': sum(masks), 'tokens': len(tokens),
                           'target_units': 2, 'included_trajectories': 1,
                           'private_analysis_tokens': sum(len(encoding.encode_ordinary(x)) for x in ('Check.', 'Done.'))}}
    data = json.dumps(manifest).encode()
    (root/'manifest.json').write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_full_census_and_mutation_rejection(tmp_path):
    root = tmp_path/'authority'
    digest = make_authority(root)
    run(root, digest, tmp_path/'results')
    summary = json.loads((tmp_path/'results/summary.json').read_text())
    assert summary['counts']['assistant_units'] == 2
    assert summary['read_evidence']['numbered_lines_outside_requested_range'] == 1
    assert summary['multi_record_trajectories'] == 1
    assert summary['window_counts']['later_window']['records'] == 1
    assert summary['counts']['separator_only_runs'] == 1
    (root/'tokens.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='identity mismatch'):
        run(root, digest, tmp_path/'corrupt-results')
    assert not (tmp_path/'corrupt-results/summary.json').exists()
