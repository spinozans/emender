#!/usr/bin/env python3
"""Read-only, source-bound census of converted Open-SWE action/observation risks.

No model calls, tool execution, authority mutation, or training admission.
Samples use deterministic bottom-k hashes, never model results.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

import numpy as np
import tiktoken

INDEX = struct.Struct('<QQQB7x')
RS = '\x1e'
CAT_HEADER = re.compile(r"^Here's the result of running `cat -n` on [^\n]+:\n")
NUMBERED_LINE = re.compile(r'^ *([0-9]+)\t', re.MULTILINE)
SEED = 'e97-open-swe-semantic-census-v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def publish(path, obj):
    path = Path(path)
    payload = (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+'\n').encode()
    fd, name = tempfile.mkstemp(prefix='.partial-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(name, path)  # no replacement, even on concurrent publication
        d = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        os.unlink(name)


def parse_turn(text):
    if text.endswith(RS):
        text = text[:-1]
    line, sep, body = text.partition('\n')
    if not sep or not line.startswith('Analysis: '):
        raise ValueError('missing analysis line')
    encoded = line[len('Analysis: '):]
    reasoning = json.loads(encoded)
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError('invalid analysis string')
    if json.dumps(reasoning, ensure_ascii=False, separators=(',', ':')) != encoded:
        raise ValueError('noncanonical analysis string')
    if body.startswith('Final: '):
        return reasoning, 'final', None
    action, sep, arguments = body.partition('\nArguments: ')
    if not sep or not action.startswith('Action: '):
        raise ValueError('missing action/arguments')
    tool = action[len('Action: '):]
    if tool not in {'read', 'bash', 'edit', 'write'}:
        raise ValueError('unsupported converted tool')
    args = json.loads(arguments)
    if not isinstance(args, dict):
        raise ValueError('non-object arguments')
    return reasoning, tool, args


def observation_from_gap(gap, next_assistant):
    prefix = '\n\nTool:\n'
    if not gap.startswith(prefix):
        raise ValueError('action has no observation boundary')
    if next_assistant:
        suffix = '\n\nAssistant:\n'
        if not gap.endswith(suffix):
            raise ValueError('missing following assistant header')
        gap = gap[:-len(suffix)]
    return gap[len(prefix):]


def read_evidence(args, observation):
    """Classify evidence only; unknown formats do not count as passes."""
    if (set(args) != {'path', 'offset', 'limit'} or not isinstance(args['path'], str)
            or type(args['offset']) is not int or args['offset'] < 1
            or type(args['limit']) is not int or args['limit'] < 1):
        return {'status': 'invalid_read_arguments'}
    match = CAT_HEADER.match(observation)
    if not match:
        return {'status': 'unclassified_observation_format'}
    numbers = [int(m.group(1)) for m in NUMBERED_LINE.finditer(observation[match.end():])]
    if not numbers:
        return {'status': 'cat_header_without_numbered_lines'}
    low, high = args['offset'], args['offset'] + args['limit'] - 1
    outside = [n for n in numbers if not low <= n <= high]
    return {'status': 'numbered_lines_outside_requested_range' if outside else 'numbered_lines_within_requested_range',
            'requested_first': low, 'requested_last': high,
            'observed_first': numbers[0], 'observed_last': numbers[-1],
            'observed_min': min(numbers), 'observed_max': max(numbers),
            'numbered_lines': len(numbers), 'outside_lines': len(outside),
            'strictly_increasing': all(a < b for a, b in zip(numbers, numbers[1:]))}


def spans(mask):
    starts = np.flatnonzero((mask == 1) & np.r_[True, mask[:-1] == 0])
    ends = np.flatnonzero((mask == 1) & np.r_[mask[1:] == 0, True]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def sample_add(samples, group, row, k=3):
    key = hashlib.sha256((SEED+'\0'+row['identity']+'\0'+str(row['turn_index'])).encode()).hexdigest()
    samples[group].append((key, row))
    samples[group].sort(key=lambda x: x[0])
    del samples[group][k:]


def verify_file(root, desc):
    name = desc['path']
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('unsafe manifest path')
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('non-regular authority file')
    if path.stat().st_size != desc['bytes'] or sha(path) != desc['sha256']:
        raise ValueError(f'identity mismatch: {path}')
    return path


def run(authority, manifest_sha256, output):
    authority, output = Path(authority).resolve(), Path(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    mp = authority / 'manifest.json'
    if sha(mp) != manifest_sha256:
        raise ValueError('manifest identity mismatch')
    manifest = json.loads(mp.read_text())
    if manifest.get('representation') != 'private-analysis' or manifest.get('training_eligible') is not False:
        raise ValueError('expected non-trainable private-analysis candidate')
    descriptors = {**manifest['outputs'], **manifest['source_snapshots'],
                   **manifest['auxiliary_artifacts']}
    paths = {name: verify_file(authority, d) for name, d in descriptors.items()}
    identity = {'manifest_sha256': manifest_sha256, 'audit_sha256': sha(__file__),
                'files': descriptors, 'sampling_seed': SEED,
                'sample_policy': 'bottom 3 SHA256 identities per tool/window-start stratum and per read evidence status; split=0 only; bounded excerpts',
                'scope': 'full converted-record census; no raw-source or filesystem replay, no model inference',
                'training_eligible': False}
    publish(output/'identity.json', identity)
    print('SEMANTIC_AUDIT_INPUTS_VERIFIED', flush=True)
    encoding = tiktoken.get_encoding('p50k_base')
    tokens = np.memmap(paths['tokens'], dtype='<u4', mode='r')
    masks = np.memmap(paths['mask'], dtype='u1', mode='r')
    index = paths['index'].read_bytes()
    rows = [json.loads(line) for line in paths['metadata'].read_text().splitlines()]
    if len(index) != len(rows)*INDEX.size or len(tokens) != len(masks):
        raise ValueError('index/stream shape mismatch')
    counts, tools, read_status, limits = Counter(), Counter(), Counter(), Counter()
    repos, splits, windows = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    samples = defaultdict(list)
    trajectory_records, trajectory_targets = Counter(), Counter()
    offset_expected = 0
    for number, row in enumerate(rows):
        offset, length, target_count, split = INDEX.unpack_from(index, number*INDEX.size)
        if (offset != offset_expected or length != row['tokens'] or target_count != row['targets']
                or split != row['split'] or split not in {0, 1}):
            raise ValueError(f'index metadata mismatch: {number}')
        offset_expected += length
        record, mask = tokens[offset:offset+length], masks[offset:offset+length]
        if len(record) != length or int(mask.sum()) != target_count or np.any(mask > 1):
            raise ValueError('mask/count mismatch')
        text = encoding.decode(record.tolist(), errors='strict')
        if hashlib.sha256(text.encode()).hexdigest() != row['serialization_sha256']:
            raise ValueError('serialized record mismatch')
        window = int(row['identity'].rsplit(':', 1)[1])
        later = 'later_window' if window > 0 else 'first_window'
        trajectory = row['trajectory_identity']
        trajectory_records[trajectory] += 1
        trajectory_targets[trajectory] += target_count
        windows[later].update(records=1, targets=target_count, input_tokens=length)
        splits[str(split)].update(records=1, targets=target_count, input_tokens=length)
        repos[row['repo']].update(records=1, targets=target_count, input_tokens=length)
        counts.update(records=1, targets=target_count, input_tokens=length)
        ranges = spans(mask)
        record_units = 0
        for j, (start, end) in enumerate(ranges):
            target = encoding.decode(record[start:end].tolist(), errors='strict')
            if target == RS:
                counts['separator_only_runs'] += 1
                continue
            if not encoding.decode(record[max(0, start-16):start].tolist()).endswith('Assistant:\n'):
                raise ValueError('target header mismatch')
            reasoning, tool, args = parse_turn(target)
            analysis_tokens = len(encoding.encode_ordinary(reasoning))
            if analysis_tokens > manifest['analysis_policy']['admission_token_cap']:
                raise ValueError('analysis cap mismatch')
            counts['private_analysis_tokens'] += analysis_tokens
            tools[tool] += 1
            counts['assistant_units'] += 1
            record_units += 1
            observation = ''
            if tool != 'final':
                if j+1 >= len(ranges):
                    raise ValueError('action without trailing context/separator')
                next_start, next_end = ranges[j+1]
                is_separator = encoding.decode(record[next_start:next_end].tolist()) == RS
                gap = encoding.decode(record[end:next_start].tolist(), errors='strict')
                observation = observation_from_gap(gap, not is_separator)
            else:
                counts['final_turns'] += 1
            evidence = None
            if tool == 'read':
                evidence = read_evidence(args, observation)
                read_status[evidence['status']] += 1
                limits[str(args.get('limit'))] += 1
                if evidence.get('outside_lines', 0):
                    counts['read_observation_lines_outside_bounds'] += evidence['outside_lines']
            # Only train-split snippets are exposed for subsequent manual review.
            if split == 0:
                example = {'identity': row['identity'], 'trajectory_identity': trajectory,
                           'source_file': row['source_file'], 'source_row': row['source_row'],
                           'repo': row['repo'], 'window_index': window, 'turn_index': record_units-1,
                           'tool': tool, 'target_excerpt': target[:900],
                           'observation_head': observation[:900], 'observation_tail': observation[-400:],
                           'read_evidence': evidence}
                sample_add(samples, tool+'/'+later, example)
                if evidence:
                    sample_add(samples, 'read-status/'+evidence['status'], example)
            if window > 0 and record_units == 1:
                counts['later_window_first_decisions'] += 1
        if record_units != row['target_units']:
            raise ValueError('record target-unit count mismatch')
        if (number+1) % 1000 == 0:
            print(f'SEMANTIC_AUDIT_PROGRESS records={number+1}/{len(rows)} units={counts["assistant_units"]}', flush=True)
    if offset_expected != len(tokens):
        raise ValueError('unaccounted stream tail')
    checks = {'records': 'records', 'targets': 'assistant_target_tokens',
              'input_tokens': 'tokens', 'assistant_units': 'target_units',
              'private_analysis_tokens': 'private_analysis_tokens'}
    for actual, expected in checks.items():
        if counts[actual] != manifest['counts'][expected]:
            raise ValueError(f'aggregate mismatch: {actual}')
    if len(trajectory_records) != manifest['counts']['included_trajectories']:
        raise ValueError('trajectory identity count mismatch')
    # Verify inputs again: mutable historical filesystem permissions are not an identity guarantee.
    for desc in descriptors.values():
        verify_file(authority, desc)
    if sha(mp) != manifest_sha256:
        raise ValueError('manifest changed during audit')
    split_trajectories = {tid for tid, n in trajectory_records.items() if n > 1}
    result = {'schema': 'emender-open-swe-semantic-census-v1', 'status': 'completed',
              'identity': identity, 'counts': dict(counts), 'tools': dict(tools),
              'read_evidence': dict(read_status), 'read_limit_histogram': dict(limits),
              'window_counts': dict(windows), 'split_counts': dict(splits),
              'unique_trajectories': len(trajectory_records),
              'multi_record_trajectories': len(split_trajectories),
              'targets_in_multi_record_trajectories': sum(trajectory_targets[t] for t in split_trajectories),
              'repository_counts': dict(sorted(repos.items())),
              'sample_groups': {k: len(v) for k, v in samples.items()},
              'limitations': ['Line-range evidence is based on recognized source cat-n headers and numbered lines, not filesystem replay.',
                              'Within-range observations are not proof of complete semantic or runtime equivalence.',
                              'Window counts measure exposure to context resets, not proven causal dependency loss.',
                              'Source path normalization, omitted user messages, and raw-source fidelity require a subsequent source audit.',
                              'No source is newly admitted and no model capability is measured.']}
    publish(output/'samples.json', {k: [dict(v, sample_key=h) for h, v in values] for k, values in samples.items()})
    publish(output/'summary.json', result)
    print('SEMANTIC_AUDIT_COMPLETE '+json.dumps({'counts': dict(counts), 'read_evidence': dict(read_status),
                                               'multi_record_trajectories': len(split_trajectories)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.authority, a.manifest_sha256, a.output)


if __name__ == '__main__':
    main()
