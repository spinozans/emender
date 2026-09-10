#!/usr/bin/env python3
"""Reconcile converted units with pinned raw Open-SWE; census semantic risk origins.

Executes only selected, hash-verified project converter functions, never commands
from trajectory data. No filesystem/tool replay or training admission.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import tiktoken

from scripts.audit_e97_open_swe_semantics import (
    INDEX, RS, observation_from_gap, parse_turn, publish, read_evidence,
    sample_add, sha, spans, verify_file,
)


def unit_hash(assistant, observation):
    return hashlib.sha256(json.dumps([assistant, observation], ensure_ascii=False,
                                    separators=(',', ':')).encode()).digest()


def load_converter(authority, manifest, encoding):
    """No module import or top-level training/build execution from the archive."""
    env = {'re': re, 'json': json, 'codec': SimpleNamespace(RS=RS),
           'E97_PI_AGENT_ANALYSIS_SYSTEM_V1': manifest['system_prompt'],
           'MAX_PRIVATE_ANALYSIS_BYTES': manifest['analysis_policy']['admission_byte_cap'],
           '_ANALYSIS_TOKEN_CAP': manifest['analysis_policy']['admission_token_cap'],
           '_ANALYSIS_ENCODING': encoding}
    for key, names, assignment in (
        ('action_normalizer_source', {'normalize_text', 'normalize_path', 'normalize_command', 'action_from_call'}, '_WORKSPACE'),
        ('builder_source', {'_finish_from_call', '_reasoning_prefix', 'normalize_private_messages'}, None),
    ):
        path = verify_file(authority, manifest['source_snapshots'][key])
        tree = ast.parse(path.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in selected} != names or any(n.decorator_list for n in selected):
            raise ValueError('unexpected archived converter function layout')
        if assignment:
            assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == assignment for t in n.targets)]
            if len(assignments) != 1:
                raise ValueError('missing archived workspace expression')
            selected = assignments + selected
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
        exec(compile(module, str(path), 'exec'), env)
    return env


def source_read_category(values):
    if 'view_range' not in values or values['view_range'] is None:
        return 'no_view_range'
    value = values['view_range']
    if isinstance(value, list) and len(value) == 2:
        try:
            return 'open_ended_view_range' if int(value[1]) < 0 else 'finite_view_range'
        except (TypeError, ValueError):
            pass
    return 'other_view_range'


def raw_diagnostics(row, converter):
    counts, ranges, roots = Counter(), Counter(), Counter()
    examples = []
    messages = row['messages']
    finished = False
    for i, m in enumerate(messages):
        if m.get('role') == 'user':
            counts['user_messages_after_finish' if finished else 'user_messages_before_finish'] += 1
        if m.get('role') != 'assistant' or finished:
            continue
        if m.get('content') and str(m['content']).strip():
            counts['assistant_messages_with_nonempty_content_field'] += 1
        for call in m.get('tool_calls') or []:
            function = call['function']
            name = function['name']
            values = json.loads(function.get('arguments') or '{}')
            if name == 'finish':
                finished = True
                continue
            following = messages[i+1] if i+1 < len(messages) else {}
            observation = converter['normalize_text'](str(following.get('content', ''))) if following.get('role') == 'tool' else None
            if name == 'str_replace_editor':
                raw_path = str(values.get('path', ''))
                if raw_path.startswith(('/workspace/', '/testbed/')):
                    roots['/'.join(raw_path.split('/')[:3])] += 1
                if raw_path.startswith('/testbed/'):
                    counts['editor_paths_under_testbed'] += 1
                if values.get('command') == 'view':
                    _, action = converter['action_from_call'](call, observation)
                    if action and action.startswith('Action: read\n'):
                        args = json.loads(action.partition('\nArguments: ')[2])
                        evidence = read_evidence(args, observation or '(no tool output)')
                        category = source_read_category(values)
                        ranges[category] += 1
                        counts['converted_reads'] += 1
                        if evidence.get('outside_lines', 0):
                            counts['out_of_bounds_reads'] += 1
                            counts['out_of_bounds/'+category] += 1
                            examples.append(('bounds/'+category, {'message_index': i,
                                'original_arguments': values, 'converted_arguments': args, 'evidence': evidence}))
                        if (observation or '').startswith('ERROR:\nInvalid `view_range` parameter:'):
                            counts['source_view_range_error_observations'] += 1
                    elif action and action.startswith('Action: bash\n'):
                        counts['directory_views_converted_using_observation'] += 1
                for field in ('old_str', 'new_str', 'file_text'):
                    value = values.get(field)
                    if isinstance(value, str) and converter['normalize_text'](value).strip() != value.strip():
                        counts['editor_payload_changed_by_observation_normalizer/'+field] += 1
                        examples.append(('payload/'+field, {'message_index': i,
                            'original_path': raw_path, 'field': field, 'payload_excerpt': value[:700],
                            'normalized_excerpt': converter['normalize_text'](value)[:700]}))
    return counts, ranges, roots, examples


def prepare_converted(authority, manifest, encoding):
    tokens = np.memmap(authority/manifest['outputs']['tokens']['path'], dtype='<u4', mode='r')
    masks = np.memmap(authority/manifest['outputs']['mask']['path'], dtype='u1', mode='r')
    index = (authority/manifest['outputs']['index']['path']).read_bytes()
    rows = [json.loads(x) for x in (authority/manifest['outputs']['metadata']['path']).read_text().splitlines()]
    trajectories = {}
    for i, row in enumerate(rows):
        tid = row['trajectory_identity']
        t = trajectories.setdefault(tid, {'digest': hashlib.sha256(), 'units': 0, 'records': 0,
            'targets': 0, 'repo': row['repo'], 'source_file': row['source_file'], 'source_row': row['source_row'],
            'split': row['split'], 'windows': []})
        if any(t[k] != row[k] for k in ('repo', 'source_file', 'source_row', 'split')):
            raise ValueError('inconsistent trajectory metadata')
        offset, length, targets, split = INDEX.unpack_from(index, i*INDEX.size)
        record, mask = tokens[offset:offset+length], masks[offset:offset+length]
        ranges = spans(mask)
        first = True
        for j, (start, end) in enumerate(ranges):
            assistant = encoding.decode(record[start:end].tolist(), errors='strict')
            if assistant == RS:
                continue
            if assistant.endswith(RS):
                assistant = assistant[:-1]
            _, tool, args = parse_turn(assistant)
            observation = None
            if tool != 'final':
                ns, ne = ranges[j+1]
                gap = encoding.decode(record[end:ns].tolist(), errors='strict')
                following = encoding.decode(record[ns:ne].tolist()) != RS
                observation = observation_from_gap(gap, following)
            if first and t['records'] > 0:
                t['windows'].append({'unit_index': t['units'], 'record_identity': row['identity'],
                    'available_prefix': encoding.decode(record[:start].tolist(), errors='strict'),
                    'path': args.get('path') if args else None})
            first = False
            t['digest'].update(unit_hash(assistant, observation))
            t['units'] += 1
        t['records'] += 1
        t['targets'] += targets
        if (i+1) % 3000 == 0:
            print(f'SOURCE_AUDIT_CONVERTED_PROGRESS records={i+1}', flush=True)
    return trajectories


def run(authority, raw_root, manifest_sha256, census_summary, census_sha256, output):
    authority, raw_root, output = Path(authority).resolve(), Path(raw_root).resolve(), Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    if sha(authority/'manifest.json') != manifest_sha256 or sha(census_summary) != census_sha256:
        raise ValueError('manifest/census identity mismatch')
    manifest = json.loads((authority/'manifest.json').read_text())
    census = json.loads(Path(census_summary).read_text())
    if manifest['training_eligible'] is not False or census['identity']['manifest_sha256'] != manifest_sha256:
        raise ValueError('invalid candidate or prior census')
    descriptors = {**manifest['outputs'], **manifest['source_snapshots'], **manifest['auxiliary_artifacts']}
    for d in descriptors.values():
        verify_file(authority, d)
    if sha(raw_root/'README.md') != manifest['dataset_card_sha256']:
        raise ValueError('raw dataset card mismatch')
    raw_files = []
    for d in manifest['input_files']:
        matches = list(raw_root.rglob(d['name']))
        if len(matches) != 1 or matches[0].is_symlink():
            raise ValueError('ambiguous raw source path')
        p = matches[0]
        if p.stat().st_size != d['bytes'] or sha(p) != d['sha256']:
            raise ValueError('raw source identity mismatch')
        raw_files.append((p, d))
    identity = {'manifest_sha256': manifest_sha256, 'census_summary_sha256': census_sha256,
                'audit_sha256': sha(__file__), 'helper_sha256': sha(Path(__file__).with_name('audit_e97_open_swe_semantics.py')),
                'raw_files': [{'path': str(p), **d} for p, d in raw_files],
                'training_eligible': False,
                'scope': 'raw included trajectories, exact normalized assistant/observation sequence, semantic-risk provenance; no commands executed'}
    publish(output/'identity.json', identity)
    print('SOURCE_AUDIT_INPUTS_VERIFIED', flush=True)
    encoding = tiktoken.get_encoding('p50k_base')
    converter = load_converter(authority, manifest, encoding)
    trajectories = prepare_converted(authority, manifest, encoding)
    counts, ranges, roots, affected = Counter(), Counter(), Counter(), Counter()
    samples = defaultdict(list)
    reports, seen = [], set()
    for path, desc in raw_files:
        source_row = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, use_threads=False):
            for row in batch.to_pylist():
                tid = 'open-swe:'+str(row['trajectory_id'])
                at = source_row
                source_row += 1
                if tid not in trajectories:
                    continue
                if tid in seen:
                    raise ValueError('duplicate source trajectory identity')
                seen.add(tid)
                t = trajectories[tid]
                if t['source_file'] != path.name or t['source_row'] != at or t['repo'] != row['repo']:
                    raise ValueError('raw-to-converted row reference mismatch')
                messages, analysis_tokens, think = converter['normalize_private_messages'](row, encoding=encoding)
                digest = hashlib.sha256()
                unit_index = 0
                window_map = {w['unit_index']: w for w in t['windows']}
                history = '\n\n'.join(role+':\n'+content for role, content in messages[:2])
                lexical = 0
                for j, (role, content) in enumerate(messages[2:], 2):
                    if role == 'assistant':
                        observation = messages[j+1][1] if j+1 < len(messages) and messages[j+1][0] == 'tool' else None
                        digest.update(unit_hash(content, observation))
                        if unit_index in window_map:
                            w = window_map[unit_index]
                            path_arg = w['path']
                            if isinstance(path_arg, str) and path_arg and path_arg not in w['available_prefix'] and path_arg in history:
                                lexical += 1
                                if t['split'] == 0:
                                    sample_add(samples, 'window/path_only_in_earlier_history',
                                        {'identity': tid, 'turn_index': unit_index, 'source_file': path.name,
                                         'source_row': at, 'record_identity': w['record_identity'], 'path': path_arg,
                                         'available_prefix_excerpt': w['available_prefix'][:900]})
                        unit_index += 1
                    history += '\n\n'+role+':\n'+content
                if unit_index != t['units'] or digest.digest() != t['digest'].digest():
                    raise ValueError('normalized source units differ from stored converted targets/observations')
                c, r, rt, examples = raw_diagnostics(row, converter)
                c['later_window_first_paths_only_in_earlier_history'] = lexical
                counts.update(c); ranges.update(r); roots.update(rt)
                counts['verified_trajectory_sequences'] += 1
                counts['verified_assistant_units'] += unit_index
                if c['out_of_bounds_reads']:
                    affected.update(bounds_trajectories=1, targets_in_bounds_affected_trajectories=t['targets'])
                if t['records'] > 1:
                    affected['multi_record_trajectories'] += 1
                if c['out_of_bounds_reads'] or t['records'] > 1:
                    affected.update(bounds_or_multi_record_trajectories=1, targets_in_bounds_or_multi_record_trajectories=t['targets'])
                if c['user_messages_before_finish'] > 1:
                    affected['multiple_prefinish_user_trajectories'] += 1
                if lexical:
                    affected['lexical_window_risk_trajectories'] += 1
                if t['split'] == 0:
                    for group, ex in examples:
                        sample_add(samples, group, {'identity': tid, 'turn_index': ex['message_index'],
                            'source_file': path.name, 'source_row': at, **ex})
                reports.append({'identity': tid, 'source_file': path.name, 'source_row': at,
                    'split': t['split'], 'records': t['records'], 'targets': t['targets'],
                    'assistant_units': unit_index, 'normalized_sequence_exact': True, 'risk_counts': dict(c)})
        print(f'SOURCE_AUDIT_SHARD_COMPLETE file={path.name} included_verified={len(seen)}', flush=True)
    if seen != set(trajectories):
        raise ValueError('missing raw trajectories')
    if counts['verified_assistant_units'] != census['counts']['assistant_units']:
        raise ValueError('assistant-unit count disagrees with census')
    if counts['out_of_bounds_reads'] != census['read_evidence'].get('numbered_lines_outside_requested_range', 0):
        raise ValueError('read mismatch count disagrees with census')
    for p, d in raw_files:
        if p.stat().st_size != d['bytes'] or sha(p) != d['sha256']:
            raise ValueError('raw source changed during audit')
    for d in descriptors.values():
        verify_file(authority, d)
    if sha(authority/'manifest.json') != manifest_sha256 or sha(census_summary) != census_sha256:
        raise ValueError('manifest/census changed during audit')
    publish(output/'trajectories.json', reports)
    publish(output/'samples.json', {k: [dict(v, sample_key=h) for h, v in values] for k, values in samples.items()})
    summary = {'schema': 'emender-open-swe-source-fidelity-v1', 'status': 'completed', 'identity': identity,
        'counts': dict(counts), 'source_read_range_categories': dict(ranges),
        'affected_trajectory_counts': dict(affected), 'editor_path_roots': dict(roots),
        'limitations': ['Exact conversion fidelity does not prove equivalence to destination tool execution.',
            'Window path flags are literal availability checks, not causal impossibility proofs.',
            'Payload-normalization flags measure potential changes to displayed evidence, not actual failed edits.',
            'Neither affected nor unflagged trajectories are newly admitted; other risks remain unverified.']}
    publish(output/'summary.json', summary)
    print('SOURCE_AUDIT_COMPLETE '+json.dumps({'counts': dict(counts), 'affected': dict(affected)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('authority', 'raw-root', 'census-summary', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--census-sha256', required=True)
    a = p.parse_args()
    run(a.authority, a.raw_root, a.manifest_sha256, a.census_summary, a.census_sha256, a.output)


if __name__ == '__main__':
    main()
