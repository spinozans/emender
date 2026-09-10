#!/usr/bin/env python3
"""Read-only source-tool and omitted-planning-field census; never execute trace commands."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
import tiktoken

from scripts.audit_e97_open_swe_semantics import publish, sample_add, sha, verify_file


def compact(text):
    return ' '.join(text.split())


def field_relation(content, reasoning):
    c, r = compact(content), compact(reasoning)
    if not c:
        return 'empty'
    if c == r:
        return 'same_as_reasoning'
    if c in r:
        return 'contained_in_reasoning'
    if r and r in c:
        return 'contains_reasoning_plus_text'
    return 'not_contained_in_reasoning'


def inspect_trajectory(row, encoding, token_cap=2048, byte_cap=65536):
    c, samples = Counter(), []
    baseline, augmented = [], []
    for i, message in enumerate(row['messages']):
        if message.get('role') != 'assistant':
            continue
        calls = message.get('tool_calls') or []
        if len(calls) != 1:
            raise ValueError('included source turn is not a single tool call')
        function = calls[0]['function']; name = function['name']
        args = json.loads(function.get('arguments') or '{}')
        reasoning = str(message.get('reasoning_content') or '')
        content = str(message.get('content') or '')
        c['assistant_messages'] += 1
        c['source_call/'+name] += 1
        relation = field_relation(content, reasoning)
        c['content_relation/'+relation] += 1
        if content.strip():
            c['content_tokens'] += len(encoding.encode_ordinary(content))
            if relation in {'contains_reasoning_plus_text', 'not_contained_in_reasoning'}:
                c['content_tokens_not_contained_in_reasoning'] += len(encoding.encode_ordinary(content))
                samples.append(('content/'+relation, i, {'content': content[:1000], 'reasoning': reasoning[:1000]}))
        if name == 'think':
            thought = str(args.get('thought') or '')
            c['think_calls'] += 1
            c['think_thought_tokens'] += len(encoding.encode_ordinary(thought))
            if thought.strip():
                c['nonempty_think_thought'] += 1
                if compact(thought) not in compact(reasoning):
                    c['think_thought_not_contained_in_reasoning'] += 1
                    samples.append(('think/distinct_thought_argument', i,
                                    {'reasoning': reasoning[:1000], 'thought': thought[:1600]}))
            baseline.append(reasoning)
            augmented.extend([reasoning, thought])
            continue
        current = '\n\n'.join([*baseline, reasoning]).strip()
        with_thought = '\n\n'.join([*augmented, reasoning]).strip()
        base_tokens = len(encoding.encode_ordinary(current))
        thought_tokens = len(encoding.encode_ordinary(with_thought))
        c['baseline_analysis_tokens'] += base_tokens
        c['hypothetical_analysis_with_thought_tokens'] += thought_tokens
        c['analysis_units'] += 1
        if c['analysis_units'] == 1:
            c['initial_executable_call/'+name] += 1
        if base_tokens > token_cap or len(current.encode()) > byte_cap:
            raise ValueError('source baseline disagrees with admitted analysis bounds')
        if thought_tokens > token_cap or len(with_thought.encode()) > byte_cap:
            c['hypothetical_thought_fold_overcap_units'] += 1
            samples.append(('think/hypothetical_fold_exceeds_cap', i,
                            {'baseline_tokens': base_tokens, 'with_thought_tokens': thought_tokens}))
        baseline.clear(); augmented.clear()
        if name == 'execute_bash':
            command = str(args.get('command', ''))
            c['bash_calls'] += 1
            if args.get('is_input') is True or str(args.get('is_input', '')).lower() == 'true':
                c['bash_is_input_true'] += 1
                samples.append(('bash/stdin', i, {'arguments': args, 'command_excerpt': command[:1000]}))
            if 'timeout' in args:
                c['bash_explicit_timeout'] += 1
                if isinstance(args['timeout'], (float, int)) and args['timeout'] > 120:
                    c['bash_timeout_over_120'] += 1
            # Lexical screening only: matches in heredocs/strings do not prove session dependence.
            if re.search(r'(^|[;&\n])\s*(cd\s|export\s|source\s|\.\s)', command):
                c['bash_session_mutation_lexical_candidates'] += 1
                samples.append(('bash/session_lexical_candidate', i, {'command_excerpt': command[:1400]}))
        if name == 'finish':
            final = str(args.get('message', 'Task completed.'))
            if final != compact(final):
                c['finish_message_whitespace_changed'] += 1
            break
    if baseline:
        raise ValueError('orphan think sequence')
    return c, samples


def run(authority, raw_root, manifest_sha256, native_tools, native_sha256, output):
    authority, raw_root, output = Path(authority).resolve(), Path(raw_root).resolve(), Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    if sha(authority/'manifest.json') != manifest_sha256 or sha(native_tools) != native_sha256:
        raise ValueError('manifest/native identity mismatch')
    m = json.loads((authority/'manifest.json').read_text())
    if m['training_eligible'] is not False:
        raise ValueError('expected non-trainable candidate')
    metadata = verify_file(authority, m['outputs']['metadata'])
    expected = {}
    for line in metadata.read_text().splitlines():
        r = json.loads(line)
        item = (r['source_file'], r['source_row'], r['split'])
        previous = expected.setdefault(r['trajectory_identity'], item)
        if previous != item:
            raise ValueError('inconsistent record metadata')
    if sha(raw_root/'README.md') != m['dataset_card_sha256']:
        raise ValueError('dataset card mismatch')
    files = []
    for d in m['input_files']:
        ps = list(raw_root.rglob(d['name']))
        if len(ps) != 1 or ps[0].is_symlink():
            raise ValueError('ambiguous source file')
        p = ps[0]
        if p.stat().st_size != d['bytes'] or sha(p) != d['sha256']:
            raise ValueError('source mismatch')
        files.append((p, d))
    identity = {'manifest_sha256': manifest_sha256, 'native_tools_sha256': native_sha256,
                'native_tools_snapshot': str(native_tools), 'probe_sha256': sha(__file__),
                'raw_inputs': m['input_files'], 'training_eligible': False,
                'scope': 'source declarations and text-field retention; no tool execution or new serialization'}
    publish(output/'identity.json', identity)
    encoding = tiktoken.get_encoding('p50k_base')
    counts, affected = Counter(), Counter()
    specs, samples, seen = {}, defaultdict(list), set()
    print('CONTRACT_AUDIT_INPUTS_VERIFIED', flush=True)
    for p, descriptor in files:
        index = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=8, use_threads=False):
            for row in batch.to_pylist():
                at = index; index += 1
                tid = 'open-swe:'+str(row['trajectory_id'])
                if tid not in expected:
                    continue
                if tid in seen or expected[tid][:2] != (p.name, at):
                    raise ValueError('source row identity mismatch')
                seen.add(tid)
                c, examples = inspect_trajectory(row, encoding,
                    m['analysis_policy']['admission_token_cap'], m['analysis_policy']['admission_byte_cap'])
                counts.update(c)
                for key in ('think_calls', 'hypothetical_thought_fold_overcap_units', 'bash_is_input_true',
                            'bash_explicit_timeout', 'bash_session_mutation_lexical_candidates',
                            'content_tokens_not_contained_in_reasoning'):
                    if c[key]:
                        affected[key] += 1
                for spec in row.get('tools') or []:
                    spec = json.loads(spec) if isinstance(spec, str) else spec
                    raw = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
                    digest = hashlib.sha256(raw).hexdigest()
                    entry = specs.setdefault(digest, {'specification': spec, 'trajectories': 0})
                    entry['trajectories'] += 1
                if expected[tid][2] == 0:
                    for group, turn, example in examples:
                        # Bound raw argument strings as well as commentary excerpts.
                        if 'arguments' in example:
                            example['arguments_excerpt'] = json.dumps(example.pop('arguments'), ensure_ascii=False)[:1400]
                        sample_add(samples, group, {'identity': tid, 'turn_index': turn,
                            'source_file': p.name, 'source_row': at, **example})
        print(f'CONTRACT_AUDIT_SHARD_COMPLETE file={p.name} included={len(seen)}', flush=True)
    if seen != set(expected) or len(seen) != m['counts']['included_trajectories']:
        raise ValueError('missing included sources')
    if counts['baseline_analysis_tokens'] != m['counts']['private_analysis_tokens'] or counts['analysis_units'] != m['counts']['target_units']:
        raise ValueError('baseline reconstruction count mismatch')
    for p, d in files:
        if p.stat().st_size != d['bytes'] or sha(p) != d['sha256']:
            raise ValueError('raw source changed during audit')
    verify_file(authority, m['outputs']['metadata'])
    if sha(authority/'manifest.json') != manifest_sha256 or sha(native_tools) != native_sha256:
        raise ValueError('manifest/native input changed during audit')
    result = {'schema': 'emender-open-swe-contract-census-v1', 'status': 'completed',
        'identity': identity, 'included_trajectories': len(seen), 'counts': dict(counts),
        'affected_trajectories': dict(affected), 'tool_specification_variants': len(specs),
        'limitations': ['Text containment is lexical, not a semantic assessment of unique information.',
            'Hypothetical thought folding only measures a cap; it does not authorize merging public and private channels.',
            'Shell regex flags do not establish actual reliance on persistent state.',
            'Declared tool specifications and a native source snapshot are not runtime replay qualification.']}
    publish(output/'tool-specifications.json', specs)
    publish(output/'samples.json', {k: [dict(v, sample_key=h) for h, v in values] for k, values in samples.items()})
    publish(output/'summary.json', result)
    print('CONTRACT_AUDIT_COMPLETE '+json.dumps({'counts': dict(counts), 'affected': dict(affected)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    for name in ('authority', 'raw-root', 'native-tools', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--native-sha256', required=True)
    a = p.parse_args()
    run(a.authority, a.raw_root, a.manifest_sha256, a.native_tools, a.native_sha256, a.output)


if __name__ == '__main__':
    main()
